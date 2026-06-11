"""
objectives.py
=============
Objective functions for calibration and evaluation metrics for post-fit analysis.

Design
------
An objective function takes (w_fit, w_obs, iv_fit, iv_obs, prices_fit,
prices_obs) and returns a scalar to minimise.

A metric function takes the same arguments and returns a scalar for reporting.

Both are fully decoupled from any specific model — the calibration engine
passes the fitted and observed quantities and the objective/metric doesn't
need to know how they were computed.

Adding a new objective or metric means writing one function and registering
it in the OBJECTIVES or METRICS dict at the bottom.

Objectives implemented
----------------------
    w_mse       : mean squared error on total variance  (default, fast)
    iv_mse      : mean squared error on implied volatility
    iv_wmse     : spread-weighted MSE on implied vol  (workshop approach)
    price_mse   : mean squared error on option prices (requires prices)
    iv_rmse     : root MSE on implied vol

Metrics implemented
-------------------
    iv_rmse     : RMSE on implied vol
    iv_rrmse    : relative RMSE on implied vol
    price_rmse  : RMSE on option prices
    price_rrmse : relative RMSE on option prices
    w_rmse      : RMSE on total variance
    max_iv_err  : maximum absolute error on implied vol
"""

import numpy as np
from typing import Optional
from scipy.stats import norm as _norm   # module-level: no per-call import overhead


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES HELPER  (inverse option / coin-margined)
# ─────────────────────────────────────────────────────────────────────────────

def _bs_call_price(k, mark_iv, t):
    """
    Vectorised Black-Scholes call price normalised by the forward (F=1).

    k       : log-strike  log(K/F),  array-like
    mark_iv : implied volatility,     array-like, same shape as k
    t       : scalar time to expiry

    All numpy operations — no Python loop, no per-call import.
    """
    mark_iv = np.asarray(mark_iv, dtype=float)
    k       = np.asarray(k,       dtype=float)
    mark_iv = np.maximum(mark_iv, 1e-8)           # guard against zero IV
    sqt     = np.sqrt(np.maximum(np.asarray(t, dtype=float), 1e-10))
    d1      = -k / (mark_iv * sqt) + mark_iv * sqt / 2
    d2      = d1 - mark_iv * sqt
    price   = _norm.cdf(d1) - np.exp(k) * _norm.cdf(d2)
    return np.maximum(price, 0.0)


def _prices_from_iv(k, iv, t):
    """Vectorised BS call prices — passes full arrays to _bs_call_price."""
    return _bs_call_price(np.asarray(k, dtype=float),
                          np.asarray(iv, dtype=float), t)


# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE FUNCTIONS
# Each returns a scalar >= 0 to be minimised.
# Signature: f(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=None) -> float
# ─────────────────────────────────────────────────────────────────────────────

def obj_w_mse(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=None):
    """Mean squared error on total implied variance."""
    return float(np.nanmean((w_fit - w_obs) ** 2))


def obj_iv_mse(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=None):
    """Mean squared error on implied volatility."""
    return float(np.nanmean((iv_fit - iv_obs) ** 2))


def obj_iv_wmse(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=None):
    """
    Spread-weighted MSE on implied vol (Gatheral workshop approach).
    If spreads are not provided, falls back to unweighted iv_mse.
    """
    if spreads is None or np.all(spreads == 0):
        return obj_iv_mse(w_fit, w_obs, iv_fit, iv_obs, k, t)
    weights = 1.0 / np.maximum(spreads, 1e-6) ** 2
    return float(np.nansum(weights * (iv_fit - iv_obs) ** 2) / np.nansum(weights))


def obj_price_mse(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=None):
    """Mean squared error on call option prices (normalised by forward)."""
    prices_fit = _prices_from_iv(k, iv_fit, t)
    prices_obs = _prices_from_iv(k, iv_obs, t)
    return float(np.nanmean((prices_fit - prices_obs) ** 2))


def obj_iv_rmse(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=None):
    """Root mean squared error on implied volatility."""
    return float(np.sqrt(np.nanmean((iv_fit - iv_obs) ** 2)))

# ─────────────────────────────────────────────────────────────────────────────
# VEGA-WEIGHTED OBJECTIVE  (Gatheral VW4 approach)
# ─────────────────────────────────────────────────────────────────────────────

def _bs_vega(k, iv, t):
    """
    Normalised Black-Scholes vega for a call/put (F=1, K=exp(k)).

    vega = sqrt(t) * K * phi(d1)   where phi is the standard normal PDF.

    This is the sensitivity of the option price to a 1-unit change in IV,
    and is the natural weight for an IV-space fitting objective: a 1-vol-point
    error at ATM causes a much larger price error than the same error deep OTM.

    Parameters
    ----------
    k   : log-strike  log(K/F),  array-like
    iv  : implied vol (annualised),  array-like, same shape as k
    t   : scalar time to expiry in years

    Returns
    -------
    vega : np.ndarray, same shape as k, strictly positive
    """
    
    k   = np.asarray(k,  dtype=float)
    iv  = np.asarray(iv, dtype=float)
    sqt = np.sqrt(np.maximum(t, 1e-10))
    iv  = np.maximum(iv, 1e-6)
    d1  = (-k / (iv * sqt)) + (iv * sqt / 2.0)
    # K = exp(k), so vega = sqrt(t) * exp(k) * phi(d1)
    return sqt * np.exp(k) * _norm.pdf(d1)


def obj_iv_vega_wmse(w_fit, w_obs, iv_fit, iv_obs, k, t,
                     spreads=None, vegas=None, bid=None, ask=None):
    """
    Vega-weighted MSE on implied vol (Gatheral VW4, slide 'Choice of objective').

    Weight each point by its Black-Scholes vega squared, so that ATM options
    (where vega is largest) dominate the fit and deep-wing noise is suppressed.

    If bid-ask spreads are also provided, the final weight combines both:
        weight_i = vega_i² / spread_i²
    so the fit is simultaneously ATM-biased and liquidity-aware. When spreads
    are absent, pure vega-weighting is used.

    Parameters
    ----------
    vegas : array-like, optional
        Precomputed Black-Scholes vegas (e.g. from df['vega']). When provided
        these are used directly, skipping the internal _bs_vega() call. Pass
        via calibrate_snapshot(..., vega_col='vega') or the global calibrators.

    Notes
    -----
    - For very short maturities (T < 1 week) vega is small everywhere because
      sqrt(T) is tiny; the weighting is still valid but becomes less
      discriminating across strikes.
    - Weights are normalised so the objective is scale-invariant.
    """
    k      = np.asarray(k,      dtype=float)
    iv_fit = np.asarray(iv_fit, dtype=float)
    iv_obs = np.asarray(iv_obs, dtype=float)

    vega    = np.asarray(vegas, dtype=float) if vegas is not None else _bs_vega(k, iv_obs, t)
    vega_sq = vega ** 2

    # Derive spreads from bid/ask if not provided directly.
    if spreads is None and bid is not None and ask is not None:
        spreads = np.asarray(ask, dtype=float) - np.asarray(bid, dtype=float)

    if spreads is not None and not np.all(spreads == 0):
        spread_sq = np.maximum(spreads, 1e-6) ** 2
        weights   = vega_sq / spread_sq       # vega²/spread²: ATM + liquidity
    else:
        weights = vega_sq                     # pure vega weighting

    w_sum = np.nansum(weights)
    if w_sum < 1e-30:                         # degenerate slice guard
        return obj_iv_mse(w_fit, w_obs, iv_fit, iv_obs, k, t)

    return float(np.nansum(weights * (iv_fit - iv_obs) ** 2) / w_sum)


def obj_band_loss(w_fit, w_obs, iv_fit, iv_obs, k, t,
                  spreads=None, vegas=None, bid=None, ask=None,
                  vega_weight=False, mid_anchor=0.0):
    """
    Bid-ask band loss — directly targets "stay inside the IV book".

    Loss per point is zero when iv_fit is inside [bid, ask], and quadratic
    in the excess otherwise:

        L_i = max(0, iv_fit_i - ask_i)^2 + max(0, bid_i - iv_fit_i)^2

    This is a smooth, differentiable proxy for maximising the hit-rate.
    Unlike `iv_rmse`, it does not pull the fit toward `mark_iv` — once the
    smile is inside the book the gradient vanishes, so the optimiser is
    free to balance other constraints (no-arb penalties, smoothness, etc.).

    Parameters
    ----------
    vega_weight : bool
        If True, weight each point by vega^2 so ATM (tight, liquid)
        contracts dominate. Recommended for crypto where deep-wing IV
        spreads are huge and would otherwise let wings drift far.
    mid_anchor : float, optional
        Small coefficient (e.g. 1e-4) on a tiebreaker term that pulls
        iv_fit toward the mid (bid+ask)/2 when already inside the band.
        Keeps the smile from drifting to a corner of the band on
        under-determined slices. Set to 0 to disable.

    Falls back to `iv_mse` if bid/ask are not provided.
    """
    if bid is None or ask is None:
        return obj_iv_mse(w_fit, w_obs, iv_fit, iv_obs, k, t)

    iv_fit = np.asarray(iv_fit, dtype=float)
    bid    = np.asarray(bid,    dtype=float)
    ask    = np.asarray(ask,    dtype=float)

    above = np.maximum(0.0, iv_fit - ask)
    below = np.maximum(0.0, bid - iv_fit)
    per_point = above ** 2 + below ** 2

    if mid_anchor and mid_anchor > 0:
        mid = 0.5 * (bid + ask)
        per_point = per_point + mid_anchor * (iv_fit - mid) ** 2

    if vega_weight:
        vega = (np.asarray(vegas, dtype=float)
                if vegas is not None else _bs_vega(k, iv_obs, t))
        weights = vega ** 2
        wsum    = np.nansum(weights)
        if wsum < 1e-30:
            return float(np.nanmean(per_point))
        return float(np.nansum(weights * per_point) / wsum)

    return float(np.nanmean(per_point))


def obj_vega_wmse_band(w_fit, w_obs, iv_fit, iv_obs, k, t,
                       spreads=None, vegas=None, bid=None, ask=None,
                       band_weight=10.0):
    """
    Hybrid: vega-weighted MSE on (iv_fit − mark_iv), PLUS an additive band
    penalty that fires only when iv_fit leaves [bid, ask].

        L = vega_wmse(iv_fit, iv_obs) + band_weight · band_loss(iv_fit, bid, ask)

    This keeps a strong gradient everywhere (from vega_wmse) so low-parameter
    models like SABR don't ping-pong between local minima, while still
    explicitly penalising any drift outside the IV book. The band_weight
    controls how aggressively the fit is dragged inside the spread.

    Falls back to pure vega_wmse if bid/ask are not provided.
    """
    base = obj_iv_vega_wmse(w_fit, w_obs, iv_fit, iv_obs, k, t,
                            spreads=spreads, vegas=vegas)
    if bid is None or ask is None:
        return base

    iv_fit = np.asarray(iv_fit, dtype=float)
    bid    = np.asarray(bid,    dtype=float)
    ask    = np.asarray(ask,    dtype=float)
    above  = np.maximum(0.0, iv_fit - ask)
    below  = np.maximum(0.0, bid - iv_fit)
    band   = float(np.nanmean(above ** 2 + below ** 2))
    return base + band_weight * band


OBJECTIVES = {
    "w_mse":           obj_w_mse,
    "iv_mse":          obj_iv_mse,
    "iv_wmse":         obj_iv_wmse,
    "price_mse":       obj_price_mse,
    "iv_rmse":         obj_iv_rmse,
    "vega_wmse":       obj_iv_vega_wmse,
    "band":            obj_band_loss,
    "vega_wmse_band":  obj_vega_wmse_band,
}


def get_objective(name: str):
    """Return an objective function by name string."""
    key = name.lower()
    if key not in OBJECTIVES:
        raise ValueError(f"Unknown objective '{name}'. Available: {list(OBJECTIVES.keys())}")
    return OBJECTIVES[key]


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# Each returns a scalar for reporting — not used during optimisation.
# Signature: f(w_fit, w_obs, iv_fit, iv_obs, k, t, bid=None, ask=None) -> float
# ─────────────────────────────────────────────────────────────────────────────

def metric_iv_rmse(w_fit, w_obs, iv_fit, iv_obs, k, t, bid=None, ask=None):
    """Absolute RMSE on implied vol (in vol points)."""
    return float(np.sqrt(np.nanmean((iv_fit - iv_obs) ** 2)))


def metric_iv_rrmse(w_fit, w_obs, iv_fit, iv_obs, k, t, bid=None, ask=None):
    """Relative RMSE on implied vol (as fraction of observed iv)."""
    rel = (iv_fit - iv_obs) / np.maximum(iv_obs, 1e-6)
    return float(np.sqrt(np.nanmean(rel ** 2)))


def metric_price_rmse(w_fit, w_obs, iv_fit, iv_obs, k, t, bid=None, ask=None):
    """Absolute RMSE on call prices."""
    prices_fit = _prices_from_iv(k, iv_fit, t)
    prices_obs = _prices_from_iv(k, iv_obs, t)
    return float(np.sqrt(np.nanmean((prices_fit - prices_obs) ** 2)))


def metric_price_rrmse(w_fit, w_obs, iv_fit, iv_obs, k, t, bid=None, ask=None):
    """Relative RMSE on call prices."""
    prices_fit = _prices_from_iv(k, iv_fit, t)
    prices_obs = _prices_from_iv(k, iv_obs, t)
    rel = (prices_fit - prices_obs) / np.maximum(prices_obs, 1e-8)
    return float(np.sqrt(np.nanmean(rel ** 2)))

def metric_spread_hit_rate(w_fit, w_obs, iv_fit, iv_obs, k, t, bid=None, ask=None):
    """
    Fraction of fitted IVs that fall inside the observed bid-ask IV spread.
    `bid` and `ask` here are expected to be bid_iv / ask_iv arrays
    (not bid_price / ask_price) to avoid call/put + forward/spot ambiguity
    introduced by converting through Black-Scholes.
    """
    if bid is None or ask is None:
        return float("nan")
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    inside = (iv_fit >= bid) & (iv_fit <= ask)
    valid = np.isfinite(iv_fit) & np.isfinite(bid) & np.isfinite(ask)
    if not valid.any():
        return float("nan")
    return float(np.mean(inside[valid]))

def metric_w_rmse(w_fit, w_obs, iv_fit, iv_obs, k, t, bid=None, ask=None):
    """Absolute RMSE on total variance."""
    return float(np.sqrt(np.nanmean((w_fit - w_obs) ** 2)))


def metric_max_iv_err(w_fit, w_obs, iv_fit, iv_obs, k, t, bid=None, ask=None):
    """Maximum absolute error on implied vol across all strikes."""
    return float(np.nanmax(np.abs(iv_fit - iv_obs)))

def metric_iv_vwrmse(w_fit, w_obs, iv_fit, iv_obs, k, t, bid=None, ask=None):
    vega = _bs_vega(k, iv_obs, t)
    num  = np.nansum(vega * (iv_fit - iv_obs) ** 2)
    den  = np.nansum(vega)
    return float(np.sqrt(num / den)) if den > 0 else np.nan


METRICS = {
    "iv_rmse":      metric_iv_rmse,
    "iv_rrmse":     metric_iv_rrmse,
    "price_rmse":   metric_price_rmse,
    "price_rrmse":  metric_price_rrmse,
    "w_rmse":       metric_w_rmse,
    "max_iv_err":   metric_max_iv_err,
    "spread_hit_rate": metric_spread_hit_rate,
    "vwrmse":   metric_iv_vwrmse
}

# Default set reported after every calibration
DEFAULT_METRICS = ["iv_rrmse", "spread_hit_rate", 'price_rrmse', "w_rmse", "vwrmse"]


def get_metric(name: str):
    """Return a metric function by name string."""
    key = name.lower()
    if key not in METRICS:
        raise ValueError(f"Unknown metric '{name}'. Available: {list(METRICS.keys())}")
    return METRICS[key]


def evaluate_all(w_fit, w_obs, iv_fit, iv_obs, k, t,
                 bid=None, ask=None, metric_names=None) -> dict:
    """
    Compute all (or a subset of) metrics and return as a dict.

    Parameters
    ----------
    bid, ask     : arrays of observed bid/ask prices (optional, used by
                   spread-based metrics like spread_hit_rate)
    metric_names : list of str or None  (None = DEFAULT_METRICS)
    """
    names = metric_names if metric_names is not None else DEFAULT_METRICS
    return {
        name: get_metric(name)(w_fit, w_obs, iv_fit, iv_obs, k, t, bid, ask)
        for name in names
    }
