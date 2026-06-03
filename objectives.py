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
    sqt     = np.sqrt(max(t, 1e-10))
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


def obj_iv_vega_wmse(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=None, vegas=None):
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

    if spreads is not None and not np.all(spreads == 0):
        spread_sq = np.maximum(spreads, 1e-6) ** 2
        weights   = vega_sq / spread_sq       # vega²/spread²: ATM + liquidity
    else:
        weights = vega_sq                     # pure vega weighting

    w_sum = np.nansum(weights)
    if w_sum < 1e-30:                         # degenerate slice guard
        return obj_iv_mse(w_fit, w_obs, iv_fit, iv_obs, k, t)

    return float(np.nansum(weights * (iv_fit - iv_obs) ** 2) / w_sum)

OBJECTIVES = {
    "w_mse":      obj_w_mse,
    "iv_mse":     obj_iv_mse,
    "iv_wmse":    obj_iv_wmse,
    "price_mse":  obj_price_mse,
    "iv_rmse":    obj_iv_rmse,
    "vega_wmse": obj_iv_vega_wmse
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
# Signature: f(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=None) -> float
# ─────────────────────────────────────────────────────────────────────────────

def metric_iv_rmse(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=None):
    """Absolute RMSE on implied vol (in vol points)."""
    return float(np.sqrt(np.nanmean((iv_fit - iv_obs) ** 2)))


def metric_iv_rrmse(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=None):
    """Relative RMSE on implied vol (as fraction of observed iv)."""
    rel = (iv_fit - iv_obs) / np.maximum(iv_obs, 1e-6)
    return float(np.sqrt(np.nanmean(rel ** 2)))


def metric_price_rmse(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=None):
    """Absolute RMSE on call prices."""
    prices_fit = _prices_from_iv(k, iv_fit, t)
    prices_obs = _prices_from_iv(k, iv_obs, t)
    return float(np.sqrt(np.nanmean((prices_fit - prices_obs) ** 2)))


def metric_price_rrmse(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=None):
    """Relative RMSE on call prices."""
    prices_fit = _prices_from_iv(k, iv_fit, t)
    prices_obs = _prices_from_iv(k, iv_obs, t)
    rel = (prices_fit - prices_obs) / np.maximum(prices_obs, 1e-8)
    return float(np.sqrt(np.nanmean(rel ** 2)))


def metric_w_rmse(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=None):
    """Absolute RMSE on total variance."""
    return float(np.sqrt(np.nanmean((w_fit - w_obs) ** 2)))


def metric_max_iv_err(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=None):
    """Maximum absolute error on implied vol across all strikes."""
    return float(np.nanmax(np.abs(iv_fit - iv_obs)))


METRICS = {
    "iv_rmse":      metric_iv_rmse,
    "iv_rrmse":     metric_iv_rrmse,
    "price_rmse":   metric_price_rmse,
    "price_rrmse":  metric_price_rrmse,
    "w_rmse":       metric_w_rmse,
    "max_iv_err":   metric_max_iv_err,
}

# Default set reported after every calibration
DEFAULT_METRICS = ["iv_rmse", "iv_rrmse", "price_rmse", "w_rmse"]


def get_metric(name: str):
    """Return a metric function by name string."""
    key = name.lower()
    if key not in METRICS:
        raise ValueError(f"Unknown metric '{name}'. Available: {list(METRICS.keys())}")
    return METRICS[key]


def evaluate_all(w_fit, w_obs, iv_fit, iv_obs, k, t,
                 spreads=None, metric_names=None) -> dict:
    """
    Compute all (or a subset of) metrics and return as a dict.

    Parameters
    ----------
    metric_names : list of str or None  (None = DEFAULT_METRICS)
    """
    names = metric_names or DEFAULT_METRICS
    return {
        name: get_metric(name)(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads)
        for name in names
    }
