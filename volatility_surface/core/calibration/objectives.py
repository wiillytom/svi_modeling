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
import inspect as _inspect
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


# ─────────────────────────────────────────────────────────────────────────────
# MONEYNESS-AWARE WEIGHTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _standardized_moneyness(k, iv_obs, t):
    """z = k / (sigma * sqrt(t)) — log-moneyness in units of one std move."""
    k      = np.asarray(k,      dtype=float)
    iv_obs = np.asarray(iv_obs, dtype=float)
    sqt    = np.sqrt(max(float(t), 1e-10))
    return k / np.maximum(iv_obs * sqt, 1e-8)


def _huber(r, delta):
    """
    Elementwise Huber loss, normalised to coincide with r**2 in the quadratic
    region so it is a drop-in replacement for squared error:

        r**2                       for |r| <= delta
        delta * (2|r| - delta)     for |r| >  delta

    Continuous with continuous first derivative at |r| = delta.
    """
    a = np.abs(np.asarray(r, dtype=float))
    return np.where(a <= delta, a ** 2, delta * (2.0 * a - delta))


def _local_dk(k):
    """
    Trapezoidal local strike spacing for each log-moneyness point — turns a
    discrete sum into a discretised integral over k, removing the strike-density
    tilt (dense ATM listings no longer dominate just by being numerous).
    """
    k     = np.asarray(k, dtype=float)
    order = np.argsort(k)
    ks    = k[order]
    dk    = np.empty_like(ks)
    if ks.size == 1:
        dk[:] = 1.0
    else:
        dk[1:-1] = 0.5 * (ks[2:] - ks[:-2])
        dk[0]    = ks[1]  - ks[0]
        dk[-1]   = ks[-1] - ks[-2]
    out        = np.empty_like(dk)
    out[order] = dk
    return np.maximum(out, 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# (A) STANDARDIZED-MONEYNESS POWER-WEIGHTED OBJECTIVE
# ─────────────────────────────────────────────────────────────────────────────

def obj_iv_zweighted(w_fit, w_obs, iv_fit, iv_obs, k, t,
                     spreads=None, vegas=None, bid=None, ask=None,
                     beta=0.4, huber_delta=None, density_correct=False):
    """
    Vega weighting with a standardized-moneyness boost that undoes vega's
    Gaussian decay in a controlled way.

        weight_j = vega_j * exp(beta * z_j**2) / spread_j**2 * [dk_j]

    where z_j = k_j / (sigma_j sqrt(t)).  Because vega ∝ exp(-z**2 / 2):

        beta = 0    → pure vega           (ATM-dominated, current behaviour)
        beta = 0.5  → uniform weight per unit of standardized moneyness
                      (every z-quantile of the smile weighted equally)
        beta > 0.5  → active wing emphasis (use with care)

    Equivalently weight_j ≈ vega_j**(1 - 2 beta): a smooth power family that
    nests pure vega (p=1) and unweighted (p=0) and gives a single, monotone,
    interpretable knob on wing attention — strictly more expressive than a
    convex WRMSE / IV-RMSE blend.

    Parameters
    ----------
    beta : float in [0, ~0.6]
        Wing-emphasis strength (see above).
    huber_delta : float or None
        If set, residuals pass through a Huber loss (quadratic for
        |Δσ| <= huber_delta, linear beyond) so stale / one-sided wing quotes
        cannot dominate.  None → plain squared error.
    density_correct : bool
        If True, multiply weights by the local strike spacing dk_j so the
        objective approximates an integral over moneyness rather than a sum over
        (ATM-clustered) quotes.
    spreads / bid / ask : optional liquidity inputs.  If spreads is None but
        bid/ask are given, spreads = ask - bid.  Division by spread**2 makes the
        moneyness boost self-regulating: it up-weights wings while the liquidity
        term suppresses illiquid ones, letting the data arbitrate.
    """
    k      = np.asarray(k,      dtype=float)
    iv_fit = np.asarray(iv_fit, dtype=float)
    iv_obs = np.asarray(iv_obs, dtype=float)

    vega    = np.asarray(vegas, dtype=float) if vegas is not None else _bs_vega(k, iv_obs, t)
    z       = _standardized_moneyness(k, iv_obs, t)
    weights = vega * np.exp(beta * z ** 2)

    if spreads is None and bid is not None and ask is not None:
        spreads = np.asarray(ask, dtype=float) - np.asarray(bid, dtype=float)
    if spreads is not None and not np.all(np.asarray(spreads) == 0):
        weights = weights / np.maximum(np.asarray(spreads, dtype=float), 1e-6) ** 2

    if density_correct:
        weights = weights * _local_dk(k)

    resid = iv_fit - iv_obs
    loss  = _huber(resid, huber_delta) if (huber_delta and huber_delta > 0) else resid ** 2

    wsum = np.nansum(weights)
    if wsum < 1e-30:
        return obj_iv_mse(w_fit, w_obs, iv_fit, iv_obs, k, t)
    return float(np.nansum(weights * loss) / wsum)


# ─────────────────────────────────────────────────────────────────────────────
# (BASELINE) CONVEX COMBINATION  WRMSE / IV-RMSE
# ─────────────────────────────────────────────────────────────────────────────

def obj_iv_convex_blend(w_fit, w_obs, iv_fit, iv_obs, k, t,
                        spreads=None, vegas=None, bid=None, ask=None,
                        kappa=0.5, combine="mse"):
    """
    The convex-combination baseline:  (1-kappa)*WRMSE + kappa*IV-RMSE.

    WRMSE is the (linear) vega-weighted vol RMSE matching the FactSet paper;
    IV-RMSE is the unweighted vol RMSE.

    combine : {"mse", "rmse"}
        "mse"  (default, recommended): blend the two *MSEs* under a single sqrt,
               so kappa is a fixed, reproducible weight ratio.
        "rmse" (the naive form): blend the two *RMSEs*.  Kept only so the
               pathology can be reproduced — the outer sqrts make kappa's
               effective weight state-dependent (gradient ∝ 1/RMSE per term).
    """
    iv_fit = np.asarray(iv_fit, dtype=float)
    iv_obs = np.asarray(iv_obs, dtype=float)
    vega   = np.asarray(vegas, dtype=float) if vegas is not None else _bs_vega(k, iv_obs, t)

    resid = iv_fit - iv_obs
    vsum  = np.nansum(vega)
    wmse  = (float(np.nansum(vega * resid ** 2) / vsum) if vsum > 1e-30
             else float(np.nanmean(resid ** 2)))
    mse   = float(np.nanmean(resid ** 2))

    if combine == "rmse":
        return float((1.0 - kappa) * np.sqrt(wmse) + kappa * np.sqrt(mse))
    return float(np.sqrt((1.0 - kappa) * wmse + kappa * mse))


# ─────────────────────────────────────────────────────────────────────────────
# (C) REPLICATION / BARRIER-BAND WEIGHTED OBJECTIVE
# ─────────────────────────────────────────────────────────────────────────────

def obj_iv_replication(w_fit, w_obs, iv_fit, iv_obs, k, t,
                       spreads=None, vegas=None, bid=None, ask=None,
                       k_lo=-2.0, k_hi=0.0, boost=5.0, p=1.0, smooth=0.1,
                       huber_delta=None):
    """
    Weight each strike by how much a target exotic loads on it — a crude but
    purpose-built proxy for the static-replication density.  A down-and-out /
    knock-in book loads on the put wing, so the default band [k_lo, k_hi] sits
    below ATM; set the band to wherever your barrier / exotic concentrates.

        weight_j = vega_j**p * (1 + boost * smooth_box(k_j; k_lo, k_hi)) / spread_j**2

    boost controls how strongly the replication band is emphasised over the rest
    of the smile; p in [0,1] tunes the residual vega tilt (p=1 keeps full vega
    outside the band, p=0 makes the base flat).
    """
    k      = np.asarray(k,      dtype=float)
    iv_fit = np.asarray(iv_fit, dtype=float)
    iv_obs = np.asarray(iv_obs, dtype=float)

    vega    = np.asarray(vegas, dtype=float) if vegas is not None else _bs_vega(k, iv_obs, t)
    base_w  = np.power(np.maximum(vega, 1e-30), p)
    weights = base_w * (1.0 + boost * _smooth_box(k, k_lo, k_hi, smooth))

    if spreads is None and bid is not None and ask is not None:
        spreads = np.asarray(ask, dtype=float) - np.asarray(bid, dtype=float)
    if spreads is not None and not np.all(np.asarray(spreads) == 0):
        weights = weights / np.maximum(np.asarray(spreads, dtype=float), 1e-6) ** 2

    resid = iv_fit - iv_obs
    loss  = _huber(resid, huber_delta) if (huber_delta and huber_delta > 0) else resid ** 2

    wsum = np.nansum(weights)
    if wsum < 1e-30:
        return obj_iv_mse(w_fit, w_obs, iv_fit, iv_obs, k, t)
    return float(np.nansum(weights * loss) / wsum)


# ─────────────────────────────────────────────────────────────────────────────
# FACTORIES  (bake hyperparameters into a calibrator-ready objective)
# The calibrator calls objective(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads)
# and auto-injects vegas/bid/ask by inspecting the signature, so every factory
# returns a closure exposing exactly those names.
# ─────────────────────────────────────────────────────────────────────────────

def make_iv_zweighted(beta=0.3, huber_delta=None, density_correct=False):
    """Configured standardized-moneyness power-weighted objective (A)."""
    def _obj(w_fit, w_obs, iv_fit, iv_obs, k, t,
             spreads=None, vegas=None, bid=None, ask=None):
        return obj_iv_zweighted(w_fit, w_obs, iv_fit, iv_obs, k, t,
                                spreads=spreads, vegas=vegas, bid=bid, ask=ask,
                                beta=beta, huber_delta=huber_delta,
                                density_correct=density_correct)
    _obj.__name__ = ("iv_zweighted(beta=%g" % beta
                     + (",huber=%g" % huber_delta if huber_delta else "")
                     + (",dk" if density_correct else "") + ")")
    return _obj


# ─────────────────────────────────────────────────────────────────────────────
# EXACT UNIFORM-IN-Z OBJECTIVE
# ─────────────────────────────────────────────────────────────────────────────

def obj_iv_uniform_z(w_fit, w_obs, iv_fit, iv_obs, k, t,
                     spreads=None, vegas=None, bid=None, ask=None,
                     huber_delta=None):
    """
    Exact uniform-per-unit-of-standardized-moneyness weighting.

    Unlike iv_zweighted(beta=0.5), which achieves approximate uniformity by
    cancelling vega's Gaussian decay via exp(0.5*z²), this objective weights
    each point by its **local dz spacing** — the direct discretisation of the
    integral ∫(σ̂-σ)² dz over standardized moneyness z = k/(σ√t):

        L = Σ_j Δz_j * (σ̂_j - σ_j)²  /  Σ_j Δz_j

    where Δz_j is the trapezoidal half-width between neighbouring z values.

    Why this is exact where iv_zweighted(beta=0.5) is approximate
    -------------------------------------------------------------
    iv_zweighted relies on vega ∝ exp(-z²/2), which holds under the Gaussian
    approximation that ignores:
      (a) the 1/(S√t) prefactor (varying across strikes/times),
      (b) the d₁ vs d₂ asymmetry at large |z|.
    In the deep wings — precisely where we're now adding weight — this
    approximation degrades and the realized weight under beta=0.5 is
    mildly non-uniform in z.

    This objective uses the actual z-coordinates directly, so "every equal
    interval of standardized moneyness gets equal fitting authority" holds
    exactly regardless of how deep the wing is.

    Optional liquidity guard
    -----------------------
    If bid/ask spreads are provided, the dz weights are divided by spread²
    so illiquid wide-spread wing quotes are not over-emphasised.

    Optional Huber loss
    -------------------
    Set huber_delta to a small positive float (e.g. 0.02 = 2 vol points) to
    use a Huber loss instead of squared error in the wings — useful when
    stale/one-sided OTM quotes would otherwise dominate.
    """
    k      = np.asarray(k,      dtype=float)
    iv_fit = np.asarray(iv_fit, dtype=float)
    iv_obs = np.asarray(iv_obs, dtype=float)

    z       = _standardized_moneyness(k, iv_obs, t)
    weights = _local_dk(z)          # local spacing in z — the exact measure

    if spreads is None and bid is not None and ask is not None:
        spreads = np.asarray(ask, dtype=float) - np.asarray(bid, dtype=float)
    if spreads is not None and not np.all(np.asarray(spreads) == 0):
        weights = weights / np.maximum(np.asarray(spreads, dtype=float), 1e-6) ** 2

    resid = iv_fit - iv_obs
    loss  = _huber(resid, huber_delta) if (huber_delta and huber_delta > 0) else resid ** 2

    wsum = np.nansum(weights)
    if wsum < 1e-30:
        return obj_iv_mse(w_fit, w_obs, iv_fit, iv_obs, k, t)
    return float(np.nansum(weights * loss) / wsum)


def make_iv_uniform_z(huber_delta=None):
    """Factory for the exact uniform-in-z objective (no hyperparameter to tune)."""
    def _obj(w_fit, w_obs, iv_fit, iv_obs, k, t,
             spreads=None, vegas=None, bid=None, ask=None):
        return obj_iv_uniform_z(w_fit, w_obs, iv_fit, iv_obs, k, t,
                                spreads=spreads, bid=bid, ask=ask,
                                huber_delta=huber_delta)
    _obj.__name__ = "iv_uniform_z" + ((",huber=%g" % huber_delta) if huber_delta else "")
    return _obj


OBJECTIVES = {
    "w_mse":           obj_w_mse,
    "iv_mse":          obj_iv_mse,
    "iv_wmse":         obj_iv_wmse,
    "price_mse":       obj_price_mse,
    "iv_rmse":         obj_iv_rmse,
    "vega_wmse":       obj_iv_vega_wmse,
    "band":            obj_band_loss,
    "vega_wmse_band":  obj_vega_wmse_band,
    "iv_zweighted":    obj_iv_zweighted,     # (A)  beta=0 default → pure-vega
    "iv_uniform_z":    obj_iv_uniform_z,     # exact uniform-in-z (no Gaussian approx)
    "iv_convex_blend": obj_iv_convex_blend,  # baseline  kappa=0.5
    "iv_replication":  obj_iv_replication,   # (C)  put-wing band by default
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


def metric_iv_vwrrmse(w_fit, w_obs, iv_fit, iv_obs, k, t, bid=None, ask=None):
    """Vega-weighted *relative* RMSE in vol space: sqrt(Σ v·(Δσ/σ)² / Σ v)."""
    iv_fit = np.asarray(iv_fit, float)
    iv_obs = np.asarray(iv_obs, float)
    vega   = _bs_vega(k, iv_obs, t)
    rel    = (iv_fit - iv_obs) / np.maximum(iv_obs, 1e-6)
    num    = np.nansum(vega * rel ** 2)
    den    = np.nansum(vega)
    return float(np.sqrt(num / den)) if den > 0 else np.nan


# Near-money / wing RMSE split (the two halves of the ATM-vs-OTM trade-off).
# |z| <= 1 is "near the money"; |z| > 1 is "the wing".
_Z_ATM_DEFAULT = 1.0

def metric_iv_rmse_atm(w_fit, w_obs, iv_fit, iv_obs, k, t, bid=None, ask=None):
    """Unweighted vol RMSE on near-money points (|z| <= 1)."""
    z = _standardized_moneyness(k, iv_obs, t)
    m = np.abs(z) <= _Z_ATM_DEFAULT
    if not np.any(m):
        return float("nan")
    r = np.asarray(iv_fit, float)[m] - np.asarray(iv_obs, float)[m]
    return float(np.sqrt(np.nanmean(r ** 2)))


def metric_iv_rmse_otm(w_fit, w_obs, iv_fit, iv_obs, k, t, bid=None, ask=None):
    """Unweighted vol RMSE on wing points (|z| > 1) — the LSV / barrier-relevant half."""
    z = _standardized_moneyness(k, iv_obs, t)
    m = np.abs(z) > _Z_ATM_DEFAULT
    if not np.any(m):
        return float("nan")
    r = np.asarray(iv_fit, float)[m] - np.asarray(iv_obs, float)[m]
    return float(np.sqrt(np.nanmean(r ** 2)))


def metric_price_vwrrmse(w_fit, w_obs, iv_fit, iv_obs, k, t, bid=None, ask=None):
    """
    Vega-weighted relative price RMSE — the natural companion to iv_zweighted.

    Combines two ideas:
      - *Relative* (%) error: (P_fit - P_obs) / P_obs, so a 1% mispricing on a
        cheap OTM option counts the same as a 1% mispricing on an expensive ATM
        option, instead of being drowned out in absolute terms.
      - *Vega weighting*: each relative error is scaled by vega, so strikes where
        a vol mis-calibration translates into meaningful P&L still dominate, and
        nearly-zero-delta deep wings (where prices are tiny and relative errors
        blow up numerically) are naturally suppressed.

    Formula:
        VWRRMSE_price = sqrt( sum_j[ v_j * ((P_fit_j - P_obs_j) / P_obs_j)^2 ]
                              / sum_j v_j )

    Connection to iv_zweighted
    --------------------------
    For small errors, ΔP ≈ vega * Δσ, so the relative price error satisfies
        ΔP/P ≈ (vega/P) * Δσ.
    iv_zweighted minimises a vol-space error weighted by vega^(1-2β); this metric
    evaluates the resulting *price* accuracy in percentage terms with one power of
    vega, making the two coherent companions: same vega bias, different space.
    """
    vega     = _bs_vega(k, iv_obs, t)
    p_fit    = _prices_from_iv(k, iv_fit, t)
    p_obs    = _prices_from_iv(k, iv_obs, t)
    rel_err  = (p_fit - p_obs) / np.maximum(p_obs, 1e-8)
    vsum     = np.nansum(vega)
    if vsum < 1e-30:
        return float(np.sqrt(np.nanmean(rel_err ** 2)))
    return float(np.sqrt(np.nansum(vega * rel_err ** 2) / vsum))


METRICS = {
    "iv_rmse":          metric_iv_rmse,
    "iv_rrmse":         metric_iv_rrmse,
    "price_rmse":       metric_price_rmse,
    "price_rrmse":      metric_price_rrmse,
    "price_vwrrmse":    metric_price_vwrrmse,
    "w_rmse":           metric_w_rmse,
    "max_iv_err":       metric_max_iv_err,
    "spread_hit_rate":  metric_spread_hit_rate,
    "vwrmse":           metric_iv_vwrmse,
    "vwrrmse":          metric_iv_vwrrmse,
    "iv_rmse_atm":      metric_iv_rmse_atm,
    "iv_rmse_otm":      metric_iv_rmse_otm,
}

# Default set reported after every calibration
DEFAULT_METRICS = ["iv_rrmse", "spread_hit_rate", "price_vwrrmse", "w_rmse",
                   "vwrmse", "vwrrmse", "iv_rmse_atm", "iv_rmse_otm"]


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


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS  (post-fit — evaluate whether the OTM-attention problem is fixed)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# BETA OPTIMISATION  (find the best wing-emphasis strength for a dataset)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_beta(
    df,
    calibrate_fn,
    calibrate_kwargs:  dict  = None,
    beta_grid:         list  = None,
    target_metric:     str   = "spread_hit_rate",
    maximize:          bool  = True,
    huber_delta:       float = None,
    density_correct:   bool  = False,
    plot:              bool  = True,
    verbose:           bool  = True,
):
    """
    Grid-search over beta in iv_zweighted to maximise/minimise a target metric,
    and compare the best beta against iv_uniform_z (the exact version).

    The grid evaluates *in-sample* performance — use this to understand the
    β-vs-metric curve and pick the knee, not as a guarantee of out-of-sample
    improvement.  For out-of-sample selection, run this on a held-out snapshot.

    Parameters
    ----------
    df              : cleaned snapshot DataFrame (output of load_snapshot / clean_df)
    calibrate_fn    : one of calibrate_snapshot, calibrate_global_essvi, etc.
    calibrate_kwargs: extra kwargs forwarded to calibrate_fn (model, bid_col, …)
                      Do NOT include `objective` — it is set by this function.
    beta_grid       : list of beta values to try.  Default: 0, 0.1, …, 0.6.
    target_metric   : metric name to optimise (must be in METRICS).
    maximize        : True = higher is better (spread_hit_rate);
                      False = lower is better (vwrmse, price_vwrrmse, …).
    huber_delta     : optional Huber delta forwarded to iv_zweighted.
    density_correct : whether to apply the dz density correction in iv_zweighted.
    plot            : if True, draw the metric-vs-beta curve.

    Returns
    -------
    dict with keys
        best_beta        : float — optimal beta from the grid
        best_score       : float — metric value at best_beta
        uniform_z_score  : float — same metric for iv_uniform_z (exact version)
        beta_scores      : dict  {beta: score} for the full grid
        results          : dict  {beta: calibration result} for all betas
                           plus 'uniform_z' for the exact objective
    """
    if beta_grid is None:
        beta_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    kw = dict(calibrate_kwargs or {})
    kw.setdefault("verbose", False)

    def _score(result):
        vals = result["metrics"].get(target_metric, [])
        if not vals:
            return float("nan")
        v = np.nanmean(vals)
        return float(v)

    beta_scores = {}
    all_results = {}

    if verbose:
        print(f"Optimising beta for '{target_metric}'  "
              f"({'max' if maximize else 'min'})\n")
        print(f"  {'beta':>6}  {'score':>10}")
        print("  " + "─" * 20)

    for beta in beta_grid:
        obj = make_iv_zweighted(beta=beta, huber_delta=huber_delta,
                                density_correct=density_correct)
        r   = calibrate_fn(df, objective=obj, **kw)
        s   = _score(r)
        beta_scores[beta] = s
        all_results[beta] = r
        if verbose:
            print(f"  {beta:>6.2f}  {s:>10.4f}")

    # Also run the exact uniform-z objective for direct comparison
    if verbose:
        print(f"  {'unif-z':>6}  ", end="")
    obj_uz  = make_iv_uniform_z(huber_delta=huber_delta)
    r_uz    = calibrate_fn(df, objective=obj_uz, **kw)
    s_uz    = _score(r_uz)
    all_results["uniform_z"] = r_uz
    if verbose:
        print(f"{s_uz:>10.4f}  ← iv_uniform_z (exact)")

    best_beta  = max(beta_scores, key=beta_scores.__getitem__) if maximize \
                 else min(beta_scores, key=beta_scores.__getitem__)
    best_score = beta_scores[best_beta]

    if verbose:
        print(f"\n  Best beta = {best_beta}  ({target_metric}={best_score:.4f})")
        diff = s_uz - best_score if maximize else best_score - s_uz
        direction = "better" if diff > 0 else "worse"
        print(f"  iv_uniform_z vs best beta: Δ={s_uz - best_score:+.4f} "
              f"({'uniform_z ' + direction})")

    if plot:
        import matplotlib.pyplot as plt
        betas  = list(beta_scores.keys())
        scores = [beta_scores[b] for b in betas]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(betas, scores, 'o-', color='steelblue', lw=2, ms=7,
                markeredgecolor='white', markeredgewidth=1.2, label='iv_zweighted(β)')
        ax.axhline(s_uz, color='tomato', ls='--', lw=1.8,
                   label=f'iv_uniform_z (exact) = {s_uz:.4f}')
        ax.axvline(best_beta, color='steelblue', ls=':', lw=1.2, alpha=0.7)
        ax.scatter([best_beta], [best_score], color='gold', s=120, zorder=5,
                   edgecolors='steelblue', linewidths=1.5,
                   label=f'best β={best_beta} ({best_score:.4f})')

        ax.set_xlabel('β  (wing-emphasis strength in iv_zweighted)', fontsize=10)
        ax.set_ylabel(target_metric, fontsize=10)
        ax.set_title(f'Beta optimisation  —  {target_metric}', fontsize=12,
                     fontweight='bold')
        # ax.text(0.5, 1.02,
        #         'Dashed red = exact iv_uniform_z  •  gold star = optimal β',
        #         transform=ax.transAxes, ha='center', va='bottom',
        #         fontsize=9, color='gray')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('beta_optimisation.png', dpi=150, bbox_inches='tight')
        plt.show()

    return {
        "best_beta":        best_beta,
        "best_score":       best_score,
        "uniform_z_score":  s_uz,
        "beta_scores":      beta_scores,
        "results":          all_results,
    }
