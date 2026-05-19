"""
svi_objective_functions.py
==========================
Drop-in objective functions for SVI snapshot calibration.

Designed to plug directly into svi_snapshot_calibration.py via the
`objective_fn` parameter added to fit_single_slice().

Each objective function has the same signature:

    obj_fn(w_model, w_obs, iv_model, iv_obs, vega, bid_ask_width) -> float

Where all arrays are aligned (same strikes, same slice).

Usage
-----
    from svi_objective_functions import OBJECTIVES, build_objective
    from svi_snapshot_calibration import fit_single_slice

    params = fit_single_slice(
        k_obs, w_obs, t,
        iv_obs=iv_obs,
        vega=vega,
        bid_ask_width=spread,
        objective_fn="vega_weighted_iv",   # <-- swap here
    )

Objective keys
--------------
    "sse_total_var"         SSE on total variance w (current default in your code)
    "sse_iv"                SSE on implied vol (uniform weights)
    "vega_weighted_iv"      Vega-weighted SSE on implied vol  [recommended]
    "bid_ask_normalized"    SSE normalised by bid-ask half-spread
    "price_sse"             SSE on Black-Scholes option prices (mid)
    "combined"              Vega-weighted IV + bid-ask normalised, equal weight

Notes on arbitrage penalties
-----------------------------
Arbitrage penalties (butterfly, calendar) are handled upstream in fit_single_slice
and are objective-function agnostic.  The functions here return only the *fit*
component of the loss.  The caller (fit_single_slice) adds penalties on top.
"""

import numpy as np
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bs_price(k: np.ndarray, iv: np.ndarray, t: float,
              is_call: bool = True) -> np.ndarray:
    """
    Undiscounted Black-Scholes price for a forward-normalised option.

    F = 1  (prices expressed as fractions of the forward)
    K = exp(k)

    Parameters
    ----------
    k       : log-strike array  log(K/F)
    iv      : implied vol array
    t       : time to expiry (scalar)
    is_call : True for calls, False for puts

    Returns
    -------
    array of undiscounted BS prices (fraction of forward)
    """
    k  = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    safe_iv = np.maximum(iv, 1e-8)
    sqrt_t  = np.sqrt(max(t, 1e-8))

    d1 = (-k + 0.5 * safe_iv ** 2 * t) / (safe_iv * sqrt_t)
    d2 = d1 - safe_iv * sqrt_t

    if is_call:
        return norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
    else:
        return np.exp(k) * norm.cdf(-d2) - norm.cdf(-d1)


def _bs_vega(k: np.ndarray, iv: np.ndarray, t: float) -> np.ndarray:
    """
    Black-Scholes vega (∂price/∂σ) for a forward-normalised option.
    Same for calls and puts.

    Returns vega per unit of the forward.
    """
    k  = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    safe_iv = np.maximum(iv, 1e-8)
    sqrt_t  = np.sqrt(max(t, 1e-8))

    d1 = (-k + 0.5 * safe_iv ** 2 * t) / (safe_iv * sqrt_t)
    return norm.pdf(d1) * sqrt_t          # always positive


# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE FUNCTIONS
# Each takes full context; unused args are ignored via **kwargs pattern.
# ─────────────────────────────────────────────────────────────────────────────

def sse_total_var(w_model, w_obs, **kwargs) -> float:
    """
    SSE on total implied variance  w = σ² · t.

    This is the current default in svi_snapshot_calibration.py.
    Fitting in variance space is natural for SVI (which is parameterised in w),
    but errors in w do not map cleanly to P&L.

    Bias: treats a 1bp² variance error the same regardless of moneyness or t.
    """
    return float(np.nanmean((w_model - w_obs) ** 2))


def sse_iv(iv_model, iv_obs, **kwargs) -> float:
    """
    Uniform SSE on implied volatility  σ_BS.

    Simple and interpretable (errors in vol points), but treats a 1-vol-point
    miss ATM the same as at a 5-delta wing.  ATM errors dominate P&L in
    practice, so this is biased against where money is made.

    Use case: quick diagnostic, not production calibration.
    """
    return float(np.nanmean((iv_model - iv_obs) ** 2))


def vega_weighted_iv(iv_model, iv_obs, vega, **kwargs) -> float:
    """
    Vega-weighted SSE on implied vol.

    Loss = mean( vega_i * (σ_model_i - σ_obs_i)² )

    Rationale
    ---------
    Since  ΔPrice ≈ vega · Δσ,  this objective minimises a first-order
    approximation of squared price errors.  It puts calibration effort where
    option P&L is actually sensitive.

    This is the most economically coherent single-term objective for
    a market-maker whose risk is predominantly vega-driven.

    Critical note: vega-weighting is NOT the same as price-SSE — they differ
    by nonlinear correction terms, which matter most in wings with high gamma.
    """
    vega = np.asarray(vega, dtype=float)
    w    = np.maximum(vega, 1e-12)          # avoid zero weights for deep OTM
    return float(np.nansum(w * (iv_model - iv_obs) ** 2) / np.nansum(w))


def bid_ask_normalized(iv_model, iv_obs, bid_ask_width, **kwargs) -> float:
    """
    SSE on implied vol, normalised by the bid-ask half-spread.

    Loss = mean( ((σ_model - σ_mid) / (0.5 * spread))² )

    Rationale
    ---------
    The bid-ask spread is the market's own signal of uncertainty at each strike.
    A fit error within the spread is "free"; one outside is penalised
    proportionally to how far it violates the market.

    This is statistically the most defensible objective — it is a maximum
    likelihood estimator under the assumption that mid-IV is uniformly
    distributed within the spread.

    Practical caveat: crypto spreads (your BTC data) are often stale or very
    wide for OTM strikes, which can under-constrain the wings.  Apply a floor
    on the spread to avoid degenerate weights.

    Parameters
    ----------
    bid_ask_width : array  bid-ask spread IN IV POINTS  (ask_iv - bid_iv)
                           NOT half-spread — we divide by 2 internally.
    """
    half_spread = np.maximum(np.asarray(bid_ask_width, dtype=float) / 2.0,
                             1e-4)           # floor: 0.01 vol point minimum
    normalised  = (iv_model - iv_obs) / half_spread
    return float(np.nanmean(normalised ** 2))


def price_sse(k_obs, iv_model, iv_obs, t, **kwargs) -> float:
    """
    SSE on undiscounted Black-Scholes mid prices (forward-normalised).

    Loss = mean( (C_model(k) - C_obs(k))² )

    Rationale
    ---------
    Prices are the actual traded quantities.  Minimising price errors is the
    most literal interpretation of "fit the market".

    Critical defect: OTM option prices are tiny, so the objective is
    dominated by near-ATM options.  Wings (which carry tail and skew
    information) are effectively ignored.

    Mitigation: combine with a wing-specific penalty if you care about
    risk-neutral density shape.

    Implementation note: we use calls for k >= 0, puts for k < 0 (forward-
    normalised put-call parity), which is standard for avoiding near-zero
    prices on both sides.
    """
    k_obs    = np.asarray(k_obs, dtype=float)
    iv_model = np.asarray(iv_model, dtype=float)
    iv_obs   = np.asarray(iv_obs, dtype=float)

    call_mask = k_obs >= 0.0
    put_mask  = ~call_mask

    price_model = np.empty_like(k_obs)
    price_obs   = np.empty_like(k_obs)

    if call_mask.any():
        price_model[call_mask] = _bs_price(k_obs[call_mask], iv_model[call_mask], t, is_call=True)
        price_obs[call_mask]   = _bs_price(k_obs[call_mask], iv_obs[call_mask],   t, is_call=True)

    if put_mask.any():
        price_model[put_mask] = _bs_price(k_obs[put_mask], iv_model[put_mask], t, is_call=False)
        price_obs[put_mask]   = _bs_price(k_obs[put_mask], iv_obs[put_mask],   t, is_call=False)

    return float(np.nanmean((price_model - price_obs) ** 2))


def combined(iv_model, iv_obs, vega, bid_ask_width, **kwargs) -> float:
    """
    Equal-weight combination of vega_weighted_iv and bid_ask_normalized.

    This balances two complementary objectives:
    - vega_weighted_iv   : prioritises where P&L risk is largest
    - bid_ask_normalized : respects the market's own uncertainty signal

    In practice this hybrid is robust: vega-weighting anchors the ATM,
    bid-ask normalisation keeps the wings from being over-fitted.

    alpha controls the blend (0 = pure vega-weighted, 1 = pure bid-ask).
    Default alpha = 0.5.
    """
    alpha = kwargs.get("alpha", 0.5)
    v = vega_weighted_iv(iv_model, iv_obs, vega)
    b = bid_ask_normalized(iv_model, iv_obs, bid_ask_width)
    return (1.0 - alpha) * v + alpha * b


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY  — maps string keys to functions
# ─────────────────────────────────────────────────────────────────────────────

OBJECTIVES = {
    "sse_total_var":      sse_total_var,
    "sse_iv":             sse_iv,
    "vega_weighted_iv":   vega_weighted_iv,
    "bid_ask_normalized": bid_ask_normalized,
    "price_sse":          price_sse,
    "combined":           combined,
}


def build_objective(name: str):
    """
    Return the objective function by name.  Raises KeyError with helpful message.
    """
    if name not in OBJECTIVES:
        raise KeyError(
            f"Unknown objective '{name}'. "
            f"Available: {list(OBJECTIVES.keys())}"
        )
    return OBJECTIVES[name]


# ─────────────────────────────────────────────────────────────────────────────
# MODIFIED fit_single_slice — drop-in replacement
# Copy this into svi_snapshot_calibration.py (replaces the existing function).
# ─────────────────────────────────────────────────────────────────────────────

def fit_single_slice_with_obj(
        k_obs, w_obs, t,
        iv_obs=None,
        vega=None,
        bid_ask_width=None,
        prev_params=None,
        next_params=None,
        penalty_cal=500.0,
        use_global_init=True,
        objective_fn="sse_total_var",
        obj_alpha=0.5,
        # ── imports from svi_snapshot_calibration ──
        _unpack=None,
        _pack=None,
        svi_raw_dict=None,
        min_g_value=None,
        crossedness=None,
        differential_evolution=None,
        minimize=None,
):
    """
    Replacement for fit_single_slice() that accepts any objective function.

    New parameters
    --------------
    iv_obs        : array  observed implied vols (σ_BS).  Required for all
                           objectives except 'sse_total_var'.
    vega          : array  BS vega at each strike.  Required for
                           'vega_weighted_iv' and 'combined'.
                           Computed from iv_obs internally if not provided.
    bid_ask_width : array  bid-ask spread in IV points.  Required for
                           'bid_ask_normalized' and 'combined'.
                           Falls back to a uniform 2-vol-point spread if None.
    objective_fn  : str or callable
                    One of the keys in OBJECTIVES, or your own function with
                    the signature described at the top of this file.
    obj_alpha     : float  blend parameter for 'combined' objective [0, 1].

    All other parameters are identical to the original fit_single_slice().

    Note
    ----
    This function intentionally accepts the helper functions from
    svi_snapshot_calibration as keyword arguments so it can be used standalone
    or pasted into that module.  When pasting, remove those kwargs and call
    the helpers directly.
    """
    import numpy as _np

    k_obs = _np.asarray(k_obs, dtype=float)
    w_obs = _np.asarray(w_obs, dtype=float)

    # ── Derive iv_obs from w_obs if not provided ───────────────────────────
    if iv_obs is None:
        iv_obs = _np.sqrt(_np.maximum(w_obs / max(t, 1e-8), 0.0))
    iv_obs = _np.asarray(iv_obs, dtype=float)

    # ── Compute vega if not provided ───────────────────────────────────────
    if vega is None:
        vega = _bs_vega(k_obs, iv_obs, t)
    vega = _np.asarray(vega, dtype=float)

    # ── Default bid-ask: uniform 2 vol-point spread if not provided ────────
    if bid_ask_width is None:
        bid_ask_width = _np.full_like(k_obs, 0.02)
    bid_ask_width = _np.asarray(bid_ask_width, dtype=float)

    # ── Resolve objective function ─────────────────────────────────────────
    if callable(objective_fn):
        _obj_fn = objective_fn
    else:
        _obj_fn = build_objective(objective_fn)

    # ATM anchor
    atm_w = float(_np.interp(0.0, _np.sort(k_obs), w_obs[_np.argsort(k_obs)]))

    def objective(x):
        p     = _unpack(x)
        w_fit = svi_raw_dict(k_obs, p)

        # Convert fitted w to iv for objectives that need it
        iv_fit = _np.sqrt(_np.maximum(w_fit / max(t, 1e-8), 0.0))

        # ── Core fit term (objective-specific) ────────────────────────────
        fit_err = _obj_fn(
            w_model       = w_fit,
            w_obs         = w_obs,
            iv_model      = iv_fit,
            iv_obs        = iv_obs,
            vega          = vega,
            bid_ask_width = bid_ask_width,
            k_obs         = k_obs,
            t             = t,
            alpha         = obj_alpha,
        )

        # ── Penalty: negative total variance ──────────────────────────────
        min_w   = p["a"] + p["b"] * p["sig"] * _np.sqrt(1 - p["rho"] ** 2)
        neg_pen = max(0.0, -min_w) * 1e4

        # ── Penalty: butterfly arbitrage (soft) ───────────────────────────
        mg      = min_g_value(**p)
        but_pen = max(0.0, -mg) * 1e3

        # ── Penalty: calendar spread with neighbours ───────────────────────
        cal_pen = 0.0
        if prev_params is not None:
            cal_pen += crossedness(prev_params, p) * penalty_cal
        if next_params is not None:
            cal_pen += crossedness(p, next_params) * penalty_cal

        return fit_err + neg_pen + but_pen + cal_pen

    # ── Initial guess ──────────────────────────────────────────────────────
    x0 = _pack(atm_w * 0.9, 0.1, -0.7, 0.0, 0.3)

    if use_global_init:
        bounds = [
            (-0.5, atm_w * 2),
            (_np.log(1e-4), _np.log(5.0)),
            (-3.0, 3.0),
            (-2.0, 2.0),
            (_np.log(1e-4), _np.log(3.0)),
        ]
        de_res = differential_evolution(
            objective, bounds,
            seed=42, maxiter=300, tol=1e-7,
            popsize=8, mutation=(0.5, 1.5), recombination=0.9,
            workers=1,
        )
        x0 = de_res.x

    # ── Local polish ───────────────────────────────────────────────────────
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-9, "fatol": 1e-9})

    return _unpack(res.x)


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def compare_objectives(k_obs, w_obs, t, iv_obs=None,
                       vega=None, bid_ask_width=None) -> dict:
    """
    Given a set of observed strikes/variances, compute what each objective
    function returns for the current model values.  Useful for diagnostics:
    run this after calibration to see how each metric looks on the fitted slice.

    Parameters
    ----------
    k_obs, w_obs, t : as in fit_single_slice
    iv_obs          : implied vols (computed from w_obs if None)
    vega            : BS vega (computed if None)
    bid_ask_width   : spread in IV points (defaults to 0.02 if None)

    Returns
    -------
    dict  {objective_name: value}
    """
    k_obs = np.asarray(k_obs, dtype=float)
    w_obs = np.asarray(w_obs, dtype=float)

    if iv_obs is None:
        iv_obs = np.sqrt(np.maximum(w_obs / max(t, 1e-8), 0.0))
    if vega is None:
        vega = _bs_vega(k_obs, iv_obs, t)
    if bid_ask_width is None:
        bid_ask_width = np.full_like(k_obs, 0.02)

    # Residuals: compare obs vs obs (should be zero — useful as a smoke test)
    # In production: pass w_model from a fitted SVI slice instead of w_obs
    ctx = dict(
        w_model       = w_obs,
        w_obs         = w_obs,
        iv_model      = iv_obs,
        iv_obs        = iv_obs,
        vega          = vega,
        bid_ask_width = bid_ask_width,
        k_obs         = k_obs,
        t             = t,
        alpha         = 0.5,
    )
    return {name: float(fn(**ctx)) for name, fn in OBJECTIVES.items()}


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("svi_objective_functions — self test")
    print("=" * 60)

    # Synthetic slice: T=0.25, flat smile at 50% vol
    rng  = np.random.default_rng(0)
    t    = 0.25
    k    = np.linspace(-0.5, 0.5, 15)
    iv   = 0.50 + 0.05 * k ** 2 + rng.normal(0, 0.005, len(k))
    w    = iv ** 2 * t
    vega = _bs_vega(k, iv, t)
    sprd = np.maximum(0.04 - 0.02 * np.exp(-k ** 2 / 0.1), 0.01)  # wider wings

    # Perturb model slightly
    iv_model = iv + rng.normal(0, 0.01, len(k))
    w_model  = iv_model ** 2 * t

    ctx = dict(
        w_model=w_model, w_obs=w,
        iv_model=iv_model, iv_obs=iv,
        vega=vega, bid_ask_width=sprd,
        k_obs=k, t=t, alpha=0.5,
    )

    print(f"\n{'Objective':<25}  {'Value':>12}")
    print("-" * 40)
    for name, fn in OBJECTIVES.items():
        val = fn(**ctx)
        print(f"{name:<25}  {val:12.6f}")

    print("\nAll objectives computed successfully.")
    print("\nAvailable objective keys:")
    for k_ in OBJECTIVES:
        print(f"  '{k_}'")
