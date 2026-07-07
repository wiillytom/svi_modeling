"""
calibrator.py
=============
Model-agnostic calibration engine for a single order-book snapshot.

The engine knows nothing about the specific model or objective function.
It receives a VolModel instance, an objective function, and a dataframe,
and returns a standardised result dict that the plotting functions consume.

Usage
-----
    from vol_models  import get_model
    from objectives  import get_objective
    from calibrator  import load_snapshot, calibrate_snapshot, summary

    df    = load_snapshot("path/to/daily_options_data.parquet")
    model = get_model("svi")                   # or "ssvi", "sabr"
    obj   = get_objective("iv_wmse")           # or "w_mse", "price_mse", ...

    result = calibrate_snapshot(df, model=model, objective=obj)
    summary(result)

    # Compare two models on the same snapshot
    result_svi  = calibrate_snapshot(df, model=get_model("svi"))
    result_sabr = calibrate_snapshot(df, model=get_model("sabr", beta=1.0))
    compare(result_svi, result_sabr)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, minimize_scalar
import warnings
import time as _time
warnings.filterwarnings("ignore")

from volatility_surface.models.vol_models import VolModel, RawSVI, SABR, get_model
from volatility_surface.core.calibration.objectives import (
    get_objective, evaluate_all, DEFAULT_METRICS, _standardized_moneyness,
)


# ─────────────────────────────────────────────────────────────────────────────
# CALENDAR SPREAD CHECK  (model-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def crossedness(model: VolModel, p1: dict, p2: dict,
                k_range=(-5.0, 5.0), n=2000) -> float:
    """
    Maximum amount by which slice p1 (earlier expiry) exceeds slice p2 (later).
    Zero means no calendar spread arbitrage between these two slices.
    Works for any model.
    """
    k_grid = np.linspace(k_range[0], k_range[1], n)
    diff   = model.w(k_grid, p1) - model.w(k_grid, p2)
    return float(max(0.0, diff.max()))


# ─────────────────────────────────────────────────────────────────────────────
# ARRAY BINDING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

import inspect as _inspect

def _bind_extra(objective, vegas=None, bid=None, ask=None):
    """
    Bind precomputed per-slice arrays (vegas, bid, ask) to an objective in a
    single pass.

    Inspects the *original* objective's signature once, then builds one wrapper
    that injects only the arrays the objective actually declares.

    Why a single function matters
    -----------------------------
    Chaining _bind_vegas then _bind_bid_ask breaks silently: the wrapper
    produced by the first call has a fixed signature that hides the inner
    function's parameter names from the second call's inspector, so the second
    binding is always a no-op.  Inspecting the original objective before any
    wrapping avoids this entirely.
    """
    sig        = _inspect.signature(objective).parameters
    want_vegas = vegas is not None and "vegas" in sig
    want_ba    = (bid is not None and ask is not None
                  and "bid" in sig and "ask" in sig)

    if not want_vegas and not want_ba:
        return objective          # nothing to bind — return original unchanged

    _obj = objective
    _v   = np.asarray(vegas, dtype=float) if want_vegas else None
    _b   = np.asarray(bid,   dtype=float) if want_ba   else None
    _a   = np.asarray(ask,   dtype=float) if want_ba   else None

    def _bound(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=None):
        kw = {}
        if _v is not None:
            kw["vegas"] = _v
        if _b is not None:
            kw["bid"] = _b
            kw["ask"] = _a
        return _obj(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads, **kw)

    _bound.__name__ = getattr(objective, "__name__", "objective")
    return _bound


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-SLICE CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def fit_single_slice(
    k_obs:        np.ndarray,
    w_obs:        np.ndarray,
    iv_obs:       np.ndarray,
    t:            float,
    model:        VolModel,
    objective,
    spreads:      np.ndarray = None,
    prev_params:  dict = None,
    next_params:  dict = None,
    penalty_cal:  float = 500.0,
    penalty_but:  float = 200.0,
    use_global_init: bool = True,
    init_params:  dict = None,
    nm_maxiter:   int   = 5000,
    nm_tol:       float = 1e-9,
) -> dict:
    """
    Fit one expiry slice using the given model and objective.

    Parameters
    ----------
    k_obs, w_obs, iv_obs : observed log-strikes, total variances, implied vols
    t            : time to expiry
    model        : VolModel instance
    objective    : callable(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads) -> float
    spreads      : bid-ask vol spreads for weighted objectives (optional)
    prev_params  : fitted params for the immediately shorter expiry (calendar constraint)
    next_params  : fitted params for the immediately longer expiry (calendar constraint)
    penalty_cal  : weight on calendar spread violation penalty
    penalty_but  : weight on butterfly arbitrage violation penalty
    use_global_init : use differential_evolution for a robust global start
    init_params  : optional warm-start parameter dict.  When given it overrides
                   model.initial_guess as the starting point (pair with
                   use_global_init=False for a pure local polish from a known
                   good fit, e.g. the Pareto pass-2 refit).

    Returns
    -------
    dict of fitted parameters
    """
    k_obs  = np.asarray(k_obs,  dtype=float)
    w_obs  = np.asarray(w_obs,  dtype=float)
    iv_obs = np.asarray(iv_obs, dtype=float)

    # For SABR and other models that need t inside params
    def _inject_t(params):
        if "t" in model.initial_guess(k_obs, w_obs, t):
            return {**params, "t": t}
        return params

    def objective_fn(x):
        params  = _inject_t(model.unpack(x))
        w_fit   = model.w(k_obs, params)
        iv_fit  = model.iv(k_obs, params, t)

        # Primary objective
        fit_err = objective(w_fit, w_obs, iv_fit, iv_obs, k_obs, t, spreads)

        # Penalty: negative total variance
        min_w   = float(np.min(w_fit))
        neg_pen = max(0.0, -min_w) * 1e4

        # Penalty: butterfly arbitrage (soft)
        mg      = model.min_g(params)
        but_pen = max(0.0, -mg) * penalty_but

        # Penalty: calendar spread with neighbours
        cal_pen = 0.0
        if prev_params is not None:
            cal_pen += crossedness(model, prev_params, params) * penalty_cal
        if next_params is not None:
            cal_pen += crossedness(model, params, next_params) * penalty_cal

        return fit_err + neg_pen + but_pen + cal_pen

    # ── Initial guess ─────────────────────────────────────────────────────────
    p0 = init_params if init_params is not None else model.initial_guess(k_obs, w_obs, t)
    x0 = model.pack(p0)

    if use_global_init:
        bounds = model.bounds()
        de_res = differential_evolution(
            objective_fn, bounds,
            seed=42, maxiter=300, tol=1e-7,
            popsize=8, mutation=(0.5, 1.5), recombination=0.9,
            workers=1,
        )
        x0 = de_res.x

    # ── Local polish ──────────────────────────────────────────────────────────
    res = minimize(
        objective_fn, x0, method="Nelder-Mead",
        options={"maxiter": nm_maxiter, "xatol": nm_tol, "fatol": nm_tol},
    )
    return _inject_t(model.unpack(res.x))


# ─────────────────────────────────────────────────────────────────────────────
# SNAPSHOT LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_snapshot(path: str,
                  timestamp=None,
                  iv_col: str = "mark_iv",
                  forward_col: str = None,
                  strike_col: str = None) -> pd.DataFrame:
    """
    Load a parquet file and extract a single order-book snapshot.
    Returns a DataFrame with columns  k, w, t, mark_iv  ready for calibration.
    """
    df = pd.read_parquet(path)

    if "file_timestamp" not in df.columns:
        raise ValueError("Column 'file_timestamp' not found.")

    if timestamp is None:
        timestamp = df["file_timestamp"].iloc[0]
        print(f"Using first snapshot: {timestamp}")
    df = df[df["file_timestamp"] == timestamp].copy()
    print(f"  {len(df)} rows in snapshot")

    # Time to maturity
    if "t" not in df.columns:
        if "expiry" in df.columns:
            ref = pd.to_datetime(timestamp)
            df["t"] = (pd.to_datetime(df["expiry"]) - ref).dt.total_seconds() / (365.25 * 24 * 3600)
        elif "time_to_maturity" in df.columns:
            df["t"] = df["time_to_maturity"]
        else:
            raise ValueError("Cannot determine time to maturity.")

    df["t"] = df["t"].round(4)   # collapse near-duplicate expiries

    # Forward price
    if forward_col is None:
        for col in ["forward", "future_price", "underlying_price", "index_price"]:
            if col in df.columns:
                forward_col = col
                break

    # Strike
    if strike_col is None:
        for col in ["strike", "strike_price"]:
            if col in df.columns:
                strike_col = col
                break

    # Log-strike
    if "k" not in df.columns:
        if forward_col and strike_col:
            df["k"] = np.log(df[strike_col] / df[forward_col])
        else:
            raise ValueError("Cannot compute log-strike.")

    # Implied vol
    if df[iv_col].max() > 5:
        df[iv_col] = df[iv_col] / 100.0
    df["mark_iv"] = df[iv_col]

    # Total variance
    if "w" not in df.columns:
        df["w"] = df["mark_iv"] ** 2 * df["t"]

    # Clean
    before = len(df)
    df = df[(df["t"] > 1e-5) & (df["mark_iv"] > 1e-4) & (df["mark_iv"] < 5.0) & (df["w"] > 0)]
    df = df.dropna(subset=["k", "w", "t", "mark_iv"])
    print(f"  {len(df)} rows after cleaning (dropped {before - len(df)})")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# FULL SNAPSHOT CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_snapshot(
    df:                   pd.DataFrame,
    model:                VolModel = None,
    objective                      = None,
    metrics:              list     = None,
    penalty_cal:          float    = 500.0,
    penalty_but:          float    = 200.0,
    min_points_per_slice: int      = 5,
    use_global_init:      bool     = True,
    spread_col:           str      = None,
    vega_col:             str      = None,
    bid_col:              str      = None,
    ask_col:              str      = None,
    enforce_arbfree:      bool     = False,
    arbfree_tol:          float    = 1e-6,
    obloj:                bool     = False,
    verbose:              bool     = True,
) -> dict:
    """
    Calibrate the vol surface for every expiry in a single snapshot.

    Parameters
    ----------
    df            : output of load_snapshot()
    model         : VolModel instance (default: RawSVI)
    objective     : callable or str  (default: "iv_wmse")
    metrics       : list of metric name strings to evaluate after fit
    spread_col    : column in df with bid-ask vol spread (optional)
    penalty_cal   : calendar spread penalty weight
    penalty_but   : butterfly arbitrage penalty weight
    enforce_arbfree : if True, runs enforce_calendar_arbfree on the result —
                      if any adjacent pair still has crossedness > arbfree_tol,
                      the surface is refitted jointly with strong calendar
                      enforcement (model-specific path; currently supported
                      for SABR). Use with per-slice models like SABR.
    arbfree_tol   : tolerance used by enforce_arbfree (default 1e-6)
    obloj         : SABR only — use Obłój (2008)'s corrected z(K, F) in place
                    of Hagan (2002)'s geometric-mean form.  If the supplied
                    `model` is SABR, this flag toggles its `obloj` attribute
                    for the duration of the call.  If `model` is None, a SABR
                    instance is constructed with `obloj=True`.
                    NOTE: for β = 1 (the SABR default) Hagan and Obłój
                    coincide exactly — this flag only changes results when
                    β < 1.

    Returns
    -------
    dict with keys:
        model_name  : str
        objective   : str
        expiries    : list of floats
        params      : list of param dicts  (one per expiry)
        metrics     : dict of {metric_name: list of per-slice values}
        n_points    : list of ints
    """
    if model is None:
        # When obloj=True with no explicit model, default to a SABR(obloj=True)
        # rather than RawSVI — the flag is SABR-specific so it would have no
        # effect on RawSVI and the user's intent would be silently lost.
        model = SABR(obloj=True) if obloj else RawSVI()
    elif obloj:
        # Honour the obloj flag if the supplied model knows about it.
        if isinstance(model, SABR):
            model.obloj = True
        elif verbose:
            print(f"  [obloj=True ignored — {model.name} has no obloj formula]")
    if objective is None:
        objective = get_objective("iv_wmse")
    elif isinstance(objective, str):
        objective = get_objective(objective)

    metric_names = metrics or DEFAULT_METRICS
    expiries     = sorted(df["t"].unique())
    n            = len(expiries)

    if verbose:
        _formula = ""
        if isinstance(model, SABR):
            _formula = "  (Obłój)" if model.obloj else "  (Hagan)"
            if model.obloj and abs(model.beta - 1.0) < 1e-12:
                _formula += " — no-op at β=1"
        print(f"\nModel     : {model.name}{_formula}")
        print(f"Objective : {objective.__name__}")
        print(f"Slices    : {n}\n")

    fitted_params = [None] * n
    slice_metrics = {m: [] for m in metric_names}

    for i, t_exp in enumerate(expiries):
        subset = df[df["t"] == t_exp]

        if len(subset) < min_points_per_slice:
            if verbose:
                print(f"  [{i+1}/{n}] T={t_exp:.4f}  SKIPPED ({len(subset)} pts)")
            fitted_params[i] = fitted_params[i-1] if i > 0 else None
            for m in metric_names:
                slice_metrics[m].append(np.nan)
            continue

        k_obs  = subset["k"].values
        w_obs  = subset["w"].values
        iv_obs = subset["mark_iv"].values
    
        spreads = subset[spread_col].values if spread_col and spread_col in df.columns else None
        vegas   = subset[vega_col].values   if vega_col  and vega_col  in df.columns else None
        bid     = subset[bid_col].values    if bid_col   and bid_col   in df.columns else None
        ask     = subset[ask_col].values    if ask_col   and ask_col   in df.columns else None

        prev_p = fitted_params[i-1] if i > 0 and fitted_params[i-1] is not None else None

        slice_obj = _bind_extra(objective, vegas=vegas, bid=bid, ask=ask)
        params = fit_single_slice(
            k_obs, w_obs, iv_obs, t_exp,
            model=model,
            objective=slice_obj,
            spreads=spreads,
            prev_params=prev_p,
            next_params=None,
            penalty_cal=penalty_cal,
            penalty_but=penalty_but,
            use_global_init=use_global_init,
        )
        fitted_params[i] = params

        # Evaluate metrics
        w_fit  = model.w(k_obs, params)
        iv_fit = model.iv(k_obs, params, t_exp)
        evals  = evaluate_all(w_fit, w_obs, iv_fit, iv_obs, k_obs, t_exp,
                               bid=bid, ask=ask, metric_names=metric_names)
        for m in metric_names:
            slice_metrics[m].append(evals[m])

        if verbose:
            mg    = model.min_g(params)
            cross = crossedness(model, prev_p, params) if prev_p else 0.0
            flag  = "⚠ butterfly" if mg < 0 else "ok"
            met_str = "  ".join(f"{m}={v:.4f}" for m, v in evals.items())
            print(f"  [{i+1}/{n}] T={t_exp:.4f}  n={len(subset):3d}  "
                  f"cal={cross:.1e}  [{flag}]  {met_str}")

    # Filter skipped slices
    valid       = [(t, p) for t, p in zip(expiries, fitted_params) if p is not None]
    valid_idx   = [i for i, p in enumerate(fitted_params) if p is not None]
    expiries_out = [v[0] for v in valid]
    params_out   = [v[1] for v in valid]
    metrics_out  = {m: [slice_metrics[m][i] for i in valid_idx] for m in metric_names}
    n_pts        = [len(df[df["t"] == t]) for t in expiries_out]

    result = {
        "model_name": model.name,
        "objective":  objective.__name__,
        "expiries":   expiries_out,
        "params":     params_out,
        "metrics":    metrics_out,
        "n_points":   n_pts,
        "_model":     model,   # kept for plotting
    }

    if enforce_arbfree and len(params_out) >= 2:
        result = enforce_calendar_arbfree(
            result, df,
            objective=objective,
            tol=arbfree_tol,
            verbose=verbose,
            vega_col=vega_col,
            bid_col=bid_col,
            ask_col=ask_col,
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL WARM-START WORKER  (module-level so joblib/loky can pickle it)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_ssvi_warm_start_slice(sl, ssvi_model, objective, use_de=True):
    """
    Fit a single per-slice SSVI for warm-starting global eSSVI/SSVI.
    Module-level so the loky backend can pickle it without cloudpickle.

    use_de=True  → differential_evolution + NM polish  (~1–2 s per slice)
    use_de=False → NM only from initial guess          (~0.1 s per slice)
                   Sufficient when many slices are available and a rough
                   warm-start is fine.
    """
    try:
        p = fit_single_slice(
            sl["k"], sl["w"], sl["iv"], sl["t"],
            model=ssvi_model,
            objective=objective,
            use_global_init=use_de,
        )
        return float(p["theta"]), float(p["eta"]), float(p["gamma"]), float(p["rho"])
    except Exception:
        theta = float(np.interp(0.0,
                                np.sort(sl["k"]),
                                sl["w"][np.argsort(sl["k"])]))
        return theta, 1.5, 0.4, -0.7


def calibrate_global_essvi(
    df:                pd.DataFrame,
    model,                             # eSSVI instance
    objective          = None,
    penalty_cal:       float = 500.0,
    penalty_but:       float = 200.0,
    smoothness_weight: float = 0.0,
    vega_col:          str   = None,
    bid_col:           str   = None,
    ask_col:           str   = None,
    n_jobs:            int   = 1,
    speed_mode:        str   = "auto",
    verbose:           bool  = True,
) -> dict:
    """
    Global calibration for eSSVI, optionally with time-smoothness on the smile
    shape parameters.

    Only the rho(theta) shape parameters are shared across all slices:
        rho_0   : short-maturity correlation limit   (global)
        rho_inf : long-maturity correlation limit    (global)
        lam     : exponential decay speed            (global)

    The smile shape parameters are fitted per slice:
        theta_i : ATM total variance   (per-slice)
        eta_i   : vol-of-vol scaling   (per-slice)
        gamma_i : skew decay exponent  (per-slice)

    Total params: 3 + 3*n

    Time-smoothness (smoothness_weight > 0)
    ---------------------------------------
    eSSVI gives rho a smooth time dependence through rho(theta(t)), but
    eta_i and gamma_i are free per-slice — which can produce visibly
    jagged surfaces where the smile shape changes wildly between adjacent
    expiries.  Setting smoothness_weight > 0 adds a fractional-change
    penalty on (eta_i, gamma_i):

        pen = smoothness_weight * Σ_{i≥1} [
                (Δη_i / η_{i-1})² + (Δγ_i / γ_{i-1})²
              ] / Δt_i

    where Δt_i = t_i − t_{i-1}.  The denominator turns the penalty into a
    "rate of change" — adjacent expiries are smoothed more strongly than
    distant ones for the same absolute parameter change.  Fractional
    differences keep the scale comparable across η (typically 1–3) and
    γ (typically 0.1–0.5).

    Choosing smoothness_weight
        0.0   → pure eSSVI (default, unchanged behaviour)
        0.01  → light smoothing for stability without distorting tight fits
        0.05  → moderate smoothing, useful for sparse / noisy snapshots
        0.5   → strong smoothing, smile shape forced near-constant in t

    speed_mode
    ----------
        "auto"     → thorough if n_slices ≤ 25, fast otherwise  (default)
        "thorough" → SSVI warm-start + NM + NM polish
                     Best for n ≤ 25.
        "fast"     → SSVI warm-start + L-BFGS-B polish
                     Required for large surfaces (n > 25, e.g. multi-day
                     aggregations or every-tick recalibration).

    Speed design
    ------------
    - eSSVI.min_g() is overridden to use the O(1) Gatheral-Jacquier analytical
      condition instead of the 600-point finite-difference scan.
    - Calendar constraints in the hot loop use a 100-point crossedness check
      (vs 2000 pts in the general function) — 20× cheaper.
    - "thorough" mode no longer runs a global Differential Evolution phase
      between the two NM passes. Benchmarked on real ETH and BTC snapshots:
      DE (popsize=5, maxiter=150) never beat the NM warm-start on this
      3+3n-dim objective, tying it at best and occasionally converging to a
      badly arbitrage-penalized solution instead. Removing it saves ~40s per
      cold fit with no measured loss in fit quality.
    """
    if objective is None:
        objective = get_objective("iv_wmse")
    elif isinstance(objective, str):
        objective = get_objective(objective)

    expiries = sorted(df["t"].unique())
    n        = len(expiries)

    # ── Build per-slice data ──────────────────────────────────────────────────
    slices = []
    for t in expiries:
        sub = df[df["t"] == t]
        slices.append({
            "t":    t,
            "k":    sub["k"].values,
            "w":    sub["w"].values,
            "iv":   sub["mark_iv"].values,
            "vegas": sub[vega_col].values if vega_col and vega_col in df.columns else None,
            "bid":   sub[bid_col].values  if bid_col  and bid_col  in df.columns else None,
            "ask":   sub[ask_col].values  if ask_col  and ask_col  in df.columns else None,
        })

    # Pre-bind per-slice objectives with precomputed vegas and bid/ask arrays.
    # Single-pass binding: inspects the original objective once to avoid
    # the chaining bug where wrapped signatures hide inner parameters.
    slice_objectives = [
        _bind_extra(objective, vegas=sl["vegas"], bid=sl["bid"], ask=sl["ask"])
        for sl in slices
    ]

    # ── Warm-start: per-slice SSVI fits (with DE for robustness) ─────────────
    # use_global_init=True so short maturities with extreme IV (200-400%)
    # are fitted properly — the SSVI.min_g analytical override makes DE fast.
    ssvi_model  = get_model("ssvi")
    init_thetas = np.empty(n)
    init_etas   = np.full(n, 1.5)
    init_gammas = np.full(n, 0.4)
    init_rhos   = np.full(n, -0.7)   # collected for rho-curve fitting below

    # ── Speed mode decision ───────────────────────────────────────────────────
    # In "fast" mode (auto-triggered when n>25) the polish uses L-BFGS-B
    # instead of Nelder-Mead, which is the only way to stay under a minute
    # when n is large.
    #
    # The per-slice SSVI warm-start no longer uses DE (previously toggled by
    # a `_warm_de` flag): benchmarked across every available ETH and BTC
    # snapshot, DE-based warm-start (popsize=8, maxiter=300 per slice) never
    # changed the final result after the NM phases that follow — identical
    # loss every time, at ~10x the warm-start cost. Same conclusion as the
    # global DE phase removed above. See project notes for the benchmark.
    _use_fast = (speed_mode == "fast") or (speed_mode == "auto" and n > 25)

    if verbose:
        _mode_str = "fast (L-BFGS-B polish)" if _use_fast else "thorough (NM + NM)"
        print(f"  speed_mode    : {speed_mode} → {_mode_str}")

    # ── Parallel warm-start when n_jobs != 1 ─────────────────────────────────
    _have_joblib = False
    if n_jobs != 1:
        try:
            from joblib import Parallel as _Parallel, delayed as _delayed
            _have_joblib = True
        except ImportError:
            if verbose:
                print("  joblib not found — falling back to sequential warm-start")

    _t_ws = _time.time()
    if n_jobs != 1 and _have_joblib:
        if verbose:
            print(f"  SSVI warm-start (NM only, n_jobs={n_jobs}, {n} slices) ...")
        _ws_results = _Parallel(
            n_jobs=n_jobs,
            verbose=10 if verbose else 0,    # joblib prints progress every ~10 tasks
        )(
            _delayed(_fit_ssvi_warm_start_slice)(sl, ssvi_model, objective, False)
            for sl in slices
        )
        for i, (theta, eta, gamma, rho) in enumerate(_ws_results):
            init_thetas[i] = theta
            init_etas[i]   = eta
            init_gammas[i] = gamma
            init_rhos[i]   = rho
    else:
        if verbose:
            print(f"  SSVI warm-start (NM only, sequential, {n} slices) ...")
        for i, sl in enumerate(slices):
            theta, eta, gamma, rho = _fit_ssvi_warm_start_slice(
                sl, ssvi_model, objective, False
            )
            init_thetas[i] = theta
            init_etas[i]   = eta
            init_gammas[i] = gamma
            init_rhos[i]   = rho
    if verbose:
        print(f"  SSVI warm-start done in {_time.time()-_t_ws:.1f}s")

    init_thetas = np.maximum.accumulate(init_thetas)

    # ── Rho-curve init: fit exponential to per-slice SSVI rho values ──────────
    # This gives data-informed rho_0 / rho_inf / lam rather than blind defaults,
    # placing the global optimisation start much closer to the true optimum.
    from scipy.optimize import curve_fit as _curve_fit

    def _rho_curve(theta, rho_0, rho_inf, lam):
        return rho_inf + (rho_0 - rho_inf) * np.exp(-lam * theta)

    init_rho_0, init_rho_inf, init_lam = init_rhos[0], init_rhos[-1], 2.0
    try:
        (init_rho_0, init_rho_inf, init_lam), _ = _curve_fit(
            _rho_curve, init_thetas, init_rhos,
            p0=[init_rhos[0], init_rhos[-1], 2.0],
            bounds=([-0.9999, -0.9999, 1e-2], [0.9999, 0.9999, 50.0]),
            maxfev=5000,
        )
    except Exception:
        pass   # keep the defaults if the curve fit diverges

    # ── Parametrisation ───────────────────────────────────────────────────────
    # Layout: [arctanh(rho_0), arctanh(rho_inf), log(lam),
    #          log(eta_0..n-1), logit_gamma_0..n-1, log(theta_0..n-1)]

    def pack_global(rho_0, rho_inf, lam, etas, gammas, thetas):
        gammas_c = np.clip(gammas, 1e-6, 0.5 - 1e-6)
        return np.concatenate([
            [np.arctanh(np.clip(rho_0,  -0.9999, 0.9999))],
            [np.arctanh(np.clip(rho_inf,-0.9999, 0.9999))],
            [np.log(max(lam, 1e-9))],
            np.log(np.maximum(etas,   1e-9)),
            np.log(gammas_c / (0.5 - gammas_c)),
            np.log(np.maximum(thetas, 1e-9)),
        ])

    def unpack_global(x):
        rho_0   = float(np.tanh(x[0]))
        rho_inf = float(np.tanh(x[1]))
        lam     = float(np.exp(x[2]))
        etas    = np.exp(x[3 : 3 + n])
        gr      = np.exp(x[3 + n : 3 + 2*n])
        gammas  = 0.5 * gr / (1 + gr)
        thetas  = np.exp(x[3 + 2*n :])
        return rho_0, rho_inf, lam, etas, gammas, thetas

    # ── Precompute Δt for smoothness penalty ──────────────────────────────────
    ts_arr   = np.array([sl["t"] for sl in slices], dtype=float)
    dts_safe = np.maximum(np.diff(ts_arr), 1e-3) if n > 1 else np.array([])

    # ── Objective ─────────────────────────────────────────────────────────────
    # Calendar arbitrage strategy: for SSVI-type smiles, w(k=0) = theta exactly,
    # so theta monotonicity catches ATM calendar arb with zero overhead.
    # A full 2000-pt crossedness scan is run once in verbose reporting after
    # the fit converges — it is never inside the hot loop.
    def objective_fn(x):
        rho_0, rho_inf, lam, etas, gammas, thetas = unpack_global(x)
        total = 0.0

        for i, sl in enumerate(slices):
            params = {
                "theta":   float(thetas[i]),
                "eta":     float(etas[i]),
                "gamma":   float(gammas[i]),
                "rho_0":   rho_0,
                "rho_inf": rho_inf,
                "lam":     lam,
            }
            w_fit  = model.w(sl["k"], params)
            iv_fit = model.iv(sl["k"], params, sl["t"])

            total += slice_objectives[i](w_fit, sl["w"], iv_fit, sl["iv"],
                                        sl["k"], sl["t"], None)

            # Butterfly: fast analytical check via eSSVI.min_g override
            mg = model.min_g(params)
            total += max(0.0, -mg) * penalty_but

            # Calendar: theta non-decreasing is the ATM condition (w(0)=theta)
            if i > 0:
                total += max(0.0, thetas[i-1] - thetas[i]) * penalty_cal * 10

        # ── Time-smoothness on (eta, gamma) ───────────────────────────────────
        # Fractional differences make the penalty scale-invariant between
        # eta (~1–3) and gamma (~0.1–0.5).  Division by Δt expresses the
        # penalty as a per-year rate of change, so adjacent expiries are
        # smoothed more than distant ones.
        if smoothness_weight > 0.0 and n > 1:
            eta_rel = np.diff(etas)   / np.maximum(etas[:-1],   1e-2)
            gam_rel = np.diff(gammas) / np.maximum(gammas[:-1], 5e-3)
            total  += smoothness_weight * float(
                np.sum((eta_rel ** 2 + gam_rel ** 2) / dts_safe)
            )

        return total

    # ── Pack warm-start ───────────────────────────────────────────────────────
    x0 = pack_global(
        rho_0=init_rho_0, rho_inf=init_rho_inf, lam=init_lam,
        etas=init_etas, gammas=init_gammas, thetas=init_thetas,
    )

    bounds = (
        [(-3.5, 3.5),                          # arctanh rho_0
         (-3.5, 3.5),                          # arctanh rho_inf
         (np.log(1e-2), np.log(50.0))]         # log lam
        + [(np.log(1e-3), np.log(10.0))] * n   # log eta per slice
        + [(-5.0, 5.0)] * n                    # gamma transform per slice
        + [(np.log(1e-5), np.log(10.0))] * n   # log theta per slice
    )

    if verbose:
        print(f"\nModel     : {model.name} (global rho, per-slice eta/gamma/theta)")
        print(f"Objective : {objective.__name__}")
        print(f"Slices    : {n}  |  Params: {len(x0)} total  "
              f"(3 global + 3×{n} per-slice)")
        print(f"  rho warm-start: rho_0={init_rho_0:.3f}  "
              f"rho_inf={init_rho_inf:.3f}  lam={init_lam:.3f}")
        if smoothness_weight > 0.0:
            print(f"  time-smoothness on (eta, gamma): "
                  f"smoothness_weight={smoothness_weight}\n")
        else:
            print(f"  time-smoothness: off (smoothness_weight=0)\n")

    # ── Optimisation phases ───────────────────────────────────────────────────
    if _use_fast:
        # FAST: skip DE, polish with L-BFGS-B (O(dim) per step instead of O(dim²))
        if verbose:
            print(f"  Phase: L-BFGS-B polish from warm-start ({len(x0)} params) ...")
            _tp = _time.time()
        res = minimize(
            objective_fn, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-7},
        )
        if verbose:
            print(f"    L-BFGS-B: {_time.time()-_tp:.1f}s  "
                  f"loss={res.fun:.4e}  nit={res.nit}")
    else:
        # THOROUGH: NM warm-start → NM polish (3+3n bounded grid)
        #
        # The DE phase that used to sit between these two NM passes was
        # removed after benchmarking: on real ETH and BTC snapshots, DE
        # (popsize=5, maxiter=150) never beat the NM warm-start on this
        # objective — it tied it at best, and on one BTC snapshot converged
        # to a badly arbitrage-penalized solution ~3500x worse than NM.
        # Seeding DE's population around the NM warm-start's own optimum
        # also never improved on it. See project notes for the benchmark.
        if verbose:
            print(f"  Phase 1 NM warm-start ({len(x0)} params) ...")
            _tp = _time.time()
        res_warm = minimize(
            objective_fn, x0, method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6},
        )
        if verbose:
            print(f"    Phase 1 NM:  {_time.time()-_tp:.1f}s  "
                  f"loss={res_warm.fun:.4e}")
            print(f"  Phase 2 NM polish ...")
            _tp = _time.time()
        res = minimize(
            objective_fn, res_warm.x, method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-9, "fatol": 1e-9},
        )
        if verbose:
            print(f"    Phase 2 NM:  {_time.time()-_tp:.1f}s  "
                  f"loss={res.fun:.4e}")

    rho_0, rho_inf, lam, etas, gammas, thetas = unpack_global(res.x)

    if verbose:
        print(f"  rho_0={rho_0:.3f}  rho_inf={rho_inf:.3f}  lam={lam:.3f}\n")

    # ── Build result dict ─────────────────────────────────────────────────────
    params_out  = []
    metrics_out = {m: [] for m in DEFAULT_METRICS}

    for i, sl in enumerate(slices):

        p = {
            "theta":   float(thetas[i]),
            "eta":     float(etas[i]),
            "gamma":   float(gammas[i]),
            "rho_0":   rho_0,
            "rho_inf": rho_inf,
            "lam":     lam,
        }
        params_out.append(p)

        w_fit  = model.w(sl["k"], p)
        iv_fit = model.iv(sl["k"], p, sl["t"])
        evals  = evaluate_all(w_fit, sl["w"], iv_fit, sl["iv"], sl["k"], sl["t"],
                              bid=sl.get("bid"), ask=sl.get("ask"))
        for m in DEFAULT_METRICS:
            metrics_out[m].append(evals[m])

        if verbose:
            cal_ok = "ok" if i == 0 else (
                "⚠ cal" if crossedness(model, params_out[i-1], p) > 1e-6 else "ok"
            )
            print(f"  T={sl['t']:.4f}  theta={thetas[i]:.5f}  "
                  f"rho={model._rho(thetas[i], p):.3f}  "
                  f"vwrmse={evals['vwrmse']:.4f}  [{cal_ok}]")

    return {
        "model_name": model.name,
        "objective":  objective.__name__,
        "expiries":   [sl["t"] for sl in slices],
        "params":     params_out,
        "metrics":    metrics_out,
        "n_points":   [len(df[df["t"] == sl["t"]]) for sl in slices],
        "_model":     model,
    }

def calibrate_global_ssvi(
    df:           pd.DataFrame,
    model         = None,
    objective     = None,
    penalty_cal:  float = 500.0,
    penalty_but:  float = 200.0,
    vega_col:     str   = None,
    bid_col:      str   = None,
    ask_col:      str   = None,
    verbose:      bool  = True,
) -> dict:
    """
    Global calibration for SSVI — shared (eta, gamma, rho) across all maturities,
    only theta fitted per slice.

    This is the Gatheral-Jacquier (2014) surface SSVI:

        phi(theta)  = eta / theta^gamma          (global eta, gamma)
        rho         = constant                   (global rho)
        theta_i     = ATM total variance          (per-slice)

    Total params: 3 + n

    Under eta*(1+|rho|) <= 2, gamma in (0, 0.5], and theta non-decreasing,
    the surface is provably free of static arbitrage (GJ 2014 Theorem 4.4).
    No crossedness check is needed in the hot loop — theta monotonicity is
    sufficient for the global surface.

    This is the baseline that eSSVI generalises: every global SSVI solution
    is a special case of eSSVI with rho_0 = rho_inf = rho (any lam).

    Speed design
    ------------
    - 3 + n params vs 4n for per-slice SSVI → much smaller search space.
    - Warm-start pools per-slice SSVI fits and takes the median eta/gamma/rho.
    - DE budget can be larger than eSSVI (13 params vs 33 for n=10).
    """
    from volatility_surface.models.vol_models import get_model as _get_model

    if model is None:
        model = _get_model("ssvi")
    if objective is None:
        objective = get_objective("iv_wmse")
    elif isinstance(objective, str):
        objective = get_objective(objective)

    expiries = sorted(df["t"].unique())
    n        = len(expiries)

    # ── Build per-slice data ──────────────────────────────────────────────────
    slices = []
    for t in expiries:
        sub = df[df["t"] == t]
        slices.append({
            "t":    t,
            "k":    sub["k"].values,
            "w":    sub["w"].values,
            "iv":   sub["mark_iv"].values,
            "vegas": sub[vega_col].values if vega_col and vega_col in df.columns else None,
            "bid":   sub[bid_col].values  if bid_col  and bid_col  in df.columns else None,
            "ask":   sub[ask_col].values  if ask_col  and ask_col  in df.columns else None,
        })

    # Pre-bind per-slice objectives with precomputed vegas and bid/ask arrays.
    slice_objectives = [
        _bind_extra(objective, vegas=sl["vegas"], bid=sl["bid"], ask=sl["ask"])
        for sl in slices
    ]

    # ── Warm-start: pool per-slice SSVI fits ──────────────────────────────────
    ssvi_model  = model
    init_thetas = np.empty(n)
    init_etas   = np.full(n, 1.5)
    init_gammas = np.full(n, 0.4)
    init_rhos   = np.full(n, -0.7)

    for i, sl in enumerate(slices):
        try:
            p = fit_single_slice(
                sl["k"], sl["w"], sl["iv"], sl["t"],
                model=ssvi_model,
                objective=objective,
                use_global_init=True,
            )
            init_thetas[i] = p["theta"]
            init_etas[i]   = p["eta"]
            init_gammas[i] = p["gamma"]
            init_rhos[i]   = p["rho"]
        except Exception:
            init_thetas[i] = float(np.interp(
                0.0, np.sort(sl["k"]), sl["w"][np.argsort(sl["k"])]
            ))

    init_thetas = np.maximum.accumulate(init_thetas)
    # Median is more robust than mean against outlier slices
    init_eta   = float(np.median(init_etas))
    init_gamma = float(np.median(init_gammas))
    init_rho   = float(np.median(init_rhos))

    # ── Parametrisation ───────────────────────────────────────────────────────
    # Layout: [arctanh(rho), log(eta), logit_gamma, log(theta_0), ..., log(theta_n-1)]

    def pack_global(rho, eta, gamma, thetas):
        gamma_c = np.clip(gamma, 1e-6, 0.5 - 1e-6)
        return np.concatenate([
            [np.arctanh(np.clip(rho, -0.9999, 0.9999))],
            [np.log(max(eta, 1e-9))],
            [np.log(gamma_c / (0.5 - gamma_c))],
            np.log(np.maximum(thetas, 1e-9)),
        ])

    def unpack_global(x):
        rho    = float(np.tanh(x[0]))
        eta    = float(np.exp(x[1]))
        gr     = np.exp(x[2])
        gamma  = float(0.5 * gr / (1 + gr))
        thetas = np.exp(x[3:])
        return rho, eta, gamma, thetas

    # ── Objective ─────────────────────────────────────────────────────────────
    # For global SSVI, theta monotonicity alone is sufficient to guarantee
    # surface-wide calendar-spread freedom (GJ 2014).  No crossedness call needed.
    def objective_fn(x):
        rho, eta, gamma, thetas = unpack_global(x)
        total = 0.0

        for i, sl in enumerate(slices):
            params = {
                "theta": float(thetas[i]),
                "eta":   eta,
                "gamma": gamma,
                "rho":   rho,
            }
            w_fit  = model.w(sl["k"], params)
            iv_fit = model.iv(sl["k"], params, sl["t"])

            total += slice_objectives[i](w_fit, sl["w"], iv_fit, sl["iv"], sl["k"], sl["t"], None)

            # Butterfly: analytical per-slice check
            mg = model.min_g(params)
            total += max(0.0, -mg) * penalty_but

            # Calendar: theta non-decreasing (necessary and sufficient for global SSVI)
            if i > 0:
                total += max(0.0, thetas[i-1] - thetas[i]) * penalty_cal * 10

        return total

    # ── Pack warm-start ───────────────────────────────────────────────────────
    x0 = pack_global(init_rho, init_eta, init_gamma, init_thetas)

    bounds = (
        [(-3.5, 3.5)]                           # arctanh rho
        + [(np.log(1e-3), np.log(10.0))]        # log eta
        + [(-5.0, 5.0)]                          # gamma transform
        + [(np.log(1e-5), np.log(10.0))] * n    # log theta per slice
    )

    if verbose:
        print(f"\nModel     : SSVI (global eta/gamma/rho, per-slice theta)")
        print(f"Objective : {objective.__name__}")
        print(f"Slices    : {n}  |  Params: {len(x0)} total  "
              f"(3 global + {n} per-slice theta)")
        print(f"  warm-start: eta={init_eta:.3f}  gamma={init_gamma:.3f}  "
              f"rho={init_rho:.3f}\n")

    # ── Phase 1: quick local NM from the data-informed warm-start ─────────────
    res_warm = minimize(
        objective_fn, x0, method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6},
    )

    # ── Phase 2: global DE (larger budget than eSSVI — only 3+n params) ───────
    de_res = differential_evolution(
        objective_fn, bounds,
        seed=42, maxiter=300, tol=1e-6,
        popsize=8, mutation=(0.5, 1.5), recombination=0.9,
        workers=1,
    )

    # ── Phase 3: final polish from best candidate ─────────────────────────────
    x_best = res_warm.x if res_warm.fun <= de_res.fun else de_res.x
    res = minimize(
        objective_fn, x_best, method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-9, "fatol": 1e-9},
    )

    rho, eta, gamma, thetas = unpack_global(res.x)

    if verbose:
        print(f"  eta={eta:.4f}  gamma={gamma:.4f}  rho={rho:.4f}\n")

    # ── Build result dict ─────────────────────────────────────────────────────
    params_out  = []
    metrics_out = {m: [] for m in DEFAULT_METRICS}

    for i, sl in enumerate(slices):
        p = {
            "theta": float(thetas[i]),
            "eta":   eta,
            "gamma": gamma,
            "rho":   rho,
        }
        params_out.append(p)

        w_fit  = model.w(sl["k"], p)
        iv_fit = model.iv(sl["k"], p, sl["t"])
        evals  = evaluate_all(w_fit, sl["w"], iv_fit, sl["iv"], sl["k"], sl["t"],
                              bid=sl.get("bid"), ask=sl.get("ask"))
        for m in DEFAULT_METRICS:
            metrics_out[m].append(evals[m])

        if verbose:
            cal_ok = "ok" if i == 0 else (
                "⚠ cal" if crossedness(model, params_out[i-1], p) > 1e-6 else "ok"
            )
            arb_ok = "ok" if model.min_g(p) >= 0 else "⚠ butterfly"
            print(f"  T={sl['t']:.4f}  theta={thetas[i]:.5f}  "
                  f"vwrmse={evals['vwrmse']:.4f}  [{arb_ok}]  [{cal_ok}]")

    return {
        "model_name": "SSVI (global)",
        "objective":  objective.__name__,
        "expiries":   [sl["t"] for sl in slices],
        "params":     params_out,
        "metrics":    metrics_out,
        "n_points":   [len(df[df["t"] == sl["t"]]) for sl in slices],
        "_model":     model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SABR  (per-slice params, joint calendar-arb enforcement)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_global_sabr(
    df:           pd.DataFrame,
    model         = None,
    objective     = None,
    penalty_cal:  float = 5000.0,
    penalty_but:  float = 200.0,
    n_cal_grid:   int   = 200,
    cal_k_range:  tuple = (-3.0, 3.0),
    vega_col:     str   = None,
    bid_col:      str   = None,
    ask_col:      str   = None,
    warm_start:   dict  = None,
    max_escalations: int = 3,
    verbose:      bool  = True,
) -> dict:
    """
    Joint global SABR calibration with explicit calendar-spread arbitrage
    enforcement.

    SABR has no natural surface-level parametrisation (unlike SSVI's theta
    monotonicity or eSSVI's rho(theta)), so every (alpha_i, rho_i, nu_i) is
    fitted per slice — 3*n parameters total — but jointly, so the optimiser
    sees calendar interactions between adjacent expiries.

    The calendar penalty enforces

        w(k, t_{i+1})  ≥  w(k, t_i)        for every k in cal_k_range
                                            for every i = 0, ..., n-2

    on a dense k-grid.  The Hagan SABR approximation is essentially always
    butterfly-free on the relevant (k, t) range, but a small butterfly
    penalty is kept as a safety net.

    Escalation
    ----------
    If the first joint NM polish still leaves calendar crossedness > 1e-6
    on any adjacent pair, the calendar penalty is multiplied by 10 and the
    fit is repeated, up to ``max_escalations`` times (default 3 → up to
    1000× the base penalty).  This recovers arb-freedom without sacrificing
    fit on pairs that were already fine.

    Parameters
    ----------
    df            : snapshot dataframe (output of load_snapshot)
    model         : SABR instance  (default: get_model("sabr"))
    objective     : callable or string  (default: "iv_wmse")
    penalty_cal   : base weight on integrated calendar crossedness
    penalty_but   : per-slice butterfly penalty (safety net)
    n_cal_grid    : number of k-points used in the calendar penalty
    cal_k_range   : (k_min, k_max) range for the calendar penalty grid
    warm_start    : optional previous result dict; if given, its params are
                    used as the per-slice starting point (nearest-T match),
                    skipping the per-slice warm-start phase.
    max_escalations : maximum number of penalty bump rounds.

    Returns
    -------
    Same result-dict format as calibrate_snapshot.
    """
    from volatility_surface.models.vol_models import get_model as _get_model

    if model is None:
        model = _get_model("sabr")
    if model.name != "SABR":
        raise ValueError(f"calibrate_global_sabr expects a SABR model, got {model.name!r}")
    if objective is None:
        objective = get_objective("iv_wmse")
    elif isinstance(objective, str):
        objective = get_objective(objective)

    expiries = sorted(df["t"].unique())
    n        = len(expiries)
    if n < 2:
        raise ValueError("calibrate_global_sabr needs at least 2 expiries.")

    # ── Build per-slice data ──────────────────────────────────────────────────
    slices = []
    for t in expiries:
        sub = df[df["t"] == t]
        slices.append({
            "t":    t,
            "k":    sub["k"].values,
            "w":    sub["w"].values,
            "iv":   sub["mark_iv"].values,
            "vegas": sub[vega_col].values if vega_col and vega_col in df.columns else None,
            "bid":   sub[bid_col].values  if bid_col  and bid_col  in df.columns else None,
            "ask":   sub[ask_col].values  if ask_col  and ask_col  in df.columns else None,
        })

    slice_objectives = [
        _bind_extra(objective, vegas=sl["vegas"], bid=sl["bid"], ask=sl["ask"])
        for sl in slices
    ]

    # ── Warm-start ────────────────────────────────────────────────────────────
    init_alphas = np.empty(n)
    init_rhos   = np.empty(n)
    init_nus    = np.empty(n)

    if warm_start is not None and "params" in warm_start:
        ws_T = np.asarray(warm_start["expiries"], dtype=float)
        ws_p = warm_start["params"]
        for i, t in enumerate(expiries):
            p = ws_p[int(np.argmin(np.abs(ws_T - t)))]
            init_alphas[i] = float(p.get("alpha", 0.5))
            init_rhos[i]   = float(p.get("rho",   -0.5))
            init_nus[i]    = float(p.get("nu",    0.5))
        if verbose:
            print(f"  warm-start from provided result ({len(ws_p)} slices)")
    else:
        if verbose:
            print(f"  SABR per-slice warm-start (DE+NM, {n} slices) ...")
            _t_ws = _time.time()
        for i, sl in enumerate(slices):
            try:
                p = fit_single_slice(
                    sl["k"], sl["w"], sl["iv"], sl["t"],
                    model=model,
                    objective=slice_objectives[i],
                    use_global_init=True,
                )
                init_alphas[i] = p["alpha"]
                init_rhos[i]   = p["rho"]
                init_nus[i]    = p["nu"]
            except Exception:
                atm_w  = float(np.interp(0.0, np.sort(sl["k"]),
                                          sl["w"][np.argsort(sl["k"])]))
                init_alphas[i] = float(np.sqrt(max(atm_w / sl["t"], 1e-6)))
                init_rhos[i]   = -0.5
                init_nus[i]    = 0.5
        if verbose:
            print(f"  warm-start done in {_time.time()-_t_ws:.1f}s")

    # ── Dense k-grid for the calendar penalty ─────────────────────────────────
    k_cal = np.linspace(cal_k_range[0], cal_k_range[1], n_cal_grid)
    dk    = (cal_k_range[1] - cal_k_range[0]) / (n_cal_grid - 1)

    # ── Pack / unpack ─────────────────────────────────────────────────────────
    # Layout: [log(alpha)_i, arctanh(rho)_i, log(nu)_i]  for i = 0..n-1
    def pack_global(alphas, rhos, nus):
        return np.concatenate([
            np.log(np.maximum(alphas, 1e-9)),
            np.arctanh(np.clip(rhos, -0.9999, 0.9999)),
            np.log(np.maximum(nus, 1e-9)),
        ])

    def unpack_global(x):
        alphas = np.exp(x[0     : n])
        rhos   = np.tanh(x[n    : 2*n])
        nus    = np.exp(x[2*n   : 3*n])
        return alphas, rhos, nus

    # ── Objective factory (lets us re-bind the penalty on escalation) ─────────
    def make_objective_fn(pen_cal):
        def _obj(x):
            alphas, rhos, nus = unpack_global(x)
            total = 0.0
            w_cal = np.empty((n, n_cal_grid))

            for i, sl in enumerate(slices):
                params = {
                    "alpha": float(alphas[i]),
                    "rho":   float(rhos[i]),
                    "nu":    float(nus[i]),
                    "t":     sl["t"],
                }
                w_fit  = model.w(sl["k"], params)
                iv_fit = model.iv(sl["k"], params, sl["t"])
                total += slice_objectives[i](
                    w_fit, sl["w"], iv_fit, sl["iv"],
                    sl["k"], sl["t"], None,
                )

                mg = model.min_g(params)
                total += max(0.0, -mg) * penalty_but

                w_cal[i] = model.w(k_cal, params)

            # Calendar: integrated crossedness over k for each adjacent pair.
            # diff = w(k, t_i) - w(k, t_{i+1}) > 0  ⇒  violation.
            # Use trapezoidal integration of the violation to keep the penalty
            # smooth in k (no kink-amplification from a max-only metric).
            for i in range(n - 1):
                viol = np.maximum(w_cal[i] - w_cal[i+1], 0.0)
                # area under the violation curve (trapezoidal) — penalises
                # wide-spread violations
                area = 0.5 * dk * (viol[0] + viol[-1] + 2.0 * float(viol[1:-1].sum()))
                # plus a max term — penalises tall localised violations
                area += float(viol.max())
                total += area * pen_cal

            return total
        return _obj

    bounds = (
        [(np.log(1e-4), np.log(2.0))]  * n +    # log alpha
        [(-3.5, 3.5)]                  * n +    # arctanh rho
        [(np.log(1e-4), np.log(10.0))] * n      # log nu
    )

    if verbose:
        print(f"\nModel     : SABR (global, per-slice with calendar enforcement)")
        print(f"Objective : {objective.__name__}")
        print(f"Slices    : {n}  |  Params: {3*n} total  (3 per slice)")
        print(f"  calendar grid: {n_cal_grid} pts on k ∈ {cal_k_range}")
        print(f"  base penalty_cal={penalty_cal}, max_escalations={max_escalations}\n")

    # ── Iterative-penalty optimisation ────────────────────────────────────────
    x_cur     = pack_global(init_alphas, init_rhos, init_nus)
    pen_cur   = float(penalty_cal)
    last_res  = None

    for round_i in range(max_escalations + 1):
        obj_fn = make_objective_fn(pen_cur)
        if verbose:
            print(f"  Round {round_i+1}/{max_escalations+1}: NM polish "
                  f"(penalty_cal={pen_cur:.1f}) ...")
            _tp = _time.time()

        res = minimize(
            obj_fn, x_cur, method="Nelder-Mead",
            options={
                "maxiter": max(3000, 800 * n),
                "xatol":   1e-8,
                "fatol":   1e-8,
                "adaptive": True,
            },
        )
        x_cur    = res.x
        last_res = res

        # Check residual calendar violation on the dense grid
        alphas, rhos, nus = unpack_global(x_cur)
        max_viol = 0.0
        for i in range(n - 1):
            p_i = {"alpha": float(alphas[i]),   "rho": float(rhos[i]),
                   "nu":    float(nus[i]),      "t":   slices[i]["t"]}
            p_j = {"alpha": float(alphas[i+1]), "rho": float(rhos[i+1]),
                   "nu":    float(nus[i+1]),    "t":   slices[i+1]["t"]}
            w_i = model.w(k_cal, p_i)
            w_j = model.w(k_cal, p_j)
            max_viol = max(max_viol, float(np.max(w_i - w_j)))

        if verbose:
            print(f"    NM: {_time.time()-_tp:.1f}s  loss={res.fun:.4e}  "
                  f"worst calendar crossedness = {max_viol:+.3e}")

        if max_viol <= 1e-6 or round_i == max_escalations:
            break
        pen_cur *= 10.0
        if verbose:
            print(f"    violation > 1e-6 → escalating penalty to {pen_cur:.1f}")

    alphas, rhos, nus = unpack_global(x_cur)

    # ── Build result dict ─────────────────────────────────────────────────────
    params_out  = []
    metrics_out = {m: [] for m in DEFAULT_METRICS}

    for i, sl in enumerate(slices):
        p = {
            "alpha": float(alphas[i]),
            "rho":   float(rhos[i]),
            "nu":    float(nus[i]),
            "t":     sl["t"],
        }
        params_out.append(p)
        w_fit  = model.w(sl["k"], p)
        iv_fit = model.iv(sl["k"], p, sl["t"])
        evals  = evaluate_all(w_fit, sl["w"], iv_fit, sl["iv"], sl["k"], sl["t"],
                              bid=sl.get("bid"), ask=sl.get("ask"))
        for m in DEFAULT_METRICS:
            metrics_out[m].append(evals[m])

        if verbose:
            cross = 0.0 if i == 0 else crossedness(
                model, params_out[i-1], p, k_range=cal_k_range, n=n_cal_grid,
            )
            cal_tag = "ok" if cross <= 1e-6 else f"⚠ cal {cross:+.2e}"
            print(f"  T={sl['t']:.4f}  alpha={p['alpha']:.4f}  "
                  f"rho={p['rho']:+.3f}  nu={p['nu']:.3f}  "
                  f"vwrmse={evals['vwrmse']:.4f}  [{cal_tag}]")

    return {
        "model_name": "SABR (global arb-free)",
        "objective":  objective.__name__,
        "expiries":   [sl["t"] for sl in slices],
        "params":     params_out,
        "metrics":    metrics_out,
        "n_points":   [len(df[df["t"] == sl["t"]]) for sl in slices],
        "_model":     model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CALENDAR-ARB-FREE POST-PROCESSOR  (model-agnostic dispatcher)
# ─────────────────────────────────────────────────────────────────────────────

def enforce_calendar_arbfree(
    result:       dict,
    df:           pd.DataFrame,
    objective     = None,
    tol:          float = 1e-6,
    k_range:      tuple = (-3.0, 3.0),
    n_grid:       int   = 200,
    verbose:      bool  = True,
    **kwargs,
) -> dict:
    """
    Project a calibrated surface onto the calendar-arbitrage-free manifold.

    Scans every adjacent expiry pair on a dense k-grid; if any pair has
        max_k [ w(k, t_i)  -  w(k, t_{i+1}) ]  >  tol
    the entire surface is refitted jointly with a strong calendar penalty,
    using the current params as warm-start.

    Currently supported models
    --------------------------
        SABR    →  refit via calibrate_global_sabr
        SSVI / eSSVI  →  already calendar-arb-free by construction (theta
                         monotonicity) when fitted via calibrate_global_ssvi
                         / calibrate_global_essvi, so the function is a no-op
                         and just reports.
        other   →  raises NotImplementedError.

    Parameters
    ----------
    result      : calibration result dict (must contain '_model')
    df          : the snapshot dataframe used to produce `result`
    objective   : objective for the refit  (default: same as in `result`)
    tol         : crossedness tolerance below which a pair is treated as free
    k_range     : k-range scanned for violations and used in the joint refit
    n_grid      : grid resolution for the scan and refit penalty
    verbose     : print diagnostics
    **kwargs    : extra args forwarded to the model-specific joint refit
                  (e.g. penalty_cal, max_escalations for SABR)

    Returns
    -------
    Either the original `result` (if no violation) or a new joint-refit result.
    """
    model = result.get("_model")
    if model is None:
        raise ValueError("result must contain '_model' (calibrator output).")

    params   = result["params"]
    expiries = result["expiries"]

    k_grid = np.linspace(k_range[0], k_range[1], n_grid)
    violations = []
    worst = 0.0
    for i in range(1, len(params)):
        w_prev = model.w(k_grid, params[i-1])
        w_cur  = model.w(k_grid, params[i])
        cross  = float(np.max(w_prev - w_cur))
        if cross > tol:
            violations.append((i, cross))
        worst = max(worst, cross)

    if verbose:
        print(f"\n  Calendar-arb scan over k ∈ {k_range} ({n_grid} pts), "
              f"tol={tol:.1e}")
        print(f"    worst crossedness across {len(params)-1} pairs: "
              f"{worst:+.3e}")

    if not violations:
        if verbose:
            print(f"    ✓ surface is calendar-arb-free — returning unchanged.\n")
        return result

    if verbose:
        print(f"    ✗ {len(violations)} pair(s) violate the constraint:")
        for i, c in violations:
            print(f"        pair ({expiries[i-1]:.4f} → {expiries[i]:.4f}): "
                  f"crossedness = {c:+.3e}")
        print(f"  → joint refit with calendar enforcement\n")

    obj = objective if objective is not None else result.get("objective")

    name = (model.name or "").lower()
    if name == "sabr":
        return calibrate_global_sabr(
            df, model=model, objective=obj,
            warm_start=result,
            n_cal_grid=n_grid, cal_k_range=k_range,
            verbose=verbose, **kwargs,
        )

    if name in ("ssvi", "essvi"):
        raise NotImplementedError(
            f"{model.name} surfaces are made calendar-arb-free by construction "
            f"when calibrated via calibrate_global_ssvi / calibrate_global_essvi "
            f"(theta monotonicity). If you see violations from a per-slice fit, "
            f"refit the surface globally instead of post-processing."
        )

    raise NotImplementedError(
        f"enforce_calendar_arbfree does not yet support model {model.name!r}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# FAST UPDATE FOR MARKET MAKING
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_global_essvi_update(
    df:           pd.DataFrame,
    prev_result:  dict,
    model         = None,
    objective     = None,
    mode:         str   = "full_polish",
    vega_col:     str   = None,
    bid_col:      str   = None,
    ask_col:      str   = None,
    penalty_cal:  float = 500.0,
    penalty_but:  float = 200.0,
    verbose:      bool  = False,
) -> dict:
    """
    Fast recalibration of eSSVI from a previous result.

    Designed for tick-by-tick market-making updates where the previous fit is a
    good warm start and a full DE pass would be too slow.

    Two modes
    ---------
    "theta_only"  (~50–200 ms)
        Holds every shape parameter (rho_0, rho_inf, lam, eta_i, gamma_i)
        fixed at their previous values and refits only the n ATM variance
        levels θᵢ using independent 1-D Brent searches.  Use when smile
        shape is stable and only the ATM level has moved.

    "full_polish"  (~1–3 s)
        Packs the previous result as the starting point for a short
        Nelder-Mead run (maxiter=500).  Skips the SSVI warm-start loop and
        the DE phase entirely.  Use after a modest market move where shape
        may have shifted slightly.

    Parameters
    ----------
    df           : current snapshot (same format as calibrate_global_essvi)
    prev_result  : dict from calibrate_global_essvi or a prior update call
    model        : eSSVI instance  (default: taken from prev_result["_model"])
    objective    : callable or str  (default: iv_wmse)
    mode         : "theta_only" | "full_polish"
    vega_col     : column for precomputed vegas  (optional)
    bid_col      : column for bid IV  (optional)
    ask_col      : column for ask IV  (optional)
    penalty_cal  : calendar-spread penalty weight
    penalty_but  : butterfly-arbitrage penalty weight
    verbose      : print timing and per-slice metrics

    Returns
    -------
    Same result-dict format as calibrate_global_essvi.
    """
    t0 = _time.time()

    # ── Defaults ──────────────────────────────────────────────────────────────
    if model is None:
        model = prev_result.get("_model")
        if model is None:
            raise ValueError(
                "model not found in prev_result; pass the model explicitly."
            )
    if objective is None:
        objective = get_objective("iv_wmse")
    elif isinstance(objective, str):
        objective = get_objective(objective)

    expiries = sorted(df["t"].unique())
    n        = len(expiries)

    # ── Build per-slice data ──────────────────────────────────────────────────
    slices = []
    for t in expiries:
        sub = df[df["t"] == t]
        slices.append({
            "t":    t,
            "k":    sub["k"].values,
            "w":    sub["w"].values,
            "iv":   sub["mark_iv"].values,
            "vegas": sub[vega_col].values if vega_col and vega_col in df.columns else None,
            "bid":   sub[bid_col].values  if bid_col  and bid_col  in df.columns else None,
            "ask":   sub[ask_col].values  if ask_col  and ask_col  in df.columns else None,
        })

    slice_objectives = [
        _bind_extra(objective, vegas=sl["vegas"], bid=sl["bid"], ask=sl["ask"])
        for sl in slices
    ]

    # ── Nearest-T matching: map each current expiry to prev params ────────────
    old_T      = np.asarray(prev_result["expiries"])
    old_params = prev_result["params"]

    def _nearest_prev(t_query):
        return old_params[int(np.argmin(np.abs(old_T - t_query)))]

    # ── Extract global rho-curve params ───────────────────────────────────────
    p_ref   = old_params[0]
    rho_0   = float(p_ref.get("rho_0",   p_ref.get("rho", -0.7)))
    rho_inf = float(p_ref.get("rho_inf", rho_0))
    lam     = float(p_ref.get("lam", 2.0))

    # =========================================================================
    # MODE: theta_only
    # =========================================================================
    if mode == "theta_only":
        if verbose:
            print(f"\n  [theta_only] Refitting {n} θᵢ  "
                  f"(all shape params fixed from previous fit)")

        prev_theta = None
        thetas_out = np.empty(n)

        for i, sl in enumerate(slices):
            pp = _nearest_prev(sl["t"])
            shape = {
                "rho_0":   rho_0,
                "rho_inf": rho_inf,
                "lam":     lam,
                "eta":     float(pp.get("eta",   1.5)),
                "gamma":   float(pp.get("gamma", 0.4)),
            }
            # Lower bound enforces calendar monotonicity
            theta_lb = max(prev_theta if prev_theta is not None else 1e-5, 1e-5)
            theta_ub = 10.0

            # Capture loop variables via default args (avoids late-binding bugs)
            def _loss_1d(log_theta,
                         _shape=shape, _sl=sl, _so=slice_objectives[i],
                         _prev=prev_theta):
                theta  = np.exp(log_theta)
                params = {**_shape, "theta": theta}
                w_fit  = model.w(_sl["k"], params)
                iv_fit = model.iv(_sl["k"], params, _sl["t"])
                err    = _so(w_fit, _sl["w"], iv_fit, _sl["iv"],
                             _sl["k"], _sl["t"], None)
                mg     = model.min_g(params)
                err   += max(0.0, -mg) * penalty_but
                if _prev is not None:
                    err += max(0.0, _prev - theta) * penalty_cal * 10
                return err

            res_1d = minimize_scalar(
                _loss_1d,
                bounds=(np.log(theta_lb), np.log(theta_ub)),
                method="bounded",
                options={"xatol": 1e-7},
            )
            theta_i = max(float(np.exp(res_1d.x)),
                          prev_theta if prev_theta is not None else 1e-5)
            thetas_out[i] = theta_i
            prev_theta    = theta_i

        # ── Build result ──────────────────────────────────────────────────────
        params_out  = []
        metrics_out = {m: [] for m in DEFAULT_METRICS}

        for i, sl in enumerate(slices):
            pp = _nearest_prev(sl["t"])
            p  = {
                "theta":   float(thetas_out[i]),
                "eta":     float(pp.get("eta",   1.5)),
                "gamma":   float(pp.get("gamma", 0.4)),
                "rho_0":   rho_0,
                "rho_inf": rho_inf,
                "lam":     lam,
            }
            params_out.append(p)
            w_fit  = model.w(sl["k"], p)
            iv_fit = model.iv(sl["k"], p, sl["t"])
            evals  = evaluate_all(w_fit, sl["w"], iv_fit, sl["iv"], sl["k"], sl["t"],
                                  bid=sl.get("bid"), ask=sl.get("ask"))
            for m in DEFAULT_METRICS:
                metrics_out[m].append(evals[m])
            if verbose:
                print(f"    T={sl['t']:.4f}  θ={thetas_out[i]:.5f}  "
                      f"iv_rmse={evals['iv_rmse']:.4f}")

        elapsed = _time.time() - t0
        if verbose:
            print(f"  [theta_only] done in {elapsed*1000:.1f} ms")

        return {
            "model_name": model.name + " (theta_only update)",
            "objective":  objective.__name__,
            "expiries":   [sl["t"] for sl in slices],
            "params":     params_out,
            "metrics":    metrics_out,
            "n_points":   [len(df[df["t"] == sl["t"]]) for sl in slices],
            "_model":     model,
        }

    # =========================================================================
    # MODE: full_polish
    # =========================================================================
    # Pack/unpack helpers (same parametrisation as calibrate_global_essvi)
    def _pack(rho_0, rho_inf, lam, etas, gammas, thetas):
        gc = np.clip(gammas, 1e-6, 0.5 - 1e-6)
        return np.concatenate([
            [np.arctanh(np.clip(rho_0,  -0.9999, 0.9999))],
            [np.arctanh(np.clip(rho_inf,-0.9999, 0.9999))],
            [np.log(max(lam, 1e-9))],
            np.log(np.maximum(etas, 1e-9)),
            np.log(gc / (0.5 - gc)),
            np.log(np.maximum(thetas, 1e-9)),
        ])

    def _unpack(x):
        r0   = float(np.tanh(x[0]))
        ri   = float(np.tanh(x[1]))
        la   = float(np.exp(x[2]))
        etas = np.exp(x[3 : 3 + n])
        gr   = np.exp(x[3 + n : 3 + 2*n])
        gams = 0.5 * gr / (1.0 + gr)
        ths  = np.exp(x[3 + 2*n :])
        return r0, ri, la, etas, gams, ths

    # Warm-start from nearest previous params per slice
    pm = [_nearest_prev(sl["t"]) for sl in slices]
    etas0   = np.array([float(p.get("eta",   1.5))  for p in pm])
    gammas0 = np.array([float(p.get("gamma", 0.4))  for p in pm])
    thetas0 = np.maximum.accumulate(
        np.maximum([float(p.get("theta", 0.05)) for p in pm], 1e-5)
    )
    x0 = _pack(rho_0, rho_inf, lam, etas0, gammas0, thetas0)

    def _objective_fn(x):
        r0, ri, la, etas, gams, ths = _unpack(x)
        total = 0.0
        for i, sl in enumerate(slices):
            params = {
                "theta":   float(ths[i]),
                "eta":     float(etas[i]),
                "gamma":   float(gams[i]),
                "rho_0":   r0,
                "rho_inf": ri,
                "lam":     la,
            }
            w_fit  = model.w(sl["k"], params)
            iv_fit = model.iv(sl["k"], params, sl["t"])
            total += slice_objectives[i](w_fit, sl["w"], iv_fit, sl["iv"],
                                         sl["k"], sl["t"], None)
            total += max(0.0, -model.min_g(params)) * penalty_but
            if i > 0:
                total += max(0.0, ths[i-1] - ths[i]) * penalty_cal * 10
        return total

    if verbose:
        print(f"\n  [full_polish] NM polish from previous result  "
              f"({len(x0)} params,  maxiter=500)")

    res = minimize(
        _objective_fn, x0, method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-7, "fatol": 1e-7},
    )

    r0, ri, la, etas, gams, ths = _unpack(res.x)

    # ── Build result ──────────────────────────────────────────────────────────
    params_out  = []
    metrics_out = {m: [] for m in DEFAULT_METRICS}

    for i, sl in enumerate(slices):
        p = {
            "theta":   float(ths[i]),
            "eta":     float(etas[i]),
            "gamma":   float(gams[i]),
            "rho_0":   r0,
            "rho_inf": ri,
            "lam":     la,
        }
        params_out.append(p)
        w_fit  = model.w(sl["k"], p)
        iv_fit = model.iv(sl["k"], p, sl["t"])
        evals  = evaluate_all(w_fit, sl["w"], iv_fit, sl["iv"], sl["k"], sl["t"],
                              bid=sl.get("bid"), ask=sl.get("ask"))
        for m in DEFAULT_METRICS:
            metrics_out[m].append(evals[m])
        if verbose:
            print(f"    T={sl['t']:.4f}  θ={ths[i]:.5f}  "
                  f"iv_rmse={evals['iv_rmse']:.4f}")

    elapsed = _time.time() - t0
    if verbose:
        print(f"  [full_polish] done in {elapsed*1000:.1f} ms")

    return {
        "model_name": model.name + " (polished update)",
        "objective":  objective.__name__,
        "expiries":   [sl["t"] for sl in slices],
        "params":     params_out,
        "metrics":    metrics_out,
        "n_points":   [len(df[df["t"] == sl["t"]]) for sl in slices],
        "_model":     model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SABR fast-path update (warm-start, no DE)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_snapshot_sabr_update(
    df:           pd.DataFrame,
    prev_result:  dict,
    model         = None,
    objective     = None,
    vega_col:     str   = None,
    bid_col:      str   = None,
    ask_col:      str   = None,
    penalty_cal:  float = 500.0,
    penalty_but:  float = 200.0,
    min_points_per_slice: int = 5,
    nm_maxiter:   int   = 500,
    nm_tol:       float = 1e-7,
    verbose:      bool  = False,
) -> dict:
    """
    Fast per-slice SABR recalibration from a previous result.

    Skips the differential-evolution global search entirely and runs only a
    Nelder-Mead polish from the previous fit's (alpha, rho, nu).  Equivalent
    to `calibrate_snapshot(..., use_global_init=False, init_params=...)`
    applied per slice, with the previous-snapshot parameters mapped onto the
    current expiries by nearest-T match.

    Typical latency: ~50–150 ms per slice — orders of magnitude faster than a
    cold `calibrate_snapshot`, suitable for tick-by-tick re-fits when the SABR
    parameters drift only modestly between snapshots.

    Parameters
    ----------
    df           : current snapshot (same schema as calibrate_snapshot)
    prev_result  : dict from a previous calibrate_snapshot / sabr_update call.
                   Used both for the SABR model instance (prev_result["_model"]
                   if `model` is None) and the per-slice warm-start params.
    model        : SABR instance.  Defaults to prev_result["_model"] when None.
    objective    : callable or str.  Defaults to "iv_wmse".
    """
    if model is None:
        model = prev_result.get("_model")
        if model is None:
            raise ValueError("model not in prev_result; pass it explicitly.")
    if not isinstance(model, SABR):
        raise TypeError(f"calibrate_snapshot_sabr_update expects a SABR model, got {type(model).__name__}")
    if objective is None:
        objective = get_objective("iv_wmse")
    elif isinstance(objective, str):
        objective = get_objective(objective)

    metric_names = DEFAULT_METRICS
    expiries     = sorted(df["t"].unique())
    n            = len(expiries)

    # ── Map prev params onto current expiries by nearest-T match ─────────────
    old_T      = np.asarray(prev_result["expiries"])
    old_params = prev_result["params"]
    def _nearest_prev(t_query):
        return old_params[int(np.argmin(np.abs(old_T - t_query)))]

    if verbose:
        print(f"\n[sabr_update] {n} slices, warm-start NM only\n")
    t0 = _time.time()

    fitted_params = [None] * n
    slice_metrics = {m: [] for m in metric_names}
    prev_p_for_cal = None

    for i, t_exp in enumerate(expiries):
        subset = df[df["t"] == t_exp]
        if len(subset) < min_points_per_slice:
            fitted_params[i] = fitted_params[i-1] if i > 0 else None
            for m in metric_names:
                slice_metrics[m].append(np.nan)
            continue

        k_obs  = subset["k"].values
        w_obs  = subset["w"].values
        iv_obs = subset["mark_iv"].values

        spreads = None
        vegas   = subset[vega_col].values if vega_col and vega_col in df.columns else None
        bid     = subset[bid_col].values  if bid_col  and bid_col  in df.columns else None
        ask     = subset[ask_col].values  if ask_col  and ask_col  in df.columns else None

        slice_obj = _bind_extra(objective, vegas=vegas, bid=bid, ask=ask)
        warm = _nearest_prev(t_exp)
        # Drop SABR's "t" key (not a calibration param) — fit_single_slice will reinject it
        warm_init = {k: v for k, v in warm.items() if k != "t"}

        t_slice = _time.time()
        params = fit_single_slice(
            k_obs, w_obs, iv_obs, t_exp,
            model=model,
            objective=slice_obj,
            spreads=spreads,
            prev_params=prev_p_for_cal,
            next_params=None,
            penalty_cal=penalty_cal,
            penalty_but=penalty_but,
            use_global_init=False,        # ← the whole point: no DE
            init_params=warm_init,        # ← warm start from previous fit
            nm_maxiter=nm_maxiter,         # ← looser NM than cold fit
            nm_tol=nm_tol,
        )
        slice_ms = (_time.time() - t_slice) * 1000
        fitted_params[i] = params
        prev_p_for_cal   = params

        w_fit  = model.w(k_obs, params)
        iv_fit = model.iv(k_obs, params, t_exp)
        evals  = evaluate_all(w_fit, w_obs, iv_fit, iv_obs, k_obs, t_exp,
                              bid=bid, ask=ask, metric_names=metric_names)
        for m in metric_names:
            slice_metrics[m].append(evals[m])

        if verbose:
            print(f"  [{i+1}/{n}] T={t_exp:.4f}  warm-NM {slice_ms:5.1f} ms  "
                  f"vwrmse={evals.get('vwrmse', float('nan')):.4f}")

    valid       = [(t, p) for t, p in zip(expiries, fitted_params) if p is not None]
    valid_idx   = [i for i, p in enumerate(fitted_params) if p is not None]
    expiries_out = [v[0] for v in valid]
    params_out   = [v[1] for v in valid]
    metrics_out  = {m: [slice_metrics[m][i] for i in valid_idx] for m in metric_names}
    n_pts        = [len(df[df["t"] == t]) for t in expiries_out]

    if verbose:
        print(f"\n[sabr_update] total {(_time.time()-t0)*1000:.0f} ms")

    return {
        "model_name": model.name + " (warm-start update)",
        "objective":  objective.__name__,
        "expiries":   expiries_out,
        "params":     params_out,
        "metrics":    metrics_out,
        "n_points":   n_pts,
        "_model":     model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def summary(result: dict):
    """Print a formatted summary of the calibration result."""
    print(f"\n{'─'*70}")
    print(f"  Model     : {result['model_name']}")
    print(f"  Objective : {result['objective']}")
    print(f"{'─'*70}")

    metric_names = list(result["metrics"].keys())
    header = f"{'T':>8}  {'n':>4}  " + "  ".join(f"{m:>12}" for m in metric_names)
    print(header)
    print("─" * len(header))

    for t, p, n, in zip(result["expiries"], result["params"], result["n_points"]):
        met_vals = [result["metrics"][m][result["expiries"].index(t)]
                    for m in metric_names]
        met_str  = "  ".join(f"{v:>12.6f}" for v in met_vals)
        print(f"{t:8.4f}  {n:4d}  {met_str}")

    print(f"\n  Mean metrics across slices:")
    for m in metric_names:
        vals = [v for v in result["metrics"][m] if not np.isnan(v)]
        print(f"    {m:<18}: {np.mean(vals):.6f}")
    print(f"{'─'*70}\n")


def metric_cal_arb_violations(result: dict,
                              k_grid: np.ndarray | None = None) -> float:
    """
    Fraction of (strike, expiry-pair) cells where total variance decreases
    with maturity — i.e. calendar-arbitrage violations.

    A return value of 0.0 means the surface is calendar-arb-free on the
    sampled grid. eSSVI/SSVI typically score 0 by construction; SVI and
    SABR can have >0 unless `enforce_arbfree=True` was used.
    """
    model    = result.get("_model")
    expiries = result["expiries"]
    params   = result["params"]
    if model is None or len(expiries) < 2:
        return 0.0
    if k_grid is None:
        k_grid = np.linspace(-0.5, 0.5, 101)
    violations = 0
    total      = 0
    for i in range(len(expiries) - 1):
        if params[i] is None or params[i + 1] is None:
            continue
        w_i   = model.w(k_grid, params[i])
        w_ip1 = model.w(k_grid, params[i + 1])
        violations += int(np.sum(w_ip1 < w_i - 1e-12))
        total      += len(k_grid)
    return float(violations / total) if total else 0.0


def metric_atm_smoothness(result: dict) -> float:
    """
    Standard deviation of ATM-IV first differences across consecutive
    expiries. Lower = smoother term structure. Useful for distinguishing
    eSSVI (smooth by construction) from free-fit SVI/SABR.
    """
    model    = result.get("_model")
    expiries = result["expiries"]
    params   = result["params"]
    if model is None or len(expiries) < 3:
        return 0.0
    atm = []
    for p, t in zip(params, expiries):
        if p is None:
            continue
        atm.append(float(model.iv(np.array([0.0]), p, t)[0]))
    if len(atm) < 3:
        return 0.0
    return float(np.std(np.diff(atm)))


def compare(*results, include_structural: bool = True) -> pd.DataFrame:
    """
    Compare multiple calibration results (different models or objectives)
    on the same snapshot. Returns a DataFrame of mean metrics per model.

    Parameters
    ----------
    include_structural : bool
        If True, also report `cal_arb_violations` (fraction of grid cells
        with calendar arbitrage) and `atm_smoothness` (std of ATM-IV
        first differences across expiries). These reveal what eSSVI buys
        you over free-fit SVI/SABR — namely no arbitrage and a smoother
        term structure — which per-slice hit-rate alone won't show.

    Example
    -------
    compare(result_svi, result_ssvi, result_sabr)
    """
    rows = []
    for r in results:
        row = {"model": r["model_name"], "objective": r["objective"]}
        for m, vals in r["metrics"].items():
            clean = [v for v in vals if not np.isnan(v)]
            row[m] = np.mean(clean) if clean else np.nan
        if include_structural:
            row["cal_arb_viol"]   = metric_cal_arb_violations(r)
            row["atm_smoothness"] = metric_atm_smoothness(r)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    print(df.to_string())
    return df
