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
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings("ignore")

from vol_models import VolModel, RawSVI, get_model
from objectives import get_objective, evaluate_all, DEFAULT_METRICS


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
    p0 = model.initial_guess(k_obs, w_obs, t)
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
        options={"maxiter": 5000, "xatol": 1e-9, "fatol": 1e-9},
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
        model = RawSVI()
    if objective is None:
        objective = get_objective("iv_wmse")
    elif isinstance(objective, str):
        objective = get_objective(objective)

    metric_names = metrics or DEFAULT_METRICS
    expiries     = sorted(df["t"].unique())
    n            = len(expiries)

    if verbose:
        print(f"\nModel     : {model.name}")
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

        prev_p = fitted_params[i-1] if i > 0 and fitted_params[i-1] is not None else None

        params = fit_single_slice(
            k_obs, w_obs, iv_obs, t_exp,
            model=model,
            objective=objective,
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
                               spreads=spreads, metric_names=metric_names)
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

    return {
        "model_name": model.name,
        "objective":  objective.__name__,
        "expiries":   expiries_out,
        "params":     params_out,
        "metrics":    metrics_out,
        "n_points":   n_pts,
        "_model":     model,   # kept for plotting
    }



def calibrate_global_essvi(
    df:           pd.DataFrame,
    model,                        # eSSVI instance
    objective     = None,
    penalty_cal:  float = 500.0,
    penalty_but:  float = 200.0,
    verbose:      bool  = True,
) -> dict:
    """
    Global calibration for eSSVI: rho_0, rho_m, a are shared across all slices;
    theta (and optionally eta, gamma) are fitted per slice.

    This matches the Hendriks & Martini (2019) calibration design where the
    rho(theta) function is a surface-level parameter estimated jointly.
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
            "t":      t,
            "k":      sub["k"].values,
            "w":      sub["w"].values,
            "iv":     sub["mark_iv"].values,
        })

    # ── Parametrisation ───────────────────────────────────────────────────────
    # Global params:  rho_0, rho_m, a, eta, gamma   (shared)
    # Per-slice:      theta_i                         (one per expiry)
    # Total params:   5 + n

    def pack_global(rho_0, rho_m, a, eta, gamma, thetas):
        gamma_c = np.clip(gamma, 1e-6, 0.5 - 1e-6)
        return np.concatenate([
            [np.arctanh(np.clip(rho_0,  -0.9999, 0.9999))],
            [np.arctanh(np.clip(rho_m,  -0.9999, 0.9999))],
            [np.log(max(a,     1e-9))],
            [np.log(max(eta,   1e-9))],
            [np.log(gamma_c / (0.5 - gamma_c))],
            np.log(np.maximum(thetas, 1e-9)),
        ])

    def unpack_global(x):
        rho_0 = float(np.tanh(x[0]))
        rho_m = float(np.tanh(x[1]))
        a     = float(np.exp(x[2]))
        eta   = float(np.exp(x[3]))
        gr    = np.exp(x[4])
        gamma = float(0.5 * gr / (1 + gr))
        thetas = np.exp(x[5:])
        return rho_0, rho_m, a, eta, gamma, thetas

    def objective_fn(x):
        rho_0, rho_m, a, eta, gamma, thetas = unpack_global(x)
        total = 0.0
        prev_params = None

        for i, (sl, theta) in enumerate(zip(slices, thetas)):
            params = {
                "theta": float(theta),
                "eta":   eta,
                "gamma": gamma,
                "rho_0": rho_0,
                "rho_m": rho_m,
                "a":     a,
            }
            w_fit  = model.w(sl["k"], params)
            iv_fit = model.iv(sl["k"], params, sl["t"])

            # Fit error
            total += objective(w_fit, sl["w"], iv_fit, sl["iv"],
                               sl["k"], sl["t"], None)

            # Butterfly penalty
            mg     = model.min_g(params)
            total += max(0.0, -mg) * penalty_but

            # Calendar spread penalty
            if prev_params is not None:
                total += crossedness(model, prev_params, params) * penalty_cal

            # Enforce theta non-decreasing (calendar spread at surface level)
            if i > 0:
                total += max(0.0, thetas[i-1] - theta) * penalty_cal * 10

            prev_params = params

        return total

    # ── Initial guess ─────────────────────────────────────────────────────────
    atm_thetas = np.array([
        float(np.interp(0.0, np.sort(sl["k"]), sl["w"][np.argsort(sl["k"])]))
        for sl in slices
    ])
    # Enforce monotonicity in initial theta
    atm_thetas = np.maximum.accumulate(atm_thetas)

    x0 = pack_global(
        rho_0=-0.8, rho_m=-0.3, a=0.5,
        eta=1.5, gamma=0.4,
        thetas=atm_thetas,
    )

    # Global bounds: 5 surface params + n theta params
    bounds = (
        [(-3.5, 3.5), (-3.5, 3.5),                   # arctanh rho_0, rho_m
         (np.log(1e-2), np.log(10.0)),                # log a
         (np.log(1e-3), np.log(10.0)),                # log eta
         (-5.0, 5.0)]                                 # gamma transform
        + [(np.log(1e-5), np.log(10.0))] * n          # log theta per slice
    )

    if verbose:
        print(f"\nModel     : {model.name} (global)")
        print(f"Objective : {objective.__name__}")
        print(f"Slices    : {n}  |  Params: {len(x0)} total\n")

    de_res = differential_evolution(
        objective_fn, bounds,
        seed=42, maxiter=500, tol=1e-7,
        popsize=10, mutation=(0.5, 1.5), recombination=0.9,
        workers=1,
    )
    res = minimize(
        objective_fn, de_res.x, method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-9},
    )

    rho_0, rho_m, a, eta, gamma, thetas = unpack_global(res.x)

    if verbose:
        print(f"  rho_0={rho_0:.3f}  rho_m={rho_m:.3f}  a={a:.3f}")
        print(f"  eta={eta:.3f}  gamma={gamma:.3f}\n")

    # ── Build result dict ─────────────────────────────────────────────────────
    params_out   = []
    metrics_out  = {m: [] for m in DEFAULT_METRICS}

    for sl, theta in zip(slices, thetas):
        p = {"theta": float(theta), "eta": eta, "gamma": gamma,
             "rho_0": rho_0, "rho_m": rho_m, "a": a}
        params_out.append(p)

        w_fit  = model.w(sl["k"], p)
        iv_fit = model.iv(sl["k"], p, sl["t"])
        evals  = evaluate_all(w_fit, sl["w"], iv_fit, sl["iv"],
                               sl["k"], sl["t"])
        for m in DEFAULT_METRICS:
            metrics_out[m].append(evals[m])

        if verbose:
            print(f"  T={sl['t']:.4f}  theta={theta:.5f}  "
                  f"rho={model._rho(theta, p):.3f}  "
                  f"iv_rmse={evals['iv_rmse']:.4f}")

    return {
        "model_name": model.name,
        "objective":  objective.__name__,
        "expiries":   [sl["t"] for sl in slices],
        "params":     params_out,
        "metrics":    metrics_out,
        "n_points":   [len(df[df["t"] == sl["t"]]) for sl in slices],
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


def compare(*results) -> pd.DataFrame:
    """
    Compare multiple calibration results (different models or objectives)
    on the same snapshot. Returns a DataFrame of mean metrics per model.

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
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    print(df.to_string())
    return df


def add_fitted_cols(df: pd.DataFrame, result: dict) -> pd.DataFrame:
    """Add w_fit and iv_fit columns to the snapshot dataframe."""
    model  = result.get("_model")
    if model is None:
        model = get_model("svi")   # fallback

    df = df.copy()
    df["w_fit"]  = np.nan
    df["iv_fit"] = np.nan

    for t, p in zip(result["expiries"], result["params"]):
        mask   = df["t"] == t
        k_vals = df.loc[mask, "k"].values
        w_fit  = model.w(k_vals, p)
        df.loc[mask, "w_fit"]  = w_fit
        df.loc[mask, "iv_fit"] = model.mark_iv(k_vals, p, t)

    return df
