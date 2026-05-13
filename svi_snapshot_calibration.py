"""
SVI Snapshot Calibration
========================
Calibrates the full SVI surface for a single order-book snapshot.

Designed for production use: take the latest (or any specific) file_timestamp
from your daily/hourly parquet, and calibrate one slice per expiry in seconds.

Typical runtime: < 5 seconds for a snapshot with 10-15 expiries.

Usage
-----
    from svi_snapshot_calibration import load_snapshot, calibrate_snapshot, summary

    df_snap = load_snapshot("path/to/daily_options_data.parquet")
    result  = calibrate_snapshot(df_snap)
    summary(result)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# RAW SVI
# ─────────────────────────────────────────────────────────────────────────────

def svi_raw(k, a, b, rho, m, sig):
    """Total implied variance w(k) under raw SVI."""
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2))


def svi_raw_dict(k, p):
    return svi_raw(k, p["a"], p["b"], p["rho"], p["m"], p["sig"])


# ─────────────────────────────────────────────────────────────────────────────
# BUTTERFLY ARBITRAGE CHECK  g(k) >= 0
# ─────────────────────────────────────────────────────────────────────────────

def g_func(k, a, b, rho, m, sig):
    k     = np.asarray(k, dtype=float)
    discr = np.sqrt((k - m) ** 2 + sig ** 2)
    w     = a + b * (rho * (k - m) + discr)
    dw    = b * rho + b * (k - m) / discr
    d2w   = b * sig ** 2 / discr ** 3
    return (1 - k * dw / (2 * w)) ** 2 - (dw ** 2 / 4) * (1 / w + 0.25) + d2w / 2


def min_g_value(a, b, rho, m, sig, k_range=(-4.0, 4.0), n=600):
    k_grid = np.linspace(k_range[0], k_range[1], n)
    return float(np.min(g_func(k_grid, a, b, rho, m, sig)))


def has_butterfly_arb(params, k_range=(-4.0, 4.0)):
    return min_g_value(**params, k_range=k_range) < -1e-6


# ─────────────────────────────────────────────────────────────────────────────
# CALENDAR SPREAD CHECK (crossedness between two consecutive slices)
# ─────────────────────────────────────────────────────────────────────────────

def crossedness(p1, p2, k_range=(-5.0, 5.0), n=2000):
    """
    Maximum amount by which slice p1 (earlier) exceeds slice p2 (later).
    Zero means no calendar spread arbitrage between these two slices.
    """
    k_grid = np.linspace(k_range[0], k_range[1], n)
    diff   = svi_raw_dict(k_grid, p1) - svi_raw_dict(k_grid, p2)
    return float(max(0.0, diff.max()))


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-SLICE FIT
# ─────────────────────────────────────────────────────────────────────────────

def _pack(a, b, rho, m, sig):
    """Map constrained params to unconstrained reals for the optimiser."""
    return np.array([
        a,
        np.log(max(b, 1e-9)),           # b  > 0
        np.arctanh(np.clip(rho, -0.9999, 0.9999)),
        m,
        np.log(max(sig, 1e-9)),         # sig > 0
    ])


def _unpack(x):
    return {
        "a":   x[0],
        "b":   float(np.exp(x[1])),
        "rho": float(np.tanh(x[2])),
        "m":   x[3],
        "sig": float(np.exp(x[4])),
    }


def fit_single_slice(k_obs, w_obs, t,
                     prev_params=None,
                     next_params=None,
                     penalty_cal=500.0,
                     use_global_init=True):
    """
    Fit raw SVI to one expiry slice.

    Parameters
    ----------
    k_obs        : array  observed log-strikes
    w_obs        : array  observed total implied variances
    t            : float  time to expiry (years)
    prev_params  : dict or None  fitted params for the previous (shorter) expiry
    next_params  : dict or None  fitted params for the next (longer) expiry
    penalty_cal  : float  penalty weight for calendar spread violations
    use_global_init : bool  use differential_evolution for a robust global start

    Returns
    -------
    dict  raw SVI params {a, b, rho, m, sig}
    """
    k_obs = np.asarray(k_obs, dtype=float)
    w_obs = np.asarray(w_obs, dtype=float)

    # ATM total variance as a simple anchor
    atm_w = float(np.interp(0.0, np.sort(k_obs), w_obs[np.argsort(k_obs)]))

    def objective(x):
        p  = _unpack(x)
        wf = svi_raw_dict(k_obs, p)

        # Weighted squared error (weight by 1/spread proxy ~ uniform here)
        fit_err = float(np.nanmean((wf - w_obs) ** 2))

        # Penalty: negative total variance
        min_w = p["a"] + p["b"] * p["sig"] * np.sqrt(1 - p["rho"] ** 2)
        neg_pen = max(0.0, -min_w) * 1e4

        # Penalty: butterfly arbitrage (soft)
        mg = min_g_value(**p)
        but_pen = max(0.0, -mg) * 1e3

        # Penalty: calendar spread with neighbours
        cal_pen = 0.0
        if prev_params is not None:
            cal_pen += crossedness(prev_params, p) * penalty_cal
        if next_params is not None:
            cal_pen += crossedness(p, next_params) * penalty_cal

        return fit_err + neg_pen + but_pen + cal_pen

    # ── Initial guess: use ATM variance to set a sensible starting point ─────
    a0   = atm_w * 0.9
    b0   = 0.1
    rho0 = -0.7
    m0   = 0.0
    sig0 = 0.3
    x0   = _pack(a0, b0, rho0, m0, sig0)

    if use_global_init:
        # Differential evolution gives a robust global start (fast for 5 params)
        bounds = [
            (-0.5, atm_w * 2),          # a
            (np.log(1e-4), np.log(5.0)), # log b
            (-3.0, 3.0),                 # arctanh rho
            (-2.0, 2.0),                 # m
            (np.log(1e-4), np.log(3.0)), # log sig
        ]
        de_res = differential_evolution(
            objective, bounds,
            seed=42, maxiter=300, tol=1e-7,
            popsize=8, mutation=(0.5, 1.5), recombination=0.9,
            workers=1,
        )
        x0 = de_res.x

    # ── Local polish ──────────────────────────────────────────────────────────
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-9, "fatol": 1e-9})

    return _unpack(res.x)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  —  single snapshot
# ─────────────────────────────────────────────────────────────────────────────

def load_snapshot(path: str,
                  timestamp=None,
                  iv_col: str = "mark_iv",
                  forward_col: str = None,
                  strike_col: str = None) -> pd.DataFrame:
    """
    Load a parquet file and extract a single order-book snapshot.

    Parameters
    ----------
    path          : str   path to parquet (daily, hourly, or 30min)
    timestamp     : optional — if None, the FIRST file_timestamp is used.
                    Pass a specific value to select a different snapshot.
    iv_col        : str   column name for implied volatility  (default 'mark_iv')
    forward_col   : str   column for forward/futures price (auto-detected if None)
    strike_col    : str   column for strike price (auto-detected if None)

    Returns
    -------
    DataFrame with columns  k, w, t  ready for calibration
    """
    df = pd.read_parquet(path)

    # ── Select snapshot ───────────────────────────────────────────────────────
    if "file_timestamp" not in df.columns:
        raise ValueError("Column 'file_timestamp' not found in dataframe.")

    if timestamp is None:
        timestamp = df["file_timestamp"].iloc[0]
        print(f"Using first snapshot: {timestamp}")
    else:
        print(f"Using snapshot: {timestamp}")

    df = df[df["file_timestamp"] == timestamp].copy()
    print(f"  {len(df)} rows in this snapshot")

    # ── Time to maturity ──────────────────────────────────────────────────────
    if "t" not in df.columns:
        if "expiry" in df.columns:
            ref = pd.to_datetime(timestamp)
            df["t"] = (pd.to_datetime(df["expiry"]) - ref).dt.total_seconds() / (365.25 * 24 * 3600)
        elif "time_to_maturity" in df.columns:
            df["t"] = df["time_to_maturity"]
        else:
            raise ValueError("Cannot determine time to maturity. Add a 't' column or 'expiry' column.")

    # ── Forward price ─────────────────────────────────────────────────────────
    if forward_col is None:
        for col in ["forward", "future_price", "underlying_price", "index_price"]:
            if col in df.columns:
                forward_col = col
                break

    # ── Strike column ─────────────────────────────────────────────────────────
    if strike_col is None:
        for col in ["strike", "strike_price"]:
            if col in df.columns:
                strike_col = col
                break

    # ── Log-strike k = log(K/F) ───────────────────────────────────────────────
    if "k" not in df.columns:
        if forward_col and strike_col:
            df["k"] = np.log(df[strike_col] / df[forward_col])
        else:
            raise ValueError(
                f"Cannot compute log-strike. Provide forward_col and strike_col, "
                f"or add a 'k' column. Available columns: {list(df.columns)}"
            )

    # ── Implied vol: ensure it's in (0, 1) range ──────────────────────────────
    if df[iv_col].max() > 5:
        df[iv_col] = df[iv_col] / 100.0

    # ── Total implied variance w = sigma^2 * t ────────────────────────────────
    if "w" not in df.columns:
        df["w"] = df[iv_col] ** 2 * df["t"]

    # ── Clean: drop bad rows ──────────────────────────────────────────────────
    before = len(df)
    df = df[df["t"]     > 1e-5].copy()
    df = df[df[iv_col]  > 1e-4].copy()
    df = df[df[iv_col]  < 5.0 ].copy()
    df = df[df["w"]     > 0   ].copy()
    df = df.dropna(subset=["k", "w", "t"])
    print(f"  {len(df)} rows after cleaning (dropped {before - len(df)})")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# FULL SNAPSHOT CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_snapshot(df: pd.DataFrame,
                        penalty_cal: float = 500.0,
                        min_points_per_slice: int = 5,
                        use_global_init: bool = True,
                        verbose: bool = True) -> dict:
    """
    Calibrate the SVI surface for every expiry in a single snapshot.

    Parameters
    ----------
    df                    : output of load_snapshot()
    penalty_cal           : calendar spread penalty weight
    min_points_per_slice  : skip expiries with fewer data points than this
    use_global_init       : use differential_evolution for robust initialisation
    verbose               : print progress

    Returns
    -------
    dict with keys:
        expiries    : list of t values (sorted)
        svi_params  : list of raw SVI dicts (one per expiry)
        n_points    : list of number of data points per slice
    """
    expiries = sorted(df["t"].unique())
    n        = len(expiries)

    if verbose:
        print(f"\nCalibrating {n} expiry slices…")

    svi_params = [None] * n

    # Forward pass: fit each slice, passing the previous fitted slice as constraint
    for i, t_exp in enumerate(expiries):
        subset = df[df["t"] == t_exp]

        if len(subset) < min_points_per_slice:
            if verbose:
                print(f"  [{i+1}/{n}] T={t_exp:.4f}  SKIPPED ({len(subset)} points < {min_points_per_slice})")
            # Fill with previous slice if available, else skip
            svi_params[i] = svi_params[i-1] if i > 0 and svi_params[i-1] else None
            continue

        k_obs = subset["k"].values
        w_obs = subset["w"].values

        # Use the previously fitted slice as prev_params (calendar constraint)
        # We don't have next_params on the forward pass, so no look-ahead needed
        prev_p = svi_params[i-1] if i > 0 and svi_params[i-1] is not None else None

        params = fit_single_slice(
            k_obs, w_obs, t_exp,
            prev_params=prev_p,
            next_params=None,
            penalty_cal=penalty_cal,
            use_global_init=use_global_init,
        )
        svi_params[i] = params

        if verbose:
            mg    = min_g_value(**params)
            cross = crossedness(prev_p, params) if prev_p else 0.0
            flag  = "⚠ butterfly" if mg < 0 else "ok"
            print(f"  [{i+1}/{n}] T={t_exp:.4f}  n={len(subset):3d}  "
                  f"min_g={mg:+.4f}  cal_cross={cross:.2e}  [{flag}]")

    # Filter out any None entries (skipped slices)
    valid = [(t, p) for t, p in zip(expiries, svi_params) if p is not None]
    expiries_out = [v[0] for v in valid]
    params_out   = [v[1] for v in valid]
    n_pts        = [len(df[df["t"] == t]) for t in expiries_out]

    return {
        "expiries":   expiries_out,
        "svi_params": params_out,
        "n_points":   n_pts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def summary(result: dict):
    """Print a clean summary table of the calibrated surface."""
    print(f"\n{'T':>8}  {'n':>4}  {'a':>9}  {'b':>8}  {'rho':>7}  "
          f"{'m':>7}  {'sig':>8}  {'min_g':>8}  {'cal_arb':>9}")
    print("─" * 82)
    prev = None
    for t, p, n in zip(result["expiries"], result["svi_params"], result["n_points"]):
        mg    = min_g_value(**p)
        cross = crossedness(prev, p) if prev else 0.0
        flag  = " ⚠" if (mg < 0 or cross > 1e-6) else ""
        print(f"{t:8.4f}  {n:4d}  {p['a']:9.6f}  {p['b']:8.5f}  {p['rho']:7.4f}  "
              f"{p['m']:7.4f}  {p['sig']:8.5f}  {mg:8.4f}  {cross:9.2e}{flag}")
        prev = p


def add_fitted_cols(df: pd.DataFrame, result: dict) -> pd.DataFrame:
    """
    Add columns  w_fit  and  iv_fit  to the snapshot dataframe.
    """
    df = df.copy()
    df["w_fit"]  = np.nan
    df["iv_fit"] = np.nan

    for t, p in zip(result["expiries"], result["svi_params"]):
        mask   = df["t"] == t
        w_fit  = svi_raw_dict(df.loc[mask, "k"].values, p)
        df.loc[mask, "w_fit"]  = w_fit
        df.loc[mask, "iv_fit"] = np.sqrt(np.maximum(w_fit / t, 0.0))

    return df


def jw_table(result: dict) -> pd.DataFrame:
    """
    Return a DataFrame of SVI Jump-Wings parameters for each expiry.
    Useful for quick inspection by traders.
    """
    rows = []
    for t, p in zip(result["expiries"], result["svi_params"]):
        wt      = svi_raw(0.0, **p)
        sqrt_wt = np.sqrt(max(wt, 1e-12))
        discr_m = np.sqrt(p["m"] ** 2 + p["sig"] ** 2)
        rows.append({
            "T":       t,
            "vt":      wt / t,
            "psit":    (p["b"] / (2 * sqrt_wt)) * (-p["m"] / discr_m + p["rho"]),
            "pt":      p["b"] * (1 - p["rho"]) / sqrt_wt,
            "ct":      p["b"] * (1 + p["rho"]) / sqrt_wt,
            "varmint": (p["a"] + p["b"] * p["sig"] * np.sqrt(1 - p["rho"] ** 2)) / t,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys

    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "2 - Data", "Parquets",
        "daily_options_data.parquet"
    )

    if not os.path.exists(DATA_PATH):
        print(f"Data not found at {DATA_PATH}. Update DATA_PATH.")
        sys.exit(0)

    # Load the first snapshot
    df_snap = load_snapshot(DATA_PATH)          # uses first file_timestamp by default

    # To use a specific timestamp:
    # df_snap = load_snapshot(DATA_PATH, timestamp="2024-01-15 08:00:00")

    # Calibrate
    result = calibrate_snapshot(df_snap, verbose=True)

    # Summary table
    summary(result)

    # Jump-Wings parameters (trader-friendly view)
    print("\nJump-Wings parameters:")
    print(jw_table(result).to_string(index=False))

    # Add fitted values back to the dataframe
    df_fitted = add_fitted_cols(df_snap, result)
    print("\nFit quality sample:")
    print(df_fitted[["t", "k", "w", "w_fit", "iv_fit"]].head(15).to_string(index=False))
