"""
vol_calibration_benchmark.py
============================
Benchmarks multiple objective functions against each other for a given
snapshot, using model-agnostic metrics that are independent of the
objective used during calibration.

The key design principle: metrics used for COMPARISON must be
different from the objectives used for CALIBRATION.
Otherwise you are just checking that each objective minimises itself.

Evaluation metrics (all objective-agnostic)
-------------------------------------------
Absolute IV metrics:
    rmse_iv          Root mean squared error on implied vol
    mae_iv           Mean absolute error on implied vol (robust to outliers)
    max_violation    Worst single IV miss (useful for risk management)

Relative IV metrics  (percentage of observed IV level):
    rrmse_iv         Relative RMSE on IV  = rmse(|err|/iv_obs)
    rmae_iv          Relative MAE on IV   = mean(|err|/iv_obs)
    max_rel_viol     Worst relative IV miss

Relative price metrics  (percentage of observed price):
    rrmse_price      Relative RMSE on price = rmse(|err_price|/price_obs)
    rmae_price       Relative MAE on price
    max_rel_price    Worst relative price miss

    These are more meaningful than absolute metrics for BTC options where
    wing vols can be 2-4x ATM vol. A 1 vol point error at 20% ATM vol
    is a 5% relative error; the same error at 80% wing vol is only 1.25%.

Absolute price metrics:
    rmse_price       RMSE on forward-normalised BS prices (coin units)
    mae_price        MAE on forward-normalised BS prices

Spread metrics:
    ba_capture       Fraction of strikes where model IV is within the bid-ask

Arbitrage metrics:
    butterfly_ok     Whether the calibrated slice is butterfly-arbitrage-free
    calendar_ok      Whether the slice respects calendar spread constraint

Usage
-----
    from vol_calibration_benchmark import benchmark_objectives, print_benchmark

    results = benchmark_objectives(
        df_snap,
        model_spec   = SVI_ModelSpec(),
        objectives   = ["sse_total_var", "vega_weighted_iv",
                        "bid_ask_normalized", "combined"],
    )
    print_benchmark(results)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

from vol_calibration_engine  import ModelSpec, fit_slice, SVI_ModelSpec
from svi_objective_functions import build_objective, OBJECTIVES


# ─────────────────────────────────────────────────────────────────────────────
# BS PRICE HELPER  (forward-normalised, coin numeraire F=1)
# ─────────────────────────────────────────────────────────────────────────────

def _bs_price(k: np.ndarray, iv: np.ndarray, t: float,
              is_call: bool = True) -> np.ndarray:
    k, iv   = np.asarray(k, float), np.asarray(iv, float)
    safe_iv = np.maximum(iv, 1e-8)
    sqrt_t  = np.sqrt(max(t, 1e-8))
    d1 = (-k + 0.5 * safe_iv ** 2 * t) / (safe_iv * sqrt_t)
    d2 = d1 - safe_iv * sqrt_t
    if is_call:
        return norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
    else:
        return np.exp(k) * norm.cdf(-d2) - norm.cdf(-d1)


# ─────────────────────────────────────────────────────────────────────────────
# PER-SLICE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_slice_metrics(
        k_obs:        np.ndarray,
        iv_obs:       np.ndarray,
        iv_model:     np.ndarray,
        t:            float,
        bid_iv:       Optional[np.ndarray] = None,
        ask_iv:       Optional[np.ndarray] = None,
        model_spec:   Optional[ModelSpec]  = None,
        params:       Optional[dict]       = None,
        prev_params:  Optional[dict]       = None,
) -> dict:
    """
    Compute all evaluation metrics for one calibrated slice.
    All metrics are independent of the objective used to calibrate.

    Parameters
    ----------
    k_obs       : log-strike array
    iv_obs      : observed mid implied vols
    iv_model    : model implied vols at the same strikes
    t           : time to expiry
    bid_iv      : bid implied vols (optional — needed for bid_ask_capture)
    ask_iv      : ask implied vols (optional)
    model_spec  : ModelSpec instance (needed for arbitrage checks)
    params      : calibrated params dict (needed for arbitrage checks)
    prev_params : previous slice params (needed for calendar check)

    Returns
    -------
    dict of scalar metrics
    """
    k_obs    = np.asarray(k_obs,    float)
    iv_obs   = np.asarray(iv_obs,   float)
    iv_model = np.asarray(iv_model, float)

    err_iv = iv_model - iv_obs

    # ── Absolute IV metrics ────────────────────────────────────────────────
    rmse_iv  = float(np.sqrt(np.nanmean(err_iv ** 2)))
    mae_iv   = float(np.nanmean(np.abs(err_iv)))
    max_viol = float(np.nanmax(np.abs(err_iv)))

    # ── Relative IV metrics ────────────────────────────────────────────────
    # Expressed as a fraction of the observed IV level at each strike.
    # More meaningful than absolute metrics when wing vols are 2-4x ATM
    # (common in BTC options).
    #
    # Critical note: relative metrics are unstable when iv_obs is very small.
    # We floor at 1% vol to avoid division instability for near-zero vols.
    iv_floor    = np.maximum(iv_obs, 0.01)
    rel_err_iv  = err_iv / iv_floor
    rrmse_iv    = float(np.sqrt(np.nanmean(rel_err_iv ** 2)))
    rmae_iv     = float(np.nanmean(np.abs(rel_err_iv)))
    max_rel_viol = float(np.nanmax(np.abs(rel_err_iv)))

    # ── Price metrics ───────────────────────────────────────────────────────
    call_mask   = k_obs >= 0
    price_obs   = np.where(call_mask,
                           _bs_price(k_obs, iv_obs,   t, is_call=True),
                           _bs_price(k_obs, iv_obs,   t, is_call=False))
    price_model = np.where(call_mask,
                           _bs_price(k_obs, iv_model, t, is_call=True),
                           _bs_price(k_obs, iv_model, t, is_call=False))
    err_price   = price_model - price_obs

    # Absolute price metrics
    rmse_price  = float(np.sqrt(np.nanmean(err_price ** 2)))
    mae_price   = float(np.nanmean(np.abs(err_price)))

    # Relative price metrics
    # Floor price at 1e-6 of forward to avoid division by near-zero OTM prices.
    # This floor matters: a deep OTM option priced at 1e-5 with a 1e-6 model
    # error has a 10% relative error but is economically irrelevant.
    # You may want to exclude strikes below a price threshold entirely.
    price_floor     = np.maximum(np.abs(price_obs), 1e-6)
    rel_err_price   = err_price / price_floor
    rrmse_price     = float(np.sqrt(np.nanmean(rel_err_price ** 2)))
    rmae_price      = float(np.nanmean(np.abs(rel_err_price)))
    max_rel_price   = float(np.nanmax(np.abs(rel_err_price)))

    # ── Bid-ask capture ─────────────────────────────────────────────────────
    # Fraction of strikes where model IV sits inside [bid_iv, ask_iv]
    if bid_iv is not None and ask_iv is not None:
        bid_iv = np.asarray(bid_iv, float)
        ask_iv = np.asarray(ask_iv, float)
        inside = (iv_model >= bid_iv) & (iv_model <= ask_iv)
        ba_capture = float(np.mean(inside))
    else:
        ba_capture = np.nan

    # ── Arbitrage metrics ───────────────────────────────────────────────────
    butterfly_ok  = np.nan
    calendar_ok   = np.nan
    min_g         = np.nan
    cal_cross     = np.nan

    if model_spec is not None and params is not None:
        min_g        = float(model_spec.check_butterfly(params))
        butterfly_ok = float(min_g >= -1e-6)

        if prev_params is not None:
            cal_cross   = float(model_spec.check_calendar(prev_params, params))
            calendar_ok = float(cal_cross <= 1e-6)

    return {
        "rmse_iv":       rmse_iv,
        "mae_iv":        mae_iv,
        "max_violation": max_viol,
        "rrmse_iv":      rrmse_iv,
        "rmae_iv":       rmae_iv,
        "max_rel_viol":  max_rel_viol,
        "rmse_price":    rmse_price,
        "mae_price":     mae_price,
        "rrmse_price":   rrmse_price,
        "rmae_price":    rmae_price,
        "max_rel_price": max_rel_price,
        "ba_capture":    ba_capture,
        "butterfly_ok":  butterfly_ok,
        "calendar_ok":   calendar_ok,
        "min_g":         min_g,
        "cal_cross":     cal_cross,
        "n_strikes":     len(k_obs),
        "t":             t,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK: calibrate with each objective, evaluate with common metrics
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_objectives(
        df:              pd.DataFrame,
        model_spec:      ModelSpec,
        objectives:      Optional[list] = None,
        penalty_butterfly: float = 1e3,
        penalty_calendar:  float = 500.0,
        min_points:        int   = 5,
        use_global_init:   bool  = True,
        verbose:           bool  = True,
        extra_params_fn    = None,
) -> pd.DataFrame:
    """
    Calibrate the surface with each objective function and evaluate
    all slices using objective-agnostic metrics.

    Parameters
    ----------
    df          : output of load_snapshot() — needs k, w, t columns.
                  Optionally bid_iv and ask_iv for bid-ask capture metric.
    model_spec  : ModelSpec instance
    objectives  : list of objective keys to benchmark.
                  Defaults to all keys in OBJECTIVES.
    verbose     : print progress

    Returns
    -------
    DataFrame with one row per (objective, expiry) combination.
    Use .groupby("objective").mean() for a surface-level summary.
    """
    if objectives is None:
        objectives = list(OBJECTIVES.keys())

    has_ba = "bid_iv" in df.columns and "ask_iv" in df.columns
    expiries = sorted(df["t"].unique())

    all_rows = []

    for obj_name in objectives:
        if verbose:
            print(f"\n{'─'*55}")
            print(f"Objective: {obj_name}")
            print(f"{'─'*55}")

        obj_fn       = build_objective(obj_name)
        prev_params  = None
        slice_params = {}

        for i, t_exp in enumerate(expiries):
            sub = df[df["t"] == t_exp]
            if len(sub) < min_points:
                if verbose:
                    print(f"  T={t_exp:.4f}  SKIPPED")
                continue

            k_obs  = sub["k"].values
            w_obs  = sub["w"].values
            iv_obs = np.sqrt(np.maximum(w_obs / max(t_exp, 1e-8), 0.0))

            bid_iv = sub["bid_iv"].values if has_ba else None
            ask_iv = sub["ask_iv"].values if has_ba else None

            bid_ask_width = (ask_iv - bid_iv) if has_ba else None

            extra = {}
            if extra_params_fn is not None:
                extra = extra_params_fn(t_exp, sub)

            # ── Calibrate ──────────────────────────────────────────────────
            params = fit_slice(
                k_obs          = k_obs,
                iv_obs         = iv_obs,
                t              = t_exp,
                model_spec     = model_spec,
                objective_fn   = obj_fn,
                prev_params    = prev_params,
                bid_ask_width  = bid_ask_width,
                penalty_butterfly = penalty_butterfly,
                penalty_calendar  = penalty_calendar,
                use_global_init   = use_global_init,
                extra_params      = extra,
            )
            slice_params[t_exp] = params

            # ── Evaluate with objective-agnostic metrics ───────────────────
            iv_model = model_spec.iv_from_params(k_obs, t_exp, params)

            metrics = compute_slice_metrics(
                k_obs       = k_obs,
                iv_obs      = iv_obs,
                iv_model    = iv_model,
                t           = t_exp,
                bid_iv      = bid_iv,
                ask_iv      = ask_iv,
                model_spec  = model_spec,
                params      = params,
                prev_params = prev_params,
            )

            row = {"objective": obj_name, **metrics}
            all_rows.append(row)

            prev_params = params

            if verbose:
                ba_str = f"  ba_capture={metrics['ba_capture']:.2%}" if has_ba else ""
                print(f"  T={t_exp:.4f}  n={metrics['n_strikes']:3d}  "
                      f"rmse_iv={metrics['rmse_iv']:.5f}  "
                      f"mae_iv={metrics['mae_iv']:.5f}  "
                      f"rmse_price={metrics['rmse_price']:.6f}"
                      f"{ba_str}")

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLES
# ─────────────────────────────────────────────────────────────────────────────

def surface_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-slice results into a surface-level summary,
    one row per objective.

    Aggregation rules
    -----------------
    rmse_iv, mae_iv, max_violation, rmse_price, mae_price : mean across slices
    ba_capture   : mean (average fraction of strikes inside spread)
    butterfly_ok : min  (one bad slice = surface is arbitrageable)
    calendar_ok  : min  (same reasoning)
    """
    agg = {
        # Absolute IV
        "rmse_iv":       "mean",
        "mae_iv":        "mean",
        "max_violation": "mean",
        # Relative IV
        "rrmse_iv":      "mean",
        "rmae_iv":       "mean",
        "max_rel_viol":  "mean",
        # Absolute price
        "rmse_price":    "mean",
        "mae_price":     "mean",
        # Relative price
        "rrmse_price":   "mean",
        "rmae_price":    "mean",
        "max_rel_price": "mean",
        # Spread and arbitrage — aggregate conservatively
        "ba_capture":    "mean",
        "butterfly_ok":  "min",
        "calendar_ok":   "min",
    }
    # Only aggregate columns that exist and are not all-NaN
    valid_agg = {k: v for k, v in agg.items()
                 if k in results.columns and results[k].notna().any()}

    summary = (results
               .groupby("objective")
               .agg(valid_agg)
               .round(6))

    # Rank by rrmse_iv (relative) as primary — more meaningful than absolute
    # for BTC where wing vols can be 2-4x ATM
    rank_col = "rrmse_iv" if "rrmse_iv" in summary.columns else "rmse_iv"
    summary["rank"] = summary[rank_col].rank().astype(int)
    summary = summary.sort_values("rank")

    return summary


def print_benchmark(results: pd.DataFrame, show_slices: bool = False):
    """
    Print a clean benchmark report split into metric groups.

    Parameters
    ----------
    results     : output of benchmark_objectives()
    show_slices : if True, also print the per-slice breakdown
    """
    summary = surface_summary(results)

    fmt = {
        "rmse_iv":       ("{:.5f}",  "abs vol"),
        "mae_iv":        ("{:.5f}",  "abs vol"),
        "max_violation": ("{:.5f}",  "abs vol"),
        "rrmse_iv":      ("{:.2%}",  "rel vol"),
        "rmae_iv":       ("{:.2%}",  "rel vol"),
        "max_rel_viol":  ("{:.2%}",  "rel vol"),
        "rmse_price":    ("{:.6f}",  "abs price"),
        "mae_price":     ("{:.6f}",  "abs price"),
        "rrmse_price":   ("{:.2%}",  "rel price"),
        "rmae_price":    ("{:.2%}",  "rel price"),
        "max_rel_price": ("{:.2%}",  "rel price"),
        "ba_capture":    ("{:.2%}",  "spread"),
        "butterfly_ok":  ("{:.0f}",  "arbitrage"),
        "calendar_ok":   ("{:.0f}",  "arbitrage"),
        "rank":          ("{:.0f}",  "rank"),
    }

    # Group metrics for cleaner display
    groups = {
        "RELATIVE METRICS (% of observed level)": [
            "rrmse_iv", "rmae_iv", "max_rel_viol",
            "rrmse_price", "rmae_price", "max_rel_price",
        ],
        "ABSOLUTE METRICS": [
            "rmse_iv", "mae_iv", "max_violation",
            "rmse_price", "mae_price",
        ],
        "SPREAD & ARBITRAGE": [
            "ba_capture", "butterfly_ok", "calendar_ok", "rank",
        ],
    }

    for group_title, cols in groups.items():
        available = [c for c in cols if c in summary.columns]
        if not available:
            continue

        print("\n" + "=" * 70)
        print(f"  {group_title}")
        print("=" * 70)

        sub = summary[available].copy()
        for col in available:
            if col in fmt:
                f, _ = fmt[col]
                sub[col] = sub[col].apply(
                    lambda x: f.format(x) if pd.notna(x) else "n/a"
                )
        print(sub.to_string())

    print("\n" + "─" * 70)
    print("Notes:")
    print("  All metrics lower-is-better except ba_capture (higher=better).")
    print("  Relative metrics: error as % of observed level at each strike.")
    print("  rrmse_iv / rmae_iv : % of observed IV — key metric for BTC wings.")
    print("  rrmse_price        : % of observed price — unstable for deep OTM.")
    print("  butterfly_ok / calendar_ok : 1=clean, 0=arbitrage violation exists.")
    print("  Ranking based on rrmse_iv (relative IV RMSE).")

    if show_slices:
        print("\n" + "=" * 70)
        print("  PER-SLICE BREAKDOWN")
        print("=" * 70)
        slice_cols = ["objective", "t", "n_strikes",
                      "rmse_iv", "rrmse_iv", "rmae_iv",
                      "rrmse_price", "ba_capture",
                      "butterfly_ok", "calendar_ok"]
        cols = [c for c in slice_cols if c in results.columns]
        print(results[cols].to_string(index=False))


def plot_benchmark(results: pd.DataFrame):
    """
    Plot per-slice rmse_iv for each objective across expiries.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    objectives = results["objective"].unique()
    expiries   = sorted(results["t"].unique())

    # ── Left: RMSE IV per slice ──────────────────────────────────────────
    ax = axes[0]
    for obj in objectives:
        sub = results[results["objective"] == obj].sort_values("t")
        ax.plot(sub["t"], sub["rmse_iv"] * 100, marker="o", label=obj)
    ax.set_xlabel("Time to expiry (years)")
    ax.set_ylabel("RMSE implied vol (vol points)")
    ax.set_title("IV fit quality by expiry")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Right: bid-ask capture per slice (if available) ──────────────────
    ax = axes[1]
    if results["ba_capture"].notna().any():
        for obj in objectives:
            sub = results[results["objective"] == obj].sort_values("t")
            ax.plot(sub["t"], sub["ba_capture"] * 100, marker="o", label=obj)
        ax.set_ylabel("Bid-ask capture (%)")
        ax.set_title("Fraction of strikes within bid-ask spread")
        ax.axhline(100, color="k", linestyle="--", alpha=0.3, label="perfect")
    else:
        for obj in objectives:
            sub = results[results["objective"] == obj].sort_values("t")
            ax.plot(sub["t"], sub["rmse_price"] * 1e4, marker="o", label=obj)
        ax.set_ylabel("RMSE price (bps of forward)")
        ax.set_title("Price fit quality by expiry")

    ax.set_xlabel("Time to expiry (years)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SELF TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("vol_calibration_benchmark — self test")
    print("=" * 65)

    rng = np.random.default_rng(42)

    # Synthetic snapshot: 4 expiries, 12 strikes each
    true_spec = SVI_ModelSpec()
    expiries  = [0.083, 0.25, 0.50, 1.0]      # 1m, 3m, 6m, 1y
    rows = []

    true_params = [
        {"a": 0.04,  "b": 0.20, "rho": -0.70, "m": 0.0,  "sig": 0.20},
        {"a": 0.06,  "b": 0.18, "rho": -0.65, "m": 0.02, "sig": 0.22},
        {"a": 0.08,  "b": 0.15, "rho": -0.60, "m": 0.03, "sig": 0.25},
        {"a": 0.10,  "b": 0.12, "rho": -0.55, "m": 0.04, "sig": 0.28},
    ]

    for t, tp in zip(expiries, true_params):
        k      = np.linspace(-0.4, 0.4, 12)
        iv_mid = true_spec.iv_from_params(k, t, tp)
        noise  = rng.normal(0, 0.003, len(k))
        spread = np.maximum(0.03 - 0.015 * np.exp(-k**2 / 0.05), 0.008)

        for ki, iv_m, sp, nz in zip(k, iv_mid, spread, noise):
            rows.append({
                "t":      t,
                "k":      ki,
                "w":      (iv_m + nz) ** 2 * t,
                "bid_iv": iv_m - sp / 2,
                "ask_iv": iv_m + sp / 2,
            })

    df = pd.DataFrame(rows)

    # Run benchmark on a subset of objectives for speed
    objectives = ["sse_total_var", "vega_weighted_iv",
                  "bid_ask_normalized", "combined"]

    results = benchmark_objectives(
        df,
        model_spec  = SVI_ModelSpec(),
        objectives  = objectives,
        use_global_init = True,
        verbose     = True,
    )

    print_benchmark(results, show_slices=True)
