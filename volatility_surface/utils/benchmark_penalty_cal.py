"""
Sweep the calendar-arb penalty (`penalty_cal`) for per-slice Raw SVI and SABR
calibrations across several snapshots.

For each (model, snapshot, penalty) we record:
  fit quality   : vwrmse, iv_rrmse, spread_hit_rate
  calendar arb  : worst crossedness across adjacent pairs,
                  integrated crossedness (sum),
                  # of adjacent pairs with crossedness > 1e-6
  timing        : wall-clock seconds

Outputs:
  benchmark_penalty_cal.pkl  raw per-run rows + summary DataFrame
  benchmark_penalty_cal.png  fit-vs-penalty, violations-vs-penalty, and a
                             Pareto scatter per model

Suggested defaults sweep penalty across four decades (10 → 1e5).  With the
built-in ETH archive this is ~1 h wall time; use `--n-snapshots 1` for a
smoke test that finishes in a couple of minutes.

Usage:
    python volatility_surface/benchmark_penalty_cal.py
    python volatility_surface/benchmark_penalty_cal.py --n-snapshots 1
    python volatility_surface/benchmark_penalty_cal.py \\
        --penalties 100 500 5000 50000 --n-snapshots 2
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volatility_surface.core.calibration.calibrator import (
    calibrate_snapshot, crossedness, load_snapshot,
)
from volatility_surface.models.vol_models import RawSVI, SABR


PARQUET   = str(PROJECT_ROOT / "2 - Data" / "parquets" / "eth_options_data_cleaned.parquet")
ARB_TOL   = 1e-6                                            # match live-chain banner threshold
MODELS    = {"RawSVI": RawSVI, "SABR": SABR}
COLORS    = {"RawSVI": "#6A0DAD", "SABR": "#D45E5E"}


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def calendar_stats(res: dict, tol: float = ARB_TOL) -> tuple[float, float, int]:
    """(worst crossedness, integrated crossedness, # of violated adjacent pairs)."""
    model    = res["_model"]
    order    = np.argsort(res["expiries"])
    worst    = 0.0
    total    = 0.0
    n_viol   = 0
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        c = crossedness(model, res["params"][a], res["params"][b])
        worst  = max(worst, c)
        total += c
        n_viol += int(c > tol)
    return worst, total, n_viol


def fit_metrics(res: dict) -> dict:
    m = res["metrics"]
    return {
        "vwrmse":          float(np.nanmean(m.get("vwrmse",          [np.nan]))),
        "iv_rrmse":        float(np.nanmean(m.get("iv_rrmse",        [np.nan]))),
        "spread_hit_rate": float(np.nanmean(m.get("spread_hit_rate", [np.nan]))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# The sweep
# ─────────────────────────────────────────────────────────────────────────────

def load_otm(ts: str) -> pd.DataFrame:
    df = load_snapshot(PARQUET, timestamp=ts)
    otm = ((df["option_type"] == "C") & (df["k"] >= 0)) | \
          ((df["option_type"] == "P") & (df["k"] <= 0))
    return df[otm].copy()


def run_one(model_name: str, ctor, snap_ts: str, df_otm: pd.DataFrame,
            penalty: float) -> dict:
    """One calibration.  Returns a row for the results frame."""
    t0 = time.perf_counter()
    res = calibrate_snapshot(
        df_otm,
        model=ctor(),
        objective="vega_wmse",
        vega_col="vega", bid_col="bid_iv", ask_col="ask_iv",
        penalty_cal=penalty,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    fm = fit_metrics(res)
    worst, total, n_viol = calendar_stats(res)
    return {
        "model":         model_name,
        "snapshot":      snap_ts,
        "penalty":       penalty,
        "elapsed_s":     elapsed,
        "worst_cross":   worst,
        "total_cross":   total,
        "n_violations":  n_viol,
        **fm,
    }


def run_sweep(penalties: list[float], snap_ts_list: list[str]) -> pd.DataFrame:
    n_runs = len(penalties) * len(snap_ts_list) * len(MODELS)
    print(f"Running {n_runs} calibrations "
          f"({len(MODELS)} models × {len(penalties)} penalties × {len(snap_ts_list)} snapshots)\n")

    rows: list[dict] = []
    counter = 0
    for ts in snap_ts_list:
        df_otm = load_otm(ts)
        for model_name, ctor in MODELS.items():
            for pen in penalties:
                counter += 1
                row = run_one(model_name, ctor, ts, df_otm, pen)
                rows.append(row)
                print(f"  [{counter:>3d}/{n_runs}]  {ts}  {model_name:6s}  "
                      f"pen={pen:>8.0f}   {row['elapsed_s']:>5.1f}s   "
                      f"vwrmse={row['vwrmse']:.4f}   "
                      f"worst_cross={row['worst_cross']:.2e}   "
                      f"n_viol={row['n_violations']}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Summary + plots + recommendation
# ─────────────────────────────────────────────────────────────────────────────

def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = (df.groupby(["model", "penalty"])
             .agg(vwrmse_mean          = ("vwrmse",          "mean"),
                  iv_rrmse_mean        = ("iv_rrmse",        "mean"),
                  hit_rate_mean        = ("spread_hit_rate", "mean"),
                  worst_cross_mean     = ("worst_cross",     "mean"),
                  worst_cross_max      = ("worst_cross",     "max"),
                  total_cross_mean     = ("total_cross",     "mean"),
                  n_violations_mean    = ("n_violations",    "mean"),
                  elapsed_s_mean       = ("elapsed_s",       "mean"))
             .reset_index()
             .sort_values(["model", "penalty"]))
    return agg


def recommend(summary: pd.DataFrame, tol: float = ARB_TOL) -> pd.DataFrame:
    """Pick the smallest penalty per model that keeps the mean worst_cross under
    `tol`; if none do, return the penalty with the smallest violation."""
    picks = []
    for model, sub in summary.groupby("model"):
        sub = sub.sort_values("penalty")
        arb_free = sub[sub["worst_cross_mean"] <= tol]
        if not arb_free.empty:
            best = arb_free.iloc[0]                # smallest penalty that works
        else:
            best = sub.sort_values("worst_cross_mean").iloc[0]  # least bad
        picks.append({
            "model":              model,
            "recommended_penalty": float(best["penalty"]),
            "worst_cross":         float(best["worst_cross_mean"]),
            "vwrmse":              float(best["vwrmse_mean"]),
            "hit_rate":            float(best["hit_rate_mean"]),
            "meets_arb_tol":       best["worst_cross_mean"] <= tol,
        })
    return pd.DataFrame(picks)


def _style_ax(ax):
    ax.grid(True, alpha=0.3, color="#B284E0")
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_color("#cccccc")


def plot_sweep(df: pd.DataFrame, summary: pd.DataFrame, savepath: str) -> plt.Figure:
    fig = plt.figure(figsize=(16, 4.5 * len(MODELS)), facecolor="white")
    gs  = fig.add_gridspec(len(MODELS), 3, hspace=0.42, wspace=0.32)

    for r, model in enumerate(MODELS):
        agg = summary[summary["model"] == model].sort_values("penalty")
        color = COLORS[model]

        # Column 0 — fit error vs penalty
        ax = fig.add_subplot(gs[r, 0])
        ax.semilogx(agg["penalty"], agg["vwrmse_mean"], "o-", color=color, lw=2, ms=6)
        ax.set(xlabel="penalty_cal (log scale)", ylabel="mean vwrmse",
               title=f"{model}  —  fit error vs penalty")
        _style_ax(ax)

        # Column 1 — worst calendar violation vs penalty
        ax = fig.add_subplot(gs[r, 1])
        ax.loglog(agg["penalty"], np.maximum(agg["worst_cross_mean"], 1e-12),
                  "o-", color=color, lw=2, ms=6, label="mean over snapshots")
        ax.loglog(agg["penalty"], np.maximum(agg["worst_cross_max"], 1e-12),
                  "s--", color=color, alpha=0.5, ms=6, label="max over snapshots")
        ax.axhline(ARB_TOL, ls=":", c="red", label=f"tol = {ARB_TOL:g}")
        ax.set(xlabel="penalty_cal (log scale)", ylabel="worst crossedness",
               title=f"{model}  —  calendar violation vs penalty")
        ax.legend(fontsize=8)
        _style_ax(ax)

        # Column 2 — Pareto per-run scatter
        ax = fig.add_subplot(gs[r, 2])
        sub = df[df["model"] == model]
        for pen in sorted(sub["penalty"].unique()):
            s = sub[sub["penalty"] == pen]
            ax.scatter(np.maximum(s["worst_cross"], 1e-12), s["vwrmse"],
                       s=42, alpha=0.7,
                       label=f"pen={pen:g}")
        ax.set(xscale="log", xlabel="worst crossedness",
               ylabel="vwrmse",
               title=f"{model}  —  Pareto (each dot = one snapshot)")
        ax.axvline(ARB_TOL, ls=":", c="red")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        _style_ax(ax)

    fig.suptitle("Sweeping penalty_cal on RawSVI and SABR — ETH archive",
                 fontsize=14, color="#6A0DAD", y=0.995)
    fig.savefig(savepath, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"\nSaved figure → {savepath}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-snapshots", type=int, default=3,
                    help="how many ETH snapshots to run on (default 3, ~1 h)")
    ap.add_argument("--penalties", nargs="+", type=float,
                    default=[10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 100000.0],
                    help="penalty_cal grid (default 10 → 1e5)")
    ap.add_argument("--out", default="benchmark_penalty_cal",
                    help="output basename (no extension)")
    args = ap.parse_args()

    df_all = pd.read_parquet(PARQUET)
    snap_ts_list = sorted(df_all["file_timestamp"].unique())[:args.n_snapshots]
    print(f"Snapshots: {snap_ts_list}\n")

    t_start = time.time()
    df = run_sweep(args.penalties, snap_ts_list)
    print(f"\nTotal wall time: {(time.time() - t_start)/60:.1f} min")

    summary = make_summary(df)
    print("\n=== Summary (means across snapshots) ===")
    print(summary.round(4).to_string(index=False))

    rec = recommend(summary)
    print("\n=== Recommended penalty per model ===")
    print(rec.round(6).to_string(index=False))

    out = Path(args.out)
    with open(out.with_suffix(".pkl"), "wb") as f:
        pickle.dump({"df": df, "summary": summary, "recommendation": rec}, f)
    print(f"Saved data → {out.with_suffix('.pkl')}")

    plot_sweep(df, summary, str(out.with_suffix(".png")))


if __name__ == "__main__":
    main()
