"""
Benchmark a plethora of objective functions on **global eSSVI** across all
ETH snapshots.

Objectives swept:
  1)  iv_mse                          — unweighted IV MSE (control)
  2)  iv_wmse                         — spread-only weighting (1/spread²)
  3)  vega_wmse                       — vega² / spread² (FactSet form)
  4)  iv_zweighted(β=0)               — vega / spread² (linear-vega)
  5)  iv_zweighted(β=0.2)             — mild wing emphasis
  6)  iv_zweighted(β=0.4)             — older empirical optimum
  7)  iv_uniform_z                    — exact uniform-in-z
  8)  iv_convex_blend(κ=0.5)          — convex blend baseline
  9)  vega_wmse_band(λ=10)            — vega WMSE + bid/ask band penalty
 10)  band(vega_weight=True, mid=1e-4) — trust-the-book extreme

Per slice we record: spread_hit_rate, vwrmse, iv_rmse_atm.

Outputs:
  - benchmark_objectives_essvi.pkl
  - benchmark_objectives_essvi.png

Run:
    python volatility_surface/benchmark_objectives.py [--max-snapshots N] [--quick]
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volatility_surface.core.calibration.calibrator import (
    calibrate_global_essvi, load_snapshot,
)
from volatility_surface.core.calibration.objectives import (
    obj_iv_mse, obj_iv_wmse, obj_iv_vega_wmse, obj_iv_zweighted,
    obj_iv_uniform_z, obj_iv_convex_blend, obj_vega_wmse_band, obj_band_loss,
)
from volatility_surface.models.vol_models import eSSVI


PARQUET = str(PROJECT_ROOT / "2 - Data" / "parquets" / "eth_options_data_cleaned.parquet")
METRICS = ["spread_hit_rate", "vwrmse", "iv_rmse_atm"]


def named(fn, name):
    fn.__name__ = name
    return fn


def obj_iv_wmse_ba(w_fit, w_obs, iv_fit, iv_obs, k, t,
                   spreads=None, vegas=None, bid=None, ask=None):
    """iv_wmse wrapper that also accepts bid/ask — needed so _bind_extra
    can inject the per-slice spread for global fits."""
    if spreads is None and bid is not None and ask is not None:
        spreads = np.abs(np.asarray(ask, dtype=float)
                         - np.asarray(bid, dtype=float))
    return obj_iv_wmse(w_fit, w_obs, iv_fit, iv_obs, k, t, spreads=spreads)


OBJECTIVES = {
    "iv_mse":                obj_iv_mse,
    "iv_wmse":               named(obj_iv_wmse_ba, "iv_wmse"),
    "vega_wmse":             obj_iv_vega_wmse,
    "iv_zweighted_b0":       named(partial(obj_iv_zweighted,    beta=0.0),   "iv_zweighted_b0"),
    "iv_zweighted_b0.2":     named(partial(obj_iv_zweighted,    beta=0.2),   "iv_zweighted_b0.2"),
    "iv_zweighted_b0.4":     named(partial(obj_iv_zweighted,    beta=0.4),   "iv_zweighted_b0.4"),
    "iv_uniform_z":          obj_iv_uniform_z,
    "iv_convex_blend_k0.5":  named(partial(obj_iv_convex_blend, kappa=0.5),  "iv_convex_blend_k0.5"),
    "vega_wmse_band_l10":    named(partial(obj_vega_wmse_band,  band_weight=10.0),                 "vega_wmse_band_l10"),
    "band_vega_midanchor":   named(partial(obj_band_loss,       vega_weight=True, mid_anchor=1e-4), "band_vega_midanchor"),
}


# Discrete purple palette (10 distinguishable shades).
PALETTE = [
    "#3D0A66", "#5C0E99", "#7B14CC", "#9A1AFF",
    "#B14CFF", "#C27FFF", "#D2A6FF", "#E0C2FF",
    "#A6F", "#EBD9F5",
]
# Map names to colours so the legend stays consistent across panels.
COLOR_MAP = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(OBJECTIVES)}


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(parquet_path: str, max_snapshots: int | None = None,
                  quick: bool = False, random_n: int | None = None,
                  seed: int = 42) -> dict:
    df_all = pd.read_parquet(parquet_path)
    timestamps = sorted(df_all["file_timestamp"].unique())
    if random_n is not None:
        rng = np.random.default_rng(seed)
        n = min(random_n, len(timestamps))
        timestamps = sorted(rng.choice(timestamps, size=n, replace=False).tolist())
        print(f"Random sample of {n} snapshots (seed={seed}).")
    elif max_snapshots:
        timestamps = timestamps[:max_snapshots]
    print(f"Running on {len(timestamps)} snapshot(s) × {len(OBJECTIVES)} objectives "
          f"= {len(timestamps) * len(OBJECTIVES)} eSSVI fits.\n")

    per_slice = {name: {m: [] for m in METRICS} for name in OBJECTIVES}
    per_snapshot = {name: {m: [] for m in METRICS} for name in OBJECTIVES}
    snapshot_labels: list[str] = []

    t0 = time.time()
    for s_i, ts in enumerate(timestamps):
        print(f"────────── snapshot {s_i+1}/{len(timestamps)}  {ts} ──────────")
        try:
            df = load_snapshot(parquet_path, timestamp=ts)
        except Exception as e:
            print(f"  load_snapshot failed: {e}")
            continue

        if quick:
            expiries = sorted(df["t"].unique())[:4]
            df = df[df["t"].isin(expiries)].reset_index(drop=True)

        snapshot_labels.append(str(ts))

        for name, objective in OBJECTIVES.items():
            t_start = time.time()
            try:
                res = calibrate_global_essvi(
                    df,
                    model=eSSVI(),
                    objective=objective,
                    vega_col="vega",
                    bid_col="bid_iv",
                    ask_col="ask_iv",
                    verbose=False,
                )
            except Exception as e:
                print(f"  {name:24s} FAILED: {e}")
                for m in METRICS:
                    per_snapshot[name][m].append(np.nan)
                continue

            elapsed = time.time() - t_start
            metric_means = {}
            for m in METRICS:
                vals = [v for v in res["metrics"].get(m, [])
                        if v is not None and not np.isnan(v)]
                per_slice[name][m].extend(vals)
                mean_v = float(np.nanmean(res["metrics"][m])) if vals else np.nan
                per_snapshot[name][m].append(mean_v)
                metric_means[m] = mean_v

            print(f"  {name:24s} {elapsed:5.1f}s   "
                  f"hit={metric_means['spread_hit_rate']:.3f}  "
                  f"vwrmse={metric_means['vwrmse']:.4f}  "
                  f"atm={metric_means['iv_rmse_atm']:.4f}")
        print()

    print(f"Total wall time: {time.time() - t0:.1f}s")
    return {
        "per_slice":    per_slice,
        "per_snapshot": per_snapshot,
        "snapshots":    snapshot_labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3, color="#B284E0")
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
    ax.tick_params(colors="#444444")


def plot_results(bench: dict, savepath: str | None = None) -> plt.Figure:
    names = list(OBJECTIVES.keys())
    per_slice    = bench["per_slice"]
    per_snapshot = bench["per_snapshot"]
    snapshots    = bench["snapshots"]
    colors = [COLOR_MAP[n] for n in names]

    fig = plt.figure(figsize=(18, 11), facecolor="white")
    gs  = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.0, 1.1],
                            hspace=0.55, wspace=0.32)

    # Row 1 — boxplot per metric across all slices
    for col, metric in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, col])
        data = [np.array(per_slice[n][metric]) for n in names]
        bp = ax.boxplot(data, labels=names, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="black",
                                       markeredgecolor="black", markersize=4),
                        flierprops=dict(marker=".", markersize=4,
                                        markeredgecolor="#888"))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        for median in bp["medians"]:
            median.set_color("black")
        ax.set_title(f"per-slice {metric}", color="#6A0DAD", fontsize=12)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7.5)
        _style_ax(ax)

    # Row 2 — per-snapshot mean line plot for each metric
    for col, metric in enumerate(METRICS):
        ax = fig.add_subplot(gs[1, col])
        x = np.arange(len(snapshots))
        for n, c in zip(names, colors):
            ax.plot(x, per_snapshot[n][metric], marker="o", lw=1.2,
                    color=c, label=n, alpha=0.85, markersize=4)
        ax.set_title(f"mean {metric} per snapshot",
                     color="#6A0DAD", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([s.split(" ")[0] for s in snapshots],
                           rotation=45, ha="right", fontsize=6.5)
        _style_ax(ax)

    # Row 3 col 0 — Pareto scatter: hit_rate vs vwrmse, one star per objective
    ax = fig.add_subplot(gs[2, 0])
    for n, c in zip(names, colors):
        xs = per_snapshot[n]["vwrmse"]
        ys = per_snapshot[n]["spread_hit_rate"]
        ax.scatter(xs, ys, color=c, alpha=0.35, s=22, edgecolor="white")
        ax.scatter(np.nanmean(xs), np.nanmean(ys), color=c, s=180,
                   marker="*", edgecolor="black", zorder=5, label=n)
    ax.set_xlabel("mean vwrmse  (lower = better)")
    ax.set_ylabel("mean spread_hit_rate  (higher = better)")
    ax.set_title("Pareto: hit rate vs vwrmse\n(★ = objective mean across snapshots)",
                 color="#6A0DAD", fontsize=11)
    _style_ax(ax)

    # Row 3 col 1 — same Pareto but iv_rmse_atm vs spread_hit_rate
    ax = fig.add_subplot(gs[2, 1])
    for n, c in zip(names, colors):
        xs = per_snapshot[n]["iv_rmse_atm"]
        ys = per_snapshot[n]["spread_hit_rate"]
        ax.scatter(xs, ys, color=c, alpha=0.35, s=22, edgecolor="white")
        ax.scatter(np.nanmean(xs), np.nanmean(ys), color=c, s=180,
                   marker="*", edgecolor="black", zorder=5, label=n)
    ax.set_xlabel("mean iv_rmse_atm  (lower = better)")
    ax.set_ylabel("mean spread_hit_rate  (higher = better)")
    ax.set_title("Pareto: hit rate vs ATM RMSE",
                 color="#6A0DAD", fontsize=11)
    _style_ax(ax)

    # Row 3 col 2 — legend-only panel so labels are readable
    ax_leg = fig.add_subplot(gs[2, 2])
    ax_leg.axis("off")
    handles = [plt.Line2D([0], [0], marker="*", linestyle="none",
                          color=COLOR_MAP[n], markeredgecolor="black",
                          markersize=10, label=n)
               for n in names]
    ax_leg.legend(handles=handles, title="Objective", loc="center",
                  frameon=False, fontsize=9, title_fontsize=10)

    fig.suptitle("Objective benchmark on ETH snapshots — global eSSVI",
                 color="#6A0DAD", fontsize=14, y=0.995)

    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches="tight", facecolor="white")
        print(f"Saved figure → {savepath}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def summary_table(bench: dict) -> pd.DataFrame:
    rows = []
    for name in OBJECTIVES:
        row = {"objective": name}
        for m in METRICS:
            arr = np.asarray(bench["per_slice"][name][m], dtype=float)
            row[f"{m}_mean"]   = float(np.nanmean(arr)) if arr.size else np.nan
            row[f"{m}_median"] = float(np.nanmedian(arr)) if arr.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-snapshots", type=int, default=None)
    ap.add_argument("--random", type=int, default=None,
                    help="pick N random snapshots instead of the first --max-snapshots")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="restrict each snapshot to its first 4 expiries")
    ap.add_argument("--out", default="benchmark_objectives_essvi")
    args = ap.parse_args()

    bench = run_benchmark(PARQUET, args.max_snapshots, args.quick,
                          random_n=args.random, seed=args.seed)

    summary = summary_table(bench)
    print("\n=== Summary (means / medians across all slices) ===")
    print(summary.to_string(index=False, float_format="%.4f"))

    out_root = Path(args.out)
    with open(out_root.with_suffix(".pkl"), "wb") as f:
        pickle.dump({"bench": bench, "summary": summary}, f)
    print(f"\nSaved data   → {out_root.with_suffix('.pkl')}")

    plot_results(bench, savepath=str(out_root.with_suffix(".png")))


if __name__ == "__main__":
    main()
