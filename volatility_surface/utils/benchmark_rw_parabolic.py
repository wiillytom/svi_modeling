"""
Benchmark the Reiswich-Wystup simplified parabolic smile against the
existing per-slice / global models on the ETH archive parquet.

Models compared per maturity slice:
  - RW-paper     : 3-point extraction (Reiswich-Wystup, "paper" recipe)
  - RW-LS        : same 3-parameter form, full-chain LS fit
  - RawSVI       : 5-param per-slice SVI, vega-weighted MSE
  - eSSVI        : 3 global + 3 per-slice, vega-weighted MSE
  - SABR         : 3 per-slice (α, ρ, ν), vega-weighted MSE

Per-slice metrics: iv_rmse, vwrmse, spread_hit_rate, iv_rmse_atm.
Outputs:
  - benchmark_rw.pkl   (raw per-slice metrics)
  - benchmark_rw.png   (boxplot + a representative smile overlay)

Usage:
  python volatility_surface/benchmark_rw_parabolic.py [--max-snapshots N]
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
    calibrate_global_essvi, calibrate_snapshot, load_snapshot,
)
from volatility_surface.core.calibration.objectives import evaluate_all
from volatility_surface.models.vol_models import RawSVI, SABR, eSSVI
from volatility_surface.models.rw_parabolic import (
    fit_paper_style, fit_least_squares, sigma_of_K,
)


PARQUET = str(PROJECT_ROOT / "2 - Data" / "parquets" / "eth_options_data_cleaned.parquet")
METRICS = ["iv_rmse", "vwrmse", "spread_hit_rate", "iv_rmse_atm"]
MODELS  = ["RW-paper", "RW-LS", "RawSVI", "eSSVI", "SABR"]
PALETTE = ["#9A6BB5", "#6A0DAD", "#3D8BBA", "#1F77B4", "#D45E5E"]


# ─────────────────────────────────────────────────────────────────────────────
# Per-slice scoring under each model
# ─────────────────────────────────────────────────────────────────────────────

def _slice_metrics(iv_fit: np.ndarray, iv_obs: np.ndarray,
                   k_obs: np.ndarray, t: float,
                   bid: np.ndarray, ask: np.ndarray) -> dict:
    w_fit = iv_fit ** 2 * t
    w_obs = iv_obs ** 2 * t
    return evaluate_all(w_fit, w_obs, iv_fit, iv_obs, k_obs, t,
                        bid=bid, ask=ask, metric_names=METRICS)


def score_rw(p, k_obs, iv_obs, bid_iv, ask_iv, t, F):
    K = F * np.exp(k_obs)
    iv_fit = sigma_of_K(K, F, t, p.sigma_atm, p.sigma_rr, p.sigma_s)
    iv_fit = np.atleast_1d(iv_fit)
    return _slice_metrics(iv_fit, iv_obs, k_obs, t, bid_iv, ask_iv)


def score_global_result(res: dict, model, df_otm: pd.DataFrame) -> dict[float, dict]:
    """Map expiry -> per-slice metrics for an eSSVI / RawSVI result."""
    out = {}
    for t, params in zip(res["expiries"], res["params"]):
        sub = df_otm[np.isclose(df_otm["t"], t)]
        if sub.empty:
            continue
        k = sub["k"].values
        iv_obs = sub["mark_iv"].values
        iv_fit = model.iv(k, params, t)
        out[t] = _slice_metrics(iv_fit, iv_obs, k,
                                t=t,
                                bid=sub["bid_iv"].values,
                                ask=sub["ask_iv"].values)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-snapshot driver
# ─────────────────────────────────────────────────────────────────────────────

def run_snapshot(df: pd.DataFrame) -> dict:
    """Fit every model on the OTM-filtered snapshot and return per-slice scores
    for each model, plus enough state to plot a representative smile."""
    otm_C = (df["option_type"] == "C") & (df["k"] >= 0)
    otm_P = (df["option_type"] == "P") & (df["k"] <= 0)
    df_otm = df[otm_C | otm_P].copy()

    out: dict = {m: {} for m in MODELS}

    # ── Global models (eSSVI + RawSVI per-slice + SABR per-slice) ──────────
    t0 = time.perf_counter()
    res_essvi = calibrate_global_essvi(
        df_otm, model=eSSVI(), objective="vega_wmse",
        vega_col="vega", bid_col="bid_iv", ask_col="ask_iv", verbose=False,
    )
    t_essvi = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_svi = calibrate_snapshot(
        df_otm, model=RawSVI(), objective="vega_wmse",
        vega_col="vega", bid_col="bid_iv", ask_col="ask_iv", verbose=False,
    )
    t_svi = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_sabr = calibrate_snapshot(
        df_otm, model=SABR(), objective="vega_wmse",
        vega_col="vega", bid_col="bid_iv", ask_col="ask_iv", verbose=False,
    )
    t_sabr = time.perf_counter() - t0

    out["eSSVI"]  = score_global_result(res_essvi, res_essvi["_model"], df_otm)
    out["RawSVI"] = score_global_result(res_svi,   res_svi["_model"],   df_otm)
    out["SABR"]   = score_global_result(res_sabr,  res_sabr["_model"],  df_otm)

    # ── Reiswich-Wystup per-slice ──────────────────────────────────────────
    expiries = sorted(df_otm["t"].unique())
    for t in expiries:
        sub = df_otm[np.isclose(df_otm["t"], t)].sort_values("k")
        if len(sub) < 5:
            continue
        F = float(sub["underlying_price"].iloc[0])
        k_obs  = sub["k"].values
        iv_obs = sub["mark_iv"].values
        bid_iv = sub["bid_iv"].values
        ask_iv = sub["ask_iv"].values

        try:
            p_paper = fit_paper_style(k_obs, iv_obs, t, F)
            out["RW-paper"][t] = score_rw(p_paper, k_obs, iv_obs, bid_iv, ask_iv, t, F)
        except Exception as e:
            print(f"  RW-paper failed at T={t:.4f}: {e!r}")

        try:
            p_ls = fit_least_squares(k_obs, iv_obs, t, F)
            out["RW-LS"][t] = score_rw(p_ls, k_obs, iv_obs, bid_iv, ask_iv, t, F)
        except Exception as e:
            print(f"  RW-LS failed at T={t:.4f}: {e!r}")

    return {
        "scores":   out,
        "timings":  {"eSSVI": t_essvi, "RawSVI": t_svi, "SABR": t_sabr},
        "df_otm":   df_otm,
        "res_essvi": res_essvi,
        "res_svi":   res_svi,
        "res_sabr":  res_sabr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate + plots
# ─────────────────────────────────────────────────────────────────────────────

def collect(all_runs: list[dict]) -> dict[str, dict[str, list[float]]]:
    """Flatten per-snapshot results into {model: {metric: [...slice values...]}}."""
    out = {m: {k: [] for k in METRICS} for m in MODELS}
    for run in all_runs:
        for m in MODELS:
            for t, vals in run["scores"][m].items():
                for k in METRICS:
                    out[m][k].append(vals.get(k, np.nan))
    return out


def plot_results(agg, snap_for_demo: dict, savepath: str) -> plt.Figure:
    fig = plt.figure(figsize=(18, 10), facecolor="white")
    gs  = fig.add_gridspec(2, 4, height_ratios=[1, 1.1], hspace=0.42, wspace=0.32)

    # Row 1 — boxplots per metric
    for col, metric in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, col])
        data = [np.array(agg[m][metric], dtype=float) for m in MODELS]
        bp = ax.boxplot(data, labels=MODELS, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="black",
                                       markeredgecolor="black", markersize=4),
                        flierprops=dict(marker=".", markersize=3, markeredgecolor="#888"))
        for patch, c in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for m in bp["medians"]:
            m.set_color("black")
        ax.set_title(metric, fontsize=12, color="#6A0DAD")
        ax.set_xticklabels(MODELS, rotation=25, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, color="#B284E0")
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_color("#cccccc")

    # Row 2 — one representative smile, eSSVI vs RW-paper vs RW-LS
    df_otm = snap_for_demo["df_otm"]
    expiries = sorted(df_otm["t"].unique())
    t_demo = expiries[len(expiries) // 2]
    sub = df_otm[np.isclose(df_otm["t"], t_demo)].sort_values("k")
    F = float(sub["underlying_price"].iloc[0])
    k = sub["k"].values
    iv = sub["mark_iv"].values

    p_paper = fit_paper_style(k, iv, t_demo, F)
    p_ls    = fit_least_squares(k, iv, t_demo, F)
    K_grid  = F * np.exp(np.linspace(k.min(), k.max(), 200))
    iv_paper = sigma_of_K(K_grid, F, t_demo, p_paper.sigma_atm, p_paper.sigma_rr, p_paper.sigma_s)
    iv_ls    = sigma_of_K(K_grid, F, t_demo, p_ls.sigma_atm,    p_ls.sigma_rr,    p_ls.sigma_s)

    res_e = snap_for_demo["res_essvi"]
    i_e   = int(np.argmin(np.abs(np.asarray(res_e["expiries"]) - t_demo)))
    iv_e  = res_e["_model"].iv(np.log(K_grid / F), res_e["params"][i_e], t_demo)

    ax = fig.add_subplot(gs[1, :])
    ax.scatter(k, iv, s=14, color="#444", label="market", zorder=4)
    ax.plot(np.log(K_grid / F), iv_e,    "-",  color="#1F77B4", lw=2.2, label="eSSVI fit")
    ax.plot(np.log(K_grid / F), iv_paper, "--", color="#9A6BB5", lw=2.2, label="RW paper-style")
    ax.plot(np.log(K_grid / F), iv_ls,    "-",  color="#6A0DAD", lw=2.2, label="RW LS-refined")
    ax.set_xlabel("log-moneyness  k = ln(K/F)")
    ax.set_ylabel("IV")
    ax.set_title(
        f"Representative smile  T = {t_demo:.4f}y ({t_demo*365:.0f}d)   "
        f"snapshot {snap_for_demo.get('ts','?')}",
        fontsize=12, color="#6A0DAD",
    )
    ax.grid(True, alpha=0.3, color="#B284E0")
    ax.set_facecolor("white")
    ax.legend(fontsize=9, frameon=False)
    for sp in ax.spines.values():
        sp.set_color("#cccccc")

    fig.suptitle("Reiswich-Wystup parabolic vs eSSVI / RawSVI / SABR (ETH)",
                 color="#6A0DAD", fontsize=14, y=0.995)
    fig.savefig(savepath, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Saved figure → {savepath}")
    return fig


def summary_table(agg) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        row = {"model": m}
        for k in METRICS:
            arr = np.asarray(agg[m][k], dtype=float)
            row[f"{k}_mean"]   = float(np.nanmean(arr)) if arr.size else np.nan
            row[f"{k}_median"] = float(np.nanmedian(arr)) if arr.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-snapshots", type=int, default=None)
    ap.add_argument("--out", default="benchmark_rw")
    args = ap.parse_args()

    df_all = pd.read_parquet(PARQUET)
    timestamps = sorted(df_all["file_timestamp"].unique())
    if args.max_snapshots:
        timestamps = timestamps[:args.max_snapshots]
    print(f"Running on {len(timestamps)} snapshot(s).")

    all_runs = []
    t_global = time.time()
    for i, ts in enumerate(timestamps):
        print(f"\n──────── snapshot {i+1}/{len(timestamps)}   {ts} ────────")
        df = load_snapshot(PARQUET, timestamp=ts)
        try:
            run = run_snapshot(df)
            run["ts"] = ts
            all_runs.append(run)
            print(f"  timings — eSSVI {run['timings']['eSSVI']:.1f}s  "
                  f"RawSVI {run['timings']['RawSVI']:.1f}s  "
                  f"SABR {run['timings']['SABR']:.1f}s")
        except Exception as e:
            print(f"  snapshot failed: {e!r}")
    print(f"\nTotal wall time: {time.time() - t_global:.1f}s")

    agg = collect(all_runs)
    summary = summary_table(agg)
    print("\n=== Summary (across all slices, ETH archive) ===")
    print(summary.to_string(index=False, float_format="%.4f"))

    out = Path(args.out)
    with open(out.with_suffix(".pkl"), "wb") as f:
        pickle.dump({"agg": agg, "summary": summary}, f)
    print(f"Saved data → {out.with_suffix('.pkl')}")

    if all_runs:
        plot_results(agg, all_runs[0], savepath=str(out.with_suffix(".png")))


if __name__ == "__main__":
    main()
