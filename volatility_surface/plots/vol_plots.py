"""
vol_plots.py
============
Model-agnostic plotting functions for a calibrated volatility surface.

All plots consume the standardised result dict from calibrator.py and
the VolModel instance stored inside it. No model-specific code here.

Functions
---------
    plot_surface(result)            3D implied vol surface
    plot_slices(result, df)         Per-slice smile + RND subplots
    plot_total_variance(result)     Gatheral-style total variance overlay
    plot_metrics(result)            Bar chart of per-slice fit metrics
    plot_compare(results, df)       Overlay multiple model fits on one slice
    plot_all(result, df)            Generate all plots at once
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from scipy.interpolate import PchipInterpolator

from volatility_surface.models.vol_models import VolModel, get_model


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_model(result: dict) -> VolModel:
    return result.get("_model") or get_model("svi")


def _k_range(df_slice, global_range=(-2.0, 2.0), padding=0.25):
    if df_slice is not None and len(df_slice) > 0:
        lo = max(df_slice["k"].min() - padding, global_range[0])
        hi = min(df_slice["k"].max() + padding, global_range[1])
    else:
        lo, hi = global_range
    return lo, hi


def _scatter_market(ax, df_slice, t):
    """Overlay market bid/ask or mid observations."""
    if df_slice is None or len(df_slice) == 0:
        return
    if "bid_iv" in df_slice.columns and "ask_iv" in df_slice.columns:
        ax.scatter(df_slice["k"], df_slice["bid_iv"] * 100,
                   color="#e05252", s=12, alpha=0.7, marker="x",
                   label="Bid", zorder=4)
        ax.scatter(df_slice["k"], df_slice["ask_iv"] * 100,
                   color="#5288e0", s=12, alpha=0.7, marker="x",
                   label="Ask", zorder=4)
    elif "mark_iv" in df_slice.columns:
        ax.scatter(df_slice["k"], df_slice['mark_iv'] * 100,
                   color="black", s=14, alpha=0.6, marker="o",
                   label="Mid", zorder=4)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  3D SURFACE
# ─────────────────────────────────────────────────────────────────────────────

def plot_surface(result: dict, k_range=(-2.0, 2.0), n_k=200, n_t=100,
                 colormap="plasma", figsize=(12, 7)) -> plt.Figure:
    """3D implied volatility surface across all fitted expiries."""
    model      = _get_model(result)
    expiries   = np.array(result["expiries"])
    params_list = result["params"]

    k_grid = np.linspace(k_range[0], k_range[1], n_k)
    t_grid = np.linspace(expiries.min(), expiries.max(), n_t)

    iv_surface = np.zeros((n_t, n_k))
    for j, kv in enumerate(k_grid):
        w_at_exp = np.array([model.w(np.array([kv]), p)[0] for p in params_list])
        interp   = PchipInterpolator(expiries, w_at_exp, extrapolate=True)
        w_interp = np.maximum(interp(t_grid), 1e-10)
        iv_surface[:, j] = np.sqrt(w_interp / t_grid)

    T_mesh, K_mesh = np.meshgrid(t_grid, k_grid, indexing="ij")

    fig = plt.figure(figsize=figsize, facecolor="white")
    ax  = fig.add_subplot(111, projection="3d", facecolor="white")

    surf = ax.plot_surface(K_mesh, T_mesh, iv_surface, cmap=colormap,
                           linewidth=0, antialiased=True, alpha=0.92,
                           rcount=n_t, ccount=n_k)

    for t_exp, p in zip(expiries, params_list):
        iv_atm = model.iv(np.array([0.0]), p, t_exp)[0]
        ax.scatter(0.0, t_exp, iv_atm, color="black", s=18, zorder=5)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08)
    cbar.set_label("Implied Vol", color="black", fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

    ax.set_xlabel("Log-strike  k", color="black", labelpad=8)
    ax.set_ylabel("Time to expiry  T", color="black", labelpad=8)
    ax.set_zlabel("Implied Vol  σ", color="black", labelpad=8)
    ax.set_title(f"{result['model_name']} — Implied Volatility Surface",
                 color="black", fontsize=12, pad=14)

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#333333")
    ax.tick_params(colors="black")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PER-SLICE SMILE + RND
# ─────────────────────────────────────────────────────────────────────────────

def plot_slices(result: dict, df=None, n_cols=3,
                k_plot_range=(-2.0, 2.0), n_grid=400,
                figsize=None) -> plt.Figure:
    """One subplot per expiry: fitted smile on left axis, RND on right axis."""
    model      = _get_model(result)
    expiries   = result["expiries"]
    params_list = result["params"]
    n_slices   = len(expiries)

    n_cols = min(n_cols, n_slices)
    n_rows = int(np.ceil(n_slices / n_cols))
    if figsize is None:
        figsize = (5.5 * n_cols, 4.2 * n_rows)

    fig    = plt.figure(figsize=figsize, facecolor="#ffffff")
    colors = [0.994324, 0.716681, 0.177208, 1.      ]#plt.cm.plasma(np.linspace(0.5, 1.25, n_slices))

    for idx, (t_exp, p) in enumerate(zip(expiries, params_list)):
        ax_iv  = fig.add_subplot(n_rows, n_cols, idx + 1)
        ax_rnd = ax_iv.twinx()

        df_slice = df[df["t"] == t_exp] if df is not None else None
        k_lo, k_hi = _k_range(df_slice, global_range=k_plot_range)
        k_grid     = np.linspace(k_lo, k_hi, n_grid)

        # Fitted smile
        iv_fit  = model.iv(k_grid, p, t_exp)
        ax_iv.plot(k_grid, iv_fit * 100, color=colors, lw=2.0,
                   label=f"{model.name} fit", zorder=3)

        # Market observations
        _scatter_market(ax_iv, df_slice, t_exp)

        # RND
        rnd       = model.butterfly_density(k_grid, p)
        has_neg   = model.has_butterfly_arb(p)
        rnd_color = "#ff6b35" if has_neg else "#fbfffb"#"#7ecf7e"
        ax_rnd.fill_between(k_grid, rnd, alpha=0.22, color=rnd_color, zorder=1)
        ax_rnd.plot(k_grid, rnd, color=rnd_color, lw=1.0, alpha=0.7,
                    label="RND", zorder=2)

        ax_iv.axvline(0, color="#555555", lw=0.8, ls="--", zorder=0)

        days      = t_exp * 365.25
        title_str = f"T={t_exp:.4f} ({days:.1f}d)"
        if has_neg:
            title_str += "  ⚠ neg RND"

        ax_iv.set_title(title_str, color="black", fontsize=8.5, pad=5)
        ax_iv.set_xlabel("Log-strike  k", color="#111111", fontsize=7.5)
        ax_iv.set_ylabel("Impl. Vol (%)", color=colors, fontsize=7.5)
        ax_rnd.set_ylabel("RND", color=rnd_color, fontsize=7.5)
        ax_iv.tick_params(colors="#111111", labelsize=7)
        ax_iv.tick_params(axis="y", colors=colors)
        ax_rnd.tick_params(colors=rnd_color, labelsize=7)
        ax_iv.set_facecolor("#ffffff")
        for spine in list(ax_iv.spines.values()) + list(ax_rnd.spines.values()):
            spine.set_edgecolor("#333333")

        if idx == 0:
            lines = (ax_iv.get_legend_handles_labels()[0]
                     + ax_rnd.get_legend_handles_labels()[0])
            labels = (ax_iv.get_legend_handles_labels()[1]
                      + ax_rnd.get_legend_handles_labels()[1])
            ax_iv.legend(lines, labels, fontsize=6.5, facecolor="#ffffff",
                         edgecolor="#444444", labelcolor="black", loc="upper right")

    # Hide unused axes
    for idx in range(n_slices, n_rows * n_cols):
        fig.add_subplot(n_rows, n_cols, idx + 1).set_visible(False)

    fig.suptitle(f"{result['model_name']} — Smile & Risk-Neutral Density",
                 color="black", fontsize=12, y=1.01)
    fig.patch.set_facecolor("#ffffff")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TOTAL VARIANCE OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

def plot_total_variance(result: dict, k_range=(-2.0, 2.0), n_grid=400,
                        figsize=(10, 5), cmap="rainbow",
                        n_cbar_ticks=8) -> plt.Figure:
    """
    Gatheral-style total variance plot.
    Colour encodes maturity continuously — no per-line legend.
    Crossed lines = calendar spread arbitrage.
    """
    model       = _get_model(result)
    expiries    = np.array(result["expiries"])
    params_list = result["params"]

    k_grid   = np.linspace(k_range[0], k_range[1], n_grid)
    colormap = plt.get_cmap(cmap)
    t_min, t_max = expiries.min(), expiries.max()
    norm = mcolors.Normalize(vmin=t_min, vmax=t_max)

    fig, ax = plt.subplots(figsize=figsize, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    for t_exp, p in zip(expiries, params_list):
        color = colormap(norm(t_exp))
        ax.plot(k_grid, model.w(k_grid, p), color=color, lw=1.4, alpha=0.85)

    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    ax.set_xlabel("Log-strike  k", color="black")
    ax.set_ylabel("Total implied variance  w = σ²T", color="black")
    ax.set_title(f"{result['model_name']} — Total Variance (no lines should cross)",
                 color="black")
    ax.tick_params(colors="#111111")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    sm   = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Time to expiry  T (years)", color="black", fontsize=9)
    tick_vals = np.linspace(t_min, t_max, n_cbar_ticks)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{t:.3f}" for t in tick_vals])
    cbar.ax.yaxis.set_tick_params(color="black", labelsize=7.5)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")
    cbar.outline.set_edgecolor("#444444")

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  METRICS BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics(result: dict, figsize=None) -> plt.Figure:
    """
    Bar chart of per-slice fit metrics.
    One subplot per metric, bars coloured by expiry.
    """
    expiries     = result["expiries"]
    metrics_dict = result["metrics"]
    metric_names = list(metrics_dict.keys())
    n_metrics    = len(metric_names)

    if figsize is None:
        figsize = (5 * n_metrics, 4)

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, facecolor="#ffffff")
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(expiries)))
    x      = np.arange(len(expiries))
    labels = [f"{t:.3f}" for t in expiries]

    for ax, metric in zip(axes, metric_names):
        vals = metrics_dict[metric]
        bars = ax.bar(x, vals, color=colors, edgecolor="#333333", linewidth=0.5)
        ax.set_facecolor("#ffffff")
        ax.set_title(metric, color="black", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right",
                           color="#111111", fontsize=7)
        ax.tick_params(colors="#111111")
        ax.set_xlabel("Expiry T", color="#111111", fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

        mean_val = np.nanmean(vals)
        ax.axhline(mean_val, color="black", lw=1.0, ls="--", alpha=0.6)
        ax.text(len(expiries) - 0.5, mean_val * 1.02,
                f"mean={mean_val:.4f}", color="black", fontsize=6.5, ha="right")

    fig.suptitle(f"{result['model_name']} — Fit Metrics by Expiry",
                 color="black", fontsize=11, y=1.02)
    fig.patch.set_facecolor("#ffffff")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMPARE MULTIPLE MODELS ON ONE SLICE
# ─────────────────────────────────────────────────────────────────────────────

def plot_compare(results: list, df=None, t_exp: float = None,
                 k_plot_range=(-2.0, 2.0), n_grid=400,
                 figsize=(10, 5)) -> plt.Figure:
    """
    Overlay fitted smiles from multiple models/objectives on a single expiry.

    Parameters
    ----------
    results  : list of result dicts from calibrate_snapshot()
    df       : optional snapshot DataFrame for market observations
    t_exp    : which expiry to plot (defaults to the median expiry)
    """
    # Pick expiry
    all_exp = results[0]["expiries"]
    if t_exp is None:
        t_exp = float(np.median(all_exp))
    else:
        t_exp = min(all_exp, key=lambda t: abs(t - t_exp))

    df_slice = df[df["t"] == t_exp] if df is not None else None
    k_lo, k_hi = _k_range(df_slice, global_range=k_plot_range)
    k_grid  = np.linspace(k_lo, k_hi, n_grid)

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(results)))

    fig, (ax_iv, ax_rnd) = plt.subplots(2, 1, figsize=figsize,
                                         facecolor="#ffffff", sharex=True)
    for ax in [ax_iv, ax_rnd]:
        ax.set_facecolor("#ffffff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.tick_params(colors="#111111")

    # Market observations on both
    _scatter_market(ax_iv, df_slice, t_exp)

    for res, color in zip(results, colors):
        model = _get_model(res)
        try:
            idx = res["expiries"].index(t_exp)
        except ValueError:
            idx = int(np.argmin([abs(e - t_exp) for e in res["expiries"]]))
        p = res["params"][idx]

        iv_fit  = model.iv(k_grid, p, t_exp)
        rnd     = model.butterfly_density(k_grid, p)
        label   = f"{res['model_name']} ({res['objective']})"

        ax_iv.plot(k_grid,  iv_fit * 100, color=color, lw=2.0, label=label)
        ax_rnd.plot(k_grid, rnd,           color=color, lw=1.5, label=label, alpha=0.8)
        ax_rnd.fill_between(k_grid, rnd,   color=color, alpha=0.12)

    for ax in [ax_iv, ax_rnd]:
        ax.axvline(0, color="#555555", lw=0.8, ls="--")

    ax_iv.set_ylabel("Implied Vol (%)", color="black")
    ax_rnd.set_ylabel("Risk-Neutral Density", color="black")
    ax_rnd.set_xlabel("Log-strike  k", color="black")
    ax_iv.legend(fontsize=7.5, facecolor="#ffffff", edgecolor="#444444",
                 labelcolor="black")

    days = t_exp * 365.25
    fig.suptitle(f"Model comparison — T={t_exp:.4f} ({days:.1f}d)",
                 color="black", fontsize=11)
    fig.patch.set_facecolor("#ffffff")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CONVENIENCE: ALL PLOTS AT ONCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(result: dict, df=None, save_prefix: str = None):
    """
    Generate surface, slices, total variance, and metrics plots.

    Parameters
    ----------
    save_prefix : str or None  — if given, saves PNGs with this prefix

    Returns
    -------
    (fig_surface, fig_slices, fig_totalvar, fig_metrics)
    """
    figs = {
        "surface":  plot_surface(result),
        "slices":   plot_slices(result, df=df),
        "totalvar": plot_total_variance(result),
        "metrics":  plot_metrics(result),
    }

    if save_prefix:
        for name, fig in figs.items():
            path = f"{save_prefix}_{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved {path}")

    return tuple(figs.values())
