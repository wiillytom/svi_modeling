"""
SVI Plotting Functions
======================
Two plotting utilities for a calibrated SVI snapshot:

1. plot_surface(result)
   3D implied volatility surface across all expiries.

2. plot_slices(result, df=None)
   One subplot per expiry showing:
     - Fitted SVI implied vol smile
     - Market bid/ask scatter (if df passed)
     - Risk-neutral density (RND) on a secondary axis

Both functions return the matplotlib Figure so you can save or display them.

Usage
-----
    from svi_plots import plot_surface, plot_slices

    fig1 = plot_surface(result)
    fig2 = plot_slices(result, df=df_snap)
    plt.show()

The `result` dict is the output of calibrate_snapshot() from svi_snapshot_calibration.py.
The `df` dataframe needs columns  k, w, t  and optionally  mark_iv, bid_iv, ask_iv.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _svi_w(k, p):
    """Total implied variance from raw SVI params dict."""
    a, b, rho, m, sig = p["a"], p["b"], p["rho"], p["m"], p["sig"]
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2))


def _svi_iv(k, p, t):
    """Implied volatility from raw SVI."""
    w = _svi_w(k, p)
    return np.sqrt(np.maximum(w / t, 0.0))


def _rnd(k, p):
    """
    Risk-neutral density from the Breeden-Litzenberger formula applied to the
    SVI smile, expressed as a density over log-moneyness k.

    p(k) = g(k) / sqrt(2*pi*w(k)) * exp(-d_minus(k)^2 / 2)

    where  g(k)  is the butterfly-arbitrage indicator function and
    d_minus(k) = -k/sqrt(w) - sqrt(w)/2.
    """
    k      = np.asarray(k, dtype=float)
    a, b, rho, m, sig = p["a"], p["b"], p["rho"], p["m"], p["sig"]

    discr  = np.sqrt((k - m) ** 2 + sig ** 2)
    w      = a + b * (rho * (k - m) + discr)
    w      = np.maximum(w, 1e-10)

    dw     = b * rho + b * (k - m) / discr
    d2w    = b * sig ** 2 / discr ** 3

    # g(k) from Gatheral eq (2.1)
    g = (
        (1 - k * dw / (2 * w)) ** 2
        - (dw ** 2 / 4) * (1 / w + 0.25)
        + d2w / 2
    )

    d_minus = -k / np.sqrt(w) - np.sqrt(w) / 2.0
    density = g / np.sqrt(2 * np.pi * w) * np.exp(-d_minus ** 2 / 2)

    return np.maximum(density, 0.0)   # clip negatives (butterfly arb region)


def _k_range_for_slice(df_slice, padding=0.3):
    """Sensible k range from the data, with some extrapolation padding."""
    if df_slice is not None and len(df_slice) > 0:
        lo = df_slice["k"].min() - padding
        hi = df_slice["k"].max() + padding
    else:
        lo, hi = -2.0, 2.0
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# 1.  3D SURFACE PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_surface(result, k_range=(-2.0, 2.0), n_k=200, n_t=100,
                 colormap="plasma", figsize=(12, 7)):
    """
    3D plot of the fitted implied volatility surface.

    Parameters
    ----------
    result   : dict from calibrate_snapshot()
    k_range  : (float, float)  log-strike range to plot
    n_k      : int  number of strike grid points
    n_t      : int  number of maturity interpolation points
    colormap : str  matplotlib colormap
    figsize  : tuple

    Returns
    -------
    matplotlib Figure
    """
    expiries   = np.array(result["expiries"])
    svi_params = result["svi_params"]

    k_grid = np.linspace(k_range[0], k_range[1], n_k)
    t_grid = np.linspace(expiries.min(), expiries.max(), n_t)

    # Build iv surface: for each t, interpolate params across expiries
    # Simple approach: for each t find the bracketing expiries and interpolate w
    from scipy.interpolate import PchipInterpolator

    iv_surface = np.zeros((n_t, n_k))
    for j, k_val in enumerate(k_grid):
        w_at_expiries = np.array([_svi_w(k_val, p) for p in svi_params])
        interp = PchipInterpolator(expiries, w_at_expiries, extrapolate=True)
        w_interp = np.maximum(interp(t_grid), 1e-10)
        iv_surface[:, j] = np.sqrt(w_interp / t_grid)

    T_mesh, K_mesh = np.meshgrid(t_grid, k_grid, indexing="ij")

    fig = plt.figure(figsize=figsize, facecolor="#0d0d0d")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#0d0d0d")

    surf = ax.plot_surface(
        K_mesh, T_mesh, iv_surface,
        cmap=colormap, linewidth=0, antialiased=True, alpha=0.92,
        rcount=n_t, ccount=n_k,
    )

    # Scatter the actual calibrated ATM points
    for t_exp, p in zip(expiries, svi_params):
        atm_iv = _svi_iv(0.0, p, t_exp)
        ax.scatter(0.0, t_exp, atm_iv, color="white", s=18, zorder=5)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08)
    cbar.set_label("Implied Volatility", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("Log-strike  k", color="white", labelpad=8)
    ax.set_ylabel("Time to expiry  T", color="white", labelpad=8)
    ax.set_zlabel("Implied Vol  σ", color="white", labelpad=8)
    ax.set_title("SVI Implied Volatility Surface", color="white",
                 fontsize=13, pad=14)

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#333333")

    ax.tick_params(colors="white")
    ax.xaxis.line.set_color("white")
    ax.yaxis.line.set_color("white")
    ax.zaxis.line.set_color("white")

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PER-SLICE: SMILE + RND
# ─────────────────────────────────────────────────────────────────────────────

def plot_slices(result, df=None, n_cols=3, k_plot_range=(-2.0, 2.0),
                n_grid=400, figsize=None):
    """
    One subplot per expiry showing the fitted smile and the risk-neutral density.

    Parameters
    ----------
    result       : dict from calibrate_snapshot()
    df           : optional DataFrame with columns k, t, and optionally bid_iv / ask_iv
                   (the market observations are overlaid as scatter points)
    n_cols       : int  number of columns in the subplot grid
    k_plot_range : (float, float)  log-strike range for the smooth fitted curves
    n_grid       : int  number of points in the smooth curve
    figsize      : tuple or None (auto-sized)

    Returns
    -------
    matplotlib Figure
    """
    expiries   = result["expiries"]
    svi_params = result["svi_params"]
    n_slices   = len(expiries)

    n_cols = min(n_cols, n_slices)
    n_rows = int(np.ceil(n_slices / n_cols))

    if figsize is None:
        figsize = (5.5 * n_cols, 4.2 * n_rows)

    fig = plt.figure(figsize=figsize, facecolor="#111111")

    # Consistent colour palette across slices
    slice_colors = plt.cm.plasma(np.linspace(0.15, 0.85, n_slices))

    for idx, (t_exp, p, color) in enumerate(zip(expiries, svi_params, slice_colors)):

        ax_iv  = fig.add_subplot(n_rows, n_cols, idx + 1)
        ax_rnd = ax_iv.twinx()

        # ── Data range ────────────────────────────────────────────────────────
        df_slice = df[df["t"] == t_exp] if df is not None else None
        k_lo, k_hi = _k_range_for_slice(df_slice, padding=0.25)
        k_lo = max(k_lo, k_plot_range[0])
        k_hi = min(k_hi, k_plot_range[1])
        k_grid = np.linspace(k_lo, k_hi, n_grid)

        # ── Fitted smile ──────────────────────────────────────────────────────
        iv_fit = _svi_iv(k_grid, p, t_exp)
        ax_iv.plot(k_grid, iv_fit * 100, color=color, lw=2.0,
                   label="SVI fit", zorder=3)

        # ── Market data scatter ───────────────────────────────────────────────
        if df_slice is not None and len(df_slice) > 0:
            # Try bid/ask first, fall back to mark_iv or w
            if "bid_iv" in df_slice.columns and "ask_iv" in df_slice.columns:
                ax_iv.scatter(df_slice["k"], df_slice["bid_iv"] * 100,
                              color="#e05252", s=12, alpha=0.7,
                              label="Bid", zorder=4, marker="v")
                ax_iv.scatter(df_slice["k"], df_slice["ask_iv"] * 100,
                              color="#5288e0", s=12, alpha=0.7,
                              label="Ask", zorder=4, marker="^")
            elif "mark_iv" in df_slice.columns:
                ax_iv.scatter(df_slice["k"], df_slice["mark_iv"] * 100,
                              color="white", s=14, alpha=0.6,
                              label="Mid", zorder=4, marker="o")
            else:
                # derive from w
                w_obs  = df_slice["w"].values
                iv_obs = np.sqrt(np.maximum(w_obs / t_exp, 0.0)) * 100
                ax_iv.scatter(df_slice["k"], iv_obs,
                              color="white", s=14, alpha=0.6,
                              label="Mid", zorder=4, marker="o")

        # ── RND ───────────────────────────────────────────────────────────────
        rnd = _rnd(k_grid, p)
        has_neg = np.any(rnd <= 0)   # flag if butterfly arb present
        rnd_color = "#ff6b35" if has_neg else "#7ecf7e"
        ax_rnd.fill_between(k_grid, rnd, alpha=0.25, color=rnd_color, zorder=1)
        ax_rnd.plot(k_grid, rnd, color=rnd_color, lw=1.0,
                    alpha=0.7, zorder=2, label="RND")

        # ── ATM line ──────────────────────────────────────────────────────────
        ax_iv.axvline(0, color="#555555", lw=0.8, ls="--", zorder=0)

        # ── Styling ───────────────────────────────────────────────────────────
        days = t_exp * 365.25
        title_str = f"T = {t_exp:.4f}  ({days:.1f}d)"
        if has_neg:
            title_str += "  ⚠ neg RND"

        ax_iv.set_title(title_str, color="white", fontsize=8.5, pad=5)
        ax_iv.set_xlabel("Log-strike  k", color="#aaaaaa", fontsize=7.5)
        ax_iv.set_ylabel("Impl. Vol (%)", color=color, fontsize=7.5)
        ax_rnd.set_ylabel("RND", color=rnd_color, fontsize=7.5)

        ax_iv.tick_params(colors="#aaaaaa", labelsize=7)
        ax_rnd.tick_params(colors=rnd_color, labelsize=7)

        for spine in ax_iv.spines.values():
            spine.set_edgecolor("#333333")
        for spine in ax_rnd.spines.values():
            spine.set_edgecolor("#333333")

        ax_iv.set_facecolor("#1a1a1a")
        ax_iv.yaxis.label.set_color(color)
        ax_iv.tick_params(axis="y", colors=color)

        # Legend — only show on first subplot to avoid clutter
        if idx == 0:
            lines_iv,  labels_iv  = ax_iv.get_legend_handles_labels()
            lines_rnd, labels_rnd = ax_rnd.get_legend_handles_labels()
            ax_iv.legend(lines_iv + lines_rnd, labels_iv + labels_rnd,
                         fontsize=6.5, facecolor="#1a1a1a",
                         edgecolor="#444444", labelcolor="white",
                         loc="upper right")

    # Hide unused subplots
    total_axes = n_rows * n_cols
    for idx in range(n_slices, total_axes):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        ax.set_visible(False)

    fig.suptitle("SVI Smile & Risk-Neutral Density — per expiry",
                 color="white", fontsize=12, y=1.01)

    fig.patch.set_facecolor("#111111")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TOTAL VARIANCE PLOT  (Gatheral-style: no lines should cross)
# ─────────────────────────────────────────────────────────────────────────────

def plot_total_variance(result, k_range=(-2.0, 2.0), n_grid=400, figsize=(9, 5)):
    """
    Overlay total variance w(k) for all slices on one axes.
    Crossed lines = calendar spread arbitrage.

    Returns
    -------
    matplotlib Figure
    """
    expiries   = result["expiries"]
    svi_params = result["svi_params"]
    n_slices   = len(expiries)

    colors  = plt.cm.rainbow(np.linspace(0.0, 1.0, n_slices))
    k_grid  = np.linspace(k_range[0], k_range[1], n_grid)

    fig, ax = plt.subplots(figsize=figsize, facecolor="#111111")
    ax.set_facecolor("#1a1a1a")

    for t_exp, p, c in zip(expiries, svi_params, colors):
        w_fit = _svi_w(k_grid, p)
        label = f"T={t_exp:.3f}"
        ax.plot(k_grid, w_fit, color=c, lw=1.6, label=label)

    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    ax.set_xlabel("Log-strike  k", color="white")
    ax.set_ylabel("Total implied variance  w = σ²T", color="white")
    ax.set_title("Total Variance Plot  (no lines should cross)", color="white")
    ax.tick_params(colors="#aaaaaa")

    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    legend = ax.legend(fontsize=6.5, ncol=max(1, n_slices // 8),
                       facecolor="#1a1a1a", edgecolor="#444444",
                       labelcolor="white", loc="upper left")

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: plot everything at once
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(result, df=None, save_prefix=None):
    """
    Generate all three plots and optionally save them.

    Parameters
    ----------
    result      : dict from calibrate_snapshot()
    df          : optional snapshot DataFrame
    save_prefix : str or None — if given, saves as
                  {save_prefix}_surface.png
                  {save_prefix}_slices.png
                  {save_prefix}_totalvar.png

    Returns
    -------
    (fig_surface, fig_slices, fig_totalvar)
    """
    fig_surface  = plot_surface(result)
    fig_slices   = plot_slices(result, df=df)
    fig_totalvar = plot_total_variance(result)

    if save_prefix:
        fig_surface .savefig(f"{save_prefix}_surface.png",  dpi=150, bbox_inches="tight")
        fig_slices  .savefig(f"{save_prefix}_slices.png",   dpi=150, bbox_inches="tight")
        fig_totalvar.savefig(f"{save_prefix}_totalvar.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {save_prefix}_surface.png / _slices.png / _totalvar.png")

    return fig_surface, fig_slices, fig_totalvar


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Minimal self-test with synthetic data
    import matplotlib
    matplotlib.use("Agg")   # headless — swap to TkAgg / Qt5Agg for interactive

    # Fake calibration result (3 expiries)
    fake_result = {
        "expiries":   [0.1, 0.25, 0.5],
        "svi_params": [
            {"a": 0.01, "b": 0.15, "rho": -0.7, "m": 0.05, "sig": 0.15},
            {"a": 0.02, "b": 0.12, "rho": -0.6, "m": 0.04, "sig": 0.20},
            {"a": 0.03, "b": 0.10, "rho": -0.5, "m": 0.03, "sig": 0.25},
        ],
        "n_points": [30, 28, 25],
    }

    fig1, fig2, fig3 = plot_all(fake_result, save_prefix="/tmp/svi_test")
    print("Self-test complete — check /tmp/svi_test_*.png")
