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
    plot_price_vs_bidask(result, df)  Per-slice fitted price vs market bid/ask
    plot_total_variance(result)     Gatheral-style total variance overlay
    plot_vega(result)               Black-Scholes vega (or vega**power) by expiry
    plot_metrics(result)            Bar chart of per-slice fit metrics
    plot_compare(results, df)       Overlay multiple model fits on one slice
    plot_fitted_term_structure(result, df)    Every fitted expiry, one strike axis
    plot_rho_comparison(res_ssvi, res_essvi)  SSVI (const) vs eSSVI rho(theta)
    plot_all(result, df)            Generate all plots at once
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.cm import ScalarMappable
from scipy.interpolate import PchipInterpolator
from datetime import datetime
import seaborn as sns

from volatility_surface.models.vol_models import VolModel, get_model


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL PALETTE — RdPu
# ─────────────────────────────────────────────────────────────────────────────

_CMAP_FULL = plt.get_cmap("RdPu")
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "RdPu_trimmed",
    [_CMAP_FULL(x) for x in np.linspace(0.25, 1.0, 256)],
)
PALETTE = [CMAP(x) for x in np.linspace(0.0, 1.0, 8)]
PALETTE_HEX = [mcolors.to_hex(c) for c in PALETTE]

BG_COLOR     = "#ffffff"
FIT_COLOR    = PALETTE[4]       # mid-dark purple-pink for fitted curves
RND_OK_COLOR = PALETTE[1]       # light pink for well-behaved RND
RND_ARB_COLOR = PALETTE[7]      # deep magenta for butterfly-arb RND
BID_COLOR    = PALETTE[2]       # market bid scatter
ASK_COLOR    = PALETTE[6]       # market ask scatter


def _n_colors(n: int):
    """Sample `n` evenly spaced colours from the RdPu colourmap (skip the
    near-white bottom end)."""
    return [CMAP(x) for x in np.linspace(0.25, 0.95, n)]


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
                   color=BID_COLOR, s=12, alpha=0.7, marker="x",
                   label="Bid", zorder=4)
        ax.scatter(df_slice["k"], df_slice["ask_iv"] * 100,
                   color=ASK_COLOR, s=12, alpha=0.7, marker="x",
                   label="Ask", zorder=4)
    elif "mark_iv" in df_slice.columns:
        ax.scatter(df_slice["k"], df_slice['mark_iv'] * 100,
                   color="black", s=14, alpha=0.6, marker="o",
                   label="Mid", zorder=4)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  3D SURFACE
# ─────────────────────────────────────────────────────────────────────────────

def plot_surface(result: dict, k_range=(-2.0, 2.0), n_k=200, n_t=100,
                 colormap=None, figsize=(12, 7), elev=30, azim=-60) -> plt.Figure:
    """
    3D implied volatility surface across all fitted expiries.

    Parameters
    ----------
    elev, azim : initial camera angle (matplotlib 3D convention: elev = tilt
        above the k-T plane, azim = rotation around the vertical axis).
        Call again with different values for a different static view, e.g.
        plot_surface(result, elev=15, azim=100). If your matplotlib backend
        is interactive (e.g. %matplotlib widget in Jupyter, or any GUI
        backend/window instead of an inline static image), you can also
        click-and-drag the rendered figure to rotate freely — elev/azim
        here only set where it starts.
    """
    if colormap is None:
        colormap = CMAP
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

    fig = plt.figure(figsize=figsize, facecolor=BG_COLOR)
    ax  = fig.add_subplot(111, projection="3d", facecolor=BG_COLOR)

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
    ax.view_init(elev=elev, azim=azim)

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = True
        pane.set_facecolor(BG_COLOR)
        pane.set_edgecolor("#333333")
    ax.tick_params(colors="black")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PER-SLICE SMILE + RND
# ─────────────────────────────────────────────────────────────────────────────

def plot_slices(result: dict, df=None, n_cols=3,
                k_plot_range=(-2.0, 2.0), n_grid=400,
                figsize=None, slice_idx=None) -> plt.Figure:
    """One subplot per expiry: fitted smile on left axis, RND on right axis.

    slice_idx : optional list of integer indices into result['expiries'] to
                restrict the plot to a subset of maturities, e.g. [2, 5, 8, 11].
                None (default) plots every expiry.
    """
    model      = _get_model(result)
    expiries   = result["expiries"]
    params_list = result["params"]
    if slice_idx is not None:
        expiries    = [expiries[i]    for i in slice_idx]
        params_list = [params_list[i] for i in slice_idx]
    n_slices   = len(expiries)

    n_cols = min(n_cols, n_slices)
    n_rows = int(np.ceil(n_slices / n_cols))
    if figsize is None:
        figsize = (5.5 * n_cols, 4.2 * n_rows)

    fig    = plt.figure(figsize=figsize, facecolor=BG_COLOR)

    for idx, (t_exp, p) in enumerate(zip(expiries, params_list)):
        ax_iv  = fig.add_subplot(n_rows, n_cols, idx + 1)
        ax_rnd = ax_iv.twinx()

        df_slice = df[df["t"] == t_exp] if df is not None else None
        k_lo, k_hi = _k_range(df_slice, global_range=k_plot_range)
        k_grid     = np.linspace(k_lo, k_hi, n_grid)

        # Fitted smile
        iv_fit  = model.iv(k_grid, p, t_exp)
        ax_iv.plot(k_grid, iv_fit * 100, color=FIT_COLOR, lw=2.0,
                   label=f"{model.name} fit", zorder=3)

        # Market observations
        _scatter_market(ax_iv, df_slice, t_exp)

        # RND
        rnd       = model.butterfly_density(k_grid, p)
        has_neg   = model.has_butterfly_arb(p)
        rnd_color = RND_ARB_COLOR if has_neg else RND_OK_COLOR
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
        ax_iv.set_ylabel("Impl. Vol (%)", color=FIT_COLOR, fontsize=7.5)
        ax_rnd.set_ylabel("RND", color=rnd_color, fontsize=7.5)
        ax_iv.tick_params(colors="#111111", labelsize=7)
        ax_iv.tick_params(axis="y", colors=FIT_COLOR)
        ax_rnd.tick_params(colors=rnd_color, labelsize=7)
        ax_iv.set_facecolor(BG_COLOR)
        for spine in list(ax_iv.spines.values()) + list(ax_rnd.spines.values()):
            spine.set_edgecolor("#333333")

        if idx == 0:
            lines = (ax_iv.get_legend_handles_labels()[0]
                     + ax_rnd.get_legend_handles_labels()[0])
            labels = (ax_iv.get_legend_handles_labels()[1]
                      + ax_rnd.get_legend_handles_labels()[1])
            ax_iv.legend(lines, labels, fontsize=6.5, facecolor=BG_COLOR,
                         edgecolor="#444444", labelcolor="black", loc="upper right")

    # Hide unused axes
    for idx in range(n_slices, n_rows * n_cols):
        fig.add_subplot(n_rows, n_cols, idx + 1).set_visible(False)

    fig.suptitle(f"{result['model_name']} — Smile & Risk-Neutral Density",
                 color="black", fontsize=12, y=1.01)
    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2b.  PER-SLICE FITTED PRICE vs MARKET BID/ASK
# ─────────────────────────────────────────────────────────────────────────────
def plot_price_vs_bidask(result: dict, df, n_cols=3, r: float = 0.0,
                         k_plot_range=(-2.0, 2.0), n_grid=400,
                         figsize=None, slice_idx=None) -> plt.Figure:
    """
    One subplot per expiry: the fitted vol smile converted to option price
    (Black-76 — forward-normalized, F=1, matching this codebase's k=log(K/F)
    convention throughout), against the market bid/ask price band, plotted
    against strike K. Companion to `plot_slices`, in price space instead of
    vol space.

    Below k=0 the curve is priced as an OTM put, at/above k=0 as an OTM
    call — matching this codebase's OTM-only convention. The model is
    always evaluated in log-moneyness k (that's the only space model.iv()
    and the call/put split understand) and converted to strike only for
    the x-axis, via K = F * exp(k).

    Parameters
    ----------
    result : calibration result dict (must contain '_model', 'expiries', 'params')
    df     : snapshot DataFrame with columns k, t, option_type, bid_price,
             ask_price, strike, underlying_price
    r      : Black-76 discount rate (default 0 — undiscounted)
    slice_idx : optional list of integer indices into result['expiries'] to
             restrict the plot to a subset of maturities, e.g. [2, 5, 8, 11].
             None (default) plots every expiry.
    """
    from volatility_surface.core.pricing.pricing_models import gk_price

    model      = _get_model(result)
    expiries   = result["expiries"]
    params_list = result["params"]
    if slice_idx is not None:
        expiries    = [expiries[i]    for i in slice_idx]
        params_list = [params_list[i] for i in slice_idx]
    n_slices   = len(expiries)

    n_cols = min(n_cols, n_slices)
    n_rows = int(np.ceil(n_slices / n_cols))
    if figsize is None:
        figsize = (5.5 * n_cols, 4.2 * n_rows)

    fig = plt.figure(figsize=figsize, facecolor=BG_COLOR)

    for idx, (t_exp, p) in enumerate(zip(expiries, params_list)):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)

        df_slice = df[df["t"] == t_exp] if df is not None else None
        k_lo, k_hi = _k_range(df_slice, global_range=k_plot_range)
        k_grid     = np.linspace(k_lo, k_hi, n_grid)

        # Representative forward for this maturity, for the continuous fitted
        # curve's x-axis only (median across the slice's own rows — each row
        # already carries its own precise underlying_price, used directly
        # below wherever we convert an OBSERVED row's price to USD).
        F_repr = float(df_slice["underlying_price"].median()) if df_slice is not None and len(df_slice) > 0 else 1.0
        K_grid_usd = F_repr * np.exp(k_grid)

        # Fitted price curve: IV -> Black-76 price, OTM put for k<0, OTM call for k>=0.
        # Black-76 = Garman-Kohlhagen with q=r (forward already embeds carry,
        # so no extra (r-q) drift should be injected into d1) and S=F=1.
        iv_fit   = model.iv(k_grid, p, t_exp)
        K_norm   = np.exp(k_grid)
        opt_type = np.where(k_grid >= 0, "c", "p")
        price_fit = gk_price(1.0, K_norm, t_exp, r, r, iv_fit, opt_type)
        ax.plot(K_grid_usd, price_fit, color=FIT_COLOR, lw=1.0,
               label=f"{model.name} fit", zorder=3)

        # Market bid/ask price band, against real strikes
        days = t_exp * 365.25
        avg_price_err_rel = None
        avg_price_err_usd = None
        if df_slice is not None and len(df_slice) > 0:
            df_slice = df_slice.sort_values("strike")
            K_atm = df_slice.loc[df_slice["k"].abs().idxmin(), "strike"]

            ax.scatter(df_slice["strike"], df_slice["bid_price"], color=BID_COLOR, s=12,
                      alpha=0.8, marker="x", label="Bid", zorder=4)
            ax.scatter(df_slice["strike"], df_slice["ask_price"], color=ASK_COLOR, s=12,
                      alpha=0.8, marker="x", label="Ask", zorder=4)
            ax.axvline(K_atm, color="#555555", lw=0.8, ls="--", zorder=0)

            # Average price error RELATIVE to the option's own premium, but
            # measured as distance to the BID/ASK SPREAD rather than to the
            # mid: zero whenever the fitted price already falls inside
            # [bid, ask] (matching the spirit of the IV-space spread_hit_rate
            # metric), and the shortfall/excess past the nearer edge as a
            # fraction of mid price otherwise. Same 1e-8 floor
            # objectives.py's metric_price_rrmse uses, so a near-zero
            # deep-OTM premium doesn't blow up the ratio. Also reported in
            # fiat (USD) — dist_to_spread is in the same normalized
            # (fraction-of-spot) units as everything else here, so convert
            # to USD using each ROW'S OWN underlying_price (not a single
            # slice-wide forward) — rigorous in case the forward drifted
            # slightly between quotes captured within the same snapshot/slice.
            k_obs      = df_slice["k"].to_numpy()
            F_obs      = df_slice["underlying_price"].to_numpy()
            iv_fit_obs = model.iv(k_obs, p, t_exp)
            price_fit_obs = gk_price(1.0, np.exp(k_obs), t_exp, r, r, iv_fit_obs,
                                     df_slice["option_type"].to_numpy())
            bid_obs = df_slice["bid_price"].to_numpy()
            ask_obs = df_slice["ask_price"].to_numpy()
            mid_obs = df_slice["mid_price"].to_numpy()
            dist_to_spread = np.maximum(0.0, np.maximum(bid_obs - price_fit_obs,
                                                         price_fit_obs - ask_obs))
            err_rel = dist_to_spread / np.maximum(mid_obs, 1e-8)
            avg_price_err_rel = float(np.mean(err_rel))
            avg_price_err_usd = float(np.mean(dist_to_spread * F_obs))
            print(f"  T={t_exp:.4f} ({days:.1f}d)  avg_price_error={avg_price_err_rel:.2%} "
                  f"of premium beyond spread (${avg_price_err_usd:,.2f})  "
                  f"(n={len(df_slice)} strikes)")

        title_str = f"T={t_exp:.4f} ({days:.1f}d)"
        if avg_price_err_rel is not None:
            title_str += (f"\naverage distance from spread={avg_price_err_rel:.1%} of "
                          f"option premium (${avg_price_err_usd:,.2f})")
        ax.set_title(title_str, color="black", fontsize=8.5, pad=5)
        ax.set_xlabel("Strike", color="#111111", fontsize=7.5)
        ax.set_ylabel("Price  (fraction of spot)", color="#111111", fontsize=7.5)
        ax.tick_params(colors="#111111", labelsize=7)
        ax.set_facecolor(BG_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

        if idx == 0:
            ax.legend(fontsize=6.5, facecolor=BG_COLOR, edgecolor="#444444",
                     labelcolor="black", loc="upper right")

    for idx in range(n_slices, n_rows * n_cols):
        fig.add_subplot(n_rows, n_cols, idx + 1).set_visible(False)

    fig.suptitle(f"{result['model_name']} — Fitted Price vs Market Bid/Ask",
                 color="black", fontsize=12, y=1.01)
    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    return fig
# ─────────────────────────────────────────────────────────────────────────────
# 3.  TOTAL VARIANCE OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

def plot_total_variance(result: dict, k_range=(-2.0, 2.0), n_grid=400,
                        figsize=(10, 5), cmap=None,
                        n_cbar_ticks=8, shuffle_colors=False,
                        penalty_cal=None, snapshot_ts=None) -> plt.Figure:
    """
    Gatheral-style total variance plot.
    Colour encodes maturity continuously — no per-line legend.
    Crossed lines = calendar spread arbitrage.

    shuffle_colors : if True, curves alternate between the dark end and the
        light end of the colormap in T-order — every other curve (1st, 3rd,
        5th, ...) is drawn from the dark half, the rest (2nd, 4th, 6th, ...)
        from the light half. This guarantees every ADJACENT pair of curves
        (the ones most likely to be hugging each other in a well-fit no-arb
        chain) gets high contrast, unlike a smooth gradient where adjacent
        maturities get near-identical colours right where distinguishing
        them matters most. Falls back to a legend (T-labelled) instead of
        the continuous colorbar, since the colorbar would otherwise
        misrepresent the (now alternating) line colours.
    penalty_cal : the penalty_cal value the calibration was actually run
        with (not stored in `result` — pass the same value you gave
        calibrate_snapshot/calibrate_global_*). Purely for display in the
        title; has no effect on the plot itself. Omit to leave it off.
    snapshot_ts : the snapshot's file_timestamp (also not stored in
        `result` — pass the same SNAP_TS you loaded the data with).
        Purely for display in the title. Omit to leave it off.
    """
    from volatility_surface.core.calibration.calibrator import crossedness

    model       = _get_model(result)
    expiries    = np.array(result["expiries"])
    params_list = result["params"]

    k_grid   = np.linspace(k_range[0], k_range[1], n_grid)
    colormap = CMAP if cmap is None else plt.get_cmap(cmap)
    t_min, t_max = expiries.min(), expiries.max()
    norm = mcolors.Normalize(vmin=t_min, vmax=t_max)

    # Worst calendar crossedness across EVERY pair of slices (not just
    # adjacent ones) — matches the rigorous check, not just the one the
    # calibration objective itself enforces.
    worst_cross = 0.0
    for i in range(len(params_list)):
        for j in range(i + 1, len(params_list)):
            worst_cross = max(worst_cross, crossedness(model, params_list[i], params_list[j]))

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    if shuffle_colors:
        n = len(expiries)
        # Two FIXED representative colours (not a gradient per bucket) — a
        # gradient within each bucket inevitably drifts its tail end toward
        # the other bucket's range as n grows, killing contrast right where
        # it's needed. A flat dark/light split guarantees every adjacent
        # pair gets the same fixed high contrast regardless of n.
        positions = np.where(np.arange(n) % 2 == 0, 0.9, 0.3)
    else:
        positions = norm(expiries)

    for t_exp, p, pos in zip(expiries, params_list, positions):
        color = colormap(pos)
        ax.plot(k_grid, model.w(k_grid, p), color=color, lw=1.4, alpha=0.85,
               label=f"T={t_exp:.4f}")

    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    ax.set_xlabel("Log-strike  k", color="black")
    ax.set_ylabel("Total implied variance  w = σ²T", color="black")
    subtitle = f"worst crossedness={worst_cross:.4f}"
    if penalty_cal is not None:
        subtitle += f"   penalty_cal={penalty_cal:g}"
    if snapshot_ts is not None:
        subtitle += f"   snapshot={snapshot_ts}"
    ax.set_title(f"{result['model_name']} — Total Variance (no lines should cross)\n{subtitle}",
                 color="black")
    ax.tick_params(colors="#111111")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    if shuffle_colors:
        ax.legend(fontsize=6.5, ncol=2, loc="best", facecolor=BG_COLOR,
                 edgecolor="#444444", labelcolor="black")
        fig.tight_layout()
        return fig

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
# 3b.  BLACK-SCHOLES VEGA BY EXPIRY
# ─────────────────────────────────────────────────────────────────────────────

def plot_vega(result: dict, k_range=(-2.0, 2.0), n_grid=400,
             figsize=(10, 5), cmap=None, n_cbar_ticks=8,
             power=1, snapshot_ts=None) -> plt.Figure:
    """
    Black-Scholes vega (or vega**power) against log-strike, one curve per
    fitted expiry, colour-coded by maturity — same style as
    plot_total_variance (same colourbar, same per-expiry overlay). Vega is
    computed from each slice's own FITTED smile (model.iv), not raw market
    quotes, so it reflects exactly what the vega_wmse-family objectives
    actually see during calibration.

    Useful for seeing directly why vega-weighted objectives concentrate so
    heavily at the money: vega itself is already a decaying, roughly
    Gaussian bump peaked at k=0 (set power=2 to see the vega_wmse default's
    actual per-point weight, which decays roughly twice as fast in log
    terms as plain vega — see obj_iv_vega_wmse's docstring for why squaring
    is the economically-motivated choice, a first-order price-space
    equivalence, not just an empirical preference).

    Parameters
    ----------
    power : plot vega**power instead of raw vega (e.g. power=2 to match
        vega_wmse's default weighting, power=1 for vega_wmse_linear's).
    snapshot_ts : optional, for display in the title only (not stored in
        `result` — pass the same SNAP_TS you loaded the data with).
    """
    from volatility_surface.core.calibration.objectives import _bs_vega

    model       = _get_model(result)
    expiries    = np.array(result["expiries"])
    params_list = result["params"]

    k_grid   = np.linspace(k_range[0], k_range[1], n_grid)
    colormap = CMAP if cmap is None else plt.get_cmap(cmap)
    t_min, t_max = expiries.min(), expiries.max()
    norm = mcolors.Normalize(vmin=t_min, vmax=t_max)

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    for t_exp, p in zip(expiries, params_list):
        color = colormap(norm(t_exp))
        iv    = model.iv(k_grid, p, t_exp)
        vega  = _bs_vega(k_grid, iv, t_exp)
        ax.plot(k_grid, vega ** power, color=color, lw=1.4, alpha=0.85)

    ax.axvline(0, color="#555555", lw=0.8, ls="--")
    ax.set_xlabel("Log-strike  k", color="black")
    ax.set_ylabel("Vega" if power == 1 else f"Vega$^{power}$", color="black")
    title = f"Black-Scholes Vega" + ("" if power == 1 else f"$^{power}$")
    if snapshot_ts is not None:
        title += f"\nsnapshot={snapshot_ts}"
    ax.set_title(title, color="black")
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

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, facecolor=BG_COLOR)
    if n_metrics == 1:
        axes = [axes]

    colors = _n_colors(len(expiries))
    x      = np.arange(len(expiries))
    labels = [f"{t:.3f}" for t in expiries]

    for ax, metric in zip(axes, metric_names):
        vals = metrics_dict[metric]
        bars = ax.bar(x, vals, color=colors, edgecolor="#333333", linewidth=0.5)
        ax.set_facecolor(BG_COLOR)
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
    fig.patch.set_facecolor(BG_COLOR)
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

    colors = _n_colors(len(results))

    fig, (ax_iv, ax_rnd) = plt.subplots(2, 1, figsize=figsize,
                                         facecolor=BG_COLOR, sharex=True)
    for ax in [ax_iv, ax_rnd]:
        ax.set_facecolor(BG_COLOR)
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
    ax_iv.legend(fontsize=7.5, facecolor=BG_COLOR, edgecolor="#444444",
                 labelcolor="black")

    days = t_exp * 365.25
    fig.suptitle(f"Model comparison — T={t_exp:.4f} ({days:.1f}d)",
                 color="black", fontsize=11)
    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5b.  FITTED TERM STRUCTURE IN STRIKE SPACE — one snapshot, every expiry
# ─────────────────────────────────────────────────────────────────────────────

def plot_fitted_term_structure(result: dict, df: pd.DataFrame, k_pad: float = 0.6,
                               asset_label: str = None, xaxis: str = 'moneyness',figsize=(10, 5)) -> plt.Figure:
    """
    Overlay every fitted expiry's smile from ONE snapshot on a shared strike
    axis, coloured by expiry (short = light, long = dark) with the project's
    trimmed RdPu colormap. Classic post-calibration diagnostic (Reiswich-
    Wystup / Sepp style): short expiries show narrow, steep-winged coverage;
    long expiries flatten out and broaden.

    Parameters
    ----------
    result      : calibration result dict (must contain '_model', 'expiries', 'params')
    df          : the snapshot DataFrame the fit was calibrated on (used for
                  each slice's observed k-range and the underlying forward)
    k_pad       : how far beyond each slice's observed k-range to extend the
                  fitted curve
    asset_label : optional ticker (e.g. "ETH") shown in the title next to
                  the underlying price
    xaxis       : ['strike','moneyness']
    """
    model    = _get_model(result)
    expiries = result["expiries"]
    colors   = _n_colors(len(expiries))

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.tick_params(colors="#111111")

    for i, (t, params) in enumerate(zip(expiries, result["params"])):
        if params is None:
            continue
        s_sub = df[np.isclose(df["t"], t)]
        if s_sub.empty:
            continue
        F      = float(s_sub["underlying_price"].iloc[0])
        k_grid = np.linspace(s_sub["k"].min() - k_pad, s_sub["k"].max() + k_pad, 200)
        iv     = model.iv(k_grid, params, t)
        if xaxis=='strike':
            ax.plot(F * np.exp(k_grid), iv * 100, lw=1.6,
               color=colors[i], label=f"{t*365:.0f}d")
        elif xaxis=='moneyness':
            ax.plot((k_grid), iv * 100, lw=1.6,
               color=colors[i], label=f"{t*365:.0f}d")

    ax.set(xlabel="strike (USD)", ylabel="fitted IV (%)",
           title="fitted vols in strike space")
    ax.legend(fontsize=7, ncol=2, loc="upper right", facecolor=BG_COLOR,
             edgecolor="#444444", labelcolor="black")

    F_snap  = float(df["underlying_price"].median())
    snap_ts = df["file_timestamp"].iloc[0] if "file_timestamp" in df.columns else ""
    label   = f"{asset_label} " if asset_label else ""
    fig.suptitle(f"Term structure of fitted {result['model_name']} vols   —   "
                f"{snap_ts}   {label}≈ ${F_snap:,.0f}",
                fontsize=12, color="#6A0DAD", y=1.02)
    fig.patch.set_facecolor(BG_COLOR)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5c.  RHO(THETA) TERM STRUCTURE — SSVI (constant) vs eSSVI (maturity-dependent)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rho_comparison(res_ssvi: dict, res_essvi: dict, n_grid: int = 200,
                        figsize=(7, 5)) -> plt.Figure:
    """
    Compare the fitted correlation term structure between a global SSVI fit
    (constant rho) and a global eSSVI fit (rho(theta) = rho_inf +
    (rho_0-rho_inf)*exp(-lam*theta)), plotted against time to expiry T.
    Same idea as Figure 2 (right panel) in Hendriks & Martini's eSSVI paper
    (SSRN id2971502), but with T on the x-axis instead of theta_T — rho only
    depends on theta in the model, so theta(T) is interpolated (monotone
    cubic, same method plot_surface uses) from the fitted per-slice
    (T_i, theta_i) pairs to get a smooth curve in T.

    Parameters
    ----------
    res_ssvi  : result dict from calibrate_global_ssvi
    res_essvi : result dict from calibrate_global_essvi
    n_grid    : resolution of the eSSVI rho(T) curve
    """
    model_essvi = _get_model(res_essvi)
    rho_ssvi    = float(res_ssvi["params"][0]["rho"])

    expiries_essvi = np.array(res_essvi["expiries"], dtype=float)
    thetas_essvi   = np.array([p["theta"] for p in res_essvi["params"]])
    p_ref          = res_essvi["params"][0]   # rho_0/rho_inf/lam are shared

    theta_of_t   = PchipInterpolator(expiries_essvi, thetas_essvi, extrapolate=True)
    t_grid       = np.linspace(expiries_essvi.min(), expiries_essvi.max(), n_grid)
    rho_essvi_curve  = np.array([model_essvi._rho(th, p_ref) for th in theta_of_t(t_grid)])
    rho_essvi_slices = np.array([model_essvi._rho(th, p_ref) for th in thetas_essvi])

    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.tick_params(colors="#111111")

    ax.axhline(rho_ssvi, color=PALETTE[2], lw=2.0, label="SSVI (constant $\\rho$)")
    ax.plot(t_grid, rho_essvi_curve, color=PALETTE[6], lw=2.0,
           label="eSSVI  $\\rho(T)$")
    ax.scatter(expiries_essvi, rho_essvi_slices, color=PALETTE[6], s=28,
              zorder=5, edgecolor="#333333", linewidth=0.5)

    ax.set_xlabel(r"$T$  (time to expiry, years)", color="black")
    ax.set_ylabel(r"$\rho$", color="black")
    ax.legend(fontsize=9, facecolor=BG_COLOR, edgecolor="#444444", labelcolor="black")

    fig.suptitle("Correlation term structure — SSVI vs eSSVI", color="black", fontsize=11)
    fig.patch.set_facecolor(BG_COLOR)
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


# ─────────────────────────────────────────────────────────────────────────────
# METRIC TIME-SERIES  (comparison dict → one line per model)
# ─────────────────────────────────────────────────────────────────────────────

# Metrics where values are already in [0,1] and should be displayed as %
_PCT_METRICS = {"spread_hit_rate"}

# Metrics that are "lower is better" (affects nothing visually but useful to know)
_LOWER_IS_BETTER = {
    "vwrmse", "vwrrmse", "iv_rmse", "iv_rrmse", "w_rmse",
    "price_rmse", "price_rrmse", "price_vwrrmse", "max_iv_err",
    "iv_rmse_atm", "iv_rmse_otm",
}

_METRIC_YLABELS = {
    "spread_hit_rate": "Spread Hit Rate",
    "vwrmse":          "VW-RMSE  (vol, absolute)",
    "vwrrmse":         "VW-RRMSE  (vol, relative)",
    "iv_rrmse":        "IV Relative RMSE",
    "iv_rmse":         "IV RMSE  (vol pts)",
    "w_rmse":          "Total-Var RMSE",
    "price_rmse":      "Price RMSE",
    "price_rrmse":     "Price Relative RMSE",
    "price_vwrrmse":   "Price VW-RRMSE",
    "max_iv_err":      "Max IV Error  (vol pts)",
    "iv_rmse_atm":     "ATM IV RMSE  (vol pts)",
    "iv_rmse_otm":     "OTM IV RMSE  (vol pts)",
}


def plot_metric_by_date(
    comparison_dict,
    metric:      str   = "spread_hit_rate",
    models:      list  = None,
    colors:      dict  = None,
    markers:     dict  = None,
    title:       str   = None,
    subtitle:    str   = None,
    pct_format:  bool  = None,
    scale:       float = 1.0,
    figsize:     tuple = (12, 7),
    savepath:    str   = None,
    annotate:    bool  = True,
) -> plt.Figure:
    """
    Plot any scalar metric from a comparison dict over time.

    Parameters
    ----------
    comparison_dict : dict  {date_str → object with attributes = metric names}
        e.g. comparison_dict['2026-05-22'].vwrmse == {'Raw SVI': 0.01, ...}
    metric      : attribute name on the comparison objects, e.g. 'vwrmse',
                  'spread_hit_rate', 'iv_rrmse', 'price_vwrrmse', …
    models      : list of model names to plot.  Default: all keys in the first entry.
    colors      : dict {model: hex color}.  Default palette used if not given.
    markers     : dict {model: marker char}.
    title       : plot title.  Default: '<metric> by Model & Date'.
    subtitle    : small text below title.  Default: none.
    pct_format  : True = format y-axis as %; False = raw float; None = auto-detect.
    scale       : multiply all values by this before plotting (e.g. 100 to convert
                  decimal vol to %).  Default 1.0.
    figsize     : figure size.
    savepath    : if given, save PNG to this path.
    annotate    : if True, label each data point with its value.

    Returns
    -------
    matplotlib Figure
    """
    dates    = sorted(comparison_dict.keys())
    dates_dt = [datetime.strptime(d, '%Y-%m-%d') for d in dates]

    # Infer models from the first entry if not specified
    if models is None:
        first = getattr(next(iter(comparison_dict.values())), metric, None)
        if first is None:
            raise ValueError(f"Metric '{metric}' not found on comparison objects.")
        models = list(first.keys())

    # Default style
    default_colors  = PALETTE_HEX
    default_markers = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']
    colors  = colors  or {m: default_colors[i % len(default_colors)]
                          for i, m in enumerate(models)}
    markers = markers or {m: default_markers[i % len(default_markers)]
                          for i, m in enumerate(models)}

    # Gather values
    values = {m: [] for m in models}
    for date in dates:
        entry = getattr(comparison_dict[date], metric, {})
        for m in models:
            values[m].append(entry.get(m, float('nan')) * scale)

    # Auto-detect % formatting
    if pct_format is None:
        pct_format = (metric in _PCT_METRICS) and (scale == 1.0)

    # Titles
    if title is None:
        title = f"{metric.replace('_', ' ').upper()} by Model & Date"

    ylabel = _METRIC_YLABELS.get(metric, metric)
    if scale != 1.0:
        ylabel += f"  (×{scale})"

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    for model in models:
        ax.plot(
            dates_dt, values[model],
            label=model,
            color=colors[model],
            marker=markers[model],
            linewidth=2, markersize=7,
            markeredgewidth=1.2, markeredgecolor='white',
        )
        if annotate:
            for x, y in zip(dates_dt, values[model]):
                if np.isfinite(y):
                    label = f'{y:.0%}' if pct_format else f'{y:.4f}'
                    ax.annotate(label, xy=(x, y),
                                xytext=(0, 10), textcoords='offset points',
                                ha='center', fontsize=7.5, color=colors[model])

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(dates_dt, rotation=35, ha='right', fontsize=9)

    if pct_format:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold',
                 pad=26 if subtitle else 14)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=10, color='gray')

    ax.legend(fontsize=10, framealpha=0.85)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.grid(axis='x', linestyle=':', alpha=0.25)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f"Saved → {savepath}")

    plt.show()
    return fig
