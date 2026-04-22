"""
SVI Parametrization Interactive Explorer
=========================================
Based on Gatheral & Jacquier (2014) "Arbitrage-free SVI volatility surfaces"

Parametrizations implemented:
  - Raw SVI:     w(k) = a + b * { rho*(k-m) + sqrt((k-m)^2 + sigma^2) }
  - Natural SVI: equivalent reformulation with (Delta, mu, rho, omega, zeta)
  - SVI-JW:      trader-friendly parametrization with (v_t, psi_t, p_t, c_t, v_tilde_t)
                 NOTE: JW has an explicit dependence on time-to-expiry t.

The app always plots the RAW SVI formula. In JW mode, sliders control JW parameters
which are then converted to raw parameters via Lemma 3.2 of the paper.

WARNING on JW inversion (Lemma 3.2):
  The inversion requires beta = rho - 2*psi_t*sqrt(w_t)/b in [-1, 1].
  This is equivalent to -p_t <= 2*psi_t <= c_t (convexity of smile).
  If this is violated the conversion will clamp/warn and the curve may be invalid.
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons, Button
import warnings

matplotlib.use("TkAgg")  # Works on most systems; fall back to Qt5Agg if needed


# ─────────────────────────────────────────────────────────────────────────────
# Core SVI mathematics
# ─────────────────────────────────────────────────────────────────────────────

def raw_svi_total_variance(k, a, b, rho, m, sigma):
    """
    Raw SVI total implied variance.
    w(k) = a + b * { rho*(k-m) + sqrt((k-m)^2 + sigma^2) }

    Parameters (from Gatheral & Jacquier §3.1):
      a     : level of variance                    (a in R,  a + b*sigma*sqrt(1-rho^2) >= 0)
      b     : slope / tightness of smile           (b >= 0)
      rho   : correlation / rotation               (|rho| < 1)
      m     : horizontal translation               (m in R)
      sigma : ATM curvature smoother               (sigma > 0)
    """
    z = k - m
    return a + b * (rho * z + np.sqrt(z**2 + sigma**2))


def raw_svi_implied_vol(k, a, b, rho, m, sigma):
    """Total variance -> implied vol (for plotting readability)."""
    w = raw_svi_total_variance(k, a, b, rho, m, sigma)
    w = np.maximum(w, 1e-10)
    return np.sqrt(w)


def compute_rnd(k, a, b, rho, m, sigma):
    """
    Risk-neutral density via Breeden-Litzenberger (eq. 5 of the paper):
        p(k) = g(k) / sqrt(2*pi*w(k)) * exp(-d_minus(k)^2 / 2)

    where g(k) is the butterfly-arbitrage function (Lemma 2.2):
        g(k) = (1 - k*w'/(2w))^2 - (w')^2/4 * (1/w + 1/4) + w''/2

    We compute w', w'' numerically on the same grid for robustness.
    Returns rnd array (same shape as k), clipped to >= 0.
    Note: negative values of g indicate butterfly arbitrage.
    """
    w  = raw_svi_total_variance(k, a, b, rho, m, sigma)
    w  = np.maximum(w, 1e-10)

    # Numerical first and second derivatives (central differences)
    dk  = k[1] - k[0]
    wp  = np.gradient(w, dk)     # w'
    wpp = np.gradient(wp, dk)    # w''

    # g(k) — Lemma 2.2
    term1 = (1.0 - k * wp / (2.0 * w)) ** 2
    term2 = (wp ** 2) / 4.0 * (1.0 / w + 0.25)
    g     = term1 - term2 + wpp / 2.0

    # d_minus
    d_minus = -k / np.sqrt(w) - np.sqrt(w) / 2.0

    rnd = g / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * d_minus ** 2)

    # Negative rnd signals arbitrage — keep raw values so caller can detect it
    return rnd, g


# ─────────────────────────────────────────────────────────────────────────────
# Natural SVI  <->  Raw SVI   (Lemma 3.1)
# ─────────────────────────────────────────────────────────────────────────────

def natural_to_raw(Delta, mu, rho, omega, zeta):
    """
    Natural SVI -> Raw SVI  (Lemma 3.1, eq. 3.3)
      a     = Delta + omega/2 * (1 - rho^2)
      b     = omega * zeta / 2
      rho   = rho
      m     = mu - rho / zeta
      sigma = sqrt(1 - rho^2) / zeta
    """
    a     = Delta + omega / 2.0 * (1.0 - rho**2)
    b     = omega * zeta / 2.0
    m     = mu - rho / zeta
    sigma = np.sqrt(max(1.0 - rho**2, 1e-12)) / zeta
    return a, b, rho, m, sigma


def raw_to_natural(a, b, rho, m, sigma):
    """
    Raw SVI -> Natural SVI  (Lemma 3.1, eq. 3.4)
      omega = 2*b*sigma / sqrt(1-rho^2)
      zeta  = sqrt(1-rho^2) / sigma
      mu    = m + rho*sigma / sqrt(1-rho^2)
      Delta = a - omega/2 * (1-rho^2)
    """
    denom = np.sqrt(max(1.0 - rho**2, 1e-12))
    omega = 2.0 * b * sigma / denom
    zeta  = denom / sigma
    mu    = m + rho * sigma / denom
    Delta = a - omega / 2.0 * (1.0 - rho**2)
    return Delta, mu, rho, omega, zeta


# ─────────────────────────────────────────────────────────────────────────────
# SVI-JW  <->  Raw SVI   (eq. 3.5 and Lemma 3.2)
# ─────────────────────────────────────────────────────────────────────────────

def raw_to_jw(a, b, rho, m, sigma, t):
    """
    Raw SVI -> SVI-JW  (eq. 3.5).
    Requires t > 0.

    Returns (v_t, psi_t, p_t, c_t, v_tilde_t)
    where:
      v_t      = ATM implied variance (annualised)
      psi_t    = ATM volatility skew  d(sigma_BS)/dk |_{k=0}
      p_t      = slope of put (left) wing
      c_t      = slope of call (right) wing
      v_tilde_t= minimum implied variance (annualised)
    """
    if t <= 0:
        raise ValueError("t must be strictly positive for JW parametrization")

    # ATM total variance
    w_t = a + b * (-rho * m + np.sqrt(m**2 + sigma**2))
    v_t = w_t / t

    sqrt_wt = np.sqrt(max(w_t, 1e-12))

    psi_t     = (b / (2.0 * sqrt_wt)) * (-m / np.sqrt(m**2 + sigma**2) + rho)
    p_t       = b * (1.0 - rho) / sqrt_wt
    c_t       = b * (1.0 + rho) / sqrt_wt
    v_tilde_t = (a + b * sigma * np.sqrt(max(1.0 - rho**2, 0.0))) / t

    return v_t, psi_t, p_t, c_t, v_tilde_t


def jw_to_raw(v_t, psi_t, p_t, c_t, v_tilde_t, t):
    """
    SVI-JW -> Raw SVI  (Lemma 3.2).

    The inversion requires beta = rho - 2*psi_t*sqrt(w_t)/b in [-1, 1].
    This is equivalent to convexity:  -p_t <= 2*psi_t <= c_t.

    Returns (a, b, rho, m, sigma) and a warning string (empty if OK).
    """
    warning = ""
    w_t = v_t * t

    # b and rho from direct expressions
    sqrt_wt = np.sqrt(max(w_t, 1e-12))
    b   = sqrt_wt / 2.0 * (c_t + p_t)
    rho = 1.0 - p_t * sqrt_wt / max(b, 1e-12)
    rho = np.clip(rho, -0.9999, 0.9999)

    # a from v_tilde
    a = v_tilde_t * t - b * np.sqrt(max(1.0 - rho**2, 0.0))  # * sigma will be added
    # We need sigma first; a is refined below

    # beta and alpha for m and sigma
    beta = rho - 2.0 * psi_t * sqrt_wt / max(b, 1e-12)
    if abs(beta) > 1.0:
        warning = (f"⚠  beta = {beta:.3f} outside [-1,1].\n"
                   f"   Convexity violated: need -p_t ≤ 2·ψ_t ≤ c_t.\n"
                   f"   Clamping beta; curve may be unreliable.")
        beta = np.clip(beta, -0.9999, 0.9999)

    alpha = np.sign(beta) * np.sqrt(max(1.0 / beta**2 - 1.0, 0.0))

    denom = (-rho + np.sign(alpha) * np.sqrt(1.0 + alpha**2)
             - alpha * np.sqrt(max(1.0 - rho**2, 0.0)))
    if abs(denom) < 1e-12:
        # m = 0 branch
        m = 0.0
        sigma_val = (w_t - a) / max(b, 1e-12) if abs(b) > 1e-12 else sigma
        a = v_tilde_t * t - b * sigma_val * np.sqrt(max(1.0 - rho**2, 0.0))
    else:
        m = (v_t - v_tilde_t) * t / (b * denom)
        sigma_val = alpha * m

    # Ensure sigma > 0
    if sigma_val <= 0:
        warning += f"\n⚠  sigma = {sigma_val:.4f} ≤ 0; clamped to 1e-4."
        sigma_val = 1e-4

    a = v_tilde_t * t - b * sigma_val * np.sqrt(max(1.0 - rho**2, 0.0))

    # Final positivity check
    if a + b * sigma_val * np.sqrt(max(1.0 - rho**2, 0.0)) < 0:
        warning += "\n⚠  w_min < 0: arbitrage possible."

    return a, b, rho, m, sigma_val, warning


# ─────────────────────────────────────────────────────────────────────────────
# Default parameter sets
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_RAW = dict(a=0.04, b=0.1, rho=-0.4, m=0.0, sigma=0.2)
DEFAULT_T   = 1.0  # time to expiry for JW

# Compute default JW from default Raw
_jw0 = raw_to_jw(**DEFAULT_RAW, t=DEFAULT_T)
DEFAULT_JW = dict(v_t=_jw0[0], psi_t=_jw0[1], p_t=_jw0[2],
                  c_t=_jw0[3], v_tilde_t=_jw0[4])

# Compute default Natural from default Raw
_nat0 = raw_to_natural(**DEFAULT_RAW)
DEFAULT_NAT = dict(Delta=_nat0[0], mu=_nat0[1], rho=_nat0[2],
                   omega=_nat0[3], zeta=_nat0[4])


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (light, modern, quant-desk feel)
# ─────────────────────────────────────────────────────────────────────────────

C = dict(
    bg        = "#F7F8FA",
    panel     = "#FFFFFF",
    accent    = "#1A56DB",    # strong blue
    accent2   = "#E02020",    # red for ATM marker
    accent3   = "#16A34A",    # green
    text      = "#111827",
    subtext   = "#6B7280",
    grid      = "#E5E7EB",
    slider_bg = "#EEF2FF",
    raw_line  = "#1A56DB",
    nat_line  = "#16A34A",
    jw_line   = "#E02020",
    shadow    = "#C7D2FE",
    rnd_fill  = "#F59E0B",    # amber fill for RND
    rnd_neg   = "#FCA5A5",    # light red when RND goes negative (arbitrage)
)

PARAM_COLORS = {
    "a":     "#1A56DB",
    "b":     "#7C3AED",
    "rho":   "#DB2777",
    "m":     "#D97706",
    "sigma": "#16A34A",
}


# ─────────────────────────────────────────────────────────────────────────────
# Main GUI class
# ─────────────────────────────────────────────────────────────────────────────

class SVIExplorer:
    """
    Interactive matplotlib GUI for SVI parametrization explorer.
    Layout:
      Left panel   : Smile plot (total variance) + Implied vol plot
      Right panel  : Sliders (mode-dependent) + Raw parameter readout
      Bottom strip : Mode selector (Raw / Natural / JW) + T slider (JW only)
    """

    K_RANGE   = np.linspace(-2.0, 2.0, 400)   # log-strike range
    PLOT_YLIM = (0.0, 0.6)                     # vol plot y-axis (fixed for clarity)
    W_YLIM    = (0.0, 0.35)                    # total var y-axis

    # Slider specs: (label, min, max, init, step)
    SLIDERS_RAW = [
        ("a",     -0.02, 0.20, DEFAULT_RAW["a"],     0.001),
        ("b",      0.00, 0.50, DEFAULT_RAW["b"],     0.005),
        ("ρ",     -0.99, 0.99, DEFAULT_RAW["rho"],   0.01 ),
        ("m",     -1.00, 1.00, DEFAULT_RAW["m"],     0.01 ),
        ("σ",      0.01, 1.00, DEFAULT_RAW["sigma"], 0.01 ),
    ]

    SLIDERS_NAT = [
        ("Δ",     -0.10, 0.20, DEFAULT_NAT["Delta"], 0.001),
        ("μ",     -1.00, 1.00, DEFAULT_NAT["mu"],    0.01 ),
        ("ρ",     -0.99, 0.99, DEFAULT_NAT["rho"],   0.01 ),
        ("ω",      0.00, 0.60, DEFAULT_NAT["omega"], 0.005),
        ("ζ",      0.10, 5.00, DEFAULT_NAT["zeta"],  0.05 ),
    ]

    SLIDERS_JW = [
        ("v̄ₜ",    0.001, 0.30, DEFAULT_JW["v_t"],      0.001),
        ("ψₜ",   -0.50, 0.50, DEFAULT_JW["psi_t"],    0.005),
        ("pₜ",    0.001, 1.00, DEFAULT_JW["p_t"],      0.005),
        ("cₜ",    0.001, 1.00, DEFAULT_JW["c_t"],      0.005),
        ("ṽₜ",    0.001, 0.20, DEFAULT_JW["v_tilde_t"],0.001),
    ]

    def __init__(self):
        self.mode = "Raw"           # "Raw" | "Natural" | "JW"
        self.t    = DEFAULT_T
        self._warning = ""

        self._build_figure()
        self._build_plots()
        self._build_sliders()
        self._build_mode_selector()
        self._build_reset_button()
        self._build_raw_readout()
        self._update()

    # ──────────────────────────────────────────────────────────────────────
    # Figure / axes layout
    # ──────────────────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(15, 8), facecolor=C["bg"])
        self.fig.canvas.manager.set_window_title("SVI Parametrization Explorer")

        # Top-level grid: [plots | controls]
        gs_top = gridspec.GridSpec(
            1, 2,
            width_ratios=[1.45, 1],
            left=0.04, right=0.98,
            top=0.93, bottom=0.10,
            wspace=0.06,
        )

        # Left: two stacked plots
        gs_left = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_top[0], hspace=0.35
        )
        self.ax_var = self.fig.add_subplot(gs_left[0])
        self.ax_vol = self.fig.add_subplot(gs_left[1])

        # Right: sliders + readout
        gs_right = gridspec.GridSpecFromSubplotSpec(
            3, 1,
            subplot_spec=gs_top[1],
            height_ratios=[0.38, 0.38, 0.24],
            hspace=0.05,
        )
        self.ax_sliders = self.fig.add_axes(
            [0.0, 0.0, 1.0, 1.0],
            label="sliders_container"
        )
        self.ax_sliders.set_visible(False)

        self._gs_right = gs_right   # save for later

    def _style_ax(self, ax, title, ylabel, ylim):
        ax.set_facecolor(C["panel"])
        for spine in ax.spines.values():
            spine.set_color(C["grid"])
            spine.set_linewidth(0.8)
        ax.tick_params(colors=C["subtext"], labelsize=8)
        ax.set_xlabel("Log-strike  k", color=C["subtext"], fontsize=9)
        ax.set_ylabel(ylabel, color=C["subtext"], fontsize=9)
        ax.set_title(title, color=C["text"], fontsize=10, fontweight="bold", pad=6)
        ax.set_xlim(self.K_RANGE[0], self.K_RANGE[-1])
        ax.set_ylim(*ylim)
        ax.axvline(0, color=C["grid"], linewidth=0.8, linestyle="--", zorder=0)
        ax.grid(True, color=C["grid"], linewidth=0.5, zorder=0)

    # ──────────────────────────────────────────────────────────────────────
    # Plot objects
    # ──────────────────────────────────────────────────────────────────────

    def _build_plots(self):
        self._style_ax(self.ax_var, "Total Implied Variance  w(k)",
                       "w(k) = σ²·T", self.W_YLIM)
        self._style_ax(self.ax_vol, "Implied Volatility  σ_BS(k)",
                       "σ_BS(k)", self.PLOT_YLIM)

        k = self.K_RANGE
        w0 = raw_svi_total_variance(k, **DEFAULT_RAW)
        s0 = np.sqrt(np.maximum(w0, 1e-10))

        self.line_var, = self.ax_var.plot(k, w0, color=C["raw_line"],
                                          linewidth=2.0, zorder=3)
        self.line_vol, = self.ax_vol.plot(k, s0, color=C["raw_line"],
                                          linewidth=2.0, zorder=3)

        # ATM dot
        self.atm_var = self.ax_var.scatter([0], [w0[len(k)//2]],
                                           color=C["accent2"], s=40, zorder=5)
        self.atm_vol = self.ax_vol.scatter([0], [s0[len(k)//2]],
                                           color=C["accent2"], s=40, zorder=5)

        # ── RND overlay on the vol plot (secondary y-axis) ──────────────────
        self.ax_rnd = self.ax_vol.twinx()
        self.ax_rnd.set_ylabel("Risk-neutral density  p(k)",
                               color=C["rnd_fill"], fontsize=8, labelpad=4)
        self.ax_rnd.tick_params(axis="y", colors=C["rnd_fill"], labelsize=7)
        self.ax_rnd.set_zorder(self.ax_vol.get_zorder() - 1)
        self.ax_vol.patch.set_visible(False)   # let RND axis background show

        # Compute initial RND
        rnd0, g0 = compute_rnd(k, **DEFAULT_RAW)

        # Filled area: positive part (valid) in amber, negative part in red
        rnd_pos = np.where(rnd0 >= 0, rnd0, 0.0)
        rnd_neg = np.where(rnd0 <  0, rnd0, 0.0)

        self.rnd_fill_pos = self.ax_rnd.fill_between(
            k, 0, rnd_pos,
            color=C["rnd_fill"], alpha=0.18, zorder=1, linewidth=0
        )
        self.rnd_fill_neg = self.ax_rnd.fill_between(
            k, 0, rnd_neg,
            color=C["rnd_neg"], alpha=0.45, zorder=1, linewidth=0
        )
        self.rnd_line, = self.ax_rnd.plot(
            k, rnd0,
            color=C["rnd_fill"], linewidth=0.9, alpha=0.55, zorder=2
        )

        # Axis limits — will be rescaled in _update
        rnd_max = max(np.nanmax(rnd_pos) * 1.3, 0.1)
        self.ax_rnd.set_ylim(-rnd_max * 0.15, rnd_max)

        # Small legend entry
        from matplotlib.patches import Patch
        rnd_patch = Patch(facecolor=C["rnd_fill"], alpha=0.4, label="RND  p(k)")
        self.ax_vol.legend(handles=[rnd_patch], loc="upper right",
                           fontsize=7.5, framealpha=0.7,
                           edgecolor=C["grid"], facecolor=C["panel"])

        # Warning text
        self.warn_text = self.ax_vol.text(
            0.01, 0.97, "", transform=self.ax_vol.transAxes,
            color="#DC2626", fontsize=7.5, va="top", ha="left",
            wrap=True, zorder=10
        )

    # ──────────────────────────────────────────────────────────────────────
    # Slider factory
    # ──────────────────────────────────────────────────────────────────────

    def _make_slider(self, ax, label, vmin, vmax, vinit, color):
        sl = Slider(
            ax, label, vmin, vmax,
            valinit=vinit,
            color=color,
            track_color=C["slider_bg"],
        )
        sl.label.set_fontsize(9)
        sl.label.set_color(C["text"])
        sl.valtext.set_fontsize(8.5)
        sl.valtext.set_color(C["subtext"])
        sl.on_changed(lambda val: self._update())
        return sl

    def _build_sliders(self):
        """Create three sets of sliders (Raw, Natural, JW) and show only one."""
        self._slider_axes   = {"Raw": [], "Natural": [], "JW": []}
        self._sliders       = {"Raw": [], "Natural": [], "JW": []}
        self._slider_specs  = {
            "Raw":     self.SLIDERS_RAW,
            "Natural": self.SLIDERS_NAT,
            "JW":      self.SLIDERS_JW,
        }

        # Right panel bounding box (in figure coordinates) for sliders
        gs = self._gs_right
        r0 = gs[0].get_position(self.fig)   # top sub-panel
        r1 = gs[1].get_position(self.fig)   # mid sub-panel

        left   = r0.x0 + 0.01
        width  = r0.width - 0.02
        top    = r0.y1 - 0.01
        total_h = (r0.y1 - r1.y0) - 0.03   # total height for 5+1 sliders

        N = 6   # 5 param sliders + 1 T slider
        row_h   = total_h / N
        pad     = row_h * 0.18
        sl_h    = row_h - 2 * pad

        param_colors = [
            C["accent"], "#7C3AED", "#DB2777", "#D97706", "#16A34A"
        ]

        for mode, specs in self._slider_specs.items():
            for i, (lbl, vmin, vmax, vinit, _vstep) in enumerate(specs):
                y = top - (i + 1) * row_h + pad
                ax = self.fig.add_axes([left, y, width, sl_h])
                ax.set_facecolor(C["bg"])
                col = param_colors[i % len(param_colors)]
                sl  = self._make_slider(ax, lbl, vmin, vmax, vinit, col)
                self._slider_axes[mode].append(ax)
                self._sliders[mode].append(sl)

        # T-slider (only visible in JW mode) — placed just below param sliders
        t_y = top - N * row_h + pad
        self._ax_t = self.fig.add_axes([left, t_y, width, sl_h])
        self._sl_t = self._make_slider(self._ax_t, "T (expiry)", 0.05, 5.0,
                                       DEFAULT_T, "#64748B")

        self._show_sliders(self.mode)

    def _show_sliders(self, mode):
        """Show only the relevant set of slider axes."""
        for m, axlist in self._slider_axes.items():
            vis = (m == mode)
            for ax in axlist:
                ax.set_visible(vis)
        self._ax_t.set_visible(mode == "JW")

    # ──────────────────────────────────────────────────────────────────────
    # Raw parameter readout panel
    # ──────────────────────────────────────────────────────────────────────

    def _build_raw_readout(self):
        gs = self._gs_right
        r2 = gs[2].get_position(self.fig)
        ax = self.fig.add_axes([r2.x0 + 0.01, r2.y0 + 0.01,
                                 r2.width - 0.02, r2.height - 0.02])
        ax.set_facecolor(C["panel"])
        for sp in ax.spines.values():
            sp.set_color(C["grid"])
            sp.set_linewidth(0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Raw SVI parameters", fontsize=8.5,
                     color=C["subtext"], pad=3)
        self._ax_readout = ax

        names  = ["a", "b", "ρ", "m", "σ"]
        colors = [C["accent"], "#7C3AED", "#DB2777", "#D97706", "#16A34A"]
        self._readout_texts = []
        for i, (nm, col) in enumerate(zip(names, colors)):
            x = 0.05 + (i % 5) * 0.19
            y = 0.42
            self._ax_readout.text(x, y + 0.35, nm, transform=ax.transAxes,
                                  color=col, fontsize=9, fontweight="bold",
                                  ha="center")
            t = ax.text(x, y, "—", transform=ax.transAxes,
                        color=col, fontsize=8.5, ha="center")
            self._readout_texts.append(t)

    def _update_readout(self, a, b, rho, m, sigma):
        vals = [a, b, rho, m, sigma]
        for t, v in zip(self._readout_texts, vals):
            t.set_text(f"{v:.4f}")

    # ──────────────────────────────────────────────────────────────────────
    # Mode selector (RadioButtons)
    # ──────────────────────────────────────────────────────────────────────

    def _build_mode_selector(self):
        ax = self.fig.add_axes([0.04, 0.01, 0.18, 0.07])
        ax.set_facecolor(C["bg"])
        self._radio = RadioButtons(
            ax,
            labels=("Raw SVI", "Natural SVI", "SVI-JW"),
            activecolor=C["accent"],
        )
        for label in self._radio.labels:
            label.set_fontsize(9)
            label.set_color(C["text"])
        self._radio.on_clicked(self._on_mode_change)

    def _on_mode_change(self, label):
        mode_map = {"Raw SVI": "Raw", "Natural SVI": "Natural", "SVI-JW": "JW"}
        self.mode = mode_map[label]
        self._show_sliders(self.mode)
        self._update()
        self.fig.canvas.draw_idle()

    # ──────────────────────────────────────────────────────────────────────
    # Reset button
    # ──────────────────────────────────────────────────────────────────────

    def _build_reset_button(self):
        ax = self.fig.add_axes([0.24, 0.015, 0.07, 0.05])
        self._btn_reset = Button(ax, "Reset", color=C["slider_bg"],
                                 hovercolor=C["shadow"])
        self._btn_reset.label.set_fontsize(8.5)
        self._btn_reset.label.set_color(C["text"])
        self._btn_reset.on_clicked(self._reset)

    def _reset(self, event=None):
        specs = self._slider_specs[self.mode]
        for sl, spec in zip(self._sliders[self.mode], specs):
            sl.set_val(spec[2])
        self._sl_t.set_val(DEFAULT_T)

    # ──────────────────────────────────────────────────────────────────────
    # Main update loop
    # ──────────────────────────────────────────────────────────────────────

    def _get_raw_params(self):
        """Read sliders and return (a, b, rho, m, sigma) + warning string."""
        warning = ""
        sls = self._sliders[self.mode]

        if self.mode == "Raw":
            a, b, rho, m, sigma = [s.val for s in sls]

        elif self.mode == "Natural":
            Delta, mu, rho, omega, zeta = [s.val for s in sls]
            a, b, rho, m, sigma = natural_to_raw(Delta, mu, rho, omega, zeta)

        elif self.mode == "JW":
            v_t, psi_t, p_t, c_t, v_tilde_t = [s.val for s in sls]
            self.t = self._sl_t.val
            a, b, rho, m, sigma, warning = jw_to_raw(
                v_t, psi_t, p_t, c_t, v_tilde_t, self.t
            )
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        # Hard clamp to valid domain
        b     = max(b, 0.0)
        sigma = max(sigma, 1e-4)
        rho   = np.clip(rho, -0.9999, 0.9999)

        return a, b, rho, m, sigma, warning

    def _update(self, val=None):
        try:
            a, b, rho, m, sigma, warning = self._get_raw_params()
        except Exception as e:
            self.warn_text.set_text(f"Computation error: {e}")
            self.fig.canvas.draw_idle()
            return

        k  = self.K_RANGE
        w  = raw_svi_total_variance(k, a, b, rho, m, sigma)
        iv = np.sqrt(np.maximum(w, 1e-10))

        self.line_var.set_ydata(w)
        self.line_vol.set_ydata(iv)

        # ATM value (k=0)
        w_atm  = raw_svi_total_variance(np.array([0.0]), a, b, rho, m, sigma)[0]
        iv_atm = np.sqrt(max(w_atm, 1e-10))
        self.atm_var.set_offsets([[0.0, w_atm]])
        self.atm_vol.set_offsets([[0.0, iv_atm]])

        # Auto-rescale y axes to data (with some headroom)
        w_max  = np.nanmax(w)
        iv_max = np.nanmax(iv)
        if np.isfinite(w_max):
            self.ax_var.set_ylim(0, max(w_max * 1.15, 0.02))
        if np.isfinite(iv_max):
            self.ax_vol.set_ylim(0, max(iv_max * 1.15, 0.05))

        # Colour the curve by mode
        color_map = {"Raw": C["raw_line"], "Natural": C["nat_line"], "JW": C["jw_line"]}
        col = color_map[self.mode]
        self.line_var.set_color(col)
        self.line_vol.set_color(col)

        self._update_readout(a, b, rho, m, sigma)
        self.warn_text.set_text(warning)

        # ── RND update ───────────────────────────────────────────────────────
        rnd, g = compute_rnd(k, a, b, rho, m, sigma)

        # Remove old fills (PolyCollections can't be set_data'd)
        self.rnd_fill_pos.remove()
        self.rnd_fill_neg.remove()

        rnd_pos = np.where(rnd >= 0, rnd, 0.0)
        rnd_neg_arr = np.where(rnd < 0, rnd, 0.0)

        self.rnd_fill_pos = self.ax_rnd.fill_between(
            k, 0, rnd_pos,
            color=C["rnd_fill"], alpha=0.18, zorder=1, linewidth=0
        )
        self.rnd_fill_neg = self.ax_rnd.fill_between(
            k, 0, rnd_neg_arr,
            color=C["rnd_neg"], alpha=0.45, zorder=1, linewidth=0
        )
        self.rnd_line.set_ydata(rnd)

        # Rescale RND axis
        rnd_max = np.nanmax(rnd_pos)
        rnd_min = np.nanmin(rnd_neg_arr)
        if np.isfinite(rnd_max) and rnd_max > 0:
            top_lim = rnd_max * 1.30
            bot_lim = min(rnd_min * 1.3, -top_lim * 0.05)
            self.ax_rnd.set_ylim(bot_lim, top_lim)

        # Append arbitrage warning if g goes negative anywhere
        if np.any(g < 0):
            arb_warn = "\n⚠  g(k) < 0 detected — butterfly arbitrage present."
            self.warn_text.set_text(warning + arb_warn)
            self.rnd_line.set_color(C["rnd_neg"])
        else:
            self.rnd_line.set_color(C["rnd_fill"])

        # Update title with ATM vol
        self.ax_vol.set_title(
            f"Implied Volatility  σ_BS(k)      [ATM σ = {iv_atm:.3f}]",
            color=C["text"], fontsize=10, fontweight="bold", pad=6
        )

        self.fig.canvas.draw_idle()

    # ──────────────────────────────────────────────────────────────────────

    def run(self):
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = SVIExplorer()
    app.run()
