"""
Interactive SVI / SVI-JW volatility smile explorer.

Run with:
    streamlit run volatility_surface/streamlit_svi.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


PURPLE = "#6A0DAD"
LIGHT_PURPLE = "#B284E0"
PASTEL_PURPLE = "#C9A0DC"


def raw_svi_w(k, a, b, rho, m, sigma):
    """Gatheral raw SVI total variance."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def raw_svi_density(k, a, b, rho, m, sigma):
    """Gatheral risk-neutral density g(k) / sqrt(2 pi w) * exp(-d_-^2/2)."""
    discr = np.sqrt((k - m) ** 2 + sigma ** 2)
    w = np.maximum(a + b * (rho * (k - m) + discr), 1e-10)
    dw = b * rho + b * (k - m) / discr
    d2w = b * sigma ** 2 / discr ** 3
    g = (1 - k * dw / (2 * w)) ** 2 - (dw ** 2 / 4) * (1 / w + 0.25) + d2w / 2
    d_minus = -k / np.sqrt(w) - np.sqrt(w) / 2
    return g / np.sqrt(2 * np.pi * w) * np.exp(-d_minus ** 2 / 2), g


SIGMA_MAX = 5.0


def jw_to_raw(v_t, psi_t, p_t, c_t, v_tilde_t, T):
    """Convert SVI-JW parameters (at maturity T) to raw SVI (a, b, rho, m, sigma).

    Returns (a, b, rho, m, sigma, warning) where `warning` is None or a string
    describing why the JW inputs are mutually inconsistent.
    """
    w_t = v_t * T
    sqrt_w = np.sqrt(w_t)

    b = sqrt_w * (c_t + p_t) / 2.0
    rho = (c_t - p_t) / (c_t + p_t)

    alpha = rho - 2.0 * psi_t * sqrt_w / b
    alpha_c = float(np.clip(alpha, -0.9999, 0.9999))
    one_minus_alpha_sq = 1.0 - alpha_c ** 2

    factor = (1.0 - rho * alpha_c) / np.sqrt(one_minus_alpha_sq) - np.sqrt(max(1.0 - rho ** 2, 1e-12))
    rhs = w_t - v_tilde_t * T

    warning = None
    if abs(b * factor) < 1e-10:
        if abs(rhs) < 1e-6:
            sigma = 0.1
        else:
            sigma = SIGMA_MAX
            warning = (
                "JW inputs are degenerate: with ρ≈0 and ψ_t≈0 the smile must satisfy v_t = ṽ_t, "
                f"but v_t·T − ṽ_t·T = {rhs:+.4f}. σ has been capped at {SIGMA_MAX}."
            )
    else:
        sigma = rhs / (b * factor)
        if sigma <= 0 or sigma > SIGMA_MAX:
            warning = (
                f"JW inputs imply σ = {sigma:.2f}, which is outside [1e-4, {SIGMA_MAX}]. "
                "Capped — the smile shown is the closest representable SVI slice, not your exact JW inputs."
            )
            sigma = float(np.clip(sigma, 1e-4, SIGMA_MAX))

    m = sigma * alpha_c / np.sqrt(one_minus_alpha_sq)
    a = v_tilde_t * T - b * sigma * np.sqrt(max(1.0 - rho ** 2, 1e-12))

    return a, b, rho, m, sigma, warning


def make_figure(k, iv, density, g, params_text, y_max=2.0):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    ax.set_facecolor("white")

    ax_d = ax.twinx()
    ax_d.fill_between(k, np.maximum(density, 0.0), color=PASTEL_PURPLE, alpha=0.45,
                      label="Risk-neutral density", zorder=1)
    ax_d.plot(k, density, color=PASTEL_PURPLE, lw=1.6, zorder=1)
    neg = g < 0
    if neg.any():
        ax_d.fill_between(k, density, where=neg, color="#E07A7A", alpha=0.45,
                          label="Butterfly arb (g < 0)", zorder=2)
    d_top = max(float(np.nanmax(density)) * 1.15, 1e-6)
    ax_d.set_ylim(0.0, d_top)
    ax_d.set_ylabel("Risk-neutral density  $q(k)$", fontsize=11, color=PASTEL_PURPLE)
    ax_d.tick_params(axis="y", colors=PASTEL_PURPLE)
    ax_d.spines["right"].set_color(PASTEL_PURPLE)
    ax_d.grid(False)

    ax.plot(k, iv, color=PURPLE, lw=2.4, label="Implied volatility", zorder=3)
    ax.axvline(0.0, color="#999999", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("Log-moneyness  k = log(K/F)", fontsize=11)
    ax.set_ylabel("Implied volatility  $\\sigma_{BS}(k)$", fontsize=11, color=PURPLE)
    ax.set_title("SVI volatility smile  &  risk-neutral density", fontsize=13, color=PURPLE)
    ax.set_ylim(0.0, y_max)
    ax.grid(True, alpha=0.3, color=LIGHT_PURPLE)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    ax.tick_params(colors="#444444")
    ax.tick_params(axis="y", colors=PURPLE)
    ax.set_zorder(ax_d.get_zorder() + 1)
    ax.patch.set_visible(False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_d.get_legend_handles_labels()
    if lines1 or lines2:
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc="upper right", frameon=False, fontsize=9)
    return fig


def make_variance_figure(k, w, params_text, y_max=1.5):
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(k, w, color=PURPLE, lw=2.4, label="Total variance w(k)")
    ax.axvline(0.0, color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("Log-moneyness  k", fontsize=11)
    ax.set_ylabel("Total variance  $w(k) = \\sigma^2 T$", fontsize=11)
    ax.set_title("SVI total variance", fontsize=13, color=PURPLE)
    ax.set_ylim(0.0, y_max)
    ax.grid(True, alpha=0.3, color=LIGHT_PURPLE)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    ax.tick_params(colors="#444444")
    return fig


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="SVI Explorer", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #6A0DAD; }
    .stSlider [data-baseweb="slider"] > div > div > div { background: #6A0DAD; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("SVI volatility smile explorer")
st.caption("Drag the sliders to see how each parameter shapes the smile.")

param_mode = st.sidebar.radio(
    "Parametrization",
    ("Raw SVI", "SVI Jump-Wings"),
)

st.sidebar.markdown("### Maturity & range")
T = st.sidebar.slider(
    "Time to maturity  T  (years)",
    min_value=0.01, max_value=2.0, value=0.25, step=0.01,
)
k_max = st.sidebar.slider(
    "Log-moneyness range  ±k",
    min_value=0.2, max_value=3.0, value=1.0, step=0.05,
)
iv_y_max = st.sidebar.slider(
    "IV  y-axis max",
    min_value=0.2, max_value=4.0, value=2.0, step=0.1,
)
show_variance = st.sidebar.checkbox("Also show total variance w(k)", value=False)

k = np.linspace(-k_max, k_max, 401)

if param_mode == "Raw SVI":
    st.subheader("Raw SVI parameters")
    st.latex(r"w(k) = a + b\,\bigl[\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2}\,\bigr]")

    st.sidebar.markdown("### Raw SVI parameters")
    a = st.sidebar.slider("a  (vertical shift)", -0.05, 0.30, 0.04, step=0.001, format="%.3f")
    b = st.sidebar.slider("b  (wing slope)", 0.0, 2.0, 0.40, step=0.01)
    rho = st.sidebar.slider("ρ  (skew, in (-1,1))", -0.999, 0.999, -0.60, step=0.01)
    m = st.sidebar.slider("m  (smile center)", -1.0, 1.0, 0.0, step=0.01)
    sigma_p = st.sidebar.slider("σ  (smoothness, >0)", 0.001, 2.0, 0.20, step=0.001, format="%.3f")

    w = raw_svi_w(k, a, b, rho, m, sigma_p)
    w_pos = np.maximum(w, 1e-10)
    iv = np.sqrt(w_pos / T)

    min_w = a + b * sigma_p * np.sqrt(max(1 - rho ** 2, 0.0))
    valid = (b >= 0) and (abs(rho) < 1) and (sigma_p > 0) and (min_w >= 0)
    params_text = (
        f"a   = {a:+.4f}\n"
        f"b   = {b:+.4f}\n"
        f"rho = {rho:+.4f}\n"
        f"m   = {m:+.4f}\n"
        f"sig = {sigma_p:+.4f}\n"
        f"T   = {T:.3f}\n"
        f"min w = {min_w:+.4f}  "
        f"{'(ok)' if valid else '(NEG! arb)'}"
    )

else:
    st.subheader("SVI Jump-Wings parameters  (Gatheral)")
    st.markdown(
        "All parameters have a direct market interpretation at a fixed maturity $T$:"
    )
    st.latex(r"""
    \begin{aligned}
    v_t            &= \sigma^2_{\text{ATM}} \quad \text{(ATM variance)} \\
    \psi_t         &= \tfrac{1}{2\sqrt{w_t}}\,\partial_k w(0) \quad \text{(ATM skew)} \\
    p_t            &= -\sqrt{w_t}\,\text{left wing slope} \\
    c_t            &= +\sqrt{w_t}\,\text{right wing slope} \\
    \tilde v_t     &= \min_k w(k)/T \quad \text{(minimum variance)}
    \end{aligned}
    """)

    st.sidebar.markdown("### SVI-JW parameters")
    v_t = st.sidebar.slider("v_t   (ATM variance)", 0.001, 1.0, 0.30, step=0.001, format="%.3f")
    psi_t = st.sidebar.slider("ψ_t   (ATM skew)", -2.0, 2.0, -0.20, step=0.01)
    p_t = st.sidebar.slider("p_t   (put wing slope)", 0.01, 4.0, 0.80, step=0.01)
    c_t = st.sidebar.slider("c_t   (call wing slope)", 0.01, 4.0, 0.30, step=0.01)
    v_tilde = st.sidebar.slider("ṽ_t   (min variance)", 0.0, 1.0, 0.24, step=0.001, format="%.3f")

    a, b, rho, m, sigma_p, jw_warning = jw_to_raw(v_t, psi_t, p_t, c_t, v_tilde, T)
    if jw_warning:
        st.warning(jw_warning)
    w = raw_svi_w(k, a, b, rho, m, sigma_p)
    w_pos = np.maximum(w, 1e-10)
    iv = np.sqrt(w_pos / T)

    min_w = a + b * sigma_p * np.sqrt(max(1 - rho ** 2, 0.0))
    valid = (b >= 0) and (abs(rho) < 1) and (sigma_p > 0) and (min_w >= 0)
    params_text = (
        f"-- JW --\n"
        f"v_t     = {v_t:+.4f}\n"
        f"psi_t   = {psi_t:+.4f}\n"
        f"p_t     = {p_t:+.4f}\n"
        f"c_t     = {c_t:+.4f}\n"
        f"v_tilde = {v_tilde:+.4f}\n"
        f"-- raw --\n"
        f"a={a:+.3f}  b={b:+.3f}  rho={rho:+.3f}\n"
        f"m={m:+.3f}  sig={sigma_p:+.3f}\n"
        f"{'(ok)' if valid else '(arb!)'}"
    )

density, g_vals = raw_svi_density(k, a, b, rho, m, sigma_p)
fig = make_figure(k, iv, density, g_vals, params_text, y_max=iv_y_max)
st.pyplot(fig, clear_figure=True)

if show_variance:
    fig2 = make_variance_figure(k, w, params_text, y_max=iv_y_max ** 2 * T)
    st.pyplot(fig2, clear_figure=True)

with st.expander("Reference  —  Gatheral SVI cheatsheet"):
    st.markdown(
        """
- **Raw SVI** (Gatheral 2004):
  $w(k) = a + b\\,[\\rho(k-m) + \\sqrt{(k-m)^2+\\sigma^2}]$
  Wing slopes: left $= b(1-\\rho)$, right $= b(1+\\rho)$.
  No-static-arb (slice level): $b\\ge 0$, $|\\rho|<1$, $\\sigma>0$,
  $a + b\\sigma\\sqrt{1-\\rho^2} \\ge 0$.

- **SVI-JW** (Gatheral–Jacquier 2014): re-parametrization with market-interpretable
  quantities at maturity $T$. The slice is the same — only the inputs change.
  Conversion uses $w_t = v_t T$:
  $b = \\tfrac{\\sqrt{w_t}}{2}(c_t + p_t)$,
  $\\rho = (c_t-p_t)/(c_t+p_t)$.
        """
    )
