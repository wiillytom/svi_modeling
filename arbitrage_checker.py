"""
arbitrage_checker.py
====================
Static-arbitrage diagnostics for calibrated implied-volatility surfaces.

A total-variance surface w(k, t) = σ_BS(k, t)² · t is free of static
arbitrage iff two pointwise conditions hold across the whole (k, t) domain:

1.  Calendar-spread no-arbitrage  —  Gatheral & Jacquier (2014), Lemma 2.1
    ──────────────────────────────────────────────────────────────────────
        ∂w(k, t) / ∂t  ≥  0    for every (k, t)

    Total implied variance must be non-decreasing in maturity at every
    log-strike.  A calendar spread (long longer-dated, short shorter-dated
    at the same strike) must have non-negative value at inception; violating
    this gives a riskless arbitrage between the two maturities.

2.  Butterfly no-arbitrage  —  Gatheral & Jacquier (2014), Lemma 2.2
    ──────────────────────────────────────────────────────────────────
    The risk-neutral density implied by w(·, t) is non-negative iff

        g(k, t)  ≡  (1 − k w′ / (2 w))²
                  − (w′)² / 4 · (1/w + 1/4)
                  + w″ / 2
                  ≥  0      for every (k, t),

    where w′ = ∂w/∂k and w″ = ∂²w/∂k².  This is the Breeden-Litzenberger
    condition (Breeden & Litzenberger 1978) translated into total-variance
    space.  When g(k, t) < 0 the implied density is locally negative,
    which would let a long butterfly position generate a riskless profit.

What this script does
---------------------
For a calibrated result dict (as produced by `calibrate_snapshot`,
`calibrate_global_essvi`, `calibrate_global_ssvi`, …) it:

    • builds a dense (k, t) grid with k ∈ [-2, 2] and t interpolated
      between the fitted expiries using a monotone cubic spline
      (PCHIP — preserves local monotonicity, no oscillation),
    • evaluates the surface and its first/second derivatives by
      central finite differences with steps dk = 1e-3, dt = 5e-4,
    • computes g(k, t) on the grid,
    • flags every cell where dw/dt < −1e-6 (calendar) or g < −1e-6
      (butterfly),
    • returns a structured summary and offers a 2×2 visualisation.

References
----------
Gatheral, J. and A. Jacquier (2014), "Arbitrage-free SVI volatility
    surfaces", Quantitative Finance 14(1), 59-71.
Breeden, D. T. and R. H. Litzenberger (1978), "Prices of state-contingent
    claims implicit in option prices", Journal of Business 51(4), 621-651.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_TOL = 1.0e-6     # numerical tolerance for "violation" (absorbs FD noise)
_DK  = 1.0e-3     # FD step in log-strike
_DT  = 5.0e-4     # FD step in maturity


# ─────────────────────────────────────────────────────────────────────────────
# CORE: BUILD SURFACE AND DERIVATIVES ON A DENSE (k, t) GRID
# ─────────────────────────────────────────────────────────────────────────────

def _build_surface(result, n_strikes, n_times, dk=_DK, dt=_DT):
    """
    Evaluate the calibrated surface and its first/second derivatives on a
    dense (k, t) grid.

    Time interpolation
    ------------------
    For each fitted expiry t_j the model gives w(·, t_j) exactly.  Between
    expiries we interpolate w along t at each fixed k using a monotone cubic
    spline (PCHIP), which preserves local monotonicity of the data and does
    not introduce spurious oscillations between fitted points.  This is the
    interpolation rule recommended in Gatheral-Jacquier (2014, §4) for
    surface assembly.

    Finite differences
    ------------------
        ∂w/∂t  : central FD using the PCHIP interpolator at  t ± dt
        ∂w/∂k  : central FD using the model itself at        k ± dk
                  (re-interpolated across t)
        ∂²w/∂k² : central FD on the same three values

    Off-grid evaluation of the model at k ± dk and re-interpolation in t
    keeps the derivative consistent with the way the surface was
    constructed.

    Returns
    -------
    k_grid : (n_strikes,) log-strike grid in [-2, 2]
    t_grid : (n_times,)   maturity grid in [min(expiries), max(expiries)]
    W      : (n_strikes, n_times) total variance
    dwdt   : (n_strikes, n_times) ∂w/∂t
    dwdk   : (n_strikes, n_times) ∂w/∂k
    d2wdk2 : (n_strikes, n_times) ∂²w/∂k²
    """
    model       = result["_model"]
    expiries    = np.asarray(result["expiries"], dtype=float)
    params_list = list(result["params"])

    if len(expiries) < 2:
        raise ValueError(
            f"check_arbitrage needs at least 2 fitted expiries to interpolate "
            f"the calendar dimension, got {len(expiries)}."
        )

    # Defensive sort — the result dict is supposed to be sorted already
    if np.any(np.diff(expiries) <= 0):
        order        = np.argsort(expiries)
        expiries     = expiries[order]
        params_list  = [params_list[i] for i in order]

    k_grid = np.linspace(-2.0,  2.0, n_strikes)
    t_grid = np.linspace(expiries.min(), expiries.max(), n_times)

    # Evaluate w on three log-strike grids at every fitted expiry
    k_plus  = k_grid + dk
    k_minus = k_grid - dk

    n_e = len(expiries)
    W_mid_e = np.empty((n_strikes, n_e))
    W_p_e   = np.empty((n_strikes, n_e))
    W_m_e   = np.empty((n_strikes, n_e))
    for j, p in enumerate(params_list):
        W_mid_e[:, j] = model.w(k_grid,  p)
        W_p_e  [:, j] = model.w(k_plus,  p)
        W_m_e  [:, j] = model.w(k_minus, p)

    # PCHIP across maturities, evaluated on the dense t grid.
    # axis=0 means "interpolate along the first axis of the y-array",
    # which is the expiries axis once we transpose to shape (n_e, n_strikes).
    interp_mid   = PchipInterpolator(expiries, W_mid_e.T, axis=0, extrapolate=True)
    interp_plus  = PchipInterpolator(expiries, W_p_e  .T, axis=0, extrapolate=True)
    interp_minus = PchipInterpolator(expiries, W_m_e  .T, axis=0, extrapolate=True)

    W   = interp_mid  (t_grid).T            # at (k_grid,  t_grid)
    Wp  = interp_plus (t_grid).T            # at (k_grid+dk, t_grid)
    Wm  = interp_minus(t_grid).T            # at (k_grid-dk, t_grid)
    Wtp = interp_mid  (t_grid + dt).T       # at (k_grid,  t_grid+dt)
    Wtm = interp_mid  (t_grid - dt).T       # at (k_grid,  t_grid-dt)

    dwdt   = (Wtp - Wtm) / (2.0 * dt)
    dwdk   = (Wp  - Wm ) / (2.0 * dk)
    d2wdk2 = (Wp - 2.0 * W + Wm) / (dk * dk)

    return k_grid, t_grid, W, dwdt, dwdk, d2wdk2


def _gatheral_g(W, dwdk, d2wdk2, k_grid):
    """
    Gatheral-Jacquier (2014, eq. 2.4) function whose positivity is
    equivalent to the Breeden-Litzenberger non-negative density condition:

        g(k, t) = (1 − k w′ / (2 w))²
                − (w′)² / 4  · (1/w + 1/4)
                + w″ / 2

    Returns g of the same shape as W.
    """
    k_col  = k_grid[:, np.newaxis]
    safe_W = np.maximum(W, 1.0e-12)
    term1 = (1.0 - k_col * dwdk / (2.0 * safe_W)) ** 2
    term2 = -(dwdk ** 2) * 0.25 * (1.0 / safe_W + 0.25)
    term3 = 0.5 * d2wdk2
    return term1 + term2 + term3


def _violation_stats(field, k_grid, t_grid, tol):
    """
    Common diagnostic block for a 2-D scalar `field` whose positivity is
    the no-arb condition.  A violation is field < -tol.

    Returns (is_free, worst_value, worst_k, worst_t, violation_pct).
    """
    mask    = field < -tol
    n_total = field.size

    # Worst point = global minimum (the most violating, or least-positive)
    idx_min = int(np.nanargmin(field))
    idx_2d  = np.unravel_index(idx_min, field.shape)
    worst_v = float(field[idx_2d])
    worst_k = float(k_grid[idx_2d[0]])
    worst_t = float(t_grid[idx_2d[1]])

    pct  = 100.0 * float(mask.sum()) / float(n_total)
    free = bool(not mask.any())
    return free, worst_v, worst_k, worst_t, pct, mask


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def check_arbitrage(result: dict,
                    n_strikes: int = 500,
                    n_times:   int = 200) -> dict:
    """
    Check a calibrated surface for static arbitrage on a dense (k, t) grid.

    Conditions
    ----------
    Calendar : ∂w/∂t ≥ 0           (GJ 2014, Lemma 2.1)
    Butterfly: g(k, t)  ≥ 0       (GJ 2014, Lemma 2.2)

    A small numerical tolerance of −1e-6 is used so finite-difference
    discretisation noise alone does not flag a violation.

    Parameters
    ----------
    result    : dict with keys '_model', 'expiries', 'params' (as produced
                by calibrate_snapshot / calibrate_global_essvi /
                calibrate_global_ssvi).
    n_strikes : log-strike grid size on [-2, 2]                (default 500)
    n_times   : maturity grid size on [t_min, t_max]           (default 200)

    Returns
    -------
    dict with the following keys:
        calendar_free            : bool
        butterfly_free           : bool
        calendar_worst           : float  — minimum ∂w/∂t found
        calendar_worst_k         : float  — log-strike of the minimum
        calendar_worst_t         : float  — maturity of the minimum
        calendar_violation_pct   : float  — % of grid points where ∂w/∂t < −tol
        butterfly_worst          : float  — minimum g found
        butterfly_worst_k        : float
        butterfly_worst_t        : float
        butterfly_violation_pct  : float
        summary                  : str    — human-readable summary
    """
    k_grid, t_grid, W, dwdt, dwdk, d2wdk2 = _build_surface(
        result, n_strikes, n_times,
    )
    g = _gatheral_g(W, dwdk, d2wdk2, k_grid)

    cal_free, cal_w, cal_k, cal_t, cal_pct, _ = _violation_stats(
        dwdt, k_grid, t_grid, _TOL,
    )
    but_free, but_w, but_k, but_t, but_pct, _ = _violation_stats(
        g,    k_grid, t_grid, _TOL,
    )

    name    = result.get("model_name", "?")
    cal_tag = "PASS" if cal_free else "FAIL"
    but_tag = "PASS" if but_free else "FAIL"
    bar     = "─" * 78
    lines = [
        bar,
        f"  Static-arbitrage check — {name}",
        f"  Grid: {n_strikes} strikes × {n_times} maturities   "
        f"(k ∈ [-2, 2],  t ∈ [{t_grid[0]:.4f}, {t_grid[-1]:.4f}])",
        f"  FD steps: dk={_DK}  dt={_DT}     tolerance={_TOL}",
        bar,
        f"  Calendar  [{cal_tag}]  worst ∂w/∂t = {cal_w:+.3e}  "
        f"@ (k={cal_k:+.3f}, t={cal_t:.4f})  "
        f"violations: {cal_pct:.3f}% of grid",
        f"  Butterfly [{but_tag}]  worst g      = {but_w:+.3e}  "
        f"@ (k={but_k:+.3f}, t={but_t:.4f})  "
        f"violations: {but_pct:.3f}% of grid",
        bar,
    ]
    summary = "\n".join(lines)

    return {
        "calendar_free":           cal_free,
        "butterfly_free":          but_free,
        "calendar_worst":          cal_w,
        "calendar_worst_k":        cal_k,
        "calendar_worst_t":        cal_t,
        "calendar_violation_pct":  cal_pct,
        "butterfly_worst":         but_w,
        "butterfly_worst_k":       but_k,
        "butterfly_worst_t":       but_t,
        "butterfly_violation_pct": but_pct,
        "summary":                 summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_arbitrage_map(result: dict,
                       n_strikes: int = 500,
                       n_times:   int = 200):
    """
    2×2 visual diagnostic for the static-arbitrage check.

        [0, 0]  heatmap of w(k, t)
        [0, 1]  heatmap of ∂w/∂t,  calendar violations overlaid in red
        [1, 0]  heatmap of g(k, t), butterfly violations overlaid in red
        [1, 1]  bar chart of violation % per maturity slice
                (calendar in orange, butterfly in red)

    Parameters
    ----------
    result    : calibration result dict
    n_strikes : log-strike grid size on [-2, 2]           (default 500)
    n_times   : maturity grid size on [t_min, t_max]      (default 200)

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    k_grid, t_grid, W, dwdt, dwdk, d2wdk2 = _build_surface(
        result, n_strikes, n_times,
    )
    g = _gatheral_g(W, dwdk, d2wdk2, k_grid)

    cal_mask = dwdt < -_TOL
    but_mask = g    < -_TOL
    cal_pct  = 100.0 * float(cal_mask.sum()) / float(cal_mask.size)
    but_pct  = 100.0 * float(but_mask.sum()) / float(but_mask.size)

    expiries = np.asarray(result["expiries"], dtype=float)
    name     = result.get("model_name", "?")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Static-arbitrage map — {name}    "
        f"calendar {cal_pct:.2f}%   butterfly {but_pct:.2f}%",
        fontsize=13,
    )

    # ── [0,0] total variance w(k, t) ──────────────────────────────────────────
    ax = axes[0, 0]
    im = ax.pcolormesh(t_grid, k_grid, W, cmap="viridis", shading="auto")
    plt.colorbar(im, ax=ax, label="w")
    for t_fit in expiries:
        ax.axvline(t_fit, color="white", lw=0.4, alpha=0.4)
    ax.set_xlabel("t (maturity)")
    ax.set_ylabel("k (log-strike)")
    ax.set_title("Total variance  w(k, t)\nwhite lines = fitted expiries")

    # ── [0,1] ∂w/∂t with calendar violations ─────────────────────────────────
    ax = axes[0, 1]
    abs_max = max(abs(np.nanmin(dwdt)), abs(np.nanmax(dwdt)), 1e-12)
    im = ax.pcolormesh(t_grid, k_grid, dwdt,
                       cmap="RdBu", vmin=-abs_max, vmax=abs_max, shading="auto")
    plt.colorbar(im, ax=ax, label="∂w/∂t")
    if cal_mask.any():
        ax.contourf(t_grid, k_grid, cal_mask.astype(float),
                    levels=[0.5, 1.5], colors=["red"], alpha=0.45)
    ax.set_xlabel("t (maturity)")
    ax.set_ylabel("k (log-strike)")
    ax.set_title(
        f"∂w/∂t   —   calendar violations in red  ({cal_pct:.3f}%)"
    )

    # ── [1,0] g(k, t) with butterfly violations ──────────────────────────────
    ax = axes[1, 0]
    abs_max = max(abs(np.nanmin(g)), abs(np.nanmax(g)), 1e-12)
    im = ax.pcolormesh(t_grid, k_grid, g,
                       cmap="RdBu", vmin=-abs_max, vmax=abs_max, shading="auto")
    plt.colorbar(im, ax=ax, label="g")
    if but_mask.any():
        ax.contourf(t_grid, k_grid, but_mask.astype(float),
                    levels=[0.5, 1.5], colors=["red"], alpha=0.45)
    ax.set_xlabel("t (maturity)")
    ax.set_ylabel("k (log-strike)")
    ax.set_title(
        f"Gatheral g(k, t)  —  butterfly violations in red  ({but_pct:.3f}%)"
    )

    # ── [1,1] violation % per maturity slice ─────────────────────────────────
    ax = axes[1, 1]
    cal_per_t = 100.0 * cal_mask.sum(axis=0) / cal_mask.shape[0]
    but_per_t = 100.0 * but_mask.sum(axis=0) / but_mask.shape[0]

    t_step = t_grid[1] - t_grid[0] if len(t_grid) > 1 else 0.01
    bar_w  = t_step * 0.4
    ax.bar(t_grid - bar_w / 2, cal_per_t, width=bar_w,
           color="tab:orange", label="Calendar  (∂w/∂t < 0)", edgecolor="none")
    ax.bar(t_grid + bar_w / 2, but_per_t, width=bar_w,
           color="tab:red",    label="Butterfly (g < 0)",     edgecolor="none")
    ax.set_xlabel("t (maturity)")
    ax.set_ylabel("% of strikes in violation")
    ax.set_title("Violation density per maturity slice")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    y_top = max(cal_per_t.max(), but_per_t.max(), 1.0) * 1.15
    ax.set_ylim(0, y_top)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    return fig
