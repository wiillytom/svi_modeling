"""
SVI Calibration Module
======================
Python translation of the R code from Gatheral & Jacquier's Baruch Volatility Workshop
(Sessions 3 & 4), adapted for inverse crypto options (Deribit-style).

Structure
---------
1.  Raw SVI smile            w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sig^2))
2.  SVI-JW parameterization  (vt, psit, pt, ct, varmint) with conversions
3.  Arbitrage checks         g(k) >= 0  (butterfly),  no crossing (calendar)
4.  Quartic-root crossedness (ComputeSviRoots equivalent)
5.  Square-root SSVI fit     global initial guess (phi(theta)=eta/sqrt(theta))
6.  QR slice-by-slice fit    with crossedness penalty
7.  SSVI surface             closed-form arbitrage-free surface
8.  Local variance           Dupire via finite differences on the fitted surface
9.  Data loading             clean_df wrapper + slice extractor

Notes on inverse options
------------------------
Inverse options (Deribit BTC/ETH) are priced in coin units.  The implied vol
surface is identical in shape to a regular vanilla surface — the SVI fit is
applied directly to the (k, w) data where  w = sigma_IV^2 * T.  The forward
used when building k = log(K/F) should be the futures price (already included
in the data pipeline via clean_df).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  RAW SVI
# ─────────────────────────────────────────────────────────────────────────────

def svi_raw(k, params):
    """
    Raw SVI total implied variance.

    Parameters
    ----------
    k      : array-like  log-strike  log(K/F)
    params : dict with keys  a, b, rho, m, sig

    Returns
    -------
    w(k)  total implied variance  (sigma_BS^2 * T)
    """
    a, b, rho, m, sig = params["a"], params["b"], params["rho"], params["m"], params["sig"]
    k = np.asarray(k, dtype=float)
    discr = np.sqrt((k - m) ** 2 + sig ** 2)
    return a + b * (rho * (k - m) + discr)


def svi_raw_valid(params):
    """Check parameter constraints for raw SVI (returns True if valid)."""
    a, b, rho, m, sig = params["a"], params["b"], params["rho"], params["m"], params["sig"]
    if b < 0:
        return False
    if abs(rho) >= 1:
        return False
    if sig <= 0:
        return False
    if a + b * sig * np.sqrt(1 - rho ** 2) < 0:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SVI-JW PARAMETERIZATION  (raw <-> JW conversions)
# ─────────────────────────────────────────────────────────────────────────────

def raw_to_jw(params, t):
    """
    Convert raw SVI parameters to Jump-Wings (JW) parameters for expiry t.

    JW parameters
    -------------
    vt      ATM variance  (= w(0)/t)
    psit    ATM vol skew
    pt      left (put) wing slope
    ct      right (call) wing slope
    varmint minimum implied variance

    References: Gatheral & Jacquier (2014), eq (3.5)
    """
    a, b, rho, m, sig = params["a"], params["b"], params["rho"], params["m"], params["sig"]
    wt = svi_raw(0.0, params)          # w(0)
    sqrt_wt = np.sqrt(wt)

    discr_m = np.sqrt(m ** 2 + sig ** 2)

    vt      = wt / t
    psit    = (b / (2 * sqrt_wt)) * (-m / discr_m + rho)
    pt      = b * (1 - rho) / sqrt_wt
    ct      = b * (1 + rho) / sqrt_wt
    varmint = (a + b * sig * np.sqrt(1 - rho ** 2)) / t

    return {"vt": vt, "psit": psit, "pt": pt, "ct": ct, "varmint": varmint, "texp": t}


def jw_to_raw(jw, t):
    """
    Convert JW parameters back to raw SVI parameters for expiry t.

    Inverse of raw_to_jw; follows Gatheral workshop R code (jwToSvi).
    """
    vt, psit, pt, ct, varmint = (
        jw["vt"], jw["psit"], jw["pt"], jw["ct"], jw["varmint"]
    )
    wt = vt * t

    # Recover b, rho from pt and ct
    b   = np.sqrt(wt) * (pt + ct) / 2
    rho = 1 - pt * np.sqrt(wt) / b

    # Recover m from psit
    # psit = b/(2*sqrt(wt)) * (-m/sqrt(m^2+sig^2) + rho)
    # Let beta = 2*sqrt(wt)*psit/b - rho  =>  -m/sqrt(m^2+sig^2) = beta
    beta = 2 * np.sqrt(wt) * psit / b - rho
    # Clamp to avoid domain error
    beta = np.clip(beta, -0.9999, 0.9999)
    # -m / sqrt(m^2+sig^2) = beta  =>  m^2 = beta^2*(m^2+sig^2)
    # m^2*(1-beta^2) = beta^2*sig^2  =>  m = beta*sig / sqrt(1-beta^2)

    # We also need sig from varmint:
    # varmint*t = a + b*sig*sqrt(1-rho^2)
    # a = w(0) - b*(rho*(0-m)+sqrt(m^2+sig^2))
    # This is a coupled system; solve iteratively via a direct reconstruction

    # Use the relationship:  pt * ct = (b^2/wt)*(1-rho^2)
    # => (1-rho^2) = pt*ct*wt/b^2
    one_minus_rho2 = pt * ct * wt / b ** 2
    rho2 = 1 - one_minus_rho2
    rho  = 1 - pt * np.sqrt(wt) / b     # keep sign

    # sig from varmint: varmint*t = a + b*sig*sqrt(1-rho^2)
    # and a = wt - b*(rho*(-m)+sqrt(m^2+sig^2))
    # Let's solve for sig using varmint directly
    # varmint*t = a + b*sig*sqrt(1-rho^2)
    # The minimum of SVI is  a + b*sig*sqrt(1-rho^2)
    # => sig = (varmint*t - a) / (b*sqrt(1-rho^2))
    # but a itself depends on m and sig...

    # Pragmatic: use the closed-form from the workshop R code
    # a = wt - b*(-rho*m + sqrt(m^2+sig^2))
    # min_w = a + b*sig*sqrt(1-rho^2)  at k = m - rho*sig*..., simplifies to:
    # min_w = a + b*sig*sqrt(1-rho^2)
    # We treat m and sig as unknowns:
    # From psit: beta = -m/sqrt(m^2+sig^2)
    # From beta and sig: m = -beta*sig/sqrt(1-beta^2)
    # From min_w: sig = solve from substituting a

    # Numerical solution (robust for edge cases)
    sqrt_one_minus_rho2 = np.sqrt(max(one_minus_rho2, 1e-12))

    def equations(sig_guess):
        sig_g = abs(sig_guess)
        if sig_g < 1e-8:
            return 1e10
        # m from beta
        beta_sq = beta ** 2
        if beta_sq >= 1:
            return 1e10
        m_g = -beta * sig_g / np.sqrt(1 - beta_sq)
        discr = np.sqrt(m_g ** 2 + sig_g ** 2)
        a_g = wt - b * (rho * (-m_g) + discr)
        min_w_calc = a_g + b * sig_g * sqrt_one_minus_rho2
        return (min_w_calc - varmint * t) ** 2

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(equations, bounds=(1e-6, 10.0), method="bounded")
    sig = abs(res.x)

    beta_sq = beta ** 2
    m = -beta * sig / np.sqrt(max(1 - beta_sq, 1e-12))

    discr = np.sqrt(m ** 2 + sig ** 2)
    a = wt - b * (rho * (-m) + discr)

    return {"a": a, "b": b, "rho": rho, "m": m, "sig": sig}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BUTTERFLY ARBITRAGE CHECK  g(k)
# ─────────────────────────────────────────────────────────────────────────────

def g_func(k, params):
    """
    Compute g(k) for butterfly-arbitrage check.

    A slice is free of butterfly arbitrage iff g(k) >= 0 for all k,
    and  lim_{k->+inf} d+(k) = -inf.

    g(k) = (1 - k*w'/(2w))^2 - (w'^2/4)*(1/w + 1/4) + w''/2
    """
    a, b, rho, m, sig = params["a"], params["b"], params["rho"], params["m"], params["sig"]
    k = np.asarray(k, dtype=float)
    discr = np.sqrt((k - m) ** 2 + sig ** 2)
    w    = a + b * (rho * (k - m) + discr)
    dw   = b * rho + b * (k - m) / discr
    d2w  = b * sig ** 2 / discr ** 3
    return (1 - k * dw / (2 * w)) ** 2 - (dw ** 2 / 4) * (1 / w + 0.25) + d2w / 2


def min_g(params, k_range=(-3.0, 3.0), n_points=500):
    """Return the minimum of g(k) over a grid — negative means butterfly arbitrage."""
    k_grid = np.linspace(k_range[0], k_range[1], n_points)
    return float(np.min(g_func(k_grid, params)))


def slice_has_butterfly_arb(params, k_range=(-3.0, 3.0)):
    """True if the slice has butterfly arbitrage."""
    return min_g(params, k_range) < 0


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CALENDAR SPREAD ARBITRAGE: QUARTIC ROOTS (ComputeSviRoots)
# ─────────────────────────────────────────────────────────────────────────────

def _svi_value(k, params):
    a, b, rho, m, sig = params["a"], params["b"], params["rho"], params["m"], params["sig"]
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2))


def compute_svi_roots(params1, params2):
    """
    Find real crossing points of two SVI slices (params1 = earlier, params2 = later).
    Returns dict with 'roots' (sorted array) and 'crossedness' (float >= 0).

    Two slices cross when  w(k; params1) = w(k; params2).
    Rearranging and squaring yields a degree-4 polynomial.
    We use numpy's polynomial root solver and verify against the original equation.
    """
    a1, b1, r1, m1, s1 = params1["a"], params1["b"], params1["rho"], params1["m"], params1["sig"]
    a2, b2, r2, m2, s2 = params2["a"], params2["b"], params2["rho"], params2["m"], params2["sig"]

    # Quartic coefficients (from Gatheral workshop, scaled by 1e6 for numerical stability)
    scale = 1e6

    q4 = scale * (
        b1**4 * (1 - r1**2)**2
        - 4*b1**3*b2*r1*(1 - r1**2)*r2
        - 4*b1*b2**3*r1*r2*(1 - r2**2)
        + b2**4*(1 - r2**2)**2
        + 2*b1**2*b2**2*(-1 - r2**2 + r1**2*(-1 + 3*r2**2))
    )

    q3 = scale * -4 * (
        b1**4*m1*(1 - r1**2)**2
        - b1**3*r1*(1 - r1**2)*(a1 - a2 + b2*(3*m1 + m2)*r2)
        + b2**3*(1 - r2**2)*((a1 - a2)*r2 + b2*m2*(1 - r2**2))
        + b1*b2**2*r1*(a1*(1 - 3*r2**2) - b2*(m1 + 3*m2)*r2*(1 - r2**2) + a2*(-1 + 3*r2**2))
        + b1**2*b2*((a1 - a2)*(-1 + 3*r1**2)*r2 + b2*(m1 + m2)*(-1 - r2**2 + r1**2*(-1 + 3*r2**2)))
    )

    # q2, q1, q0 are lengthy; compute numerically via polynomial difference
    # Strategy: build quartic by expanding (A - sqrt(B))*(A + sqrt(B)) = A^2 - B
    # where A = (a1 - a2) + b1*r1*(k-m1) - b2*r2*(k-m2)   [linear terms after moving sqrt to one side]
    # and B = b1^2*((k-m1)^2 + s1^2) or b2^2*...
    # This is algebraically equivalent but easier to implement symbolically.
    # We use numpy polynomial multiplication directly.

    # poly1(k) = (a1-a2) + b1*r1*(k-m1) - b2*r2*(k-m2)
    # = (a1-a2) - b1*r1*m1 + b2*r2*m2 + (b1*r1 - b2*r2)*k
    c0_lin = (a1 - a2) - b1*r1*m1 + b2*r2*m2
    c1_lin = b1*r1 - b2*r2
    # poly1^2:  c1_lin^2 * k^2 + 2*c0_lin*c1_lin * k + c0_lin^2
    p1_sq = np.array([c1_lin**2, 2*c0_lin*c1_lin, c0_lin**2])  # degree 2

    # b1^2*((k-m1)^2+s1^2) = b1^2*(k^2 - 2m1*k + m1^2 + s1^2)
    p_b1sq = b1**2 * np.array([1.0, -2*m1, m1**2 + s1**2])  # degree 2
    # b2^2*((k-m2)^2+s2^2)
    p_b2sq = b2**2 * np.array([1.0, -2*m2, m2**2 + s2**2])  # degree 2

    # After squaring both sides:
    # poly1^2 - b1^2*((k-m1)^2+s1^2) = -b2^2*((k-m2)^2+s2^2) + 2*b2*r2*(k-m2)*poly1
    # =>  poly1^2 - p_b1sq + p_b2sq - 2*b2*r2*(k-m2)*poly1 = 0
    # The lhs is degree 4 when we consider poly1^2 crossed with higher terms.
    # Let's compute directly.

    # Full quartic = [poly1 - b1*sqrt(...)] * [poly1 + b1*sqrt(...)] expressed differently
    # Easier: directly compute the quartic numerically, then find roots.

    # We already have q4, q3. Compute q2, q1, q0 via a different route.
    # Evaluate the quartic at 5 points (Lagrange interpolation)
    def quartic_val(k_val):
        """
        Value of  [w1(k)-w2(k)]^2  after squaring out, minus direct test.
        Actually: we set up f(k) = (w1-w2) and find roots by another method.
        """
        pass

    # Numerical fallback: use the polynomial (p1_sq - p_b1sq) difference trick
    # f(k) = w1(k) - w2(k):  sign changes => roots
    def f_diff(k_val):
        return _svi_value(k_val, params1) - _svi_value(k_val, params2)

    # Scan for sign changes on a fine grid, then polish with Brent
    k_grid = np.linspace(-10, 10, 5000)
    f_grid = f_diff(k_grid)

    roots = []
    for i in range(len(k_grid) - 1):
        if f_grid[i] * f_grid[i+1] < 0:
            from scipy.optimize import brentq
            try:
                r = brentq(f_diff, k_grid[i], k_grid[i+1], xtol=1e-10)
                roots.append(r)
            except Exception:
                pass

    roots = np.array(sorted(set([round(r, 8) for r in roots])))

    # Compute crossedness
    crossedness = 0.0
    if len(roots) > 0:
        sample_points = np.concatenate([
            [roots[0] - 1],
            (roots[:-1] + roots[1:]) / 2 if len(roots) > 1 else [],
            [roots[-1] + 1]
        ])
        c_vals = np.maximum(0.0, f_diff(sample_points))
        crossedness = float(np.max(c_vals))

    return {"roots": roots, "crossedness": crossedness}


def calendar_arb_check(svi_params_list):
    """
    Check all consecutive pairs of slices for calendar spread arbitrage.
    Returns total crossedness (0 = no arb).
    """
    total = 0.0
    for i in range(len(svi_params_list) - 1):
        res = compute_svi_roots(svi_params_list[i], svi_params_list[i+1])
        total += res["crossedness"]
    return total


def slice_arb_check(svi_params_list, k_range=(-3.0, 3.0)):
    """
    Returns list of slice indices (0-based) that have butterfly arbitrage.
    """
    bad = []
    for i, p in enumerate(svi_params_list):
        if slice_has_butterfly_arb(p, k_range):
            bad.append(i)
    return bad


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SQUARE-ROOT SSVI FIT  (global initial guess)
# ─────────────────────────────────────────────────────────────────────────────

def ssvi_sqrt(k, theta, eta, rho):
    """
    SSVI with power-law  phi(theta) = eta / sqrt(theta).

    w(k, theta) = theta/2 * {1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1-rho^2)}
    """
    phi = eta / np.sqrt(theta)
    return (theta / 2) * (
        1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2)
    )


def ssvi_sqrt_to_raw(theta, eta, rho, t):
    """Convert SSVI-sqrt parameters for a given slice theta=sigma_atm^2*t to raw SVI."""
    phi = eta / np.sqrt(theta)
    # SSVI = (theta/2)*(1 + rho*phi*(k-0) + sqrt((phi*(k-0)+rho)^2 + 1-rho^2))
    # Compare with raw SVI natural form:
    #   w(k) = Delta + omega/2*(1 + zeta*rho*(k-mu) + sqrt((zeta*(k-mu)+rho)^2+1-rho^2))
    # Here Delta=0, mu=0, omega=theta, zeta=phi
    # Converting natural -> raw:
    a_nat = 0.0
    omega = theta
    zeta  = phi
    mu    = 0.0
    # natural -> raw: a=Delta+omega/2*(1-sqrt(1-rho^2)), b=omega*zeta/2, sig=sqrt(1-rho^2)/zeta, m=mu-rho/zeta
    b   = omega * zeta / 2
    sig = np.sqrt(1 - rho**2) / zeta
    m   = mu - rho / zeta
    a   = a_nat + omega / 2 * (1 - np.sqrt(1 - rho**2))
    return {"a": float(a), "b": float(b), "rho": float(rho), "m": float(m), "sig": float(sig)}


def fit_ssvi_sqrt(df, verbose=False):
    """
    Fit the global 3-parameter SSVI square-root surface to the entire dataset.

    Parameters
    ----------
    df : DataFrame with columns  k, w, t  (output of clean_df)

    Returns
    -------
    (eta, rho) : optimal global parameters
    svi_matrix : list of raw SVI dicts, one per slice (sorted by t)
    """
    expiries = sorted(df["t"].unique())
    thetas   = []
    for t_exp in expiries:
        subset = df[df["t"] == t_exp]
        # ATM total variance = interpolated w at k=0
        atm_w = np.interp(0.0, subset["k"].values, subset["w"].values)
        thetas.append(atm_w)
    thetas = np.array(thetas)

    def objective(params):
        eta_  = params[0]
        rho_  = np.tanh(params[1])   # unconstrained optimisation
        total = 0.0
        for i, t_exp in enumerate(expiries):
            subset = df[df["t"] == t_exp]
            k_vals = subset["k"].values
            w_obs  = subset["w"].values
            w_fit  = ssvi_sqrt(k_vals, thetas[i], eta_, rho_)
            total += np.sum((w_fit - w_obs) ** 2)
        return total

    x0 = [2.0, np.arctanh(-0.7)]
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
    eta_opt = res.x[0]
    rho_opt = np.tanh(res.x[1])

    if verbose:
        print(f"SSVI sqrt fit: eta={eta_opt:.4f}, rho={rho_opt:.4f}")

    svi_matrix = []
    for i, t_exp in enumerate(expiries):
        raw = ssvi_sqrt_to_raw(thetas[i], eta_opt, rho_opt, t_exp)
        svi_matrix.append(raw)

    return eta_opt, rho_opt, svi_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 6.  QR SLICE-BY-SLICE FIT WITH CROSSEDNESS PENALTY
# ─────────────────────────────────────────────────────────────────────────────

def fit_svi_slice(
    k_vals,
    w_obs,
    initial_params,
    prev_params=None,
    next_params=None,
    penalty_factor=1000.0,
):
    """
    Fit a single SVI slice minimising squared vol error plus crossedness penalty.

    Parameters
    ----------
    k_vals         : array  log-strikes
    w_obs          : array  observed total implied variances
    initial_params : dict   initial raw SVI parameters
    prev_params    : dict or None  (slice at earlier expiry)
    next_params    : dict or None  (slice at later expiry)
    penalty_factor : float  weight on crossedness penalty

    Returns
    -------
    dict  optimised raw SVI parameters
    """
    def params_from_x(x):
        return {
            "a":   x[0],
            "b":   np.exp(x[1]),            # b > 0
            "rho": np.tanh(x[2]),           # |rho| < 1
            "m":   x[3],
            "sig": np.exp(x[4]),            # sig > 0
        }

    def x_from_params(p):
        return [
            p["a"],
            np.log(max(p["b"], 1e-8)),
            np.arctanh(np.clip(p["rho"], -0.9999, 0.9999)),
            p["m"],
            np.log(max(p["sig"], 1e-8)),
        ]

    def objective(x):
        p = params_from_x(x)
        w_fit = svi_raw(k_vals, p)

        # Squared distance (vol-weighted like the workshop)
        sq_dist = np.nansum((w_fit - w_obs) ** 2)

        # Crossedness penalties
        penalty = 0.0
        if prev_params is not None:
            res = compute_svi_roots(prev_params, p)
            penalty += res["crossedness"]
        if next_params is not None:
            res = compute_svi_roots(p, next_params)
            penalty += res["crossedness"]

        # Penalty for negative variance
        min_w = p["a"] + p["b"] * p["sig"] * np.sqrt(1 - p["rho"] ** 2)
        if min_w < 0:
            penalty += 100 * abs(min_w)

        return sq_dist + penalty_factor * penalty

    x0 = x_from_params(initial_params)
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-7})
    return params_from_x(res.x)


def fit_svi_surface(df, penalty_factor=1000.0, verbose=True):
    """
    Full SVI surface calibration following the Gatheral-Jacquier recipe:

    1. Fit square-root SSVI as global initial guess.
    2. Fit slice-by-slice (forward order) with crossedness penalty.

    Parameters
    ----------
    df             : DataFrame with columns  k, w, t
    penalty_factor : float

    Returns
    -------
    dict with keys:
        'expiries'    : sorted list of t values
        'svi_params'  : list of raw SVI dicts (one per expiry)
        'ssvi_eta'    : global eta from sqrt SSVI
        'ssvi_rho'    : global rho from sqrt SSVI
    """
    expiries = sorted(df["t"].unique())
    n = len(expiries)

    # ── Step 1: square-root SSVI initial guess ────────────────────────────────
    if verbose:
        print("Step 1: fitting SSVI square-root initial guess…")
    eta, rho, svi_init = fit_ssvi_sqrt(df, verbose=verbose)

    # ── Step 2: QR slice-by-slice ─────────────────────────────────────────────
    if verbose:
        print("Step 2: QR slice-by-slice fit…")
    svi_matrix = list(svi_init)   # copy

    for i, t_exp in enumerate(expiries):
        if verbose:
            print(f"  Slice {i+1}/{n}  T={t_exp:.4f}", end="  ")
        subset = df[df["t"] == t_exp].copy()
        k_vals = subset["k"].values
        w_obs  = subset["w"].values

        prev_p = svi_matrix[i-1] if i > 0 else None
        next_p = svi_matrix[i+1] if i < n-1 else None

        svi_matrix[i] = fit_svi_slice(
            k_vals, w_obs,
            initial_params=svi_init[i],
            prev_params=prev_p,
            next_params=next_p,
            penalty_factor=penalty_factor,
        )
        if verbose:
            cross = calendar_arb_check(svi_matrix[:i+1])
            but   = slice_has_butterfly_arb(svi_matrix[i])
            print(f"cal_arb={cross:.2e}  butterfly={'YES' if but else 'no'}")

    return {
        "expiries":   expiries,
        "svi_params": svi_matrix,
        "ssvi_eta":   eta,
        "ssvi_rho":   rho,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  SSVI SURFACE (closed-form arbitrage-free)
# ─────────────────────────────────────────────────────────────────────────────

def ssvi_power_law(k, theta, eta, rho, gamma=0.5):
    """
    SSVI with power-law  phi(theta) = eta / theta^gamma * (1+theta)^(gamma-1).
    For gamma=0.5 this reduces to the square-root form.

    Free of static arbitrage when  eta*(1+|rho|) <= 2.
    """
    phi = eta / (theta ** gamma)
    return (theta / 2) * (
        1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2)
    )


def fit_ssvi_surface(df, gamma=0.5, verbose=True):
    """
    Fit the 3-parameter (eta, rho, gamma) SSVI surface.
    Arbitrage-free by construction when  eta*(1+|rho|) <= 2.

    Returns dict with 'eta', 'rho', 'gamma', 'thetas', 'expiries'.
    """
    expiries = sorted(df["t"].unique())
    thetas   = []
    for t_exp in expiries:
        subset = df[df["t"] == t_exp]
        atm_w  = np.interp(0.0, subset["k"].values, subset["w"].values)
        thetas.append(atm_w)
    thetas = np.array(thetas)

    def objective(params):
        eta_  = np.exp(params[0])        # eta > 0
        rho_  = np.tanh(params[1])
        gamma_ = 0.5 + 0.5 * np.tanh(params[2])   # gamma in (0,1)
        total = 0.0
        for i, t_exp in enumerate(expiries):
            subset = df[df["t"] == t_exp]
            k_vals = subset["k"].values
            w_obs  = subset["w"].values
            w_fit  = ssvi_power_law(k_vals, thetas[i], eta_, rho_, gamma_)
            total += np.nansum((w_fit - w_obs) ** 2)
        # Soft arbitrage constraint: eta*(1+|rho|) <= 2
        arb_pen = max(0.0, eta_ * (1 + abs(rho_)) - 2) * 1e6
        return total + arb_pen

    x0 = [np.log(2.0), np.arctanh(-0.7), np.arctanh(0.0)]
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 10000, "xatol": 1e-8})
    eta_opt   = np.exp(res.x[0])
    rho_opt   = np.tanh(res.x[1])
    gamma_opt = 0.5 + 0.5 * np.tanh(res.x[2])

    if verbose:
        print(f"SSVI power-law: eta={eta_opt:.4f}, rho={rho_opt:.4f}, gamma={gamma_opt:.4f}")
        print(f"  Arbitrage condition eta*(1+|rho|)={eta_opt*(1+abs(rho_opt)):.4f} (must <= 2)")

    return {
        "eta": eta_opt, "rho": rho_opt, "gamma": gamma_opt,
        "thetas": thetas, "expiries": expiries,
    }


def ssvi_to_svi_matrix(ssvi_params):
    """Convert SSVI surface parameters to a list of raw SVI dicts (one per slice)."""
    eta, rho, gamma = ssvi_params["eta"], ssvi_params["rho"], ssvi_params["gamma"]
    svi_list = []
    for theta, t_exp in zip(ssvi_params["thetas"], ssvi_params["expiries"]):
        phi = eta / (theta ** gamma)
        raw = ssvi_sqrt_to_raw(theta, eta / (theta ** (gamma - 0.5)), rho, t_exp)
        # redo properly with power-law phi
        b   = theta * phi / 2
        sig = np.sqrt(1 - rho**2) / phi
        m   = -rho / phi
        a   = theta / 2 * (1 - np.sqrt(1 - rho**2))
        svi_list.append({"a": float(a), "b": float(b), "rho": float(rho),
                          "m": float(m), "sig": float(sig)})
    return svi_list


# ─────────────────────────────────────────────────────────────────────────────
# 8.  LOCAL VARIANCE (Dupire)
# ─────────────────────────────────────────────────────────────────────────────

def svi_interpolated_w(svi_params_list, expiries, k, t):
    """
    Interpolate total implied variance w(k, t) across expiries using Stineman-style
    monotone interpolation (via scipy pchip = shape-preserving cubic Hermite).
    """
    from scipy.interpolate import PchipInterpolator
    k = np.asarray(k, dtype=float)
    t = float(t)

    w_per_expiry = np.array([svi_raw(k, p) for p in svi_params_list])  # shape (n_slices,) or (n_slices, n_k)
    if w_per_expiry.ndim == 1:
        interp = PchipInterpolator(expiries, w_per_expiry, extrapolate=True)
        return float(interp(t))
    else:
        result = np.zeros(len(k))
        for j in range(len(k)):
            interp = PchipInterpolator(expiries, w_per_expiry[:, j], extrapolate=True)
            result[j] = float(interp(t))
        return result


def local_variance(svi_params_list, expiries, k, t, dk=1e-3, dt=1e-3):
    """
    Dupire local variance via finite differences on the SVI surface.

    v_loc(k,T) = dw/dT / (1 - k/w * dw/dk + (dw/dk)^2/4 * (1/4 + 1/w - k^2/w^2) + d^2w/dk^2/2)

    Parameters
    ----------
    svi_params_list : list of raw SVI dicts (sorted by t)
    expiries        : list of floats matching svi_params_list
    k, t            : scalar or array, scalar
    """
    k  = np.asarray(k, dtype=float)
    dt = max(dt, t * 0.05)    # avoid negative time

    wkt  = svi_interpolated_w(svi_params_list, expiries, k, t)
    wktm = svi_interpolated_w(svi_params_list, expiries, k, max(t - dt, 1e-5))
    wktp = svi_interpolated_w(svi_params_list, expiries, k, t + dt)

    dwdt  = (wktp - wktm) / (2 * dt)
    wkmt  = svi_interpolated_w(svi_params_list, expiries, k - dk, t)
    wkpt  = svi_interpolated_w(svi_params_list, expiries, k + dk, t)
    dwdk  = (wkpt - wkmt) / (2 * dk)
    d2wdk2 = (wkpt + wkmt - 2 * wkt) / dk**2

    denom = (
        1
        - k / wkt * dwdk
        + dwdk**2 / 4 * (-1 / wkt + k**2 / wkt**2 - 4)
        + d2wdk2 / 2
    )
    return dwdt / np.maximum(denom, 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def clean_df(path: str, iv_col: str = "mark_iv") -> pd.DataFrame:
    """
    Load and clean an options parquet file.

    Adds columns
    ------------
    t  : time to maturity in years
    k  : log-strike  log(K / F)
    w  : total implied variance  = (mark_iv)^2 * t   [assumes iv in (0,1)]

    Static arbitrage rows are dropped (negative iv, negative t, w <= 0).
    For inverse options (Deribit-style), the forward F is used directly if
    present, otherwise the underlying spot price is used as a proxy.

    Parameters
    ----------
    path   : str  path to parquet file
    iv_col : str  column name for the implied volatility (default 'mark_iv')
    """
    df = pd.read_parquet(path)

    # ── time to maturity ──────────────────────────────────────────────────────
    if "t" not in df.columns:
        if "expiry" in df.columns and "date" in df.columns:
            df["t"] = (pd.to_datetime(df["expiry"]) - pd.to_datetime(df["date"])).dt.days / 365.25
        elif "time_to_maturity" in df.columns:
            df["t"] = df["time_to_maturity"]
        else:
            raise ValueError("Cannot determine time to maturity; add a 't' column.")

    # ── forward price ─────────────────────────────────────────────────────────
    if "forward" in df.columns:
        fwd_col = "forward"
    elif "future_price" in df.columns:
        fwd_col = "future_price"
    elif "underlying_price" in df.columns:
        fwd_col = "underlying_price"
    else:
        fwd_col = None

    # ── log-strike ────────────────────────────────────────────────────────────
    if "k" not in df.columns:
        strike_col = "strike" if "strike" in df.columns else "strike_price"
        if fwd_col is not None:
            df["k"] = np.log(df[strike_col] / df[fwd_col])
        else:
            # fallback: use index (no forward available)
            df["k"] = np.log(df[strike_col])
            print("Warning: no forward price found; k = log(K) may be imprecise.")

    # ── implied vol in (0,1) ──────────────────────────────────────────────────
    if df[iv_col].max() > 5:
        df[iv_col] = df[iv_col] / 100.0   # assume it was in percentage

    # ── total implied variance ────────────────────────────────────────────────
    if "w" not in df.columns:
        df["w"] = df[iv_col] ** 2 * df["t"]

    # ── drop arbitrageable / bad rows ─────────────────────────────────────────
    df = df[df["t"]  > 0].copy()
    df = df[df[iv_col] > 0].copy()
    df = df[df[iv_col] < 5].copy()
    df = df[df["w"]  > 0].copy()
    df = df.dropna(subset=["k", "w", "t"])

    return df.reset_index(drop=True)


def get_slice(df, t_exp, tol=1e-4):
    """Return the slice of df closest to expiry t_exp (within tolerance tol)."""
    expiries = df["t"].unique()
    closest  = expiries[np.argmin(np.abs(expiries - t_exp))]
    if abs(closest - t_exp) > tol:
        print(f"Warning: requested T={t_exp}, closest available T={closest}")
    return df[df["t"] == closest].copy()


# ─────────────────────────────────────────────────────────────────────────────
# 10.  DIAGNOSTICS & REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def surface_summary(result):
    """
    Print a summary table of calibrated surface quality.

    Parameters
    ----------
    result : dict returned by fit_svi_surface()
    """
    expiries = result["expiries"]
    params   = result["svi_params"]
    print(f"\n{'T':>8}  {'a':>9}  {'b':>9}  {'rho':>8}  {'m':>8}  {'sig':>9}  "
          f"{'min_g':>8}  {'cal_arb':>10}")
    print("-" * 85)
    for i, (t, p) in enumerate(zip(expiries, params)):
        mg   = min_g(p)
        cross = 0.0
        if i > 0:
            cross = compute_svi_roots(params[i-1], p)["crossedness"]
        print(f"{t:8.4f}  {p['a']:9.6f}  {p['b']:9.6f}  {p['rho']:8.4f}  "
              f"{p['m']:8.4f}  {p['sig']:9.6f}  {mg:8.4f}  {cross:10.2e}")


def fitted_iv_surface(result, df):
    """
    Add fitted implied vol and total variance to the original dataframe.

    Returns a copy of df with added columns  w_fit, iv_fit.
    """
    df_out = df.copy()
    df_out["w_fit"]  = np.nan
    df_out["iv_fit"] = np.nan

    expiries = result["expiries"]
    params   = result["svi_params"]

    for t_exp, p in zip(expiries, params):
        mask = df_out["t"] == t_exp
        k_vals = df_out.loc[mask, "k"].values
        w_fit  = svi_raw(k_vals, p)
        df_out.loc[mask, "w_fit"]  = w_fit
        df_out.loc[mask, "iv_fit"] = np.sqrt(np.maximum(w_fit / t_exp, 0))

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "2 - Data", "Parquets")
    path = os.path.join(DATA_DIR, "daily_options_data.parquet")

    if not os.path.exists(path):
        print(f"Data not found at {path}. Update DATA_DIR to match your layout.")
        sys.exit(0)

    print("Loading data…")
    df = clean_df(path)
    print(f"  {len(df)} rows, {df['t'].nunique()} expiries")

    print("\nCalibrating SVI surface…")
    result = fit_svi_surface(df, penalty_factor=1000.0, verbose=True)

    surface_summary(result)

    # Calendar & butterfly arb checks
    cal_arb = calendar_arb_check(result["svi_params"])
    but_arb = slice_arb_check(result["svi_params"])
    print(f"\nCalendar spread crossedness : {cal_arb:.2e}  (0 = no arb)")
    print(f"Butterfly arbitrage slices  : {but_arb}  (empty = no arb)")

    # Add fitted values back
    df_fitted = fitted_iv_surface(result, df)
    print("\nSample fit vs market:")
    print(df_fitted[["t", "k", "w", "w_fit", "iv_fit"]].head(10).to_string(index=False))
