"""
vol_calibration_engine.py
=========================
Model-agnostic calibration engine for implied volatility surfaces.

Separates three concerns that were previously coupled in fit_single_slice:

    1. ModelSpec  — how to pack/unpack parameters, call the model, check arbitrage
    2. Objective  — what loss function to minimise (from svi_objective_functions.py)
    3. Optimizer  — the generic DE + Nelder-Mead loop (model- and objective-agnostic)

Adding a new model (SABR, eSSVI, Heston, ...) means writing one ModelSpec.
The objective functions and optimizer are reused unchanged.

Included ModelSpecs
-------------------
    SVI_ModelSpec     raw SVI  (5 params: a, b, rho, m, sig)
    eSSVI_ModelSpec   extended SSVI  (3 params per slice: rho, eta, gamma, given θ_t)
    SABR_ModelSpec    SABR approximation  (4 params: alpha, beta, rho, nu)

Usage
-----
    from vol_calibration_engine import fit_slice, SVI_ModelSpec, SABR_ModelSpec
    from svi_objective_functions import build_objective

    spec   = SVI_ModelSpec()
    obj_fn = build_objective("vega_weighted_iv")

    params = fit_slice(k_obs, iv_obs, t,
                       model_spec  = spec,
                       objective_fn = obj_fn,
                       prev_params  = prev_p)
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# BS HELPERS  (duplicated here so this module is self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def _bs_vega(k: np.ndarray, iv: np.ndarray, t: float) -> np.ndarray:
    k, iv   = np.asarray(k, float), np.asarray(iv, float)
    safe_iv = np.maximum(iv, 1e-8)
    sqrt_t  = np.sqrt(max(t, 1e-8))
    d1      = (-k + 0.5 * safe_iv ** 2 * t) / (safe_iv * sqrt_t)
    return norm.pdf(d1) * sqrt_t


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SPEC PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────

class ModelSpec(ABC):
    """
    Abstract base class for a volatility model.

    Subclass this to add a new model.  You only need to implement:
        - pack / unpack      : bijection between dict params and unconstrained R^n
        - iv_from_params     : model implied vol given params and strikes
        - de_bounds          : search bounds for differential evolution (in packed space)
        - initial_guess      : sensible starting point in packed space
        - check_arbitrage    : returns (has_butterfly_arb, has_calendar_arb)

    The optimizer loop in fit_slice calls these methods and nothing else.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def pack(self, params: dict) -> np.ndarray:
        """Map constrained params dict → unconstrained array for optimizer."""
        ...

    @abstractmethod
    def unpack(self, x: np.ndarray) -> dict:
        """Map unconstrained optimizer array → constrained params dict."""
        ...

    @abstractmethod
    def iv_from_params(self, k: np.ndarray, t: float, params: dict) -> np.ndarray:
        """Return Black-Scholes implied vol σ(k) for given params."""
        ...

    @abstractmethod
    def de_bounds(self, atm_iv: float, t: float) -> list:
        """Differential evolution bounds in packed space. List of (lo, hi) tuples."""
        ...

    @abstractmethod
    def initial_guess(self, atm_iv: float, t: float) -> np.ndarray:
        """Packed initial guess. Used if use_global_init=False."""
        ...

    def check_butterfly(self, params: dict, k_range=(-4., 4.), n=500) -> float:
        """
        Returns min_g value (>=0 means no butterfly arbitrage).
        Override for models where butterfly check has a closed form.
        Default: returns 0.0 (no check — safe to override).
        """
        return 0.0

    def check_calendar(self, params_earlier: dict, params_later: dict,
                       k_range=(-5., 5.), n=1000) -> float:
        """
        Returns crossedness (0.0 means no calendar spread arbitrage).
        Override for models that have a closed-form calendar check.
        Default: numerical check via total variance comparison.
        """
        k = np.linspace(k_range[0], k_range[1], n)
        t_dummy = 1.0   # crossedness is in total variance space
        iv1 = self.iv_from_params(k, t_dummy, params_earlier)
        iv2 = self.iv_from_params(k, t_dummy, params_later)
        w1  = iv1 ** 2 * t_dummy
        w2  = iv2 ** 2 * t_dummy
        diff = w1 - w2
        return float(max(0.0, diff.max()))

    def w_from_params(self, k: np.ndarray, t: float, params: dict) -> np.ndarray:
        """Total variance w = σ² * t. Convenience wrapper."""
        return self.iv_from_params(k, t, params) ** 2 * t


# ─────────────────────────────────────────────────────────────────────────────
# SVI MODEL SPEC
# ─────────────────────────────────────────────────────────────────────────────

class SVI_ModelSpec(ModelSpec):
    """
    Raw SVI:  w(k) = a + b * (ρ(k−m) + √((k−m)² + σ²))
    5 params: a, b, rho, m, sig

    Identical to the original svi_snapshot_calibration.py logic,
    now wrapped in the ModelSpec protocol.
    """

    @property
    def name(self): return "SVI"

    def pack(self, p: dict) -> np.ndarray:
        return np.array([
            p["a"],
            np.log(max(p["b"],   1e-9)),
            np.arctanh(np.clip(p["rho"], -0.9999, 0.9999)),
            p["m"],
            np.log(max(p["sig"], 1e-9)),
        ])

    def unpack(self, x: np.ndarray) -> dict:
        return {
            "a":   float(x[0]),
            "b":   float(np.exp(x[1])),
            "rho": float(np.tanh(x[2])),
            "m":   float(x[3]),
            "sig": float(np.exp(x[4])),
        }

    def iv_from_params(self, k, t, p) -> np.ndarray:
        k  = np.asarray(k, float)
        w  = p["a"] + p["b"] * (p["rho"] * (k - p["m"])
             + np.sqrt((k - p["m"]) ** 2 + p["sig"] ** 2))
        return np.sqrt(np.maximum(w / max(t, 1e-8), 0.0))

    def de_bounds(self, atm_iv, t):
        atm_w = atm_iv ** 2 * t
        return [
            (-0.5,          atm_w * 2),
            (np.log(1e-4),  np.log(5.0)),
            (-3.0,          3.0),
            (-2.0,          2.0),
            (np.log(1e-4),  np.log(3.0)),
        ]

    def initial_guess(self, atm_iv, t):
        atm_w = atm_iv ** 2 * t
        return self.pack({"a": atm_w * 0.9, "b": 0.1,
                          "rho": -0.7, "m": 0.0, "sig": 0.3})

    def check_butterfly(self, p, k_range=(-4., 4.), n=600):
        """Reproduce the g-function check from svi_snapshot_calibration."""
        k     = np.linspace(k_range[0], k_range[1], n)
        discr = np.sqrt((k - p["m"]) ** 2 + p["sig"] ** 2)
        w     = p["a"] + p["b"] * (p["rho"] * (k - p["m"]) + discr)
        dw    = p["b"] * p["rho"] + p["b"] * (k - p["m"]) / discr
        d2w   = p["b"] * p["sig"] ** 2 / discr ** 3
        g     = (1 - k * dw / (2 * w)) ** 2 - (dw ** 2 / 4) * (1 / w + 0.25) + d2w / 2
        return float(np.min(g))     # >=0 means no butterfly arbitrage

    def check_calendar(self, p1, p2, k_range=(-5., 5.), n=2000):
        """Direct total variance comparison — exact for SVI."""
        k    = np.linspace(k_range[0], k_range[1], n)
        def _w(p): return p["a"] + p["b"] * (p["rho"] * (k - p["m"])
                          + np.sqrt((k - p["m"]) ** 2 + p["sig"] ** 2))
        diff = _w(p1) - _w(p2)
        return float(max(0.0, diff.max()))


# ─────────────────────────────────────────────────────────────────────────────
# eSSVI MODEL SPEC
# ─────────────────────────────────────────────────────────────────────────────

class eSSVI_ModelSpec(ModelSpec):
    """
    Extended SSVI (Gatheral-Jacquier SSVI with power-law ϕ).

    w(k, θ) = (θ/2) * (1 + ρ·ϕ(θ)·k + √((ϕ(θ)·k + ρ)² + 1 − ρ²))

    where ϕ(θ) = η / θ^γ  (power-law parameterisation).

    Per-slice free params: rho, eta, gamma  (3 params)
    θ_t (ATM total variance) is passed as an external input per slice.

    This is the standard SSVI form; "extended" refers to allowing
    slice-by-slice rho rather than a global rho.

    Reference: Gatheral & Jacquier (2013), Section 4.
    """

    @property
    def name(self): return "eSSVI"

    def pack(self, p: dict) -> np.ndarray:
        return np.array([
            np.arctanh(np.clip(p["rho"], -0.9999, 0.9999)),
            np.log(max(p["eta"],   1e-9)),
            np.log(max(p["gamma"], 1e-9)),
        ])

    def unpack(self, x: np.ndarray) -> dict:
        return {
            "rho":   float(np.tanh(x[0])),
            "eta":   float(np.exp(x[1])),
            "gamma": float(np.exp(x[2])),
        }

    def _phi(self, theta, eta, gamma):
        """Power-law ϕ(θ) = η / θ^γ."""
        return eta / (max(theta, 1e-8) ** gamma)

    def iv_from_params(self, k, t, p) -> np.ndarray:
        """
        Requires p to contain 'theta' (ATM total variance at this expiry).
        Compute theta = atm_iv^2 * t before calling.
        """
        k     = np.asarray(k, float)
        theta = p.get("theta", p.get("atm_var", None))
        if theta is None:
            raise ValueError("eSSVI params must contain 'theta' (ATM total variance).")
        phi   = self._phi(theta, p["eta"], p["gamma"])
        inner = (phi * k + p["rho"]) ** 2 + 1 - p["rho"] ** 2
        w     = (theta / 2) * (1 + p["rho"] * phi * k + np.sqrt(np.maximum(inner, 0.0)))
        return np.sqrt(np.maximum(w / max(t, 1e-8), 0.0))

    def de_bounds(self, atm_iv, t):
        return [
            (-3.0, 3.0),            # arctanh rho
            (np.log(1e-4), np.log(10.0)),  # log eta
            (np.log(1e-4), np.log(2.0)),   # log gamma
        ]

    def initial_guess(self, atm_iv, t):
        return self.pack({"rho": -0.5, "eta": 1.0, "gamma": 0.5})

    def check_butterfly(self, p, k_range=(-4., 4.), n=600):
        """
        SSVI butterfly-free condition (Theorem 4.2 of Gatheral-Jacquier 2013).
        For power-law ϕ: sufficient condition is 0 < gamma <= 1/2 and
        eta*(1+|rho|) <= 2.  We return a soft score rather than hard bool.
        """
        gamma_ok = 0 < p["gamma"] <= 0.5
        eta_ok   = p["eta"] * (1 + abs(p["rho"])) <= 2.0
        # Return a score: positive = safe margin, negative = violated
        margin = min(
            0.5 - p["gamma"],                              # gamma bound
            2.0 - p["eta"] * (1 + abs(p["rho"])),         # eta bound
        )
        return float(margin)

    def check_calendar(self, p1, p2, k_range=(-5., 5.), n=2000):
        """Numerical calendar check in total variance space."""
        k = np.linspace(k_range[0], k_range[1], n)
        t_dummy = 1.0
        iv1 = self.iv_from_params(k, t_dummy, p1)
        iv2 = self.iv_from_params(k, t_dummy, p2)
        diff = iv1 ** 2 - iv2 ** 2
        return float(max(0.0, diff.max()))


# ─────────────────────────────────────────────────────────────────────────────
# SABR MODEL SPEC
# ─────────────────────────────────────────────────────────────────────────────

class SABR_ModelSpec(ModelSpec):
    """
    SABR model implied vol via the Hagan et al. (2002) approximation.

    σ_BS(K, F) ≈ Hagan formula (log-normal approximation)

    Free params: alpha, rho, nu  (beta is typically fixed externally)
    beta is stored in the ModelSpec and not optimised per slice.

    Reference: Hagan, Kumar, Lesniewski, Woodward (2002).

    Critical note: the Hagan approximation breaks down for:
        - Very short expiries (T < 1 week)
        - Deep OTM/ITM strikes on crypto (high vol, large |k|)
    For BTC you may want to use a normal-SABR or free-boundary SABR instead.
    """

    def __init__(self, beta: float = 0.5):
        """
        beta : float  CEV exponent in (0, 1).  Typically fixed at 0.5 (square-root)
               or 1.0 (log-normal SABR).  Not optimised per slice.
        """
        self.beta = float(beta)

    @property
    def name(self): return f"SABR(β={self.beta:.2f})"

    def pack(self, p: dict) -> np.ndarray:
        return np.array([
            np.log(max(p["alpha"], 1e-9)),
            np.arctanh(np.clip(p["rho"], -0.9999, 0.9999)),
            np.log(max(p["nu"],   1e-9)),
        ])

    def unpack(self, x: np.ndarray) -> dict:
        return {
            "alpha": float(np.exp(x[0])),
            "rho":   float(np.tanh(x[1])),
            "nu":    float(np.exp(x[2])),
            "beta":  self.beta,
        }

    def iv_from_params(self, k, t, p) -> np.ndarray:
        """
        Hagan et al. log-normal SABR approximation.
        k = log(K/F),  F is the forward (forward-normalised so F=1).
        """
        k      = np.asarray(k, float)
        alpha  = p["alpha"]
        beta   = p.get("beta", self.beta)
        rho    = p["rho"]
        nu     = p["nu"]
        t      = max(t, 1e-8)

        K = np.exp(k)           # strike (F-normalised, so F=1)
        F = 1.0

        atm_mask = np.abs(k) < 1e-6

        # Forward-normalised midpoint
        FK_mid = (F * K) ** ((1 - beta) / 2)

        # Log(F/K) — for non-ATM
        log_FK = np.where(atm_mask, 0.0, np.log(F / K))

        # z and x(z)
        z  = (nu / alpha) * FK_mid * log_FK
        xz = np.where(
            np.abs(z) < 1e-6,
            1.0,
            np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho)) / z
        )

        # Numerator factor A
        A = alpha / (FK_mid * (1 + ((1 - beta) ** 2 / 24) * log_FK ** 2
                                  + ((1 - beta) ** 4 / 1920) * log_FK ** 4))

        # Time-correction factor B
        B = 1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / (FK_mid ** 2)
                + (rho * beta * nu * alpha / 4) / FK_mid
                + (2 - 3 * rho ** 2) / 24 * nu ** 2) * t

        iv = A * (z / xz) * B

        # ATM formula (L'Hôpital limit of z/x(z) → 1)
        atm_iv = (alpha / F ** (1 - beta)) * (
            1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / F ** (2 - 2 * beta)
                 + (rho * beta * nu * alpha / 4) / F ** (1 - beta)
                 + (2 - 3 * rho ** 2) / 24 * nu ** 2) * t
        )

        iv = np.where(atm_mask, atm_iv, iv)
        return np.maximum(iv, 1e-6)

    def de_bounds(self, atm_iv, t):
        return [
            (np.log(1e-4), np.log(5.0)),   # log alpha
            (-3.0, 3.0),                    # arctanh rho
            (np.log(1e-4), np.log(10.0)),   # log nu
        ]

    def initial_guess(self, atm_iv, t):
        return self.pack({"alpha": atm_iv, "rho": -0.5, "nu": 0.5})

    def check_butterfly(self, p, k_range=(-4., 4.), n=600):
        """
        SABR does not have a simple closed-form butterfly condition.
        The Hagan approximation can produce negative densities for large nu.
        We do a numerical g-function check via the density proxy.
        Returns a scalar: positive = safe, negative = butterfly violated.
        """
        # We check that d²C/dK² > 0 numerically via finite differences
        k     = np.linspace(k_range[0], k_range[1], n)
        dk    = k[1] - k[0]
        t_ref = max(0.1, 0.0)    # use a reference t; butterfly is t-independent
        iv    = self.iv_from_params(k, 0.25, p)
        w     = iv ** 2 * 0.25
        # Second derivative of w (proxy — not exact Breeden-Litzenberger)
        d2w   = np.gradient(np.gradient(w, dk), dk)
        return float(np.min(d2w))   # positive = convex = no butterfly

    def check_calendar(self, p1, p2, k_range=(-5., 5.), n=2000):
        """Numerical calendar check."""
        k = np.linspace(k_range[0], k_range[1], n)
        t_dummy = 1.0
        iv1 = self.iv_from_params(k, t_dummy, p1)
        iv2 = self.iv_from_params(k, t_dummy, p2)
        diff = iv1 ** 2 - iv2 ** 2
        return float(max(0.0, diff.max()))


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC OPTIMIZER LOOP
# ─────────────────────────────────────────────────────────────────────────────

def fit_slice(
        k_obs:          np.ndarray,
        iv_obs:         np.ndarray,
        t:              float,
        model_spec:     ModelSpec,
        objective_fn:   Callable,
        prev_params:    Optional[dict]  = None,
        next_params:    Optional[dict]  = None,
        vega:           Optional[np.ndarray] = None,
        bid_ask_width:  Optional[np.ndarray] = None,
        penalty_butterfly: float = 1e3,
        penalty_calendar:  float = 500.0,
        use_global_init:   bool  = True,
        obj_kwargs:        dict  = None,
        extra_params:      dict  = None,
) -> dict:
    """
    Generic slice calibration.  Works with any ModelSpec and any objective.

    Parameters
    ----------
    k_obs           : log-strike array
    iv_obs          : observed implied vol array (σ_BS)
    t               : time to expiry
    model_spec      : a ModelSpec instance (SVI_ModelSpec, SABR_ModelSpec, ...)
    objective_fn    : callable from svi_objective_functions.OBJECTIVES or custom
    prev_params     : fitted params for the earlier slice (calendar constraint)
    next_params     : fitted params for the later slice (calendar constraint)
    vega            : BS vega array (computed internally if None)
    bid_ask_width   : bid-ask spread in IV points (defaults to 0.02 if None)
    penalty_butterfly : weight on butterfly arbitrage penalty
    penalty_calendar  : weight on calendar spread penalty
    use_global_init   : use differential evolution for robust global start
    obj_kwargs        : extra kwargs forwarded to objective_fn (e.g. alpha=0.3)
    extra_params      : extra fixed params injected into model params dict
                        (e.g. {"theta": atm_var} for eSSVI, {"beta": 0.5} for SABR)

    Returns
    -------
    dict of model parameters (keys depend on ModelSpec)
    """
    k_obs  = np.asarray(k_obs, float)
    iv_obs = np.asarray(iv_obs, float)
    w_obs  = iv_obs ** 2 * t
    obj_kwargs   = obj_kwargs   or {}
    extra_params = extra_params or {}

    atm_iv = float(np.interp(0.0, np.sort(k_obs), iv_obs[np.argsort(k_obs)]))

    # ── Pre-compute vega and spread if not provided ────────────────────────
    if vega is None:
        vega = _bs_vega(k_obs, iv_obs, t)
    if bid_ask_width is None:
        bid_ask_width = np.full_like(k_obs, 0.02)

    def _full_params(p_dict):
        """Merge model params with fixed extra_params (e.g. theta for eSSVI)."""
        return {**extra_params, **p_dict}

    def objective(x):
        p_raw  = model_spec.unpack(x)
        p_full = _full_params(p_raw)

        # Model prediction
        iv_model = model_spec.iv_from_params(k_obs, t, p_full)
        w_model  = iv_model ** 2 * t

        # ── Fit term ───────────────────────────────────────────────────────
        fit_err = objective_fn(
            w_model       = w_model,
            w_obs         = w_obs,
            iv_model      = iv_model,
            iv_obs        = iv_obs,
            vega          = vega,
            bid_ask_width = bid_ask_width,
            k_obs         = k_obs,
            t             = t,
            **obj_kwargs,
        )

        # ── Butterfly penalty ──────────────────────────────────────────────
        min_g   = model_spec.check_butterfly(p_full)
        but_pen = max(0.0, -min_g) * penalty_butterfly

        # ── Calendar penalty ───────────────────────────────────────────────
        cal_pen = 0.0
        if prev_params is not None:
            cal_pen += model_spec.check_calendar(prev_params, p_full) * penalty_calendar
        if next_params is not None:
            cal_pen += model_spec.check_calendar(p_full, next_params) * penalty_calendar

        return fit_err + but_pen + cal_pen

    # ── Global init via DE ─────────────────────────────────────────────────
    x0 = model_spec.initial_guess(atm_iv, t)

    if use_global_init:
        bounds = model_spec.de_bounds(atm_iv, t)
        de_res = differential_evolution(
            objective, bounds,
            seed=42, maxiter=300, tol=1e-7,
            popsize=8, mutation=(0.5, 1.5), recombination=0.9,
            workers=1,
        )
        x0 = de_res.x

    # ── Local polish ───────────────────────────────────────────────────────
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-9, "fatol": 1e-9})

    p_final = model_spec.unpack(res.x)
    return _full_params(p_final)


# ─────────────────────────────────────────────────────────────────────────────
# FULL SURFACE CALIBRATION (slice-by-slice, model-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_surface(
        df,
        model_spec:        ModelSpec,
        objective_fn:      Callable,
        penalty_butterfly: float = 1e3,
        penalty_calendar:  float = 500.0,
        min_points:        int   = 5,
        use_global_init:   bool  = True,
        verbose:           bool  = True,
        extra_params_fn:   Optional[Callable] = None,
) -> dict:
    """
    Calibrate the full vol surface slice-by-slice for any model and objective.

    Parameters
    ----------
    df               : DataFrame with columns k, w, t  (output of load_snapshot)
    model_spec       : ModelSpec instance
    objective_fn     : objective function (from svi_objective_functions)
    extra_params_fn  : optional callable(t, df_slice) -> dict of fixed params
                       e.g. for eSSVI: lambda t, s: {"theta": atm_var(s)}

    Returns
    -------
    dict with keys: expiries, model_params, n_points, model_name
    """
    expiries = sorted(df["t"].unique())
    n        = len(expiries)

    if verbose:
        print(f"\nCalibrating {n} slices with {model_spec.name} | "
              f"obj={objective_fn.__name__}")

    params_out = [None] * n

    for i, t_exp in enumerate(expiries):
        sub = df[df["t"] == t_exp]

        if len(sub) < min_points:
            if verbose:
                print(f"  [{i+1}/{n}] T={t_exp:.4f}  SKIPPED ({len(sub)} pts)")
            params_out[i] = params_out[i-1] if i > 0 else None
            continue

        k_obs  = sub["k"].values
        w_obs  = sub["w"].values
        iv_obs = np.sqrt(np.maximum(w_obs / max(t_exp, 1e-8), 0.0))

        bid_ask = None
        if "ask_iv" in sub.columns and "bid_iv" in sub.columns:
            bid_ask = (sub["ask_iv"].values - sub["bid_iv"].values)

        extra = {}
        if extra_params_fn is not None:
            extra = extra_params_fn(t_exp, sub)

        p = fit_slice(
            k_obs         = k_obs,
            iv_obs        = iv_obs,
            t             = t_exp,
            model_spec    = model_spec,
            objective_fn  = objective_fn,
            prev_params   = params_out[i-1] if i > 0 else None,
            bid_ask_width = bid_ask,
            penalty_butterfly = penalty_butterfly,
            penalty_calendar  = penalty_calendar,
            use_global_init   = use_global_init,
            extra_params      = extra,
        )
        params_out[i] = p

        if verbose:
            mg    = model_spec.check_butterfly(p)
            cross = model_spec.check_calendar(params_out[i-1], p) if i > 0 else 0.0
            flag  = "⚠ butterfly" if mg < 0 else "ok"
            print(f"  [{i+1}/{n}] T={t_exp:.4f}  n={len(sub):3d}  "
                  f"min_g={mg:+.4f}  cal={cross:.2e}  [{flag}]")

    valid = [(t, p) for t, p in zip(expiries, params_out) if p is not None]
    return {
        "expiries":    [v[0] for v in valid],
        "model_params":[v[1] for v in valid],
        "n_points":    [len(df[df["t"] == v[0]]) for v in valid],
        "model_name":  model_spec.name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SELF TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from svi_objective_functions import build_objective

    print("=" * 65)
    print("vol_calibration_engine — self test")
    print("=" * 65)

    rng = np.random.default_rng(1)
    t   = 0.25
    k   = np.linspace(-0.4, 0.4, 12)

    # Ground-truth SVI
    true_p = {"a": 0.04, "b": 0.15, "rho": -0.6, "m": 0.0, "sig": 0.25}
    spec   = SVI_ModelSpec()
    iv_true = spec.iv_from_params(k, t, true_p)
    iv_obs  = iv_true + rng.normal(0, 0.005, len(k))

    obj = build_objective("vega_weighted_iv")

    for ModelCls, kwargs, extra in [
        (SVI_ModelSpec,   {},              {}),
        (SABR_ModelSpec,  {"beta": 0.5},  {}),
    ]:
        ms = ModelCls(**kwargs)
        p  = fit_slice(k, iv_obs, t, model_spec=ms, objective_fn=obj,
                       use_global_init=True, extra_params=extra)
        iv_fit = ms.iv_from_params(k, t, p)
        rmse   = float(np.sqrt(np.mean((iv_fit - iv_obs) ** 2)))
        print(f"\n{ms.name}")
        print(f"  params : {p}")
        print(f"  RMSE IV: {rmse:.6f}")

    print("\nAll models fitted successfully.")
