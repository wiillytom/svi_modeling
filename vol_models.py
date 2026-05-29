"""
vol_models.py
=============
Abstract base class for volatility smile models, plus concrete implementations.

Each model must implement:
    - w(k, params)          : total implied variance as a function of log-strike
    - bounds()              : parameter bounds for the optimiser
    - initial_guess(k, w)   : data-driven starting point
    - pack(params) / unpack(x): convert between dict and unconstrained real vector
    - validate(params)      : True if params satisfy model constraints
    - name                  : string identifier

Adding a new model means subclassing VolModel and implementing those methods.
Nothing else in the codebase needs to change.

Models implemented
------------------
    RawSVI      : Gatheral raw SVI (5 params)
    SSVI        : Surface SVI with power-law phi (3 global params + per-slice theta)
    SABR        : Hagan et al. SABR approximation (4 params)
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT BASE
# ─────────────────────────────────────────────────────────────────────────────

class VolModel(ABC):
    """
    Abstract base class for a single-slice volatility smile model.

    All models operate on:
        k  : log-strike  log(K/F)
        w  : total implied variance  sigma_BS^2 * T
        t  : time to expiry in years
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @abstractmethod
    def w(self, k: np.ndarray, params: dict) -> np.ndarray:
        """
        Total implied variance w(k) for a given parameter dict.
        Must return non-negative values for all k.
        """

    @abstractmethod
    def bounds(self) -> list:
        """
        List of (lo, hi) bounds in the *unconstrained* space used by the optimiser.
        Length must match pack() output.
        """

    @abstractmethod
    def initial_guess(self, k: np.ndarray, w_obs: np.ndarray, t: float) -> dict:
        """Data-driven initial parameter dict."""

    @abstractmethod
    def pack(self, params: dict) -> np.ndarray:
        """Convert params dict -> unconstrained real vector for the optimiser."""

    @abstractmethod
    def unpack(self, x: np.ndarray) -> dict:
        """Convert unconstrained real vector -> params dict."""

    def validate(self, params: dict) -> bool:
        """Return True if params satisfy all model constraints (optional override)."""
        return True

    def iv(self, k: np.ndarray, params: dict, t: float) -> np.ndarray:
        """Implied volatility sigma(k) = sqrt(w(k) / t)."""
        return np.sqrt(np.maximum(self.w(k, params) / t, 0.0))

    def butterfly_density(self, k: np.ndarray, params: dict) -> np.ndarray:
        """
        Risk-neutral density from Breeden-Litzenberger, expressed over k.
        Valid for any model where w(k) is twice differentiable.
        Uses finite differences — models can override for analytic derivatives.
        """
        k   = np.asarray(k, dtype=float)
        dk  = 1e-4
        w   = np.maximum(self.w(k, params), 1e-10)
        dw  = (self.w(k + dk, params) - self.w(k - dk, params)) / (2 * dk)
        d2w = (self.w(k + dk, params) + self.w(k - dk, params) - 2 * w) / dk**2

        g = (
            (1 - k * dw / (2 * w)) ** 2
            - (dw**2 / 4) * (1 / w + 0.25)
            + d2w / 2
        )
        d_minus  = -k / np.sqrt(w) - np.sqrt(w) / 2
        density  = g / np.sqrt(2 * np.pi * w) * np.exp(-d_minus**2 / 2)
        return np.maximum(density, 0.0)

    def min_g(self, params: dict, k_range=(-4.0, 4.0), n=600) -> float:
        """Minimum of g(k) — negative means butterfly arbitrage."""
        k_grid = np.linspace(k_range[0], k_range[1], n)
        dk  = 1e-4
        w   = np.maximum(self.w(k_grid, params), 1e-10)
        dw  = (self.w(k_grid + dk, params) - self.w(k_grid - dk, params)) / (2 * dk)
        d2w = (self.w(k_grid + dk, params) + self.w(k_grid - dk, params) - 2 * w) / dk**2
        g   = (1 - k_grid * dw / (2 * w))**2 - (dw**2 / 4) * (1/w + 0.25) + d2w/2
        return float(np.min(g))

    def has_butterfly_arb(self, params: dict) -> bool:
        return self.min_g(params) < -1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 1.  RAW SVI
# ─────────────────────────────────────────────────────────────────────────────

class RawSVI(VolModel):
    """
    Gatheral raw SVI parameterisation.
    w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sig^2))

    Parameters: a, b, rho, m, sig
    """

    name = "Raw SVI"

    def w(self, k, params):
        k   = np.asarray(k, dtype=float)
        a, b, rho, m, sig = (
            params["a"], params["b"], params["rho"], params["m"], params["sig"]
        )
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sig**2))

    # analytic g(k) — faster and more accurate than finite differences
    def butterfly_density(self, k, params):
        k   = np.asarray(k, dtype=float)
        a, b, rho, m, sig = (
            params["a"], params["b"], params["rho"], params["m"], params["sig"]
        )
        discr  = np.sqrt((k - m)**2 + sig**2)
        w      = np.maximum(a + b * (rho * (k - m) + discr), 1e-10)
        dw     = b * rho + b * (k - m) / discr
        d2w    = b * sig**2 / discr**3
        g      = (1 - k * dw / (2*w))**2 - (dw**2/4)*(1/w + 0.25) + d2w/2
        d_minus = -k / np.sqrt(w) - np.sqrt(w) / 2
        return np.maximum(g / np.sqrt(2 * np.pi * w) * np.exp(-d_minus**2 / 2), 0.0)

    def min_g(self, params, k_range=(-4.0, 4.0), n=600):
        k   = np.linspace(k_range[0], k_range[1], n)
        a, b, rho, m, sig = (
            params["a"], params["b"], params["rho"], params["m"], params["sig"]
        )
        discr = np.sqrt((k - m)**2 + sig**2)
        w     = np.maximum(a + b * (rho*(k-m) + discr), 1e-10)
        dw    = b * rho + b * (k-m) / discr
        d2w   = b * sig**2 / discr**3
        g     = (1 - k*dw/(2*w))**2 - (dw**2/4)*(1/w + 0.25) + d2w/2
        return float(np.min(g))

    def validate(self, params):
        b, rho, sig = params["b"], params["rho"], params["sig"]
        min_w = params["a"] + b * sig * np.sqrt(1 - rho**2)
        return b >= 0 and abs(rho) < 1 and sig > 0 and min_w >= 0

    def pack(self, params):
        return np.array([
            params["a"],
            np.log(max(params["b"],   1e-9)),
            np.arctanh(np.clip(params["rho"], -0.9999, 0.9999)),
            params["m"],
            np.log(max(params["sig"], 1e-9)),
        ])

    def unpack(self, x):
        return {
            "a":   float(x[0]),
            "b":   float(np.exp(x[1])),
            "rho": float(np.tanh(x[2])),
            "m":   float(x[3]),
            "sig": float(np.exp(x[4])),
        }

    def bounds(self):
        return [
            (-1.0,  2.0),          # a
            (np.log(1e-4), np.log(5.0)),  # log b
            (-3.5,  3.5),          # arctanh rho
            (-3.0,  3.0),          # m
            (np.log(1e-4), np.log(3.0)),  # log sig
        ]

    def initial_guess(self, k, w_obs, t):
        atm_w = float(np.interp(0.0, np.sort(k), w_obs[np.argsort(k)]))
        return {"a": atm_w * 0.9, "b": 0.1, "rho": -0.7, "m": 0.0, "sig": 0.3}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SSVI  (power-law phi, per-slice)
# ─────────────────────────────────────────────────────────────────────────────

class SSVI(VolModel):
    """
    Surface SVI with power-law phi(theta) = eta / theta^gamma.

    Per-slice parameters: theta (ATM total variance), eta, rho, gamma.
    Free of static arbitrage when  eta*(1+|rho|) <= 2  and  gamma in (0, 0.5].

    In per-slice mode (default), eta, rho, gamma are fitted per expiry.
    For a truly global SSVI fit use SSVIGlobal below.
    """

    name = "SSVI"

    def w(self, k, params):
        k     = np.asarray(k, dtype=float)
        theta = params["theta"]
        eta   = params["eta"]
        rho   = params["rho"]
        gamma = params["gamma"]
        phi   = eta / (theta ** gamma)
        return (theta / 2) * (
            1 + rho * phi * k + np.sqrt((phi * k + rho)**2 + 1 - rho**2)
        )

    def validate(self, params):
        eta, rho, gamma = params["eta"], params["rho"], params["gamma"]
        theta = params["theta"]
        return (
            eta > 0
            and abs(rho) < 1
            and 0 < gamma <= 0.5
            and theta > 0
            and eta * (1 + abs(rho)) <= 2
        )

    def pack(self, params):
        return np.array([
            np.log(max(params["theta"], 1e-9)),
            np.log(max(params["eta"],   1e-9)),
            np.arctanh(np.clip(params["rho"], -0.9999, 0.9999)),
            np.log(params["gamma"] / (0.5 - params["gamma"] + 1e-9)),  # gamma in (0,0.5)
        ])

    def unpack(self, x):
        gamma_raw = np.exp(x[3])
        gamma     = 0.5 * gamma_raw / (1 + gamma_raw)   # maps R -> (0, 0.5)
        return {
            "theta": float(np.exp(x[0])),
            "eta":   float(np.exp(x[1])),
            "rho":   float(np.tanh(x[2])),
            "gamma": float(gamma),
        }

    def bounds(self):
        return [
            (np.log(1e-5), np.log(5.0)),   # log theta
            (np.log(1e-3), np.log(10.0)),  # log eta
            (-3.5, 3.5),                   # arctanh rho
            (-5.0, 5.0),                   # gamma transform
        ]

    def initial_guess(self, k, w_obs, t):
        atm_w = float(np.interp(0.0, np.sort(k), w_obs[np.argsort(k)]))
        return {"theta": atm_w, "eta": 2.0, "rho": -0.7, "gamma": 0.4}

    def min_g(self, params, k_range=(-4.0, 4.0), n=50):
        """
        Analytical per-slice butterfly check using closed-form SSVI derivatives.

        For  w(k) = (theta/2)*(1 + rho*phi*k + sqrt((phi*k+rho)^2 + 1-rho^2))
        the first and second derivatives are exact:
            dw/dk  = (theta*phi/2) * (rho + (phi*k+rho)/D)
            d²w/dk² = (theta*phi²/2) * (1-rho²) / D³
        where D = sqrt((phi*k+rho)² + 1-rho²).

        Evaluates g(k) on 50 points — ~15× faster than the base-class
        600-point finite-difference scan, and checks the actual single-slice
        condition rather than the stricter surface-level condition.
        """
        k     = np.linspace(k_range[0], k_range[1], n)
        theta = params["theta"]
        phi   = params["eta"] / (theta ** params["gamma"])
        rho   = params["rho"]
        P     = phi * k + rho
        D     = np.sqrt(np.maximum(P**2 + (1 - rho**2), 1e-10))
        W     = np.maximum((theta / 2) * (1 + rho * phi * k + D), 1e-10)
        dW    = (theta * phi / 2) * (rho + P / D)
        d2W   = (theta * phi**2 / 2) * (1 - rho**2) / D**3
        g     = (1 - k * dW / (2 * W))**2 - (dW**2 / 4) * (1/W + 0.25) + d2W / 2
        return float(np.min(g))


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SABR  (Hagan et al. lognormal approximation)
# ─────────────────────────────────────────────────────────────────────────────

class SABR(VolModel):
    """
    SABR model implied vol approximation (Hagan et al. 2002).

    Parameters: alpha (ATM vol level), beta (CEV exponent, typically fixed at 1
    for lognormal), rho (spot-vol correlation), nu (vol of vol).

    Note: SABR is parameterised in implied vol space, not total variance.
    The w() method converts back to total variance for consistency with the
    rest of the framework.

    For crypto options, beta=1 (lognormal SABR) is the standard choice.
    """

    name = "SABR"

    def __init__(self, beta: float = 1.0):
        """
        Parameters
        ----------
        beta : float  CEV exponent, fixed before calibration (default 1.0)
        """
        self.beta = beta

    def _sabr_iv(self, k, params, t):
        """
        Hagan et al. (2002) approximation for implied vol.
        k     : log-strike (note: SABR originally uses K and F separately;
                we reconstruct K = F*exp(k) and set F=1 by convention)
        """
        alpha = params["alpha"]
        rho   = params["rho"]
        nu    = params["nu"]
        beta  = self.beta
        F     = 1.0           # normalised forward
        K     = np.exp(k)

        # Avoid numerical issues at ATM
        eps   = 1e-7
        atm   = np.abs(k) < eps

        # Mid-point for the log-expansion
        FK    = F * K
        FK_b  = FK ** ((1 - beta) / 2)

        log_FK = np.where(atm, 0.0, np.log(F / K))

        z      = (nu / alpha) * FK_b * log_FK
        safe_z = np.where(np.abs(z) < eps, 1.0, z)   # avoid 0/0 in both branches
        with np.errstate(divide="ignore", invalid="ignore"):
            x_z = np.where(
                np.abs(z) < eps,
                1.0,
                np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho)) / safe_z,
            )

        A = alpha / (FK_b * (1 + ((1-beta)**2/24)*log_FK**2
                             + ((1-beta)**4/1920)*log_FK**4))
        B = 1.0 / x_z

        C = (
            1
            + ((1-beta)**2 / 24 * alpha**2 / FK**(2*(1-beta))
               + rho*beta*nu*alpha / (4 * FK**((1-beta)))
               + (2 - 3*rho**2) / 24 * nu**2) * t
        )

        iv = A * B * C
        # ATM correction
        iv_atm = (
            alpha / F**(1-beta)
            * (1 + ((1-beta)**2/24 * alpha**2/F**(2*(1-beta))
                    + rho*beta*nu*alpha/(4*F**(1-beta))
                    + (2-3*rho**2)/24*nu**2) * t)
        )
        return np.where(atm, iv_atm, iv)

    def w(self, k, params):
        k  = np.asarray(k, dtype=float)
        t  = params["t"]    # t must be in params for SABR
        iv = self._sabr_iv(k, params, t)
        return np.maximum(iv, 0.0)**2 * t

    def iv(self, k, params, t):
        params_t = {**params, "t": t}
        return np.maximum(self._sabr_iv(k, params_t, t), 0.0)

    def validate(self, params):
        return (
            params["alpha"] > 0
            and abs(params["rho"]) < 1
            and params["nu"] > 0
        )

    def pack(self, params):
        return np.array([
            np.log(max(params["alpha"], 1e-9)),
            np.arctanh(np.clip(params["rho"], -0.9999, 0.9999)),
            np.log(max(params["nu"], 1e-9)),
        ])

    def unpack(self, x):
        return {
            "alpha": float(np.exp(x[0])),
            "rho":   float(np.tanh(x[1])),
            "nu":    float(np.exp(x[2])),
        }

    def bounds(self):
        return [
            (np.log(1e-4), np.log(5.0)),   # log alpha
            (-3.5, 3.5),                   # arctanh rho
            (np.log(1e-4), np.log(10.0)),  # log nu
        ]

    def initial_guess(self, k, w_obs, t):
        atm_w  = float(np.interp(0.0, np.sort(k), w_obs[np.argsort(k)]))
        atm_iv = np.sqrt(max(atm_w / t, 1e-6))
        return {"alpha": atm_iv, "rho": -0.5, "nu": 0.5, "t": t}

# ─────────────────────────────────────────────────────────────────────────────
# 4.  eSSVI  (Hendriks & Martini 2019 — maturity-dependent rho)
# ─────────────────────────────────────────────────────────────────────────────

class eSSVI(VolModel):
    """
    Extended SSVI (Hendriks & Martini, Journal of Computational Finance 2019).

    Same formula as SSVI but rho is a function of theta:

        w(k, theta) = (theta/2) * (
            1 + rho(theta)*phi(theta)*k
            + sqrt((phi(theta)*k + rho(theta))^2 + 1 - rho(theta)^2)
        )

        phi(theta)   = eta / theta^gamma           (power-law, same as SSVI)
        rho(theta)   = rho_inf + (rho_0 - rho_inf) * exp(-lam * theta)

    The exponential form is well-behaved for all theta > 0:
        rho(0)   = rho_0   (short-maturity limit)
        rho(inf) = rho_inf (long-maturity limit)

    Empirically rho is more negative for short maturities and flattens out
    for long maturities. SSVI with constant rho cannot capture this.

    Parameters
    ----------
    theta   : ATM total variance for this slice  (> 0)
    eta     : vol-of-vol scaling                 (> 0)
    gamma   : skew decay exponent                (0 < gamma <= 0.5)
    rho_0   : short-maturity correlation limit   (|rho_0| < 1)
    rho_inf : long-maturity correlation limit    (|rho_inf| < 1)
    lam     : decay speed                        (lam > 0)
    """

    name = "eSSVI"

    def _rho(self, theta, params):
        rho_0   = params["rho_0"]
        rho_inf = params["rho_inf"]
        lam     = params["lam"]
        return rho_inf + (rho_0 - rho_inf) * np.exp(-lam * theta)

    def w(self, k, params):
        k     = np.asarray(k, dtype=float)
        theta = params["theta"]
        eta   = params["eta"]
        gamma = params["gamma"]
        rho   = self._rho(theta, params)
        phi   = eta / (theta ** gamma)
        return (theta / 2) * (
            1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2)
        )

    def validate(self, params):
        rho = self._rho(params["theta"], params)
        # no-arb requires eta*(1+|rho|) <= 2 for the worst-case rho on [0,inf)
        max_rho_abs = max(abs(params["rho_0"]), abs(params["rho_inf"]))
        return (
            params["theta"] > 0
            and params["eta"] > 0
            and 0 < params["gamma"] <= 0.5
            and abs(params["rho_0"]) < 1
            and abs(params["rho_inf"]) < 1
            and params["lam"] > 0
            and params["eta"] * (1 + max_rho_abs) <= 2
        )

    def pack(self, params):
        gamma = np.clip(params["gamma"], 1e-6, 0.5 - 1e-6)
        return np.array([
            np.log(max(params["theta"],  1e-9)),
            np.log(max(params["eta"],    1e-9)),
            np.log(gamma / (0.5 - gamma)),
            np.arctanh(np.clip(params["rho_0"],   -0.9999, 0.9999)),
            np.arctanh(np.clip(params["rho_inf"], -0.9999, 0.9999)),
            np.log(max(params["lam"], 1e-9)),
        ])

    def unpack(self, x):
        gamma_raw = np.exp(x[2])
        gamma     = 0.5 * gamma_raw / (1 + gamma_raw)
        return {
            "theta":   float(np.exp(x[0])),
            "eta":     float(np.exp(x[1])),
            "gamma":   float(gamma),
            "rho_0":   float(np.tanh(x[3])),
            "rho_inf": float(np.tanh(x[4])),
            "lam":     float(np.exp(x[5])),
        }

    def bounds(self):
        return [
            (np.log(1e-5), np.log(10.0)),           # log theta
            (np.log(1e-3), np.log(10.0)),           # log eta
            (-5.0, 5.0),                             # gamma transform
            (-3.5, 3.5),                             # arctanh rho_0
            (-3.5, 3.5),                             # arctanh rho_inf
            (np.log(1e-2), np.log(50.0)),           # log lam
        ]

    def initial_guess(self, k, w_obs, t):
        atm_w = float(np.interp(0.0, np.sort(k), w_obs[np.argsort(k)]))
        return {
            "theta":   atm_w,
            "eta":     1.5,
            "gamma":   0.4,
            "rho_0":   -0.8,
            "rho_inf": -0.3,
            "lam":     2.0,
        }

    def min_g(self, params, k_range=(-4.0, 4.0), n=50):
        """
        Analytical per-slice butterfly check using closed-form eSSVI derivatives.

        Same formula as SSVI.min_g but uses rho(theta) from _rho() rather than
        a fixed rho parameter.  Checks the actual single-slice g(k)>=0 condition
        rather than the stricter surface-level Gatheral-Jacquier condition —
        critical for short maturities where per-slice solutions can be
        butterfly-free even when eta*(1+|rho|) > 2 at the surface level.

        ~15× faster than the base-class 600-point finite-difference scan.
        """
        k     = np.linspace(k_range[0], k_range[1], n)
        theta = params["theta"]
        rho   = self._rho(theta, params)
        phi   = params["eta"] / (theta ** params["gamma"])
        P     = phi * k + rho
        D     = np.sqrt(np.maximum(P**2 + (1 - rho**2), 1e-10))
        W     = np.maximum((theta / 2) * (1 + rho * phi * k + D), 1e-10)
        dW    = (theta * phi / 2) * (rho + P / D)
        d2W   = (theta * phi**2 / 2) * (1 - rho**2) / D**3
        g     = (1 - k * dW / (2 * W))**2 - (dW**2 / 4) * (1/W + 0.25) + d2W / 2
        return float(np.min(g))

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY  —  add new models here
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "svi":  RawSVI,
    "ssvi": SSVI,
    "sabr": SABR,
    'essvi':eSSVI
}


def get_model(name: str, **kwargs) -> VolModel:
    """
    Retrieve a model instance by name string.

    Parameters
    ----------
    name   : str   one of 'svi', 'ssvi', 'sabr'
    kwargs : passed to the model constructor (e.g. beta=1.0 for SABR)

    Example
    -------
    model = get_model("sabr", beta=1.0)
    model = get_model("svi")
    """
    key = name.lower()
    if key not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODELS.keys())}")
    return MODELS[key](**kwargs)
