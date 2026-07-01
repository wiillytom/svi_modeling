"""
Reiswich–Wystup (2010) **Simplified Parabolic** smile parametrisation.

Reference: D. Reiswich, U. Wystup, "FX Volatility Smile Construction",
CPQF Working Paper 20 (2010).  Theorem 1, equations (34)–(35).

The smile is a parabola in delta-space anchored at three quotes:
  - sigma_ATM       at the ATM-Δ-neutral strike  (Δ_ATM = 0.5 for fwd delta)
  - sigma_RR        the 25Δ risk-reversal  =  sigma(K_25C) - sigma(K_25P)
  - sigma_S         the *smile* strangle  =  (sigma(K_25C)+sigma(K_25P))/2 − sigma_ATM

For forward delta with Δ̃ = 0.25 and Δ_ATM = 0.5, the formula collapses to
the classical Malz (1997) form:
    σ(Δ) = σ_ATM − 2·σ_RR·(Δ − 0.5) + 16·σ_S·(Δ − 0.5)²

This module exposes:
    fit_paper_style(k_obs, iv_obs, T, F)   3-point extraction (paper-faithful)
    fit_least_squares(k_obs, iv_obs, T, F) 3-parameter LS fit to the full chain
    sigma_of_K(K, F, T, σ_ATM, σ_RR, σ_S)  smile at a single strike (implicit solve)
    w_of_k(k, F, T, σ_ATM, σ_RR, σ_S)      total-variance form, useful as a w(k)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, least_squares
from scipy.stats import norm


# Defaults for the standard FX/Deribit-style quote
DELTA_QUOTE = 0.25        # the |Δ| of the quoted risk-reversal / strangle
DELTA_ATM   = 0.5         # forward-delta-neutral ATM ⇒ Δ_ATM = 0.5
A_FWD       = 1.0         # call delta minus put delta for forward delta = 1


# ─────────────────────────────────────────────────────────────────────────────
# Fitted parameter container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RWParams:
    sigma_atm: float
    sigma_rr:  float
    sigma_s:   float        # smile strangle (NOT the quoted strangle)
    F:         float
    T:         float

    def smile(self, K):
        return sigma_of_K(K, self.F, self.T, self.sigma_atm, self.sigma_rr, self.sigma_s)

    def __repr__(self) -> str:
        return (f"RWParams(σ_ATM={self.sigma_atm:.4f}  σ_RR={self.sigma_rr:+.4f}  "
                f"σ_S={self.sigma_s:.4f}  F={self.F:.4g}  T={self.T:.4g})")


# ─────────────────────────────────────────────────────────────────────────────
# Forward delta and inverse map
# ─────────────────────────────────────────────────────────────────────────────

def _d1(F: float, K: np.ndarray, T: float, sigma: np.ndarray) -> np.ndarray:
    return (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))


def forward_delta(F: float, K, T: float, sigma, call: bool = True) -> float:
    d1 = _d1(F, K, T, sigma)
    return norm.cdf(d1) if call else norm.cdf(d1) - 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Parabolic formula in delta-space
# ─────────────────────────────────────────────────────────────────────────────

def _malz_parabola(delta: float, sigma_atm: float, sigma_rr: float, sigma_s: float) -> float:
    """σ(Δ) = σ_ATM − 2·σ_RR·(Δ − 0.5) + 16·σ_S·(Δ − 0.5)²"""
    x = delta - DELTA_ATM
    return sigma_atm - 2.0 * sigma_rr * x + 16.0 * sigma_s * x * x


def sigma_of_K(K, F: float, T: float, sigma_atm: float, sigma_rr: float, sigma_s: float):
    """Map strike → smile vol.  Vectorised over K."""
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    out = np.empty_like(K_arr)
    for i, k in enumerate(K_arr):
        def residual(sigma):
            d1 = (np.log(F / k) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
            delta = norm.cdf(d1)
            return sigma - _malz_parabola(delta, sigma_atm, sigma_rr, sigma_s)
        # bracket: residual sign changes between very low and very high vol
        try:
            out[i] = brentq(residual, 1e-3, 5.0, maxiter=200)
        except Exception:
            out[i] = sigma_atm        # fail-soft
    return out if out.size > 1 else float(out[0])


def w_of_k(k_log_moneyness, F: float, T: float,
           sigma_atm: float, sigma_rr: float, sigma_s: float) -> np.ndarray:
    """Total variance w(k) at log-moneyness k = ln(K/F)."""
    K = F * np.exp(np.asarray(k_log_moneyness, dtype=float))
    sigma = sigma_of_K(K, F, T, sigma_atm, sigma_rr, sigma_s)
    return sigma ** 2 * T


# ─────────────────────────────────────────────────────────────────────────────
# Calibration variant A — paper-faithful 3-point extraction
# ─────────────────────────────────────────────────────────────────────────────

def fit_paper_style(k_obs: np.ndarray, iv_obs: np.ndarray, T: float, F: float,
                    delta_quote: float = DELTA_QUOTE) -> RWParams:
    """Extract σ_ATM, σ_RR and σ_S from a per-strike IV chain in the way the
    Reiswich-Wystup paper builds the smile from FX market quotes.

    Steps:
      1. Build a PCHIP interpolant of σ_market(k).
      2. Find the ATM-Δ-neutral strike K_ATM by fixed-point iteration on
         K = F · exp(σ_market(K)² · T / 2).
      3. Brent-root-search for K_25C, K_25P such that forward-call-delta at
         (K, σ_market(K)) equals +0.25 / −0.25 respectively.
      4. σ_ATM = σ_market(K_ATM)
         σ_RR  = σ_market(K_25C) − σ_market(K_25P)
         σ_S   = (σ_market(K_25C) + σ_market(K_25P))/2 − σ_ATM.
    """
    k_obs = np.asarray(k_obs, dtype=float)
    iv_obs = np.asarray(iv_obs, dtype=float)
    order = np.argsort(k_obs)
    k_sorted, iv_sorted = k_obs[order], iv_obs[order]
    interp = PchipInterpolator(k_sorted, iv_sorted, extrapolate=True)

    def sigma_market_at_strike(K: float) -> float:
        k = float(np.log(K / F))
        return float(interp(np.clip(k, k_sorted[0], k_sorted[-1])))

    # 2. Fixed-point for K_ATM
    sigma_atm = float(np.mean(iv_obs))
    for _ in range(50):
        K_atm = F * np.exp(0.5 * sigma_atm ** 2 * T)
        new = sigma_market_at_strike(K_atm)
        if abs(new - sigma_atm) < 1e-9:
            sigma_atm = new
            break
        sigma_atm = new
    K_atm = F * np.exp(0.5 * sigma_atm ** 2 * T)

    # 3. Find K_25C, K_25P
    def call_delta_err(K):
        sigma = sigma_market_at_strike(K)
        return forward_delta(F, K, T, sigma, call=True) - delta_quote

    def put_delta_err(K):
        sigma = sigma_market_at_strike(K)
        return forward_delta(F, K, T, sigma, call=False) - (-delta_quote)

    K_min = float(F * np.exp(k_sorted[0]))
    K_max = float(F * np.exp(k_sorted[-1]))
    try:
        K_25C = brentq(call_delta_err, K_atm, K_max, maxiter=200)
        K_25P = brentq(put_delta_err,  K_min, K_atm, maxiter=200)
    except ValueError:
        # Chain doesn't reach Δ=±0.25 — fall back to wing extremes
        K_25C = K_max
        K_25P = K_min

    sigma_25c = sigma_market_at_strike(K_25C)
    sigma_25p = sigma_market_at_strike(K_25P)

    sigma_rr = sigma_25c - sigma_25p
    sigma_s  = 0.5 * (sigma_25c + sigma_25p) - sigma_atm

    return RWParams(sigma_atm, sigma_rr, sigma_s, F=F, T=T)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration variant B — full-chain least squares on (σ_ATM, σ_RR, σ_S)
# ─────────────────────────────────────────────────────────────────────────────

def fit_least_squares(k_obs: np.ndarray, iv_obs: np.ndarray, T: float, F: float,
                      weights: np.ndarray | None = None,
                      seed: RWParams | None = None) -> RWParams:
    """Fit the same 3-parameter parabolic form by minimising weighted residuals
    over the WHOLE chain (closer to how RawSVI / eSSVI are calibrated).

    Initial guess comes from the paper-style fit if `seed` is None.
    """
    if seed is None:
        seed = fit_paper_style(k_obs, iv_obs, T, F)

    k_obs = np.asarray(k_obs, dtype=float)
    iv_obs = np.asarray(iv_obs, dtype=float)
    K_obs = F * np.exp(k_obs)
    if weights is None:
        weights = np.ones_like(iv_obs)

    def residuals(x):
        sa, sr, ss = x
        sigma_model = sigma_of_K(K_obs, F, T, sa, sr, ss)
        return np.sqrt(weights) * (sigma_model - iv_obs)

    x0 = np.array([seed.sigma_atm, seed.sigma_rr, seed.sigma_s])
    res = least_squares(residuals, x0, method="lm", max_nfev=300)
    sa, sr, ss = res.x
    return RWParams(float(sa), float(sr), float(ss), F=F, T=T)
