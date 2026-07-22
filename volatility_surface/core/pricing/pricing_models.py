"""
Black-Scholes and Garman-Kohlhagen pricing models.

Both are special cases of the same formula. Garman-Kohlhagen generalises
Black-Scholes by adding a continuous yield q on the underlying — the foreign
risk-free rate in FX, the dividend yield for equities, or for crypto the
staking yield / perpetual funding rate.

    d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Call = S * exp(-q*T) * N(d1)  - K * exp(-r*T) * N(d2)
    Put  = K * exp(-r*T) * N(-d2) - S * exp(-q*T) * N(-d1)

Black-Scholes is recovered with q = 0.

Why both for crypto:
    - BS assumes the forward equals S * exp(r*T) (USD cost of carry only).
    - GK lets the forward equal S * exp((r - q) * T), so q absorbs the
      basis / funding the market is pricing into the curve.
    - If the observed basis is non-zero, GK should produce a tighter fit
      (and a smoother implied-vol smile) than BS at the same r.

All functions are vectorised over (S, K, T, r, q, sigma, price). option_type
accepts a scalar ('c'/'p', case-insensitive, 'call'/'put' also fine) or an
array-like of the same length.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _call_put_sign(option_type) -> np.ndarray | float:
    """Map call/put flag to +1 / -1. Accepts scalar or array-like of strings."""
    if isinstance(option_type, str):
        return 1.0 if option_type.strip().lower()[0] == 'c' else -1.0
    arr = np.asarray(option_type)
    first = np.array([str(x).strip().lower()[0] for x in arr.ravel()]).reshape(arr.shape)
    return np.where(first == 'c', 1.0, -1.0)


def _d1_d2(S, K, T, r, q, sigma):
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    sqrtT = np.sqrt(T)
    vol_t = sigma * sqrtT
    d1 = (np.log(np.asarray(S) / np.asarray(K)) + (np.asarray(r) - np.asarray(q) + 0.5 * sigma ** 2) * T) / vol_t
    d2 = d1 - vol_t
    return d1, d2


# ----------------------------------------------------------------------------
# Garman-Kohlhagen (workhorse — BS is the q=0 case)
# ----------------------------------------------------------------------------

def gk_price(S, K, T, r, q, sigma, option_type='c'):
    """Garman-Kohlhagen European option price."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    eta = _call_put_sign(option_type)
    disc_K = np.exp(-np.asarray(r) * np.asarray(T))
    disc_S = np.exp(-np.asarray(q) * np.asarray(T))
    return eta * (np.asarray(S) * disc_S * norm.cdf(eta * d1)
                  - np.asarray(K) * disc_K * norm.cdf(eta * d2))


def gk_vega(S, K, T, r, q, sigma):
    """Vega = dPrice/dSigma. Same for call and put."""
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return np.asarray(S) * np.exp(-np.asarray(q) * np.asarray(T)) * norm.pdf(d1) * np.sqrt(np.asarray(T))


def gk_vanna(S, K, T, r, q, sigma):
    """Vanna = d^2 Price / (dS dSigma). Same for call and put."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return -np.exp(-np.asarray(q) * np.asarray(T)) * norm.pdf(d1) * d2 / np.asarray(sigma)


def gk_volga(S, K, T, r, q, sigma):
    """Volga (= vomma) = d^2 Price / dSigma^2. Same for call and put."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return gk_vega(S, K, T, r, q, sigma) * d1 * d2 / np.asarray(sigma)


def _gk_price_eta(S, K, T, r, q, sigma, eta):
    """Same formula as `gk_price`, but takes the numeric call/put sign `eta`
    directly instead of an option_type flag — avoids re-parsing option_type
    strings on every Newton iteration in `gk_implied_vol`."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_K = np.exp(-np.asarray(r) * np.asarray(T))
    disc_S = np.exp(-np.asarray(q) * np.asarray(T))
    return eta * (np.asarray(S) * disc_S * norm.cdf(eta * d1)
                  - np.asarray(K) * disc_K * norm.cdf(eta * d2))


def gk_implied_vol(price, S, K, T, r, q, option_type='c',
                   tol=1e-8, max_iter=100, sigma_init=0.6):
    """Vectorised Newton-Raphson implied vol under Garman-Kohlhagen.

    Falls back to bisection bracket [1e-6, 5.0] on rows where Newton fails to
    converge (vega collapses, deep ITM/OTM). Returns NaN for arbitrage-violating
    quotes (price outside intrinsic / upper bound).

    Only `valid` rows are ever computed on, and the Newton working set shrinks
    every iteration as rows converge (their indices are dropped, not just
    masked out of the convergence check) — with a fixed max_iter, operating on
    the full array every pass means every row pays for the slowest-converging
    row in the batch; on a large, moneyness-heterogeneous set that's most of
    the wasted cost. Same idea for the bisection fallback: it only ever
    touches the (usually small) subset that didn't converge under Newton,
    instead of re-running 80 passes over every row. ~45x faster on a 2M-row
    heterogeneous benchmark than a plain full-array version, same tol/max_iter.

    Known limitation shared with any absolute-tolerance Newton IV solver
    (this one, and the un-optimised version before it): for extreme deep-OTM /
    short-T rows, the model price can underflow to ~0 across a wide range of
    sigma, so `abs(diff) < tol` looks satisfied without sigma being anywhere
    near meaningful — the root just isn't identifiable from a price this deep
    in float64 underflow, regardless of solver. Benchmarked on 2M synthetic
    rows (deliberately wide moneyness incl. extreme wings): median abs
    difference vs. a non-shrinking reference is 6e-11 (i.e. exact), and 99.3%
    of the rows that DO differ by more than 1e-3 are exactly this underflow
    case. On real filtered market data (post volume/no-arb/OTM filtering,
    where this degenerate case essentially doesn't survive) max abs difference
    was 6.6e-6. Peter Jaeckel's "Let's Be Rational" (used by vollib) handles
    this regime properly by solving in a transformed price space instead of
    raw price difference — worth adopting if this ever needs to be airtight,
    but no maintained vectorised implementation was available (see
    requirements.txt note on py_vollib_vectorized's numba incompatibility).
    """
    price = np.asarray(price, dtype=float)
    orig_shape = price.shape
    price = price.ravel()
    S, K, T, r, q = [np.broadcast_to(np.asarray(x, dtype=float), orig_shape).ravel().copy() for x in (S, K, T, r, q)]
    eta = _call_put_sign(option_type)
    eta = np.full(price.shape, eta) if np.isscalar(eta) else np.broadcast_to(np.asarray(eta, dtype=float), orig_shape).ravel()

    # arbitrage bounds
    disc_K = np.exp(-r * T)
    disc_S = np.exp(-q * T)
    intrinsic = np.maximum(eta * (S * disc_S - K * disc_K), 0.0)
    upper = np.where(eta > 0, S * disc_S, K * disc_K)
    valid = (price > intrinsic - 1e-12) & (price < upper + 1e-12) & (T > 0) & (S > 0) & (K > 0)

    sigma = np.full(price.shape, np.nan, dtype=float)
    idx = np.flatnonzero(valid)  # active working set — shrinks as rows converge

    if idx.size:
        p_i, S_i, K_i, T_i, r_i, q_i, eta_i = (a[idx] for a in (price, S, K, T, r, q, eta))
        sig_i = np.full(idx.size, sigma_init)

        with np.errstate(divide='ignore', invalid='ignore'):
            for it in range(max_iter):
                pr = _gk_price_eta(S_i, K_i, T_i, r_i, q_i, sig_i, eta_i)
                diff = p_i - pr
                # Never allow "converged" on the very first pass: for extreme
                # deep-OTM/short-T rows the model price at sigma_init can
                # already be underflowed near zero (numerically indistinguish-
                # able from the tiny observed price) without sigma being
                # anywhere near meaningful — same degeneracy the old
                # full-batch loop has, just not visible there since it steps
                # every row unconditionally regardless of individual state.
                converged = np.abs(diff) < tol if it > 0 else np.zeros(diff.shape, dtype=bool)

                # Only step rows that haven't converged — a converged row can
                # have near-zero vega too, and diff/vega on an already-good row
                # can overshoot to the clip bound for no reason right as it's
                # about to be dropped from the active set. Write AFTER
                # stepping (not before) so the last iteration's result is
                # always the one persisted, for rows that run out of max_iter
                # without ever converging (e.g. a genuinely non-convergent,
                # oscillating row) — writing pre-step left the output exactly
                # one iteration stale, which for an oscillating row means
                # landing on the wrong phase entirely.
                v = gk_vega(S_i, K_i, T_i, r_i, q_i, sig_i)
                step = np.where(v > 1e-10, diff / np.maximum(v, 1e-10), 0.0)
                sig_i = np.where(converged, sig_i, np.clip(sig_i + step, 1e-6, 10.0))
                sigma[idx] = sig_i

                if np.all(converged):
                    break
                if np.any(converged):
                    keep = ~converged
                    idx, p_i, S_i, K_i, T_i, r_i, q_i, eta_i, sig_i = (
                        idx[keep], p_i[keep], S_i[keep], K_i[keep], T_i[keep],
                        r_i[keep], q_i[keep], eta_i[keep], sig_i[keep],
                    )

        # bisection fallback, only on whatever's left un-converged in idx
        if idx.size:
            final_diff = p_i - _gk_price_eta(S_i, K_i, T_i, r_i, q_i, sig_i, eta_i)
            bad_mask = np.abs(final_diff) > 1e-5
            if np.any(bad_mask):
                b_idx, b_price, b_S, b_K, b_T, b_r, b_q, b_eta = (
                    a[bad_mask] for a in (idx, p_i, S_i, K_i, T_i, r_i, q_i, eta_i)
                )
                lo = np.full(b_idx.size, 1e-6)
                hi = np.full(b_idx.size, 5.0)
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    p_mid = _gk_price_eta(b_S, b_K, b_T, b_r, b_q, mid, b_eta)
                    up = p_mid < b_price
                    lo = np.where(up, mid, lo)
                    hi = np.where(~up, mid, hi)
                sigma[b_idx] = 0.5 * (lo + hi)

    return sigma.reshape(orig_shape)


# ----------------------------------------------------------------------------
# Black-Scholes (q = 0)
# ----------------------------------------------------------------------------

def bs_price(S, K, T, r, sigma, option_type='c'):
    """Black-Scholes European option price (no dividend/funding yield)."""
    return gk_price(S, K, T, r, 0.0, sigma, option_type)


def bs_vega(S, K, T, r, sigma):
    return gk_vega(S, K, T, r, 0.0, sigma)


def bs_vanna(S, K, T, r, sigma):
    return gk_vanna(S, K, T, r, 0.0, sigma)


def bs_volga(S, K, T, r, sigma):
    return gk_volga(S, K, T, r, 0.0, sigma)


def bs_implied_vol(price, S, K, T, r, option_type='c', **kwargs):
    return gk_implied_vol(price, S, K, T, r, 0.0, option_type, **kwargs)


# ----------------------------------------------------------------------------
# Vanna-Volga (Castagna-Mercurio)
# ----------------------------------------------------------------------------

def vv_price(S, K, T, r, q, K_anchors, sigma_anchors, option_type='c'):
    """Vanna-Volga price using 3 anchor strikes and their market IVs.

    K_anchors      length-3 array: [K_low, K_atm, K_high] (e.g. 25-delta-put / ATM / 25-delta-call)
    sigma_anchors  length-3 array of market IVs at those strikes
    K              target strike(s) (scalar or array)
    option_type    flag(s) at the target strike(s), matching K's shape

    Mechanics (Castagna & Mercurio, 2007):
        Start with the BS/GK price at K under the flat ATM vol sigma_atm. Build
        a portfolio of the 3 anchor options that exactly replicates vega, vanna
        and volga of the target option. The cost of that portfolio (market vs
        flat-vol BS at each anchor) is the smile correction.

        C_VV(K) = C_BS(K, sigma_atm) + sum_i w_i * (C_BS(K_i, sigma_i) - C_BS(K_i, sigma_atm))

        where w_i solves the 3x3 system
            [vega(K_i)] [w] = vega(K)
            [vanna(K_i)]    = vanna(K)
            [volga(K_i)]    = volga(K)
        all evaluated under sigma_atm.

    At any anchor K=K_i, w collapses to a one-hot and VV reprices the anchor exactly.
    If sigma_anchors is flat (sigma_low = sigma_atm = sigma_high), VV reduces to BS/GK.

    S, T, r, q must be scalars (call once per expiry). K and option_type may be arrays.

    Caveat: VV is a local (Greek-matching) smile correction and breaks down for
    ultra-short maturities (~< 3 days for crypto) where the smile is too steep
    relative to BS-vega curvature. For those expiries it can produce wild
    interior prices. Empirically on ETH the method is well-behaved once T > 1
    week; filter accordingly before scoring.
    """
    K_anchors = np.asarray(K_anchors, dtype=float)
    sigma_anchors = np.asarray(sigma_anchors, dtype=float)
    if K_anchors.shape != (3,) or sigma_anchors.shape != (3,):
        raise ValueError("K_anchors and sigma_anchors must each be length-3 arrays")

    sigma_atm = float(sigma_anchors[1])

    vega_a = gk_vega(S, K_anchors, T, r, q, sigma_atm)
    vanna_a = gk_vanna(S, K_anchors, T, r, q, sigma_atm)
    volga_a = gk_volga(S, K_anchors, T, r, q, sigma_atm)
    A = np.stack([vega_a, vanna_a, volga_a])  # shape (3, 3)

    # price correction at each anchor (parity makes this option-type-invariant)
    bs_market_a = gk_price(S, K_anchors, T, r, q, sigma_anchors, 'c')
    bs_atm_a = gk_price(S, K_anchors, T, r, q, sigma_atm, 'c')
    delta_a = bs_market_a - bs_atm_a  # length 3

    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    scalar_out = (np.ndim(K) == 0)
    typ_arr = np.atleast_1d(np.asarray(option_type))
    if typ_arr.size == 1:
        typ_arr = np.broadcast_to(typ_arr, K_arr.shape)

    out = np.empty(K_arr.shape, dtype=float)
    for i, Kt in enumerate(K_arr):
        b = np.array([gk_vega(S, Kt, T, r, q, sigma_atm),
                      gk_vanna(S, Kt, T, r, q, sigma_atm),
                      gk_volga(S, Kt, T, r, q, sigma_atm)])
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.array([0.0, 1.0, 0.0])  # degenerate (collinear strikes) → flat-vol fallback
        bs_t = gk_price(S, Kt, T, r, q, sigma_atm, str(typ_arr[i]))
        out[i] = bs_t + float(np.dot(w, delta_a))

    return float(out[0]) if scalar_out else out


def _pick_vv_anchors(group: pd.DataFrame, wing_k: float) -> tuple | None:
    """Pick anchor rows by log-moneyness: closest to -wing_k, 0, +wing_k.

    Returns (idx_low, idx_atm, idx_high) or None if either wing has no quotes.
    """
    k = group['k']
    idx_atm = (k - 0.0).abs().idxmin()
    lows = group[k < 0]
    highs = group[k > 0]
    if lows.empty or highs.empty:
        return None
    idx_low = (lows['k'] - (-wing_k)).abs().idxmin()
    idx_high = (highs['k'] - wing_k).abs().idxmin()
    return idx_low, idx_atm, idx_high


def compare_pricing_models(df: pd.DataFrame,
                           r: float = 0.0,
                           q: float = 0.0,
                           wing_k: float = 0.1,
                           min_quotes_per_expiry: int = 6,
                           sigma_col: str = 'mid_iv') -> pd.DataFrame:
    """Reprice every option in `df` under three pricing models and compare to market mid.

    BS and GK are applied PER ROW using each option's own market IV
    (`(bid_iv + ask_iv) / 2` by default — override with `sigma_col='mark_iv'`).
    The only thing that distinguishes them at the row level is the forward
    assumption: BS uses F = S * exp(r*T), GK uses F = S * exp((r - q) * T).

    VV is intrinsically a smile model — it uses 3 anchor IVs per expiry (rows
    closest to k = -wing_k / 0 / +wing_k) to price every other row in the same
    expiry. It carries far less per-row info than BS/GK, so the comparison
    answers two different questions:

        BS vs GK     : does the market price options as if there's a non-zero
                       carry yield q? (Forward consistency test.)
        BS/GK vs VV  : can 3 anchor IVs per expiry reproduce the rest of the
                       smile? (Parsimony test.)

    The DataFrame must carry: strike, underlying_price, t, k, mid_price,
    option_type, bid_iv, ask_iv. Prices are normalized (Deribit convention:
    premium / spot) — S is set to 1 internally and strikes are divided by spot.

    Adds columns:
        mid_iv                                       (bid_iv + ask_iv) / 2
        bs_price_model, gk_price_model, vv_price_model
        bs_resid, gk_resid, vv_resid                 model - market
        is_anchor                                    rows used as VV anchors
    """
    out = df.copy()
    out['mid_iv'] = (out['bid_iv'] + out['ask_iv']) / 2

    # -- per-row BS and GK using each option's own IV (vectorised) --------
    S_usd = out['underlying_price'].to_numpy(dtype=float)
    K_usd = out['strike'].to_numpy(dtype=float)
    T_all = out['t'].to_numpy(dtype=float)
    sig = out[sigma_col].to_numpy(dtype=float)
    typ = out['option_type'].astype(str).to_numpy()
    K_norm_all = K_usd / S_usd

    out['bs_price_model'] = bs_price(1.0, K_norm_all, T_all, r, sig, typ)
    out['gk_price_model'] = gk_price(1.0, K_norm_all, T_all, r, q, sig, typ)
    out['vv_price_model'] = np.nan
    out['is_anchor'] = False

    # -- per-expiry VV using 3 anchors -----------------------------------
    for T_val, group in out.groupby('t'):
        if len(group) < min_quotes_per_expiry:
            continue
        anchors = _pick_vv_anchors(group, wing_k)
        if anchors is None:
            continue
        idx_low, idx_atm, idx_high = anchors

        S = float(group.loc[idx_atm, 'underlying_price'])
        K_anc = np.array([group.loc[i, 'strike'] / S for i in (idx_low, idx_atm, idx_high)])
        sig_anc = np.array([group.loc[i, sigma_col] for i in (idx_low, idx_atm, idx_high)])
        if not np.all(np.isfinite(sig_anc)) or not np.all(sig_anc > 0):
            continue

        K_norm = group['strike'].to_numpy(dtype=float) / S
        types = group['option_type'].astype(str).to_numpy()
        vv_p = vv_price(1.0, K_norm, float(T_val), r, q, K_anc, sig_anc, types)
        out.loc[group.index, 'vv_price_model'] = vv_p
        out.loc[[idx_low, idx_atm, idx_high], 'is_anchor'] = True

    market = out['mid_price'].to_numpy(dtype=float)
    out['bs_resid'] = out['bs_price_model'] - market
    out['gk_resid'] = out['gk_price_model'] - market
    out['vv_resid'] = out['vv_price_model'] - market
    return out


def compare_across_snapshots(df: pd.DataFrame,
                             r: float = 0.0,
                             q: float = 0.0,
                             wing_k: float = 0.1,
                             min_T_days: float = 3.0,
                             relative: bool = False,
                             min_price: float = 1e-4) -> pd.DataFrame:
    """Run `compare_pricing_models` on each snapshot and return per-snapshot RMSEs.

    `min_T_days` drops ultra-short expiries where VV is unstable.
    `relative=True` reports relative RMSE (dropping rows with mid_price <
    `min_price` so the deep-OTM denominator doesn't blow up).

    Returns one row per snapshot with columns:
        file_timestamp, n_options, bs_rmse, gk_rmse, vv_rmse, winner
    """
    rows = []
    for ts, snap in df.groupby('file_timestamp'):
        cmp_df = compare_pricing_models(snap.copy(), r=r, q=q, wing_k=wing_k)
        sub = cmp_df[cmp_df['t'] >= min_T_days / 365.0]
        sub = sub[~sub['is_anchor']].dropna(subset=['bs_resid', 'gk_resid', 'vv_resid'])
        if relative:
            sub = sub[sub['mid_price'] >= min_price].copy()
            for col in ('bs_resid', 'gk_resid', 'vv_resid'):
                sub[col] = sub[col] / sub['mid_price']
        if len(sub) == 0:
            continue
        rmse_bs = float(np.sqrt((sub['bs_resid'] ** 2).mean()))
        rmse_gk = float(np.sqrt((sub['gk_resid'] ** 2).mean()))
        rmse_vv = float(np.sqrt((sub['vv_resid'] ** 2).mean()))
        winner = min(('BS', rmse_bs), ('GK', rmse_gk), ('VV', rmse_vv), key=lambda x: x[1])[0]
        rows.append({
            'file_timestamp': ts,
            'n_options': len(sub),
            'bs_rmse': rmse_bs,
            'gk_rmse': rmse_gk,
            'vv_rmse': rmse_vv,
            'winner': winner,
        })
    return pd.DataFrame(rows)


def pricing_model_summary(df_compared: pd.DataFrame,
                          exclude_anchors: bool = True,
                          relative: bool = False,
                          min_price: float = 1e-4) -> pd.DataFrame:
    """RMSE per pricing model — overall and by moneyness bucket.

    relative=False : absolute RMSE in units of spot.
    relative=True  : relative RMSE = sqrt(mean(((model - market) / market) ** 2)).
                     Rows with mid_price < `min_price` are dropped to avoid the
                     deep-OTM denominator blowup (1e-4 ≈ 0.01% of spot; below
                     that the quote is essentially noise).

    VV reprices its anchors exactly, so we exclude them by default for a fair
    out-of-sample comparison. Set exclude_anchors=False to include them.
    """
    d = df_compared.dropna(subset=['bs_resid', 'gk_resid', 'vv_resid']).copy()
    if exclude_anchors and 'is_anchor' in d.columns:
        d = d[~d['is_anchor']]
    if relative:
        d = d[d['mid_price'] >= min_price].copy()
        for col in ('bs_resid', 'gk_resid', 'vv_resid'):
            d[col] = d[col] / d['mid_price']

    def rmse(sub, col):
        x = sub[col].to_numpy()
        return float(np.sqrt(np.mean(x ** 2)))

    rows = [{
        'bucket': 'overall', 'n': len(d),
        'bs_rmse': rmse(d, 'bs_resid'),
        'gk_rmse': rmse(d, 'gk_resid'),
        'vv_rmse': rmse(d, 'vv_resid'),
    }]
    buckets = [
        ('ATM |k|<0.05', d['k'].abs() < 0.05),
        ('mid 0.05-0.15', (d['k'].abs() >= 0.05) & (d['k'].abs() < 0.15)),
        ('wing |k|>=0.15', d['k'].abs() >= 0.15),
    ]
    for label, mask in buckets:
        sub = d[mask]
        if len(sub) == 0:
            continue
        rows.append({
            'bucket': label, 'n': len(sub),
            'bs_rmse': rmse(sub, 'bs_resid'),
            'gk_rmse': rmse(sub, 'gk_resid'),
            'vv_rmse': rmse(sub, 'vv_resid'),
        })
    return pd.DataFrame(rows)
