"""
Shared pricing and red-cell logic used by the live option-chain screen
(streamlit_chain.py), the GIF recorder (chain_gif.py), and the backtest engine
(engine.py). `bs_pricing` lives here so all three price off one
implementation instead of drifting apart — chain_gif.py's own docstring
already flagged the streamlit_chain.py/chain_gif.py duplication as a risk
before this module existed; a third copy for the backtest is the point to
stop duplicating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def bs_pricing(F, K, T, iv, is_call):
    """Vectorised Black-76 call/put price + greeks. r = 0.

    Moved here from streamlit_chain.py (previously duplicated a second time in
    chain_gif.py).
    """
    iv = np.maximum(iv, 1e-6)
    T_e = np.maximum(T, 1e-6)
    sqrtT = np.sqrt(T_e)
    d1 = (np.log(F / K) + 0.5 * iv ** 2 * T_e) / (iv * sqrtT)
    d2 = d1 - iv * sqrtT
    pdf_d1 = norm.pdf(d1)

    call_price = F * norm.cdf(d1) - K * norm.cdf(d2)
    put_price = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    price = np.where(is_call, call_price, put_price)

    delta = np.where(is_call, norm.cdf(d1), -norm.cdf(-d1))
    gamma = pdf_d1 / (F * iv * sqrtT)
    vega = F * pdf_d1 * sqrtT * 0.01              # per 1 vol-point
    theta = -F * pdf_d1 * iv / (2 * sqrtT) / 365  # per calendar day

    return price, delta, gamma, vega, theta


def detect_red_cells(df: pd.DataFrame, calib_result: dict, dp: int = 4, invert: bool = False) -> pd.DataFrame:
    """Flag each row of a single-snapshot option dataframe as a "red cell":
    the calibrated model's re-priced Black-76 coin price falls outside the
    row's own [bid_price, ask_price] at `dp`-decimal precision. Same
    definition as the live trading screen (streamlit_chain.py's `_styler`/
    red-cell count, duplicated in chain_gif.py) — a price-space test, not an
    IV-space one, so this stays consistent with what a trader watching the
    live screen would actually see flagged.

    `df` must be ONE snapshot (single file_timestamp), with the columns
    `clean_df_pl`/`clean_bulk_parquet_chunked` produce: strike, option_type
    ('C'/'P'), t, k, underlying_price, bid_price, ask_price. `calib_result` is
    the dict returned by any `calibrate_*`/`calibrate_*_update` function
    (needs "_model", "expiries", "params").

    Adds three columns to the returned frame:
        theo_price   model's Black-76 coin price at this row's own strike/t
        red_cell     bool, True if theo crosses outside [bid_price, ask_price]
        signal_side  'buy' (theo > ask, looks cheap), 'sell' (theo < bid,
                     looks rich), or None otherwise

    Rows with NaN/non-positive bid_price or ask_price are excluded from
    red_cell (never flagged) — synthetic/missing quotes, not real tradeable
    prices (same exclusion as the live screen's `fake_C`/`fake_P` masks).

    `invert=True` flags exactly the same cells but swaps buy<->sell — for
    testing whether the strategy's direction, not its transaction costs, is
    what's driving P&L. Note this does NOT flip transaction costs (spread-
    crossing and fees are paid regardless of side), only the directional bet.
    """
    model = calib_result["_model"]
    expiries = np.asarray(calib_result["expiries"], dtype=float)
    params_list = calib_result["params"]

    df = df.reset_index(drop=True).copy()
    theo = np.full(len(df), np.nan)
    t_arr = df["t"].to_numpy()

    for i, t_exp in enumerate(expiries):
        mask = np.isclose(t_arr, t_exp)
        if not mask.any():
            continue
        sub = df.loc[mask]
        F = float(sub["underlying_price"].iloc[0])
        k = sub["k"].to_numpy()
        iv = model.iv(k, params_list[i], t_exp)
        is_call = (sub["option_type"] == "C").to_numpy()
        price_usd, _, _, _, _ = bs_pricing(F, sub["strike"].to_numpy(), t_exp, iv, is_call)
        theo[mask] = price_usd / F

    df["theo_price"] = theo

    bid = df["bid_price"].to_numpy(dtype=float)
    ask = df["ask_price"].to_numpy(dtype=float)
    fake = np.isnan(bid) | (bid <= 0) | np.isnan(ask) | (ask <= 0) | np.isnan(theo)
    theo_r = np.round(theo, dp)
    bid_r = np.round(bid, dp)
    ask_r = np.round(ask, dp)

    cheap = (~fake) & (theo_r > ask_r)   # model says it's worth more than the ask -> buy
    rich = (~fake) & (theo_r < bid_r)    # model says it's worth less than the bid -> sell

    df["red_cell"] = cheap | rich
    if invert:
        df["signal_side"] = np.where(cheap, "sell", np.where(rich, "buy", None))
    else:
        df["signal_side"] = np.where(cheap, "buy", np.where(rich, "sell", None))
    return df


def implied_carry_rate(underlying_price: float, estimated_delivery_price: float, t: float) -> float:
    """Deribit's implied per-expiry interest-rate component: r = ln(F/S)/T,
    F = underlying_price (the per-expiry forward Deribit priced this option's
    slice against), S = estimated_delivery_price (Deribit's spot-index
    estimate). Same relationship streamlit_chain.py surfaces as its "Implied
    r" metric (`_render_tab`) — B76(F,...) == BS(S,...,r) mathematically, so
    this is exactly the cost-of-carry the market is pricing, derivable from
    columns already present on every row. No separate funding-rate data pull
    needed.
    """
    if estimated_delivery_price is None or estimated_delivery_price <= 0 or t <= 0:
        return 0.0
    return float(np.log(underlying_price / estimated_delivery_price) / t)


def detect_calendar_arb(calib_result: dict, k_range: tuple = (-3.0, 3.0), n: int = 200,
                         tol: float = 1e-6) -> list[dict]:
    """Calendar-spread arbitrage between ADJACENT expiry slices of a
    calibration result: for a fixed k, total variance w(k,T) must be
    non-decreasing in T. Violations are checked pairwise-adjacent, matching
    `enforce_calendar_arbfree`'s convention in calibrator.py.

    This is a genuinely different signal from `detect_red_cells`: that one
    compares the model to the MARKET's own bid/ask (is the smile internally
    consistent with what's quoted); this one is a pure MODEL-INTERNAL
    consistency check across time (is the calibrated surface even
    theoretically arbitrage-free), independent of any market quote. Models
    that enforce this by construction (eSSVI, SSVI, SABR — all use a
    calendar penalty or global-theta monotonicity) will essentially never
    trigger it; RawSVI, fit independently per slice with no cross-slice
    constraint, can (verified 0/300 on a real sample, but that's not a
    guarantee — see `crossedness` in calibrator.py, the same primitive this
    wraps).

    Returns one dict per violating adjacent pair (usually empty):
        {t_near, t_far, k_star, crossedness, params_near, params_far}
    `k_star` is the log-moneyness where the violation is largest — trade
    idea: at k_star, the near slice shows MORE total variance than the far
    slice, which is backwards (a genuine, static/model-free arbitrage, not a
    view on realized vol) — sell the near-expiry option and buy the
    far-expiry option, both at the strike nearest `k_star` (converted via
    each expiry's own forward, since forwards differ across expiries), same
    call/put type.
    """
    model = calib_result["_model"]
    expiries = calib_result["expiries"]
    params = calib_result["params"]
    k_grid = np.linspace(k_range[0], k_range[1], n)

    violations = []
    for i in range(len(expiries) - 1):
        diff = model.w(k_grid, params[i]) - model.w(k_grid, params[i + 1])
        j = int(np.argmax(diff))
        if diff[j] > tol:
            violations.append({
                "t_near": expiries[i], "t_far": expiries[i + 1],
                "k_star": float(k_grid[j]), "crossedness": float(diff[j]),
                "params_near": params[i], "params_far": params[i + 1],
            })
    return violations
