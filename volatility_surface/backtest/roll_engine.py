"""
Systematic roll-based option-strategy backtester — Lucic & Sepp (2024),
"Valuation and hedging of cryptocurrency inverse options" (§5.2-5.3).

This is a DIFFERENT engine from `engine.py`. `engine.py` is opportunistic
relative-value: calibrate a surface every snapshot, flag mispriced "red cells",
trade them. This module implements the paper's *systematic* strategies: a fixed
calendar of rolls (weekly Friday / monthly / quarterly), at each roll selling or
buying a FIXED structure, held to maturity, delta-hedged hourly with the inverse
perpetual, sized proportionally to the current Coin NAV (auto-compounding,
N(t0)=Pi^BTC(t0)).

`run_all_strategies` runs the WHOLE catalog (all single- and multi-leg
structures, long and short — Tables 1/3 of the paper) in a SINGLE pass over the
data: the expensive part (reading + iterating 165 daily chunks) is paid once,
and each strategy carries its own light per-snapshot state. So all ~26
strategies cost roughly the same wall-clock as one. `run_roll_backtest` is a
thin single-strategy wrapper on top.

Reused from the rest of the project (so nothing drifts):
  - `pricing_models.inverse_delta` — premium-adjusted "Net Delta" (Eq 17), the
    only correct hedge ratio for a coin-settled option (NOT plain Black delta).
  - `signal.bs_pricing` — the one Black-76 implementation the whole project
    prices off.
  - `engine.py`'s hedge-P&L convention (§5): hedge_notional is the Coin net
    delta Delta^BTC, per-hour P&L is (F_h - F_{h-1})/F_h * Delta^BTC (Eq 26).
    The F_t^{Tk}/F_t perp-vs-dated scaling of Eq (25) is deliberately omitted
    to stay identical to `engine.py` (a ~1e-3 funding-basis term for 7-day
    options).

Accounting (`accounting=`): "coin" (default, §5.1.2 Eqs 39-44) is Deribit-native
— the account is funded in coin, P&L accrues in coin, and the investor keeps a
long exposure to the underlying through realised profits. "usd" (Eqs 46-52)
funds in USD and swaps every coin increment into dollars at the spot prevailing
when it accrues, which removes that exposure; option P&L becomes
S_t V(t) - S_{t0} V(t0) (Eq 46) rather than V(t) - V(t0), so the FX move on the
premium between opening and settlement is part of the result. Position sizing
follows the book: N = Pi^BTC under coin, N = Pi^USD/S_t under USD. The paper
finds risk-adjusted performance is close between the two (§5.2.4) — what really
changes is beta to the coin.

Execution & marking (more realistic than the paper's Assumption 5.1): each leg
is TRADED across the real spread — a long leg buys at the ask, a short leg sells
at the bid — and the open book is MARKED at mid every hedge hour. The half-spread
cost therefore surfaces automatically in the NAV path the instant a position is
opened (bought at ask, immediately worth mid), and the hourly mid mark makes the
NAV continuous so daily vol/Sharpe/MaxDD are meaningful. This replaces the flat
50bp-on-mid of Assumption 5.1; `option_fee_bps` (default 0) adds any explicit
exchange fee ON TOP of the spread. Settlement is the intrinsic coin payoff
(cash-settled, no spread).

Funding (Eq 29): pass `funding_series` = a Deribit funding parquet (from
`utils.deribit_funding.fetch_funding`) for the realised hourly rate charged on
the perp position held each hour (correct Deribit sign: a long perp pays when
the rate is positive). `funding_annual_rate` is a constant fallback; 0 = omit.
"""

from __future__ import annotations

import datetime as dt
import glob
import os
import types

import numpy as np
import pandas as pd
import polars as pl

from volatility_surface.backtest.signal import bs_pricing
from volatility_surface.core.pricing.pricing_models import inverse_delta as _inverse_delta


OPTION_COST_BPS = 50.0      # Assumption 5.1: c = 50bp of traded mid premium
PERP_COST_BPS = 5.0         # Assumption 5.1: eta = 5bp of traded perp notional
MS_PER_DAY = 86_400_000
MS_PER_HOUR = 3_600_000
HOURS_PER_YEAR = 365.0 * 24.0


# --------------------------------------------------------------------------- #
# Structure catalog
# --------------------------------------------------------------------------- #
# A leg is (moneyness, option_type, qty): qty is the SIGNED unit count of the
# LONG version of the structure (+ long, - short, magnitude = ratio units).
# moneyness: "ATM" (nearest forward) or ("delta", d) (strike whose |Black delta|
# is nearest d). Contracts per leg = qty * N, N = current Coin NAV.
#
# Definitions are the paper's §5.2.3 / §5.3.2. The "Short <name>" variant is
# just every qty negated (the paper simulates long and short of each).

_BASE_STRUCTURES: dict[str, list[tuple]] = {
    # --- single leg (Table 1/2) ---
    "ATM Call":     [("ATM", "C", 1.0)],
    "25D Call":     [(("delta", 0.25), "C", 1.0)],
    "10D Call":     [(("delta", 0.10), "C", 1.0)],
    "ATM Put":      [("ATM", "P", 1.0)],
    "25D Put":      [(("delta", 0.25), "P", 1.0)],
    "10D Put":      [(("delta", 0.10), "P", 1.0)],
    # --- multi leg (Table 3/4/5) ---
    "Straddle":     [("ATM", "C", 1.0), ("ATM", "P", 1.0)],
    "25D Strangle": [(("delta", 0.25), "C", 1.0), (("delta", 0.25), "P", 1.0)],
    "10D Strangle": [(("delta", 0.10), "C", 1.0), (("delta", 0.10), "P", 1.0)],
    # call spread: short 1x 50D call, long 2x 25D call (exposure to call skew)
    "Call Spread":  [(("delta", 0.50), "C", -1.0), (("delta", 0.25), "C", 2.0)],
    # put spread: short 1x 50D put, long 2x 25D put (exposure to put skew)
    "Put Spread":   [(("delta", 0.50), "P", -1.0), (("delta", 0.25), "P", 2.0)],
    # 25D risk reversal: short 25D put, long 25D call
    "25D RR":       [(("delta", 0.25), "P", -1.0), (("delta", 0.25), "C", 1.0)],
    # 25D butterfly ratio: long ATM straddle, short 25D strangle (implied convexity)
    "25D ButterFly": [("ATM", "C", 1.0), ("ATM", "P", 1.0),
                       (("delta", 0.25), "C", -1.0), (("delta", 0.25), "P", -1.0)],
}

# back-compat with the lowercase keys used before this catalog existed
_ALIASES = {
    "short_straddle": "Short Straddle", "long_straddle": "Long Straddle",
    "short_25d_strangle": "Short 25D Strangle", "long_25d_strangle": "Long 25D Strangle",
    "short_10d_strangle": "Short 10D Strangle",
}


def build_catalog(structures: list[str] | None = None,
                  sides: tuple[str, ...] = ("Long", "Short")) -> dict[str, list[tuple]]:
    """{"Long Straddle": legs, "Short Straddle": legs, ...}. `structures`
    selects a subset of `_BASE_STRUCTURES` keys (default all); `sides` picks
    which of Long/Short to emit."""
    names = structures or list(_BASE_STRUCTURES)
    cat: dict[str, list[tuple]] = {}
    for nm in names:
        base = _BASE_STRUCTURES[nm]
        for side in sides:
            s = 1.0 if side == "Long" else -1.0
            cat[f"{side} {nm}"] = [(m, o, s * q) for (m, o, q) in base]
    return cat


def catalog_names() -> list[str]:
    return list(build_catalog())


# --------------------------------------------------------------------------- #
# Roll calendar
# --------------------------------------------------------------------------- #
def _friday_0800_grid(start_ms: int, end_ms: int) -> list[int]:
    """Every Friday 08:00 UTC in [start, end] (ms) — the weekly roll grid.
    Deribit weeklies expire Friday 08:00 UTC, so rolling then into the following
    Friday's expiry is a clean non-overlapping hand-off."""
    start = dt.datetime.fromtimestamp(start_ms / 1000, dt.timezone.utc)
    d = start.replace(hour=8, minute=0, second=0, microsecond=0)
    while d.weekday() != 4 or int(d.timestamp() * 1000) < start_ms:
        d += dt.timedelta(days=1)
    out = []
    while int(d.timestamp() * 1000) <= end_ms:
        out.append(int(d.timestamp() * 1000))
        d += dt.timedelta(days=7)
    return out


def _last_friday_grid(start_ms: int, end_ms: int, quarterly: bool = False) -> list[int]:
    """Last-Friday-of-month (monthly) or last-Friday-of-quarter (quarterly)
    08:00 UTC grid — the standard Deribit monthly/quarterly expiries."""
    by_month: dict[tuple, int] = {}
    for ts in _friday_0800_grid(start_ms, end_ms):
        d = dt.datetime.fromtimestamp(ts / 1000, dt.timezone.utc)
        by_month[(d.year, d.month)] = ts  # later Fridays overwrite -> last Friday
    out = sorted(by_month.values())
    if quarterly:
        out = [ts for ts in out
               if dt.datetime.fromtimestamp(ts / 1000, dt.timezone.utc).month in (3, 6, 9, 12)]
    return out


def _daily_0800_grid(start_ms: int, end_ms: int) -> list[int]:
    """Every day at 08:00 UTC — Deribit lists daily expiries for the front few
    days (verified on real chain data: 4 consecutive dailies), so a daily roll
    settles this morning's expiry and opens tomorrow's 1-day option."""
    d = dt.datetime.fromtimestamp(start_ms / 1000, dt.timezone.utc).replace(
        hour=8, minute=0, second=0, microsecond=0)
    while int(d.timestamp() * 1000) < start_ms:
        d += dt.timedelta(days=1)
    out = []
    while int(d.timestamp() * 1000) <= end_ms:
        out.append(int(d.timestamp() * 1000))
        d += dt.timedelta(days=1)
    return out


# --------------------------------------------------------------------------- #
# Market regimes
# --------------------------------------------------------------------------- #
#: Hand-labelled ETH regimes over the 2024-2026 dataset, as (start, end) UTC
#: dates. Deliberately NOT derived from the data: the paper's own regime split
#: (Fig 8/9) sorts monthly returns and cuts the tails at 16%, which is circular
#: when the question is "does this strategy survive a regime it did not fit to".
#: These are the user's own reading of the market and are meant to be edited.
#: Note the gaps (e.g. 2024-09-07 -> 2024-11-02) — those months belong to
#: neither camp and are simply excluded from a regime run.
REGIMES: dict[str, tuple[str, str]] = {
    "bear1": ("2024-06-05", "2024-09-07"),
    "bear2": ("2024-12-07", "2025-04-12"),
    "bear3": ("2025-09-13", "2026-08-03"),
    "bull1": ("2024-11-02", "2024-12-07"),
    "bull2": ("2025-04-12", "2025-09-13"),
}


def _to_ms(when, default=None) -> int | None:
    """Accept 'YYYY-MM-DD', a date/datetime, or epoch ms. Naive dates are UTC."""
    if when is None:
        return default
    if isinstance(when, (int, float)):
        return int(when)
    if isinstance(when, str):
        when = dt.datetime.strptime(when, "%Y-%m-%d")
    if isinstance(when, dt.date) and not isinstance(when, dt.datetime):
        when = dt.datetime(when.year, when.month, when.day)
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return int(when.timestamp() * 1000)


def resolve_window(start=None, end=None, regime: str | None = None) -> tuple[int | None, int | None]:
    """(start_ms, end_ms) from either an explicit window or a named regime.
    An explicit start/end overrides the corresponding regime bound."""
    if regime is not None:
        if regime not in REGIMES:
            raise ValueError(f"unknown regime {regime!r} — one of {list(REGIMES)}")
        r_start, r_end = REGIMES[regime]
        start = start if start is not None else r_start
        end = end if end is not None else r_end
    return _to_ms(start), _to_ms(end)


def roll_grid(start_ms: int, end_ms: int, frequency: str) -> list[int]:
    if frequency == "daily":
        return _daily_0800_grid(start_ms, end_ms)
    if frequency == "weekly":
        return _friday_0800_grid(start_ms, end_ms)
    if frequency == "monthly":
        return _last_friday_grid(start_ms, end_ms, quarterly=False)
    if frequency == "quarterly":
        return _last_friday_grid(start_ms, end_ms, quarterly=True)
    raise ValueError(f"unknown frequency {frequency!r} — daily/weekly/monthly/quarterly")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _perp_series(perp_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pl.read_parquet(perp_path).select(["timestamp_ms", "close"]).sort("timestamp_ms")
    return df["timestamp_ms"].to_numpy(), df["close"].to_numpy()


def _perp_at(perp_ts: np.ndarray, perp_close: np.ndarray, ts_ms: int) -> float:
    """Nearest perp close AT OR BEFORE ts_ms — never look ahead."""
    idx = max(np.searchsorted(perp_ts, ts_ms, side="right") - 1, 0)
    return float(perp_close[idx])


def _iter_snapshots(option_path: str, max_snaps: int | None = None,
                    start_ms: int | None = None, end_ms: int | None = None):
    """Yield (ts_ms, snapshot_df) chronologically from a single parquet or a
    directory of chunk_*.parquet. Snapshot key is `creation_timestamp_x` (see
    `engine._iter_snapshots` for why `file_timestamp` is the wrong key).

    `start_ms`/`end_ms` restrict the window (end is exclusive). Filtering happens
    in polars before materialising to pandas, so a regime run over one month of a
    three-year file does not pay to convert the other 35."""
    if os.path.isdir(option_path):
        paths = sorted(glob.glob(os.path.join(option_path, "chunk_*.parquet")))
    else:
        paths = [option_path]
    n = 0
    for path in paths:
        lf = pl.read_parquet(path)
        if start_ms is not None:
            lf = lf.filter(pl.col("creation_timestamp_x") >= start_ms)
        if end_ms is not None:
            lf = lf.filter(pl.col("creation_timestamp_x") < end_ms)
        if lf.is_empty():
            continue
        day = lf.to_pandas()
        for ts, snap in day.groupby("creation_timestamp_x", sort=True):
            snap = snap.drop_duplicates(subset="instrument_id", keep="first").reset_index(drop=True)
            yield int(ts), snap
            n += 1
            if max_snaps is not None and n >= max_snaps:
                return


# --------------------------------------------------------------------------- #
# Pricing / leg selection helpers
# --------------------------------------------------------------------------- #
def _mid_iv(row) -> float:
    b, a = row.get("bid_iv", np.nan), row.get("ask_iv", np.nan)
    if np.isfinite(b) and np.isfinite(a) and b > 0 and a > 0:
        return 0.5 * (b + a)
    mk = row.get("mark_iv", np.nan)
    return float(mk) if np.isfinite(mk) and mk > 0 else np.nan


def _mid_price(row) -> float:
    b, a = row.get("bid_price", np.nan), row.get("ask_price", np.nan)
    if np.isfinite(b) and np.isfinite(a) and b > 0 and a > 0:
        return 0.5 * (b + a)
    return float(row.get("mark_price", np.nan))


def _select_leg(expiry_slice: pd.DataFrame, F: float, moneyness, option_type: str):
    """Pick the instrument for one leg from the options at a single expiry.
    ATM: two strikes nearest F, then max open interest ("adjacent strike with
    maximum open interest", §5.3). ("delta", d): among that type's strikes, the
    two whose |Black delta| bracket d, then max open interest. Black delta from
    the strike's own mid IV (paper leaves Deribit-vs-Black open, p.21)."""
    sub = expiry_slice[(expiry_slice["option_type"] == option_type) & (expiry_slice["strike"] > 0)].copy()
    if sub.empty:
        return None
    if moneyness == "ATM":
        sub["_dist"] = (sub["strike"] - F).abs()
    else:
        _, target = moneyness
        iv = sub.apply(_mid_iv, axis=1).to_numpy()
        T = float(sub["t"].iloc[0])
        _, delta, _, _, _ = bs_pricing(F, sub["strike"].to_numpy(), T, iv, option_type == "C")
        sub["_dist"] = np.abs(np.abs(delta) - target)
    cand = sub.nsmallest(2, "_dist")
    if cand.empty:
        return None
    oi = cand.get("open_interest")
    if oi is not None and oi.notna().any():
        return cand.loc[cand["open_interest"].idxmax()]
    return cand.iloc[0]


def _select_structure(snap: pd.DataFrame, target_expiry_ms: float, legs_spec: list[tuple],
                      expiry_tol_ms: int = MS_PER_DAY):
    """Resolve abstract legs to concrete instruments at the roll snapshot.
    Returns None (skip the roll) if the target expiry isn't listed or any leg
    can't be filled — never open a partial structure."""
    uniq = np.unique(snap["expiration_timestamp_ms"].to_numpy())
    j = int(np.argmin(np.abs(uniq - target_expiry_ms)))
    if abs(uniq[j] - target_expiry_ms) > expiry_tol_ms:
        return None
    chosen = uniq[j]
    sl = snap[snap["expiration_timestamp_ms"] == chosen]
    if sl.empty:
        return None
    F = float(sl["underlying_price"].iloc[0])
    legs = []
    for moneyness, otype, qty in legs_spec:
        row = _select_leg(sl, F, moneyness, otype)
        if row is None:
            return None
        mid, iv = _mid_price(row), _mid_iv(row)
        if not (np.isfinite(mid) and mid > 0 and np.isfinite(iv) and iv > 0):
            return None
        # Cross the spread on the actual trade: a long leg (qty>0) BUYS at the
        # ask, a short leg (qty<0) SELLS at the bid. Fall back to mid only if
        # that side of the book is missing. The half-spread cost then surfaces
        # automatically once the position is marked at mid (see `_do_hedge`).
        bid = float(row.get("bid_price", np.nan))
        ask = float(row.get("ask_price", np.nan))
        if qty > 0:
            entry_exec = ask if (np.isfinite(ask) and ask > 0) else mid
        else:
            entry_exec = bid if (np.isfinite(bid) and bid > 0) else mid
        legs.append({"instrument_id": row["instrument_id"], "strike": float(row["strike"]),
                     "option_type": otype, "qty": float(qty),
                     "expiration_timestamp_ms": float(chosen),
                     "entry_mid": float(mid), "entry_exec": float(entry_exec),
                     "entry_iv": float(iv), "F_expiry": F})
    return legs


def _acct(cfg, spot: float) -> float:
    """Multiplier turning a Coin-denominated increment into the accounting
    currency: 1 under Coin accounting, the prevailing spot under USD accounting
    (Eqs 47/30 — each coin increment is swapped to USD as it accrues)."""
    return float(spot) if cfg.usd else 1.0


def _leg_net_delta(strike: float, option_type: str, F_expiry: float, iv: float, T: float) -> float:
    """Net (premium-adjusted) delta per contract, Coin units — Delta_k^BTC (Eq
    17), via `pricing_models.inverse_delta` under S=1 / K=strike/F."""
    otype = "c" if option_type == "C" else "p"
    return float(_inverse_delta(1.0, strike / F_expiry, T, 0.0, 0.0, iv, otype))


def _coin_payoff(option_type: str, strike: float, S_T: float) -> float:
    """Inverse-option settlement in Coin: (1/S_T)*max(.,0) (Eq 2)."""
    if S_T <= 0:
        return 0.0
    intrinsic = max(S_T - strike, 0.0) if option_type == "C" else max(strike - S_T, 0.0)
    return intrinsic / S_T


def _spot_at_expiry(snap: pd.DataFrame):
    if "estimated_delivery_price" in snap.columns:
        v = snap["estimated_delivery_price"].dropna()
        if not v.empty and float(v.iloc[0]) > 0:
            return float(v.iloc[0])
    return None


# --------------------------------------------------------------------------- #
# Per-strategy state machine (shared by single- and multi-strategy runs)
# --------------------------------------------------------------------------- #
def _init_state(name: str, legs_spec: list[tuple], frequency: str, grid: list[int]) -> dict:
    return {"name": name, "legs_spec": legs_spec, "frequency": frequency, "grid": grid,
            "next_idx": 0, "open_legs": None, "cum_pnl": 0.0, "hedge_notional": 0.0,
            "last_hedge_F": None, "last_hedge_ts": None,
            "events": [], "rolls": [], "nav_rows": []}


def _accrue(state: dict, ts_ms: int, F_perp: float, cfg) -> tuple[float, float]:
    """Book perp P&L (Eq 26) + funding (Eq 29) on the notional held SINCE THE
    LAST MARK, then advance the mark. Funding accrues pro-rata on the elapsed
    time rather than assuming a fixed hour, so this is correct whether it's
    called on the hourly grid or off-grid at a roll.

    Calling this before the book changes at a roll is what stops the last
    holding period's hedge P&L from being silently dropped."""
    prev = state["hedge_notional"]
    hedge_pnl = funding = 0.0
    if state["last_hedge_F"] is not None and prev != 0.0:
        fx = _acct(cfg, F_perp)  # 1 under Coin, spot under USD (Eqs 47/30)
        hedge_pnl = (F_perp - state["last_hedge_F"]) / F_perp * prev * fx
        hours = max(ts_ms - (state["last_hedge_ts"] or ts_ms), 0) / MS_PER_HOUR
        if cfg.funding_fn is not None:
            funding = -cfg.funding_fn(ts_ms) * hours * prev * fx
        elif cfg.funding_rate:
            funding = -(cfg.funding_rate / HOURS_PER_YEAR) * hours * prev * fx
        state["cum_pnl"] += hedge_pnl + funding
    state["last_hedge_F"] = F_perp
    state["last_hedge_ts"] = ts_ms
    return hedge_pnl, funding


def _compute_book(state: dict, snap_idx: pd.DataFrame, ts_ms: int,
                  spot: float, cfg) -> tuple[float, float]:
    """(target hedge notional in COIN, unrealised mark in the ACCOUNTING ccy) of
    the currently open book. Pure — no side effects, so it serves both hedging
    and marking. The target stays in coin because the perp hedge is a coin
    quantity regardless of which book funds it."""
    target = unrealized = 0.0
    if state["open_legs"] is None:
        return target, unrealized
    for leg in state["open_legs"]:
        iv, F_exp, T, mark = leg["entry_iv"], leg["F_expiry"], None, None
        if leg["instrument_id"] in snap_idx.index:
            r = snap_idx.loc[leg["instrument_id"]]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            m = _mid_iv(r)
            if np.isfinite(m):
                iv = m
            F_exp, T = float(r["underlying_price"]), float(r["t"])
            mp = _mid_price(r)
            if np.isfinite(mp) and mp > 0:
                mark = mp
        if T is None:  # instrument not quoted this snapshot: decay T off expiry
            T = max((leg["expiration_timestamp_ms"] - ts_ms) / (MS_PER_DAY * 365.0), 1e-6)
        if mark is None:  # no live mid: model-mark at current fwd/vol
            mark = float(bs_pricing(F_exp, leg["strike"], T, iv, leg["option_type"] == "C")[0]) / F_exp
        target += -(leg["contracts"] * _leg_net_delta(leg["strike"], leg["option_type"], F_exp, iv, T))
        if cfg.usd:  # Eq 46 marked to today: S_t V(t) - S_{t0} V(t0)
            unrealized += leg["contracts"] * (mark * spot - leg["entry_exec"] * leg["entry_spot"])
        else:
            unrealized += leg["contracts"] * (mark - leg["entry_exec"])
    return target, unrealized


def _rehedge(state: dict, snap_idx: pd.DataFrame, ts_ms: int, spot: float, cfg,
             force: bool = False) -> tuple[float, float]:
    """Move the perp hedge toward the book's target delta, subject to the
    no-trade band, and charge 5bp on whatever is actually traded.

    Band semantics: rebalance only once the delta drift exceeds
    `hedge_band` x current Coin NAV — an ABSOLUTE tolerance in coin units. A
    band relative to the notional itself would be meaningless for the
    delta-neutral structures (straddles sit at target ~ 0, so any relative band
    would trigger on every tick). `force=True` at a roll: the book just changed
    wholesale, so it must be hedged regardless of the band.

    Returns (rebalance_cost, unrealised_mark)."""
    target, unrealized = _compute_book(state, snap_idx, ts_ms, spot, cfg)
    drift = abs(target - state["hedge_notional"])
    # The band compares coin against coin: `drift` is a coin delta, so the NAV
    # it is measured against must be expressed in coin too (under USD accounting
    # the book is carried in dollars).
    nav = abs(cfg.initial_nav + state["cum_pnl"])
    nav_coin = nav / spot if cfg.usd else nav
    rebalance = 0.0
    if force or cfg.hedge_band <= 0.0 or drift > cfg.hedge_band * nav_coin:
        rebalance = cfg.perp_cost_frac * drift * _acct(cfg, spot)
        state["cum_pnl"] -= rebalance
        state["hedge_notional"] = target
    return rebalance, unrealized


def _settle(state: dict, S_T: float, ts_ms: int, cfg) -> None:
    """Realise a held structure at expiry (Eq 39 with V(T,T)=payoff).

    Under USD accounting this is Eq 46, N_k(S_t V_k(t) - S_{t0} V_k(t0)): each
    side of the P&L is converted at the spot prevailing WHEN IT HAPPENED, not
    both at today's — so the FX move on the premium between opening and
    settlement is part of the result, which is the whole point of the USD book.
    """
    opt = 0.0
    for leg in state["open_legs"]:
        payoff = _coin_payoff(leg["option_type"], leg["strike"], S_T)
        if cfg.usd:
            opt += leg["contracts"] * (payoff * S_T - leg["entry_exec"] * leg["entry_spot"])
        else:
            opt += leg["contracts"] * (payoff - leg["entry_exec"])
    state["cum_pnl"] += opt
    state["events"].append({"action": "settle", "timestamp": ts_ms, "option_pnl": opt,
                            "option_cost": 0.0, "hedge_pnl": 0.0, "funding": 0.0, "rebalance_cost": 0.0})
    state["rolls"].append({"action": "settle", "timestamp": ts_ms, "S_T": S_T,
                           "option_pnl": opt, "nav_after": cfg.initial_nav + state["cum_pnl"]})


def _open(state: dict, legs: list[dict], ts_ms: int, spot: float, cfg) -> None:
    """Open a structure sized to current NAV; pay entry costs (Eq 40 option cost).

    Sizing is always a CONTRACT count, which is a coin quantity: under Coin
    accounting N = Pi^BTC (Eq 31), under USD accounting N = Pi^USD/S_t (§5.2.3)
    — the same economic size, expressed from whichever book funds it."""
    nav = cfg.initial_nav + state["cum_pnl"]
    N = (nav / spot if cfg.usd else nav) * cfg.size_multiple
    cost = 0.0
    for leg in legs:
        leg["contracts"] = leg["qty"] * N
        leg["entry_spot"] = float(spot)
        # Explicit fee ON TOP of the spread (which is already paid via entry@bid/ask
        # marked at mid). Defaults to 0 — set option_fee_bps to add exchange fees.
        cost += cfg.opt_fee_frac * abs(leg["contracts"]) * leg["entry_mid"] * _acct(cfg, spot)
    state["cum_pnl"] -= cost
    state["open_legs"] = legs
    state["events"].append({"action": "open", "timestamp": ts_ms, "option_pnl": 0.0,
                            "option_cost": cost, "hedge_pnl": 0.0, "funding": 0.0, "rebalance_cost": 0.0})
    state["rolls"].append({"action": "open", "timestamp": ts_ms, "N": N, "entry_cost": cost,
                           "legs": [(l["instrument_id"], l["strike"], l["option_type"],
                                     l["qty"], l["entry_mid"]) for l in legs]})


def _hedge_event(state: dict, ts_ms: int, hedge_pnl: float, funding: float,
                 rebalance: float) -> None:
    if hedge_pnl or funding or rebalance:
        state["events"].append({"action": "hedge", "timestamp": ts_ms, "option_pnl": 0.0,
                                "option_cost": 0.0, "hedge_pnl": hedge_pnl,
                                "funding": funding, "rebalance_cost": rebalance})


def _do_rolls(state: dict, snap: pd.DataFrame, snap_idx: pd.DataFrame, ts_ms: int,
              F_perp: float, perp_ts: np.ndarray, perp_close: np.ndarray, cfg) -> None:
    grid = state["grid"]
    while state["next_idx"] < len(grid) and ts_ms >= grid[state["next_idx"]]:
        roll_ts = grid[state["next_idx"]]
        # Book the hedge P&L + funding accrued on the position held up to this
        # moment BEFORE the book changes. Previously the notional was zeroed and
        # the mark reset without booking, silently discarding the final holding
        # period's hedge P&L — the expiry hour, when the option's delta is at its
        # most extreme, so the loss was both systematic and directional.
        hedge_pnl, funding = _accrue(state, ts_ms, F_perp, cfg)
        _hedge_event(state, ts_ms, hedge_pnl, funding, 0.0)

        if state["open_legs"] is not None:
            S_T = _spot_at_expiry(snap) or _perp_at(perp_ts, perp_close, roll_ts)
            _settle(state, S_T, roll_ts, cfg)
            state["open_legs"] = None
        if state["next_idx"] + 1 < len(grid):
            target_exp = grid[state["next_idx"] + 1]
            # Tolerance must be under HALF the roll spacing, or a roll can match
            # the neighbouring expiry instead of its own target — for daily rolls
            # the just-settled expiry sits exactly one day away, i.e. right on a
            # flat 1-day tolerance. Half-spacing makes the nearest-expiry match
            # unambiguous at every frequency.
            tol = min(MS_PER_DAY, (target_exp - roll_ts) // 2)
            legs = _select_structure(snap, target_exp, state["legs_spec"], expiry_tol_ms=tol)
            if legs is not None:
                _open(state, legs, roll_ts, F_perp, cfg)
        state["next_idx"] += 1

        # Hedge the NEW book immediately rather than waiting for the next hour
        # boundary (which left it naked for up to an hour every roll). The perp
        # is NOT zeroed in between, so the cost charged here is the netted trade
        # from the old hedge straight to the new one — what a desk would do when
        # one expiry settles and the next is opened at the same moment.
        rebalance, _ = _rehedge(state, snap_idx, ts_ms, F_perp, cfg, force=True)
        _hedge_event(state, ts_ms, 0.0, 0.0, rebalance)


def _do_hedge(state: dict, snap_idx: pd.DataFrame, ts_ms: int, F_perp: float, cfg) -> None:
    """Hourly mark: accrue perp P&L + funding on the held notional, then
    rebalance toward the book's delta subject to the no-trade band."""
    hedge_pnl, funding = _accrue(state, ts_ms, F_perp, cfg)
    rebalance, unrealized = _rehedge(state, snap_idx, ts_ms, F_perp, cfg)
    _hedge_event(state, ts_ms, hedge_pnl, funding, rebalance)
    # NAV = starting coin + realised P&L + unrealised mark of the open book, so
    # the path is continuous hour-to-hour (the option no longer only "appears"
    # at settlement) — this is what makes daily vol/Sharpe/MaxDD meaningful.
    state["nav_rows"].append({"timestamp": ts_ms,
                              "coin_nav": cfg.initial_nav + state["cum_pnl"] + unrealized,
                              "cum_pnl": state["cum_pnl"], "unrealized": unrealized,
                              "coin_px": F_perp})


def _finalize(state: dict, cfg) -> dict:
    return {"events": state["events"], "nav": pd.DataFrame(state["nav_rows"]),
            "rolls": state["rolls"],
            "params": {"structure": state["name"], "frequency": state["frequency"],
                       "initial_coin": cfg.initial_coin, "size_multiple": cfg.size_multiple,
                       "accounting": cfg.accounting, "initial_nav": cfg.initial_nav,
                       "regime": cfg.regime, "window": cfg.window,
                       "nav_unit": "USD" if cfg.usd else "coin",
                       "funding_annual_rate": cfg.funding_rate,
                       "option_fee_bps": cfg.opt_fee_frac * 10_000.0,
                       "perp_cost_bps": PERP_COST_BPS, "hedge_band": cfg.hedge_band,
                       "execution": "bid/ask entry, mid MTM"}}


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def _resolve_funding(funding_series):
    """funding_series may be a parquet path (str) of Deribit funding history, a
    callable ts_ms -> interest_1h, or None. Returns a callable or None."""
    if funding_series is None:
        return None
    if callable(funding_series):
        return funding_series
    from volatility_surface.utils.deribit_funding import load_funding_lookup
    return load_funding_lookup(funding_series)


def run_all_strategies(option_path: str, perp_path: str,
                       catalog: dict[str, list[tuple]] | None = None,
                       frequencies: tuple[str, ...] = ("weekly",),
                       initial_coin: float = 1.0, size_multiple: float = 1.0,
                       funding_annual_rate: float = 0.0, funding_series=None,
                       option_fee_bps: float = 0.0, hedge_band: float = 0.05,
                       accounting: str = "coin", initial_usd: float | None = None,
                       start=None, end=None, regime: str | None = None,
                       max_snaps: int | None = None, verbose: bool = True) -> dict[str, dict]:
    """Backtest an entire catalog x frequencies in ONE pass over the option
    data. Returns {strategy_name: result_dict}; hand the whole dict to
    `roll_results.results_table` for the paper-style table. When more than one
    frequency is requested, each name is suffixed " [freq]".

    Funding (Eq 29): pass `funding_series` = path to a Deribit funding parquet
    (see `utils.deribit_funding.fetch_funding`) for realised hourly funding — the
    faithful term. `funding_annual_rate` is a constant fallback used only when
    `funding_series` is None.

    `hedge_band` (default 0.05): no-trade band on the delta hedge — rebalance
    only once the drift exceeds 5% of Coin NAV, instead of forcing a trade every
    hour. Hourly rebalancing of a short-dated ATM book is dominated by gamma
    churn near expiry and measured at ~18% of gross option P&L on this data; the
    paper itself notes desks rebalance within bands and that "optimised hedging
    produces better risk-adjusted results" (§5.2). Set 0.0 to restore the
    unconditional hourly rebalance."""
    catalog = catalog or build_catalog()
    if accounting not in ("coin", "usd"):
        raise ValueError(f"accounting must be 'coin' or 'usd', got {accounting!r}")
    cfg = types.SimpleNamespace(initial_coin=initial_coin, size_multiple=size_multiple,
                                usd=(accounting == "usd"), accounting=accounting,
                                opt_fee_frac=option_fee_bps / 10_000.0,
                                perp_cost_frac=PERP_COST_BPS / 10_000.0,
                                funding_rate=funding_annual_rate,
                                funding_fn=_resolve_funding(funding_series),
                                hedge_band=hedge_band, regime=regime, window=None)

    start_ms, end_ms = resolve_window(start, end, regime)
    cfg.window = (
        dt.datetime.fromtimestamp(start_ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d") if start_ms else None,
        dt.datetime.fromtimestamp(end_ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d") if end_ms else None,
    )

    perp_ts, perp_close = _perp_series(perp_path)
    try:
        first_ts = next(_iter_snapshots(option_path, max_snaps=1,
                                        start_ms=start_ms, end_ms=end_ms))[0]
    except StopIteration:
        # Report what the file DOES cover: an empty window is nearly always a
        # source-data gap rather than a bad window, and the difference is not
        # something the caller can see from the request alone.
        def _fmt(ms):
            return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d") if ms else "-"
        try:
            probe = (pl.scan_parquet(os.path.join(option_path, "chunk_*.parquet")
                                     if os.path.isdir(option_path) else option_path)
                       .select(pl.col("creation_timestamp_x").min().alias("lo"),
                               pl.col("creation_timestamp_x").max().alias("hi"))
                       .collect())
            have = f"{_fmt(probe['lo'][0])} .. {_fmt(probe['hi'][0])}"
        except Exception:
            have = "unknown"
        raise ValueError(
            f"no option snapshots between {_fmt(start_ms)} and {_fmt(end_ms)}"
            f"{f' (regime {regime!r})' if regime else ''}. "
            f"The file spans {have} — but coverage inside that span can still be "
            f"patchy; run utils/validate_dataset.py to see snapshots per month.")
    # The roll grid must live inside the window, not span the whole file: a
    # regime run should open its first position at the first roll date ON OR
    # AFTER the window opens, and stop at the window's end.
    end_ts = min(int(perp_ts[-1]), end_ms) if end_ms is not None else int(perp_ts[-1])

    # The account's starting value, in whichever currency funds it. Under USD
    # accounting the paper sets D_0^USD = S_0 (the price of one coin at
    # inception, §5.2.3) so the two books start economically identical and their
    # returns are directly comparable.
    S0 = _perp_at(perp_ts, perp_close, first_ts)
    cfg.initial_nav = (
        (initial_usd if initial_usd is not None else S0 * initial_coin)
        if cfg.usd else initial_coin
    )

    multi_freq = len(frequencies) > 1
    states: list[dict] = []
    for freq in frequencies:
        grid = roll_grid(first_ts, end_ts, freq)
        for name, legs in catalog.items():
            label = f"{name} [{freq}]" if multi_freq else name
            states.append(_init_state(label, legs, freq, grid))

    if verbose:
        w0 = dt.datetime.fromtimestamp(first_ts / 1000, dt.timezone.utc)
        w1 = dt.datetime.fromtimestamp(end_ts / 1000, dt.timezone.utc)
        label = f" [{regime}]" if regime else ""
        n_rolls = len(states[0]["grid"]) if states else 0
        print(f"[window]{label} {w0:%Y-%m-%d} -> {w1:%Y-%m-%d} "
              f"({(w1 - w0).days} days, {n_rolls} {frequencies[0]} roll dates)")
        if n_rolls < 4:
            print("[window] WARNING: fewer than 4 rolls in this window — annualised "
                  "figures (P.a., Sharpe) are extrapolated from very few observations")

    last_hour = None
    n = 0
    for ts_ms, snap in _iter_snapshots(option_path, max_snaps=max_snaps,
                                       start_ms=start_ms, end_ms=end_ms):
        n += 1
        hour = ts_ms // MS_PER_HOUR
        need_hedge = last_hour is None or hour != last_hour
        # Rolls now hedge the new book on the spot, so they need the indexed
        # snapshot too — but building it costs real time on 240k snapshots, so
        # only do it when something actually happens this snapshot.
        need_roll = any(st["next_idx"] < len(st["grid"]) and ts_ms >= st["grid"][st["next_idx"]]
                        for st in states)
        if not (need_hedge or need_roll):
            continue
        F_perp = _perp_at(perp_ts, perp_close, ts_ms)
        snap_idx = snap.set_index("instrument_id")
        if need_roll:
            for st in states:
                _do_rolls(st, snap, snap_idx, ts_ms, F_perp, perp_ts, perp_close, cfg)
        if need_hedge:
            for st in states:
                _do_hedge(st, snap_idx, ts_ms, F_perp, cfg)
            last_hour = hour
        if verbose and n % 20_000 == 0:
            d = dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc)
            print(f"[{n}] {d:%Y-%m-%d %H:%M} — {len(states)} strategies running")

    for st in states:
        if st["open_legs"] is not None:
            _settle(st, _perp_at(perp_ts, perp_close, ts_ms), ts_ms, cfg)
    return {st["name"]: _finalize(st, cfg) for st in states}


def run_roll_backtest(option_path: str, perp_path: str,
                      structure: str = "Short Straddle", frequency: str = "weekly",
                      initial_coin: float = 1.0, size_multiple: float = 1.0,
                      funding_annual_rate: float = 0.0, funding_series=None,
                      option_fee_bps: float = 0.0, hedge_band: float = 0.05,
                      accounting: str = "coin", initial_usd: float | None = None,
                      start=None, end=None, regime: str | None = None,
                      max_snaps: int | None = None, verbose: bool = True) -> dict:
    """Single-strategy wrapper on `run_all_strategies`. `structure` is a catalog
    name ("Short Straddle", "Long 25D Strangle", ...) or an old lowercase alias
    ("short_straddle")."""
    name = _ALIASES.get(structure, structure)
    full = build_catalog()
    if name not in full:
        raise ValueError(f"unknown structure {structure!r} — one of {list(full)}")
    res = run_all_strategies(option_path, perp_path, catalog={name: full[name]},
                             frequencies=(frequency,), initial_coin=initial_coin,
                             size_multiple=size_multiple, funding_annual_rate=funding_annual_rate,
                             funding_series=funding_series, option_fee_bps=option_fee_bps,
                             hedge_band=hedge_band, accounting=accounting,
                             initial_usd=initial_usd, start=start, end=end, regime=regime,
                             max_snaps=max_snaps, verbose=verbose)
    return res[name]
