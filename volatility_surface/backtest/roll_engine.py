"""
Systematic roll-based option-strategy backtester — Lucic & Sepp (2024),
"Valuation and hedging of cryptocurrency inverse options" (§5.2-5.3).

This is a DIFFERENT engine from `engine.py`. `engine.py` is opportunistic
relative-value: it calibrates a surface every snapshot, flags mispriced
"red cells", and trades them as they appear. This module instead implements
the paper's *systematic* strategies: a fixed calendar of rolls (weekly Friday
/ monthly / quarterly), at each roll selling or buying a FIXED structure
(ATM straddle to start; strangles/spreads/RR/butterfly drop in via the leg
spec), held to maturity, delta-hedged on an hourly grid with the inverse
perpetual, and sized proportionally to the current Coin NAV (auto-compounding,
Eq N(t0)=Pi^BTC(t0)).

What it reuses from the rest of the project (so nothing drifts):
  - `pricing_models.inverse_delta` — the premium-adjusted "Net Delta" (Eq 17),
    the ONLY correct hedge ratio for a coin-settled option. NOT plain Black
    delta.
  - `signal.bs_pricing` — the one Black-76 implementation the live screen,
    the GIF recorder and `engine.py` all price off.
  - The hedge-P&L convention of `engine.py` §5: hedge_notional is the
    Coin-denominated net delta Delta^BTC, and per-hour P&L is
    (F_h - F_{h-1})/F_h * Delta^BTC — exactly Eq (26)'s last form. We deliberately
    mirror `engine.py` rather than also carry the F_t^{Tk}/F_t perp-vs-dated
    scaling of Eq (25): for the 7-day options this is a funding-basis term of
    order 1e-3 that `engine.py` already omits; keeping the two engines
    identical here is worth more than that second-order correction. Documented
    so it's a choice, not an oversight.

Accounting: Coin-based (§5.1.2 "Coin accounting", Eqs 39-44) — the native
Deribit measure and the one the paper's headline short-straddle figure (Fig 6)
uses. USD accounting (Eqs 46-52) is a thin wrapper on top and is left for the
next phase.

Funding (DeltaFunding^BTC, Eq 29): the paper uses realised Deribit hourly
funding. The perp series we have is OHLCV close only (no funding column, no
index), so `funding_annual_rate` defaults to 0.0 and is reported as its own
P&L line. Short-ATM funding ran ~-8%/yr in the paper (Fig 3) — non-trivial but
not sign-flipping over a ~5.5-month window; pass a constant or a callable to
approximate it once a funding/basis series is available.
"""

from __future__ import annotations

import datetime as dt
import glob
import os

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
# Structure specification
# --------------------------------------------------------------------------- #
# A structure is a list of legs; each leg is (moneyness, option_type, qty_sign):
#   moneyness  : "ATM"  -> strike nearest the forward
#                ("delta", 0.25) -> strike whose |Black delta| is nearest 0.25
#   option_type: "C" or "P"
#   qty_sign   : +1 long, -1 short.  The per-leg contract count is
#                qty_sign * size_multiple * N, N = current Coin NAV.
# Extend the paper's other structures by editing this dict only.

STRUCTURES: dict[str, list[tuple]] = {
    "short_straddle": [("ATM", "C", -1.0), ("ATM", "P", -1.0)],
    "long_straddle":  [("ATM", "C", +1.0), ("ATM", "P", +1.0)],
    "short_25d_strangle": [(("delta", 0.25), "C", -1.0), (("delta", 0.25), "P", -1.0)],
    "long_25d_strangle":  [(("delta", 0.25), "C", +1.0), (("delta", 0.25), "P", +1.0)],
    "short_10d_strangle": [(("delta", 0.10), "C", -1.0), (("delta", 0.10), "P", -1.0)],
    # spreads / RR / butterfly follow the same pattern; add when Phase 3 lands.
}


# --------------------------------------------------------------------------- #
# Roll calendar
# --------------------------------------------------------------------------- #
def _friday_0800_grid(start_ms: int, end_ms: int) -> list[int]:
    """Every Friday 08:00:00 UTC timestamp (ms) in [start_ms, end_ms] — the
    weekly roll grid. Deribit weeklies expire Friday 08:00 UTC, so a roll at
    Friday 08:00 into the following Friday's expiry is a clean, non-overlapping
    hand-off: the held weekly settles exactly as the new one is opened."""
    start = dt.datetime.fromtimestamp(start_ms / 1000, dt.timezone.utc)
    d = start.replace(hour=8, minute=0, second=0, microsecond=0)
    # advance to the first Friday-08:00 at or after start
    while d.weekday() != 4 or int(d.timestamp() * 1000) < start_ms:
        d += dt.timedelta(days=1)
    out = []
    while int(d.timestamp() * 1000) <= end_ms:
        out.append(int(d.timestamp() * 1000))
        d += dt.timedelta(days=7)
    return out


def _last_friday_0800_grid(start_ms: int, end_ms: int, quarterly: bool = False) -> list[int]:
    """Last-Friday-of-month (monthly) or last-Friday-of-quarter (quarterly)
    08:00 UTC grid — the paper's other two roll frequencies. A month's last
    Friday is the standard Deribit monthly expiry."""
    weeklies = _friday_0800_grid(start_ms, end_ms)
    by_month: dict[tuple, int] = {}
    for ts in weeklies:
        d = dt.datetime.fromtimestamp(ts / 1000, dt.timezone.utc)
        by_month[(d.year, d.month)] = ts  # later Fridays overwrite -> last Friday
    last_fridays = sorted(by_month.values())
    if not quarterly:
        return last_fridays
    out = []
    for ts in last_fridays:
        d = dt.datetime.fromtimestamp(ts / 1000, dt.timezone.utc)
        if d.month in (3, 6, 9, 12):
            out.append(ts)
    return out


def roll_grid(start_ms: int, end_ms: int, frequency: str) -> list[int]:
    if frequency == "weekly":
        return _friday_0800_grid(start_ms, end_ms)
    if frequency == "monthly":
        return _last_friday_0800_grid(start_ms, end_ms, quarterly=False)
    if frequency == "quarterly":
        return _last_friday_0800_grid(start_ms, end_ms, quarterly=True)
    raise ValueError(f"unknown frequency {frequency!r} — weekly/monthly/quarterly")


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


def _iter_snapshots(option_path: str, max_snaps: int | None = None):
    """Yield (ts_ms, snapshot_df) in chronological order from either a single
    parquet or a directory of chunk_*.parquet files. Snapshot key is
    `creation_timestamp_x` (true ms-precision snapshot boundary — see
    `engine._iter_snapshots` for why `file_timestamp` is the wrong key)."""
    if os.path.isdir(option_path):
        paths = sorted(glob.glob(os.path.join(option_path, "chunk_*.parquet")))
    else:
        paths = [option_path]
    n = 0
    for path in paths:
        day = pl.read_parquet(path).to_pandas()
        if day.empty:
            continue
        for ts, snap in day.groupby("creation_timestamp_x", sort=True):
            snap = snap.drop_duplicates(subset="instrument_id", keep="first").reset_index(drop=True)
            yield int(ts), snap
            n += 1
            if max_snaps is not None and n >= max_snaps:
                return


# --------------------------------------------------------------------------- #
# Leg / strike selection
# --------------------------------------------------------------------------- #
def _mid_iv(row: pd.Series) -> float:
    b, a = row.get("bid_iv", np.nan), row.get("ask_iv", np.nan)
    if np.isfinite(b) and np.isfinite(a) and b > 0 and a > 0:
        return 0.5 * (b + a)
    mk = row.get("mark_iv", np.nan)
    return float(mk) if np.isfinite(mk) and mk > 0 else np.nan


def _mid_price(row: pd.Series) -> float:
    b, a = row.get("bid_price", np.nan), row.get("ask_price", np.nan)
    if np.isfinite(b) and np.isfinite(a) and b > 0 and a > 0:
        return 0.5 * (b + a)
    return float(row.get("mark_price", np.nan))


def _select_leg(expiry_slice: pd.DataFrame, F: float, moneyness, option_type: str) -> pd.Series | None:
    """Pick the instrument for one leg from the options at a single expiry.

    ATM: the two strikes nearest the forward F, then the one with higher open
    interest ("adjacent strike with maximum open interest", §5.3).
    ("delta", d): among that type's strikes, the two whose |Black delta| bracket
    the target d, then max open interest between them. Black delta N(d1) is
    computed from the strike's own mid IV — the paper leaves Deribit-vs-Black
    delta open (p.21); Black delta from mid IV is the self-contained choice and
    is nearly moot for the ATM structures we start with."""
    sub = expiry_slice[expiry_slice["option_type"] == option_type].copy()
    sub = sub[sub["strike"] > 0]
    if sub.empty:
        return None

    if moneyness == "ATM":
        sub["_dist"] = (sub["strike"] - F).abs()
        cand = sub.nsmallest(2, "_dist")
    else:
        _, target = moneyness  # ("delta", 0.25)
        iv = sub.apply(_mid_iv, axis=1).to_numpy()
        T = float(sub["t"].iloc[0])
        _, delta, _, _, _ = bs_pricing(F, sub["strike"].to_numpy(), T, iv,
                                        (option_type == "C"))
        sub["_dist"] = np.abs(np.abs(delta) - target)
        cand = sub.nsmallest(2, "_dist")

    if cand.empty:
        return None
    oi = cand.get("open_interest")
    if oi is not None and oi.notna().any():
        return cand.loc[cand["open_interest"].idxmax()]
    return cand.iloc[0]


def _select_structure(snap: pd.DataFrame, target_expiry_ms: float, legs_spec: list[tuple],
                       expiry_tol_ms: int = MS_PER_DAY) -> list[dict] | None:
    """Resolve a structure's abstract legs to concrete instruments at the roll
    snapshot. Returns None (skip the roll) if the target expiry isn't listed or
    any leg can't be filled — never open a partial structure."""
    exp = snap["expiration_timestamp_ms"].to_numpy()
    uniq = np.unique(exp)
    j = int(np.argmin(np.abs(uniq - target_expiry_ms)))
    if abs(uniq[j] - target_expiry_ms) > expiry_tol_ms:
        return None
    chosen_expiry = uniq[j]
    sl = snap[snap["expiration_timestamp_ms"] == chosen_expiry]
    if sl.empty:
        return None
    F = float(sl["underlying_price"].iloc[0])

    legs = []
    for moneyness, otype, qty_sign in legs_spec:
        row = _select_leg(sl, F, moneyness, otype)
        if row is None:
            return None
        mid = _mid_price(row)
        iv = _mid_iv(row)
        if not (np.isfinite(mid) and mid > 0 and np.isfinite(iv) and iv > 0):
            return None
        legs.append({
            "instrument_id": row["instrument_id"],
            "strike": float(row["strike"]),
            "option_type": otype,
            "qty_sign": float(qty_sign),
            "expiration_timestamp_ms": float(chosen_expiry),
            "entry_mid": float(mid),
            "entry_iv": float(iv),
            "F_expiry": F,
        })
    return legs


# --------------------------------------------------------------------------- #
# Greeks for the held book
# --------------------------------------------------------------------------- #
def _leg_net_delta(leg: dict, F_expiry: float, iv: float, T: float) -> float:
    """Net (premium-adjusted) delta per contract for one held leg, in Coin
    units — `pricing_models.inverse_delta` under the project's S=1 / K=strike/F
    normalisation. This is Delta_k^BTC in Eq (17)/(25)."""
    K_norm = leg["strike"] / F_expiry
    otype = "c" if leg["option_type"] == "C" else "p"
    return float(_inverse_delta(1.0, K_norm, T, 0.0, 0.0, iv, otype))


def _coin_payoff(option_type: str, strike: float, S_T: float) -> float:
    """Inverse-option settlement value in Coin: (1/S_T)*max(.,0) (Eq 2)."""
    if S_T <= 0:
        return 0.0
    intrinsic = max(S_T - strike, 0.0) if option_type == "C" else max(strike - S_T, 0.0)
    return intrinsic / S_T


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_roll_backtest(option_path: str,
                      perp_path: str,
                      structure: str = "short_straddle",
                      frequency: str = "weekly",
                      initial_coin: float = 1.0,
                      size_multiple: float = 1.0,
                      funding_annual_rate: float = 0.0,
                      max_snaps: int | None = None,
                      verbose: bool = True) -> dict:
    """Backtest a systematic roll structure with hourly delta-hedging, Coin
    accounting.

    option_path : single parquet OR a directory of chunk_*.parquet (cleaned
                  schema: strike, option_type C/P, t, k, underlying_price,
                  estimated_delivery_price, bid_price/ask_price, bid_iv/ask_iv,
                  open_interest, expiration_timestamp_ms, creation_timestamp_x).
    perp_path   : perp OHLCV parquet (timestamp_ms, close) — the hedging instr.
    structure   : key into STRUCTURES.
    frequency   : "weekly" | "monthly" | "quarterly".
    initial_coin: Pi_0^BTC, the starting Coin NAV (paper uses 1).
    size_multiple: leg contracts = qty_sign * size_multiple * NAV. 1.0 = the
                  paper's N(t0)=Pi^BTC(t0) sizing (auto-compounding).
    funding_annual_rate: constant annualised funding applied to the perp hedge
                  (Eq 29 proxy). 0.0 = omitted (default; see module docstring).

    Returns a dict: {"events", "nav", "rolls", "params"}. `nav` is a DataFrame
    (timestamp, coin_nav, cum_pnl) sampled at hedge times; hand `events`/`nav`
    to `roll_results.summarize_rolls` for headline metrics.
    """
    if structure not in STRUCTURES:
        raise ValueError(f"unknown structure {structure!r} — one of {list(STRUCTURES)}")
    legs_spec = STRUCTURES[structure]

    perp_ts, perp_close = _perp_series(perp_path)

    # roll calendar spans the option data
    first_ts = next(_iter_snapshots(option_path, max_snaps=1))[0]
    # cheap end estimate: last perp timestamp (>= last option ts in practice)
    grid = roll_grid(first_ts, int(perp_ts[-1]), frequency)
    grid_set = sorted(grid)

    coin_nav = float(initial_coin)
    cum_pnl = 0.0                       # realised cumulative Coin P&L (Eq 43)
    open_legs: list[dict] | None = None  # currently held structure
    next_roll_idx = 0                    # pointer into grid_set

    hedge_notional = 0.0                 # Delta^BTC currently offset by the perp, Coin units
    last_hedge_F = None
    last_hedge_hour = None

    events: list[dict] = []
    nav_rows: list[dict] = []
    rolls: list[dict] = []

    def _settle(legs, S_T, ts_ms):
        """Realise a held structure at expiry (Eq 39 with V(T,T)=payoff)."""
        nonlocal cum_pnl
        opt_pnl = 0.0
        for leg in legs:
            payoff = _coin_payoff(leg["option_type"], leg["strike"], S_T)
            # M_k (V(T) - V(t0)); M_k = qty_sign * contracts
            opt_pnl += leg["contracts"] * (payoff - leg["entry_mid"])
        cum_pnl += opt_pnl
        rolls.append({"action": "settle", "timestamp": ts_ms, "S_T": S_T,
                      "option_pnl": opt_pnl, "coin_nav_after": initial_coin + cum_pnl})
        events.append({"action": "settle", "timestamp": ts_ms, "option_pnl": opt_pnl,
                       "option_cost": 0.0, "hedge_pnl": 0.0, "funding": 0.0, "rebalance_cost": 0.0})

    def _open(legs, ts_ms):
        """Open a structure sized to current NAV; pay entry costs (Eq 40 cost
        term on the option side)."""
        nonlocal cum_pnl, coin_nav
        coin_nav = initial_coin + cum_pnl
        N = coin_nav * size_multiple
        entry_cost = 0.0
        for leg in legs:
            leg["contracts"] = leg["qty_sign"] * N
            entry_cost += OPTION_COST_BPS / 10_000.0 * abs(leg["contracts"]) * leg["entry_mid"]
        cum_pnl -= entry_cost
        rolls.append({"action": "open", "timestamp": ts_ms, "structure": structure,
                      "N": N, "entry_cost": entry_cost,
                      "legs": [(l["instrument_id"], l["strike"], l["option_type"],
                                l["qty_sign"], l["entry_mid"]) for l in legs]})
        events.append({"action": "open", "timestamp": ts_ms, "option_pnl": 0.0,
                       "option_cost": entry_cost, "hedge_pnl": 0.0, "funding": 0.0,
                       "rebalance_cost": 0.0})

    def _spot_at_expiry(snap):
        col = "estimated_delivery_price"
        if col in snap.columns:
            v = snap[col].dropna()
            if not v.empty and float(v.iloc[0]) > 0:
                return float(v.iloc[0])
        return None

    n = 0
    for ts_ms, snap in _iter_snapshots(option_path, max_snaps=max_snaps):
        n += 1

        # ---- 1. roll boundary: settle expiring, open next ----
        while next_roll_idx < len(grid_set) and ts_ms >= grid_set[next_roll_idx]:
            roll_ts = grid_set[next_roll_idx]
            if open_legs is not None:
                S_T = _spot_at_expiry(snap) or _perp_at(perp_ts, perp_close, roll_ts)
                _settle(open_legs, S_T, roll_ts)
                open_legs = None
                hedge_notional = 0.0  # book flat until new structure is hedged below

            # open next: target expiry = the following roll on the grid
            if next_roll_idx + 1 < len(grid_set):
                target_expiry = grid_set[next_roll_idx + 1]
                legs = _select_structure(snap, target_expiry, legs_spec)
                if legs is not None:
                    _open(legs, roll_ts)
                    open_legs = legs
            next_roll_idx += 1
            last_hedge_F = None  # force a hedge re-seed on the new book

        # ---- 2. hourly delta-hedge + mark ----
        hour = ts_ms // MS_PER_HOUR
        if last_hedge_hour is None or hour != last_hedge_hour:
            F_perp = _perp_at(perp_ts, perp_close, ts_ms)

            hedge_pnl = 0.0
            if last_hedge_F is not None and hedge_notional != 0.0:
                hedge_pnl = (F_perp - last_hedge_F) / F_perp * hedge_notional  # Eq 26
                cum_pnl += hedge_pnl

            # recompute target Delta^BTC of the held book from current IV
            target_notional = 0.0
            if open_legs is not None:
                idx = snap.set_index("instrument_id")
                for leg in open_legs:
                    iv, F_exp, T = leg["entry_iv"], leg["F_expiry"], None
                    if leg["instrument_id"] in idx.index:
                        r = idx.loc[leg["instrument_id"]]
                        r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
                        iv = _mid_iv(r) if np.isfinite(_mid_iv(r)) else iv
                        F_exp = float(r["underlying_price"])
                        T = float(r["t"])
                    if T is None:
                        # instrument not quoted this snapshot: decay T off expiry
                        T = max((leg["expiration_timestamp_ms"] - ts_ms) / (MS_PER_DAY * 365.0), 1e-6)
                    nd = _leg_net_delta(leg, F_exp, iv, T)
                    target_notional += -(leg["contracts"] * nd)  # offset the book delta

            rebalance_cost = PERP_COST_BPS / 10_000.0 * abs(target_notional - hedge_notional)
            funding = 0.0
            if funding_annual_rate and target_notional != 0.0:
                funding = -(funding_annual_rate / HOURS_PER_YEAR) * target_notional
            cum_pnl += funding - rebalance_cost

            if hedge_pnl or rebalance_cost or funding:
                events.append({"action": "hedge", "timestamp": ts_ms, "option_pnl": 0.0,
                               "option_cost": 0.0, "hedge_pnl": hedge_pnl, "funding": funding,
                               "rebalance_cost": rebalance_cost})

            hedge_notional = target_notional
            last_hedge_F = F_perp
            last_hedge_hour = hour

            coin_nav = initial_coin + cum_pnl
            nav_rows.append({"timestamp": ts_ms, "coin_nav": coin_nav, "cum_pnl": cum_pnl})

        if verbose and n % 20_000 == 0:
            d = dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc)
            print(f"[{n}] {d:%Y-%m-%d %H:%M} — NAV {coin_nav:.4f} coin, {len(rolls)} roll events")

    # settle any structure still open at end of data (mark to last payoff proxy)
    if open_legs is not None:
        S_T = _perp_at(perp_ts, perp_close, ts_ms)
        _settle(open_legs, S_T, ts_ms)

    return {
        "events": events,
        "nav": pd.DataFrame(nav_rows),
        "rolls": rolls,
        "params": {"structure": structure, "frequency": frequency,
                   "initial_coin": initial_coin, "size_multiple": size_multiple,
                   "funding_annual_rate": funding_annual_rate,
                   "option_cost_bps": OPTION_COST_BPS, "perp_cost_bps": PERP_COST_BPS},
    }
