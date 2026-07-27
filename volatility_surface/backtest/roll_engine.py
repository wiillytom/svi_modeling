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

Accounting: Coin-based (§5.1.2, Eqs 39-44) — Deribit-native, the measure the
paper's headline figures use. USD accounting (Eqs 46-52) is a thin wrapper for
a later phase.

Funding (Eq 29): the perp series is OHLCV close only (no funding column), so
`funding_annual_rate` defaults to 0.0 and is reported as its own P&L line. Pass
a constant to approximate it (short-ATM funding ran ~-8%/yr in the paper).
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


def roll_grid(start_ms: int, end_ms: int, frequency: str) -> list[int]:
    if frequency == "weekly":
        return _friday_0800_grid(start_ms, end_ms)
    if frequency == "monthly":
        return _last_friday_grid(start_ms, end_ms, quarterly=False)
    if frequency == "quarterly":
        return _last_friday_grid(start_ms, end_ms, quarterly=True)
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
    """Yield (ts_ms, snapshot_df) chronologically from a single parquet or a
    directory of chunk_*.parquet. Snapshot key is `creation_timestamp_x` (see
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
        legs.append({"instrument_id": row["instrument_id"], "strike": float(row["strike"]),
                     "option_type": otype, "qty": float(qty),
                     "expiration_timestamp_ms": float(chosen),
                     "entry_mid": float(mid), "entry_iv": float(iv), "F_expiry": F})
    return legs


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
            "last_hedge_F": None, "events": [], "rolls": [], "nav_rows": []}


def _settle(state: dict, S_T: float, ts_ms: int, initial_coin: float) -> None:
    """Realise a held structure at expiry (Eq 39 with V(T,T)=payoff)."""
    opt = 0.0
    for leg in state["open_legs"]:
        payoff = _coin_payoff(leg["option_type"], leg["strike"], S_T)
        opt += leg["contracts"] * (payoff - leg["entry_mid"])
    state["cum_pnl"] += opt
    state["events"].append({"action": "settle", "timestamp": ts_ms, "option_pnl": opt,
                            "option_cost": 0.0, "hedge_pnl": 0.0, "funding": 0.0, "rebalance_cost": 0.0})
    state["rolls"].append({"action": "settle", "timestamp": ts_ms, "S_T": S_T,
                           "option_pnl": opt, "coin_nav_after": initial_coin + state["cum_pnl"]})


def _open(state: dict, legs: list[dict], ts_ms: int, cfg) -> None:
    """Open a structure sized to current NAV; pay entry costs (Eq 40 option cost)."""
    N = (cfg.initial_coin + state["cum_pnl"]) * cfg.size_multiple
    cost = 0.0
    for leg in legs:
        leg["contracts"] = leg["qty"] * N
        cost += cfg.opt_cost_frac * abs(leg["contracts"]) * leg["entry_mid"]
    state["cum_pnl"] -= cost
    state["open_legs"] = legs
    state["events"].append({"action": "open", "timestamp": ts_ms, "option_pnl": 0.0,
                            "option_cost": cost, "hedge_pnl": 0.0, "funding": 0.0, "rebalance_cost": 0.0})
    state["rolls"].append({"action": "open", "timestamp": ts_ms, "N": N, "entry_cost": cost,
                           "legs": [(l["instrument_id"], l["strike"], l["option_type"],
                                     l["qty"], l["entry_mid"]) for l in legs]})


def _do_rolls(state: dict, snap: pd.DataFrame, ts_ms: int,
              perp_ts: np.ndarray, perp_close: np.ndarray, cfg) -> None:
    grid = state["grid"]
    while state["next_idx"] < len(grid) and ts_ms >= grid[state["next_idx"]]:
        roll_ts = grid[state["next_idx"]]
        if state["open_legs"] is not None:
            S_T = _spot_at_expiry(snap) or _perp_at(perp_ts, perp_close, roll_ts)
            _settle(state, S_T, roll_ts, cfg.initial_coin)
            state["open_legs"] = None
            state["hedge_notional"] = 0.0
        if state["next_idx"] + 1 < len(grid):
            legs = _select_structure(snap, grid[state["next_idx"] + 1], state["legs_spec"])
            if legs is not None:
                _open(state, legs, roll_ts, cfg)
        state["next_idx"] += 1
        state["last_hedge_F"] = None  # reseed hedge on the new book


def _do_hedge(state: dict, snap_idx: pd.DataFrame, ts_ms: int, F_perp: float, cfg) -> None:
    hedge_pnl = 0.0
    if state["last_hedge_F"] is not None and state["hedge_notional"] != 0.0:
        hedge_pnl = (F_perp - state["last_hedge_F"]) / F_perp * state["hedge_notional"]  # Eq 26
        state["cum_pnl"] += hedge_pnl

    target = 0.0
    if state["open_legs"] is not None:
        for leg in state["open_legs"]:
            iv, F_exp, T = leg["entry_iv"], leg["F_expiry"], None
            if leg["instrument_id"] in snap_idx.index:
                r = snap_idx.loc[leg["instrument_id"]]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                m = _mid_iv(r)
                if np.isfinite(m):
                    iv = m
                F_exp, T = float(r["underlying_price"]), float(r["t"])
            if T is None:  # instrument not quoted this snapshot: decay T off expiry
                T = max((leg["expiration_timestamp_ms"] - ts_ms) / (MS_PER_DAY * 365.0), 1e-6)
            target += -(leg["contracts"] * _leg_net_delta(leg["strike"], leg["option_type"], F_exp, iv, T))

    rebalance = cfg.perp_cost_frac * abs(target - state["hedge_notional"])
    funding = 0.0
    if cfg.funding_rate and target != 0.0:
        funding = -(cfg.funding_rate / HOURS_PER_YEAR) * target
    state["cum_pnl"] += funding - rebalance
    if hedge_pnl or rebalance or funding:
        state["events"].append({"action": "hedge", "timestamp": ts_ms, "option_pnl": 0.0,
                                "option_cost": 0.0, "hedge_pnl": hedge_pnl, "funding": funding,
                                "rebalance_cost": rebalance})
    state["hedge_notional"] = target
    state["last_hedge_F"] = F_perp
    state["nav_rows"].append({"timestamp": ts_ms, "coin_nav": cfg.initial_coin + state["cum_pnl"],
                              "cum_pnl": state["cum_pnl"], "coin_px": F_perp})


def _finalize(state: dict, cfg) -> dict:
    return {"events": state["events"], "nav": pd.DataFrame(state["nav_rows"]),
            "rolls": state["rolls"],
            "params": {"structure": state["name"], "frequency": state["frequency"],
                       "initial_coin": cfg.initial_coin, "size_multiple": cfg.size_multiple,
                       "funding_annual_rate": cfg.funding_rate,
                       "option_cost_bps": OPTION_COST_BPS, "perp_cost_bps": PERP_COST_BPS}}


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def run_all_strategies(option_path: str, perp_path: str,
                       catalog: dict[str, list[tuple]] | None = None,
                       frequencies: tuple[str, ...] = ("weekly",),
                       initial_coin: float = 1.0, size_multiple: float = 1.0,
                       funding_annual_rate: float = 0.0,
                       max_snaps: int | None = None, verbose: bool = True) -> dict[str, dict]:
    """Backtest an entire catalog x frequencies in ONE pass over the option
    data. Returns {strategy_name: result_dict}; hand the whole dict to
    `roll_results.results_table` for the paper-style table. When more than one
    frequency is requested, each name is suffixed " [freq]"."""
    catalog = catalog or build_catalog()
    cfg = types.SimpleNamespace(initial_coin=initial_coin, size_multiple=size_multiple,
                                opt_cost_frac=OPTION_COST_BPS / 10_000.0,
                                perp_cost_frac=PERP_COST_BPS / 10_000.0,
                                funding_rate=funding_annual_rate)

    perp_ts, perp_close = _perp_series(perp_path)
    first_ts = next(_iter_snapshots(option_path, max_snaps=1))[0]
    end_ts = int(perp_ts[-1])

    multi_freq = len(frequencies) > 1
    states: list[dict] = []
    for freq in frequencies:
        grid = roll_grid(first_ts, end_ts, freq)
        for name, legs in catalog.items():
            label = f"{name} [{freq}]" if multi_freq else name
            states.append(_init_state(label, legs, freq, grid))

    last_hour = None
    n = 0
    for ts_ms, snap in _iter_snapshots(option_path, max_snaps=max_snaps):
        n += 1
        for st in states:
            _do_rolls(st, snap, ts_ms, perp_ts, perp_close, cfg)
        hour = ts_ms // MS_PER_HOUR
        if last_hour is None or hour != last_hour:
            F_perp = _perp_at(perp_ts, perp_close, ts_ms)
            snap_idx = snap.set_index("instrument_id")
            for st in states:
                _do_hedge(st, snap_idx, ts_ms, F_perp, cfg)
            last_hour = hour
        if verbose and n % 20_000 == 0:
            d = dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc)
            print(f"[{n}] {d:%Y-%m-%d %H:%M} — {len(states)} strategies running")

    for st in states:
        if st["open_legs"] is not None:
            _settle(st, _perp_at(perp_ts, perp_close, ts_ms), ts_ms, cfg.initial_coin)
    return {st["name"]: _finalize(st, cfg) for st in states}


def run_roll_backtest(option_path: str, perp_path: str,
                      structure: str = "Short Straddle", frequency: str = "weekly",
                      initial_coin: float = 1.0, size_multiple: float = 1.0,
                      funding_annual_rate: float = 0.0,
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
                             max_snaps=max_snaps, verbose=verbose)
    return res[name]
