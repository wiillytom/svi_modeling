"""
Backtest engine for the red-cell relative-value strategy on ETH options.

Strategy, in one paragraph: every snapshot, calibrate a global eSSVI surface,
flag "red cells" (`signal.detect_red_cells` — the same theo-vs-bid/ask-price
test the live trading screen uses), enter a vega-sized position on each new
one, delta-hedge the whole book with ETH-PERPETUAL (using `inverse_delta`,
the Lucic & Sepp 2024 premium-adjusted hedge ratio — NOT the plain
Black-Scholes delta, which is the wrong hedge ratio for a coin-settled
option), and exit a position when its red-cell signal nulls out, when a
better opportunity appears at the same strike+expiry, or — failing both — at
expiry, settled at intrinsic value.

See /Users/macbookair/.claude/plans/sharded-crunching-shannon.md for the full
design rationale and the open design choices this implements.
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np
import pandas as pd
import polars as pl

from volatility_surface.core.calibration.calibrator import (
    calibrate_global_essvi, calibrate_global_essvi_update,
    calibrate_snapshot, calibrate_snapshot_sabr_update,
)
from volatility_surface.models.vol_models import eSSVI, RawSVI
from volatility_surface.backtest.signal import detect_red_cells, implied_carry_rate, detect_calendar_arb
from volatility_surface.backtest import live_plot


FULL_CALIB_INTERVAL_MS = 15 * 60 * 1000   # full eSSVI recalibration cadence
OPTION_COST_BPS = 50.0                     # Lucic & Sepp Assumption 5.1
PERP_COST_BPS = 5.0                        # Lucic & Sepp Assumption 5.1
TARGET_VEGA_DOLLARS = 1000.0
HOURS_PER_YEAR = 365.0 * 24.0


def _make_calibrator(model_name: str):
    """Returns (model_instance, full_fn, update_fn) with a uniform call
    signature: full_fn(snap, bid_col=, ask_col=, verbose=) -> result dict;
    update_fn(snap, prev_result, bid_col=, ask_col=, verbose=) -> result dict.
    Lets `run_backtest` swap models without the main loop caring which one —
    e.g. to check whether eSSVI's structural fitting limitation (documented
    elsewhere in this project: ~18-28% "red cell" floor purely from its
    3-params/slice functional form, independent of real mispricing) is
    inflating how many red cells looked tradeable, by comparing against
    RawSVI's richer 5-params/slice per-expiry fit.
    """
    if model_name == "essvi":
        model = eSSVI()

        def full_fn(snap, **kw):
            return calibrate_global_essvi(snap, model, speed_mode="fast", **kw)

        def update_fn(snap, prev_result, **kw):
            return calibrate_global_essvi_update(snap, prev_result, mode="theta_only", **kw)

        return model, full_fn, update_fn

    if model_name in ("svi", "rawsvi"):
        model = RawSVI()

        def full_fn(snap, **kw):
            return calibrate_snapshot(snap, model, **kw)

        def update_fn(snap, prev_result, **kw):
            return calibrate_snapshot_sabr_update(snap, prev_result, model=model, **kw)

        return model, full_fn, update_fn

    raise ValueError(f"unknown model_name {model_name!r} — use 'essvi' or 'svi'")


def _load_perp_series(perp_path: str) -> pd.DataFrame:
    df = pl.read_parquet(perp_path).select(["timestamp_ms", "close"]).sort("timestamp_ms").to_pandas()
    return df


def _perp_price_at(perp_ts: np.ndarray, perp_close: np.ndarray, ts_ms: int) -> float:
    """Nearest perp price AT OR BEFORE ts_ms — never look ahead."""
    idx = np.searchsorted(perp_ts, ts_ms, side="right") - 1
    idx = max(idx, 0)
    return float(perp_close[idx])


def _progress_bar(current: int, total: int, width: int = 30) -> str:
    frac = current / total if total else 1.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({frac * 100:5.1f}%)"


def _iter_snapshots(chunk_dir: str, max_chunks: int | None = None, progress: bool = False):
    """Yield (creation_timestamp_x, snapshot_df) across every chunk parquet in
    `chunk_dir`, in chronological order. Chunk files are named
    chunk_00000.parquet, chunk_00001.parquet, ... in day order
    (`clean_bulk_parquet_chunked`), so sorting filenames preserves global
    chronology; within a chunk, groupby+sort_values orders snapshots too.

    Groups by `creation_timestamp_x` (the true snapshot key established by
    `_clean_df_pl_core`'s own snapshot-detection, ms precision), NOT by
    `file_timestamp` — that column is a minute-TRUNCATED display string, and
    on any source polled faster than once/minute (confirmed on the live
    gatherer's output: 24 genuinely distinct snapshots ~1.5-3s apart all
    truncating to the same "HH:MM" string) grouping by it silently merges
    many real snapshots into one Frankenstein "snapshot" — wrong (conflates
    different points in time into a single calibration) and the actual
    reason a single calibration call was taking ~100s instead of a fraction
    of a second (264 fake expiry-slices from 24 merged real ones, instead of
    the true ~20-30).

    `max_chunks` limits how many chunk FILES (not snapshots) are read — pass
    1 to try the engine on a single day before committing to a full run.

    `progress`: if True, prints a live, in-place progress bar over snapshots
    within the current day (throttled to ~5 updates/sec so it doesn't add
    meaningful overhead of its own) — separate from `run_backtest`'s own
    periodic "[N] ... open positions" line, which reports on the run overall
    rather than position within the current day."""
    paths = sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.parquet")))
    if max_chunks is not None:
        paths = paths[:max_chunks]
    n_days = len(paths)
    for day_idx, path in enumerate(paths):
        day_df = pl.read_parquet(path).to_pandas()
        if day_df.empty:
            continue
        n_snaps = day_df["creation_timestamp_x"].nunique()
        last_print = 0.0
        for i, (ts, snap) in enumerate(day_df.groupby("creation_timestamp_x", sort=True)):
            # Defensive: some sources (observed on the live gatherer's output)
            # duplicate every row within a snapshot verbatim, which breaks the
            # instrument_id-indexed lookups below (`.loc[inst_id]` returning a
            # DataFrame instead of a Series). One row per instrument per
            # snapshot is always correct regardless of the source.
            snap = snap.drop_duplicates(subset="instrument_id", keep="first")
            if progress:
                now = time.time()
                if now - last_print > 0.2 or i == n_snaps - 1:
                    bar = _progress_bar(i + 1, n_snaps)
                    print(f"\rDay {day_idx + 1}/{n_days} {os.path.basename(path)} {bar}", end="", flush=True)
                    last_print = now
            yield ts, snap.reset_index(drop=True)
        if progress:
            print()  # newline so the next day's bar (or later prints) start fresh


def _slot_key(row) -> tuple:
    """A (strike, expiry) slot — a call and a put at the same strike/expiry
    compete for one capital slot (rotation scope decided in the plan). Uses
    `expiration_timestamp_ms` (constant for a listed instrument) rather than
    `t` (which decays every snapshot as time passes)."""
    return (float(row["strike"]), float(row["expiration_timestamp_ms"]))


def _option_notional_cost(mid_price: float, contracts: float) -> float:
    return OPTION_COST_BPS / 10_000.0 * mid_price * contracts


def _nearest_leg(snap: pd.DataFrame, t_target: float, k_target: float) -> pd.Series | None:
    """Nearest listed instrument to (t_target, k_target) for a calendar-spread
    leg. Matches this project's OTM convention (call for k>=0, put for k<=0)
    since forwards differ across expiries — the same k_target maps to a
    different strike at each expiry (K = F * exp(k)), so this can't just look
    up a strike directly, it has to search each expiry's own listed strikes
    by k. Returns None if that expiry has no quotes in this snapshot."""
    sub = snap[np.isclose(snap["t"], t_target)]
    if sub.empty:
        return None
    option_type = "C" if k_target >= 0 else "P"
    sub = sub[sub["option_type"] == option_type]
    if sub.empty:
        return None
    idx = (sub["k"] - k_target).abs().idxmin()
    return sub.loc[idx]


def run_backtest(chunk_dir: str, perp_path: str,
                  model_name: str = "essvi",
                  target_vega_dollars: float = TARGET_VEGA_DOLLARS,
                  full_calib_interval_ms: int = FULL_CALIB_INTERVAL_MS,
                  max_chunks: int | None = None,
                  verbose: bool = True,
                  invert_signal: bool = False,
                  min_edge_multiple: float = 2.0,
                  enable_calendar_arb: bool = False,
                  calendar_arb_tol: float = 1e-6,
                  live_plot_pnl: bool = False,
                  live_plot_interval_secs: float = 3.0) -> list[dict]:
    """Run the backtest over every snapshot in `chunk_dir`'s cleaned chunk
    parquets (must have been produced with `otm_only=False` — the strategy
    needs the full ITM+OTM universe, not just the calibration-ready OTM one).

    `model_name`: "essvi" (global eSSVI, `calibrate_global_essvi[_update]`) or
    "svi" (per-slice RawSVI, `calibrate_snapshot`/`calibrate_snapshot_sabr_update`
    for the warm-started path). RawSVI fits each expiry independently with 5
    parameters instead of eSSVI's global 2-parameter power-law curvature —
    tighter per-slice fit, no arb-free guarantee across slices. Useful to
    check whether eSSVI's documented structural fitting limitation is
    inflating the red-cell count with model-form artifacts rather than real
    mispricing.

    `max_chunks`: limit to the first N chunk files (= N days, one chunk per
    day from `clean_bulk_parquet_chunked`) instead of the whole directory —
    pass 1 to time a single day before committing to the full range.

    `invert_signal`: flag exactly the same red cells but take the OPPOSITE
    side (sell what looked cheap, buy what looked rich) — a diagnostic for
    whether the strategy's direction, rather than its transaction costs, is
    what's driving P&L. Does NOT flip transaction costs (spread-crossing and
    fees are paid regardless of side) or the delta hedge (still hedges
    whatever position is actually held), only the entry/exit direction.

    `min_edge_multiple`: only enter (or rotate into) a red cell whose
    mispricing exceeds this multiple of its own half-spread. Default 2.0 =
    require the edge to clear the round-trip cost of crossing the spread
    twice (once to enter, once to exit) — a real snapshot check found most
    raw red cells (any crossing at 4dp, no minimum size, matching the live
    screen's definition) don't clear even this bar, meaning they lose money
    by construction before the explicit transaction cost is even applied.
    Set to 0 to disable filtering and trade every red cell (the original
    behaviour, matching the live screen exactly). Does not affect null-out
    exits, which still use the unfiltered red-cell flag.

    `enable_calendar_arb`: also trade calendar-spread arbitrage between
    adjacent expiry slices (`signal.detect_calendar_arb`) — a genuinely
    different, model-INTERNAL signal from the red cells (which compare the
    model to the market's own quotes): this checks whether the calibrated
    surface is even theoretically arbitrage-free across time, independent of
    any bid/ask. Only meaningful for `model_name="svi"` (RawSVI has no
    cross-slice constraint so it CAN violate this; eSSVI/SSVI/SABR enforce it
    by construction and will essentially never trigger it). On real data
    checked this session (30 snapshots, 300 adjacent pairs), this fired
    exactly 0 times — RawSVI's global-init-assisted per-slice fits stayed
    correctly ordered throughout, so don't expect much action from this on
    similar data; it's here in case your full multi-month range behaves
    differently (e.g. during sharp moves). When triggered: sells the
    near-expiry leg, buys the far-expiry leg, both at the strike nearest the
    violation's log-moneyness (`k_star`, converted via each expiry's own
    forward — they differ across expiries) — a static, model-free arbitrage
    if the violation is real, not a view on realized vol. Exits both legs
    together when the violation resolves (recomputed each snapshot) or the
    near leg's own time-to-expiry decays to zero.

    `live_plot_pnl`: redraw a cumulative-P&L chart in the notebook cell every
    `live_plot_interval_secs` (see `live_plot.LivePnLPlot`) — a static
    matplotlib figure redrawn via `IPython.display.clear_output`, not an
    interactive widget. Requires an IPython/Jupyter environment; silently
    disabled (with one warning) outside of one. When on, this also takes over
    the text progress bar's job (`_iter_snapshots`'s bar would otherwise be
    wiped by `clear_output` on every redraw), so only one of the two shows.

    Returns a list of event dicts (one per entry/exit) — hand this to
    `results.py` for aggregation. Each event has: instrument_id, strike,
    option_type, side, action ('entry'/'exit'), reason (for exits: 'null',
    'rotation', 'expiry'), timestamp, price, contracts, option_pnl (for
    exits), option_cost, and running hedge state at that point.
    """
    if live_plot_pnl and not live_plot.is_available():
        print("live_plot_pnl=True but IPython/matplotlib aren't available — ignoring (need a Jupyter environment).")
        live_plot_pnl = False
    plotter = live_plot.LivePnLPlot(interval_secs=live_plot_interval_secs) if live_plot_pnl else None

    perp_df = _load_perp_series(perp_path)
    perp_ts = perp_df["timestamp_ms"].to_numpy()
    perp_close = perp_df["close"].to_numpy()

    model, calibrate_full, calibrate_update = _make_calibrator(model_name)
    last_calib_result = None
    last_full_calib_ts = None

    positions: dict[str, dict] = {}       # instrument_id -> position dict
    slot_owner: dict[tuple, str] = {}     # slot_key -> instrument_id currently holding it
    calendar_positions: dict[tuple, dict] = {}  # (t_near, t_far) -> {"near": pos, "far": pos}

    hedge_notional = 0.0                  # coin-denominated aggregate hedge (perp units), signed
    last_hedge_F = None
    last_hedge_hour = None

    events: list[dict] = []
    n_snapshots = 0

    for ts, snap in _iter_snapshots(chunk_dir, max_chunks=max_chunks, progress=verbose and not live_plot_pnl):
        n_snapshots += 1
        ts_ms = int(snap["creation_timestamp_x"].iloc[0])

        # ---- 1. calibration (full every full_calib_interval_ms, warm-start otherwise) ----
        # For eSSVI, speed_mode="fast" (L-BFGS-B) instead of "auto" (picks
        # "thorough" NM+NM for n<=25 slices, which this data has): ~3.75x
        # faster per call on real data (3.69s vs 13.86s), confirmed the
        # dominant cost in a full-day profile (69 full recalibrations ~= 11
        # of 35 min). A prior session found "fast" ~96% equivalent to
        # "thorough" with one documented rare failure on a specific snapshot
        # — acceptable here (backtest signal noise on a rare snapshot, not
        # live risk). RawSVI's calibrate_snapshot has no such knob.
        if last_calib_result is None or (ts_ms - last_full_calib_ts) >= full_calib_interval_ms:
            result = calibrate_full(snap, bid_col="bid_iv", ask_col="ask_iv", verbose=False)
            last_full_calib_ts = ts_ms
        else:
            result = calibrate_update(snap, last_calib_result, bid_col="bid_iv", ask_col="ask_iv", verbose=False)
        last_calib_result = result

        # ---- 2. signal ----
        flagged = detect_red_cells(snap, result, invert=invert_signal)
        flagged = flagged.set_index("instrument_id", drop=False)
        red = flagged[flagged["red_cell"]]

        # Precompute the single best (max edge*vega) candidate per slot ONCE
        # per snapshot — used by both the exits/rotation pass and the entries
        # pass below via an O(1) dict lookup. The previous version re-filtered
        # the whole `red` table with a pandas boolean mask for EVERY open
        # position, every snapshot (O(positions x red_cells) pandas ops) —
        # the dominant cost of a 21-minutes-for-one-day run. This also
        # unifies what used to be two slightly different edge formulas (one
        # in the exits pass, one in entries) into one.
        #
        # `slot_best` is computed from `tradeable`, NOT the raw `red` table:
        # a real snapshot check found the majority of raw red cells have a
        # mispricing SMALLER than the round-trip cost of crossing the spread
        # twice (once to enter, once to exit) — trading those loses money by
        # construction even with a perfect, instantly-converging signal,
        # before the explicit transaction cost is even applied. `min_edge_multiple`
        # filters those out of consideration for BOTH fresh entries and
        # rotation, so we never rotate out of a held position for an
        # "improvement" that isn't itself worth entering (which would leave
        # the slot empty — strictly worse than holding the original).
        # Null-out exits are unaffected — those still use the unfiltered
        # `flagged`/`red_cell` below, matching the live screen's definition
        # exactly (has the mispricing that justified this trade disappeared,
        # not "is what's left still big enough to newly enter").
        slot_best: dict[tuple, pd.Series] = {}
        if not red.empty:
            red = red.copy()
            red["_slot"] = list(zip(red["strike"].astype(float), red["expiration_timestamp_ms"].astype(float)))
            touch = np.where(red["signal_side"] == "buy", red["ask_price"], red["bid_price"])
            red["_raw_edge"] = (red["theo_price"] - touch).abs()
            red["_half_spread"] = (red["ask_price"] - red["bid_price"]) / 2.0
            red["_edge"] = red["_raw_edge"] * red["vega"]
            tradeable = red[red["_raw_edge"] > min_edge_multiple * red["_half_spread"]]
            if not tradeable.empty:
                best_idx = tradeable.groupby("_slot")["_edge"].idxmax()
                slot_best = {slot: tradeable.loc[idx] for slot, idx in best_idx.items()}

        F_perp = _perp_price_at(perp_ts, perp_close, ts_ms)

        # ---- 3. exits: null-out, then rotation ----
        for inst_id in list(positions.keys()):
            pos = positions[inst_id]
            if inst_id not in flagged.index:
                continue  # instrument not quoted this snapshot — leave position open, mark next time it reappears
            row = flagged.loc[inst_id]
            still_red = bool(row["red_cell"])

            if not still_red:
                _close_position(pos, row, "null", events)
                del positions[inst_id]
                del slot_owner[pos["slot"]]
                continue

            # rotation: slot_best[slot] is the argmax over ALL red rows at
            # this slot, including this position's own — if it's someone
            # else, that candidate has strictly higher edge*vega.
            best_row = slot_best.get(pos["slot"])
            if best_row is not None and best_row["instrument_id"] != inst_id:
                _close_position(pos, row, "rotation", events)
                del positions[inst_id]
                del slot_owner[pos["slot"]]
                # the winning candidate is `best_row`, picked up by the entries pass below

        # ---- 4. entries: for every slot not currently owned, take its
        # precomputed best candidate. Matters even on a slot freed by
        # rotation THIS same snapshot: the instrument just rotated out of is
        # often still individually red (that's not why it was rotated out —
        # a better candidate was), so grabbing an arbitrary red row for a
        # freed slot can immediately re-enter the very position rotation just
        # closed — `slot_best` is guaranteed to be the actual winner. Also
        # guards against entering both a call and a put at the same
        # strike+expiry if both happen to be red at once (rotation scope is
        # one position per slot regardless of type). ----
        for slot, row in slot_best.items():
            if slot in slot_owner:
                continue
            inst_id = row["instrument_id"]
            vega = float(row["vega"])
            if not np.isfinite(vega) or vega <= 0:
                continue
            # `vega` is coin-denominated (ETH per vol-point, matching bid/ask
            # price convention) — target_vega_dollars is USD, so convert via
            # underlying_price before dividing, or contracts comes out ~1000x
            # too large (confirmed: off by exactly the ETH price on a real row).
            vega_usd = vega * float(row["underlying_price"])
            contracts = target_vega_dollars / vega_usd
            side = row["signal_side"]
            entry_price = float(row["ask_price"] if side == "buy" else row["bid_price"])
            mid_price = float((row["bid_price"] + row["ask_price"]) / 2.0)
            cost = _option_notional_cost(mid_price, contracts)

            pos = {
                "instrument_id": inst_id, "slot": slot,
                "strike": float(row["strike"]), "option_type": row["option_type"],
                "expiration_timestamp_ms": float(row["expiration_timestamp_ms"]),
                "side": side, "entry_time": ts_ms, "entry_price": entry_price,
                "contracts": contracts, "vega_at_entry": vega,
                "inverse_delta_at_entry": float(row["inverse_delta"]),
            }
            positions[inst_id] = pos
            slot_owner[slot] = inst_id
            events.append({
                "instrument_id": inst_id, "action": "entry", "reason": None,
                "timestamp": ts_ms, "side": side, "price": entry_price,
                "contracts": contracts, "option_cost": cost, "option_pnl": None,
            })

        # ---- 4.5 calendar-spread arbitrage (only fires in practice for
        # model_name="svi" — eSSVI/SSVI/SABR enforce this by construction) ----
        if enable_calendar_arb:
            violations = detect_calendar_arb(result, tol=calendar_arb_tol)
            still_valid_keys = set()
            for v in violations:
                near_row = _nearest_leg(snap, v["t_near"], v["k_star"])
                far_row = _nearest_leg(snap, v["t_far"], v["k_star"])
                if near_row is None or far_row is None:
                    continue
                key = (float(near_row["expiration_timestamp_ms"]), float(far_row["expiration_timestamp_ms"]))
                still_valid_keys.add(key)
                if key in calendar_positions:
                    continue  # already open, leave it running

                # Sell near (calendar-arb says it shows MORE total variance
                # than the far leg, which is backwards), buy far (cheap in
                # variance terms) — validate both legs BEFORE opening either,
                # so a bad price/vega on one side never leaves an orphaned,
                # unpaired entry event for the other.
                leg_specs = []
                ok = True
                for row2, side in [(near_row, "sell"), (far_row, "buy")]:
                    vega = float(row2["vega"])
                    entry_price = row2["ask_price"] if side == "buy" else row2["bid_price"]
                    if not (np.isfinite(vega) and vega > 0 and np.isfinite(entry_price) and entry_price > 0):
                        ok = False
                        break
                    vega_usd = vega * float(row2["underlying_price"])
                    contracts = target_vega_dollars / vega_usd
                    mid_price = float((row2["bid_price"] + row2["ask_price"]) / 2.0)
                    leg_specs.append({
                        "instrument_id": row2["instrument_id"], "side": side,
                        "entry_time": ts_ms, "entry_price": float(entry_price), "contracts": contracts,
                        "inverse_delta_at_entry": float(row2["inverse_delta"]),
                        "cost": _option_notional_cost(mid_price, contracts),
                    })
                if not ok:
                    continue
                legs = {"near": leg_specs[0], "far": leg_specs[1]}
                calendar_positions[key] = legs
                for leg in legs.values():
                    events.append({
                        "instrument_id": leg["instrument_id"], "action": "entry", "reason": "calendar_arb",
                        "timestamp": ts_ms, "side": leg["side"], "price": leg["entry_price"],
                        "contracts": leg["contracts"], "option_cost": leg["cost"], "option_pnl": None,
                    })

            # close any open calendar spread whose violation has resolved
            for key in list(calendar_positions.keys()):
                if key in still_valid_keys:
                    continue
                legs = calendar_positions.pop(key)
                for leg in legs.values():
                    inst_id = leg["instrument_id"]
                    if inst_id in flagged.index:
                        row2 = flagged.loc[inst_id]
                        exit_price = float(row2["bid_price"] if leg["side"] == "buy" else row2["ask_price"])
                        mid_price = float((row2["bid_price"] + row2["ask_price"]) / 2.0)
                    else:
                        exit_price = leg["entry_price"]  # not quoted this snapshot — stale fallback
                        mid_price = leg["entry_price"]
                    sign = 1.0 if leg["side"] == "buy" else -1.0
                    pnl = sign * (exit_price - leg["entry_price"]) * leg["contracts"]
                    cost = _option_notional_cost(mid_price, leg["contracts"])
                    events.append({
                        "instrument_id": inst_id, "action": "exit", "reason": "calendar_resolved",
                        "timestamp": ts_ms, "side": leg["side"], "price": exit_price,
                        "contracts": leg["contracts"], "option_cost": cost, "option_pnl": pnl,
                    })

        # ---- 5. hourly delta-hedge rebalance + funding ----
        hour = ts_ms // 3_600_000
        if last_hedge_hour is None or hour != last_hedge_hour:
            if last_hedge_F is not None and hedge_notional != 0.0:
                hedge_pnl = (F_perp - last_hedge_F) / F_perp * hedge_notional
                events.append({"instrument_id": None, "action": "hedge_pnl", "reason": None,
                                "timestamp": ts_ms, "side": None, "price": F_perp,
                                "contracts": hedge_notional, "option_cost": 0.0, "option_pnl": hedge_pnl})

            calendar_legs = [leg for legs in calendar_positions.values() for leg in legs.values()]
            target_notional = -sum(
                p["inverse_delta_at_entry"] * p["contracts"] * (1.0 if p["side"] == "buy" else -1.0)
                for p in list(positions.values()) + calendar_legs
            )
            rebalance_cost = PERP_COST_BPS / 10_000.0 * abs(target_notional - hedge_notional)

            # implied funding rate: median across currently quoted genuine dated-future slices this snapshot
            dated = snap[snap["underlying_index"] != "index_price"] if "underlying_index" in snap.columns else snap
            if not dated.empty:
                r_implied = float(np.median([
                    implied_carry_rate(row2["underlying_price"], row2["estimated_delivery_price"], row2["t"])
                    for _, row2 in dated.drop_duplicates("expiration_timestamp_ms").iterrows()
                ]))
            else:
                r_implied = 0.0
            funding_cost = -(r_implied / HOURS_PER_YEAR) * target_notional

            events.append({"instrument_id": None, "action": "hedge_rebalance", "reason": None,
                            "timestamp": ts_ms, "side": None, "price": F_perp,
                            "contracts": target_notional - hedge_notional,
                            "option_cost": rebalance_cost, "option_pnl": funding_cost})

            hedge_notional = target_notional
            last_hedge_F = F_perp
            last_hedge_hour = hour

        if plotter is not None:
            status = f"[{n_snapshots} snapshots] {len(positions)} open positions, {len(events)} events so far"
            plotter.maybe_render(events, status=status)
        elif verbose and n_snapshots % 500 == 0:
            label = snap["file_timestamp"].iloc[0] if "file_timestamp" in snap.columns else ts
            print(f"[{n_snapshots}] {label} — {len(positions)} open positions, {len(events)} events so far")

    # ---- 6. fallback: settle any still-open positions at the last known price (intrinsic if truly past expiry) ----
    for inst_id, pos in positions.items():
        events.append({"instrument_id": inst_id, "action": "exit", "reason": "end_of_data",
                        "timestamp": ts_ms, "side": pos["side"], "price": None,
                        "contracts": pos["contracts"], "option_cost": 0.0, "option_pnl": None})
    for legs in calendar_positions.values():
        for leg in legs.values():
            events.append({"instrument_id": leg["instrument_id"], "action": "exit", "reason": "end_of_data",
                            "timestamp": ts_ms, "side": leg["side"], "price": None,
                            "contracts": leg["contracts"], "option_cost": 0.0, "option_pnl": None})

    if plotter is not None:
        plotter.maybe_render(events, status=f"Done — {len(events)} events total.", force=True)

    return events


def _close_position(pos: dict, row: pd.Series, reason: str, events: list[dict]) -> None:
    exit_price = float(row["bid_price"] if pos["side"] == "buy" else row["ask_price"])
    sign = 1.0 if pos["side"] == "buy" else -1.0
    pnl = sign * (exit_price - pos["entry_price"]) * pos["contracts"]
    mid_price = float((row["bid_price"] + row["ask_price"]) / 2.0)
    cost = _option_notional_cost(mid_price, pos["contracts"])
    events.append({
        "instrument_id": pos["instrument_id"], "action": "exit", "reason": reason,
        "timestamp": int(row["creation_timestamp_x"]), "side": pos["side"], "price": exit_price,
        "contracts": pos["contracts"], "option_cost": cost, "option_pnl": pnl,
    })
