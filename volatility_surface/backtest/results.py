"""
Aggregation for `engine.run_backtest`'s event log.
"""

from __future__ import annotations

import pandas as pd


def _pair_entries_exits(df: pd.DataFrame) -> pd.DataFrame:
    """Match each entry to its following exit, per instrument_id. At most one
    open position per instrument_id at a time (enforced by the engine's
    slot-ownership bookkeeping), so entries/exits for a given instrument
    strictly alternate — a simple sequential pairing is correct."""
    records = []
    opt_events = df[df["instrument_id"].notna()].copy()
    # A position entered on the very last snapshot gets force-closed by the
    # engine's end-of-data fallback using that SAME snapshot's timestamp, so
    # entry and exit can tie on `timestamp` — sort_values isn't guaranteed
    # stable on ties, so break ties explicitly (entry before exit) rather than
    # rely on sort order alone.
    opt_events["_action_order"] = (opt_events["action"] == "exit").astype(int)
    opt_events = opt_events.sort_values(["timestamp", "_action_order"])
    for inst_id, grp in opt_events.groupby("instrument_id"):
        pending_entry = None
        for _, r in grp.iterrows():
            if r["action"] == "entry":
                pending_entry = r
            elif r["action"] == "exit" and pending_entry is not None:
                records.append({
                    "instrument_id": inst_id,
                    "entry_time": pending_entry["timestamp"],
                    "exit_time": r["timestamp"],
                    "holding_minutes": (r["timestamp"] - pending_entry["timestamp"]) / 60_000.0,
                    "reason": r["reason"],
                    "pnl": r["option_pnl"],
                    "entry_cost": pending_entry["option_cost"],
                    "exit_cost": r["option_cost"],
                })
                pending_entry = None
    return pd.DataFrame(records)


def summarize(events: list[dict]) -> dict:
    """Aggregate `engine.run_backtest`'s event list into headline metrics plus
    a per-exit-reason and a per-trade breakdown, so it's clear which mechanism
    (signal converging, rotation, or running out of data) is actually
    generating the P&L rather than just a single net number."""
    df = pd.DataFrame(events)
    if df.empty:
        return {"n_trades": 0}

    exits = df[df["action"] == "exit"]
    trades = _pair_entries_exits(df)

    total_option_pnl = exits["option_pnl"].dropna().sum()
    total_option_cost = df["option_cost"].sum()
    total_hedge_pnl = df.loc[df["action"] == "hedge_pnl", "option_pnl"].sum()
    total_funding_cost = df.loc[df["action"] == "hedge_rebalance", "option_pnl"].sum()
    total_rebalance_cost = df.loc[df["action"] == "hedge_rebalance", "option_cost"].sum()
    net_pnl = (total_option_pnl - total_option_cost + total_hedge_pnl
               + total_funding_cost - total_rebalance_cost)

    closed = trades.dropna(subset=["pnl"])
    by_reason = closed.groupby("reason")["pnl"].agg(["sum", "mean", "count"]) if not closed.empty else pd.DataFrame()

    return {
        "n_trades": len(trades),
        "n_closed_with_pnl": len(closed),
        "net_pnl": net_pnl,
        "total_option_pnl": total_option_pnl,
        "total_option_cost": total_option_cost,
        "total_hedge_pnl": total_hedge_pnl,
        "total_funding_cost": total_funding_cost,
        "total_rebalance_cost": total_rebalance_cost,
        "hit_rate": float((closed["pnl"] > 0).mean()) if not closed.empty else float("nan"),
        "avg_holding_minutes": float(closed["holding_minutes"].mean()) if not closed.empty else float("nan"),
        "pnl_by_exit_reason": by_reason,
        "trades": trades,
    }


def print_summary(summary: dict) -> None:
    if summary.get("n_trades", 0) == 0:
        print("No trades.")
        return
    print(f"Trades: {summary['n_trades']} ({summary['n_closed_with_pnl']} closed with a realized P&L)")
    print(f"Hit rate: {summary['hit_rate']:.1%}")
    print(f"Avg holding time: {summary['avg_holding_minutes']:.1f} min")
    print()
    print(f"Option P&L (gross):     {summary['total_option_pnl']:+.4f}")
    print(f"Option transaction cost: -{summary['total_option_cost']:.4f}")
    print(f"Hedge P&L:              {summary['total_hedge_pnl']:+.4f}")
    print(f"Implied funding cost:   {summary['total_funding_cost']:+.4f}")
    print(f"Hedge rebalance cost:   -{summary['total_rebalance_cost']:.4f}")
    print(f"NET P&L:                {summary['net_pnl']:+.4f}")
    print()
    print("P&L by exit reason:")
    print(summary["pnl_by_exit_reason"])


def running_pnl_series(events: list[dict]) -> pd.DataFrame:
    """Cumulative net P&L over time, built directly from the event stream —
    one-shot / post-hoc version (recomputes from scratch each call). For a
    live-updating chart DURING a run, use `live_plot.LivePnLPlot` instead,
    which updates incrementally rather than reprocessing every event each
    time it's called.

    Same P&L components `summarize()` totals, just accumulated in event
    order instead of summed at the end: option P&L net of transaction cost on
    entries and exits, plus hedge P&L, plus implied funding cost net of
    rebalance cost.
    """
    df = pd.DataFrame(events)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "cum_pnl"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    pnl_delta = df["option_pnl"].fillna(0.0) - df["option_cost"].fillna(0.0)
    df = df.assign(cum_pnl=pnl_delta.cumsum())
    return df[["timestamp", "cum_pnl"]]
