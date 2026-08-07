"""
Pull perpetual-futures OHLCV history from Deribit's public API (no auth) — the
price series the roll backtester delta-hedges against.

Companion to `deribit_funding.py`: that module fetches the CARRY on a perp
position (Eq 29), this one fetches the PRICE path that position moves with
(Eq 26). The backtester needs both, and they are separate inputs
(`--perp` / `--funding`).

Output schema matches `2 - Data/parquets/eth_perp_1min_jan-jun2025.parquet`
exactly (timestamp_ms, open, high, low, close, volume, cost, datetime), so a
freshly pulled file is a drop-in replacement wherever that one is used.

`get_tradingview_chart_data` returns bar-START timestamps, so the `close` of the
bar stamped 10:00 is really the price at 10:01. `_perp_at` looks up the last bar
at or before a snapshot, which therefore peeks up to one resolution-step ahead —
1 minute at the default resolution, i.e. noise for an hourly-rebalanced
strategy, and the same convention the existing file already uses. Pull at a
coarser resolution only if that trade-off is acceptable for the use case.
"""

from __future__ import annotations

import datetime as dt
import json
import time
import urllib.parse
import urllib.request

import pandas as pd

_URL = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"

#: The endpoint returns at most ~5000 bars per call and TRUNCATES SILENTLY past
#: that — asking for 5 days of 1-minute data returned 5001 of the 7200 bars, so
#: every chunk left a 2199-minute hole and the pull looked like it had succeeded
#: (69.7% coverage, 158 evenly spaced gaps). Chunks are therefore sized from the
#: resolution, and `fetch_perp` verifies the returned count against the ceiling.
_MAX_BARS = 5000
_SAFE_BARS = 4000   # margin, so a boundary bar or DST-style oddity can't tip it over


def _get(instrument: str, start_ms: int, end_ms: int, resolution: str,
         retries: int = 3) -> dict:
    q = urllib.parse.urlencode({"instrument_name": instrument,
                                "start_timestamp": start_ms,
                                "end_timestamp": end_ms,
                                "resolution": resolution})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{_URL}?{q}", timeout=60) as r:
                return json.loads(r.read()).get("result", {})
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return {}


def fetch_perp(instrument: str, start_ms: int, end_ms: int,
               resolution: str = "1", verbose: bool = True) -> pd.DataFrame:
    """OHLCV bars for `instrument` over [start_ms, end_ms].

    `resolution` is Deribit's, in minutes as a string ("1", "5", "60") or "1D".
    Returns a frame sorted by time and deduplicated on timestamp — chunk borders
    overlap by one bar, which would otherwise leave duplicate timestamps that
    break `_perp_at`'s searchsorted assumption of a strictly increasing series.
    """
    # Size the request window so it can never exceed the endpoint's bar ceiling.
    minutes = 1440 if str(resolution).upper() == "1D" else int(resolution)
    step = _SAFE_BARS * minutes * 60_000

    frames = []
    lo = start_ms
    truncated = 0
    while lo < end_ms:
        hi = min(lo + step, end_ms)
        r = _get(instrument, lo, hi, resolution)
        ticks = r.get("ticks") or []
        if len(ticks) >= _MAX_BARS:
            truncated += 1
        if ticks:
            frames.append(pd.DataFrame({
                "timestamp_ms": ticks,
                "open": r["open"], "high": r["high"], "low": r["low"],
                "close": r["close"], "volume": r["volume"], "cost": r["cost"],
            }))
        if verbose:
            d = dt.datetime.fromtimestamp(lo / 1000, dt.timezone.utc)
            print(f"  {instrument} {d:%Y-%m-%d}: {len(ticks)} bars "
                  f"(total {sum(len(f) for f in frames):,})")
        lo = hi
        time.sleep(0.15)  # be polite

    if not frames:
        return pd.DataFrame(columns=["timestamp_ms", "open", "high", "low",
                                     "close", "volume", "cost", "datetime"])
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates("timestamp_ms")
            .sort_values("timestamp_ms")
            .reset_index(drop=True))
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    if truncated:
        print(f"  WARNING: {truncated} request(s) hit the {_MAX_BARS}-bar ceiling — "
              f"data is missing. Lower _SAFE_BARS.")
    # Coverage check: silent truncation is the failure mode this endpoint has, so
    # report it here rather than letting a holey series reach the backtest, where
    # `_perp_at` would quietly reuse a stale price across every gap.
    step_ms = minutes * 60_000
    slots = df["timestamp_ms"].to_numpy() // step_ms
    span = int(slots.max() - slots.min()) + 1
    if verbose:
        print(f"  coverage: {len(df):,} / {span:,} expected bars ({len(df) / span:.1%})")
    return df


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instrument", default="ETH-PERPETUAL")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (UTC); default = now")
    ap.add_argument("--resolution", default="1", help="minutes, or '1D'")
    ap.add_argument("--out", required=True, help="output parquet path")
    a = ap.parse_args()

    s = int(dt.datetime.strptime(a.start, "%Y-%m-%d")
              .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    e = (int(dt.datetime.strptime(a.end, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
         if a.end else int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000))

    df = fetch_perp(a.instrument, s, e, resolution=a.resolution)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_parquet(a.out)
    print(f"\nsaved {len(df):,} bars -> {a.out}")
