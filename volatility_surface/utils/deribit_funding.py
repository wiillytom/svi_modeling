"""
Pull realised perpetual-futures funding-rate history from Deribit's public API
(no auth) for use as the DeltaFunding term (Eq 29) in the roll backtester.

Deribit `public/get_funding_rate_history` returns hourly points with:
  timestamp, index_price, prev_index_price, interest_1h, interest_8h
`interest_1h` is the funding fraction accrued that hour (a long perp position
pays it when positive) — the most literal input for the paper's hourly Eq 29
sum, so that's what we keep. We also derive an annualised rate (x 24 x 365) for
reporting / for feeding the engine's annualised-rate interface.

The endpoint caps the number of points per call, so `fetch_funding` walks the
window in chunks and concatenates.
"""

from __future__ import annotations

import time
import urllib.request
import urllib.parse
import json

import numpy as np
import pandas as pd

_URL = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
_HOURS_PER_YEAR = 24 * 365
_CHUNK_MS = 20 * 24 * 3_600_000  # 20 days per request (well under the point cap)


def _get(instrument: str, start_ms: int, end_ms: int, retries: int = 3) -> list[dict]:
    q = urllib.parse.urlencode({"instrument_name": instrument,
                                "start_timestamp": start_ms, "end_timestamp": end_ms})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{_URL}?{q}", timeout=30) as r:
                return json.loads(r.read()).get("result", [])
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return []


def fetch_funding(instrument: str, start_ms: int, end_ms: int,
                  verbose: bool = True) -> pd.DataFrame:
    """Hourly funding history for `instrument` over [start_ms, end_ms].
    Columns: timestamp_ms, interest_1h, interest_8h, index_price, funding_annual.
    Deduplicated on timestamp, sorted ascending."""
    rows: list[dict] = []
    lo = start_ms
    while lo < end_ms:
        hi = min(lo + _CHUNK_MS, end_ms)
        chunk = _get(instrument, lo, hi)
        rows.extend(chunk)
        if verbose:
            print(f"  {instrument} {lo} -> {hi}: {len(chunk)} pts (total {len(rows)})")
        lo = hi
        time.sleep(0.2)  # be polite
    if not rows:
        return pd.DataFrame(columns=["timestamp_ms", "interest_1h", "interest_8h",
                                     "index_price", "funding_annual"])
    df = pd.DataFrame(rows).rename(columns={"timestamp": "timestamp_ms"})
    df = df.drop_duplicates("timestamp_ms").sort_values("timestamp_ms").reset_index(drop=True)
    df["funding_annual"] = df["interest_1h"].astype(float) * _HOURS_PER_YEAR
    return df[["timestamp_ms", "interest_1h", "interest_8h", "index_price", "funding_annual"]]


def load_funding_lookup(path: str):
    """Load a saved funding parquet into a step-function lookup:
    `f(ts_ms) -> interest_1h` (the funding fraction for the hour containing ts,
    using the last published rate at or before ts; 0.0 before the first point).
    Returned callable is what `roll_engine` consumes as its funding source."""
    df = pd.read_parquet(path).sort_values("timestamp_ms")
    ts = df["timestamp_ms"].to_numpy()
    rate = df["interest_1h"].to_numpy(dtype=float)

    def lookup(ts_ms: int) -> float:
        i = np.searchsorted(ts, ts_ms, side="right") - 1
        return float(rate[i]) if i >= 0 else 0.0

    return lookup
