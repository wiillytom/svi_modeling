"""
Sanity-check a cleaned option parquet before trusting a backtest built on it.

Written for the 2024-2026 hourly build, which is stitched from at least three
different Deribit export vintages (raw ticker / instruments-merged / later
variants). Each vintage change is a chance for something to shift silently —
a units convention, a thinner chain, a gap in coverage — and none of those
raise. This reports the things that would quietly corrupt a result rather
than crash it.

    python volatility_surface/utils/validate_dataset.py \
        --path "2 - Data/parquets/eth_hourly_2024_2026.parquet"

Anything printed with [WARN] deserves a look before running strategies; [FAIL]
means the backtest cannot use the file as-is.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl

#: Columns `roll_engine` reads. Missing any of these is fatal.
REQUIRED = ["creation_timestamp_x", "instrument_id", "strike", "option_type",
            "expiration_timestamp_ms", "underlying_price", "estimated_delivery_price",
            "bid_price", "ask_price", "bid_iv", "ask_iv", "mark_iv", "t", "k",
            "vega", "open_interest"]

MS_H = 3_600_000


def _hdr(t: str) -> None:
    print(f"\n{'=' * 68}\n{t}\n{'=' * 68}")


def validate(path: str, expect_interval_min: int = 60) -> int:
    df = pl.read_parquet(path)
    problems = 0

    _hdr("1. COVERAGE")
    ts = df["creation_timestamp_x"]
    lo, hi = int(ts.min()), int(ts.max())
    n_snap = ts.n_unique()
    d0 = dt.datetime.fromtimestamp(lo / 1000, dt.timezone.utc)
    d1 = dt.datetime.fromtimestamp(hi / 1000, dt.timezone.utc)
    print(f"  rows           : {df.height:,}")
    print(f"  snapshots      : {n_snap:,}")
    print(f"  range          : {d0:%Y-%m-%d %H:%M} -> {d1:%Y-%m-%d %H:%M} UTC "
          f"({(d1 - d0).days} days)")
    print(f"  rows/snapshot  : {df.height / max(n_snap, 1):.0f}")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"  [FAIL] missing columns required by roll_engine: {missing}")
        problems += 1
    else:
        print("  all columns required by roll_engine are present")

    _hdr("2. HOURLY GRID — a missing hour is a missed hedge; a missing 08:00 is a missed roll")
    hours = sorted({int(t) // MS_H for t in ts.unique()})
    expected = (hours[-1] - hours[0]) + 1
    gaps = np.diff(np.array(hours))
    n_missing = expected - len(hours)
    print(f"  distinct hours : {len(hours):,} of {expected:,} expected "
          f"({len(hours) / expected:.1%} coverage)")
    if n_missing:
        big = np.where(gaps > 1)[0]
        worst = sorted(((int(gaps[i] - 1), hours[i]) for i in big), reverse=True)[:5]
        print(f"  [WARN] {n_missing:,} hours absent, in {len(big):,} gaps. Largest:")
        for n, h in worst:
            print(f"           {dt.datetime.fromtimestamp(h * MS_H / 1000, dt.timezone.utc):%Y-%m-%d %H:%M}"
                  f"  +{n} h missing")
        problems += 1
    else:
        print("  no gaps")

    # Friday 08:00 UTC is where weekly rolls happen — check those specifically.
    fri = [h for h in hours
           if (d := dt.datetime.fromtimestamp(h * MS_H / 1000, dt.timezone.utc)).weekday() == 4
           and d.hour == 8]
    weeks = (d1 - d0).days / 7
    print(f"  Friday 08:00 snapshots present: {len(fri)} (~{weeks:.0f} weeks in range)")
    if len(fri) < 0.9 * weeks:
        print("  [WARN] weekly roll dates are missing — those rolls will be skipped")
        problems += 1

    _hdr("3. VALUE RANGES — a units change between vintages is silent, not an error")
    checks = [
        ("mark_iv", 0.05, 5.0, "decimal vol, e.g. 0.65 = 65%"),
        ("bid_iv", 0.05, 5.0, "decimal vol"),
        ("ask_iv", 0.05, 5.0, "decimal vol"),
        ("t", 0.0, 3.0, "years to expiry, must be > 0"),
        ("bid_price", 0.0, 1.5, "COIN-denominated premium, not USD"),
        ("ask_price", 0.0, 1.5, "COIN-denominated premium"),
        ("vega", 0.0, 1.0, "coin per vol point"),
    ]
    for col, lo_ok, hi_ok, why in checks:
        if col not in df.columns:
            continue
        s = df[col].drop_nulls()
        if s.is_empty():
            print(f"  [FAIL] {col:10s} all null")
            problems += 1
            continue
        mn, med, mx = float(s.min()), float(s.median()), float(s.max())
        bad = float(((s < lo_ok) | (s > hi_ok)).mean()) * 100
        flag = "[WARN]" if bad > 1.0 else "      "
        print(f"  {flag} {col:10s} min {mn:10.4f} | med {med:9.4f} | max {mx:11.4f} "
              f"| {bad:5.2f}% outside [{lo_ok}, {hi_ok}]  ({why})")
        if bad > 1.0:
            problems += 1

    _hdr("4. CONTRACT FIELDS")
    ot = df["option_type"].value_counts().sort("option_type")
    print(f"  option_type    : {dict(zip(ot['option_type'].to_list(), ot['count'].to_list()))}")
    if set(df["option_type"].unique().to_list()) - {"C", "P"}:
        print("  [FAIL] option_type contains values other than C/P — filters will drop them")
        problems += 1
    print(f"  instrument_id  : dtype {df['instrument_id'].dtype}, e.g. {df['instrument_id'][0]!r}")
    if df["instrument_id"].dtype != pl.Utf8:
        print("  [WARN] instrument_id is not a string — vintages may disagree on the key")
        problems += 1
    n_expired = int((df["t"] <= 0).sum())
    if n_expired:
        print(f"  [WARN] {n_expired:,} rows with t <= 0 (already-expired contracts)")
        problems += 1

    _hdr("5. CHAIN WIDTH OVER TIME — a vintage change that thins the chain shows up here")
    per = (df.with_columns(
               (pl.col("creation_timestamp_x") * 1e-3).cast(pl.Int64).alias("_s"))
             .with_columns(pl.from_epoch("_s", time_unit="s").dt.strftime("%Y-%m").alias("month"))
             .group_by("month")
             .agg(pl.col("creation_timestamp_x").n_unique().alias("snaps"),
                  pl.len().alias("rows"),
                  pl.col("expiration_timestamp_ms").n_unique().alias("expiries"))
             .sort("month"))
    print(f"  {'month':9s} {'snapshots':>10s} {'rows/snap':>10s} {'expiries':>9s}")
    rps = []
    for r in per.iter_rows(named=True):
        v = r["rows"] / max(r["snaps"], 1)
        rps.append(v)
        print(f"  {r['month']:9s} {r['snaps']:10,d} {v:10.0f} {r['expiries']:9d}")
    if rps:
        med = float(np.median(rps))
        thin = [m["month"] for m, v in zip(per.iter_rows(named=True), rps) if v < 0.5 * med]
        if thin:
            print(f"  [WARN] months with < half the median chain width ({med:.0f}): {thin}")
            problems += 1

    _hdr("VERDICT")
    if problems == 0:
        print("  No issues found — the dataset looks usable as-is.")
    else:
        print(f"  {problems} item(s) flagged above. [FAIL] blocks a backtest; [WARN] means "
              "understand it before trusting the numbers.")
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", required=True, help="cleaned option parquet to check")
    ap.add_argument("--interval-minutes", type=int, default=60)
    args = ap.parse_args()
    raise SystemExit(0 if validate(args.path, args.interval_minutes) == 0 else 1)
