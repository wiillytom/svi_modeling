"""
Bulk-concatenate a large directory tree of raw Deribit CSV snapshots into one
cleaned parquet file, without ever holding more than one batch in memory.

Built for the ~170k-file Jan-Jun 2025 minute/5-min CSV dump — `parquet_creation`
in `data_handling.py` loads every cleaned DataFrame into a Python list before
concatenating, which doesn't scale past a few thousand files. This version
processes in batches, parallelized across CPU cores (the per-file cost is
dominated by `clean_df`'s row-wise Let's-Be-Rational IV inversion), and
streams each batch straight to an open ParquetWriter — same append-only
pattern as the live-gather archive, so memory use stays flat regardless of
how many files there are.

DOWNSAMPLING (`--interval-minutes`): one CSV is one snapshot (`clean_df` reads
`creation_timestamp_x.iloc[0]` and broadcasts it), so thinning the history to an
hourly grid is a matter of picking which FILES to open — the other 59 per hour
are never read at all. That matters because the IV inversion inside `clean_df`
is ~all of the cost: an hourly build over a minute dump is ~60x less work, not
just 60x less storage. Verified against the backtester: keeping the FIRST file
of each hour reproduces a minute-data backtest to 0.00e+00, while keeping the
LAST one shifts every hedge and roll by 59 minutes and moves returns by up to
1.7 percentage points — so this selects the first, and `roll_engine`'s hourly
decision grid lines up exactly.

Usage:
    # three years, hourly, into one parquet
    python volatility_surface/utils/bulk_concat.py \
        --root echanges/P.Nom/deribit/2024 \
               echanges/P.Nom/deribit/2025 \
               echanges/P.Nom/deribit/2026 \
        --out "2 - Data/parquets/eth_hourly_2024_2026.parquet" \
        --glob "eth_*.csv" --interval-minutes 60 --workers 8

    # count what would be processed, without processing it
    ... --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import re
import signal
import sys
import time
from multiprocessing import Pool
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from volatility_surface.utils.data_handling import _TS_ALIASES, clean_df


# --------------------------------------------------------------------------- #
# Timestamp extraction
# --------------------------------------------------------------------------- #
# Two naming conventions exist in this project's own data already
# (`eth_option_data_2026-05-21_15h45m.csv` and `eth_2026-06-30_17-04.csv`), and a
# third dump may use another, so try a few shapes and fall back to reading the
# file when none match.
_TS_PATTERNS = [
    re.compile(r"(\d{4})-(\d{2})-(\d{2})[_\-T ](\d{2})h(\d{2})m"),      # 2026-05-21_15h45m
    re.compile(r"(\d{4})-(\d{2})-(\d{2})[_\-T ](\d{2})[-:_.](\d{2})"),  # 2026-06-30_17-04
    re.compile(r"(\d{4})(\d{2})(\d{2})[_\-T ]?(\d{2})(\d{2})"),         # 20260630_1704
]


def _ts_from_name(path: str) -> dt.datetime | None:
    """UTC datetime parsed from the file NAME, or None if no pattern matches."""
    name = os.path.basename(path)
    for pat in _TS_PATTERNS:
        m = pat.search(name)
        if m:
            y, mo, d, h, mi = (int(x) for x in m.groups())
            try:
                return dt.datetime(y, mo, d, h, mi, tzinfo=dt.timezone.utc)
            except ValueError:
                continue
    return None


def _ts_from_csv(path: str) -> dt.datetime | None:
    """UTC datetime read from the file's first data row — the authoritative
    source, used when the filename can't be parsed. Reads one row, not the file."""
    try:
        head = pd.read_csv(path, nrows=1)
        if head.empty:
            return None
        col = next((c for c in _TS_ALIASES if c in head.columns), None)
        if col is None:  # 2024 dumps name it `creation_timestamp`, see _TS_ALIASES
            return None
        return dt.datetime.fromtimestamp(float(head[col].iloc[0]) / 1000.0, dt.timezone.utc)
    except Exception:
        return None


def _ts_worker(path: str) -> tuple[str, float | None]:
    ts = _ts_from_csv(path)
    return path, (ts.timestamp() if ts else None)


def select_files(roots: list[str], pattern: str = "*.csv",
                 interval_minutes: int = 60, workers: int = 8,
                 verbose: bool = True) -> list[str]:
    """Every CSV under `roots` thinned to one file per `interval_minutes`
    bucket, chronologically ordered.

    Within a bucket the EARLIEST file wins, so an hourly build lands on HH:00 —
    the same instants `roll_engine` hedges and rolls at (see module docstring).
    Timestamps come from the filename where possible; if more than 5% of files
    fail to parse, every timestamp is re-read from the CSVs in parallel instead,
    so an unrecognised naming scheme degrades in speed, never in correctness.
    """
    files: list[str] = []
    for root in roots:
        found = glob.glob(str(Path(root) / "**" / pattern), recursive=True)
        if verbose:
            print(f"[select] {root}: {len(found)} files matching {pattern!r}")
        files.extend(found)
    if not files:
        return []

    stamped = [(f, _ts_from_name(f)) for f in files]
    n_bad = sum(1 for _, ts in stamped if ts is None)
    if n_bad > 0.05 * len(files):
        if verbose:
            print(f"[select] filename parsing failed on {n_bad}/{len(files)} files "
                  f"— reading timestamps from CSV headers ({workers} workers)…")
        pool = Pool(workers, initializer=_ignore_sigint)
        try:
            pairs = pool.map(_ts_worker, files, chunksize=256)
        finally:
            pool.close()
            pool.join()
        stamped = [(p, dt.datetime.fromtimestamp(t, dt.timezone.utc) if t else None)
                   for p, t in pairs]
    elif n_bad and verbose:
        print(f"[select] {n_bad} files with unparseable names — dropped")

    bucket_s = interval_minutes * 60
    best: dict[int, tuple[float, str]] = {}
    for path, ts in stamped:
        if ts is None:
            continue
        epoch = ts.timestamp()
        key = int(epoch // bucket_s)
        cur = best.get(key)
        if cur is None or epoch < cur[0]:
            best[key] = (epoch, path)

    chosen = [p for _, p in sorted(best.values())]
    if verbose and chosen:
        first = dt.datetime.fromtimestamp(sorted(best.values())[0][0], dt.timezone.utc)
        last = dt.datetime.fromtimestamp(sorted(best.values())[-1][0], dt.timezone.utc)
        print(f"[select] {len(files)} files -> {len(chosen)} kept "
              f"(1 per {interval_minutes} min, {len(files)/max(len(chosen),1):.0f}x fewer)")
        print(f"[select] range {first:%Y-%m-%d %H:%M} -> {last:%Y-%m-%d %H:%M} UTC")
    return chosen


# --------------------------------------------------------------------------- #
# Cleaning pipeline
# --------------------------------------------------------------------------- #
def _ignore_sigint():
    # Workers must not handle SIGINT themselves — otherwise Ctrl+C hits every
    # worker mid-task and each prints its own traceback. Only the main
    # process should react, then terminate the pool cleanly on its own terms.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _clean_one(args: tuple[str, bool]) -> pd.DataFrame | None:
    path, filter_rows = args
    try:
        df = clean_df(path, filter_rows=filter_rows)
        return df if not df.empty else None
    except Exception as e:
        print(f"  [skip] {path}: {e!r}")
        return None


def bulk_concat(root: str | list[str], out_path: str, filter_rows: bool = True,
                workers: int = 8, batch_size: int = 500, pattern: str = "*.csv",
                interval_minutes: int | None = None, dry_run: bool = False) -> None:
    """Clean every selected CSV under `root` into one parquet.

    `root` may be a single directory or a list of them (e.g. one per year) —
    they are globbed together and the result is emitted in one chronological
    stream, so a multi-year dump split across folders still produces a single
    time-ordered file.

    `interval_minutes`: thin to one snapshot per interval before any file is
    opened (60 = hourly, the grid `roll_engine` actually uses). None keeps every
    file, the original behaviour.
    """
    roots = [root] if isinstance(root, str) else list(root)
    if interval_minutes:
        files = select_files(roots, pattern=pattern, interval_minutes=interval_minutes,
                             workers=workers)
    else:
        files = sorted(f for r in roots
                       for f in glob.glob(str(Path(r) / "**" / pattern), recursive=True))
        print(f"[bulk_concat] found {len(files)} CSV files")

    n_files = len(files)
    if n_files == 0:
        print("[bulk_concat] nothing to do")
        return
    if dry_run:
        print(f"[bulk_concat] dry run — {n_files} files would be processed, exiting")
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    col_order: list[str] | None = None   # pinned from the first batch, see below
    dropped_cols: set[str] = set()
    n_rows = 0
    n_skipped = 0
    t0 = time.time()

    pool = Pool(workers, initializer=_ignore_sigint)
    try:
        for batch_start in range(0, n_files, batch_size):
            batch_files = files[batch_start:batch_start + batch_size]
            args = [(f, filter_rows) for f in batch_files]
            results = pool.map(_clean_one, args)

            frames = [r for r in results if r is not None]
            n_skipped += len(results) - len(frames)
            if not frames:
                continue

            batch_df = pd.concat(frames, ignore_index=True)

            # Column ORDER is not stable across the raw CSVs (observed on the
            # 2024 dump: `volume`/`interest_rate` swap places between files).
            # pyarrow's cast matches fields positionally and refuses a reordered
            # schema, so pin the first batch's order and reindex every later one
            # onto it — otherwise the run dies mid-way with "Target schema's
            # field names are not matching".
            if col_order is None:
                col_order = list(batch_df.columns)
            else:
                missing = [c for c in col_order if c not in batch_df.columns]
                extra = [c for c in batch_df.columns if c not in col_order]
                if missing:
                    for c in missing:
                        batch_df[c] = pd.NA
                if extra:
                    # A column absent from the first batch can't be added to an
                    # already-open writer; drop it, but say so once per name.
                    for c in extra:
                        if c not in dropped_cols:
                            print(f"[bulk_concat] column {c!r} absent from the first "
                                  f"batch's schema — dropped for the whole file")
                            dropped_cols.add(c)
                batch_df = batch_df[col_order]

            table = pa.Table.from_pandas(batch_df, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(out_path, schema)
            elif table.schema != schema:
                table = table.cast(schema)

            writer.write_table(table)
            n_rows += len(batch_df)

            done = min(batch_start + batch_size, n_files)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta_min = (n_files - done) / rate / 60 if rate > 0 else float("nan")
            print(f"[bulk_concat] {done}/{n_files} files  "
                  f"({n_rows} rows so far, {n_skipped} files skipped)  "
                  f"{rate:.1f} files/s  ETA {eta_min:.1f} min")
    except KeyboardInterrupt:
        print("\n[bulk_concat] interrupted — stopping workers and closing "
              f"output (partial: {n_rows} rows so far) …")
        pool.terminate()
        pool.join()
        if writer is not None:
            writer.close()
        return
    else:
        pool.close()
        pool.join()

    if writer is not None:
        writer.close()
    print(f"[bulk_concat] done: {n_rows} rows, {n_skipped} files skipped "
          f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, nargs="+",
                    help="one or more directories to scan recursively for CSVs")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--glob", default="*.csv",
                    help="filename pattern, e.g. 'eth_*.csv' to take one coin only")
    ap.add_argument("--interval-minutes", type=int, default=60,
                    help="keep one snapshot per N minutes (60 = hourly, the grid the "
                         "backtester uses); pass 0 to keep every file")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true",
                    help="report how many files would be processed, then exit")
    ap.add_argument("--no-filter", action="store_true",
                    help="keep every row (skip volume/no-arb/OTM filters) — for EDA")
    args = ap.parse_args()
    bulk_concat(args.root, args.out, filter_rows=not args.no_filter,
                workers=args.workers, batch_size=args.batch_size, pattern=args.glob,
                interval_minutes=args.interval_minutes or None, dry_run=args.dry_run)
