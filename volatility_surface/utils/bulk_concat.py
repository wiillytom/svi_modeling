"""
Bulk-concatenate a large directory of raw Deribit CSV snapshots into one
cleaned parquet file, without ever holding more than one batch in memory.

Built for the ~170k-file Jan-Jun 2025 minute/5-min CSV dump — `parquet_creation`
in `data_handling.py` loads every cleaned DataFrame into a Python list before
concatenating, which doesn't scale past a few thousand files. This version
processes in batches, parallelized across CPU cores (the per-file cost is
dominated by `clean_df`'s row-wise Let's-Be-Rational IV inversion), and
streams each batch straight to an open ParquetWriter — same append-only
pattern as the live-gather archive, so memory use stays flat regardless of
how many files there are.

Usage:
    python volatility_surface/utils/bulk_concat.py \
        --root "/path/to/eth/2025/deribit" \
        --out "2 - Data/parquets/eth_2025_cleaned.parquet" \
        --workers 8
"""

from __future__ import annotations

import argparse
import glob
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

from volatility_surface.utils.data_handling import clean_df


def _clean_one(args: tuple[str, bool]) -> pd.DataFrame | None:
    path, filter_rows = args
    try:
        df = clean_df(path, filter_rows=filter_rows)
        return df if not df.empty else None
    except Exception as e:
        print(f"  [skip] {path}: {e!r}")
        return None


def bulk_concat(root: str, out_path: str, filter_rows: bool = True,
                 workers: int = 8, batch_size: int = 500) -> None:
    files = sorted(glob.glob(str(Path(root) / "**" / "*.csv"), recursive=True))
    n_files = len(files)
    print(f"[bulk_concat] found {n_files} CSV files under {root}")
    if n_files == 0:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    n_rows = 0
    n_skipped = 0
    t0 = time.time()

    with Pool(workers) as pool:
        for batch_start in range(0, n_files, batch_size):
            batch_files = files[batch_start:batch_start + batch_size]
            args = [(f, filter_rows) for f in batch_files]
            results = pool.map(_clean_one, args)

            frames = [r for r in results if r is not None]
            n_skipped += len(results) - len(frames)
            if not frames:
                continue

            batch_df = pd.concat(frames, ignore_index=True)
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

    if writer is not None:
        writer.close()
    print(f"[bulk_concat] done: {n_rows} rows, {n_skipped} files skipped "
          f"-> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory to scan recursively for CSVs")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--no-filter", action="store_true",
                     help="keep every row (skip volume/no-arb/OTM filters) — for EDA")
    args = ap.parse_args()
    bulk_concat(args.root, args.out, filter_rows=not args.no_filter,
                workers=args.workers, batch_size=args.batch_size)
