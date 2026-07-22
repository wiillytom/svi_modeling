import os
import sys

# Anchor to the script's own location, works both as script and in notebooks
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))  # utils -> volatility_surface -> Internship Natixis

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import polars as pl
from scipy.stats import norm
from volatility_surface.core.pricing.volatility_dataframe import implied_volatility_dataframe, add_bid_ask_iv

def clean_df(path: str, filter_rows: bool = True):
    """Cleans the market microstructure data
    #See if we only take otm options ? 'c' & k>0 otm calls, 'p' & k<0 otm puts

    Args:
        path (str): path of the file we want to use as the df
        filter_rows (bool): if True (default), apply the volume>0, no-arb
            boundary, OTM and bid/ask-inversion-dropna filters used to build
            the calibration parquet. If False, keep every row (still with
            all derived columns) — use this for EDA so filtering doesn't
            bias what you're looking at; see `load_uncleaned`.
    returns: pd.DataFrame
    """

    df = pd.read_csv(path)
    if df.empty:
        return df

    # Grab the snapshot timestamp from the raw frame, before any filtering —
    # a snapshot with zero rows surviving the no-arb boundary filter is a
    # normal (if illiquid) outcome, not an error; taking .iloc[0] off the
    # already-filtered frame crashed on those instead of returning empty.
    ts = df.creation_timestamp_x.iloc[0]

    if filter_rows:
        df = df[df.volume>0]
        put_boundary_mask = (df['underlying_price']*df['bid_price']<df['strike']) & (df['option_type']=='P')
        call_boundary_mask = (df['option_type']=='C') & (df['bid_price']<1)
        clean_df = df[put_boundary_mask | call_boundary_mask].copy()
    else:
        clean_df = df.copy()

    #Timestamp standardisation
    clean_df['creation_timestamp_x'] = ts
    clean_df['creation_timestamp_x'] = clean_df['creation_timestamp_x'] #2h, paris = gmt +2
    clean_df['expiration_timestamp'] = pd.to_datetime(clean_df['expiration_timestamp'], format='%d%b%y')+pd.Timedelta(hours=8)
    clean_df['expiration_timestamp_ms'] = clean_df['expiration_timestamp'].astype('int64') / 10**6
    clean_df['file_timestamp'] = pd.to_datetime(clean_df['creation_timestamp_x'], unit='ms').dt.strftime('%Y-%m-%d %H:%M')

    #Features
    clean_df['mark_iv']= clean_df['mark_iv']/100
    clean_df['t'] = (clean_df['expiration_timestamp_ms']-clean_df['creation_timestamp_x'])/(3.6e6*24*365)
    clean_df['k'] = np.log(clean_df['strike']/clean_df['underlying_price'])
    clean_df['w'] = clean_df['mark_iv']**2 * clean_df['t']
    n_d1 = norm.pdf(-clean_df['k']/np.sqrt(clean_df['w']) + np.sqrt(clean_df['w'])/2) #Gatheral d1 formula, equivalent to classical one
    clean_df['vega'] = n_d1*np.sqrt(clean_df['t']) # Vega is to be multiplied by 1 vol point; Since we express vol as decimal, this is the adequate format
    #if we expressed vol as a percentage, we would divide vega by 100: e.g. 0.42 vol, 0.3 vega <=> 42 vol, 0.003 vega

    if filter_rows:
        #OTM Filtering
        otm_call_mask = (clean_df.option_type=='C') & (clean_df.k >=0)
        otm_put_mask = (clean_df.option_type=='P') & (clean_df.k<=0)
        clean_df = clean_df[otm_call_mask | otm_put_mask]

    if clean_df.empty:
        # `.apply(..., result_type='expand')` on an empty frame returns 0
        # columns, not 2 — assigning that into ['bid_iv','ask_iv'] raises.
        # A snapshot with nothing left after filtering is a normal (illiquid)
        # outcome, not an error.
        clean_df['bid_iv'] = pd.Series(dtype=float)
        clean_df['ask_iv'] = pd.Series(dtype=float)
    else:
        #IV Quotes
        clean_df[['bid_iv','ask_iv']] = clean_df.apply(implied_volatility_dataframe, axis=1, result_type='expand')
    if filter_rows:
        clean_df.dropna(subset=['bid_iv','ask_iv'], inplace=True)
    clean_df['half_spread_iv'] = (clean_df['ask_iv'] - clean_df['bid_iv'])/2
    clean_df['half_spread'] = (clean_df['ask_price'] - clean_df['bid_price'])/2
    return clean_df


def _clean_df_pl_core(df: pl.DataFrame, filter_rows: bool = True, snapshot_gap_ms: float = 1000.0) -> pl.DataFrame:
    """Shared transform behind `clean_df_pl` (whole file in memory) and
    `clean_bulk_parquet_chunked` (day-by-day, for files too big to hold in RAM
    along with the IV solver's working arrays). See `clean_df_pl` for the
    snapshot-grouping rationale — kept here so both callers share one
    implementation instead of drifting apart.
    """
    if df.is_empty():
        return df

    df = df.sort('creation_timestamp_x')
    new_snapshot = pl.col('creation_timestamp_x').diff().fill_null(snapshot_gap_ms + 1) > snapshot_gap_ms
    df = df.with_columns(new_snapshot.cast(pl.Int64).cum_sum().alias('_snapshot_id'))
    df = df.with_columns(
        pl.col('creation_timestamp_x').min().over('_snapshot_id').alias('creation_timestamp_x')
    )

    if filter_rows:
        put_boundary = (pl.col('underlying_price') * pl.col('bid_price') < pl.col('strike')) & (pl.col('option_type') == 'P')
        call_boundary = (pl.col('option_type') == 'C') & (pl.col('bid_price') < 1)
        df = df.filter(pl.col('volume') > 0).filter(put_boundary | call_boundary)

    df = df.with_columns(
        (pl.col('expiration_timestamp').str.strptime(pl.Datetime, '%d%b%y') + pl.duration(hours=8)).alias('expiration_timestamp')
    )
    df = df.with_columns(
        pl.col('expiration_timestamp').dt.epoch(time_unit='ms').alias('expiration_timestamp_ms'),
        pl.from_epoch(pl.col('creation_timestamp_x'), time_unit='ms').dt.strftime('%Y-%m-%d %H:%M').alias('file_timestamp'),
        (pl.col('mark_iv') / 100).alias('mark_iv'),
    )
    df = df.with_columns(
        ((pl.col('expiration_timestamp_ms') - pl.col('creation_timestamp_x')) / (3.6e6 * 24 * 365)).alias('t'),
        (pl.col('strike') / pl.col('underlying_price')).log().alias('k'),
    )
    df = df.with_columns((pl.col('mark_iv') ** 2 * pl.col('t')).alias('w'))
    z = -pl.col('k') / pl.col('w').sqrt() + pl.col('w').sqrt() / 2
    n_d1 = (-(z ** 2) / 2).exp() / np.sqrt(2 * np.pi)  # Gatheral d1 formula, equivalent to classical one
    df = df.with_columns((n_d1 * pl.col('t').sqrt()).alias('vega'))

    if filter_rows:
        otm_call = (pl.col('option_type') == 'C') & (pl.col('k') >= 0)
        otm_put = (pl.col('option_type') == 'P') & (pl.col('k') <= 0)
        df = df.filter(otm_call | otm_put)

    df = df.drop('_snapshot_id')

    if df.is_empty():
        df = df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias('bid_iv'),
            pl.lit(None, dtype=pl.Float64).alias('ask_iv'),
        )
    else:
        df = add_bid_ask_iv(df)

    if filter_rows:
        df = df.drop_nulls(subset=['bid_iv', 'ask_iv'])

    df = df.with_columns(
        ((pl.col('ask_iv') - pl.col('bid_iv')) / 2).alias('half_spread_iv'),
        ((pl.col('ask_price') - pl.col('bid_price')) / 2).alias('half_spread'),
    )
    return df


def clean_df_pl(path: str, filter_rows: bool = True, snapshot_gap_ms: float = 1000.0) -> pl.DataFrame:
    """Polars port of `clean_df`, for a BULK raw parquet with many snapshots
    concatenated (e.g. a multi-month 1-min history), not one file per snapshot.

    Snapshot grouping: within one real snapshot, `creation_timestamp_x` jitters
    by only a few ms across instruments (observed ~10-20ms spread per poll on
    the raw CSVs); real snapshots are >= 1 minute apart. Rows are sorted by
    `creation_timestamp_x` and a new snapshot starts whenever the gap since the
    previous row exceeds `snapshot_gap_ms` (default 1s — well above intra-
    snapshot jitter, well below any realistic polling interval). Every row in
    a snapshot is then stamped with that snapshot's min timestamp, replacing
    the old single-file `ts = df.creation_timestamp_x.iloc[0]` broadcast.
    Output is sorted by creation_timestamp_x (order is not preserved from the
    input file).

    Loads the ENTIRE file into memory plus the IV solver's numpy working
    arrays — fine for a file of up to a few million rows, but for a much
    larger bulk dump (tens/hundreds of millions of rows) this can exceed
    available RAM and crash the process with no Python-catchable error (an OS
    OOM kill, not an exception). Use `clean_bulk_parquet_chunked` instead for
    files that large.
    """
    return _clean_df_pl_core(pl.read_parquet(path), filter_rows=filter_rows, snapshot_gap_ms=snapshot_gap_ms)


def clean_bulk_parquet_chunked(path: str, out_dir: str, filter_rows: bool = True,
                                snapshot_gap_ms: float = 1000.0, chunk_days: float = 1.0) -> None:
    """Same cleaning as `clean_df_pl`, but for a raw parquet too large to fit
    in memory alongside the IV solver's working arrays all at once (this is
    what a ~96M-row, 6-month 1-min dump needs — `clean_df_pl` OOM-crashes the
    kernel on that scale, silently, since an OS memory kill isn't a Python
    exception).

    Slices the file into `chunk_days`-sized windows of `creation_timestamp_x`
    (using polars' lazy scanner, so only each window's rows are ever
    materialised — not the whole file), cleans each window with the same
    `_clean_df_pl_core` logic `clean_df_pl` uses, and writes one output
    parquet per non-empty chunk into `out_dir`. Read the result back as one
    logical dataset (still lazily, no need to hold it all in memory) with:

        pl.scan_parquet(f"{out_dir}/*.parquet")

    Caveat: a real snapshot landing exactly on a chunk boundary (e.g. a poll
    at 23:59:59.99x ms) could be split across two chunks and end up counted as
    two partial snapshots instead of one. This is rare (at most one snapshot
    per chunk boundary — ~1 in 1440 for chunk_days=1 at 1-min frequency) and
    is not corrected for. If you need it exact, either shrink chunk_days or
    post-process rows near each boundary.
    """
    os.makedirs(out_dir, exist_ok=True)
    lf = pl.scan_parquet(path)
    bounds = lf.select(
        pl.col('creation_timestamp_x').min().alias('lo'),
        pl.col('creation_timestamp_x').max().alias('hi'),
    ).collect()
    if bounds.is_empty() or bounds['lo'][0] is None:
        return
    lo_ms, hi_ms = int(bounds['lo'][0]), int(bounds['hi'][0])
    window_ms = int(chunk_days * 24 * 3600 * 1000)

    start = lo_ms
    chunk_idx = 0
    while start <= hi_ms:
        end = start + window_ms
        chunk = lf.filter(
            (pl.col('creation_timestamp_x') >= start) & (pl.col('creation_timestamp_x') < end)
        ).collect()
        if not chunk.is_empty():
            cleaned = _clean_df_pl_core(chunk, filter_rows=filter_rows, snapshot_gap_ms=snapshot_gap_ms)
            if not cleaned.is_empty():
                out_path = os.path.join(out_dir, f'chunk_{chunk_idx:05d}.parquet')
                cleaned.write_parquet(out_path)
                print(f'chunk {chunk_idx}: {chunk.height} raw rows -> {cleaned.height} cleaned rows -> {out_path}')
            else:
                print(f'chunk {chunk_idx}: {chunk.height} raw rows -> 0 cleaned rows, skipped')
        chunk_idx += 1
        start = end


import glob
from pathlib import Path
def parquet_creation(underlying: str):
# Get all CSV files from data/ folder
    underlying=underlying.lower()
    csv_files = sorted(glob.glob(os.path.join(project_root, '2 - Data', underlying, '*.csv')))

    # Load and clean all CSV files
    global_df = pd.concat([clean_df(file) for file in csv_files], ignore_index=True)

    # Save to parquet
    global_df.to_parquet(os.path.join(project_root, '2 - Data', 'parquets', f'{underlying}_options_data_cleaned.parquet'))


def load_uncleaned(underlying: str) -> pd.DataFrame:
    """
    Concatenate every raw CSV snapshot for `underlying` with derived columns
    (t, k, w, vega, mark_iv, bid_iv, ask_iv, half_spread_iv) but WITHOUT the
    volume / no-arb-boundary / OTM / bid-ask-inversion-dropna filters that
    `clean_df` applies before writing the calibration parquet.

    For exploratory analysis only — using the filtered parquet for EDA hides
    exactly the rows (illiquid, boundary-violating, ITM) whose absence is
    itself informative. Calibration should keep using `clean_df` / the
    parquet as before.
    """
    underlying = underlying.lower()
    csv_files = sorted(glob.glob(os.path.join(project_root, '2 - Data', underlying, '*.csv')))
    return pd.concat([clean_df(f, filter_rows=False) for f in csv_files], ignore_index=True)


if __name__ == '__main__':
    parquet_creation('btc')
    parquet_creation('eth')