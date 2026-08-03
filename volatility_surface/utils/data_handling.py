import os
import re
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
from volatility_surface.core.pricing.pricing_models import bs_delta, bs_inverse_delta

# --------------------------------------------------------------------------- #
# Schema normalisation (2024 vs 2025+ Deribit dumps)
# --------------------------------------------------------------------------- #
# The 2025+ exports were built by merging the ticker feed with the instruments
# endpoint, which is what gives them `strike`, `option_type`,
# `expiration_timestamp`, `instrument_id` — and the `_x` suffix on
# `creation_timestamp_x` (a pandas merge collision artefact). The 2024 dumps are
# the raw ticker feed: they carry `creation_timestamp` and `instrument_name` but
# none of the split-out contract fields.
#
# Deribit's instrument_name encodes all of them ("ETH-27DEC24-3000-C"), and the
# expiry token is already in the `%d%b%y` form `clean_df` parses, so the 2024
# schema is fully recoverable. Resolving it once here keeps every consumer
# (clean_df, _clean_df_pl_core, roll_engine) on a single column vocabulary.
_TS_ALIASES = ("creation_timestamp_x", "creation_timestamp")

#: CURRENCY-DDMMMYY-STRIKE-C/P. Futures and perpetuals ("ETH-PERPETUAL",
#: "ETH-27DEC24") have fewer tokens and are dropped — they are not options.
_INSTRUMENT_RE = re.compile(r"^[A-Z]+-(\d{1,2}[A-Z]{3}\d{2})-([0-9.]+)-([CP])$")

#: What everything downstream needs to exist by the end of normalisation.
_REQUIRED = ("creation_timestamp_x", "strike", "option_type", "expiration_timestamp",
             "underlying_price", "bid_price", "ask_price", "mark_iv")


def _rename_ts(cols) -> str | None:
    return next((c for c in _TS_ALIASES if c in cols), None)


def _normalise_schema_pd(df):
    """Bring a raw Deribit snapshot frame to the project's column vocabulary,
    whatever export vintage it came from.

    Renames the snapshot-time alias, and — when the contract fields are absent
    (2024) — derives `expiration_timestamp` / `strike` / `option_type` /
    `instrument_id` from `instrument_name`. Rows whose name is not an option
    (perpetuals, futures) are dropped. Raises listing what is still missing, so
    an unexpected vintage fails by name here instead of as a bare KeyError deep
    inside the feature block.
    """
    ts = _rename_ts(df.columns)
    if ts is None:
        raise KeyError(f"no snapshot-time column (tried {_TS_ALIASES}); got {list(df.columns)[:15]}")
    if ts != "creation_timestamp_x":
        df = df.rename(columns={ts: "creation_timestamp_x"})

    contract_cols = ("expiration_timestamp", "strike", "option_type")
    if not all(c in df.columns for c in contract_cols):
        if "instrument_name" not in df.columns:
            raise KeyError(
                f"missing {[c for c in contract_cols if c not in df.columns]} and no "
                f"`instrument_name` to derive them from; got {list(df.columns)[:15]}")
        parsed = df["instrument_name"].astype(str).str.extract(_INSTRUMENT_RE)
        keep = parsed[0].notna()
        df = df.loc[keep].copy()
        parsed = parsed.loc[keep]
        df["expiration_timestamp"] = parsed[0]           # '27DEC24', the %d%b%y clean_df parses
        df["strike"] = parsed[1].astype(float)
        df["option_type"] = parsed[2]
    if "instrument_id" not in df.columns and "instrument_name" in df.columns:
        # instrument_name is itself a stable unique key — roll_engine only needs
        # it to track a held position across snapshots.
        df["instrument_id"] = df["instrument_name"]

    # option_type convention differs by vintage: raw ticker CSVs (and the value
    # derived from instrument_name above) use 'C'/'P', exports merged with the
    # instruments endpoint carry Deribit's 'call'/'put', and a merge that failed
    # to match leaves NaN. Every downstream filter compares against 'C'/'P'
    # exactly, so anything else silently filters the file to zero rows.
    df["option_type"] = df["option_type"].astype(str).str[0].str.upper()
    bad = ~df["option_type"].isin(["C", "P"])
    if bad.any():
        # instrument_name is authoritative when present — prefer it over a
        # column the upstream merge may have left null or malformed.
        if "instrument_name" in df.columns:
            recovered = df.loc[bad, "instrument_name"].astype(str).str.extract(_INSTRUMENT_RE)[2]
            df.loc[bad, "option_type"] = recovered
            bad = ~df["option_type"].isin(["C", "P"])
        if bad.all():
            raise ValueError(
                f"option_type is never 'C'/'P' after normalisation (sample raw values: "
                f"{df['option_type'].unique()[:6].tolist()}) — every row would be filtered out")
        if bad.any():
            df = df.loc[~bad].copy()

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns {missing} after normalisation; "
                       f"got {list(df.columns)[:15]}")
    return df


def _normalise_schema_pl(df: 'pl.DataFrame') -> 'pl.DataFrame':
    """Polars counterpart of `_normalise_schema_pd`."""
    ts = _rename_ts(df.columns)
    if ts is None:
        raise KeyError(f"no snapshot-time column (tried {_TS_ALIASES}); got {df.columns[:15]}")
    if ts != "creation_timestamp_x":
        df = df.rename({ts: "creation_timestamp_x"})

    contract_cols = ("expiration_timestamp", "strike", "option_type")
    if not all(c in df.columns for c in contract_cols):
        if "instrument_name" not in df.columns:
            raise KeyError(
                f"missing {[c for c in contract_cols if c not in df.columns]} and no "
                f"`instrument_name` to derive them from; got {df.columns[:15]}")
        df = (df.with_columns(
                  pl.col("instrument_name").cast(pl.Utf8)
                    .str.extract_groups(_INSTRUMENT_RE.pattern).alias("_p"))
                .filter(pl.col("_p").struct.field("1").is_not_null())
                .with_columns(
                    pl.col("_p").struct.field("1").alias("expiration_timestamp"),
                    pl.col("_p").struct.field("2").cast(pl.Float64).alias("strike"),
                    pl.col("_p").struct.field("3").alias("option_type"))
                .drop("_p"))
    if "instrument_id" not in df.columns and "instrument_name" in df.columns:
        df = df.with_columns(pl.col("instrument_name").alias("instrument_id"))

    # See the pandas counterpart: 'call'/'put' vs 'C'/'P' by export vintage.
    df = df.with_columns(
        pl.col("option_type").cast(pl.Utf8).str.slice(0, 1).str.to_uppercase().alias("option_type"))

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns {missing} after normalisation; "
                       f"got {df.columns[:15]}")
    return df


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
    df = _normalise_schema_pd(df)   # 2024 dumps: creation_timestamp + instrument_name only

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
    # Same three `expiration_timestamp` schemas `_clean_df_pl_core` documents:
    #   1. bare expiry DATE string ("5JUN26") — Deribit settles at 08:00 UTC, so
    #      +8h after parsing gives the real settlement instant;
    #   2. already-resolved epoch-ms int (the merged/instruments-endpoint
    #      exports) — settlement time is baked in, adding +8h double-counts it;
    #   3. a native datetime column.
    exp_raw = clean_df['expiration_timestamp']
    if pd.api.types.is_numeric_dtype(exp_raw):
        clean_df['expiration_timestamp_ms'] = exp_raw.astype('float64')
        clean_df['expiration_timestamp'] = pd.to_datetime(exp_raw, unit='ms')
    elif pd.api.types.is_datetime64_any_dtype(exp_raw):
        clean_df['expiration_timestamp_ms'] = exp_raw.astype('int64') / 10**6
    else:
        clean_df['expiration_timestamp'] = pd.to_datetime(exp_raw, format='%d%b%y') + pd.Timedelta(hours=8)
        clean_df['expiration_timestamp_ms'] = clean_df['expiration_timestamp'].astype('int64') / 10**6
    clean_df['file_timestamp'] = pd.to_datetime(clean_df['creation_timestamp_x'], unit='ms').dt.strftime('%Y-%m-%d %H:%M')

    #Features
    clean_df['mark_iv']= clean_df['mark_iv']/100
    clean_df['t'] = (clean_df['expiration_timestamp_ms']-clean_df['creation_timestamp_x'])/(3.6e6*24*365)
    clean_df['k'] = np.log(clean_df['strike']/clean_df['underlying_price'])
    clean_df['w'] = clean_df['mark_iv']**2 * clean_df['t']
    # An already-expired contract still quoted in the snapshot has t<0, hence
    # w<0, hence sqrt(w)=NaN — a legitimate outcome (the row is dropped by the
    # bid_iv/ask_iv dropna below), but numpy warns per call, which on a bulk run
    # over tens of thousands of files buries every real message. Silence the
    # expected warning rather than the rows.
    with np.errstate(invalid='ignore', divide='ignore'):
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


def _clean_df_pl_core(df: pl.DataFrame, filter_rows: bool = True, snapshot_gap_ms: float = 1000.0,
                       otm_only: bool = True) -> pl.DataFrame:
    """Shared transform behind `clean_df_pl` (whole file in memory) and
    `clean_bulk_parquet_chunked` (day-by-day, for files too big to hold in RAM
    along with the IV solver's working arrays). See `clean_df_pl` for the
    snapshot-grouping rationale — kept here so both callers share one
    implementation instead of drifting apart.

    `filter_rows` and `otm_only` are independent — a row-quality filter
    (volume>0, no-arb boundary sanity, drop failed IV solves) and a moneyness
    filter (OTM-only) respectively. They used to be one combined flag, which
    is right for building a calibration-ready dataset (vol surface fitting is
    conventionally OTM-only — ITM options are redundant with OTM ones via
    put-call parity, and are typically thinner/quirkier quotes) but wrong for
    a trading/signal universe: a red-cell / arbitrage screen needs to see ITM
    quotes too, since a put-call-parity violation on an ITM option IS a
    tradeable signal the calibration-only view would silently discard. Use
    `filter_rows=True, otm_only=False` for that "global" dataframe, and
    `otm_only=True` (default, matches the original behaviour) when the output
    feeds calibration.
    """
    if df.is_empty():
        return df

    # Schema + option_type ('call'/'put' vs 'C'/'P') normalisation both happen
    # inside `_normalise_schema_pl`. It must run FIRST: the 2024 raw-ticker
    # vintage has no `option_type` column at all (it is derived there from
    # `instrument_name`), so touching that column beforehand raised.
    df = _normalise_schema_pl(df)
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

    # Three schemas seen in the wild for `expiration_timestamp`:
    #   1. bare expiry DATE string, live gatherer's raw CSVs ("5JUN26", no
    #      time-of-day — Deribit settles at 08:00 UTC, so +8h is added after
    #      parsing to get the real settlement instant).
    #   2. already-resolved epoch-ms int, a bulk historical dump (settlement
    #      time baked in already — confirmed by decoding a sample:
    #      1738569600000 -> 2025-02-03 08:00:00 UTC, exactly midnight+8h).
    #      Adding +8h to that would double-count the offset.
    #   3. a native Datetime column (e.g. `live_btc.parquet`, which carries a
    #      Datetime(time_unit='ns') column — inherited from a pandas
    #      datetime64[ns] upstream). MUST use `.dt.epoch()`, not `.cast(Int64)`
    #      — casting a Datetime directly reads out whatever its internal time
    #      unit is (here: nanoseconds), which is 1e6x too large if you assume
    #      milliseconds. This produced t values around 5.7e7 "years" instead
    #      of a sane fraction when first hit on real data.
    dtype = df.schema['expiration_timestamp']
    if dtype == pl.Utf8:
        df = df.with_columns(
            (pl.col('expiration_timestamp').str.strptime(pl.Datetime, '%d%b%y') + pl.duration(hours=8)).alias('expiration_timestamp')
        )
        df = df.with_columns(pl.col('expiration_timestamp').dt.epoch(time_unit='ms').alias('expiration_timestamp_ms'))
    elif dtype == pl.Datetime:
        df = df.with_columns(pl.col('expiration_timestamp').dt.epoch(time_unit='ms').alias('expiration_timestamp_ms'))
    else:
        df = df.with_columns(pl.col('expiration_timestamp').cast(pl.Int64).alias('expiration_timestamp_ms'))

    # mark_iv convention also differs by source: the live gatherer's raw CSVs
    # carry Deribit's raw PERCENTAGE value (e.g. 47.32 for 47.32% vol), needing
    # /100 to get decimal — but an already-processed file (e.g. live_btc.parquet)
    # can carry it already in decimal (0.30-0.85 range here). Dividing that by
    # 100 again silently produced a w off by exactly 1e4 (mark_iv appears
    # squared in w = mark_iv**2 * t). Crypto vol realistically never exceeds
    # ~10 (1000%) in decimal form, so median > 10 is an unambiguous signal
    # it's still in percentage form and needs the /100.
    mark_iv_median = df.select(pl.col('mark_iv').median()).item()
    df = df.with_columns(
        pl.from_epoch(pl.col('creation_timestamp_x'), time_unit='ms').dt.strftime('%Y-%m-%d %H:%M').alias('file_timestamp'),
    )
    if mark_iv_median > 10:
        df = df.with_columns((pl.col('mark_iv') / 100).alias('mark_iv'))
    df = df.with_columns(
        ((pl.col('expiration_timestamp_ms') - pl.col('creation_timestamp_x')) / (3.6e6 * 24 * 365)).alias('t'),
        (pl.col('strike') / pl.col('underlying_price')).log().alias('k'),
    )
    df = df.with_columns((pl.col('mark_iv') ** 2 * pl.col('t')).alias('w'))
    z = -pl.col('k') / pl.col('w').sqrt() + pl.col('w').sqrt() / 2
    n_d1 = (-(z ** 2) / 2).exp() / np.sqrt(2 * np.pi)  # Gatheral d1 formula, equivalent to classical one
    df = df.with_columns((n_d1 * pl.col('t').sqrt()).alias('vega'))

    # Delta, from mark_iv ("Blacks delta", no skew adjustment — matches
    # Deribit's own convention). Two columns: `delta` is the regular
    # (dollar/USD) Black-Scholes delta; `inverse_delta` is the premium-adjusted
    # hedge ratio actually needed to delta-hedge a Deribit inverse (coin-
    # settled) option with the perpetual — Delta_tilde = Delta - V/S, Lucic &
    # Sepp (2024, SSRN 4606748), Corollary 1. Plain `delta` is the WRONG hedge
    # ratio for these contracts; `inverse_delta` is the one to actually trade.
    K_norm = (df['strike'] / df['underlying_price']).to_numpy()
    t_arr = df['t'].to_numpy()
    sigma_arr = df['mark_iv'].to_numpy()
    opt_arr = df['option_type'].to_numpy()
    df = df.with_columns(
        pl.Series('delta', bs_delta(1.0, K_norm, t_arr, 0.0, sigma_arr, opt_arr)),
        pl.Series('inverse_delta', bs_inverse_delta(1.0, K_norm, t_arr, 0.0, sigma_arr, opt_arr)),
    )

    if otm_only:
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


def clean_df_pl(path: str, filter_rows: bool = True, snapshot_gap_ms: float = 1000.0,
                 otm_only: bool = True) -> pl.DataFrame:
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

    `otm_only=True` (default) matches `clean_df`'s original calibration-ready
    behaviour (OTM options only). Pass `otm_only=False` for the full universe
    (ITM included) needed by a trading/signal screen — see `_clean_df_pl_core`
    docstring for why these two use cases need different filtering.

    Loads the ENTIRE file into memory plus the IV solver's numpy working
    arrays — fine for a file of up to a few million rows, but for a much
    larger bulk dump (tens/hundreds of millions of rows) this can exceed
    available RAM and crash the process with no Python-catchable error (an OS
    OOM kill, not an exception). Use `clean_bulk_parquet_chunked` instead for
    files that large.
    """
    return _clean_df_pl_core(pl.read_parquet(path), filter_rows=filter_rows,
                              snapshot_gap_ms=snapshot_gap_ms, otm_only=otm_only)


def clean_bulk_parquet_chunked(path: str, out_dir: str, filter_rows: bool = True,
                                snapshot_gap_ms: float = 1000.0, chunk_days: float = 1.0,
                                otm_only: bool = True) -> None:
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
    # The windowing below slices on the raw file's own column, before
    # `_clean_df_pl_core` gets a chance to normalise the alias, so resolve the
    # name here too (2024 dumps: `creation_timestamp`).
    schema_cols = lf.collect_schema().names()
    ts_col = next((c for c in _TS_ALIASES if c in schema_cols), None)
    if ts_col is None:
        raise KeyError(f"no snapshot-time column found (tried {_TS_ALIASES}) in {path}")
    bounds = lf.select(
        pl.col(ts_col).min().alias('lo'),
        pl.col(ts_col).max().alias('hi'),
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
            (pl.col(ts_col) >= start) & (pl.col(ts_col) < end)
        ).collect()
        if not chunk.is_empty():
            cleaned = _clean_df_pl_core(chunk, filter_rows=filter_rows, snapshot_gap_ms=snapshot_gap_ms, otm_only=otm_only)
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