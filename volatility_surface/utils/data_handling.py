import os
import sys

# Anchor to the script's own location, works both as script and in notebooks
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))  # utils -> volatility_surface -> Internship Natixis

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd 
import numpy as np 
from scipy.stats import norm 
from volatility_surface.core.pricing.volatility_dataframe import implied_volatility_dataframe

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