import pandas as pd
import numpy as np
import polars as pl
from vollib.black.implied_volatility import implied_volatility
from vollib.helpers.exceptions import PriceIsAboveMaximum, PriceIsBelowIntrinsic
from volatility_surface.core.pricing.pricing_models import bs_implied_vol, gk_implied_vol


def add_bid_ask_iv(df: pl.DataFrame, r: float = 0.0, q: float = 0.0, use_gk: bool = False) -> pl.DataFrame:
    """Vectorised bid/ask implied vol for a large polars DataFrame.

    Same math as `implied_volatility_dataframe` below, but solved for every row
    in one batch via the project's vectorised Newton-Raphson solver
    (`bs_implied_vol` / `gk_implied_vol` in pricing_models.py) instead of one
    vollib call per row — the row-wise `.apply()` is the bottleneck on
    million-row frames.

    Expects columns: bid_price, ask_price, strike, underlying_price, t, option_type.
    Forward is normalised (F=1, K = strike/underlying_price), matching the
    Deribit premium/spot convention used elsewhere. Pass use_gk=True to price
    under Garman-Kohlhagen with funding yield q instead of Black-Scholes.
    """
    K = (df['strike'] / df['underlying_price']).to_numpy()
    t = df['t'].to_numpy()
    option_type = df['option_type'].to_numpy()

    bid = np.where(df['bid_price'].to_numpy() > 0, df['bid_price'].to_numpy(), np.nan)
    ask = np.where(df['ask_price'].to_numpy() > 0, df['ask_price'].to_numpy(), np.nan)

    if use_gk:
        iv_bid = gk_implied_vol(bid, 1.0, K, t, r, q, option_type)
        iv_ask = gk_implied_vol(ask, 1.0, K, t, r, q, option_type)
    else:
        iv_bid = bs_implied_vol(bid, 1.0, K, t, r, option_type)
        iv_ask = bs_implied_vol(ask, 1.0, K, t, r, option_type)

    return df.with_columns(
        pl.Series('bid_iv', iv_bid),
        pl.Series('ask_iv', iv_ask),
    )


def implied_volatility_dataframe(row):
    """Takes a row as input and applies the implied vol function from vollib's black module


    Args:
        row (_type_): _description_
    
    returns:
        series of iv_ask and bid_ask
    """

    bid = row['bid_price']
    ask = row['ask_price']
    F = 1
    K = row['strike']/row['underlying_price']  # see if index price is better
    t = row['t']
    option_type  = row['option_type'].lower()[0]

    NAN_PAIR = (np.nan, np.nan)

    if pd.isna(bid) or bid <= 0 or pd.isna(ask) or ask <= 0:
        return NAN_PAIR
    if pd.isna(K) or K <= 0:
        return NAN_PAIR
    if pd.isna(t) or t <= 0:
        return NAN_PAIR
    if option_type == 'c' and bid > F:
        return NAN_PAIR
    if option_type == 'p' and bid > K:
        return NAN_PAIR

    try:
        iv_bid = implied_volatility(bid, F, K, 0, t, flag=option_type)
        iv_ask = implied_volatility(ask, F, K, 0, t, flag=option_type)
    except Exception:
        return NAN_PAIR
    return float(iv_bid), float(iv_ask)