import pandas as pd 
import numpy as np 
from vollib.black.implied_volatility import implied_volatility
from vollib.helpers.exceptions import PriceIsAboveMaximum, PriceIsBelowIntrinsic

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