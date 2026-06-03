import os
import time
import requests
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime

# --- CONFIGURATION ---
POLLING_INTERVAL_SECONDS = 1  # Adjust to 3600 for hourly intervals
BASE_URL = 'https://www.deribit.com/api/v2'
TARGET_COLUMNS = [
    'bid_price', 'ask_price', 'open_interest', 'mark_price', 'creation_timestamp_x', 
    'volume', 'mark_iv', 'underlying_price', 'underlying_index', 'estimated_delivery_price', 
    'mid_price', 'price_index', 'expiration_timestamp', 'strike', 'settlement_period', 
    'option_type', 'instrument_id'
]

def get_index_price(index_name: str = 'btc_usd') -> Optional[float]:
    """
    Fetches the current index price from Deribit.
    
    Args:
        index_name (str): The index identifier (default: 'btc_usd').
        
    Returns:
        Optional[float]: The index price, or None if the request fails.
    """
    url = f"{BASE_URL}/public/get_index_price"
    params = {'index_name': index_name}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('result', {}).get('index_price')
    except requests.exceptions.RequestException as e:
        print(f"Network or API error fetching index price: {e}")
        return None

def get_book_summary(currency: str = 'btc', kind: str = 'option') -> List[Dict[str, Any]]:
    """
    Fetches the market data summary for all instruments of a specific currency and kind.
    
    Args:
        currency (str): The underlying currency (default: 'btc').
        kind (str): The instrument kind (default: 'option').
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the market data.
    """
    url = f"{BASE_URL}/public/get_book_summary_by_currency"
    params = {'currency': currency, 'kind': kind}
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get('result', [])
    except requests.exceptions.RequestException as e:
        print(f"Network or API error fetching book summary: {e}")
        return []

def process_data(book_data: List[Dict[str, Any]], price_index: float) -> pd.DataFrame:
    """
    Maps raw API payloads to the strict DataFrame schema and derives necessary fields.
    
    Args:
        book_data (List[Dict[str, Any]]): Raw JSON data from the book summary endpoint.
        price_index (float): The current index price.
        
    Returns:
        pd.DataFrame: Cleaned data conforming to the required schema.
    """
    if not book_data:
        return pd.DataFrame(columns=TARGET_COLUMNS)
        
    df = pd.DataFrame(book_data)
    
    # 1. Map existing API fields to the required schema names
    if 'creation_timestamp' in df.columns:
        df = df.rename(columns={'creation_timestamp': 'creation_timestamp_x'})
    if 'instrument_name' in df.columns:
        df = df.rename(columns={'instrument_name': 'instrument_id'})
        
    # 2. Append the isolated index price
    df['price_index'] = price_index
    
    # 3. Parse derivative fields from instrument_id (Format: BASE-DATE-STRIKE-TYPE)
    if 'instrument_id' in df.columns:
        parsed = df['instrument_id'].str.split('-', expand=True)
        if parsed.shape[1] == 4:
            df['expiration_timestamp'] = parsed[1]
            df['strike'] = pd.to_numeric(parsed[2], errors='coerce')
            df['option_type'] = parsed[3]
        else:
            df['expiration_timestamp'] = pd.NA
            df['strike'] = pd.NA
            df['option_type'] = pd.NA
    else:
        df['instrument_id'] = pd.NA
        
    # 4. Fill missing or unsupported fields with NA
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
            
    # 5. Filter and enforce strict column order
    return df[TARGET_COLUMNS]
def save_to_csv(df: pd.DataFrame) -> None:
    """
    Saves the DataFrame to a new CSV file named with the current timestamp.
    """
    if df.empty:
        return
        
    # Generate timestamped filename: btc_option_data_YYYY-MM-DD_HHhmm.csv
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"eth_{timestamp}.csv"
    
    # Save to the current directory (or change to 'data/' if needed)
    path = r'/Users/macbookair/Internship Natixis/data/market_making_data/'
    df.to_csv(path+filename, index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved data to {filename}")

def run_job() -> None:
    price_index = get_index_price('btc_usd')
    book_summary = get_book_summary('btc', 'option')
    
    if price_index is not None and book_summary:
        df = process_data(book_summary, price_index)
        save_to_csv(df)
    else:
        print("Data ingestion skipped: API response missing.")

def main():
    """
    Initializes the execution loop based on configured constraints.
    """
    print(f"Engine started. Polling Deribit API every {POLLING_INTERVAL_SECONDS} seconds.")
    print("Press Ctrl+C to terminate.")
    
    while True:
        try:
            run_job()
        except Exception as e:
            print(f"Critical error during execution cycle: {e}")
            
        time.sleep(POLLING_INTERVAL_SECONDS)

if __name__ == "__main__":
    # Ensure pandas utilizes standard null handling for consistency in CSV outputs
    pd.options.mode.chained_assignment = None
    main()