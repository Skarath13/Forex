import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import time
import json

# Map of forex pairs to Yahoo Finance symbols
FOREX_SYMBOLS = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X',
    'AUDUSD': 'AUDUSD=X',
    'USDCHF': 'USDCHF=X',
    'NZDUSD': 'NZDUSD=X',
    'USDCAD': 'USDCAD=X',
    'EURJPY': 'EURJPY=X',
    'GBPJPY': 'GBPJPY=X',
    'EURGBP': 'EURGBP=X'
}

# Map of timeframes to yfinance intervals
TIMEFRAME_MAP = {
    '1H': '1h',
    '4H': '4h',
    'D': '1d',
    'W': '1wk',
    'M': '1mo'
}

def fetch_forex_data(instruments, timeframes, start_date=None, end_date=None, cache_dir='data_cache'):
    """
    Fetch historical forex data from Yahoo Finance
    
    Args:
        instruments: List of forex pairs to fetch
        timeframes: List of timeframes to fetch
        start_date: Start date for historical data (string 'YYYY-MM-DD' or datetime)
        end_date: End date for historical data (string 'YYYY-MM-DD' or datetime)
        cache_dir: Directory to cache data
        
    Returns:
        Dictionary of market data in the same format as generate_sample_data
    """
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
    if start_date is None:
        # Set start date based on longest timeframe
        # For monthly data, we need several years for better analysis
        if 'M' in timeframes:
            start_date = end_date - timedelta(days=365*3)  # 3 years for monthly
        # For weekly data, we need at least a year
        elif 'W' in timeframes:
            start_date = end_date - timedelta(days=365*2)  # 2 years for weekly
        # For daily data, we want more history for better analysis
        elif 'D' in timeframes:
            start_date = end_date - timedelta(days=500)
        else:
            start_date = end_date - timedelta(days=60)
    elif isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    market_data = {}
    correlation_data = {}
    
    # Fetch data for each instrument
    for instrument in instruments:
        market_data[instrument] = {}
        
        # Check if instrument is in the supported list
        if instrument not in FOREX_SYMBOLS:
            print(f"Warning: {instrument} is not in the supported list. Using EURUSD as fallback.")
            symbol = FOREX_SYMBOLS['EURUSD']
        else:
            symbol = FOREX_SYMBOLS[instrument]
        
        # Fetch data for each timeframe
        for timeframe in timeframes:
            # Map timeframe to yfinance interval
            if timeframe not in TIMEFRAME_MAP:
                print(f"Warning: {timeframe} is not in the supported list. Using 1H as fallback.")
                interval = TIMEFRAME_MAP['1H']
            else:
                interval = TIMEFRAME_MAP[timeframe]
                
            # Create cache filename
            cache_file = os.path.join(cache_dir, f"{instrument}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")
            
            # Check if data is cached
            if os.path.exists(cache_file) and (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days < 1:
                try:
                    # Use cached data if less than 1 day old
                    print(f"Loading cached data for {instrument} {timeframe}")
                    data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    
                    # Ensure numeric columns are actually numeric
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in data.columns:
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                    
                    # Check if any columns have non-numeric data
                    if data[['open', 'high', 'low', 'close']].isna().any().any():
                        print(f"Warning: Non-numeric data found in {instrument} {timeframe}. Regenerating.")
                        os.remove(cache_file)  # Remove bad cache
                        raise ValueError("Invalid data in cache")
                except Exception as e:
                    print(f"Error loading cached data for {instrument} {timeframe}: {e}")
                    print(f"Fetching new data...")
                    # We'll continue to the fetch section
            if not os.path.exists(cache_file) or (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days >= 1:
                # Fetch new data
                print(f"Fetching {instrument} {timeframe} data from Yahoo Finance...")
                try:
                    # Get data from Yahoo Finance
                    data = yf.download(
                        symbol,
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        progress=False
                    )
                    
                    # Check if data is empty
                    if data.empty:
                        print(f"Warning: No data returned for {instrument} {timeframe}. Using sample data.")
                        # Create sample data for this timeframe
                        data = _create_sample_data_for_instrument(instrument, timeframe, 
                                                               start_date, end_date)
                    else:
                        # Save to cache
                        data.to_csv(cache_file)
                        
                except Exception as e:
                    print(f"Error fetching {instrument} {timeframe} data: {e}")
                    print(f"Using sample data for {instrument} {timeframe}.")
                    # Create sample data for this timeframe
                    data = _create_sample_data_for_instrument(instrument, timeframe, 
                                                           start_date, end_date)
            
            # Make sure we have a DataFrame, not an empty result or tuple
            if not isinstance(data, pd.DataFrame) or data.empty:
                print(f"No data returned for {instrument} {timeframe}. Using sample data.")
                data = _create_sample_data_for_instrument(instrument, timeframe, start_date, end_date)
            else:
                # Fix column names to lowercase
                try:
                    data.columns = [col.lower() if isinstance(col, str) else col for col in data.columns]
                
                    # Make sure we have all required columns
                    if not all(col in map(str.lower, data.columns) for col in ['open', 'high', 'low', 'close']):
                        # Try to map common column names
                        col_map = {
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume',
                            'Adj Close': 'adj_close'
                        }
                        data = data.rename(columns=col_map)
                except Exception as e:
                    print(f"Error processing column names for {instrument} {timeframe}: {e}")
                    print(f"Using sample data for {instrument} {timeframe}.")
                    data = _create_sample_data_for_instrument(instrument, timeframe, start_date, end_date)
            
            # Add volume if missing
            if 'volume' not in data.columns:
                data['volume'] = np.random.uniform(100, 1000, len(data))
                
            # Store in market_data
            market_data[instrument][timeframe] = data
            
            # Store daily data for correlation calculation
            if timeframe == 'D':
                correlation_data[instrument] = data
    
    # Add correlation data to market data
    market_data['correlation_data'] = correlation_data
    
    return market_data

def _create_sample_data_for_instrument(instrument, timeframe, start_date, end_date):
    """Create sample data for an instrument and timeframe if real data is unavailable"""
    # Determine number of bars based on timeframe
    if timeframe == '1H':
        freq = 'h'  # Use lowercase 'h' to avoid FutureWarning
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    elif timeframe == '4H':
        freq = '4h'  # Use lowercase 'h' to avoid FutureWarning
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    elif timeframe == 'D':
        freq = 'D'
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    elif timeframe == 'W':
        # Weekly data - end on Friday
        freq = 'W-FRI'
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    elif timeframe == 'M':
        # Monthly data - end on last day of month
        freq = 'ME'  # Month end
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    else:
        freq = 'D'
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
    # Set base price based on instrument
    if instrument == 'EURUSD':
        base_price = 1.1
    elif instrument == 'GBPUSD':
        base_price = 1.3
    elif instrument == 'USDJPY':
        base_price = 110.0
    elif instrument == 'AUDUSD':
        base_price = 0.7
    else:
        base_price = 1.0
        
    # Generate random walk prices
    np.random.seed(int(time.time()))  # Use current time as seed
    
    # Parameters for random walk
    volatility = np.random.uniform(0.0005, 0.002)  # Base volatility
    trend = np.random.uniform(-0.0002, 0.0002)     # Trend component
    
    # Adjust parameters based on timeframe
    if timeframe == '1H':
        tf_volatility = volatility
        tf_trend = trend
    elif timeframe == '4H':
        tf_volatility = volatility * 1.5
        tf_trend = trend * 3
    elif timeframe == 'D':
        tf_volatility = volatility * 2.5
        tf_trend = trend * 5
    else:
        tf_volatility = volatility
        tf_trend = trend
    
    # Generate price data using random walk with drift
    close = [base_price]
    
    for i in range(1, len(date_range)):
        # Add trend and random component
        new_price = close[i-1] * (1 + tf_trend + np.random.normal(0, tf_volatility))
        close.append(new_price)
    
    # Generate OHLC data
    high = []
    low = []
    open_prices = []
    volume = []
    
    for i in range(len(date_range)):
        if i == 0:
            open_price = close[i] * (1 - np.random.uniform(0, tf_volatility))
        else:
            open_price = close[i-1]
        
        high_price = max(open_price, close[i]) * (1 + np.random.uniform(0, tf_volatility))
        low_price = min(open_price, close[i]) * (1 - np.random.uniform(0, tf_volatility))
        
        open_prices.append(open_price)
        high.append(high_price)
        low.append(low_price)
        volume.append(np.random.uniform(100, 1000))
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=date_range)
    
    return data

def save_market_data_info(market_data, filename='market_data_info.json'):
    """
    Save summary information about the market data
    
    Args:
        market_data: Dictionary of market data
        filename: Output filename
        
    Returns:
        Dictionary with market data summary
    """
    info = {
        "instruments": [],
        "timeframes": [],
        "date_range": {},
        "bars_count": {},
        "price_range": {}
    }
    
    # Collect instruments and timeframes
    for instrument in market_data:
        if instrument == 'correlation_data':
            continue
            
        info["instruments"].append(instrument)
        
        info["date_range"][instrument] = {}
        info["bars_count"][instrument] = {}
        info["price_range"][instrument] = {}
        
        for timeframe in market_data[instrument]:
            if timeframe not in info["timeframes"]:
                info["timeframes"].append(timeframe)
                
            data = market_data[instrument][timeframe]
            
            try:
                # Date range - handle different types of indices
                if isinstance(data.index, pd.DatetimeIndex):
                    start_date = data.index.min().strftime('%Y-%m-%d')
                    end_date = data.index.max().strftime('%Y-%m-%d')
                else:
                    # If index is not datetime, use string representation
                    start_date = str(data.index.min())
                    end_date = str(data.index.max())
                    
                info["date_range"][instrument][timeframe] = {
                    "start": start_date,
                    "end": end_date
                }
                
                # Bars count
                info["bars_count"][instrument][timeframe] = len(data)
                
                # Price range
                info["price_range"][instrument][timeframe] = {
                    "min": float(data['low'].min()),
                    "max": float(data['high'].max()),
                    "avg": float(data['close'].mean())
                }
            except Exception as e:
                print(f"Error processing data info for {instrument} {timeframe}: {e}")
                info["date_range"][instrument][timeframe] = {
                    "start": "unknown",
                    "end": "unknown"
                }
                info["bars_count"][instrument][timeframe] = len(data)
                info["price_range"][instrument][timeframe] = {
                    "min": 0.0,
                    "max": 0.0,
                    "avg": 0.0
                }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(info, f, indent=2)
        
    return info