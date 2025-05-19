import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_forex_data(instrument, start_date, end_date, timeframe='1H', volatility=None, trend=None, with_gaps=True):
    """
    Generate synthetic forex data for backtesting
    
    Args:
        instrument: Trading instrument symbol (e.g., 'EURUSD')
        start_date: Start date for data generation
        end_date: End date for data generation
        timeframe: Timeframe for the data ('1H', '4H', 'D', etc.)
        volatility: Custom volatility parameter (None for auto)
        trend: Custom trend parameter (None for random)
        with_gaps: Whether to include gaps to simulate weekend/holiday closures
        
    Returns:
        DataFrame with OHLCV data
    """
    # Parse dates
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Set default parameters based on instrument
    if volatility is None:
        # Set default volatilities based on typical instrument behavior
        volatilities = {
            'EURUSD': 0.0007,
            'GBPUSD': 0.0009,
            'USDJPY': 0.0006,
            'AUDUSD': 0.0008,
            'USDCAD': 0.0007,
            'USDCHF': 0.0006,
            'NZDUSD': 0.0008,
            'EURGBP': 0.0005,
            'EURJPY': 0.0008,
            'GBPJPY': 0.0010
        }
        volatility = volatilities.get(instrument, 0.0007)  # Default if not found
    
    if trend is None:
        # Random trend between -0.0001 and 0.0001 per bar
        trend = random.uniform(-0.0001, 0.0001)
    
    # Generate time series based on timeframe
    if timeframe == '1H':
        freq = 'H'
    elif timeframe == '4H':
        freq = '4H'
    elif timeframe == 'D':
        freq = 'D'
    elif timeframe == '15M':
        freq = '15min'
    else:
        freq = 'H'  # Default to hourly
    
    # Create date range
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Remove weekends if with_gaps is True
    if with_gaps:
        timestamps = timestamps[timestamps.dayofweek < 5]  # 0-4: Monday to Friday
    
    # Starting price - pick a realistic value based on instrument
    starting_prices = {
        'EURUSD': 1.10,
        'GBPUSD': 1.30,
        'USDJPY': 110.0,
        'AUDUSD': 0.70,
        'USDCAD': 1.30,
        'USDCHF': 0.95,
        'NZDUSD': 0.65,
        'EURGBP': 0.85,
        'EURJPY': 130.0,
        'GBPJPY': 150.0
    }
    starting_price = starting_prices.get(instrument, 1.0)
    
    # Generate price data
    num_bars = len(timestamps)
    prices = [starting_price]
    
    # Add session-specific volatility adjustments
    def get_session_factor(timestamp):
        hour = timestamp.hour
        # Asian session: lower volatility
        if 0 <= hour < 8:
            return 0.7
        # London session: higher volatility
        elif 8 <= hour < 16:
            return 1.2
        # New York session: higher volatility
        elif 12 <= hour < 21:  # Note overlap with London
            return 1.3
        # Quieter period
        else:
            return 0.8
    
    # Simulate price using geometric Brownian motion with session adjustments
    for i in range(1, num_bars):
        session_factor = get_session_factor(timestamps[i])
        daily_vol = volatility * session_factor
        
        # Add some mean reverting behavior and random shocks
        if random.random() < 0.03:  # 3% chance of a shock
            shock = random.choice([-1, 1]) * volatility * random.uniform(3, 5)
        else:
            shock = 0
            
        # Mean reversion component
        if len(prices) > 20:
            avg_20 = sum(prices[-20:]) / 20
            mean_reversion = (avg_20 - prices[-1]) * 0.05
        else:
            mean_reversion = 0
        
        # Price change with drift (trend), volatility, mean reversion and shocks
        price_change = (trend + 
                       np.random.normal(0, daily_vol) + 
                       mean_reversion + 
                       shock)
        
        # Apply change 
        new_price = prices[-1] * (1 + price_change)
        
        # Ensure price stays positive and somewhat realistic
        new_price = max(new_price, starting_price * 0.5)
        new_price = min(new_price, starting_price * 2.0)
        
        prices.append(new_price)
    
    # Generate OHLC data
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    for i in range(num_bars):
        if i == 0:
            opens.append(prices[i])
        else:
            opens.append(closes[i-1])
            
        close = prices[i]
        closes.append(close)
        
        # High and low based on volatility
        day_range = volatility * prices[i] * random.uniform(1, 3)
        highs.append(max(opens[i], closes[i]) + day_range * random.uniform(0.2, 1.0))
        lows.append(min(opens[i], closes[i]) - day_range * random.uniform(0.2, 1.0))
        
        # Volume - higher during trend moves and session overlaps
        base_vol = random.uniform(80, 120)
        if i > 0:
            price_change = abs(closes[i] - closes[i-1]) / closes[i-1]
            # More volume on bigger price moves
            vol_factor = 1 + price_change / volatility * 5
        else:
            vol_factor = 1
            
        # Session volume factor
        session_vol = get_session_factor(timestamps[i])
        
        # Final volume
        volumes.append(int(base_vol * vol_factor * session_vol))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=timestamps)
    
    return df

def generate_multi_timeframe_data(instrument, start_date, end_date, timeframes=['1H', '4H', 'D']):
    """
    Generate data for multiple timeframes
    
    Args:
        instrument: Trading instrument symbol
        start_date: Start date for data generation
        end_date: End date for data generation
        timeframes: List of timeframes to generate
        
    Returns:
        Dictionary with data for each timeframe
    """
    result = {}
    
    # Generate the smallest timeframe first
    smallest_tf = min(timeframes, key=lambda x: _get_timeframe_minutes(x))
    
    # For accurate resampling, start earlier
    buffer_days = 10
    adjusted_start = pd.to_datetime(start_date) - timedelta(days=buffer_days)
    
    # Generate base data
    base_data = generate_forex_data(
        instrument, 
        adjusted_start, 
        end_date, 
        timeframe=smallest_tf
    )
    
    # Store the smallest timeframe
    result[smallest_tf] = base_data.loc[pd.to_datetime(start_date):]
    
    # Resample for other timeframes
    for tf in timeframes:
        if tf == smallest_tf:
            continue
            
        if tf == 'D':
            freq = 'D'
        elif tf.endswith('H'):
            freq = f"{tf[:-1]}H"
        elif tf.endswith('M'):
            freq = f"{tf[:-1]}min"
        else:
            continue  # Skip unknown timeframes
        
        # Resample
        resampled = base_data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Store resampled data
        result[tf] = resampled.loc[pd.to_datetime(start_date):]
    
    return result

def _get_timeframe_minutes(timeframe):
    """Convert timeframe to minutes for comparison"""
    if timeframe.endswith('M'):
        return int(timeframe[:-1])
    elif timeframe.endswith('H'):
        return int(timeframe[:-1]) * 60
    elif timeframe == 'D':
        return 1440  # 24 hours
    return 60  # Default to 1H

def generate_test_market_data(instruments=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'], 
                             timeframes=['1H', '4H', 'D'],
                             days=100):
    """
    Generate test market data for multiple instruments and timeframes
    
    Args:
        instruments: List of instruments
        timeframes: List of timeframes
        days: Number of days of data
        
    Returns:
        Dictionary of market data for backtesting
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    market_data = {}
    correlation_data = {}
    
    for instrument in instruments:
        instrument_data = generate_multi_timeframe_data(
            instrument, 
            start_date, 
            end_date, 
            timeframes
        )
        market_data[instrument] = instrument_data
        
        # Store daily data for correlation calculation
        if 'D' in timeframes:
            correlation_data[instrument] = instrument_data['D']
        else:
            # Resample to daily if daily not provided
            correlation_data[instrument] = instrument_data[timeframes[0]].resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
    
    # Add correlation data to market_data
    market_data['correlation_data'] = correlation_data
    
    return market_data