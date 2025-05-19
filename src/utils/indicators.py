import numpy as np
import pandas as pd

def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index
    
    Args:
        data: DataFrame or Series with price data
        period: RSI calculation period
        
    Returns:
        Series containing RSI values
    """
    # Convert to pandas Series if DataFrame is provided
    if isinstance(data, pd.DataFrame):
        if 'close' in data.columns:
            close = data['close']
        else:
            close = data.iloc[:, 0]  # Use first column
    else:
        close = data
        
    # Calculate price changes
    delta = close.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over period
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence
    
    Args:
        data: DataFrame or Series with price data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        Dictionary with MACD, signal line, and histogram
    """
    # Convert to pandas Series if DataFrame is provided
    if isinstance(data, pd.DataFrame):
        if 'close' in data.columns:
            close = data['close']
        else:
            close = data.iloc[:, 0]  # Use first column
    else:
        close = data
        
    # Calculate EMAs
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd = fast_ema - slow_ema
    
    # Calculate signal line
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd - signal
    
    return {
        'macd': macd,
        'signal': signal,
        'histogram': histogram
    }

def calculate_bollinger_bands(data, period=20, std_dev=2):
    """
    Calculate Bollinger Bands
    
    Args:
        data: DataFrame or Series with price data
        period: Moving average period
        std_dev: Number of standard deviations
        
    Returns:
        Dictionary with upper, middle, and lower bands
    """
    # Convert to pandas Series if DataFrame is provided
    if isinstance(data, pd.DataFrame):
        if 'close' in data.columns:
            close = data['close']
        else:
            close = data.iloc[:, 0]  # Use first column
    else:
        close = data
        
    # Calculate middle band (SMA)
    middle_band = close.rolling(window=period).mean()
    
    # Calculate standard deviation
    std = close.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)
    
    # Calculate bandwidth
    bandwidth = (upper_band - lower_band) / middle_band
    
    return {
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band,
        'width': bandwidth
    }

def calculate_atr(data, period=14):
    """
    Calculate Average True Range
    
    Args:
        data: DataFrame with high, low, close price data
        period: ATR calculation period
        
    Returns:
        Series containing ATR values
    """
    # Ensure data has high, low, close columns
    if not all(col in data.columns for col in ['high', 'low', 'close']):
        raise ValueError("Data must contain 'high', 'low', and 'close' columns")
    
    # Calculate true range
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    
    # Get maximum of the three
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_stochastic(data, k_period=14, d_period=3):
    """
    Calculate Stochastic Oscillator
    
    Args:
        data: DataFrame with high, low, close price data
        k_period: %K period
        d_period: %D period
        
    Returns:
        Dictionary with %K and %D values
    """
    # Ensure data has high, low, close columns
    if not all(col in data.columns for col in ['high', 'low', 'close']):
        raise ValueError("Data must contain 'high', 'low', and 'close' columns")
    
    # Calculate %K
    lowest_low = data['low'].rolling(window=k_period).min()
    highest_high = data['high'].rolling(window=k_period).max()
    
    k = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
    
    # Calculate %D
    d = k.rolling(window=d_period).mean()
    
    return {
        'k': k,
        'd': d
    }

def calculate_ichimoku(data):
    """
    Calculate Ichimoku Cloud
    
    Args:
        data: DataFrame with high, low, close price data
        
    Returns:
        Dictionary with Ichimoku components
    """
    # Ensure data has high, low columns
    if not all(col in data.columns for col in ['high', 'low']):
        raise ValueError("Data must contain 'high' and 'low' columns")
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_high = data['high'].rolling(window=9).max()
    tenkan_low = data['low'].rolling(window=9).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_high = data['high'].rolling(window=26).max()
    kijun_low = data['low'].rolling(window=26).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods
    senkou_high = data['high'].rolling(window=52).max()
    senkou_low = data['low'].rolling(window=52).min()
    senkou_span_b = ((senkou_high + senkou_low) / 2).shift(26)
    
    # Calculate Chikou Span (Lagging Span): Current close shifted -26 periods
    chikou_span = data['close'].shift(-26)
    
    # Determine if price is above the cloud
    cloud_bullish = False
    cloud_bearish = False
    
    # Check the current state if enough data is available
    if len(data) > 26:
        current_price = data['close'].iloc[-1]
        current_a = senkou_span_a.iloc[-1]
        current_b = senkou_span_b.iloc[-1]
        
        cloud_top = max(current_a, current_b)
        cloud_bottom = min(current_a, current_b)
        
        cloud_bullish = current_price > cloud_top
        cloud_bearish = current_price < cloud_bottom
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span,
        'cloud_bullish': cloud_bullish,
        'cloud_bearish': cloud_bearish
    }

def detect_divergence(data, rsi, macd, stoch_k):
    """
    Detect bullish and bearish divergences
    
    Args:
        data: DataFrame with price data
        rsi: Series with RSI values
        macd: Series with MACD values
        stoch_k: Series with Stochastic %K values
        
    Returns:
        Dictionary indicating detected divergences
    """
    # Convert to pandas Series if DataFrame is provided
    if isinstance(data, pd.DataFrame):
        if 'close' in data.columns:
            close = data['close']
        else:
            close = data.iloc[:, 0]  # Use first column
    else:
        close = data
    
    # Initialize result
    divergences = {
        'bullish': False,
        'bearish': False,
        'bullish_type': [],
        'bearish_type': []
    }
    
    # Need enough data for divergence detection
    if len(close) < 10:
        return divergences
    
    # Find local minima and maxima in price
    price_window = 5  # Window to look for local extrema
    
    price_min_indices = []
    price_max_indices = []
    
    for i in range(price_window, len(close) - price_window):
        if close.iloc[i] == close.iloc[i-price_window:i+price_window+1].min():
            price_min_indices.append(i)
        elif close.iloc[i] == close.iloc[i-price_window:i+price_window+1].max():
            price_max_indices.append(i)
    
    # Need at least two minima/maxima for divergence
    if len(price_min_indices) < 2 or len(price_max_indices) < 2:
        return divergences
    
    # Check the last two price minima for bullish divergence
    if len(price_min_indices) >= 2:
        last_min_idx = price_min_indices[-1]
        prev_min_idx = price_min_indices[-2]
        
        # Price made lower low
        if close.iloc[last_min_idx] < close.iloc[prev_min_idx]:
            # Check RSI for higher low (bullish divergence)
            if rsi.iloc[last_min_idx] > rsi.iloc[prev_min_idx]:
                divergences['bullish'] = True
                divergences['bullish_type'].append('rsi')
                
            # Check MACD for higher low
            if macd.iloc[last_min_idx] > macd.iloc[prev_min_idx]:
                divergences['bullish'] = True
                divergences['bullish_type'].append('macd')
                
            # Check Stochastic for higher low
            if stoch_k.iloc[last_min_idx] > stoch_k.iloc[prev_min_idx]:
                divergences['bullish'] = True
                divergences['bullish_type'].append('stochastic')
    
    # Check the last two price maxima for bearish divergence
    if len(price_max_indices) >= 2:
        last_max_idx = price_max_indices[-1]
        prev_max_idx = price_max_indices[-2]
        
        # Price made higher high
        if close.iloc[last_max_idx] > close.iloc[prev_max_idx]:
            # Check RSI for lower high (bearish divergence)
            if rsi.iloc[last_max_idx] < rsi.iloc[prev_max_idx]:
                divergences['bearish'] = True
                divergences['bearish_type'].append('rsi')
                
            # Check MACD for lower high
            if macd.iloc[last_max_idx] < macd.iloc[prev_max_idx]:
                divergences['bearish'] = True
                divergences['bearish_type'].append('macd')
                
            # Check Stochastic for lower high
            if stoch_k.iloc[last_max_idx] < stoch_k.iloc[prev_max_idx]:
                divergences['bearish'] = True
                divergences['bearish_type'].append('stochastic')
    
    return divergences

def calculate_support_resistance(market_data, timeframes):
    """
    Calculate support and resistance levels
    
    Args:
        market_data: Dictionary of market data for different timeframes
        timeframes: List of timeframes to analyze
        
    Returns:
        Dictionary with support and resistance levels
    """
    levels = {
        'supports': [],
        'resistances': []
    }
    
    for timeframe in timeframes:
        if timeframe not in market_data:
            continue
            
        data = market_data[timeframe]
        
        # Ensure data has high, low columns
        if not all(col in data.columns for col in ['high', 'low']):
            continue
        
        # Find pivot points
        pivot_len = 5  # Number of bars to look on each side
        
        for i in range(pivot_len, len(data) - pivot_len):
            # Check for pivot high
            if data['high'].iloc[i] == data['high'].iloc[i-pivot_len:i+pivot_len+1].max():
                levels['resistances'].append(data['high'].iloc[i])
                
            # Check for pivot low
            if data['low'].iloc[i] == data['low'].iloc[i-pivot_len:i+pivot_len+1].min():
                levels['supports'].append(data['low'].iloc[i])
    
    # Remove duplicates and sort
    levels['supports'] = sorted(list(set(levels['supports'])))
    levels['resistances'] = sorted(list(set(levels['resistances'])))
    
    return levels