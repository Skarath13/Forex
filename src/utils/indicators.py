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
    
def detect_candlestick_patterns(data, min_trend_bars=5, volatility_filter=True, volume_filter=True, 
                             min_pattern_size_pct=0.01, confirmation_required=True):
    """
    Detect common candlestick patterns in price data with advanced filtering
    
    Args:
        data: DataFrame with OHLC price data
        min_trend_bars: Minimum number of bars to confirm a trend before pattern
        volatility_filter: Whether to apply volatility-based filtering
        volume_filter: Whether to apply volume-based filtering
        min_pattern_size_pct: Minimum pattern size as percentage of ATR (for relevance)
        confirmation_required: Whether to require confirmation for patterns
        
    Returns:
        Dictionary with detected patterns and their locations including metadata
    """
    # Ensure we have required data
    if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
        return {'bullish': [], 'bearish': []}
    
    # Initialize results
    patterns = {
        'bullish': [],
        'bearish': []
    }
    
    # Get number of candles
    n = len(data)
    
    # Skip if we don't have enough data
    if n < max(min_trend_bars + 3, 10):
        return patterns
    
    # Extract price data
    open_prices = data['open']
    high_prices = data['high']
    low_prices = data['low']
    close_prices = data['close']
    
    # Calculate candle properties
    candle_body = abs(close_prices - open_prices)
    candle_range = high_prices - low_prices
    body_size_pct = candle_body / candle_range  # Relative body size
    
    # True if bullish candle (close > open)
    is_bullish = close_prices > open_prices
    # True if bearish candle (close < open)
    is_bearish = close_prices < open_prices
    
    # Calculate upper and lower shadows
    upper_shadow = high_prices - np.maximum(close_prices, open_prices)
    lower_shadow = np.minimum(close_prices, open_prices) - low_prices
    
    # Calculate ATR for volatility filter
    def calculate_atr(periods=14):
        tr1 = high_prices - low_prices
        tr2 = abs(high_prices - close_prices.shift(1))
        tr3 = abs(low_prices - close_prices.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=periods).mean()
    
    atr = calculate_atr()
    
    # Calculate trend metrics for better pattern context
    def is_in_uptrend(i, bars=min_trend_bars):
        if i < bars:
            return False
        
        # Simple moving average slope
        window = min(bars, i)
        sma = close_prices.iloc[i-window:i].mean()
        return close_prices.iloc[i] > sma and close_prices.iloc[i-window:i].is_monotonic_increasing
        
    def is_in_downtrend(i, bars=min_trend_bars):
        if i < bars:
            return False
        
        # Simple moving average slope
        window = min(bars, i)
        sma = close_prices.iloc[i-window:i].mean()
        return close_prices.iloc[i] < sma and close_prices.iloc[i-window:i].is_monotonic_decreasing
    
    # Calculate additional technical indicators for confirmation
    if confirmation_required:
        # Simple RSI implementation
        def calculate_rsi(periods=14):
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        rsi = calculate_rsi()
        
        # Simple moving averages
        sma20 = close_prices.rolling(window=20).mean()
        sma50 = close_prices.rolling(window=50).mean()
    
    # Function to check if pattern is confirmed by the next candle
    def is_pattern_confirmed(index, pattern_type):
        """Check if pattern is confirmed by subsequent price action"""
        if index + 1 >= n:  # Can't confirm patterns at the end of the data
            return False
            
        # Simple confirmation rule: next candle follows the expected direction
        if pattern_type == 'bullish':
            return is_bullish.iloc[index + 1] and close_prices.iloc[index + 1] > close_prices.iloc[index]
        elif pattern_type == 'bearish':
            return is_bearish.iloc[index + 1] and close_prices.iloc[index + 1] < close_prices.iloc[index]
        return False
    
    # Function to add pattern with metadata
    def add_pattern(direction, index, pattern_name, base_strength, trend_quality=0.5):
        # Skip patterns that are too small relative to volatility
        if volatility_filter and candle_range.iloc[index] < atr.iloc[index] * min_pattern_size_pct:
            return
            
        # Apply volume filter if enabled and volume data exists
        if volume_filter and 'volume' in data.columns:
            avg_volume = data['volume'].iloc[max(0, index-5):index].mean()
            if data['volume'].iloc[index] < avg_volume * 0.8:
                return
        
        # Calculate final pattern strength based on multiple factors
        strength = base_strength
        
        # Adjust strength based on trend quality
        strength *= trend_quality
        
        # Adjust strength based on pattern size relative to volatility
        if volatility_filter and not np.isnan(atr.iloc[index]) and atr.iloc[index] > 0:
            volatility_factor = min(1.5, candle_range.iloc[index] / atr.iloc[index])
            strength *= volatility_factor
            
        # Adjust strength based on confirmation if required
        if confirmation_required:
            if index < n - 1:  # Not the last candle
                # Check next candle confirmation
                if is_pattern_confirmed(index, direction):
                    strength *= 1.2  # Boost strength for confirmed patterns
                else:
                    strength *= 0.5  # Reduce strength for unconfirmed patterns
                    
                # Check technical indicator alignment
                if not np.isnan(rsi.iloc[index]):
                    if direction == 'bullish' and rsi.iloc[index] < 30:
                        strength *= 1.2  # Bullish pattern in oversold condition
                    elif direction == 'bearish' and rsi.iloc[index] > 70:
                        strength *= 1.2  # Bearish pattern in overbought condition
                        
        # Cap strength at 1.0 and ensure it's at least 0.1
        strength = max(0.1, min(1.0, strength))
        
        # Add pattern with full metadata
        pattern_info = {
            'index': index,
            'pattern': pattern_name,
            'strength': strength,
            'price': close_prices.iloc[index],
            'volume': data['volume'].iloc[index] if 'volume' in data.columns else None,
            'time': data.index[index] if isinstance(data.index, pd.DatetimeIndex) else index,
            'confirmed': is_pattern_confirmed(index, direction) if index < n - 1 else None
        }
        
        patterns[direction].append(pattern_info)
    
    # Detect Hammer (bullish reversal pattern)
    for i in range(min_trend_bars, n):
        # Check for downtrend before pattern
        if is_in_downtrend(i):
            # Check for small body and long lower shadow
            if (is_bullish.iloc[i] and
                body_size_pct.iloc[i] < 0.3 and  # Small body
                lower_shadow.iloc[i] > 2 * candle_body.iloc[i] and  # Long lower shadow
                upper_shadow.iloc[i] < 0.3 * candle_body.iloc[i]):  # Very small upper shadow
                
                add_pattern('bullish', i, 'hammer', 0.7, trend_quality=0.8)
    
    # Detect Shooting Star (bearish reversal pattern)
    for i in range(min_trend_bars, n):
        # Check for uptrend before pattern
        if is_in_uptrend(i):
            # Check for small body and long upper shadow
            if (is_bearish.iloc[i] and
                body_size_pct.iloc[i] < 0.3 and  # Small body
                upper_shadow.iloc[i] > 2 * candle_body.iloc[i] and  # Long upper shadow
                lower_shadow.iloc[i] < 0.3 * candle_body.iloc[i]):  # Very small lower shadow
                
                add_pattern('bearish', i, 'shooting_star', 0.7, trend_quality=0.8)
    
    # Detect Engulfing Patterns
    for i in range(1, n):
        # Bullish Engulfing
        if (is_bearish.iloc[i-1] and is_bullish.iloc[i] and  # Previous bearish, current bullish
            open_prices.iloc[i] <= close_prices.iloc[i-1] and  # Current open below/at previous close
            close_prices.iloc[i] >= open_prices.iloc[i-1]):  # Current close above/at previous open
            
            # Calculate trend quality (how strong was the preceding downtrend)
            trend_quality = 0.5
            if i >= min_trend_bars:
                # Check how many of the previous bars were bearish
                bearish_count = sum(is_bearish.iloc[max(0, i-min_trend_bars):i])
                trend_quality = min(1.0, bearish_count / min_trend_bars + 0.3)
            
            add_pattern('bullish', i, 'bullish_engulfing', 0.8, trend_quality)
            
        # Bearish Engulfing
        if (is_bullish.iloc[i-1] and is_bearish.iloc[i] and  # Previous bullish, current bearish
            open_prices.iloc[i] >= close_prices.iloc[i-1] and  # Current open above/at previous close
            close_prices.iloc[i] <= open_prices.iloc[i-1]):  # Current close below/at previous open
            
            # Calculate trend quality (how strong was the preceding uptrend)
            trend_quality = 0.5
            if i >= min_trend_bars:
                # Check how many of the previous bars were bullish
                bullish_count = sum(is_bullish.iloc[max(0, i-min_trend_bars):i])
                trend_quality = min(1.0, bullish_count / min_trend_bars + 0.3)
            
            add_pattern('bearish', i, 'bearish_engulfing', 0.8, trend_quality)
    
    # Detect Doji (indecision pattern)
    for i in range(n):
        # Real doji has almost no body
        body_thresh = 0.05  # Body must be less than 5% of the range
        
        # Body is very small relative to shadows
        if (candle_body.iloc[i] < body_thresh * candle_range.iloc[i] and
            candle_range.iloc[i] > 0):  # Ensure non-zero range
            
            # Long-legged Doji (significant shadows on both sides)
            if (upper_shadow.iloc[i] > 0.3 * candle_range.iloc[i] and
                lower_shadow.iloc[i] > 0.3 * candle_range.iloc[i]):
                
                # In downtrend, could be bullish reversal
                if i >= min_trend_bars and is_in_downtrend(i):
                    add_pattern('bullish', i, 'doji', 0.5, trend_quality=0.7)
                # In uptrend, could be bearish reversal    
                elif i >= min_trend_bars and is_in_uptrend(i):
                    add_pattern('bearish', i, 'doji', 0.5, trend_quality=0.7)
    
    # Detect Morning Star (bullish reversal)
    for i in range(2, n):
        if (is_bearish.iloc[i-2] and  # First candle is bearish with significant body
            candle_body.iloc[i-2] > atr.iloc[i-2] * 0.5 and  # First candle has significant body
            candle_body.iloc[i-1] < 0.3 * candle_body.iloc[i-2] and  # Second candle has small body
            is_bullish.iloc[i] and  # Third candle is bullish
            candle_body.iloc[i] > atr.iloc[i] * 0.5 and  # Third candle has significant body
            # Gap or near-gap between 1st and 2nd candles
            max(open_prices.iloc[i-1], close_prices.iloc[i-1]) < close_prices.iloc[i-2] and
            # Third candle closes well into first candle's body
            close_prices.iloc[i] > (open_prices.iloc[i-2] + close_prices.iloc[i-2]) / 2):
            
            # Check for downtrend before the pattern
            trend_quality = 0.5
            if i >= min_trend_bars + 2:
                # Check preceding bars for downtrend
                if is_in_downtrend(i-2):
                    trend_quality = 0.9
            
            add_pattern('bullish', i, 'morning_star', 0.9, trend_quality)
    
    # Detect Evening Star (bearish reversal)
    for i in range(2, n):
        if (is_bullish.iloc[i-2] and  # First candle is bullish with significant body
            candle_body.iloc[i-2] > atr.iloc[i-2] * 0.5 and  # First candle has significant body
            candle_body.iloc[i-1] < 0.3 * candle_body.iloc[i-2] and  # Second candle has small body
            is_bearish.iloc[i] and  # Third candle is bearish
            candle_body.iloc[i] > atr.iloc[i] * 0.5 and  # Third candle has significant body
            # Gap or near-gap between 1st and 2nd candles
            min(open_prices.iloc[i-1], close_prices.iloc[i-1]) > close_prices.iloc[i-2] and
            # Third candle closes well into first candle's body
            close_prices.iloc[i] < (open_prices.iloc[i-2] + close_prices.iloc[i-2]) / 2):
            
            # Check for uptrend before the pattern
            trend_quality = 0.5
            if i >= min_trend_bars + 2:
                # Check preceding bars for uptrend
                if is_in_uptrend(i-2):
                    trend_quality = 0.9
            
            add_pattern('bearish', i, 'evening_star', 0.9, trend_quality)
    
    # Detect Hanging Man (bearish reversal)
    for i in range(min_trend_bars, n):
        # Check for uptrend
        if is_in_uptrend(i):
            # Look for small body and long lower shadow
            if (body_size_pct.iloc[i] < 0.3 and  # Small body
                lower_shadow.iloc[i] > 2 * candle_body.iloc[i] and  # Long lower shadow
                upper_shadow.iloc[i] < 0.3 * candle_body.iloc[i]):  # Very small upper shadow
                
                add_pattern('bearish', i, 'hanging_man', 0.7, trend_quality=0.8)
    
    # Detect Three White Soldiers (bullish continuation)
    for i in range(2, n):
        if (is_bullish.iloc[i-2] and is_bullish.iloc[i-1] and is_bullish.iloc[i] and  # Three bullish candles
            close_prices.iloc[i] > close_prices.iloc[i-1] > close_prices.iloc[i-2] and  # Each close higher
            open_prices.iloc[i] > open_prices.iloc[i-1] > open_prices.iloc[i-2] and  # Each open higher
            upper_shadow.iloc[i-2] < 0.2 * candle_body.iloc[i-2] and  # Small upper shadows
            upper_shadow.iloc[i-1] < 0.2 * candle_body.iloc[i-1] and
            upper_shadow.iloc[i] < 0.2 * candle_body.iloc[i] and
            # Each body should be reasonable size
            candle_body.iloc[i-2] > atr.iloc[i-2] * 0.5 and
            candle_body.iloc[i-1] > atr.iloc[i-1] * 0.5 and
            candle_body.iloc[i] > atr.iloc[i] * 0.5):
            
            # Check for prior consolidation or base pattern for better quality signal
            trend_quality = 0.7  # Default quality
            if i >= min_trend_bars + 2:
                # Look for a base or consolidation pattern before the three soldiers
                prior_range = high_prices.iloc[i-min_trend_bars:i-2].max() - low_prices.iloc[i-min_trend_bars:i-2].min()
                current_range = close_prices.iloc[i] - close_prices.iloc[i-2]
                
                # If the soldiers are breaking out of a base pattern, higher quality
                if current_range > prior_range * 0.5:
                    trend_quality = 0.9
            
            add_pattern('bullish', i, 'three_white_soldiers', 0.9, trend_quality)
    
    # Detect Three Black Crows (bearish continuation)
    for i in range(2, n):
        if (is_bearish.iloc[i-2] and is_bearish.iloc[i-1] and is_bearish.iloc[i] and  # Three bearish candles
            close_prices.iloc[i] < close_prices.iloc[i-1] < close_prices.iloc[i-2] and  # Each close lower
            open_prices.iloc[i] < open_prices.iloc[i-1] < open_prices.iloc[i-2] and  # Each open lower
            lower_shadow.iloc[i-2] < 0.2 * candle_body.iloc[i-2] and  # Small lower shadows
            lower_shadow.iloc[i-1] < 0.2 * candle_body.iloc[i-1] and
            lower_shadow.iloc[i] < 0.2 * candle_body.iloc[i] and
            # Each body should be reasonable size
            candle_body.iloc[i-2] > atr.iloc[i-2] * 0.5 and
            candle_body.iloc[i-1] > atr.iloc[i-1] * 0.5 and
            candle_body.iloc[i] > atr.iloc[i] * 0.5):
            
            # Check for prior consolidation or top pattern for better quality signal
            trend_quality = 0.7  # Default quality
            if i >= min_trend_bars + 2:
                # Look for a top pattern before the three crows
                prior_range = high_prices.iloc[i-min_trend_bars:i-2].max() - low_prices.iloc[i-min_trend_bars:i-2].min()
                current_range = close_prices.iloc[i-2] - close_prices.iloc[i]
                
                # If the crows are breaking down from a top pattern, higher quality
                if current_range > prior_range * 0.5:
                    trend_quality = 0.9
            
            add_pattern('bearish', i, 'three_black_crows', 0.9, trend_quality)
    
    # Detect Piercing Line (bullish reversal)
    for i in range(1, n):
        if (is_bearish.iloc[i-1] and is_bullish.iloc[i] and  # Previous bearish, current bullish
            open_prices.iloc[i] < close_prices.iloc[i-1] and  # Open below previous close
            close_prices.iloc[i] > (open_prices.iloc[i-1] + close_prices.iloc[i-1]) / 2 and  # Close above midpoint
            candle_body.iloc[i-1] > atr.iloc[i-1] * 0.5 and  # Previous candle has significant body
            candle_body.iloc[i] > atr.iloc[i] * 0.5):  # Current candle has significant body
            
            # Check for downtrend before the pattern
            trend_quality = 0.5
            if i >= min_trend_bars + 1:
                if is_in_downtrend(i-1):
                    trend_quality = 0.8
            
            add_pattern('bullish', i, 'piercing_line', 0.7, trend_quality)
    
    # Detect Dark Cloud Cover (bearish reversal)
    for i in range(1, n):
        if (is_bullish.iloc[i-1] and is_bearish.iloc[i] and  # Previous bullish, current bearish
            open_prices.iloc[i] > close_prices.iloc[i-1] and  # Open above previous close
            close_prices.iloc[i] < (open_prices.iloc[i-1] + close_prices.iloc[i-1]) / 2 and  # Close below midpoint
            candle_body.iloc[i-1] > atr.iloc[i-1] * 0.5 and  # Previous candle has significant body
            candle_body.iloc[i] > atr.iloc[i] * 0.5):  # Current candle has significant body
            
            # Check for uptrend before the pattern
            trend_quality = 0.5
            if i >= min_trend_bars + 1:
                if is_in_uptrend(i-1):
                    trend_quality = 0.8
            
            add_pattern('bearish', i, 'dark_cloud_cover', 0.7, trend_quality)
    
    # Detect harami patterns (reversal patterns)
    for i in range(1, n):
        # Bullish Harami
        if (is_bearish.iloc[i-1] and  # Previous bearish
            is_bullish.iloc[i] and  # Current bullish
            candle_body.iloc[i-1] > candle_body.iloc[i] * 2 and  # Previous body engulfs current
            open_prices.iloc[i] > close_prices.iloc[i-1] and  # Current open above previous close
            close_prices.iloc[i] < open_prices.iloc[i-1] and  # Current close below previous open
            candle_body.iloc[i-1] > atr.iloc[i-1] * 0.5):  # Previous candle has significant body
            
            # Check for downtrend before the pattern
            trend_quality = 0.5
            if i >= min_trend_bars + 1:
                if is_in_downtrend(i-1):
                    trend_quality = 0.8
            
            add_pattern('bullish', i, 'bullish_harami', 0.6, trend_quality)
            
        # Bearish Harami
        if (is_bullish.iloc[i-1] and  # Previous bullish
            is_bearish.iloc[i] and  # Current bearish
            candle_body.iloc[i-1] > candle_body.iloc[i] * 2 and  # Previous body engulfs current
            open_prices.iloc[i] < close_prices.iloc[i-1] and  # Current open below previous close
            close_prices.iloc[i] > open_prices.iloc[i-1] and  # Current close above previous open
            candle_body.iloc[i-1] > atr.iloc[i-1] * 0.5):  # Previous candle has significant body
            
            # Check for uptrend before the pattern
            trend_quality = 0.5
            if i >= min_trend_bars + 1:
                if is_in_uptrend(i-1):
                    trend_quality = 0.8
            
            add_pattern('bearish', i, 'bearish_harami', 0.6, trend_quality)
    
    # Optionally filter out very weak patterns or reduce to top N strongest patterns
    # This helps eliminate patterns that are technically valid but not tradable
    min_strength_threshold = 0.3
    
    # Filter out weak patterns
    patterns['bullish'] = [p for p in patterns['bullish'] if p['strength'] >= min_strength_threshold]
    patterns['bearish'] = [p for p in patterns['bearish'] if p['strength'] >= min_strength_threshold]
    
    # Sort patterns by strength (descending)
    patterns['bullish'] = sorted(patterns['bullish'], key=lambda x: x['strength'], reverse=True)
    patterns['bearish'] = sorted(patterns['bearish'], key=lambda x: x['strength'], reverse=True)
    
    return patterns

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