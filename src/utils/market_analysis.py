import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from collections import defaultdict

def analyze_market_structure(data, window=20):
    """
    Analyze market structure for trend identification
    
    Args:
        data: DataFrame with OHLC price data
        window: Window for trend analysis
        
    Returns:
        Dictionary with market structure analysis
    """
    # Ensure we have enough data
    if len(data) < window:
        return {
            'trend_direction': 'undefined',
            'trend_strength': 0,
            'swing_high_levels': [],
            'swing_low_levels': [],
            'momentum': 0
        }
    
    # Extract close prices
    if 'close' in data.columns:
        close = data['close']
    else:
        close = data.iloc[:, 0]  # Use first column
    
    # Calculate linear regression to determine trend
    x = np.arange(window)
    y = close.iloc[-window:].values
    
    # Fit linear regression
    slope, intercept = np.polyfit(x, y, 1)
    
    # Calculate R-squared to determine trend strength
    y_pred = slope * x + intercept
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Determine trend direction
    if slope > 0:
        trend_direction = 'up'
    elif slope < 0:
        trend_direction = 'down'
    else:
        trend_direction = 'sideways'
    
    # Trend strength is R-squared value, modified by slope magnitude
    trend_strength = r_squared * min(1, abs(slope) * 100)
    
    # Find swing high and low levels
    swing_window = 5  # Window to look for local extrema
    
    swing_high_levels = []
    swing_low_levels = []
    
    for i in range(swing_window, len(close) - swing_window):
        # Check high if high data is available
        if 'high' in data.columns:
            if data['high'].iloc[i] == data['high'].iloc[i-swing_window:i+swing_window+1].max():
                swing_high_levels.append(data['high'].iloc[i])
        # Otherwise use close
        elif close.iloc[i] == close.iloc[i-swing_window:i+swing_window+1].max():
            swing_high_levels.append(close.iloc[i])
            
        # Check low if low data is available
        if 'low' in data.columns:
            if data['low'].iloc[i] == data['low'].iloc[i-swing_window:i+swing_window+1].min():
                swing_low_levels.append(data['low'].iloc[i])
        # Otherwise use close
        elif close.iloc[i] == close.iloc[i-swing_window:i+swing_window+1].min():
            swing_low_levels.append(close.iloc[i])
    
    # Calculate momentum (rate of change)
    momentum = (close.iloc[-1] / close.iloc[-window]) - 1
    
    return {
        'trend_direction': trend_direction,
        'trend_strength': trend_strength,
        'swing_high_levels': swing_high_levels,
        'swing_low_levels': swing_low_levels,
        'momentum': momentum
    }

def detect_market_regime(data, window=20):
    """
    Detect market regime (trending, ranging, volatile)
    
    Args:
        data: DataFrame with price data
        window: Analysis window
        
    Returns:
        String indicating the market regime
    """
    # Ensure we have enough data
    if len(data) < window:
        return 'undefined'
    
    # Extract close prices
    if 'close' in data.columns:
        close = data['close']
    else:
        close = data.iloc[:, 0]  # Use first column
    
    # Calculate volatility (standard deviation of returns)
    returns = close.pct_change()
    volatility = returns.iloc[-window:].std()
    
    # Calculate linear regression to determine trend
    x = np.arange(window)
    y = close.iloc[-window:].values
    
    # Fit linear regression
    slope, intercept = np.polyfit(x, y, 1)
    
    # Calculate R-squared to determine trend strength
    y_pred = slope * x + intercept
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Calculate Bollinger Band width to identify ranging markets
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    
    band_width = (upper_band - lower_band) / rolling_mean
    recent_band_width = band_width.iloc[-1]
    avg_band_width = band_width.iloc[-window:].mean()
    
    # Determine regime
    if r_squared > 0.7 and abs(slope) > 0:
        return 'trending'
    elif volatility > returns.iloc[-window*3:].std() * 1.5:
        return 'volatile'
    elif recent_band_width < avg_band_width * 0.8:
        return 'ranging'
    else:
        return 'undefined'

def calculate_correlation(correlation_data, window=20):
    """
    Calculate correlation between instruments
    
    Args:
        correlation_data: Dictionary of price data for different instruments
        window: Correlation window
        
    Returns:
        Dictionary with correlation coefficients
    """
    correlations = {}
    
    instruments = list(correlation_data.keys())
    
    for i in range(len(instruments)):
        for j in range(i+1, len(instruments)):
            inst1 = instruments[i]
            inst2 = instruments[j]
            
            # Extract close prices
            if 'close' in correlation_data[inst1].columns:
                close1 = correlation_data[inst1]['close']
            else:
                close1 = correlation_data[inst1].iloc[:, 0]  # Use first column
                
            if 'close' in correlation_data[inst2].columns:
                close2 = correlation_data[inst2]['close']
            else:
                close2 = correlation_data[inst2].iloc[:, 0]  # Use first column
            
            # Calculate correlation if we have enough data
            if len(close1) >= window and len(close2) >= window:
                corr = close1.iloc[-window:].corr(close2.iloc[-window:])
                corr_key = f"{inst1}_{inst2}"
                correlations[corr_key] = corr
    
    return correlations

def check_news_events(instrument, upcoming_window_hours=24, sensitivity='high'):
    """
    Check for upcoming news events
    
    Args:
        instrument: Trading instrument
        upcoming_window_hours: How many hours ahead to check
        sensitivity: Filter for news importance ('high', 'medium', 'low')
        
    Returns:
        List of upcoming news events
    """
    # This is a placeholder function
    # In a real implementation, this would connect to a news API or database
    
    # Extract currency codes from instrument
    currencies = []
    if len(instrument) >= 6:
        currencies.append(instrument[:3])
        currencies.append(instrument[3:6])
    
    # Mock data for demonstration
    current_time = datetime.now()
    
    mock_news_events = [
        {
            'time': current_time + timedelta(hours=2),
            'currency': 'USD',
            'event': 'FOMC Statement',
            'importance': 'high'
        },
        {
            'time': current_time + timedelta(hours=5),
            'currency': 'EUR',
            'event': 'ECB Interest Rate Decision',
            'importance': 'high'
        },
        {
            'time': current_time + timedelta(hours=8),
            'currency': 'GBP',
            'event': 'UK GDP',
            'importance': 'medium'
        },
        {
            'time': current_time + timedelta(hours=12),
            'currency': 'JPY',
            'event': 'BOJ Press Conference',
            'importance': 'medium'
        }
    ]
    
    # Filter by currency
    filtered_events = [
        event for event in mock_news_events
        if event['currency'] in currencies
    ]
    
    # Filter by time window
    window_end = current_time + timedelta(hours=upcoming_window_hours)
    filtered_events = [
        event for event in filtered_events
        if current_time <= event['time'] <= window_end
    ]
    
    # Filter by importance
    importance_levels = {
        'high': ['high'],
        'medium': ['high', 'medium'],
        'low': ['high', 'medium', 'low']
    }
    
    filtered_events = [
        event for event in filtered_events
        if event['importance'] in importance_levels.get(sensitivity, ['high'])
    ]
    
    # Format events for return
    formatted_events = []
    for event in filtered_events:
        time_str = event['time'].strftime('%Y-%m-%d %H:%M')
        formatted_events.append({
            'time': time_str,
            'currency': event['currency'],
            'event': event['event'],
            'importance': event['importance']
        })
    
    return formatted_events

def analyze_sentiment(instrument):
    """
    Analyze market sentiment for an instrument
    
    Args:
        instrument: Trading instrument
        
    Returns:
        Dictionary with sentiment analysis
    """
    # This is a placeholder function
    # In a real implementation, this would connect to sentiment APIs, social media, etc.
    
    # Mock sentiment data
    sentiment = {
        'overall': 0,  # -1 to +1 scale
        'source_breakdown': {
            'news': 0,
            'social_media': 0,
            'analyst_ratings': 0,
            'positioning': 0
        },
        'change_24h': 0,
        'volume': 0
    }
    
    # Generate random sentiment for demonstration
    import random
    
    sentiment['overall'] = random.uniform(-1, 1)
    sentiment['source_breakdown']['news'] = random.uniform(-1, 1)
    sentiment['source_breakdown']['social_media'] = random.uniform(-1, 1)
    sentiment['source_breakdown']['analyst_ratings'] = random.uniform(-1, 1)
    sentiment['source_breakdown']['positioning'] = random.uniform(-1, 1)
    sentiment['change_24h'] = random.uniform(-0.5, 0.5)
    sentiment['volume'] = random.uniform(0, 100)
    
    return sentiment

def calculate_volume_profile(data, price_levels=50, lookback_periods=None, volume_threshold_pct=70, recency_weight=True):
    """
    Calculate advanced volume profile for identifying institutional supply/demand zones
    
    Args:
        data: DataFrame with OHLCV price data
        price_levels: Number of price levels to divide the range into (higher for more precision)
        lookback_periods: Number of periods to analyze (None for all data)
        volume_threshold_pct: Percentage of volume to include in value area (typically 70%)
        recency_weight: Whether to weight recent volume more heavily
        
    Returns:
        Dictionary with enhanced volume profile data including key institutional levels
    """
    # Ensure data has necessary columns
    if not all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
        return {
            'profile': {},
            'poc': None,
            'vah': None,
            'val': None,
            'volume_by_price': [],
            'institutional_levels': []
        }
    
    # Use recent data for more relevant levels if lookback specified
    if lookback_periods is not None and lookback_periods < len(data):
        working_data = data.iloc[-lookback_periods:]
    else:
        working_data = data.copy()
    
    # Get price range for the analyzed period
    price_min = working_data['low'].min()
    price_max = working_data['high'].max()
    price_range = price_max - price_min
    
    # If range is zero or very small, return empty profile
    if price_range < 0.00001:
        return {
            'profile': {},
            'poc': price_min,
            'vah': price_min,
            'val': price_min,
            'volume_by_price': [],
            'institutional_levels': []
        }
    
    # Calculate increment size for each price level
    increment = price_range / price_levels
    
    # Initialize volume by price level
    volume_by_price = defaultdict(float)
    
    # Track volume for each candle's type (bullish/bearish)
    bullish_volume_by_price = defaultdict(float)
    bearish_volume_by_price = defaultdict(float)
    
    # Track volume deltas (buying vs selling pressure)
    volume_delta_by_price = defaultdict(float)
    
    # Calculate time weights if using recency weighting
    if recency_weight:
        num_candles = len(working_data)
        half_life = num_candles / 3  # Adjust this parameter to control decay rate
        time_weights = np.exp(np.linspace(0, -num_candles/half_life, num_candles))
        time_weights = time_weights / time_weights.mean()  # Normalize
    else:
        time_weights = np.ones(len(working_data))
    
    # Process each candle
    for idx in range(len(working_data)):
        candle_open = working_data['open'].iloc[idx]
        candle_high = working_data['high'].iloc[idx]
        candle_low = working_data['low'].iloc[idx]
        candle_close = working_data['close'].iloc[idx]
        candle_volume = working_data['volume'].iloc[idx]
        time_weight = time_weights[idx]
        
        # Skip invalid data
        if candle_volume <= 0 or np.isnan(candle_volume):
            continue
        
        # Determine candle type
        is_bullish = candle_close > candle_open
        
        # Calculate how many levels this candle spans
        low_level = int((candle_low - price_min) / increment)
        high_level = int((candle_high - price_min) / increment)
        
        # Constrain to valid range
        low_level = max(0, min(price_levels - 1, low_level))
        high_level = max(0, min(price_levels - 1, high_level))
        
        # Distribute volume across price levels with sophisticated weighting
        levels_spanned = high_level - low_level + 1
        
        if levels_spanned <= 1:
            # If candle spans just one level, assign all volume to that level
            weighted_volume = candle_volume * time_weight
            volume_by_price[low_level] += weighted_volume
            
            # Assign to bullish or bearish buckets
            if is_bullish:
                bullish_volume_by_price[low_level] += weighted_volume
            else:
                bearish_volume_by_price[low_level] += weighted_volume
                
            # Update volume delta (positive for bullish, negative for bearish)
            delta = weighted_volume if is_bullish else -weighted_volume
            volume_delta_by_price[low_level] += delta
            
        else:
            # Otherwise distribute volume across levels using time-price model
            # This simulates how price spends more time at certain levels within a candle
            
            # Calculate weightings based on price action theory
            # 1. More time spent near open/close
            # 2. Less time spent in rapid movements through a level
            # 3. High volume at reversal points
            
            # Define key price points for weighting
            price_points = {
                'open': candle_open,
                'close': candle_close,
                'mid': (candle_open + candle_close) / 2,
                'high': candle_high,
                'low': candle_low
            }
            
            # Assign weights for each price level based on key points
            level_weights = {}
            
            for level in range(low_level, high_level + 1):
                level_price_mid = price_min + (level + 0.5) * increment
                
                # Calculate weight based on proximity to key price points
                # Higher weight for levels closer to open, close, and mid points
                proximity_weight = 0
                
                for point_name, point_price in price_points.items():
                    distance = abs(level_price_mid - point_price)
                    if distance < increment:
                        # Higher weight for open/close points
                        if point_name in ['open', 'close']:
                            proximity_weight += 0.4 * (1 - distance/increment)
                        # Medium weight for mid point
                        elif point_name == 'mid':
                            proximity_weight += 0.2 * (1 - distance/increment)
                        # Lower weight for high/low (unless they're also open/close)
                        else:
                            proximity_weight += 0.1 * (1 - distance/increment)
                
                # Ensure weight is positive
                proximity_weight = max(0.1, proximity_weight)
                level_weights[level] = proximity_weight
                
            # Normalize weights to sum to 1
            weight_sum = sum(level_weights.values())
            if weight_sum > 0:
                for level in level_weights:
                    level_weights[level] /= weight_sum
            
            # Distribute volume according to weights
            for level, weight in level_weights.items():
                weighted_volume = candle_volume * time_weight * weight
                volume_by_price[level] += weighted_volume
                
                # Assign to bullish or bearish buckets
                if is_bullish:
                    bullish_volume_by_price[level] += weighted_volume
                else:
                    bearish_volume_by_price[level] += weighted_volume
                    
                # Update volume delta
                delta = weighted_volume if is_bullish else -weighted_volume
                volume_delta_by_price[level] += delta
    
    # Convert to regular dict for JSON serialization
    volume_profile = dict(volume_by_price)
    
    # Identify Point of Control (POC) - price level with highest volume
    if volume_profile:
        poc_level = max(volume_profile.keys(), key=lambda k: volume_profile[k])
        poc_price = price_min + (poc_level + 0.5) * increment
    else:
        poc_level = 0
        poc_price = price_min
    
    # Calculate Value Area (typically 70% of total volume)
    total_volume = sum(volume_profile.values())
    target_volume = total_volume * (volume_threshold_pct / 100)
    
    # Sort levels by volume (descending)
    sorted_levels = sorted(volume_profile.keys(), key=lambda k: volume_profile[k], reverse=True)
    
    # Track cumulative volume and included levels
    cumulative_volume = 0
    value_area_levels = []
    
    # Add levels until we reach target volume threshold
    for level in sorted_levels:
        value_area_levels.append(level)
        cumulative_volume += volume_profile[level]
        if cumulative_volume >= target_volume:
            break
    
    # Find Value Area High and Value Area Low
    if value_area_levels:
        val_level = min(value_area_levels)
        vah_level = max(value_area_levels)
        val_price = price_min + val_level * increment
        vah_price = price_min + (vah_level + 1) * increment
    else:
        val_price = price_min
        vah_price = price_max
    
    # Create array of price levels and volumes for charting
    volume_by_price_array = []
    
    for level in range(price_levels):
        price = price_min + (level + 0.5) * increment
        volume = volume_profile.get(level, 0)
        bull_volume = bullish_volume_by_price.get(level, 0)
        bear_volume = bearish_volume_by_price.get(level, 0)
        delta = volume_delta_by_price.get(level, 0)
        
        volume_by_price_array.append({
            'price': price,
            'volume': volume,
            'bullish_volume': bull_volume,
            'bearish_volume': bear_volume,
            'delta': delta
        })
    
    # Identify institutional levels based on volume analysis
    institutional_levels = []
    
    # 1. Look for significant volume nodes (clusters)
    # Sort levels by volume, descending
    high_volume_threshold = total_volume * 0.05  # 5% of total volume for a significant level
    volume_nodes = [level for level, vol in volume_profile.items() if vol > high_volume_threshold]
    
    # 2. Look for significant volume delta changes (buying vs selling pressure)
    # Find levels with strong buying pressure
    support_levels = [
        level for level, delta in volume_delta_by_price.items() 
        if delta > 0 and volume_profile.get(level, 0) > high_volume_threshold
    ]
    
    # Find levels with strong selling pressure
    resistance_levels = [
        level for level, delta in volume_delta_by_price.items() 
        if delta < 0 and volume_profile.get(level, 0) > high_volume_threshold
    ]
    
    # Add POC and Value Area as institutional levels
    institutional_levels.append({
        'price': poc_price,
        'type': 'poc',
        'strength': 0.9,
        'description': 'Point of Control'
    })
    
    institutional_levels.append({
        'price': vah_price,
        'type': 'vah',
        'strength': 0.7,
        'description': 'Value Area High'
    })
    
    institutional_levels.append({
        'price': val_price,
        'type': 'val',
        'strength': 0.7,
        'description': 'Value Area Low'
    })
    
    # Add support levels (high volume with positive delta)
    for level in support_levels:
        price = price_min + (level + 0.5) * increment
        volume = volume_profile.get(level, 0)
        strength = min(0.85, volume / total_volume * 10)  # Scale by relative volume
        
        # Skip if too close to an existing level
        if any(abs(price - existing['price']) / price < 0.001 for existing in institutional_levels):
            continue
            
        institutional_levels.append({
            'price': price,
            'type': 'support',
            'strength': strength,
            'description': 'High Volume Support'
        })
    
    # Add resistance levels (high volume with negative delta)
    for level in resistance_levels:
        price = price_min + (level + 0.5) * increment
        volume = volume_profile.get(level, 0)
        strength = min(0.85, volume / total_volume * 10)  # Scale by relative volume
        
        # Skip if too close to an existing level
        if any(abs(price - existing['price']) / price < 0.001 for existing in institutional_levels):
            continue
            
        institutional_levels.append({
            'price': price,
            'type': 'resistance',
            'strength': strength,
            'description': 'High Volume Resistance'
        })
    
    # Look for volume gaps (low volume areas) between high volume nodes
    # These often act as acceleration zones (price moves quickly through them)
    if len(volume_nodes) >= 2:
        sorted_volume_nodes = sorted(volume_nodes)
        
        for i in range(len(sorted_volume_nodes) - 1):
            current_node = sorted_volume_nodes[i]
            next_node = sorted_volume_nodes[i + 1]
            
            # Check if there's a gap (several low volume levels between nodes)
            if next_node - current_node > 3:
                # Calculate average volume in the gap
                gap_levels = range(current_node + 1, next_node)
                gap_volumes = [volume_profile.get(level, 0) for level in gap_levels]
                avg_gap_volume = sum(gap_volumes) / len(gap_volumes) if gap_volumes else 0
                
                # If average volume in gap is low, mark it as an acceleration zone
                if avg_gap_volume < high_volume_threshold / 3:
                    gap_start_price = price_min + (current_node + 1) * increment
                    gap_end_price = price_min + next_node * increment
                    
                    institutional_levels.append({
                        'price': (gap_start_price + gap_end_price) / 2,  # Midpoint
                        'type': 'acceleration_zone',
                        'range': [gap_start_price, gap_end_price],
                        'strength': 0.6,
                        'description': 'Volume Gap (Acceleration Zone)'
                    })
    
    return {
        'profile': volume_profile,  # Raw volume by level data
        'poc': poc_price,          # Point of Control
        'vah': vah_price,          # Value Area High
        'val': val_price,          # Value Area Low
        'volume_by_price': volume_by_price_array,  # Detailed volume data for charting
        'institutional_levels': institutional_levels  # Key price levels for trading
    }

def analyze_order_flow(data, window=20, delta_period=3, volume_threshold=2.0):
    """
    Advanced order flow analysis to detect institutional buying/selling pressure
    
    Args:
        data: DataFrame with OHLCV price data
        window: Window for analysis
        delta_period: Period for analyzing delta changes (recent pressure changes)
        volume_threshold: Multiple of average volume to consider significant
        
    Returns:
        Dictionary with enhanced order flow analysis including institutional activity patterns
    """
    if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        return {
            'buying_pressure': 0,
            'selling_pressure': 0,
            'absorption': False,
            'climax': False,
            'imbalance': 0,
            'institutional_activity': {
                'present': False,
                'type': None,
                'confidence': 0,
                'signals': []
            }
        }
    
    # Ensure we have enough data
    if len(data) < window:
        return {
            'buying_pressure': 0,
            'selling_pressure': 0,
            'absorption': False,
            'climax': False,
            'imbalance': 0,
            'institutional_activity': {
                'present': False,
                'type': None,
                'confidence': 0,
                'signals': []
            }
        }
    
    # Get recent data
    recent_data = data.iloc[-window:].copy()
    
    # Calculate basic metrics
    bullish_candles = recent_data['close'] > recent_data['open']
    bearish_candles = recent_data['close'] < recent_data['open']
    neutral_candles = recent_data['close'] == recent_data['open']
    
    # Calculate volume metrics
    total_volume = recent_data['volume'].sum()
    bullish_volume = recent_data.loc[bullish_candles, 'volume'].sum() if sum(bullish_candles) > 0 else 0
    bearish_volume = recent_data.loc[bearish_candles, 'volume'].sum() if sum(bearish_candles) > 0 else 0
    neutral_volume = recent_data.loc[neutral_candles, 'volume'].sum() if sum(neutral_candles) > 0 else 0
    
    # Calculate volume-weighted price movements
    bullish_moves = recent_data.loc[bullish_candles, 'close'] - recent_data.loc[bullish_candles, 'open'] if sum(bullish_candles) > 0 else 0
    bearish_moves = recent_data.loc[bearish_candles, 'open'] - recent_data.loc[bearish_candles, 'close'] if sum(bearish_candles) > 0 else 0
    
    if isinstance(bullish_moves, pd.Series):
        bullish_weighted = (bullish_moves * recent_data.loc[bullish_candles, 'volume']).sum() if sum(bullish_candles) > 0 else 0
    else:
        bullish_weighted = 0
        
    if isinstance(bearish_moves, pd.Series):
        bearish_weighted = (bearish_moves * recent_data.loc[bearish_candles, 'volume']).sum() if sum(bearish_candles) > 0 else 0
    else:
        bearish_weighted = 0
    
    # Normalize by total volume
    buying_pressure = bullish_weighted / total_volume if total_volume > 0 else 0
    selling_pressure = bearish_weighted / total_volume if total_volume > 0 else 0
    
    # Calculate imbalance (range from -1 to 1)
    # Positive means buying pressure, negative means selling pressure
    if buying_pressure + selling_pressure > 0:
        imbalance = (buying_pressure - selling_pressure) / (buying_pressure + selling_pressure)
    else:
        imbalance = 0
    
    # Detect volume/price anomalies
    # Volume climax - spike in volume
    recent_vol_avg = recent_data['volume'].iloc[:-1].mean() if len(recent_data) > 1 else 0
    latest_vol = recent_data['volume'].iloc[-1] if len(recent_data) > 0 else 0
    climax = latest_vol > recent_vol_avg * 2
    
    # Absorption - price doesn't move despite high volume
    recent_range_avg = (recent_data['high'] - recent_data['low']).iloc[:-1].mean() if len(recent_data) > 1 else 0
    latest_range = (recent_data['high'] - recent_data['low']).iloc[-1] if len(recent_data) > 0 else 0
    absorption = (latest_vol > recent_vol_avg * 1.5) and (latest_range < recent_range_avg * 0.5)
    
    # === INSTITUTIONAL ACTIVITY ANALYSIS ===
    institutional_signals = []
    
    # 1. Calculate Delta (Difference between buying and selling volume)
    # Add calculated columns to our dataframe for easier analysis
    recent_data['candle_type'] = np.where(recent_data['close'] > recent_data['open'], 1, 
                                         np.where(recent_data['close'] < recent_data['open'], -1, 0))
    recent_data['body_size'] = abs(recent_data['close'] - recent_data['open'])
    recent_data['range_size'] = recent_data['high'] - recent_data['low']
    recent_data['body_to_range_ratio'] = np.where(recent_data['range_size'] > 0, 
                                                 recent_data['body_size'] / recent_data['range_size'], 0)
    
    # Calculate delta for each candle
    recent_data['volume_delta'] = recent_data['volume'] * recent_data['candle_type']
    
    # Cumulative delta
    recent_data['cumulative_delta'] = recent_data['volume_delta'].cumsum()
    
    # Recent delta change - look for acceleration or deceleration
    recent_delta = recent_data['volume_delta'].iloc[-delta_period:].sum()
    previous_delta = recent_data['volume_delta'].iloc[-2*delta_period:-delta_period].sum() 
    delta_acceleration = recent_delta - previous_delta
    
    # 2. Detect absorption (institutions absorbing supply or demand)
    # High volume but price not moving much - classic absorption pattern
    if absorption:
        # Determine direction of absorption
        if recent_data['close'].iloc[-1] >= recent_data['open'].iloc[-1]:  # Bullish
            institutional_signals.append({
                'type': 'bullish_absorption',
                'description': 'High volume with minimal upward price movement - likely accumulation',
                'strength': 0.8,
                'candle_index': len(recent_data) - 1
            })
        else:  # Bearish
            institutional_signals.append({
                'type': 'bearish_absorption',
                'description': 'High volume with minimal downward price movement - likely distribution',
                'strength': 0.8,
                'candle_index': len(recent_data) - 1
            })
    
    # 3. Detect stopping volume (high volume at support/resistance with reversal)
    # Last candle has high volume, small body, and price reverses
    high_volume_candle = recent_data['volume'].iloc[-1] > recent_data['volume'].iloc[-5:].mean() * 1.5
    small_body = recent_data['body_to_range_ratio'].iloc[-1] < 0.3
    lower_wick_ratio = 0
    upper_wick_ratio = 0
    
    if recent_data['range_size'].iloc[-1] > 0:
        if recent_data['candle_type'].iloc[-1] == 1:  # Bullish
            lower_wick_ratio = (recent_data['open'].iloc[-1] - recent_data['low'].iloc[-1]) / recent_data['range_size'].iloc[-1]
        elif recent_data['candle_type'].iloc[-1] == -1:  # Bearish
            upper_wick_ratio = (recent_data['high'].iloc[-1] - recent_data['open'].iloc[-1]) / recent_data['range_size'].iloc[-1]
    
    # Check for reversal pattern
    prior_trend_bearish = recent_data['close'].iloc[-3:-1].mean() < recent_data['close'].iloc[-5:-3].mean()
    prior_trend_bullish = recent_data['close'].iloc[-3:-1].mean() > recent_data['close'].iloc[-5:-3].mean()
    
    if high_volume_candle and small_body:
        # Bullish stopping volume (high volume, bearish trend, long lower wick)
        if prior_trend_bearish and recent_data['candle_type'].iloc[-1] >= 0 and lower_wick_ratio > 0.6:
            institutional_signals.append({
                'type': 'bullish_stopping_volume',
                'description': 'High volume with long lower wick after downtrend - institutional buying',
                'strength': 0.85,
                'candle_index': len(recent_data) - 1
            })
        # Bearish stopping volume (high volume, bullish trend, long upper wick)
        elif prior_trend_bullish and recent_data['candle_type'].iloc[-1] <= 0 and upper_wick_ratio > 0.6:
            institutional_signals.append({
                'type': 'bearish_stopping_volume',
                'description': 'High volume with long upper wick after uptrend - institutional selling',
                'strength': 0.85,
                'candle_index': len(recent_data) - 1
            })
    
    # 4. Detect climax volume (exhaustion)
    if climax:
        # End of major move - look for exhaustion
        extended_move = False
        if recent_data['candle_type'].iloc[-1] == 1:  # Bullish
            # Check if we're already in a strong uptrend
            up_candles = sum(recent_data['candle_type'].iloc[-5:] > 0)
            if up_candles >= 4 and recent_data['close'].iloc[-1] > recent_data['close'].iloc[-5:].mean() * 1.02:
                extended_move = True
                institutional_signals.append({
                    'type': 'bullish_climax',
                    'description': 'Very high volume after extended uptrend - potential exhaustion',
                    'strength': 0.75,
                    'candle_index': len(recent_data) - 1
                })
        elif recent_data['candle_type'].iloc[-1] == -1:  # Bearish
            # Check if we're already in a strong downtrend
            down_candles = sum(recent_data['candle_type'].iloc[-5:] < 0)
            if down_candles >= 4 and recent_data['close'].iloc[-1] < recent_data['close'].iloc[-5:].mean() * 0.98:
                extended_move = True
                institutional_signals.append({
                    'type': 'bearish_climax',
                    'description': 'Very high volume after extended downtrend - potential exhaustion',
                    'strength': 0.75,
                    'candle_index': len(recent_data) - 1
                })
        # If not at end of trend, may be acceleration
        if not extended_move:
            if recent_data['candle_type'].iloc[-1] == 1:  # Bullish
                institutional_signals.append({
                    'type': 'bullish_breakout_volume',
                    'description': 'Very high volume bullish candle - potential institutional entry',
                    'strength': 0.7,
                    'candle_index': len(recent_data) - 1
                })
            elif recent_data['candle_type'].iloc[-1] == -1:  # Bearish
                institutional_signals.append({
                    'type': 'bearish_breakout_volume',
                    'description': 'Very high volume bearish candle - potential institutional exit',
                    'strength': 0.7,
                    'candle_index': len(recent_data) - 1
                })
    
    # 5. Detect delta divergences with price
    # Price making new high but delta not confirming
    recent_price_high = recent_data['high'].iloc[-1] >= recent_data['high'].iloc[-window:-1].max()
    recent_price_low = recent_data['low'].iloc[-1] <= recent_data['low'].iloc[-window:-1].min()
    delta_confirms_high = recent_data['cumulative_delta'].iloc[-1] >= recent_data['cumulative_delta'].iloc[-window:-1].max()
    delta_confirms_low = recent_data['cumulative_delta'].iloc[-1] <= recent_data['cumulative_delta'].iloc[-window:-1].min()
    
    if recent_price_high and not delta_confirms_high:
        institutional_signals.append({
            'type': 'bearish_delta_divergence',
            'description': 'Price making new high but buying volume not confirming - potential distribution',
            'strength': 0.8,
            'candle_index': len(recent_data) - 1
        })
    
    if recent_price_low and not delta_confirms_low:
        institutional_signals.append({
            'type': 'bullish_delta_divergence',
            'description': 'Price making new low but selling volume not confirming - potential accumulation',
            'strength': 0.8,
            'candle_index': len(recent_data) - 1
        })
    
    # 6. Detect signs of accumulation/distribution
    # Series of candles with decreasing ranges but sustained high volume
    recent_ranges = recent_data['range_size'].iloc[-5:].values
    recent_volumes = recent_data['volume'].iloc[-5:].values
    avg_range = recent_data['range_size'].iloc[-10:-5].mean()
    avg_volume = recent_data['volume'].iloc[-10:-5].mean()
    
    decreasing_ranges = all(recent_ranges[i] <= recent_ranges[i-1] for i in range(1, len(recent_ranges)))
    sustained_high_volume = sum(vol > avg_volume * 1.2 for vol in recent_volumes) >= 3
    
    if decreasing_ranges and sustained_high_volume:
        avg_delta = recent_data['volume_delta'].iloc[-5:].mean()
        if avg_delta > 0:  # Bullish accumulation
            institutional_signals.append({
                'type': 'bullish_accumulation',
                'description': 'Decreasing price ranges with sustained high volume and positive delta - institutional accumulation',
                'strength': 0.9,
                'candle_index': slice(-5, None)
            })
        else:  # Bearish distribution
            institutional_signals.append({
                'type': 'bearish_distribution',
                'description': 'Decreasing price ranges with sustained high volume and negative delta - institutional distribution',
                'strength': 0.9,
                'candle_index': slice(-5, None)
            })
    
    # 7. Identify institutional tape reading patterns
    # Large delta reversals, often signaling institutional position changes
    if len(recent_data) >= 5:
        # Check for delta reversals
        delta_5_periods = recent_data['volume_delta'].iloc[-5:].values
        if (delta_5_periods[0] < 0 and delta_5_periods[-1] > 0 and 
            abs(delta_5_periods[-1]) > abs(delta_5_periods[0]) * 1.5):
            # Strong bullish reversal in delta
            institutional_signals.append({
                'type': 'bullish_delta_reversal',
                'description': 'Strong shift from selling to buying volume - institutional buying after selloff',
                'strength': 0.75,
                'candle_index': len(recent_data) - 1
            })
        elif (delta_5_periods[0] > 0 and delta_5_periods[-1] < 0 and 
              abs(delta_5_periods[-1]) > abs(delta_5_periods[0]) * 1.5):
            # Strong bearish reversal in delta
            institutional_signals.append({
                'type': 'bearish_delta_reversal',
                'description': 'Strong shift from buying to selling volume - institutional selling after rally',
                'strength': 0.75,
                'candle_index': len(recent_data) - 1
            })
    
    # Determine overall institutional activity
    institutional_present = len(institutional_signals) > 0
    institutional_type = None
    institutional_confidence = 0
    
    if institutional_present:
        # Count bullish vs bearish signals
        bullish_signals = [s for s in institutional_signals if 'bullish' in s['type']]
        bearish_signals = [s for s in institutional_signals if 'bearish' in s['type']]
        
        # Get the strongest signal's confidence
        if institutional_signals:
            institutional_confidence = max(s['strength'] for s in institutional_signals)
        
        # Determine overall direction
        if len(bullish_signals) > len(bearish_signals):
            institutional_type = 'bullish'
        elif len(bearish_signals) > len(bullish_signals):
            institutional_type = 'bearish'
        else:
            # If equal count, use the strongest signal
            if institutional_signals:
                strongest_signal = max(institutional_signals, key=lambda x: x['strength'])
                institutional_type = 'bullish' if 'bullish' in strongest_signal['type'] else 'bearish'
    
    # Create fingerprint data for visualization
    # Simple version: ratio of delta vs price change over different periods
    fingerprint = {}
    for period in [3, 5, 10, 20]:
        if len(recent_data) >= period:
            price_change = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[-period] - 1) * 100
            delta_sum = recent_data['volume_delta'].iloc[-period:].sum()
            delta_normalized = delta_sum / recent_data['volume'].iloc[-period:].sum() if recent_data['volume'].iloc[-period:].sum() > 0 else 0
            fingerprint[f'{period}_period'] = {
                'price_change_pct': price_change,
                'delta_ratio': delta_normalized,
                'efficiency': abs(price_change) / recent_data['range_size'].iloc[-period:].sum() if recent_data['range_size'].iloc[-period:].sum() > 0 else 0
            }
    
    return {
        'buying_pressure': buying_pressure,
        'selling_pressure': selling_pressure,
        'absorption': absorption,
        'climax': climax,
        'imbalance': imbalance,
        'institutional_activity': {
            'present': institutional_present,
            'type': institutional_type,
            'confidence': institutional_confidence,
            'signals': institutional_signals
        },
        'delta': {
            'recent': recent_delta,
            'previous': previous_delta,
            'acceleration': delta_acceleration,
            'cumulative': recent_data['cumulative_delta'].iloc[-1] if len(recent_data) > 0 else 0
        },
        'fingerprint': fingerprint
    }

def analyze_long_term_trends(daily_data, weekly_data=None, monthly_data=None):
    """
    Analyze long-term trends using weekly and monthly timeframes for high-level market context
    
    Args:
        daily_data: DataFrame with daily price data
        weekly_data: Optional DataFrame with weekly price data
        monthly_data: Optional DataFrame with monthly price data
        
    Returns:
        Dictionary with long-term trend analysis
    """
    results = {
        'daily_trend': {'direction': 'undefined', 'strength': 0},
        'weekly_trend': {'direction': 'undefined', 'strength': 0},
        'monthly_trend': {'direction': 'undefined', 'strength': 0},
        'trend_alignment': False,
        'support_levels': [],
        'resistance_levels': []
    }
    
    # Analyze daily trend
    if daily_data is not None and len(daily_data) >= 20:
        if 'close' in daily_data.columns:
            close = daily_data['close']
            
            # Calculate common moving averages
            daily_sma20 = close.rolling(window=20).mean()
            daily_sma50 = close.rolling(window=50).mean()
            daily_sma200 = close.rolling(window=200).mean() if len(close) >= 200 else None
            
            # Get most recent values
            current_price = close.iloc[-1]
            sma20_current = daily_sma20.iloc[-1]
            sma50_current = daily_sma50.iloc[-1]
            sma200_current = daily_sma200.iloc[-1] if daily_sma200 is not None else None
            
            # Determine trend direction
            if sma20_current > sma50_current and current_price > sma20_current:
                direction = 'up'
                strength = 0.7
                
                # Strong uptrend if above 200 SMA
                if sma200_current is not None and current_price > sma200_current:
                    strength = 0.9
            elif sma20_current < sma50_current and current_price < sma20_current:
                direction = 'down'
                strength = 0.7
                
                # Strong downtrend if below 200 SMA
                if sma200_current is not None and current_price < sma200_current:
                    strength = 0.9
            else:
                # Check if price is within a small range of SMAs
                if sma20_current is not None and sma50_current is not None:
                    avg_price = (sma20_current + sma50_current) / 2
                    price_range = abs(sma20_current - sma50_current)
                    
                    if abs(current_price - avg_price) < price_range * 0.2:
                        direction = 'sideways'
                        strength = 0.6
                    elif current_price > sma20_current:
                        direction = 'up'
                        strength = 0.4
                    elif current_price < sma20_current:
                        direction = 'down'
                        strength = 0.4
                    else:
                        direction = 'undefined'
                        strength = 0
                else:
                    direction = 'undefined'
                    strength = 0
            
            results['daily_trend'] = {
                'direction': direction,
                'strength': strength
            }
    
    # Analyze weekly trend
    if weekly_data is not None and len(weekly_data) >= 10:
        if 'close' in weekly_data.columns:
            close = weekly_data['close']
            
            # Calculate common moving averages for weekly
            weekly_sma8 = close.rolling(window=8).mean()
            weekly_sma20 = close.rolling(window=20).mean()
            
            # Get most recent values
            current_price = close.iloc[-1]
            sma8_current = weekly_sma8.iloc[-1]
            sma20_current = weekly_sma20.iloc[-1]
            
            # Determine trend direction
            if sma8_current > sma20_current and current_price > sma8_current:
                direction = 'up'
                strength = 0.8
            elif sma8_current < sma20_current and current_price < sma8_current:
                direction = 'down'
                strength = 0.8
            else:
                # Check if price is within a small range of SMAs
                avg_price = (sma8_current + sma20_current) / 2
                price_range = abs(sma8_current - sma20_current)
                
                if abs(current_price - avg_price) < price_range * 0.2:
                    direction = 'sideways'
                    strength = 0.5
                elif current_price > sma8_current:
                    direction = 'up'
                    strength = 0.4
                elif current_price < sma8_current:
                    direction = 'down'
                    strength = 0.4
                else:
                    direction = 'undefined'
                    strength = 0
            
            results['weekly_trend'] = {
                'direction': direction,
                'strength': strength
            }
    
    # Analyze monthly trend
    if monthly_data is not None and len(monthly_data) >= 6:
        if 'close' in monthly_data.columns:
            close = monthly_data['close']
            
            # Calculate common moving averages for monthly
            monthly_sma6 = close.rolling(window=6).mean()
            monthly_sma12 = close.rolling(window=12).mean() if len(close) >= 12 else None
            
            # Get most recent values
            current_price = close.iloc[-1]
            sma6_current = monthly_sma6.iloc[-1]
            sma12_current = monthly_sma12.iloc[-1] if monthly_sma12 is not None else None
            
            # Determine trend direction
            if sma12_current is not None:
                if sma6_current > sma12_current and current_price > sma6_current:
                    direction = 'up'
                    strength = 0.9
                elif sma6_current < sma12_current and current_price < sma6_current:
                    direction = 'down'
                    strength = 0.9
                else:
                    # Check if price is within a small range of SMAs
                    avg_price = (sma6_current + sma12_current) / 2
                    price_range = abs(sma6_current - sma12_current)
                    
                    if abs(current_price - avg_price) < price_range * 0.2:
                        direction = 'sideways'
                        strength = 0.5
                    elif current_price > sma6_current:
                        direction = 'up'
                        strength = 0.4
                    elif current_price < sma6_current:
                        direction = 'down'
                        strength = 0.4
                    else:
                        direction = 'undefined'
                        strength = 0
            else:
                # Simple trend detection using the 6-month SMA
                recent_prices = close.iloc[-3:]
                if all(price > sma6_current for price in recent_prices):
                    direction = 'up'
                    strength = 0.7
                elif all(price < sma6_current for price in recent_prices):
                    direction = 'down'
                    strength = 0.7
                else:
                    direction = 'sideways'
                    strength = 0.5
            
            results['monthly_trend'] = {
                'direction': direction,
                'strength': strength
            }
    
    # Check trend alignment across timeframes
    daily_dir = results['daily_trend']['direction']
    weekly_dir = results['weekly_trend']['direction']
    monthly_dir = results['monthly_trend']['direction']
    
    results['trend_alignment'] = (daily_dir == weekly_dir == monthly_dir and 
                                daily_dir != 'undefined' and daily_dir != 'sideways')
    
    # Extract significant support and resistance levels from weekly/monthly data
    if monthly_data is not None and len(monthly_data) >= 12:
        # Find swing highs and lows in monthly data (more significant levels)
        if all(col in monthly_data.columns for col in ['high', 'low']):
            swing_window = 3  # 3 months on each side
            
            # Loop through monthly data to find swing points
            for i in range(swing_window, len(monthly_data) - swing_window):
                high_window = monthly_data['high'].iloc[i-swing_window:i+swing_window+1]
                low_window = monthly_data['low'].iloc[i-swing_window:i+swing_window+1]
                
                # Check if this point is a swing high
                if monthly_data['high'].iloc[i] == high_window.max():
                    results['resistance_levels'].append({
                        'price': monthly_data['high'].iloc[i],
                        'strength': 0.9,  # Monthly resistance is strong
                        'source': 'monthly'
                    })
                
                # Check if this point is a swing low
                if monthly_data['low'].iloc[i] == low_window.min():
                    results['support_levels'].append({
                        'price': monthly_data['low'].iloc[i],
                        'strength': 0.9,  # Monthly support is strong
                        'source': 'monthly'
                    })
    
    # Add weekly levels if available (secondary importance)
    if weekly_data is not None and len(weekly_data) >= 20:
        if all(col in weekly_data.columns for col in ['high', 'low']):
            swing_window = 4  # 4 weeks on each side
            
            # Loop through recent weekly data only (last 52 weeks / 1 year)
            recent_weeks = min(52, len(weekly_data) - swing_window - 1)
            
            for i in range(len(weekly_data) - recent_weeks, len(weekly_data) - swing_window):
                high_window = weekly_data['high'].iloc[i-swing_window:i+swing_window+1]
                low_window = weekly_data['low'].iloc[i-swing_window:i+swing_window+1]
                
                # Check if this point is a swing high
                if weekly_data['high'].iloc[i] == high_window.max():
                    results['resistance_levels'].append({
                        'price': weekly_data['high'].iloc[i],
                        'strength': 0.7,  # Weekly resistance is moderately strong
                        'source': 'weekly'
                    })
                
                # Check if this point is a swing low
                if weekly_data['low'].iloc[i] == low_window.min():
                    results['support_levels'].append({
                        'price': weekly_data['low'].iloc[i],
                        'strength': 0.7,  # Weekly support is moderately strong
                        'source': 'weekly'
                    })
    
    # Combine and deduplicate levels, keeping the stronger ones
    # Group support and resistance levels that are very close together
    def merge_close_levels(levels, threshold_pct=0.005):
        if not levels:
            return []
            
        # Sort levels by price
        sorted_levels = sorted(levels, key=lambda x: x['price'])
        
        # Initialize merged list with the first level
        merged_levels = [sorted_levels[0]]
        
        # Loop through remaining levels and merge if close
        for level in sorted_levels[1:]:
            last_level = merged_levels[-1]
            
            # Check if current level is very close to the previous one
            price_diff_pct = abs(level['price'] - last_level['price']) / last_level['price']
            
            if price_diff_pct < threshold_pct:
                # If current level is stronger, replace previous
                if level['strength'] > last_level['strength']:
                    merged_levels[-1] = level
            else:
                # Otherwise add as new level
                merged_levels.append(level)
                
        return merged_levels
    
    # Merge close support and resistance levels
    results['support_levels'] = merge_close_levels(results['support_levels'])
    results['resistance_levels'] = merge_close_levels(results['resistance_levels'])
    
    return results

def check_high_impact_times(instrument):
    """
    Check if current time is optimal for trading the instrument
    
    Args:
        instrument: Trading instrument
        
    Returns:
        Boolean indicating whether it's a good time to trade
    """
    # Extract currency codes to determine relevant sessions
    sessions = []
    
    if len(instrument) >= 6:
        base = instrument[:3]
        quote = instrument[3:6]
        
        # Map currencies to sessions
        currency_sessions = {
            'USD': 'America',
            'CAD': 'America',
            'EUR': 'Europe',
            'GBP': 'Europe',
            'CHF': 'Europe',
            'JPY': 'Asia',
            'AUD': 'Asia',
            'NZD': 'Asia'
        }
        
        if base in currency_sessions:
            sessions.append(currency_sessions[base])
        if quote in currency_sessions:
            sessions.append(currency_sessions[quote])
    
    # Get current UTC time
    current_time = datetime.utcnow().time()
    
    # Define session times (UTC)
    asia_session = (time(0, 0), time(9, 0))  # 00:00-09:00 UTC
    europe_session = (time(7, 0), time(16, 0))  # 07:00-16:00 UTC
    america_session = (time(12, 0), time(21, 0))  # 12:00-21:00 UTC
    
    # Check if current time is within any relevant session
    in_session = False
    
    for session in sessions:
        if session == 'Asia' and asia_session[0] <= current_time <= asia_session[1]:
            in_session = True
            break
        elif session == 'Europe' and europe_session[0] <= current_time <= europe_session[1]:
            in_session = True
            break
        elif session == 'America' and america_session[0] <= current_time <= america_session[1]:
            in_session = True
            break
    
    # Check overlap periods (most active times)
    asia_europe_overlap = (time(7, 0), time(9, 0))  # 07:00-09:00 UTC
    europe_america_overlap = (time(12, 0), time(16, 0))  # 12:00-16:00 UTC
    
    in_overlap = (
        (asia_europe_overlap[0] <= current_time <= asia_europe_overlap[1] and 
         ('Asia' in sessions or 'Europe' in sessions)) or
        (europe_america_overlap[0] <= current_time <= europe_america_overlap[1] and 
         ('Europe' in sessions or 'America' in sessions))
    )
    
    # If it's in an overlap period, it's definitely a good time
    if in_overlap:
        return True
    
    # If it's in a relevant session, it's generally okay
    if in_session:
        return True
    
    # Otherwise, it's not an optimal time
    return False