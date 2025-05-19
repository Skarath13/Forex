import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta

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