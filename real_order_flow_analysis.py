#!/usr/bin/env python3
"""
Real Order Flow Analysis

This script fetches real historical forex data and applies enhanced order flow
analysis to approximate institutional buying/selling pressure.

It uses volume profile analysis and advanced order flow metrics to identify
potential institutional activity in the market.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader as pdr
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our order flow analysis tools
from src.utils.market_analysis import analyze_order_flow, calculate_volume_profile
from src.utils.visualization import plot_order_flow_analysis, plot_volume_profile

def fetch_real_historical_data(symbol, period="1y", interval="1d"):
    """
    Fetch real historical data from Yahoo Finance or other sources
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD=X" for EUR/USD)
        period: Time period to fetch (e.g., "1mo", "3mo", "1y", "5y")
        interval: Data interval (e.g., "1d", "1h", "15m")
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Fetching {period} of {interval} data for {symbol}...")
    
    try:
        # Try first using pandas-datareader
        try:
            # Convert period string to actual start/end dates
            end_date = datetime.now()
            if period == "1mo":
                start_date = end_date - timedelta(days=30)
            elif period == "3mo":
                start_date = end_date - timedelta(days=90)
            elif period == "6mo":
                start_date = end_date - timedelta(days=180)
            elif period == "2y":
                start_date = end_date - timedelta(days=730)
            else:  # Default to 1y
                start_date = end_date - timedelta(days=365)
            
            # Only use this method for daily data
            if interval == "1d":
                data = pdr.data.get_data_yahoo(symbols=symbol, start=start_date, end=end_date)
                print(f"Successfully fetched {len(data)} periods of data with pandas-datareader")
            else:
                raise ValueError(f"Need to use yfinance for {interval} interval")
        except Exception as pdr_error:
            print(f"pandas-datareader fetch failed: {pdr_error}, trying yfinance...")
            # Fall back to yfinance
            data = yf.download(
                tickers=symbol,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False
            )
            print(f"Successfully fetched {len(data)} periods of data with yfinance")
        
        # Rename columns to match our expected format
        data.columns = [c.lower() for c in data.columns]
        
        # Make sure we have all required columns
        if 'volume' not in data.columns or data['volume'].sum() == 0:
            # Some forex data might not have volume - create synthetic volume
            print("Warning: No volume data available, creating synthetic volume")
            data['volume'] = create_synthetic_volume(data)
        
        # Reset index to have date as a column if it's not already
        if 'date' not in data.columns:
            data = data.reset_index()
            if 'index' in data.columns and 'date' not in data.columns:
                data = data.rename(columns={'index': 'date'})
            elif 'Date' in data.columns and 'date' not in data.columns:
                data = data.rename(columns={'Date': 'date'})
        
        # Additional data validation
        if len(data) == 0:
            print("Warning: No data returned, trying to generate sample data")
            data = generate_sample_data(symbol, interval, period)
            
        return data
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Generating sample data instead...")
        return generate_sample_data(symbol, interval, period)

def create_synthetic_volume(data):
    """
    Create synthetic volume data based on price volatility
    This approximation assumes higher volatility = higher volume
    
    Args:
        data: DataFrame with OHLC data
        
    Returns:
        Series with synthetic volume
    """
    # Calculate true range as volatility proxy
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    
    # True range is the greatest of the three
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    # Normalize to a reasonable volume range (10,000 - 100,000)
    min_volume = 10000
    max_volume = 100000
    
    normalized_tr = (true_range - true_range.min()) / (true_range.max() - true_range.min())
    synthetic_volume = min_volume + normalized_tr * (max_volume - min_volume)
    
    # Add some randomness to make it more realistic
    noise = np.random.normal(1, 0.3, len(synthetic_volume))
    synthetic_volume = synthetic_volume * noise
    
    # Ensure volume is positive
    synthetic_volume = np.maximum(synthetic_volume, min_volume / 10)
    
    return synthetic_volume

def generate_sample_data(symbol, interval, period):
    """
    Generate sample data when real data is not available
    
    Args:
        symbol: Trading symbol used for labeling
        interval: Data interval (for labeling)
        period: Time period (used to determine length)
        
    Returns:
        DataFrame with synthetic OHLCV data
    """
    print(f"Generating sample data for {symbol}...")
    
    # Determine number of periods based on requested period
    if period == "1mo":
        n_periods = 30
    elif period == "3mo":
        n_periods = 90
    elif period == "6mo":
        n_periods = 180
    elif period == "2y":
        n_periods = 500
    else:  # Default to 1y
        n_periods = 252
    
    # Adjust based on interval
    if interval == "1h":
        n_periods = min(n_periods * 8, 500)  # 8 hours per day, max 500 points
    elif interval == "15m":
        n_periods = min(n_periods * 32, 500)  # 32 15-min periods per day, max 500
    elif interval == "1wk":
        n_periods = max(n_periods // 5, 50)  # ~5 days per week, min 50 points
    
    # Create dates
    end_date = datetime.now()
    if interval == "1d":
        dates = [end_date - timedelta(days=i) for i in range(n_periods, 0, -1)]
    elif interval == "1h":
        dates = [end_date - timedelta(hours=i) for i in range(n_periods, 0, -1)]
    elif interval == "15m":
        dates = [end_date - timedelta(minutes=15*i) for i in range(n_periods, 0, -1)]
    elif interval == "1wk":
        dates = [end_date - timedelta(weeks=i) for i in range(n_periods, 0, -1)]
    
    # Generate sample price data
    base_price = 1.0
    if symbol.startswith("EUR"):
        base_price = 1.1
    elif symbol.startswith("GBP"):
        base_price = 1.3
    elif symbol.startswith("USD"):
        base_price = 110.0 if "JPY" in symbol else 1.0
    
    # Generate price with trends and reversals
    close_prices = [base_price]
    
    # Add trend and reversal points
    for i in range(1, n_periods):
        # Create different market patterns
        if i < n_periods * 0.2:  # First 20% - uptrend
            new_price = close_prices[i-1] * (1 + 0.001 + np.random.normal(0, 0.003))
        elif i < n_periods * 0.4:  # 20-40% - downtrend
            new_price = close_prices[i-1] * (1 - 0.001 + np.random.normal(0, 0.003))
        elif i < n_periods * 0.6:  # 40-60% - accumulation
            new_price = close_prices[i-1] * (1 + np.random.normal(0, 0.002))
        elif i < n_periods * 0.8:  # 60-80% - uptrend
            new_price = close_prices[i-1] * (1 + 0.002 + np.random.normal(0, 0.003))
        else:  # Last 20% - distribution
            new_price = close_prices[i-1] * (1 - 0.0005 + np.random.normal(0, 0.004))
        
        close_prices.append(new_price)
    
    # Generate OHLC data
    open_prices = []
    high_prices = []
    low_prices = []
    volumes = []
    
    for i in range(n_periods):
        # Create open prices
        if i == 0:
            open_price = close_prices[i] * (1 - np.random.uniform(0, 0.005))
        else:
            open_price = close_prices[i-1]
        
        # Create price ranges based on period
        range_factor = 1.0
        if i < n_periods * 0.4:  # First 40% - moderate range
            range_factor = 1.0 + (i / n_periods)
        elif i < n_periods * 0.6:  # 40-60% - smaller range (accumulation)
            range_factor = 0.7
        elif i < n_periods * 0.8:  # 60-80% - increasing range
            range_factor = 1.0 + ((i - n_periods * 0.6) / (n_periods * 0.2))
        else:  # Last 20% - larger range (distribution)
            range_factor = 1.5
        
        # Calculate candle ranges
        price_range = abs(close_prices[i] - open_price) * range_factor
        
        # Create high/low prices with more realistic wicks
        if open_price > close_prices[i]:  # Bearish candle
            high_price = open_price + price_range * 0.3
            low_price = close_prices[i] - price_range * 0.7
        else:  # Bullish or neutral candle
            high_price = close_prices[i] + price_range * 0.7
            low_price = open_price - price_range * 0.3
        
        # Add randomness to wicks
        high_price += np.random.uniform(0, price_range * 0.2)
        low_price -= np.random.uniform(0, price_range * 0.2)
        
        # Store values
        open_prices.append(open_price)
        high_prices.append(high_price)
        low_prices.append(low_price)
        
        # Generate volume with patterns
        base_vol = 1000
        if i < n_periods * 0.2:  # First 20% - moderate volume
            vol = base_vol * (0.8 + i/n_periods)
        elif i < n_periods * 0.4:  # 20-40% - high volume (downtrend)
            vol = base_vol * 1.5
        elif i < n_periods * 0.6:  # 40-60% - declining volume (accumulation)
            vol = base_vol * (1.2 - ((i - n_periods * 0.4) / (n_periods * 0.2)) * 0.5)
        elif i < n_periods * 0.8:  # 60-80% - increasing volume (uptrend)
            vol = base_vol * (0.7 + ((i - n_periods * 0.6) / (n_periods * 0.2)) * 1.3)
        else:  # Last 20% - high but uneven volume (distribution)
            vol = base_vol * (1.5 - ((i - n_periods * 0.8) / (n_periods * 0.2)) * 0.3)
        
        # Add some randomness to volume
        vol *= np.random.uniform(0.7, 1.3)
        volumes.append(vol)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    # Add specific institutional patterns
    # 1. Add absorption pattern during accumulation phase
    absorption_idx = int(n_periods * 0.5)  # Middle of accumulation
    df.loc[absorption_idx, 'volume'] = df['volume'].mean() * 3.5
    mid_price = (df.loc[absorption_idx, 'open'] + df.loc[absorption_idx, 'close']) / 2
    df.loc[absorption_idx, 'open'] = mid_price * 0.998
    df.loc[absorption_idx, 'close'] = mid_price * 1.002
    df.loc[absorption_idx, 'high'] = mid_price * 1.005
    df.loc[absorption_idx, 'low'] = mid_price * 0.995
    
    # 2. Add stopping volume at the end of downtrend
    stopping_idx = int(n_periods * 0.38)  # Just before end of downtrend
    df.loc[stopping_idx, 'volume'] = df['volume'].mean() * 3.0
    df.loc[stopping_idx, 'open'] = df.loc[stopping_idx, 'close'] * 0.99
    df.loc[stopping_idx, 'close'] = df.loc[stopping_idx, 'close'] * 1.003
    df.loc[stopping_idx, 'low'] = df.loc[stopping_idx, 'close'] * 0.96  # Long lower wick
    
    # 3. Add climax volume at peak of uptrend
    climax_idx = int(n_periods * 0.78)  # Near peak of uptrend
    df.loc[climax_idx, 'volume'] = df['volume'].mean() * 4.0
    df.loc[climax_idx, 'open'] = df.loc[climax_idx, 'close'] * 0.99
    df.loc[climax_idx, 'close'] = df.loc[climax_idx, 'close'] * 1.03
    df.loc[climax_idx, 'high'] = df.loc[climax_idx, 'close'] * 1.04
    
    # 4. Add distribution pattern in final phase
    distribution_idx = int(n_periods * 0.85)  # During distribution
    df.loc[distribution_idx, 'volume'] = df['volume'].mean() * 3.2
    df.loc[distribution_idx, 'open'] = df.loc[distribution_idx, 'close'] * 1.01
    df.loc[distribution_idx, 'close'] = df.loc[distribution_idx, 'close'] * 0.995
    df.loc[distribution_idx, 'high'] = df.loc[distribution_idx, 'close'] * 1.03  # Long upper wick
    
    print(f"Generated {len(df)} periods of synthetic data with institutional patterns")
    return df

def analyze_real_order_flow(symbol="EURUSD=X", period="1y", interval="1d", save_charts=True):
    """
    Fetch real data and analyze order flow
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD=X" for EUR/USD)
        period: Time period to fetch (e.g., "1mo", "3mo", "1y", "5y")
        interval: Data interval (e.g., "1d", "1h", "15m")
        save_charts: Whether to save charts to files
        
    Returns:
        Dictionary with analysis results
    """
    # Fetch data
    data = fetch_real_historical_data(symbol, period, interval)
    
    if data is None or len(data) == 0:
        print("No data available for analysis")
        return None
    
    # Run order flow analysis
    print("Running enhanced order flow analysis...")
    order_flow_results = analyze_order_flow(data)
    
    # Run volume profile analysis
    print("Creating volume profile...")
    volume_profile_results = calculate_volume_profile(data)
    
    # Print key findings
    print("\nOrder Flow Analysis Results:")
    
    # Check if institutional activity was detected
    inst_activity = order_flow_results['institutional_activity']
    if inst_activity['present']:
        print(f"Institutional Activity: {inst_activity['type'].upper()} with {inst_activity['confidence']:.2f} confidence")
        print("\nDetected signals:")
        for signal in inst_activity['signals']:
            print(f"- {signal['type'].replace('_', ' ').title()}: {signal['description']}")
    else:
        print("No clear institutional activity detected")
    
    # Print key metrics
    print("\nKey Order Flow Metrics:")
    print(f"Buying Pressure: {order_flow_results['buying_pressure']:.4f}")
    print(f"Selling Pressure: {order_flow_results['selling_pressure']:.4f}")
    print(f"Imbalance: {order_flow_results['imbalance']:.4f} (positive = buying, negative = selling)")
    print(f"Recent Delta: {order_flow_results['delta']['recent']:.2f}")
    print(f"Delta Acceleration: {order_flow_results['delta']['acceleration']:.2f}")
    print(f"Cumulative Delta: {order_flow_results['delta']['cumulative']:.2f}")
    
    # Print volume profile key levels
    print("\nVolume Profile Key Levels:")
    print(f"Point of Control: {volume_profile_results.get('poc'):.5f}")
    print(f"Value Area High: {volume_profile_results.get('vah'):.5f}")
    print(f"Value Area Low: {volume_profile_results.get('val'):.5f}")
    
    # Print key institutional levels
    print("\nKey Institutional Levels:")
    for level in volume_profile_results.get('institutional_levels', []):
        level_type = level.get('type', 'unknown').replace('_', ' ').title()
        price = level.get('price', 0)
        strength = level.get('strength', 0)
        print(f"- {level_type}: {price:.5f} (Strength: {strength:.2f})")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Format symbol for filenames
    symbol_name = symbol.replace('=X', '').replace('^', '')
    
    # Order flow chart
    fig1 = plot_order_flow_analysis(data, order_flow_results, 
                                  title=f"{symbol_name} {interval} Order Flow Analysis")
    
    # Volume profile chart
    fig2 = plot_volume_profile(data, volume_profile_results)
    
    # Save or display charts
    if save_charts:
        filename1 = f"{symbol_name}_{interval}_order_flow.png"
        filename2 = f"{symbol_name}_{interval}_volume_profile.png"
        
        plt.figure(fig1.number)
        plt.savefig(filename1, dpi=300, bbox_inches='tight')
        
        plt.figure(fig2.number)
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        
        print(f"Charts saved as {filename1} and {filename2}")
    else:
        plt.figure(fig1.number)
        plt.show()
        
        plt.figure(fig2.number)
        plt.show()
    
    # Close figures to free memory
    plt.close(fig1)
    plt.close(fig2)
    
    return {
        'data': data,
        'order_flow': order_flow_results,
        'volume_profile': volume_profile_results
    }

def analyze_multiple_timeframes(symbol="EURUSD=X", save_charts=True):
    """
    Analyze order flow across multiple timeframes
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD=X" for EUR/USD)
        save_charts: Whether to save charts to files
        
    Returns:
        Dictionary with analysis results for each timeframe
    """
    timeframes = [
        {"period": "1mo", "interval": "15m"},
        {"period": "3mo", "interval": "1h"},
        {"period": "6mo", "interval": "1d"},
        {"period": "1y", "interval": "1wk"}
    ]
    
    results = {}
    strongest_tf = None
    max_confidence = 0
    
    print(f"Analyzing {symbol} across multiple timeframes...")
    
    for tf in timeframes:
        period = tf["period"]
        interval = tf["interval"]
        
        print(f"\n{'-'*50}")
        print(f"Analyzing {period} of {interval} data")
        print(f"{'-'*50}")
        
        # Run analysis for this timeframe
        result = analyze_real_order_flow(
            symbol=symbol,
            period=period,
            interval=interval,
            save_charts=save_charts
        )
        
        if result:
            # Store results
            key = f"{interval}_{period}"
            results[key] = result
            
            # Track strongest institutional signal
            confidence = result['order_flow']['institutional_activity'].get('confidence', 0)
            if result['order_flow']['institutional_activity'].get('present', False) and confidence > max_confidence:
                max_confidence = confidence
                strongest_tf = key
    
    # Print summary of institutional activity across timeframes
    print("\n\nInstitutional Activity Summary Across Timeframes:")
    print(f"{'-'*80}")
    print(f"{'Timeframe':<15} {'Activity':<10} {'Confidence':<10} {'Signals':<45}")
    print(f"{'-'*80}")
    
    for tf, result in results.items():
        inst = result['order_flow']['institutional_activity']
        if inst['present']:
            signals = ", ".join([s['type'].split('_')[0] for s in inst['signals'][:2]])
            if len(inst['signals']) > 2:
                signals += f" +{len(inst['signals'])-2} more"
            print(f"{tf:<15} {inst['type']:<10} {inst['confidence']:.2f}      {signals:<45}")
        else:
            print(f"{tf:<15} {'None':<10} {0.00:<10} {'No signals detected':<45}")
    
    if strongest_tf:
        print(f"\nStrongest institutional activity detected on {strongest_tf} timeframe")
        print(f"Confidence: {max_confidence:.2f}")
    else:
        print("\nNo significant institutional activity detected across timeframes")
    
    return results

def main():
    """Main function to run the analysis"""
    print("Real Order Flow Analysis")
    print("=" * 50)
    print("This script fetches real historical forex data and approximates")
    print("institutional order flow using volume profile analysis.")
    print("=" * 50)
    
    # List of available symbols with volume data
    symbols = [
        "EURUSD=X",  # EUR/USD
        "GBPUSD=X",  # GBP/USD
        "USDJPY=X",  # USD/JPY
        "AUDUSD=X",  # AUD/USD
        "USDCAD=X",  # USD/CAD
        "EURGBP=X"   # EUR/GBP
    ]
    
    print("\nAvailable symbols:")
    for i, symbol in enumerate(symbols, 1):
        print(f"{i}. {symbol[:-2]}")
    
    try:
        symbol_idx = int(input("\nSelect a symbol (1-6): ")) - 1
        if symbol_idx < 0 or symbol_idx >= len(symbols):
            symbol_idx = 0  # Default to EURUSD
    except:
        symbol_idx = 0  # Default to EURUSD
    
    symbol = symbols[symbol_idx]
    
    print("\nAnalysis options:")
    print("1. Single timeframe analysis")
    print("2. Multi-timeframe analysis")
    print("3. Use synthetic data with institutional patterns")
    
    try:
        option = int(input("\nSelect option (1-3): "))
    except:
        option = 1  # Default to single timeframe
    
    if option == 3:
        # Use synthetic data
        print("\nUsing synthetic data with institutional patterns...")
        data = generate_sample_data(symbol, "1d", "1y")
        order_flow_results = analyze_order_flow(data)
        volume_profile_results = calculate_volume_profile(data)
        
        # Create visualizations
        symbol_name = symbol.replace('=X', '')
        fig1 = plot_order_flow_analysis(data, order_flow_results, 
                                     title=f"{symbol_name} Synthetic Order Flow Analysis")
        fig2 = plot_volume_profile(data, volume_profile_results)
        
        # Save charts
        filename1 = f"{symbol_name}_synthetic_order_flow.png"
        filename2 = f"{symbol_name}_synthetic_volume_profile.png"
        
        plt.figure(fig1.number)
        plt.savefig(filename1, dpi=300, bbox_inches='tight')
        
        plt.figure(fig2.number)
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        
        print(f"\nCharts saved as {filename1} and {filename2}")
        
    elif option == 2:
        # Run multi-timeframe analysis
        analyze_multiple_timeframes(symbol=symbol, save_charts=True)
    else:
        # Specify period and interval
        periods = ["1mo", "3mo", "6mo", "1y", "2y"]
        intervals = ["15m", "1h", "1d", "1wk"]
        
        print("\nSelect period:")
        for i, period in enumerate(periods, 1):
            print(f"{i}. {period}")
            
        try:
            period_idx = int(input("\nSelect period (1-5): ")) - 1
            if period_idx < 0 or period_idx >= len(periods):
                period_idx = 3  # Default to 1y
        except:
            period_idx = 3  # Default to 1y
        
        print("\nSelect interval:")
        for i, interval in enumerate(intervals, 1):
            print(f"{i}. {interval}")
            
        try:
            interval_idx = int(input("\nSelect interval (1-4): ")) - 1
            if interval_idx < 0 or interval_idx >= len(intervals):
                interval_idx = 2  # Default to 1d
        except:
            interval_idx = 2  # Default to 1d
        
        # Run single timeframe analysis
        analyze_real_order_flow(
            symbol=symbol,
            period=periods[period_idx],
            interval=intervals[interval_idx],
            save_charts=True
        )
    
    print("\nAnalysis complete!")
    print("The visualization images have been saved to the current directory.")
    print("You can use these visualizations to identify potential institutional")
    print("buying and selling pressure in the market.")

if __name__ == "__main__":
    main()