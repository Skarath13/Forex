#!/usr/bin/env python3
"""
Example script demonstrating how to use the enhanced order flow analysis
to identify institutional buying/selling pressure in market data.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.market_analysis import analyze_order_flow, calculate_volume_profile
from src.utils.visualization import plot_order_flow_analysis, plot_volume_profile
from src.utils.data_generator import generate_test_market_data

def generate_sample_data_with_patterns():
    """
    Generate sample data with specific institutional order flow patterns
    """
    print("Generating sample data with institutional patterns...")
    
    # Create a base dataframe with 100 periods
    n_periods = 100
    dates = [datetime.now() - timedelta(days=n_periods-i) for i in range(n_periods)]
    
    # Initialize with some base price
    base_price = 1.2000
    close_prices = [base_price]
    
    # Create a price series with specific patterns
    for i in range(1, n_periods):
        # Create different market phases to test institutional patterns
        if i < 20:  # Initial uptrend
            new_price = close_prices[i-1] * (1 + 0.002 + np.random.normal(0, 0.003))
        elif i < 40:  # Downtrend
            new_price = close_prices[i-1] * (1 - 0.002 + np.random.normal(0, 0.003))
        elif i < 60:  # Accumulation (sideways with absorption)
            new_price = close_prices[i-1] * (1 + np.random.normal(0, 0.002))
        elif i < 80:  # Strong uptrend
            new_price = close_prices[i-1] * (1 + 0.003 + np.random.normal(0, 0.002))
        else:  # Distribution and topping
            new_price = close_prices[i-1] * (1 - 0.0005 + np.random.normal(0, 0.003))
        
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
        
        # Create price range based on phase
        if i < 20:  # Initial uptrend - moderate range
            range_factor = 1.0
        elif i < 40:  # Downtrend - larger range
            range_factor = 1.5
        elif i < 60:  # Accumulation - smaller range
            range_factor = 0.7
        elif i < 80:  # Strong uptrend - moderate range
            range_factor = 1.2
        else:  # Distribution - increasing range
            range_factor = 1.3
        
        price_range = abs(close_prices[i] - open_price) * range_factor
        
        # Create high and low prices
        if open_price > close_prices[i]:  # Bearish candle
            high_price = open_price + price_range * 0.3
            low_price = close_prices[i] - price_range * 0.7
        else:  # Bullish candle
            high_price = close_prices[i] + price_range * 0.7
            low_price = open_price - price_range * 0.3
        
        # Add randomness to high/low
        high_price += np.random.uniform(0, price_range * 0.2)
        low_price -= np.random.uniform(0, price_range * 0.2)
        
        # Store values
        open_prices.append(open_price)
        high_prices.append(high_price)
        low_prices.append(low_price)
        
        # Generate volume with patterns
        base_vol = 1000
        if i < 20:  # Initial uptrend - moderate, increasing volume
            vol = base_vol * (0.8 + i * 0.01)
        elif i < 40:  # Downtrend - high volume
            vol = base_vol * 1.5
        elif i < 60:  # Accumulation - declining volume, spikes on key bars
            vol = base_vol * (1.0 - (i - 40) * 0.01)
        elif i < 80:  # Strong uptrend - high increasing volume
            vol = base_vol * (1.0 + (i - 60) * 0.03)
        else:  # Distribution - varying volume with spikes
            vol = base_vol * (1.5 - (i - 80) * 0.02)
        
        # Add randomness
        vol *= np.random.uniform(0.8, 1.2)
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
    # 1. Absorption pattern (high volume, small range) during accumulation
    absorption_idx = 50
    df.loc[absorption_idx, 'volume'] = df['volume'].mean() * 3.5
    mid_price = (df.loc[absorption_idx, 'open'] + df.loc[absorption_idx, 'close']) / 2
    df.loc[absorption_idx, 'open'] = mid_price * 0.998
    df.loc[absorption_idx, 'close'] = mid_price * 1.001
    df.loc[absorption_idx, 'high'] = mid_price * 1.005
    df.loc[absorption_idx, 'low'] = mid_price * 0.995
    
    # 2. Stopping volume at the end of downtrend (day 38)
    stopping_idx = 38
    df.loc[stopping_idx, 'volume'] = df['volume'].mean() * 3.0
    df.loc[stopping_idx, 'open'] = df.loc[stopping_idx, 'close'] * 0.99
    df.loc[stopping_idx, 'close'] = df.loc[stopping_idx, 'close'] * 1.002
    df.loc[stopping_idx, 'low'] = df.loc[stopping_idx, 'close'] * 0.96  # Long lower wick
    
    # 3. Climax volume at end of uptrend (day 78)
    climax_idx = 78
    df.loc[climax_idx, 'volume'] = df['volume'].mean() * 4.0
    df.loc[climax_idx, 'open'] = df.loc[climax_idx, 'close'] * 0.99
    df.loc[climax_idx, 'close'] = df.loc[climax_idx, 'close'] * 1.03
    df.loc[climax_idx, 'high'] = df.loc[climax_idx, 'close'] * 1.04
    
    # 4. Distribution pattern (bearish stopping volume) in distribution phase
    distribution_idx = 90
    df.loc[distribution_idx, 'volume'] = df['volume'].mean() * 3.2
    df.loc[distribution_idx, 'open'] = df.loc[distribution_idx, 'close'] * 1.01
    df.loc[distribution_idx, 'close'] = df.loc[distribution_idx, 'close'] * 0.995
    df.loc[distribution_idx, 'high'] = df.loc[distribution_idx, 'close'] * 1.03  # Long upper wick
    
    print(f"Generated {len(df)} periods of sample data with institutional patterns")
    return df

def analyze_real_instrument():
    """
    Analyze order flow for a real instrument using the test data generator
    """
    print("Analyzing order flow for EURUSD...")
    
    # Generate test market data for EURUSD
    instruments = ['EURUSD']
    timeframes = ['1H', '4H', 'D']
    market_data = generate_test_market_data(instruments, timeframes, days=100)
    
    # Get the daily data
    df = market_data['EURUSD']['D']
    
    # Run order flow analysis
    order_flow_results = analyze_order_flow(df)
    volume_profile_results = calculate_volume_profile(df)
    
    # Plot results
    fig1 = plot_order_flow_analysis(df, order_flow_results, 
                                   title="EURUSD Daily Order Flow Analysis")
    fig2 = plot_volume_profile(df, volume_profile_results)
    
    # Show plots
    plt.figure(fig1.number)
    plt.savefig('eurusd_order_flow.png')
    
    plt.figure(fig2.number)
    plt.savefig('eurusd_volume_profile.png')
    
    # Print key findings
    print("\nEURUSD Order Flow Analysis Results:")
    print(f"Buying Pressure: {order_flow_results['buying_pressure']:.4f}")
    print(f"Selling Pressure: {order_flow_results['selling_pressure']:.4f}")
    print(f"Imbalance: {order_flow_results['imbalance']:.4f} (positive = buying, negative = selling)")
    
    # Print institutional activity
    inst_activity = order_flow_results['institutional_activity']
    if inst_activity['present']:
        print(f"\nInstitutional Activity Detected:")
        print(f"Type: {inst_activity['type']}, Confidence: {inst_activity['confidence']:.2f}")
        print("\nDetected signals:")
        for signal in inst_activity['signals']:
            print(f"- {signal['type'].replace('_', ' ').title()}: {signal['strength']:.2f}")
    else:
        print("\nNo clear institutional activity detected")
    
    print(f"\nOrder flow charts saved as eurusd_order_flow.png and eurusd_volume_profile.png")
    
    return {'data': df, 'order_flow': order_flow_results, 'volume_profile': volume_profile_results}

def run_multi_timeframe_analysis():
    """
    Analyze order flow across multiple timeframes to detect institutional activity
    """
    print("Running multi-timeframe order flow analysis...")
    
    # Generate test market data
    instruments = ['EURUSD']
    timeframes = ['1H', '4H', 'D', 'W']
    market_data = generate_test_market_data(instruments, timeframes, days=200)
    
    # Analyze order flow for each timeframe
    results = {}
    for timeframe in timeframes:
        df = market_data['EURUSD'][timeframe]
        results[timeframe] = {
            'order_flow': analyze_order_flow(df),
            'volume_profile': calculate_volume_profile(df)
        }
    
    # Print a summary of institutional activity across timeframes
    print("\nInstitutional Activity Summary Across Timeframes:")
    print("-" * 50)
    print(f"{'Timeframe':<10} {'Activity':<10} {'Confidence':<10} {'Signals':<30}")
    print("-" * 50)
    
    for tf, result in results.items():
        inst = result['order_flow']['institutional_activity']
        if inst['present']:
            signals = ", ".join([s['type'].split('_')[0] for s in inst['signals'][:2]])
            if len(inst['signals']) > 2:
                signals += f" +{len(inst['signals'])-2} more"
            print(f"{tf:<10} {inst['type']:<10} {inst['confidence']:.2f}      {signals:<30}")
        else:
            print(f"{tf:<10} {'None':<10} {0.00:<10} {'No signals detected':<30}")
    
    # Identify strongest institutional activity
    strongest_tf = max(results.keys(), 
                      key=lambda x: results[x]['order_flow']['institutional_activity'].get('confidence', 0) 
                      if results[x]['order_flow']['institutional_activity'].get('present', False) else 0)
    
    strongest_conf = results[strongest_tf]['order_flow']['institutional_activity'].get('confidence', 0)
    if strongest_conf > 0:
        print(f"\nStrongest institutional activity detected on {strongest_tf} timeframe")
        print(f"Confidence: {strongest_conf:.2f}")
        print("Signals detected:")
        for signal in results[strongest_tf]['order_flow']['institutional_activity']['signals']:
            print(f"- {signal['type'].replace('_', ' ').title()}: {signal['description']}")
            
        # Plot the order flow for the strongest timeframe
        df = market_data['EURUSD'][strongest_tf]
        fig = plot_order_flow_analysis(df, results[strongest_tf]['order_flow'],
                                     title=f"EURUSD {strongest_tf} Order Flow - Strongest Institutional Activity")
        plt.savefig(f'eurusd_{strongest_tf}_order_flow.png')
        print(f"\nOrder flow chart for strongest timeframe saved as eurusd_{strongest_tf}_order_flow.png")
    
    return results

if __name__ == "__main__":
    print("Order Flow Analysis Example")
    print("=" * 50)
    print("This example demonstrates how to use the enhanced order flow analysis")
    print("to identify institutional buying/selling pressure in market data.")
    print("=" * 50)
    
    # Choose which example to run
    choice = input("\nChoose an example to run:\n"
                  "1. Custom sample data with institutional patterns\n"
                  "2. Real instrument analysis (EURUSD)\n"
                  "3. Multi-timeframe analysis\n"
                  "Enter your choice (1-3): ")
    
    if choice == '1':
        df = generate_sample_data_with_patterns()
        order_flow_results = analyze_order_flow(df)
        volume_profile_results = calculate_volume_profile(df)
        
        # Plot results
        fig1 = plot_order_flow_analysis(df, order_flow_results, 
                                      title="Sample Data with Institutional Patterns")
        fig2 = plot_volume_profile(df, volume_profile_results)
        
        plt.figure(fig1.number)
        plt.show()
        
        plt.figure(fig2.number)
        plt.show()
        
    elif choice == '2':
        analyze_real_instrument()
        
    elif choice == '3':
        run_multi_timeframe_analysis()
        
    else:
        print("Invalid choice. Please run the script again and choose 1, 2, or 3.")