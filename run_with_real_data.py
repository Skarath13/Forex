#!/usr/bin/env python3

import numpy as np
import pandas as pd
from src.strategies.adaptive_strategy import AdaptiveStrategy, MarketRegime, TradeDirection
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode to prevent figures from popping up
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import random
import os
import subprocess
import sys
import time
from src.utils.report_generator import generate_pdf_report, analyze_metrics_by_market_cycle
from src.utils.data_fetcher import fetch_forex_data, save_market_data_info

def run_with_real_data():
    """Run the forex trading strategy with real historical data"""
    
    # Define instruments and timeframes
    instruments = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    timeframes = ['1H', '4H', 'D']
    
    # Set date range - adjust as needed
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months of data
    
    print(f"Fetching real historical forex data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    try:
        # Fetch real market data
        market_data = fetch_forex_data(instruments, timeframes, start_date, end_date)
        
        # Save market data info
        market_info = save_market_data_info(market_data, "forex_data_info.json")
        
        # Verify data quality
        data_valid = True
        for instrument in instruments:
            for timeframe in timeframes:
                if instrument not in market_data or timeframe not in market_data[instrument]:
                    print(f"Error: Missing data for {instrument} {timeframe}")
                    data_valid = False
                    continue
                    
                data = market_data[instrument][timeframe]
                
                # Check for numeric columns
                if not all(pd.api.types.is_numeric_dtype(data[col]) for col in ['open', 'high', 'low', 'close'] if col in data.columns):
                    print(f"Error: Non-numeric data in {instrument} {timeframe}")
                    data_valid = False
        
        if not data_valid:
            print("\nSome data was invalid or missing. Falling back to sample data.")
            print("This is common with free API limitations. For production use, consider a paid data provider.")
            
            # Generate sample data with same parameters
            from src.main import generate_sample_data
            market_data = generate_sample_data(instruments, timeframes, bars=1000)
            market_info = save_market_data_info(market_data, "forex_sample_data_info.json")
            
    except Exception as e:
        print(f"Error fetching real data: {e}")
        print("Falling back to sample data...")
        
        # Generate sample data
        from src.main import generate_sample_data
        market_data = generate_sample_data(instruments, timeframes, bars=1000)
        market_info = save_market_data_info(market_data, "forex_sample_data_info.json")
    
    print("\nMarket Data Summary:")
    for instrument in market_info['instruments']:
        print(f"\n{instrument}:")
        for timeframe in market_info['timeframes']:
            bars = market_info['bars_count'][instrument][timeframe]
            date_range = market_info['date_range'][instrument][timeframe]
            price_range = market_info['price_range'][instrument][timeframe]
            print(f"  {timeframe}: {bars} bars from {date_range['start']} to {date_range['end']}")
            print(f"    Price range: {price_range['min']:.5f} - {price_range['max']:.5f} (avg: {price_range['avg']:.5f})")
    
    # Create strategy instance with optimal settings
    print("\nInitializing strategy...")
    strategy_config = {
        'trend_confirmation_threshold': 2,  # Require at least 2 indicators to confirm trend
        'risk_per_trade': 0.01,            # 1% risk per trade
        'max_open_positions': 4,           # Maximum 4 positions at once
        'rsi_oversold': 30,                # Standard RSI settings
        'rsi_overbought': 70,
        'pyramiding': True,                # Allow adding to positions
        'normalize_data': True,            # Enable data normalization
        'use_machine_learning': False,     # Disable ML features for backtesting speed
        'use_sentiment_analysis': False,   # Disable sentiment analysis for backtesting
        'backtest_optimization': True,     # Enable backtest optimization
        'market_regime_detection': True    # Enable market regime detection
    }
    strategy = AdaptiveStrategy(strategy_config)
    
    # Run analysis on current data
    print("Analyzing markets...")
    analysis_results = strategy.analyze_markets(market_data)
    
    # Generate signals
    print("Generating signals...")
    signals = strategy.generate_signals(analysis_results)
    
    # Execute signals
    print("Executing signals...")
    executed_actions = strategy.execute_signals(signals, market_data)
    
    # Print signals and actions
    print("\nGenerated Signals:")
    for instrument, signal in signals.items():
        if signal['direction'] != TradeDirection.NEUTRAL:
            print(f"{instrument}: {signal['direction']} with strength {signal['strength']:.2f}")
            print(f"  Entry: {signal['entry_price']:.5f}, Stop: {signal['stop_loss']:.5f}, Target: {signal['take_profit']:.5f}")
            print(f"  Reason: {', '.join(signal['reason'])}")
    
    print("\nExecuted Actions:")
    for action in executed_actions:
        if action['type'] == 'entry':
            print(f"Entry {action['position']['direction']} on {action['position']['instrument']} at {action['position']['entry_price']:.5f}")
        elif action['type'] == 'exit':
            print(f"Exit {action['instrument']} at {action['exit_price']:.5f} ({action['reason']})")
    
    # Run a backtest simulation
    print("\nRunning backtest simulation...")
    backtest_periods = 20
    backtest_actions = run_backtest(strategy, market_data, instruments, periods=backtest_periods)
    
    print(f"\nTotal backtest actions: {len(backtest_actions)}")
    
    # Generate detailed PDF report
    print("\nGenerating detailed performance report...")
    report_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forex_real_data_report.pdf")
    
    # Generate the PDF report
    generate_pdf_report(strategy, strategy.trade_history, market_data, report_file)
    
    print(f"\nDetailed report generated: {report_file}")
    
    # Open the PDF report if possible
    try:
        if sys.platform == 'darwin':  # macOS
            subprocess.call(['open', report_file])
        elif sys.platform == 'win32':  # Windows
            os.startfile(report_file)
        elif sys.platform == 'linux':  # Linux
            subprocess.call(['xdg-open', report_file])
    except Exception as e:
        print(f"Could not automatically open the report: {e}")
        print(f"Please open the report manually at: {report_file}")

def run_backtest(strategy, market_data, instruments, periods=10, analyze_profitability=True):
    """
    Run a simple backtest simulation
    
    Args:
        strategy: Strategy instance
        market_data: Dictionary of market data
        instruments: List of instruments to trade
        periods: Number of periods to simulate
        analyze_profitability: Whether to analyze profitability by timeframe
        
    Returns:
        List of all executed actions
    """
    all_actions = []
    timeframe_actions = {'1H': [], '4H': [], 'D': []}
    timeframe_metrics = {}
    
    # Clone market data to avoid modifying original
    backtest_data = {
        k: v.copy() if not isinstance(v, dict) else {
            k2: v2.copy() for k2, v2 in v.items()
        } for k, v in market_data.items()
    }
    
    # For each period, we'll process a subset of the data
    for period in range(periods):
        period_data = {}
        
        # Prepare data for this period
        for instrument in instruments:
            period_data[instrument] = {}
            
            for timeframe in backtest_data[instrument]:
                data = backtest_data[instrument][timeframe]
                
                # Get data up to current period
                period_end = len(data) - (periods - period - 1)
                period_start = max(0, period_end - 100)  # Use 100 bars for each analysis
                
                if period_end <= period_start:
                    continue
                    
                period_data[instrument][timeframe] = data.iloc[period_start:period_end]
        
        # Add correlation data
        if 'correlation_data' in backtest_data:
            period_data['correlation_data'] = {
                k: backtest_data['correlation_data'][k].iloc[:-(periods-period-1)] 
                for k in backtest_data['correlation_data']
                if k in backtest_data['correlation_data'] and len(backtest_data['correlation_data'][k]) > (periods-period-1)
            }
        
        # Run strategy on this period's data
        analysis = strategy.analyze_markets(period_data)
        signals = strategy.generate_signals(analysis)
        executed = strategy.execute_signals(signals, period_data)
        
        # Add to overall actions
        all_actions.extend(executed)
        
        # Track actions by timeframe
        for action in executed:
            if action['type'] == 'entry':
                timeframe = action['position'].get('timeframe')
                if timeframe in timeframe_actions:
                    timeframe_actions[timeframe].append(action)
        
        # Update performance metrics
        strategy.update_performance_metrics()
        
        # Print period progress
        print(f"Period {period+1}/{periods} - Actions: {len(executed)}")
    
    # Print final performance metrics
    print("\nFinal Performance Metrics:")
    metrics = strategy.performance_metrics
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Analyze profitability by timeframe
    if analyze_profitability:
        print("\nProfitability Analysis by Timeframe:")
        
        # Create separate strategy instances for each timeframe
        for timeframe in timeframe_actions:
            if timeframe_actions[timeframe]:
                temp_strategy = AdaptiveStrategy()
                # Add actions to trade history
                for action in timeframe_actions[timeframe]:
                    entry_price = action['position']['entry_price']
                    instrument = action['position']['instrument']
                    direction = action['position']['direction']
                    
                    # Find corresponding exit action
                    exit_action = None
                    for potential_exit in all_actions:
                        if (potential_exit['type'] == 'exit' and
                            potential_exit['instrument'] == instrument and
                            'exit_price' in potential_exit):
                            exit_action = potential_exit
                            break
                    
                    # Add to trade history
                    if exit_action:
                        exit_price = exit_action['exit_price']
                        if direction == 'long':
                            profit_pct = (exit_price - entry_price) / entry_price
                        else:
                            profit_pct = (entry_price - exit_price) / entry_price
                            
                        temp_strategy.trade_history.append({
                            'instrument': instrument,
                            'direction': direction,
                            'entry_price': entry_price,
                            'entry_time': action['position']['entry_time'] if 'entry_time' in action['position'] else 0,
                            'exit_price': exit_price,
                            'exit_time': exit_action.get('exit_time', 0),
                            'profit_pct': profit_pct,
                            'profit_amount': profit_pct * action['position'].get('size', 1.0),
                            'status': 'closed',
                            'exit_reason': exit_action.get('reason', 'unknown')
                        })
                
                # Update metrics
                temp_strategy.update_performance_metrics()
                timeframe_metrics[timeframe] = temp_strategy.performance_metrics
                
                # Print metrics for this timeframe
                print(f"\n{timeframe} Timeframe:")
                print(f"Total trades: {timeframe_metrics[timeframe].get('total_trades', 0)}")
                print(f"Win rate: {timeframe_metrics[timeframe].get('win_rate', 0):.2f}")
                print(f"Profit factor: {timeframe_metrics[timeframe].get('profit_factor', 0):.2f}")
                print(f"Average win: {timeframe_metrics[timeframe].get('average_win', 0):.4f}")
                print(f"Average loss: {timeframe_metrics[timeframe].get('average_loss', 0):.4f}")
            else:
                print(f"\n{timeframe} Timeframe: No trades executed")
        
        # Determine most profitable timeframe
        best_timeframe = None
        best_profit_factor = 0
        
        for timeframe, metrics in timeframe_metrics.items():
            profit_factor = metrics.get('profit_factor', 0)
            if profit_factor > best_profit_factor:
                best_profit_factor = profit_factor
                best_timeframe = timeframe
        
        if best_timeframe:
            print(f"\nMost profitable timeframe: {best_timeframe} with profit factor {best_profit_factor:.2f}")
        else:
            print("\nNo profitable timeframe found in this backtest run")
    
    return all_actions

if __name__ == "__main__":
    run_with_real_data()