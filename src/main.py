import numpy as np
import pandas as pd
from src.strategies.adaptive_strategy import AdaptiveStrategy, MarketRegime, TradeDirection
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import random

def generate_sample_data(instruments, timeframes, bars=500):
    """
    Generate sample OHLCV data for testing
    
    Args:
        instruments: List of instruments to generate data for
        timeframes: List of timeframes
        bars: Number of bars to generate
        
    Returns:
        Dictionary of market data
    """
    market_data = {}
    correlation_data = {}
    
    # Generate data for each instrument
    for instrument in instruments:
        market_data[instrument] = {}
        
        # Set instrument-specific parameters
        volatility = random.uniform(0.0005, 0.002)  # Base volatility
        trend = random.uniform(-0.0002, 0.0002)     # Trend component
        
        # Generate data for each timeframe
        for timeframe in timeframes:
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
            
            # Generate timestamps
            if timeframe == '1H':
                start_date = datetime.now() - timedelta(hours=bars)
                dates = [start_date + timedelta(hours=i) for i in range(bars)]
            elif timeframe == '4H':
                start_date = datetime.now() - timedelta(hours=4*bars)
                dates = [start_date + timedelta(hours=4*i) for i in range(bars)]
            elif timeframe == 'D':
                start_date = datetime.now() - timedelta(days=bars)
                dates = [start_date + timedelta(days=i) for i in range(bars)]
            else:
                start_date = datetime.now() - timedelta(hours=bars)
                dates = [start_date + timedelta(hours=i) for i in range(bars)]
            
            # Generate price data using random walk with drift
            close = [random.uniform(1.0, 2.0)]  # Starting price
            
            for i in range(1, bars):
                # Add trend and random component
                new_price = close[i-1] * (1 + tf_trend + random.normalvariate(0, tf_volatility))
                close.append(new_price)
            
            # Generate OHLC data
            high = []
            low = []
            open_prices = []
            volume = []
            
            for i in range(bars):
                if i == 0:
                    open_price = close[i] * (1 - random.uniform(0, tf_volatility))
                else:
                    open_price = close[i-1]
                
                high_price = max(open_price, close[i]) * (1 + random.uniform(0, tf_volatility))
                low_price = min(open_price, close[i]) * (1 - random.uniform(0, tf_volatility))
                
                open_prices.append(open_price)
                high.append(high_price)
                low.append(low_price)
                volume.append(random.uniform(100, 1000))
            
            # Create DataFrame
            data = pd.DataFrame({
                'date': dates,
                'open': open_prices,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
            
            data.set_index('date', inplace=True)
            
            market_data[instrument][timeframe] = data
            
            # Store daily data for correlation calculation
            if timeframe == 'D':
                correlation_data[instrument] = data
    
    # Add correlation data to market data
    market_data['correlation_data'] = correlation_data
    
    return market_data

def plot_strategy_results(market_data, signals, executed_actions, instruments, timeframe):
    """
    Plot strategy results
    
    Args:
        market_data: Dictionary of market data
        signals: Dictionary of signals from strategy
        executed_actions: List of executed actions
        instruments: List of instruments to plot
        timeframe: Timeframe to plot
    """
    for instrument in instruments:
        if instrument not in market_data or timeframe not in market_data[instrument]:
            continue
            
        data = market_data[instrument][timeframe]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot price
        ax.plot(data.index, data['close'], label='Close Price', color='blue', alpha=0.7)
        
        # Plot signals
        if instrument in signals and signals[instrument]['direction'] != TradeDirection.NEUTRAL:
            signal_data = signals[instrument]
            signal_time = data.index[-1]
            
            if signal_data['direction'] == TradeDirection.LONG:
                ax.scatter(signal_time, signal_data['entry_price'], marker='^', color='green', s=100, label='Buy Signal')
                
                # Plot stop loss and take profit
                if signal_data['stop_loss'] is not None:
                    ax.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.5, label='Stop Loss')
                    
                if signal_data['take_profit'] is not None:
                    ax.axhline(y=signal_data['take_profit'], color='green', linestyle='--', alpha=0.5, label='Take Profit')
                    
            elif signal_data['direction'] == TradeDirection.SHORT:
                ax.scatter(signal_time, signal_data['entry_price'], marker='v', color='red', s=100, label='Sell Signal')
                
                # Plot stop loss and take profit
                if signal_data['stop_loss'] is not None:
                    ax.axhline(y=signal_data['stop_loss'], color='red', linestyle='--', alpha=0.5, label='Stop Loss')
                    
                if signal_data['take_profit'] is not None:
                    ax.axhline(y=signal_data['take_profit'], color='green', linestyle='--', alpha=0.5, label='Take Profit')
        
        # Plot executed actions
        for action in executed_actions:
            if action['type'] == 'entry' and action['position']['instrument'] == instrument:
                action_time = data.index[min(len(data) - 1, max(0, len(data) - 10))]  # For visualization purposes
                
                if action['position']['direction'] == 'long':
                    ax.scatter(action_time, action['position']['entry_price'], marker='^', color='green', s=200, edgecolors='black', label='Entry Long')
                else:
                    ax.scatter(action_time, action['position']['entry_price'], marker='v', color='red', s=200, edgecolors='black', label='Entry Short')
            
            elif action['type'] == 'exit' and action['instrument'] == instrument:
                action_time = data.index[min(len(data) - 1, max(0, len(data) - 5))]  # For visualization purposes
                
                ax.scatter(action_time, action['exit_price'], marker='o', color='black', s=200, label=f"Exit ({action['reason']})")
        
        # Set labels and title
        ax.set_title(f"{instrument} - {timeframe} Timeframe")
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        
        # Format x-axis to show dates nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.xticks(rotation=45)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Show plot
        plt.tight_layout()
        plt.show()

def run_backtest(strategy, market_data, instruments, periods=10):
    """
    Run a simple backtest simulation
    
    Args:
        strategy: Strategy instance
        market_data: Dictionary of market data
        instruments: List of instruments to trade
        periods: Number of periods to simulate
        
    Returns:
        List of all executed actions
    """
    all_actions = []
    
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
                
                period_data[instrument][timeframe] = data.iloc[period_start:period_end]
        
        # Add correlation data
        if 'correlation_data' in backtest_data:
            period_data['correlation_data'] = {
                k: backtest_data['correlation_data'][k].iloc[:-(periods-period-1)] 
                for k in backtest_data['correlation_data']
            }
        
        # Run strategy on this period's data
        analysis = strategy.analyze_markets(period_data)
        signals = strategy.generate_signals(analysis)
        executed = strategy.execute_signals(signals, period_data)
        
        # Add to overall actions
        all_actions.extend(executed)
        
        # Update performance metrics
        strategy.update_performance_metrics()
        
        # Print period progress
        print(f"Period {period+1}/{periods} - Actions: {len(executed)}")
    
    # Print final performance metrics
    print("\nFinal Performance Metrics:")
    metrics = strategy.performance_metrics
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    return all_actions

def main():
    """Main function to run the strategy"""
    # Define instruments and timeframes
    instruments = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    timeframes = ['1H', '4H', 'D']
    
    # Generate sample data
    print("Generating sample market data...")
    market_data = generate_sample_data(instruments, timeframes)
    
    # Create strategy instance
    print("Initializing strategy...")
    strategy = AdaptiveStrategy()
    
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
    
    # Plot results
    print("\nPlotting results...")
    plot_strategy_results(market_data, signals, executed_actions, instruments, '4H')
    
    # Run a backtest simulation
    print("\nRunning backtest simulation...")
    backtest_actions = run_backtest(strategy, market_data, instruments, periods=10)
    
    print(f"\nTotal backtest actions: {len(backtest_actions)}")

if __name__ == "__main__":
    main()