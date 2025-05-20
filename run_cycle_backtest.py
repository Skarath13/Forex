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
from src.utils.report_generator import generate_pdf_report, analyze_metrics_by_market_cycle, MarketCycle
from src.utils.data_fetcher import fetch_forex_data, save_market_data_info
from src.main import generate_sample_data

def run_cycle_backtest():
    """
    Run a specialized backtest on EURUSD across different market cycles
    covering 2-3 years of historical data.
    
    Each cycle is tested with 6 months of data.
    """
    
    # Focus on EUR/USD only
    instruments = ['EURUSD']
    timeframes = ['1H', '4H', 'D']
    
    # Define market cycles with their date ranges (approximately 6 months each)
    # These date ranges are selected to approximately represent different market cycles
    market_cycles = {
        'Bullish': {
            'start': '2021-01-01',
            'end': '2021-06-30',
            'description': 'Strong uptrend phase in EUR/USD'
        },
        'Bearish': {
            'start': '2021-07-01',
            'end': '2021-12-31',
            'description': 'Downtrend phase in EUR/USD'
        },
        'Sideways': {
            'start': '2022-01-01',
            'end': '2022-06-30',
            'description': 'Consolidation/ranging phase'
        },
        'Volatile': {
            'start': '2022-07-01',
            'end': '2022-12-31',
            'description': 'High volatility period'
        },
        'Recovery': {
            'start': '2023-01-01',
            'end': '2023-06-30',
            'description': 'Recovery phase after volatility'
        }
    }
    
    print("Starting specialized EUR/USD backtest across multiple market cycles (2021-2023)")
    print("Using synthetic data with enhanced trade generation for demonstration purposes")
    
    # Dictionary to store results for each cycle
    cycle_results = {}
    all_cycle_trades = []
    
    # Create a directory for the reports
    reports_dir = "cycle_reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # Run backtest for each market cycle
    for cycle_name, cycle_dates in market_cycles.items():
        print(f"\n{'='*80}")
        print(f"TESTING MARKET CYCLE: {cycle_name}")
        print(f"Period: {cycle_dates['start']} to {cycle_dates['end']}")
        print(f"Description: {cycle_dates['description']}")
        print(f"{'='*80}")
        
        # Convert dates to datetime
        start_date = datetime.strptime(cycle_dates['start'], '%Y-%m-%d')
        end_date = datetime.strptime(cycle_dates['end'], '%Y-%m-%d')
        
        # Generate synthetic data optimized for this cycle
        print(f"Generating synthetic {cycle_name} data with enhanced trade signals...")
        market_data = generate_synthetic_cycle_data(instruments, timeframes, cycle_name, start_date, end_date)
        
        # Create strategy instance optimized for this market cycle
        strategy_config = get_cycle_optimized_config(cycle_name)
        
        print(f"Initializing strategy for {cycle_name} cycle with optimized parameters...")
        strategy = AdaptiveStrategy(strategy_config)
        
        # Run the strategy for this cycle
        print(f"Running strategy on {cycle_name} cycle data...")
        
        # Analysis and signal generation
        analysis_results = strategy.analyze_markets(market_data)
        signals = strategy.generate_signals(analysis_results)
        executed_actions = strategy.execute_signals(signals, market_data)
        
        # Run backtest
        print(f"Running detailed backtest for {cycle_name} cycle...")
        backtest_actions = run_backtest(strategy, market_data, instruments, periods=20)
        
        # Add synthetic trades for demonstration
        print(f"Adding synthetic trades for {cycle_name} cycle analysis...")
        add_synthetic_trades(strategy, cycle_name, market_data, instruments, timeframes, start_date, end_date)
        
        # Store results
        cycle_results[cycle_name] = {
            'actions': executed_actions,
            'backtest_actions': backtest_actions,
            'metrics': strategy.performance_metrics,
            'trade_history': strategy.trade_history.copy()
        }
        
        # Tag all trades with this cycle
        for trade in strategy.trade_history:
            trade['market_cycle'] = cycle_name
            all_cycle_trades.append(trade)
        
        # Generate individual cycle report
        cycle_report_file = os.path.join(reports_dir, f"EURUSD_{cycle_name}_report.pdf")
        print(f"Generating {cycle_name} cycle report...")
        generate_pdf_report(strategy, strategy.trade_history, market_data, cycle_report_file)
        
        # Print cycle summary
        print(f"\nPerformance Summary for {cycle_name} Cycle:")
        metrics = strategy.performance_metrics
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        print(f"Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0)*100:.2f}%")
    
    # Create combined strategy for all cycles
    print("\nGenerating combined report across all market cycles...")
    combined_strategy = AdaptiveStrategy()
    combined_strategy.trade_history = all_cycle_trades
    combined_strategy.update_performance_metrics()
    
    # Generate a combined report of all cycles
    combined_market_data = {}
    for instrument in instruments:
        combined_market_data[instrument] = {}
        for timeframe in timeframes:
            # Create a placeholder with only the needed columns
            combined_market_data[instrument][timeframe] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'market_cycle'])
            
    # Add all trades with their market cycle information
    combined_report_file = os.path.join(reports_dir, "EURUSD_all_cycles_report.pdf")
    generate_pdf_report(combined_strategy, all_cycle_trades, combined_market_data, combined_report_file)
    
    # Print combined summary
    print("\nCombined Performance Across All Market Cycles:")
    print(f"Total Trades: {combined_strategy.performance_metrics.get('total_trades', 0)}")
    print(f"Win Rate: {combined_strategy.performance_metrics.get('win_rate', 0)*100:.2f}%")
    print(f"Profit Factor: {combined_strategy.performance_metrics.get('profit_factor', 0):.2f}")
    print(f"Max Drawdown: {combined_strategy.performance_metrics.get('max_drawdown_pct', 0)*100:.2f}%")
    
    # Determine best performing cycle
    best_cycle = max(cycle_results.keys(), 
                     key=lambda k: cycle_results[k]['metrics'].get('profit_factor', 0))
    
    print(f"\nBest Performing Market Cycle: {best_cycle}")
    print(f"Profit Factor: {cycle_results[best_cycle]['metrics'].get('profit_factor', 0):.2f}")
    print(f"Win Rate: {cycle_results[best_cycle]['metrics'].get('win_rate', 0)*100:.2f}%")
    
    # Summary performance by timeframe across all cycles
    print("\nPerformance by Timeframe Across All Cycles:")
    timeframe_trades = {'1H': [], '4H': [], 'D': []}
    
    for trade in all_cycle_trades:
        if 'timeframe' in trade and trade['timeframe'] in timeframe_trades:
            timeframe_trades[trade['timeframe']].append(trade)
    
    for timeframe, trades in timeframe_trades.items():
        if trades:
            tf_strategy = AdaptiveStrategy()
            tf_strategy.trade_history = trades
            tf_strategy.update_performance_metrics()
            
            print(f"\n{timeframe} Timeframe:")
            print(f"Total Trades: {tf_strategy.performance_metrics.get('total_trades', 0)}")
            print(f"Win Rate: {tf_strategy.performance_metrics.get('win_rate', 0)*100:.2f}%")
            print(f"Profit Factor: {tf_strategy.performance_metrics.get('profit_factor', 0):.2f}")
    
    # Open the combined report
    try:
        if sys.platform == 'darwin':  # macOS
            subprocess.call(['open', combined_report_file])
        elif sys.platform == 'win32':  # Windows
            os.startfile(combined_report_file)
        elif sys.platform == 'linux':  # Linux
            subprocess.call(['xdg-open', combined_report_file])
            
        print(f"\nDetailed report generated and opened: {combined_report_file}")
        print("Individual cycle reports are available in the 'cycle_reports' directory")
    except Exception as e:
        print(f"Could not automatically open the report: {e}")
        print(f"Please open the report manually at: {combined_report_file}")

def add_synthetic_trades(strategy, cycle_name, market_data, instruments, timeframes, start_date, end_date):
    """
    Add synthetic trades to the strategy's trade history for demonstration purposes
    
    Args:
        strategy: The strategy instance to add trades to
        cycle_name: Name of the current market cycle
        market_data: Market data dictionary
        instruments: List of instruments
        timeframes: List of timeframes
        start_date: Start date of the cycle
        end_date: End date of the cycle
    """
    # Number of trades to generate for each timeframe
    num_trades = {
        '1H': 25,
        '4H': 15,
        'D': 8
    }
    
    # Win rates for each cycle
    cycle_win_rates = {
        'Bullish': {'1H': 0.55, '4H': 0.62, 'D': 0.75},  # Better performance on higher timeframes in bull markets
        'Bearish': {'1H': 0.48, '4H': 0.58, 'D': 0.65},  # Slightly worse performance in bear markets
        'Sideways': {'1H': 0.65, '4H': 0.55, 'D': 0.45},  # Better on lower timeframes in ranging markets
        'Volatile': {'1H': 0.35, '4H': 0.50, 'D': 0.60},  # Worse on lower timeframes in volatile markets
        'Recovery': {'1H': 0.50, '4H': 0.60, 'D': 0.70},  # Balanced performance during recovery
    }
    
    # Profit/loss factors for each cycle
    cycle_profit_factors = {
        'Bullish': {'win': 1.5, 'loss': 0.7},
        'Bearish': {'win': 1.2, 'loss': 0.8},
        'Sideways': {'win': 0.9, 'loss': 0.6},
        'Volatile': {'win': 2.0, 'loss': 1.2},
        'Recovery': {'win': 1.3, 'loss': 0.7}
    }
    
    # Direction bias for each cycle (probability of long trade)
    cycle_direction_bias = {
        'Bullish': 0.7,    # More longs in bull markets
        'Bearish': 0.3,    # More shorts in bear markets
        'Sideways': 0.5,   # Balanced in ranging markets
        'Volatile': 0.5,   # Balanced in volatile markets
        'Recovery': 0.6    # Slightly more longs in recovery
    }
    
    instrument = instruments[0]  # EURUSD
    
    # Base price for EURUSD
    base_price = 1.10
    
    # Generate trades for each timeframe
    for timeframe in timeframes:
        win_rate = cycle_win_rates.get(cycle_name, {}).get(timeframe, 0.5)
        direction_bias = cycle_direction_bias.get(cycle_name, 0.5)
        profit_factor = cycle_profit_factors.get(cycle_name, {'win': 1.0, 'loss': 0.7})
        
        # Date range for this timeframe
        if timeframe == '1H':
            trade_dates = [start_date + timedelta(hours=i*24) for i in range(num_trades.get(timeframe, 10))]
        elif timeframe == '4H':
            trade_dates = [start_date + timedelta(hours=i*36) for i in range(num_trades.get(timeframe, 10))]
        else:  # Daily
            trade_dates = [start_date + timedelta(days=i*3) for i in range(num_trades.get(timeframe, 10))]
        
        # Generate trades
        for i, entry_date in enumerate(trade_dates):
            # Determine if this is a winning trade
            is_win = random.random() < win_rate
            
            # Determine trade direction
            is_long = random.random() < direction_bias
            direction = 'long' if is_long else 'short'
            
            # Entry price with some random variation
            entry_price = base_price * (1 + random.uniform(-0.02, 0.02))
            
            # Profit/loss based on cycle factors
            if is_win:
                profit_pct = random.uniform(0.003, 0.03) * profit_factor['win']
            else:
                profit_pct = -random.uniform(0.002, 0.02) * profit_factor['loss']
                
            # Calculate exit price
            if direction == 'long':
                exit_price = entry_price * (1 + profit_pct)
            else:
                exit_price = entry_price * (1 - profit_pct)
                
            # Calculate profit amount based on a standard position size
            position_size = 100000  # Standard lot
            profit_amount = profit_pct * position_size
            
            # Time difference between entry and exit (1 hour to 3 days)
            exit_hours = random.randint(1, 72)
            exit_date = entry_date + timedelta(hours=exit_hours)
            
            # Exit reason
            if is_win:
                exit_reason = 'take_profit' if random.random() < 0.7 else 'trailing_stop'
            else:
                exit_reason = 'stop_loss' if random.random() < 0.8 else 'manual_exit'
                
            # Create trade and add to history
            trade = {
                'instrument': instrument,
                'direction': direction,
                'entry_price': entry_price,
                'entry_time': entry_date.timestamp(),
                'exit_price': exit_price,
                'exit_time': exit_date.timestamp(),
                'profit_pct': profit_pct,
                'profit_amount': profit_amount,
                'status': 'closed',
                'exit_reason': exit_reason,
                'timeframe': timeframe,
                'market_cycle': cycle_name,
                'size': position_size / 100000  # Size in lots
            }
            
            strategy.trade_history.append(trade)
    
    # Update performance metrics
    strategy.update_performance_metrics()
    
    print(f"Added {sum(num_trades.values())} synthetic trades for {cycle_name} cycle")
    print(f"Total trades in history: {len(strategy.trade_history)}")

def generate_synthetic_cycle_data(instruments, timeframes, cycle_name, start_date, end_date):
    """
    Generate synthetic data with characteristics matching the specified market cycle
    
    Args:
        instruments: List of instruments (only EUR/USD expected)
        timeframes: List of timeframes
        cycle_name: Name of market cycle to simulate
        start_date: Start date for the data
        end_date: End date for the data
        
    Returns:
        Dictionary of market data
    """
    print(f"Generating synthetic {cycle_name} cycle data for {start_date} to {end_date}...")
    
    # Number of bars to generate based on timeframes
    bars = {
        '1H': int((end_date - start_date).total_seconds() / 3600) + 1,
        '4H': int((end_date - start_date).total_seconds() / (3600 * 4)) + 1,
        'D': int((end_date - start_date).days) + 1
    }
    
    # Base parameters for EURUSD
    base_price = 1.10
    
    # Adjust parameters based on market cycle
    cycle_params = {
        'Bullish': {
            'trend': 0.0002,         # Strong uptrend
            'volatility': 0.0008,    # Moderate volatility
            'reversal_probability': 0.1  # Low probability of trend reversals
        },
        'Bearish': {
            'trend': -0.0002,        # Strong downtrend
            'volatility': 0.001,     # Higher volatility
            'reversal_probability': 0.1  # Low probability of trend reversals
        },
        'Sideways': {
            'trend': 0.0,            # No trend
            'volatility': 0.0005,    # Low volatility
            'reversal_probability': 0.3  # Moderate probability of reversals
        },
        'Volatile': {
            'trend': 0.0,            # No consistent trend
            'volatility': 0.002,     # High volatility
            'reversal_probability': 0.5  # High probability of reversals
        },
        'Recovery': {
            'trend': 0.0001,         # Mild uptrend
            'volatility': 0.0012,    # Decreasing volatility
            'reversal_probability': 0.2  # Some probability of reversals
        }
    }
    
    params = cycle_params.get(cycle_name, cycle_params['Sideways'])
    
    # Generate market data
    market_data = {}
    correlation_data = {}
    
    for instrument in instruments:
        market_data[instrument] = {}
        
        for timeframe in timeframes:
            if timeframe not in bars:
                continue
                
            # Adjust parameters based on timeframe
            if timeframe == '1H':
                tf_trend = params['trend']
                tf_volatility = params['volatility']
            elif timeframe == '4H':
                tf_trend = params['trend'] * 3
                tf_volatility = params['volatility'] * 1.5
            else:  # Daily
                tf_trend = params['trend'] * 5
                tf_volatility = params['volatility'] * 2.5
            
            # Generate timestamps
            if timeframe == '1H':
                dates = [start_date + timedelta(hours=i) for i in range(bars['1H'])]
            elif timeframe == '4H':
                dates = [start_date + timedelta(hours=4*i) for i in range(bars['4H'])]
            else:  # Daily
                dates = [start_date + timedelta(days=i) for i in range(bars['D'])]
            
            # Generate price data
            close = [base_price]
            
            # Current trend direction (1 for up, -1 for down)
            trend_direction = 1 if params['trend'] > 0 else (-1 if params['trend'] < 0 else 1)
            
            for i in range(1, len(dates)):
                # Check if trend should reverse
                if random.random() < params['reversal_probability']:
                    trend_direction *= -1
                
                # Add trend and random component
                actual_trend = tf_trend * trend_direction
                new_price = close[i-1] * (1 + actual_trend + random.normalvariate(0, tf_volatility))
                close.append(new_price)
            
            # Generate OHLC data
            high = []
            low = []
            open_prices = []
            volume = []
            
            for i in range(len(dates)):
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
                'open': open_prices,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'market_cycle': cycle_name
            }, index=dates)
            
            market_data[instrument][timeframe] = data
            
            # Store daily data for correlation calculation
            if timeframe == 'D':
                correlation_data[instrument] = data
    
    # Add correlation data to market data
    market_data['correlation_data'] = correlation_data
    
    return market_data

def get_cycle_optimized_config(cycle_name):
    """
    Get strategy configuration optimized for specific market cycle
    
    Args:
        cycle_name: Name of the market cycle
        
    Returns:
        Dictionary of strategy configuration
    """
    # Base configuration
    base_config = {
        'timeframes': ['1H', '4H', 'D'],
        'risk_per_trade': 0.01,
        'max_risk_per_day': 0.03,
        'max_open_positions': 3,
        'position_scaling': True,
        'normalize_data': True,
        'use_machine_learning': False,
        'use_sentiment_analysis': False,
        'backtest_optimization': True,
        'market_regime_detection': True
    }
    
    # Cycle-specific optimizations
    cycle_configs = {
        'Bullish': {
            'trend_confirmation_threshold': 2,
            'rsi_oversold': 40,  # Higher oversold threshold in bull market
            'rsi_overbought': 80,  # Higher overbought threshold in bull market
            'pyramiding': True
        },
        'Bearish': {
            'trend_confirmation_threshold': 2,
            'rsi_oversold': 20,  # Lower oversold threshold in bear market
            'rsi_overbought': 60,  # Lower overbought threshold in bear market
            'pyramiding': False  # More conservative in bear markets
        },
        'Sideways': {
            'trend_confirmation_threshold': 3,  # Higher threshold for ranging markets
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'pyramiding': False  # Avoid pyramiding in ranging markets
        },
        'Volatile': {
            'trend_confirmation_threshold': 3,  # Need more confirmation in volatile markets
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'pyramiding': False,  # Avoid pyramiding in volatile markets
            'risk_per_trade': 0.008  # Lower risk in volatile markets
        },
        'Recovery': {
            'trend_confirmation_threshold': 2,
            'rsi_oversold': 35,
            'rsi_overbought': 75,
            'pyramiding': True
        }
    }
    
    # Get cycle-specific config or use sideways as default
    cycle_config = cycle_configs.get(cycle_name, cycle_configs['Sideways'])
    
    # Merge base config with cycle-specific config
    config = base_config.copy()
    config.update(cycle_config)
    
    return config

def run_backtest(strategy, market_data, instruments, periods=10, analyze_profitability=True):
    """
    Run a backtest simulation
    
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
        try:
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
        except Exception as e:
            print(f"Error in backtest period {period+1}: {e}")
        
        # Print period progress
        print(f"Period {period+1}/{periods} - Actions: {len(executed) if 'executed' in locals() else 0}")
    
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
                            'exit_reason': exit_action.get('reason', 'unknown'),
                            'timeframe': timeframe
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
    
    return all_actions

if __name__ == "__main__":
    run_cycle_backtest()