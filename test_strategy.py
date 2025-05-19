import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.strategies.adaptive_strategy import AdaptiveStrategy, MarketRegime, TradeDirection
from src.utils.data_generator import generate_test_market_data

def run_strategy_test():
    """
    Run a test of the adaptive strategy with simulated data
    """
    print("Generating test market data...")
    market_data = generate_test_market_data(
        instruments=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
        timeframes=['1H', '4H', 'D'],
        days=60
    )
    
    print(f"Generated data for {len(market_data)-1} instruments across {len(market_data['EURUSD'])} timeframes")
    
    # Initialize strategy with custom configuration
    config = {
        'risk_per_trade': 0.01,  # 1% risk per trade
        'max_risk_per_day': 0.03,  # 3% max daily risk
        'max_open_positions': 3,
        'timeframes': ['1H', '4H', 'D'],
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'avoid_news': True,
        'use_market_hours': True,
        'position_scaling': True,
        'pyramiding': True
    }
    
    strategy = AdaptiveStrategy(config)
    
    print("\nAnalyzing markets...")
    analysis = strategy.analyze_markets(market_data)
    
    # Check market regimes detected
    regimes = {}
    for instrument in analysis:
        if instrument != 'correlation_data':
            regimes[instrument] = strategy.current_regime
    
    print("\nDetected Market Regimes:")
    for instrument, regime in regimes.items():
        print(f"{instrument}: {regime.name}")
    
    print("\nGenerating signals...")
    signals = strategy.generate_signals(analysis)
    
    # Print signal details
    print("\nGenerated Signals:")
    for instrument, signal in signals.items():
        if signal['direction'] != TradeDirection.NEUTRAL:
            direction = "LONG" if signal['direction'] == TradeDirection.LONG else "SHORT"
            print(f"{instrument}: {direction} signal with strength {signal['strength']:.2f}")
            
            if signal['entry_price'] is not None:
                print(f"  Entry: {signal['entry_price']:.5f}")
                print(f"  Stop Loss: {signal['stop_loss']:.5f}")
                print(f"  Take Profit: {signal['take_profit']:.5f}")
                if 'risk_reward' in signal:
                    print(f"  Risk-Reward: {signal['risk_reward']:.2f}")
            
            print(f"  Reasons: {', '.join(signal['reason'])}")
    
    print("\nExecuting signals...")
    actions = strategy.execute_signals(signals, market_data)
    
    # Print executed actions
    print("\nExecuted Actions:")
    for action in actions:
        if action['type'] == 'entry':
            print(f"Entered {action['position']['direction']} position on {action['position']['instrument']}")
            print(f"  Entry Price: {action['position']['entry_price']:.5f}")
            print(f"  Position Size: {action['position']['size']}")
            print(f"  Stop Loss: {action['position']['stop_loss']:.5f}")
            print(f"  Take Profit: {action['position']['take_profit']:.5f}")
        elif action['type'] == 'exit':
            print(f"Exited position on {action['instrument']} at {action['exit_price']:.5f}")
            print(f"  Reason: {action['reason']}")
            print(f"  P&L: {action['profit_pct']*100:.2f}%")
    
    # Test parameter optimization
    print("\nOptimizing strategy parameters...")
    parameter_ranges = {
        'rsi_period': [7, 14, 21],
        'macd_fast': [8, 12, 16],
        'macd_slow': [21, 26, 34]
    }
    
    strategy.optimize_parameters(market_data, parameter_ranges, metric='profit_factor')
    
    print("\nOptimized Parameters:")
    print(f"  RSI Period: {strategy.config['rsi_period']}")
    print(f"  MACD Fast: {strategy.config['macd_fast']}")
    print(f"  MACD Slow: {strategy.config['macd_slow']}")
    
    # Update performance metrics
    strategy.update_performance_metrics()
    
    print("\nStrategy Performance Metrics:")
    metrics = strategy.performance_metrics
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    run_strategy_test()