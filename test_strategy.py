import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.strategies.adaptive_strategy import AdaptiveStrategy, MarketRegime, TradeDirection
from src.utils.data_generator import generate_test_market_data
from src.utils.market_analysis import analyze_order_flow, calculate_volume_profile
from src.utils.visualization import plot_order_flow_analysis, plot_volume_profile

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

def test_order_flow_analysis():
    """
    Test the enhanced order flow analysis functionality
    """
    print("\nTesting enhanced order flow analysis...")
    
    # Generate test data for a single instrument
    market_data = generate_test_market_data(
        instruments=['EURUSD'],
        timeframes=['1H', '4H', 'D'],
        days=60
    )
    
    # Get daily data for analysis
    daily_data = market_data['EURUSD']['D']
    
    # Run order flow analysis
    order_flow_results = analyze_order_flow(daily_data)
    volume_profile_results = calculate_volume_profile(daily_data)
    
    # Check for institutional activity
    inst_activity = order_flow_results['institutional_activity']
    print("\nOrder Flow Analysis Results:")
    
    if inst_activity['present']:
        print(f"Institutional Activity: {inst_activity['type'].upper()} with {inst_activity['confidence']:.2f} confidence")
        print("\nDetected patterns:")
        for signal in inst_activity['signals']:
            print(f"- {signal['type'].replace('_', ' ').title()}: {signal['description']}")
    else:
        print("No clear institutional activity detected")
    
    # Print key metrics
    print("\nKey Metrics:")
    print(f"Buying Pressure: {order_flow_results['buying_pressure']:.4f}")
    print(f"Selling Pressure: {order_flow_results['selling_pressure']:.4f}")
    print(f"Imbalance: {order_flow_results['imbalance']:.4f} (positive = buying, negative = selling)")
    print(f"Recent Delta: {order_flow_results['delta']['recent']:.2f}")
    print(f"Cumulative Delta: {order_flow_results['delta']['cumulative']:.2f}")
    
    # Create visualizations
    fig1 = plot_order_flow_analysis(daily_data, order_flow_results, 
                                  title="EURUSD Daily Order Flow Analysis")
    fig2 = plot_volume_profile(daily_data, volume_profile_results)
    
    plt.figure(fig1.number)
    plt.savefig('order_flow_analysis_test.png')
    plt.close(fig1)
    
    plt.figure(fig2.number)
    plt.savefig('volume_profile_test.png')
    plt.close(fig2)
    
    print("\nOrder flow visualizations saved to 'order_flow_analysis_test.png' and 'volume_profile_test.png'")
    print("Enhanced order flow analysis test complete!")

if __name__ == "__main__":
    # Choose which test to run
    print("Choose a test to run:")
    print("1. Run strategy test")
    print("2. Test order flow analysis")
    
    choice = input("Enter choice (1-2): ")
    
    if choice == '2':
        test_order_flow_analysis()
    else:
        run_strategy_test()