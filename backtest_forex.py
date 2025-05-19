import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging
from utils import setup_logging
from advanced_strategy import InstitutionalForexStrategy
from backtest_system import BacktestEngine, Trade
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

def generate_forex_data(pair, start_date, end_date, timeframe='1h'):
    """Generate realistic forex historical data."""
    # Base prices for different pairs
    base_prices = {
        'EUR/USD': 1.08,
        'GBP/USD': 1.26,
        'USD/JPY': 150.0,
        'AUD/USD': 0.66,
        'USD/CAD': 1.36,
        'USD/CHF': 0.91,
        'NZD/USD': 0.61
    }
    
    base_price = base_prices.get(pair, 1.0)
    
    # Generate hourly timestamps
    current_time = start_date
    timestamps = []
    while current_time <= end_date:
        timestamps.append(current_time)
        current_time += timedelta(hours=1)
    
    # Create realistic price movement
    num_bars = len(timestamps)
    
    # Add multiple overlapping trends for realism
    # Long-term trend
    long_trend = 0.0001 * np.sin(2 * np.pi * np.arange(num_bars) / (num_bars / 4))
    
    # Medium-term cycles
    medium_trend = 0.00005 * np.sin(2 * np.pi * np.arange(num_bars) / (num_bars / 12))
    
    # Short-term volatility
    short_trend = 0.00002 * np.sin(2 * np.pi * np.arange(num_bars) / (num_bars / 48))
    
    # Random walk component
    random_walk = np.random.normal(0, 0.0001, num_bars)
    
    # Session-based volatility (higher during London/NY)
    session_volatility = []
    for ts in timestamps:
        hour = ts.hour
        if 8 <= hour <= 16 or 13 <= hour <= 21:  # London/NY hours
            volatility = 0.0002
        else:
            volatility = 0.00005
        session_volatility.append(np.random.normal(0, volatility))
    
    # Combine all components
    cumulative_return = np.cumsum(long_trend + medium_trend + short_trend + random_walk + session_volatility)
    prices = base_price * np.exp(cumulative_return)
    
    # Generate OHLCV data
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        # Add intrabar volatility
        volatility = 0.0001 if 8 <= timestamp.hour <= 21 else 0.00005
        
        open_price = price * (1 + np.random.normal(0, volatility))
        high_price = price * (1 + abs(np.random.normal(0, volatility * 2)))
        low_price = price * (1 - abs(np.random.normal(0, volatility * 2)))
        close_price = price
        
        # Make sure OHLC relationships are valid
        high_price = max(open_price, close_price, high_price)
        low_price = min(open_price, close_price, low_price)
        
        # Volume (higher during active sessions)
        base_volume = 10000
        if 8 <= timestamp.hour <= 16 or 13 <= timestamp.hour <= 21:
            volume = base_volume * np.random.uniform(1.5, 3.0)
        else:
            volume = base_volume * np.random.uniform(0.5, 1.5)
        
        data.append({
            'Timestamp': timestamp,
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': int(volume)
        })
    
    df = pd.DataFrame(data)
    df.set_index('Timestamp', inplace=True)
    return df

def run_backtest(pairs, start_date, end_date, initial_balance=10000):
    """Run backtest on multiple currency pairs."""
    logger.info(f"Starting backtest from {start_date} to {end_date}")
    
    # Initialize components
    strategy = InstitutionalForexStrategy()
    engine = BacktestEngine(initial_balance)
    
    # Store data for all pairs
    all_data = {}
    
    # Generate or fetch data for each pair
    for pair in pairs:
        logger.info(f"Generating data for {pair}")
        df = generate_forex_data(pair, start_date, end_date)
        
        # Calculate indicators and volume profile
        df = strategy.calculate_indicators(df)
        df, volume_profile = strategy.calculate_volume_profile(df)
        df = strategy.identify_market_structure(df)
        
        all_data[pair] = df
    
    # Simulate trading by iterating through time
    all_timestamps = sorted(set(ts for df in all_data.values() for ts in df.index))
    
    daily_trades = {}
    position_size = 0.02  # 2% of balance per trade
    
    for timestamp in all_timestamps:
        current_date = timestamp.date()
        
        # Reset daily trade counter
        if current_date not in daily_trades:
            daily_trades[current_date] = 0
        
        # Check if we've hit daily trade limit
        if daily_trades[current_date] >= strategy.config['max_daily_trades']:
            continue
        
        # Update equity curve
        engine.equity_curve.append(engine.balance)
        
        # Check open positions for exit conditions
        closed_positions = []
        for trade_id, position in engine.open_positions.items():
            pair = position['pair']
            if timestamp in all_data[pair].index:
                current_bar = all_data[pair].loc[timestamp]
                exit_condition = strategy.check_exit_conditions(position, current_bar)
                
                if exit_condition['exit']:
                    # Close position
                    trade = position['trade']
                    trade.exit_time = timestamp
                    trade.exit_price = exit_condition['price']
                    trade.exit_reason = exit_condition['reason']
                    
                    # Calculate P&L (in pips for forex)
                    pip_value = 0.0001 if 'JPY' not in pair else 0.01
                    
                    if trade.side == 'LONG':
                        price_diff = trade.exit_price - trade.entry_price
                    else:  # SHORT
                        price_diff = trade.entry_price - trade.exit_price
                    
                    pips = price_diff / pip_value
                    trade.pnl = pips * 10 * (trade.size / 100000)  # Standard lot calculation
                    trade.pnl_pct = price_diff / trade.entry_price if trade.entry_price > 0 else 0
                    
                    # Update balance
                    engine.balance += trade.pnl
                    
                    closed_positions.append(trade_id)
                    logger.info(f"Closed {trade.side} position on {pair} at {trade.exit_price:.5f}, "
                               f"P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2%})")
        
        # Remove closed positions
        for trade_id in closed_positions:
            del engine.open_positions[trade_id]
        
        # Check for new signals
        for pair in pairs:
            if timestamp in all_data[pair].index:
                # Skip if we already have a position in this pair
                if any(pos['pair'] == pair for pos in engine.open_positions.values()):
                    continue
                
                # Check daily trade limit
                if daily_trades[current_date] >= strategy.config['max_daily_trades']:
                    break
                
                current_bar = all_data[pair].loc[timestamp]
                signal = strategy.generate_signals(all_data[pair].loc[:timestamp])
                
                if signal:
                    # Calculate position size
                    risk_amount = engine.balance * strategy.config['risk_per_trade']
                    stop_distance = abs(signal['price'] - signal['stop_loss'])
                    
                    # For forex, typically trade in lots (100,000 units)
                    pip_value = 0.0001 if 'JPY' not in pair else 0.01
                    position_units = risk_amount / (stop_distance / pip_value) if stop_distance > 0 else 10000
                    position_units = int(max(1000, position_units / 1000) * 1000)  # Round to nearest 1000, minimum 1000
                    
                    # Create trade
                    trade = Trade(
                        pair=pair,
                        entry_time=timestamp,
                        entry_price=signal['price'],
                        side=signal['type'],
                        size=position_units,
                        entry_reason=signal['reason']
                    )
                    
                    # Store position
                    trade_id = f"{pair}_{timestamp}"
                    engine.open_positions[trade_id] = {
                        'pair': pair,
                        'entry_price': signal['price'],
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'side': signal['type'],
                        'trade': trade
                    }
                    
                    engine.trades.append(trade)
                    daily_trades[current_date] += 1
                    
                    logger.info(f"Opened {signal['type']} position on {pair} at {signal['price']:.5f}, "
                               f"SL: {signal['stop_loss']:.5f}, TP: {signal['take_profit']:.5f}")
    
    # Close any remaining open positions at the end
    for trade_id, position in engine.open_positions.items():
        trade = position['trade']
        trade.exit_time = end_date
        trade.exit_price = all_data[position['pair']].iloc[-1]['Close']
        trade.exit_reason = 'End of backtest'
        
        # Calculate P&L (in pips for forex)
        pip_value = 0.0001 if 'JPY' not in position['pair'] else 0.01
        
        if trade.side == 'LONG':
            price_diff = trade.exit_price - trade.entry_price
        else:
            price_diff = trade.entry_price - trade.exit_price
        
        pips = price_diff / pip_value
        trade.pnl = pips * 10 * (trade.size / 100000)  # Standard lot calculation
        trade.pnl_pct = price_diff / trade.entry_price if trade.entry_price > 0 else 0
        engine.balance += trade.pnl
    
    # Final equity curve update
    engine.equity_curve.append(engine.balance)
    
    return engine, all_data

def main():
    """Run the forex backtest."""
    # Setup logging
    logger = setup_logging()
    
    # Configuration
    pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD']
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    initial_balance = 10000
    
    # Run backtest
    logger.info("Starting forex backtest with volume profile strategy")
    engine, data = run_backtest(pairs, start_date, end_date, initial_balance)
    
    # Calculate and display metrics
    metrics = engine.calculate_metrics()
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Balance: ${engine.balance:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Average Win: ${metrics['avg_win']:.2f}")
    print(f"Average Loss: ${metrics['avg_loss']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Avg Trade Duration: {metrics['avg_trade_duration']:.1f} hours")
    print("="*50)
    
    # Trade breakdown by pair
    pair_stats = {}
    for trade in engine.trades:
        if trade.exit_time:
            pair = trade.pair
            if pair not in pair_stats:
                pair_stats[pair] = {'trades': 0, 'pnl': 0, 'wins': 0}
            pair_stats[pair]['trades'] += 1
            pair_stats[pair]['pnl'] += trade.pnl
            if trade.pnl > 0:
                pair_stats[pair]['wins'] += 1
    
    print("\nPerformance by Pair:")
    for pair, stats in pair_stats.items():
        win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
        print(f"{pair}: {stats['trades']} trades, "
              f"Win Rate: {win_rate:.1%}, "
              f"Total P&L: ${stats['pnl']:.2f}")
    
    # Plot results
    engine.plot_results('backtest_results.png')
    
    # Save detailed trade log
    trade_log = []
    for trade in engine.trades:
        if trade.exit_time:
            trade_log.append({
                'Pair': trade.pair,
                'Entry Time': trade.entry_time,
                'Exit Time': trade.exit_time,
                'Side': trade.side,
                'Entry Price': trade.entry_price,
                'Exit Price': trade.exit_price,
                'Size': trade.size,
                'P&L': trade.pnl,
                'P&L %': trade.pnl_pct,
                'Entry Reason': trade.entry_reason,
                'Exit Reason': trade.exit_reason
            })
    
    trade_df = pd.DataFrame(trade_log)
    trade_df.to_csv('trade_log.csv', index=False)
    logger.info("Trade log saved to trade_log.csv")
    
    # Plot sample volume profile
    sample_pair = 'EUR/USD'
    sample_data = data[sample_pair].tail(200)
    
    plt.figure(figsize=(15, 10))
    
    # Price chart
    plt.subplot(2, 1, 1)
    plt.plot(sample_data.index, sample_data['Close'], label='Close Price')
    plt.plot(sample_data.index, sample_data['MA_fast'], label='MA Fast')
    plt.plot(sample_data.index, sample_data['MA_slow'], label='MA Slow')
    plt.fill_between(sample_data.index, sample_data['VAL'], sample_data['VAH'], 
                     alpha=0.2, label='Value Area')
    plt.axhline(y=sample_data['POC'].iloc[-1], color='red', linestyle='--', label='POC')
    plt.title(f'{sample_pair} Price with Volume Profile Levels')
    plt.legend()
    plt.grid(True)
    
    # Volume
    plt.subplot(2, 1, 2)
    plt.bar(sample_data.index, sample_data['Volume'], alpha=0.7)
    plt.title('Volume')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('volume_profile_example.png', dpi=300)
    plt.close()
    
    logger.info("Backtest completed successfully")

if __name__ == "__main__":
    main()