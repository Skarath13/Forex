import configparser
import time
import logging
import signal
import sys
from datetime import datetime

from utils import setup_logging
from data_handler import fetch_historical_data, fetch_live_price
from strategy import MAcrossoverStrategy
from broker_interface import BrokerInterface
from risk_manager import RiskManager

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(sig, frame):
    """Handle Ctrl+C for graceful shutdown."""
    global shutdown_flag
    print("\nShutdown signal received. Stopping bot gracefully...")
    shutdown_flag = True

def load_config(config_file='config.ini'):
    """Load configuration from file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def main():
    """Main execution loop for the Forex trading bot."""
    # Set up signal handler for graceful shutdown
    import signal as sig
    sig.signal(sig.SIGINT, signal_handler)
    
    # Initialize logging
    logger = setup_logging()
    logger.info("Starting PyFxTrader_SimpleMA v0.1")
    
    # Load configuration
    config = load_config()
    
    # Parse configuration values
    trading_pairs = [pair.strip() for pair in config['DEFAULT']['TRADING_PAIRS'].split(',')]
    timeframe = config['DEFAULT']['TIMEFRAME']
    short_ma_period = int(config['DEFAULT']['SHORT_MA_PERIOD'])
    long_ma_period = int(config['DEFAULT']['LONG_MA_PERIOD'])
    position_size = int(config['DEFAULT']['POSITION_SIZE'])
    loop_interval = int(config['DEFAULT']['LOOP_INTERVAL_SECONDS'])
    
    logger.info(f"Configuration loaded:")
    logger.info(f"  Trading pairs: {trading_pairs}")
    logger.info(f"  Timeframe: {timeframe}")
    logger.info(f"  MA periods: {short_ma_period}/{long_ma_period}")
    logger.info(f"  Position size: {position_size}")
    logger.info(f"  Loop interval: {loop_interval}s")
    
    # Initialize components
    strategy = MAcrossoverStrategy(short_period=short_ma_period, long_period=long_ma_period)
    broker = BrokerInterface()
    risk_manager = RiskManager(position_size=position_size)
    
    # Connect to broker
    if not broker.connect_broker():
        logger.error("Failed to connect to broker. Exiting.")
        return
    
    # Store positions per pair
    positions = {}
    
    # Main trading loop
    loop_count = 0
    while not shutdown_flag:
        loop_count += 1
        logger.info(f"\n--- Trading Loop #{loop_count} ---")
        logger.info(f"Time: {datetime.now()}")
        logger.info(f"Account Balance: ${broker.get_account_balance():.2f}")
        
        try:
            # Process each trading pair
            for pair in trading_pairs:
                logger.info(f"\nProcessing {pair}...")
                
                # Fetch historical data
                df = fetch_historical_data(pair, timeframe, count=max(short_ma_period, long_ma_period) + 50)
                
                # Calculate indicators
                df = strategy.calculate_moving_averages(df)
                
                # Check for signals
                signal = strategy.check_for_signal(df, pair)
                
                # Get current position for this pair
                current_position = positions.get(pair)
                
                # Process signal
                action = strategy.process_signal(signal, current_position)
                
                if action:
                    # Check risk limits before trading
                    if risk_manager.check_risk_limits(broker.get_open_trades(), broker.get_account_balance()):
                        
                        if action['action'] == 'OPEN':
                            # Calculate position size
                            size = risk_manager.calculate_position_size(
                                pair, 
                                broker.get_account_balance()
                            )
                            
                            # Place order
                            trade = broker.place_order(
                                pair=pair,
                                units=size,
                                side=action['side']
                            )
                            
                            if trade:
                                positions[pair] = {
                                    'trade_id': trade['trade_id'],
                                    'side': action['side'],
                                    'entry_price': trade['open_price'],
                                    'units': size
                                }
                        
                        elif action['action'] == 'REVERSE':
                            # Close existing position
                            if current_position:
                                broker.close_trade(current_position['trade_id'])
                                
                            # Open new position in opposite direction
                            size = risk_manager.calculate_position_size(
                                pair,
                                broker.get_account_balance()
                            )
                            
                            trade = broker.place_order(
                                pair=pair,
                                units=size,
                                side=action['open_side']
                            )
                            
                            if trade:
                                positions[pair] = {
                                    'trade_id': trade['trade_id'],
                                    'side': action['open_side'],
                                    'entry_price': trade['open_price'],
                                    'units': size
                                }
                            else:
                                # Remove closed position even if new one failed
                                positions.pop(pair, None)
                    else:
                        logger.warning(f"Risk limits exceeded. Skipping trade for {pair}")
                
                # Small delay between pairs
                time.sleep(1)
            
            # Display current positions
            logger.info("\nCurrent positions:")
            if positions:
                for pair, pos in positions.items():
                    logger.info(f"  {pair}: {pos['side']} {pos['units']} units @ {pos['entry_price']:.5f}")
            else:
                logger.info("  No open positions")
            
            # Sleep until next iteration
            if not shutdown_flag:
                logger.info(f"\nSleeping for {loop_interval} seconds until next check...")
                time.sleep(loop_interval)
                
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(60)  # Sleep 1 minute on error before retrying
    
    # Cleanup on shutdown
    logger.info("\nShutting down PyFxTrader_SimpleMA...")
    
    # Close all positions (optional for v0.1)
    if positions:
        logger.info("Closing all open positions...")
        for pair, pos in positions.items():
            try:
                broker.close_trade(pos['trade_id'])
            except Exception as e:
                logger.error(f"Error closing position for {pair}: {e}")
    
    logger.info("Shutdown complete. Goodbye!")

if __name__ == "__main__":
    main()