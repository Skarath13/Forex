import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MAcrossoverStrategy:
    def __init__(self, short_period=50, long_period=200):
        """
        Initialize Moving Average Crossover Strategy.
        
        Parameters:
        -----------
        short_period : int
            Period for the short-term moving average
        long_period : int
            Period for the long-term moving average
        """
        self.short_period = short_period
        self.long_period = long_period
        self.positions = {}  # Track current positions per pair
        
    def calculate_moving_averages(self, df):
        """
        Calculate short and long-term SMAs.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Historical price data with at least 'Close' column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added SMA columns
        """
        df = df.copy()
        df['SMA_short'] = df['Close'].rolling(window=self.short_period).mean()
        df['SMA_long'] = df['Close'].rolling(window=self.long_period).mean()
        return df
    
    def check_for_signal(self, df, pair):
        """
        Check for MA crossover signals.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Historical price data with calculated MAs
        pair : str
            Currency pair being analyzed
            
        Returns:
        --------
        dict or None
            Signal dictionary with 'type' ('BUY' or 'SELL') and other details, or None
        """
        if len(df) < self.long_period + 1:
            logger.warning(f"Insufficient data for {pair}. Need at least {self.long_period + 1} bars.")
            return None
        
        # Get current and previous MA values
        current_short = df['SMA_short'].iloc[-1]
        current_long = df['SMA_long'].iloc[-1]
        prev_short = df['SMA_short'].iloc[-2]
        prev_long = df['SMA_long'].iloc[-2]
        
        # Check for NaN values
        if any(pd.isna([current_short, current_long, prev_short, prev_long])):
            logger.warning(f"MA values contain NaN for {pair}. Skipping signal check.")
            return None
        
        # Check for crossover
        signal = None
        
        # Bullish crossover - short MA crosses above long MA
        if prev_short <= prev_long and current_short > current_long:
            signal = {
                'type': 'BUY',
                'pair': pair,
                'short_ma': current_short,
                'long_ma': current_long,
                'price': df['Close'].iloc[-1],
                'timestamp': df['Timestamp'].iloc[-1]
            }
            logger.info(f"BUY signal detected for {pair} at price {signal['price']:.5f}")
        
        # Bearish crossover - short MA crosses below long MA
        elif prev_short >= prev_long and current_short < current_long:
            signal = {
                'type': 'SELL',
                'pair': pair,
                'short_ma': current_short,
                'long_ma': current_long,
                'price': df['Close'].iloc[-1],
                'timestamp': df['Timestamp'].iloc[-1]
            }
            logger.info(f"SELL signal detected for {pair} at price {signal['price']:.5f}")
        
        return signal
    
    def process_signal(self, signal, current_position=None):
        """
        Process the signal and determine action to take.
        
        Parameters:
        -----------
        signal : dict
            Signal dictionary from check_for_signal
        current_position : dict or None
            Current position info for the pair
            
        Returns:
        --------
        dict or None
            Action to take: {'action': 'OPEN'|'CLOSE'|'REVERSE', 'side': 'BUY'|'SELL', ...}
        """
        if signal is None:
            return None
        
        pair = signal['pair']
        signal_type = signal['type']
        
        # No current position - open new position
        if current_position is None:
            return {
                'action': 'OPEN',
                'side': signal_type,
                'pair': pair,
                'price': signal['price'],
                'timestamp': signal['timestamp']
            }
        
        # Current position exists
        current_side = current_position['side']
        
        # Opposite signal - close current and open new position
        if current_side != signal_type:
            return {
                'action': 'REVERSE',
                'close_side': current_side,
                'open_side': signal_type,
                'pair': pair,
                'price': signal['price'],
                'timestamp': signal['timestamp']
            }
        
        # Same direction signal - no action needed
        return None