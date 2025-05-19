import pandas as pd
import numpy as np
from datetime import datetime, time
import logging

logger = logging.getLogger(__name__)

class InstitutionalForexStrategy:
    def __init__(self, config=None):
        """Initialize advanced forex strategy with institutional concepts."""
        self.config = config or self.get_default_config()
        
    def get_default_config(self):
        return {
            'volume_profile_lookback': 120,  # bars for volume profile
            'value_area_pct': 0.70,  # 70% of volume for value area
            'ma_fast': 20,
            'ma_slow': 50,
            'atr_period': 14,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'session_filter': True,
            'london_open': time(8, 0),   # London session
            'london_close': time(16, 0),
            'ny_open': time(13, 0),      # New York session
            'ny_close': time(21, 0),
            'min_volume_threshold': 0.7,  # Volume must be 70% of average
            'risk_per_trade': 0.01,       # 1% risk per trade
            'risk_reward_ratio': 2.0,     # 1:2 risk reward
            'max_daily_trades': 3,        # Institution discipline
            'trend_strength_threshold': 0.6
        }
    
    def calculate_volume_profile(self, df, lookback_periods=None):
        """Calculate volume profile with POC, VAH, VAL."""
        lookback = lookback_periods or self.config['volume_profile_lookback']
        
        if len(df) < lookback:
            return df
        
        # Get recent data
        recent_data = df.tail(lookback).copy()
        
        # Create price bins
        price_min = recent_data['Low'].min()
        price_max = recent_data['High'].max()
        n_bins = 50  # Number of price levels
        
        price_bins = np.linspace(price_min, price_max, n_bins)
        
        # Calculate volume at each price level
        volume_profile = {}
        
        for idx, row in recent_data.iterrows():
            # Distribute volume across the candle's range
            candle_prices = np.linspace(row['Low'], row['High'], 10)
            volume_per_level = row['Volume'] / len(candle_prices)
            
            for price in candle_prices:
                # Find the closest price bin
                closest_bin = price_bins[np.argmin(np.abs(price_bins - price))]
                if closest_bin not in volume_profile:
                    volume_profile[closest_bin] = 0
                volume_profile[closest_bin] += volume_per_level
        
        # Sort by volume to find POC and value area
        sorted_profile = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        
        # Find Point of Control (highest volume price)
        poc = sorted_profile[0][0] if sorted_profile else 0
        
        # Calculate Value Area (70% of total volume)
        total_volume = sum(v for _, v in sorted_profile)
        value_area_volume = total_volume * self.config['value_area_pct']
        
        accumulated_volume = 0
        value_area_prices = []
        
        for price, volume in sorted_profile:
            accumulated_volume += volume
            value_area_prices.append(price)
            if accumulated_volume >= value_area_volume:
                break
        
        vah = max(value_area_prices) if value_area_prices else price_max
        val = min(value_area_prices) if value_area_prices else price_min
        
        # Add to dataframe
        df['POC'] = poc
        df['VAH'] = vah
        df['VAL'] = val
        
        # Identify high/low volume nodes
        avg_volume = np.mean(list(volume_profile.values()))
        high_volume_nodes = [p for p, v in volume_profile.items() if v > avg_volume * 1.5]
        low_volume_nodes = [p for p, v in volume_profile.items() if v < avg_volume * 0.5]
        
        df['nearest_hvn'] = df['Close'].apply(
            lambda x: min(high_volume_nodes, key=lambda p: abs(p - x)) if high_volume_nodes else np.nan
        )
        df['nearest_lvn'] = df['Close'].apply(
            lambda x: min(low_volume_nodes, key=lambda p: abs(p - x)) if low_volume_nodes else np.nan
        )
        
        return df, volume_profile
    
    def identify_market_structure(self, df):
        """Identify key market structure levels."""
        if len(df) < 50:
            return df
        
        # Find swing highs and lows
        df['swing_high'] = df['High'].rolling(window=5, center=True).max() == df['High']
        df['swing_low'] = df['Low'].rolling(window=5, center=True).min() == df['Low']
        
        # Identify support and resistance levels
        swing_highs = df[df['swing_high']]['High'].tail(10)
        swing_lows = df[df['swing_low']]['Low'].tail(10)
        
        # Cluster nearby levels
        resistance_levels = self._cluster_levels(swing_highs.values)
        support_levels = self._cluster_levels(swing_lows.values)
        
        # Find nearest levels to current price
        current_price = df['Close'].iloc[-1]
        
        df['nearest_resistance'] = min(resistance_levels, key=lambda x: abs(x - current_price)) if resistance_levels else np.nan
        df['nearest_support'] = min(support_levels, key=lambda x: abs(x - current_price)) if support_levels else np.nan
        
        # Calculate trend strength using linear regression
        prices = df['Close'].tail(20).values
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # Normalize slope to get trend strength
        price_range = df['High'].tail(20).max() - df['Low'].tail(20).min()
        df['trend_strength'] = slope / price_range if price_range > 0 else 0
        
        return df
    
    def _cluster_levels(self, levels, threshold=0.0002):
        """Cluster nearby price levels."""
        if len(levels) == 0:
            return []
        
        sorted_levels = sorted(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clusters.append(np.mean(current_cluster))
        return clusters
    
    def calculate_indicators(self, df):
        """Calculate technical indicators."""
        # Moving averages
        df['MA_fast'] = df['Close'].rolling(self.config['ma_fast']).mean()
        df['MA_slow'] = df['Close'].rolling(self.config['ma_slow']).mean()
        
        # ATR for volatility
        high_low = df['High'] - df['Low']
        high_prev_close = df['High'] - df['Close'].shift(1)
        low_prev_close = df['Low'] - df['Close'].shift(1)
        
        true_range = pd.concat([high_low, high_prev_close.abs(), low_prev_close.abs()], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(self.config['atr_period']).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.config['rsi_period']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume analysis
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price action patterns
        df['bullish_engulfing'] = ((df['Close'] > df['Open']) & 
                                  (df['Close'].shift(1) < df['Open'].shift(1)) &
                                  (df['Close'] > df['Open'].shift(1)) &
                                  (df['Open'] < df['Close'].shift(1)))
        
        df['bearish_engulfing'] = ((df['Close'] < df['Open']) & 
                                  (df['Close'].shift(1) > df['Open'].shift(1)) &
                                  (df['Close'] < df['Open'].shift(1)) &
                                  (df['Open'] > df['Close'].shift(1)))
        
        return df
    
    def check_session_filter(self, timestamp):
        """Check if current time is within active trading sessions."""
        if not self.config['session_filter']:
            return True
        
        current_time = timestamp.time()
        
        # London session
        if self.config['london_open'] <= current_time <= self.config['london_close']:
            return True
        
        # New York session
        if self.config['ny_open'] <= current_time <= self.config['ny_close']:
            return True
        
        # London-NY overlap (most liquid)
        if (self.config['ny_open'] <= current_time <= self.config['london_close']):
            return True
        
        return False
    
    def generate_signals(self, df):
        """Generate trading signals based on institutional concepts."""
        if len(df) < 200:
            return None
        
        current_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]
        
        # Check session filter
        if not self.check_session_filter(current_bar.name):
            return None
        
        # Basic filters
        if pd.isna(current_bar['MA_slow']) or pd.isna(current_bar['ATR']):
            return None
        
        # Volume filter - need significant volume
        if current_bar['Volume_ratio'] < self.config['min_volume_threshold']:
            return None
        
        signal = None
        
        # Long signals
        if all([
            # Trend alignment
            current_bar['MA_fast'] > current_bar['MA_slow'],
            prev_bar['MA_fast'] <= prev_bar['MA_slow'],  # MA crossover
            
            # Price near value area low or support
            (abs(current_bar['Close'] - current_bar['VAL']) / current_bar['Close'] < 0.001 or
             abs(current_bar['Close'] - current_bar['nearest_support']) / current_bar['Close'] < 0.001),
            
            # RSI not overbought
            current_bar['RSI'] < self.config['rsi_overbought'],
            
            # Trend strength positive
            current_bar['trend_strength'] > self.config['trend_strength_threshold']
        ]):
            signal = {
                'type': 'LONG',
                'price': current_bar['Close'],
                'stop_loss': current_bar['Close'] - 2 * current_bar['ATR'],
                'take_profit': current_bar['Close'] + self.config['risk_reward_ratio'] * 2 * current_bar['ATR'],
                'reason': 'MA crossover + Value area support',
                'timestamp': current_bar.name
            }
        
        # Short signals
        elif all([
            # Trend alignment
            current_bar['MA_fast'] < current_bar['MA_slow'],
            prev_bar['MA_fast'] >= prev_bar['MA_slow'],  # MA crossover
            
            # Price near value area high or resistance
            (abs(current_bar['Close'] - current_bar['VAH']) / current_bar['Close'] < 0.001 or
             abs(current_bar['Close'] - current_bar['nearest_resistance']) / current_bar['Close'] < 0.001),
            
            # RSI not oversold
            current_bar['RSI'] > self.config['rsi_oversold'],
            
            # Trend strength negative
            current_bar['trend_strength'] < -self.config['trend_strength_threshold']
        ]):
            signal = {
                'type': 'SHORT',
                'price': current_bar['Close'],
                'stop_loss': current_bar['Close'] + 2 * current_bar['ATR'],
                'take_profit': current_bar['Close'] - self.config['risk_reward_ratio'] * 2 * current_bar['ATR'],
                'reason': 'MA crossover + Value area resistance',
                'timestamp': current_bar.name
            }
        
        # Additional pattern-based signals
        if current_bar['bullish_engulfing'] and current_bar['Close'] > current_bar['VAL']:
            signal = {
                'type': 'LONG',
                'price': current_bar['Close'],
                'stop_loss': current_bar['Low'],
                'take_profit': current_bar['Close'] + (current_bar['Close'] - current_bar['Low']) * 2,
                'reason': 'Bullish engulfing at value area',
                'timestamp': current_bar.name
            }
        
        return signal
    
    def check_exit_conditions(self, position, current_bar):
        """Check if position should be closed."""
        if position['side'] == 'LONG':
            # Stop loss hit
            if current_bar['Low'] <= position['stop_loss']:
                return {
                    'exit': True,
                    'price': position['stop_loss'],
                    'reason': 'Stop loss hit'
                }
            
            # Take profit hit
            if current_bar['High'] >= position['take_profit']:
                return {
                    'exit': True,
                    'price': position['take_profit'],
                    'reason': 'Take profit hit'
                }
            
            # Trailing stop - if price moved favorably, trail the stop
            if current_bar['Close'] > position['entry_price'] * 1.01:
                new_stop = current_bar['Close'] - 2 * current_bar['ATR']
                if new_stop > position['stop_loss']:
                    position['stop_loss'] = new_stop
            
        else:  # SHORT
            # Stop loss hit
            if current_bar['High'] >= position['stop_loss']:
                return {
                    'exit': True,
                    'price': position['stop_loss'],
                    'reason': 'Stop loss hit'
                }
            
            # Take profit hit
            if current_bar['Low'] <= position['take_profit']:
                return {
                    'exit': True,
                    'price': position['take_profit'],
                    'reason': 'Take profit hit'
                }
            
            # Trailing stop
            if current_bar['Close'] < position['entry_price'] * 0.99:
                new_stop = current_bar['Close'] + 2 * current_bar['ATR']
                if new_stop < position['stop_loss']:
                    position['stop_loss'] = new_stop
        
        return {'exit': False}