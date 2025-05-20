import numpy as np
import pandas as pd
from enum import Enum
import time
from src.utils.indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_atr, calculate_stochastic, calculate_ichimoku,
    detect_divergence, calculate_support_resistance
)
from src.utils.risk_management import (
    calculate_position_size, calculate_risk_reward,
    set_stop_loss, set_take_profit, adjust_position
)
from src.utils.market_analysis import (
    analyze_market_structure, detect_market_regime,
    calculate_correlation, check_news_events,
    analyze_sentiment, check_high_impact_times
)

class MarketRegime(Enum):
    TRENDING = 1
    RANGING = 2
    VOLATILE = 3
    UNDEFINED = 4

class TradeDirection(Enum):
    LONG = 1
    SHORT = 2
    NEUTRAL = 3

class AdaptiveStrategy:
    def __init__(self, config=None):
        # Default configuration
        self.default_config = {
            # Timeframes to analyze (now includes weekly and monthly)
            'timeframes': ['1H', '4H', 'D', 'W', 'M'],
            # Timeframe configuration - Trade ONLY on daily, use weekly/monthly just for context
            'primary_timeframe': 'D',    # Primary for entry signals (ONLY timeframe we'll trade on)
            'trend_timeframes': ['W', 'M'],  # For trend context only, not for trading
            'confirmation_timeframe': 'D',  # Use daily for self-confirmation
            
            # Risk parameters
            'risk_per_trade': 0.01,  # 1% of account per trade
            'max_risk_per_day': 0.03,  # 3% of account per day
            'max_open_positions': 3,
            'position_scaling': True,
            
            # Strategy parameters
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std_dev': 2,
            'atr_period': 14,
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            
            # Execution parameters
            'use_market_hours': True,
            'avoid_news': True,
            'auto_adjust': True,
            'pyramiding': True,
            'max_pyramid_positions': 2,
            
            # Filters
            'min_adr_percentage': 0.5,  # Minimum ADR for trading
            'correlation_threshold': 0.7,  # Pairs above this are too correlated
            'trend_confirmation_threshold': 2,  # Need at least 2 indicators to confirm trend
            
            # Advanced features
            'use_machine_learning': False,
            'adaptive_parameters': True,
            'use_sentiment_analysis': True,
            'use_tick_data': False,
            'backtest_optimization': True
        }
        
        # Override defaults with provided config
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Strategy state
        self.current_regime = MarketRegime.UNDEFINED
        self.open_positions = {}
        self.daily_risk_used = 0
        self.trade_history = []
        self.performance_metrics = {}
        self.current_correlations = {}
        self.support_resistance_levels = {}
        
        # Analysis cache to avoid redundant calculations
        self.analysis_cache = {}
        self.cache_expiry = {}
        self.cache_ttl = 300  # Cache time to live in seconds

    def analyze_markets(self, market_data):
        """
        Comprehensive market analysis across multiple timeframes and instruments
        
        Args:
            market_data: Dictionary of market data for different instruments and timeframes
            
        Returns:
            Dictionary of analysis results for each instrument
        """
        analysis_results = {}
        
        # Reset daily risk if it's a new day
        self._check_reset_daily_risk()
        
        # Update correlation matrix for risk management
        if 'correlation_data' in market_data:
            self.current_correlations = calculate_correlation(market_data['correlation_data'])
        
        # Analyze each instrument
        for instrument, data in market_data.items():
            if instrument == 'correlation_data':
                continue
                
            analysis_results[instrument] = {}
            
            # Loop through each timeframe
            for timeframe in self.config['timeframes']:
                if timeframe not in data:
                    continue
                    
                # Check if we have cached results that are still valid
                cache_key = f"{instrument}_{timeframe}"
                if self._is_cache_valid(cache_key):
                    analysis_results[instrument][timeframe] = self.analysis_cache[cache_key]
                    continue
                
                # Perform new analysis for this timeframe
                timeframe_data = data[timeframe]
                analysis = self._analyze_single_timeframe(instrument, timeframe, timeframe_data)
                
                # Cache the results
                self.analysis_cache[cache_key] = analysis
                self.cache_expiry[cache_key] = time.time() + self.cache_ttl
                
                analysis_results[instrument][timeframe] = analysis
            
            # Determine market regime by combining multiple timeframes
            self._determine_market_regime(instrument, analysis_results[instrument])
            
            # Update support/resistance levels
            self.support_resistance_levels[instrument] = calculate_support_resistance(
                market_data[instrument], self.config['timeframes']
            )
            
            # Check for news events if configured
            if self.config['avoid_news']:
                analysis_results[instrument]['news_events'] = check_news_events(instrument)
                
            # Check market hours
            if self.config['use_market_hours']:
                analysis_results[instrument]['market_hours'] = check_high_impact_times(instrument)
                
            # Add sentiment analysis if enabled
            if self.config['use_sentiment_analysis']:
                analysis_results[instrument]['sentiment'] = analyze_sentiment(instrument)
                
        return analysis_results
    
    def _analyze_single_timeframe(self, instrument, timeframe, data):
        """Analyze a single timeframe for a given instrument"""
        analysis = {}
        
        # Calculate technical indicators
        analysis['rsi'] = calculate_rsi(data, self.config['rsi_period'])
        
        macd_result = calculate_macd(
            data, 
            self.config['macd_fast'], 
            self.config['macd_slow'], 
            self.config['macd_signal']
        )
        analysis['macd'] = macd_result['macd']
        analysis['macd_signal'] = macd_result['signal']
        analysis['macd_histogram'] = macd_result['histogram']
        
        bb_result = calculate_bollinger_bands(data, self.config['bb_period'], self.config['bb_std_dev'])
        analysis['bb_upper'] = bb_result['upper']
        analysis['bb_middle'] = bb_result['middle']
        analysis['bb_lower'] = bb_result['lower']
        analysis['bb_width'] = bb_result['width']
        
        analysis['atr'] = calculate_atr(data, self.config['atr_period'])
        
        stoch_result = calculate_stochastic(data, self.config['stoch_k_period'], self.config['stoch_d_period'])
        analysis['stoch_k'] = stoch_result['k']
        analysis['stoch_d'] = stoch_result['d']
        
        # Calculate Ichimoku cloud
        ichimoku = calculate_ichimoku(data)
        analysis.update(ichimoku)
        
        # Detect divergences
        analysis['divergences'] = detect_divergence(
            data, 
            analysis['rsi'], 
            analysis['macd'], 
            analysis['stoch_k']
        )
        
        # Market structure analysis
        analysis['market_structure'] = analyze_market_structure(data)
        
        # Candlestick pattern detection (new)
        from src.utils.indicators import detect_candlestick_patterns
        analysis['candlestick_patterns'] = detect_candlestick_patterns(data)
        
        # Only calculate volume profile and order flow for timeframes Daily or smaller
        # These are computationally intensive and more relevant for shorter timeframes
        if timeframe not in ['W', 'M']:
            # Volume profile analysis (new)
            from src.utils.market_analysis import calculate_volume_profile, analyze_order_flow
            
            # Only perform volume analysis if the data contains volume
            if 'volume' in data.columns and data['volume'].sum() > 0:
                # Volume profile for price levels
                analysis['volume_profile'] = calculate_volume_profile(data)
                
                # Order flow analysis for momentum
                analysis['order_flow'] = analyze_order_flow(data)
        
        # Perform long-term trend analysis for weekly and monthly timeframes
        if timeframe in ['W', 'M'] and len(data) >= 10:
            from src.utils.market_analysis import analyze_long_term_trends
            
            # For weekly timeframe, we may want to check against monthly if available
            if timeframe == 'W':
                # Try to get monthly data from cache
                monthly_data = None
                monthly_key = f"{instrument}_M"
                if monthly_key in self.analysis_cache:
                    monthly_data = self.analysis_cache[monthly_key]
                    
                analysis['long_term_trends'] = analyze_long_term_trends(data, None, monthly_data)
            
            # For monthly timeframe, we just analyze the data itself for long-term trends
            elif timeframe == 'M':
                analysis['long_term_trends'] = analyze_long_term_trends(None, None, data)
        
        return analysis
    
    def _determine_market_regime(self, instrument, analysis_results):
        """Determine the current market regime for decision making"""
        # Use data from multiple timeframes to determine regime
        timeframe_regimes = {}
        
        for timeframe, analysis in analysis_results.items():
            # Check for trending market
            if analysis['market_structure']['trend_strength'] > 0.7:
                timeframe_regimes[timeframe] = MarketRegime.TRENDING
            # Check for ranging market
            elif analysis['bb_width'][.iloc-1] < analysis['bb_width'][.iloc-20:].mean() * 0.8:
                timeframe_regimes[timeframe] = MarketRegime.RANGING
            # Check for volatile market
            elif analysis['atr'][.iloc-1] > analysis['atr'][.iloc-20:].mean() * 1.5:
                timeframe_regimes[timeframe] = MarketRegime.VOLATILE
            else:
                timeframe_regimes[timeframe] = MarketRegime.UNDEFINED
        
        # Combine timeframe regimes with priority to larger timeframes
        for tf in reversed(self.config['timeframes']):
            if tf in timeframe_regimes:
                self.current_regime = timeframe_regimes[tf]
                break
                
    def generate_signals(self, analysis_results):
        """
        Generate trading signals based on analysis results and current market regime
        
        Prioritizes daily timeframe for signal generation, with weekly and monthly
        used for trend context and analysis
        
        Args:
            analysis_results: Dictionary of analysis results from analyze_markets
            
        Returns:
            Dictionary of trading signals for each instrument
        """
        signals = {}
        
        for instrument, analysis in analysis_results.items():
            # Skip processing if we don't have multi-timeframe data
            if len(analysis) < 2:
                continue
                
            # Default to no signal
            signal = {
                'direction': TradeDirection.NEUTRAL,
                'strength': 0,
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'timeframe': None,
                'reason': []
            }
            
            # Check if we should avoid trading due to news
            if 'news_events' in analysis and analysis['news_events']:
                signal['reason'].append(f"Avoiding trade due to news events: {analysis['news_events']}")
                signals[instrument] = signal
                continue
                
            # Check market hours if configured
            if 'market_hours' in analysis and not analysis['market_hours']:
                signal['reason'].append("Outside optimal market hours")
                signals[instrument] = signal
                continue
            
            # Set timeframes based on configuration
            primary_tf = self.config['primary_timeframe']  # Default to daily
            confirm_tf = self.config['confirmation_timeframe']  # Default to 4h
            trend_timeframes = self.config['trend_timeframes']  # Weekly and monthly
            
            # Check if our preferred timeframes are available
            if primary_tf not in analysis:
                # Fall back to largest available timeframe
                available_tfs = sorted([tf for tf in analysis.keys() 
                                     if tf not in trend_timeframes],
                                     key=lambda x: self._get_timeframe_minutes(x))
                if available_tfs:
                    primary_tf = available_tfs[-1]
                    signal['reason'].append(f"Preferred primary timeframe not available, using {primary_tf}")
                else:
                    signal['reason'].append(f"No suitable timeframes available")
                    signals[instrument] = signal
                    continue
            
            if confirm_tf not in analysis:
                # Fall back to second-largest available timeframe
                available_tfs = sorted([tf for tf in analysis.keys() 
                                      if tf != primary_tf and tf not in trend_timeframes],
                                      key=lambda x: self._get_timeframe_minutes(x))
                if available_tfs:
                    confirm_tf = available_tfs[-1]
                    signal['reason'].append(f"Preferred confirmation timeframe not available, using {confirm_tf}")
                else:
                    # If no confirmation timeframe is available, use primary as confirmation too
                    confirm_tf = primary_tf
                    signal['reason'].append(f"No confirmation timeframe available, using primary")
            
            # Check for trend alignment with higher timeframes
            trend_aligned = self._check_trend_alignment(instrument, analysis, trend_timeframes, primary_tf)
            
            # Get long-term trend analysis if available
            long_term_context = self._get_long_term_context(instrument, analysis, trend_timeframes)
            
            # Apply regime-specific signal generation
            if self.current_regime == MarketRegime.TRENDING:
                signal = self._generate_trend_following_signal(
                    instrument, 
                    analysis, 
                    primary_tf, 
                    confirm_tf,
                    long_term_context
                )
            elif self.current_regime == MarketRegime.RANGING:
                signal = self._generate_range_trading_signal(
                    instrument, 
                    analysis, 
                    primary_tf, 
                    confirm_tf,
                    long_term_context
                )
            elif self.current_regime == MarketRegime.VOLATILE:
                signal = self._generate_volatility_signal(
                    instrument, 
                    analysis, 
                    primary_tf, 
                    confirm_tf,
                    long_term_context
                )
            else:
                signal['reason'].append("Undefined market regime - no signal generated")
            
            # Add trend alignment information to signal
            if trend_aligned:
                signal['reason'].append("Signal aligned with higher timeframe trends")
                signal['strength'] = min(signal['strength'] * 1.3, 1.0)  # Boost signal strength
            else:
                signal['reason'].append("Signal not aligned with higher timeframe trends")
                signal['strength'] *= 0.7  # Reduce signal strength
                
            # ONLY allow trading on daily timeframe regardless of signal strength
            if signal['timeframe'] != 'D':
                signal['direction'] = TradeDirection.NEUTRAL
                signal['strength'] = 0
                signal['reason'].append("Signal ignored - only trading on daily timeframe")
            
            # Calculate risk parameters if we have a directional signal
            if signal['direction'] != TradeDirection.NEUTRAL and signal['entry_price'] is not None:
                # Calculate optimal position size
                position_details = calculate_position_size(
                    signal['direction'], 
                    signal['entry_price'],
                    signal['stop_loss'],
                    self.config['risk_per_trade'],
                    self.daily_risk_used,
                    self.config['max_risk_per_day'],
                    instrument
                )
                
                # Update signal with position details
                signal.update(position_details)
                
                # Calculate risk-reward ratio
                signal['risk_reward'] = calculate_risk_reward(
                    signal['direction'],
                    signal['entry_price'],
                    signal['stop_loss'],
                    signal['take_profit']
                )
                
                # Check correlation risk
                if self._has_correlation_risk(instrument):
                    signal['reason'].append("High correlation with existing positions")
                    signal['strength'] *= 0.5  # Reduce signal strength
                    
            signals[instrument] = signal
                
        return signals
        
    def _check_trend_alignment(self, instrument, analysis, trend_timeframes, primary_tf):
        """Check alignment between primary timeframe and higher timeframes"""
        if not trend_timeframes:
            return True  # No higher timeframes to check
            
        # Get primary timeframe trend direction
        if primary_tf not in analysis or 'market_structure' not in analysis[primary_tf]:
            return False
            
        primary_trend = analysis[primary_tf]['market_structure'].get('trend_direction', 'undefined')
        
        # Count how many higher timeframes align with primary
        aligned_count = 0
        higher_tf_count = 0
        
        for tf in trend_timeframes:
            if tf in analysis and 'market_structure' in analysis[tf]:
                higher_tf_count += 1
                higher_trend = analysis[tf]['market_structure'].get('trend_direction', 'undefined')
                
                if higher_trend == primary_trend and higher_trend != 'undefined':
                    aligned_count += 1
        
        # Check also for long-term trend analysis if available
        for tf in trend_timeframes:
            tf_key = f"{instrument}_{tf}"
            if tf_key in self.analysis_cache and 'long_term_trends' in self.analysis_cache[tf_key]:
                higher_tf_count += 1
                lt_trend = self.analysis_cache[tf_key]['long_term_trends'].get('direction', {}).get('direction', 'undefined')
                
                if lt_trend == primary_trend and lt_trend != 'undefined':
                    aligned_count += 1
        
        # Return true if at least 50% of higher timeframes align with primary
        return aligned_count >= (higher_tf_count / 2) if higher_tf_count > 0 else True
    
    def _get_long_term_context(self, instrument, analysis, trend_timeframes):
        """Get context from higher timeframes"""
        context = {
            'trend_direction': 'undefined',
            'trend_strength': 0,
            'key_levels': {
                'support': [],
                'resistance': []
            }
        }
        
        # Check for higher timeframe analysis in cache
        has_long_term_analysis = False
        
        # Extract trend information from weekly/monthly timeframes
        for tf in trend_timeframes:
            tf_key = f"{instrument}_{tf}"
            
            # First check if we have long-term trend analysis in the cache
            if tf_key in self.analysis_cache and 'long_term_trends' in self.analysis_cache[tf_key]:
                lt_trends = self.analysis_cache[tf_key]['long_term_trends']
                
                # Get direction and strength
                direction = lt_trends.get(f'{tf.lower()}_trend', {}).get('direction', 'undefined')
                strength = lt_trends.get(f'{tf.lower()}_trend', {}).get('strength', 0)
                
                # Only update if this timeframe has a defined direction and is stronger
                if direction != 'undefined' and strength > context['trend_strength']:
                    context['trend_direction'] = direction
                    context['trend_strength'] = strength
                
                # Add support/resistance levels
                for level in lt_trends.get('support_levels', []):
                    context['key_levels']['support'].append({
                        'price': level['price'],
                        'strength': level['strength'],
                        'source': level['source']
                    })
                
                for level in lt_trends.get('resistance_levels', []):
                    context['key_levels']['resistance'].append({
                        'price': level['price'],
                        'strength': level['strength'],
                        'source': level['source']
                    })
                
                has_long_term_analysis = True
            
            # If timeframe is available directly in current analysis, check it too
            elif tf in analysis and 'market_structure' in analysis[tf]:
                ms = analysis[tf]['market_structure']
                direction = ms.get('trend_direction', 'undefined')
                strength = ms.get('trend_strength', 0)
                
                # Only update if this is a stronger trend
                if direction != 'undefined' and strength > context['trend_strength']:
                    context['trend_direction'] = direction
                    context['trend_strength'] = strength * 0.8  # Give slight preference to long-term trend analysis
        
        # If we don't have long-term analysis yet, use regular support/resistance
        if not has_long_term_analysis and instrument in self.support_resistance_levels:
            sr_levels = self.support_resistance_levels[instrument]
            
            # Add all support levels
            if 'supports' in sr_levels:
                for level in sr_levels['supports']:
                    context['key_levels']['support'].append({
                        'price': level,
                        'strength': 0.5,  # Medium strength since we don't know the source
                        'source': 'general'
                    })
            
            # Add all resistance levels
            if 'resistances' in sr_levels:
                for level in sr_levels['resistances']:
                    context['key_levels']['resistance'].append({
                        'price': level,
                        'strength': 0.5,  # Medium strength since we don't know the source
                        'source': 'general'
                    })
        
        return context
    
    def _generate_trend_following_signal(self, instrument, analysis, primary_tf, confirm_tf, long_term_context=None):
        """
        Generate signals for trending market conditions
        
        Args:
            instrument: Trading instrument
            analysis: Analysis results
            primary_tf: Primary timeframe (typically daily)
            confirm_tf: Confirmation timeframe (typically 4H)
            long_term_context: Long-term trend context from weekly/monthly
            
        Returns:
            Signal dictionary
        """
        signal = {
            'direction': TradeDirection.NEUTRAL,
            'strength': 0,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'timeframe': primary_tf,
            'reason': ["Trend following strategy applied"]
        }
        
        # Get primary and confirmation timeframe data
        primary = analysis[primary_tf]
        confirm = analysis[confirm_tf]
        
        # Count bullish signals
        bullish_signals = 0
        bearish_signals = 0
        
        # Check primary timeframe trend
        if primary['market_structure']['trend_direction'] == 'up':
            bullish_signals += 2
            signal['reason'].append(f"Bullish trend on {primary_tf}")
        elif primary['market_structure']['trend_direction'] == 'down':
            bearish_signals += 2
            signal['reason'].append(f"Bearish trend on {primary_tf}")
            
        # Check confirmation timeframe alignment
        if confirm['market_structure']['trend_direction'] == 'up':
            bullish_signals += 1
            signal['reason'].append(f"Bullish trend on {confirm_tf}")
        elif confirm['market_structure']['trend_direction'] == 'down':
            bearish_signals += 1
            signal['reason'].append(f"Bearish trend on {confirm_tf}")
            
        # Check MACD
        if primary['macd'][.iloc-1] > primary['macd_signal'][.iloc-1] and primary['macd_histogram'][.iloc-1] > 0:
            bullish_signals += 1
            signal['reason'].append(f"Bullish MACD on {primary_tf}")
        elif primary['macd'][.iloc-1] < primary['macd_signal'][.iloc-1] and primary['macd_histogram'][.iloc-1] < 0:
            bearish_signals += 1
            signal['reason'].append(f"Bearish MACD on {primary_tf}")
            
        # Check RSI direction
        if primary['rsi'][.iloc-1] > primary['rsi'][.iloc-2] and primary['rsi'][.iloc-1] > 50:
            bullish_signals += 1
            signal['reason'].append(f"Rising RSI above 50 on {primary_tf}")
        elif primary['rsi'][.iloc-1] < primary['rsi'][.iloc-2] and primary['rsi'][.iloc-1] < 50:
            bearish_signals += 1
            signal['reason'].append(f"Falling RSI below 50 on {primary_tf}")
            
        # Check Ichimoku Cloud
        if 'cloud_bullish' in primary and primary['cloud_bullish']:
            bullish_signals += 1
            signal['reason'].append(f"Price above Ichimoku Cloud on {primary_tf}")
        elif 'cloud_bearish' in primary and primary['cloud_bearish']:
            bearish_signals += 1
            signal['reason'].append(f"Price below Ichimoku Cloud on {primary_tf}")
            
        # Check for divergences (strong reversal signals)
        if 'divergences' in primary:
            if 'bullish' in primary['divergences'] and primary['divergences']['bullish']:
                bullish_signals += 2
                signal['reason'].append(f"Bullish divergence detected on {primary_tf}")
            if 'bearish' in primary['divergences'] and primary['divergences']['bearish']:
                bearish_signals += 2
                signal['reason'].append(f"Bearish divergence detected on {primary_tf}")
                
        # Generate signal if we have enough confirmations
        if bullish_signals >= self.config['trend_confirmation_threshold'] and bullish_signals > bearish_signals:
            signal['direction'] = TradeDirection.LONG
            signal['strength'] = min(bullish_signals / 5, 1.0)  # Normalize to max 1.0
            
            # Calculate entry, stop and take profit
            last_price = self._get_last_price(instrument)
            if last_price:
                signal['entry_price'] = last_price
                signal['stop_loss'] = self._calculate_stop_loss(
                    instrument, 
                    TradeDirection.LONG, 
                    last_price, 
                    primary['atr'][.iloc-1]
                )
                signal['take_profit'] = self._calculate_take_profit(
                    instrument,
                    TradeDirection.LONG,
                    last_price,
                    signal['stop_loss']
                )
                
        elif bearish_signals >= self.config['trend_confirmation_threshold'] and bearish_signals > bullish_signals:
            signal['direction'] = TradeDirection.SHORT
            signal['strength'] = min(bearish_signals / 5, 1.0)  # Normalize to max 1.0
            
            # Calculate entry, stop and take profit
            last_price = self._get_last_price(instrument)
            if last_price:
                signal['entry_price'] = last_price
                signal['stop_loss'] = self._calculate_stop_loss(
                    instrument, 
                    TradeDirection.SHORT, 
                    last_price, 
                    primary['atr'][.iloc-1]
                )
                signal['take_profit'] = self._calculate_take_profit(
                    instrument,
                    TradeDirection.SHORT,
                    last_price,
                    signal['stop_loss']
                )
        
        return signal
        
    def _generate_range_trading_signal(self, instrument, analysis, primary_tf, confirm_tf, long_term_context=None):
        """
        Generate signals for ranging market conditions
        
        Args:
            instrument: Trading instrument
            analysis: Analysis results
            primary_tf: Primary timeframe (typically daily)
            confirm_tf: Confirmation timeframe (typically 4H)
            long_term_context: Long-term trend context from weekly/monthly
            
        Returns:
            Signal dictionary
        """
        signal = {
            'direction': TradeDirection.NEUTRAL,
            'strength': 0,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'timeframe': primary_tf,
            'reason': ["Range trading strategy applied"]
        }
        
        # Get primary and confirmation timeframe data
        primary = analysis[primary_tf]
        
        # Get support and resistance levels
        if instrument in self.support_resistance_levels:
            levels = self.support_resistance_levels[instrument]
            nearest_support = None
            nearest_resistance = None
            
            # Find nearest levels
            last_price = self._get_last_price(instrument)
            if last_price and 'supports' in levels and 'resistances' in levels:
                # Find nearest support
                supports = [s for s in levels['supports'] if s < last_price]
                if supports:
                    nearest_support = max(supports)
                    
                # Find nearest resistance
                resistances = [r for r in levels['resistances'] if r > last_price]
                if resistances:
                    nearest_resistance = min(resistances)
            
            # Check if price is near support
            if nearest_support and last_price:
                distance_to_support = (last_price - nearest_support) / last_price
                if distance_to_support < 0.005:  # Within 0.5% of support
                    signal['direction'] = TradeDirection.LONG
                    signal['strength'] = 0.7
                    signal['reason'].append(f"Price near support at {nearest_support}")
                    signal['entry_price'] = last_price
                    signal['stop_loss'] = nearest_support * 0.995  # Stop just below support
                    
                    # Find target at resistance or based on range size
                    if nearest_resistance:
                        signal['take_profit'] = nearest_resistance
                    else:
                        # Use range size for take profit
                        signal['take_profit'] = last_price + (last_price - nearest_support) * 2
                        
            # Check if price is near resistance
            elif nearest_resistance and last_price:
                distance_to_resistance = (nearest_resistance - last_price) / last_price
                if distance_to_resistance < 0.005:  # Within 0.5% of resistance
                    signal['direction'] = TradeDirection.SHORT
                    signal['strength'] = 0.7
                    signal['reason'].append(f"Price near resistance at {nearest_resistance}")
                    signal['entry_price'] = last_price
                    signal['stop_loss'] = nearest_resistance * 1.005  # Stop just above resistance
                    
                    # Find target at support or based on range size
                    if nearest_support:
                        signal['take_profit'] = nearest_support
                    else:
                        # Use range size for take profit
                        signal['take_profit'] = last_price - (nearest_resistance - last_price) * 2
        
        # Check if RSI is in extreme zones (oversold/overbought)
        if primary['rsi'][.iloc-1] < self.config['rsi_oversold']:
            # Strengthen or create bullish signal
            if signal['direction'] == TradeDirection.LONG:
                signal['strength'] = min(signal['strength'] + 0.2, 1.0)
                signal['reason'].append(f"Oversold RSI: {primary['rsi'][.iloc-1]}")
            elif signal['direction'] == TradeDirection.NEUTRAL:
                signal['direction'] = TradeDirection.LONG
                signal['strength'] = 0.6
                signal['reason'].append(f"Oversold RSI: {primary['rsi'][.iloc-1]}")
                
                # Calculate entry, stop and take profit if not already set
                if signal['entry_price'] is None:
                    last_price = self._get_last_price(instrument)
                    if last_price:
                        signal['entry_price'] = last_price
                        signal['stop_loss'] = last_price * 0.99  # 1% stop
                        signal['take_profit'] = last_price * 1.02  # 2% target
                        
        elif primary['rsi'][.iloc-1] > self.config['rsi_overbought']:
            # Strengthen or create bearish signal
            if signal['direction'] == TradeDirection.SHORT:
                signal['strength'] = min(signal['strength'] + 0.2, 1.0)
                signal['reason'].append(f"Overbought RSI: {primary['rsi'][.iloc-1]}")
            elif signal['direction'] == TradeDirection.NEUTRAL:
                signal['direction'] = TradeDirection.SHORT
                signal['strength'] = 0.6
                signal['reason'].append(f"Overbought RSI: {primary['rsi'][.iloc-1]}")
                
                # Calculate entry, stop and take profit if not already set
                if signal['entry_price'] is None:
                    last_price = self._get_last_price(instrument)
                    if last_price:
                        signal['entry_price'] = last_price
                        signal['stop_loss'] = last_price * 1.01  # 1% stop
                        signal['take_profit'] = last_price * 0.98  # 2% target
                        
        # Check if price is testing Bollinger Bands
        if primary['bb_lower'][.iloc-1] is not None and primary['bb_upper'][.iloc-1] is not None:
            last_price = self._get_last_price(instrument)
            if last_price:
                # Bullish signal if price is near lower band
                if last_price <= primary['bb_lower'][.iloc-1] * 1.001:
                    if signal['direction'] in [TradeDirection.LONG, TradeDirection.NEUTRAL]:
                        signal['direction'] = TradeDirection.LONG
                        signal['strength'] = max(signal['strength'], 0.7)
                        signal['reason'].append("Price at lower Bollinger Band")
                        if signal['entry_price'] is None:
                            signal['entry_price'] = last_price
                            signal['stop_loss'] = last_price * 0.99
                            signal['take_profit'] = primary['bb_middle'][.iloc-1]
                
                # Bearish signal if price is near upper band
                elif last_price >= primary['bb_upper'][.iloc-1] * 0.999:
                    if signal['direction'] in [TradeDirection.SHORT, TradeDirection.NEUTRAL]:
                        signal['direction'] = TradeDirection.SHORT
                        signal['strength'] = max(signal['strength'], 0.7)
                        signal['reason'].append("Price at upper Bollinger Band")
                        if signal['entry_price'] is None:
                            signal['entry_price'] = last_price
                            signal['stop_loss'] = last_price * 1.01
                            signal['take_profit'] = primary['bb_middle'][.iloc-1]
            
        return signal
    
    def _generate_volatility_signal(self, instrument, analysis, primary_tf, confirm_tf, long_term_context=None):
        """
        Generate signals for volatile market conditions - more conservative approach
        
        Args:
            instrument: Trading instrument
            analysis: Analysis results
            primary_tf: Primary timeframe (typically daily)
            confirm_tf: Confirmation timeframe (typically 4H)
            long_term_context: Long-term trend context from weekly/monthly
            
        Returns:
            Signal dictionary
        """
        signal = {
            'direction': TradeDirection.NEUTRAL,
            'strength': 0,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'timeframe': primary_tf,
            'reason': ["Volatility-based strategy applied"]
        }
        
        # Get primary timeframe data
        primary = analysis[primary_tf]
        
        # In volatile markets, we want multiple strong confirmations
        bullish_signals = 0
        bearish_signals = 0
        signal_threshold = self.config['trend_confirmation_threshold'] + 1  # Higher threshold
        
        # Check if there's a strong momentum move
        if 'market_structure' in primary and 'momentum' in primary['market_structure']:
            momentum = primary['market_structure']['momentum']
            if momentum > 0.8:  # Strong bullish momentum
                bullish_signals += 2
                signal['reason'].append(f"Strong bullish momentum: {momentum:.2f}")
            elif momentum < -0.8:  # Strong bearish momentum
                bearish_signals += 2
                signal['reason'].append(f"Strong bearish momentum: {momentum:.2f}")
                
        # Look for strong divergences (good in volatile markets)
        if 'divergences' in primary:
            if 'bullish' in primary['divergences'] and primary['divergences']['bullish']:
                bullish_signals += 2
                signal['reason'].append(f"Bullish divergence in volatile market")
            if 'bearish' in primary['divergences'] and primary['divergences']['bearish']:
                bearish_signals += 2
                signal['reason'].append(f"Bearish divergence in volatile market")
                
        # Check for oversold/overbought conditions with reversal signs
        if primary['rsi'][.iloc-1] < 30 and primary['rsi'][.iloc-1] > primary['rsi'][.iloc-2]:
            bullish_signals += 1
            signal['reason'].append(f"Oversold RSI with uptick: {primary['rsi'][.iloc-1]:.2f}")
        elif primary['rsi'][.iloc-1] > 70 and primary['rsi'][.iloc-1] < primary['rsi'][.iloc-2]:
            bearish_signals += 1
            signal['reason'].append(f"Overbought RSI with downtick: {primary['rsi'][.iloc-1]:.2f}")
            
        # Check for Bollinger Band squeeze followed by breakout (volatility expansion)
        if primary['bb_width'][.iloc-1] > primary['bb_width'][.iloc-2] * 1.1 and primary['bb_width'][.iloc-2] < primary['bb_width'][.iloc-20:].mean() * 0.8:
            last_price = self._get_last_price(instrument)
            if last_price:
                if last_price > primary['bb_upper'][.iloc-1]:
                    bullish_signals += 2
                    signal['reason'].append("Bollinger Band squeeze with upside breakout")
                elif last_price < primary['bb_lower'][.iloc-1]:
                    bearish_signals += 2
                    signal['reason'].append("Bollinger Band squeeze with downside breakout")
        
        # Check for strong MACD histogram expansion
        if abs(primary['macd_histogram'][.iloc-1]) > abs(primary['macd_histogram'][.iloc-5:]).mean() * 1.5:
            if primary['macd_histogram'][.iloc-1] > 0 and primary['macd_histogram'][.iloc-1] > primary['macd_histogram'][.iloc-2]:
                bullish_signals += 1
                signal['reason'].append("Strong MACD histogram expansion (bullish)")
            elif primary['macd_histogram'][.iloc-1] < 0 and primary['macd_histogram'][.iloc-1] < primary['macd_histogram'][.iloc-2]:
                bearish_signals += 1
                signal['reason'].append("Strong MACD histogram expansion (bearish)")
                
        # Generate signal if we have enough strong confirmations
        if bullish_signals >= signal_threshold and bullish_signals > bearish_signals + 1:
            signal['direction'] = TradeDirection.LONG
            signal['strength'] = min(bullish_signals / 6, 1.0)  # Normalize to max 1.0
            
            # Calculate entry, stop and take profit
            last_price = self._get_last_price(instrument)
            if last_price:
                signal['entry_price'] = last_price
                # Use wider stop loss in volatile markets (1.5x normal ATR)
                signal['stop_loss'] = self._calculate_stop_loss(
                    instrument, 
                    TradeDirection.LONG, 
                    last_price, 
                    primary['atr'][.iloc-1] * 1.5
                )
                # Tighter take profit in volatile markets (1:1 risk-reward initially)
                risk = last_price - signal['stop_loss']
                signal['take_profit'] = last_price + risk
                
        elif bearish_signals >= signal_threshold and bearish_signals > bullish_signals + 1:
            signal['direction'] = TradeDirection.SHORT
            signal['strength'] = min(bearish_signals / 6, 1.0)  # Normalize to max 1.0
            
            # Calculate entry, stop and take profit
            last_price = self._get_last_price(instrument)
            if last_price:
                signal['entry_price'] = last_price
                # Use wider stop loss in volatile markets (1.5x normal ATR)
                signal['stop_loss'] = self._calculate_stop_loss(
                    instrument, 
                    TradeDirection.SHORT, 
                    last_price, 
                    primary['atr'][.iloc-1] * 1.5
                )
                # Tighter take profit in volatile markets (1:1 risk-reward initially)
                risk = signal['stop_loss'] - last_price
                signal['take_profit'] = last_price - risk
        
        return signal
    
    def execute_signals(self, signals, market_data):
        """
        Execute trading signals, manage positions, and handle trade lifecycle
        
        Args:
            signals: Dictionary of trading signals from generate_signals
            market_data: Current market data
            
        Returns:
            List of executed actions (entries, exits, adjustments)
        """
        executed_actions = []
        
        # First check existing positions for exits or adjustments
        for instrument, position in list(self.open_positions.items()):
            if instrument in market_data:
                last_price = self._get_last_price(instrument, market_data)
                
                # Check for stop loss or take profit hits
                if self._check_stop_loss_hit(position, last_price):
                    # Close position at stop loss
                    action = self._close_position(instrument, last_price, "stop_loss")
                    executed_actions.append(action)
                    continue
                
                elif self._check_take_profit_hit(position, last_price):
                    # Close position at take profit
                    action = self._close_position(instrument, last_price, "take_profit")
                    executed_actions.append(action)
                    continue
                
                # Position adjustment logic for open trades
                if self.config['auto_adjust'] and instrument in signals:
                    current_signal = signals[instrument]
                    
                    # Check if signal direction matches position direction
                    position_direction = TradeDirection.LONG if position['direction'] == 'long' else TradeDirection.SHORT
                    
                    if current_signal['direction'] == position_direction:
                        # Strong signal in same direction - consider pyramiding
                        if (self.config['pyramiding'] and 
                            current_signal['strength'] > 0.7 and 
                            position['pyramid_count'] < self.config['max_pyramid_positions']):
                            
                            # Add to position
                            action = adjust_position(
                                position,
                                "add",
                                current_signal['entry_price'],
                                position['size'] * 0.5,  # Add 50% of original position
                                current_signal['stop_loss'],
                                current_signal['take_profit']
                            )
                            self.open_positions[instrument] = action['updated_position']
                            executed_actions.append(action)
                    
                    elif current_signal['direction'] != TradeDirection.NEUTRAL:
                        # Strong opposing signal - consider reducing position
                        if current_signal['strength'] > 0.8:
                            # Reduce position size
                            action = adjust_position(
                                position,
                                "reduce",
                                last_price,
                                position['size'] * 0.5  # Reduce by 50%
                            )
                            
                            if action['remaining_size'] > 0:
                                self.open_positions[instrument] = action['updated_position']
                            else:
                                del self.open_positions[instrument]
                                
                            executed_actions.append(action)
                
                # Trailing stop adjustment if enabled
                if position.get('use_trailing_stop', False):
                    position_direction = position['direction']
                    current_stop = position['stop_loss']
                    
                    if position_direction == 'long' and last_price > position['entry_price']:
                        # Calculate new stop based on ATR or fixed percentage
                        if 'atr_value' in position:
                            new_stop = last_price - position['atr_value']
                        else:
                            new_stop = last_price * 0.98  # 2% trailing stop
                            
                        # Only move stop up, never down
                        if new_stop > current_stop:
                            action = {
                                'type': 'adjust_stop',
                                'instrument': instrument,
                                'old_stop': current_stop,
                                'new_stop': new_stop,
                                'price': last_price,
                                'time': time.time(),
                                'reason': "Trailing stop adjustment"
                            }
                            position['stop_loss'] = new_stop
                            self.open_positions[instrument] = position
                            executed_actions.append(action)
                    
                    elif position_direction == 'short' and last_price < position['entry_price']:
                        # Calculate new stop based on ATR or fixed percentage
                        if 'atr_value' in position:
                            new_stop = last_price + position['atr_value']
                        else:
                            new_stop = last_price * 1.02  # 2% trailing stop
                            
                        # Only move stop down, never up
                        if new_stop < current_stop:
                            action = {
                                'type': 'adjust_stop',
                                'instrument': instrument,
                                'old_stop': current_stop,
                                'new_stop': new_stop,
                                'price': last_price,
                                'time': time.time(),
                                'reason': "Trailing stop adjustment"
                            }
                            position['stop_loss'] = new_stop
                            self.open_positions[instrument] = position
                            executed_actions.append(action)
        
        # Process new entry signals
        for instrument, signal in signals.items():
            # Skip if no directional signal or already in a position
            if (signal['direction'] == TradeDirection.NEUTRAL or 
                instrument in self.open_positions or
                signal['entry_price'] is None):
                continue
                
            # Check max positions limit
            if len(self.open_positions) >= self.config['max_open_positions']:
                continue
                
            # Check daily risk limit
            if 'risk_amount' in signal and self.daily_risk_used + signal['risk_amount'] > self.config['max_risk_per_day']:
                continue
                
            # Check signal strength threshold
            if signal['strength'] < 0.6:  # Require moderately strong signals
                continue
                
            # Check risk-reward ratio if available
            if 'risk_reward' in signal and signal['risk_reward'] < 1.5:
                continue
                
            # All checks passed, execute entry
            position = {
                'instrument': instrument,
                'direction': 'long' if signal['direction'] == TradeDirection.LONG else 'short',
                'entry_price': signal['entry_price'],
                'size': signal['position_size'] if 'position_size' in signal else 1.0,
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'entry_time': time.time(),
                'timeframe': signal['timeframe'],
                'reason': signal['reason'],
                'risk_amount': signal['risk_amount'] if 'risk_amount' in signal else 0,
                'risk_reward': signal['risk_reward'] if 'risk_reward' in signal else None,
                'pyramid_count': 0,
                'use_trailing_stop': True,  # Enable trailing stops by default
                'atr_value': signal.get('atr_value')
            }
            
            self.open_positions[instrument] = position
            self.daily_risk_used += position['risk_amount']
            
            executed_actions.append({
                'type': 'entry',
                'position': position,
                'time': position['entry_time']
            })
            
            # Add to trade history
            self.trade_history.append({
                'instrument': instrument,
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'entry_time': position['entry_time'],
                'size': position['size'],
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'status': 'open',
                'reason': position['reason']
            })
            
        return executed_actions
    
    def _close_position(self, instrument, price, reason):
        """Close a position and update records"""
        position = self.open_positions[instrument]
        
        # Calculate P&L
        if position['direction'] == 'long':
            profit_pct = (price - position['entry_price']) / position['entry_price']
        else:  # short
            profit_pct = (position['entry_price'] - price) / position['entry_price']
            
        profit_amount = profit_pct * position['size']
        
        # Create the action record
        action = {
            'type': 'exit',
            'instrument': instrument,
            'exit_price': price,
            'exit_time': time.time(),
            'profit_pct': profit_pct,
            'profit_amount': profit_amount,
            'reason': reason,
            'position': position
        }
        
        # Update trade history
        for trade in self.trade_history:
            if (trade['instrument'] == instrument and 
                trade['entry_time'] == position['entry_time'] and
                trade['status'] == 'open'):
                trade['exit_price'] = price
                trade['exit_time'] = action['exit_time']
                trade['profit_pct'] = profit_pct
                trade['profit_amount'] = profit_amount
                trade['status'] = 'closed'
                trade['exit_reason'] = reason
                break
                
        # Update risk tracking if this is same-day close
        entry_date = self._get_date_from_timestamp(position['entry_time'])
        exit_date = self._get_date_from_timestamp(action['exit_time'])
        
        if entry_date == exit_date:
            # Only reduce used risk if we're still in the same trading day
            self.daily_risk_used -= position.get('risk_amount', 0)
            
        # Remove from open positions
        del self.open_positions[instrument]
        
        return action
    
    def _check_stop_loss_hit(self, position, current_price):
        """Check if stop loss has been hit"""
        if position['direction'] == 'long':
            return current_price <= position['stop_loss']
        else:  # short
            return current_price >= position['stop_loss']
    
    def _check_take_profit_hit(self, position, current_price):
        """Check if take profit has been hit"""
        if position['direction'] == 'long':
            return current_price >= position['take_profit']
        else:  # short
            return current_price <= position['take_profit']
    
    def _calculate_stop_loss(self, instrument, direction, entry_price, atr_value):
        """Calculate stop loss based on ATR"""
        atr_multiplier = 1.5  # Default ATR multiplier
        
        if direction == TradeDirection.LONG:
            return entry_price - (atr_value * atr_multiplier)
        else:  # SHORT
            return entry_price + (atr_value * atr_multiplier)
    
    def _calculate_take_profit(self, instrument, direction, entry_price, stop_loss):
        """Calculate take profit based on risk-reward ratio"""
        risk = abs(entry_price - stop_loss)
        reward_ratio = 2.0  # Default risk-reward ratio
        
        if direction == TradeDirection.LONG:
            return entry_price + (risk * reward_ratio)
        else:  # SHORT
            return entry_price - (risk * reward_ratio)
    
    def _get_last_price(self, instrument, market_data=None):
        """Get the last price for an instrument from market data"""
        # Mock implementation - in real system, this would get the current price
        return 1.0  # Placeholder
    
    def _has_correlation_risk(self, instrument):
        """Check if adding this instrument would create correlation risk"""
        if not self.current_correlations or not self.open_positions:
            return False
            
        for open_instrument in self.open_positions:
            corr_key = f"{instrument}_{open_instrument}"
            inv_corr_key = f"{open_instrument}_{instrument}"
            
            if corr_key in self.current_correlations and self.current_correlations[corr_key] > self.config['correlation_threshold']:
                return True
            elif inv_corr_key in self.current_correlations and self.current_correlations[inv_corr_key] > self.config['correlation_threshold']:
                return True
                
        return False
    
    def _check_reset_daily_risk(self):
        """Reset daily risk if it's a new day"""
        current_date = self._get_current_date()
        if not hasattr(self, 'last_reset_date') or self.last_reset_date != current_date:
            self.daily_risk_used = 0
            self.last_reset_date = current_date
    
    def _get_current_date(self):
        """Get current date string"""
        return time.strftime("%Y-%m-%d")
        
    def _get_date_from_timestamp(self, timestamp):
        """Convert timestamp to date string"""
        return time.strftime("%Y-%m-%d", time.localtime(timestamp))
    
    def _get_timeframe_minutes(self, timeframe):
        """Convert timeframe to minutes for comparison"""
        if timeframe.endswith('m') or (timeframe.endswith('M') and timeframe != 'M'):
            return int(timeframe[:-1])
        elif timeframe.endswith('H') or timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('D') or timeframe.endswith('d'):
            return int(timeframe[:-1]) * 60 * 24
        elif timeframe == 'W':
            return 7 * 24 * 60  # 1 week = 7 days = 10080 minutes
        elif timeframe == 'M':
            return 30 * 24 * 60  # approximate 1 month = 30 days = 43200 minutes
        return 0
    
    def _is_cache_valid(self, cache_key):
        """Check if cached analysis is still valid"""
        return (cache_key in self.analysis_cache and 
                cache_key in self.cache_expiry and 
                time.time() < self.cache_expiry[cache_key])
                
    def update_performance_metrics(self):
        """Calculate and update strategy performance metrics"""
        if not self.trade_history:
            self.performance_metrics = {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'average_win': 0,
                'average_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_trade_duration': 0
            }
            return
            
        # Filter closed trades
        closed_trades = [t for t in self.trade_history if t.get('status') == 'closed']
        if not closed_trades:
            self.performance_metrics = {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'average_win': 0,
                'average_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_trade_duration': 0
            }
            return
            
        # Calculate metrics
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t.get('profit_amount', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('profit_amount', 0) <= 0]
        
        win_count = len(winning_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        total_profit = sum([t.get('profit_amount', 0) for t in winning_trades])
        total_loss = abs(sum([t.get('profit_amount', 0) for t in losing_trades]))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        average_win = total_profit / win_count if win_count > 0 else 0
        average_loss = total_loss / len(losing_trades) if losing_trades else 0
        
        largest_win = max([t.get('profit_amount', 0) for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.get('profit_amount', 0) for t in losing_trades]) if losing_trades else 0
        
        # Calculate trade durations
        durations = []
        for trade in closed_trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                duration = trade['exit_time'] - trade['entry_time']
                durations.append(duration)
                
        avg_trade_duration = sum(durations) / len(durations) if durations else 0
        
        # Calculate running equity and drawdown
        equity_curve = []
        running_equity = 0
        peak_equity = 0
        max_drawdown = 0
        
        for trade in sorted(closed_trades, key=lambda x: x.get('exit_time', 0)):
            running_equity += trade.get('profit_amount', 0)
            equity_curve.append(running_equity)
            
            if running_equity > peak_equity:
                peak_equity = running_equity
            else:
                drawdown = peak_equity - running_equity
                max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        returns = []
        prev_equity = 0
        for equity in equity_curve:
            if prev_equity > 0:
                returns.append((equity - prev_equity) / prev_equity)
            prev_equity = equity
            
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if returns else 1
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        # Update metrics
        self.performance_metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_win': average_win,
            'average_loss': average_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_trade_duration': avg_trade_duration
        }
        
    def optimize_parameters(self, historical_data, parameter_ranges, metric='profit_factor'):
        """Optimize strategy parameters using historical data"""
        if not self.config['backtest_optimization']:
            return
        
        # Simple grid search implementation (in practice, would use more advanced methods)
        best_params = None
        best_metric_value = float('-inf')
        
        # Create grid of parameter combinations (simplified example)
        # In a real implementation, use more sophisticated optimization methods
        for rsi_period in parameter_ranges.get('rsi_period', [self.config['rsi_period']]):
            for macd_fast in parameter_ranges.get('macd_fast', [self.config['macd_fast']]):
                for macd_slow in parameter_ranges.get('macd_slow', [self.config['macd_slow']]):
                    # Create test config
                    test_config = self.config.copy()
                    test_config['rsi_period'] = rsi_period
                    test_config['macd_fast'] = macd_fast
                    test_config['macd_slow'] = macd_slow
                    
                    # Run backtest with this configuration
                    test_strategy = AdaptiveStrategy(test_config)
                    performance = self._run_backtest(test_strategy, historical_data)
                    
                    # Check if this config is better
                    metric_value = performance.get(metric, float('-inf'))
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_params = test_config
        
        # Update config with best parameters if found
        if best_params:
            self.config.update(best_params)
            
    def _run_backtest(self, strategy, historical_data):
        """Run a backtest for parameter optimization (simplified)"""
        # This is a simplified backtest implementation
        # In a real system, this would be much more sophisticated
        
        # Placeholder for actual backtest logic
        # Would simulate running the strategy on historical data
        
        # Return mock performance metrics for now
        return {
            'profit_factor': 1.5,
            'win_rate': 0.6,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.1
        }
    
    def get_current_status(self):
        """Get current strategy status for monitoring and debugging"""
        return {
            'market_regime': self.current_regime.name,
            'open_positions': self.open_positions,
            'daily_risk_used': self.daily_risk_used,
            'performance': self.performance_metrics,
            'config': self.config
        }