# Adaptive Forex Trading Strategy

This project implements a comprehensive, adaptive Forex trading strategy that automatically selects appropriate trading approaches based on current market conditions. The strategy dynamically adjusts to trending, ranging, or volatile markets while integrating advanced risk management techniques.

## Features

### Multi-Timeframe Analysis
- Analyzes market data across multiple timeframes (1H, 4H, D)
- Uses larger timeframes for trend confirmation
- Identifies suitable entries on smaller timeframes

### Market Regime Detection
- Automatically identifies market conditions (trending, ranging, volatile)
- Applies different trading strategies for each regime:
  - Trend following for trending markets
  - Mean reversion for ranging markets
  - Volatility-based approaches for volatile conditions

### Advanced Technical Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- Stochastic Oscillator
- Ichimoku Cloud
- Divergence detection
- Support and resistance levels

### Risk Management
- Position sizing based on account risk percentage
- Adjustable risk-per-trade and daily risk limits
- Dynamic stop loss and take profit placement
- Correlation-based position filtering
- Pyramiding for strong trend continuation
- Trailing stops for profit maximization

### Market Analysis
- Trend strength and direction assessment
- Correlation calculation between instruments
- Market structure analysis
- News event filtering
- Sentiment analysis integration
- Trading session optimization

### Execution Features
- Signal strength quantification
- Entry, exit, and position adjustment logic
- Performance tracking and metrics
- Parameter optimization capabilities

## Project Structure

```
/Forex
  ├── src/
  │   ├── strategies/
  │   │   └── adaptive_strategy.py
  │   ├── utils/
  │   │   ├── indicators.py
  │   │   ├── risk_management.py
  │   │   └── market_analysis.py
  │   ├── data/
  │   └── main.py
  └── README.md
```

## How It Works

1. **Market Analysis**: The strategy begins by analyzing market data across multiple timeframes, calculating technical indicators, and determining the current market regime.

2. **Signal Generation**: Based on the market analysis, the strategy generates trading signals with different approaches for trending, ranging, and volatile markets.

3. **Risk Assessment**: Before execution, each signal is evaluated for risk management, including position sizing, correlation risk, and daily risk limits.

4. **Trade Execution**: Trades are executed with precise entry, stop loss, and take profit levels based on market volatility and structure.

5. **Position Management**: Open positions are actively managed with trailing stops, pyramid opportunities, and early exit signals when market conditions change.

6. **Performance Tracking**: The strategy continuously monitors and calculates performance metrics to evaluate its effectiveness.

## Getting Started

### Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib (for visualization)

### Installation

1. Clone this repository
2. Install required dependencies:
   ```
   pip install numpy pandas matplotlib
   ```

### Running the Strategy

```python
from src.strategies.adaptive_strategy import AdaptiveStrategy

# Initialize the strategy
strategy = AdaptiveStrategy()

# Analyze markets
analysis = strategy.analyze_markets(market_data)

# Generate signals
signals = strategy.generate_signals(analysis)

# Execute signals
actions = strategy.execute_signals(signals, market_data)
```

## Customization

The strategy can be customized through the configuration dictionary, allowing you to adjust:

- Risk parameters
- Technical indicator settings
- Execution preferences
- Filter thresholds
- Advanced features

## Extending the Strategy

This framework is designed to be extensible. You can add new:

- Technical indicators
- Market regime detection methods
- Signal generation approaches
- Risk management techniques

## Disclaimer

This trading strategy is provided for educational and research purposes only. Past performance is not indicative of future results. Trading financial markets involves substantial risk of loss and is not suitable for all investors.