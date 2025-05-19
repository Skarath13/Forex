# PyFxTrader_SimpleMA - v0.2

An advanced Python Forex trading system implementing institutional trading concepts including Volume Profile analysis, market structure identification, and session-based trading. The system now includes a comprehensive backtesting framework with detailed performance analytics.

**Disclaimer:** This is experimental software. Trading Forex involves significant risk of loss. Use this software at your own risk. **Always test thoroughly on a demo account before considering live trading.** This is not financial advice.

## Table of Contents

1.  [Features (v0.1)](#features-v01)
2.  [Project Structure](#project-structure)
3.  [Prerequisites](#prerequisites)
4.  [Installation](#installation)
5.  [Configuration](#configuration)
6.  [Usage](#usage)
7.  [Strategy Details](#strategy-details)
8.  [Future Enhancements (Roadmap)](#future-enhancements-roadmap)
9.  [Contributing](#contributing)
10. [License](#license)

## Features (v0.2)

### Institutional Trading Features:
*   **Volume Profile Analysis:**
    *   Point of Control (POC) identification
    *   Value Area High/Low (VAH/VAL) calculations
    *   High Volume Nodes (HVN) and Low Volume Nodes (LVN) detection
    *   Dynamic volume distribution analysis

*   **Market Structure Analysis:**
    *   Automatic support and resistance level identification
    *   Swing high/low detection
    *   Trend strength calculation using linear regression
    *   Price level clustering for significant zones

*   **Session-Based Trading:**
    *   London session (08:00-16:00)
    *   New York session (13:00-21:00)
    *   Automated session filtering for optimal liquidity
    *   Higher volatility modeling during active sessions

*   **Advanced Risk Management:**
    *   ATR-based stop loss and take profit
    *   Position sizing based on account risk
    *   Maximum daily trade limits
    *   Risk-reward ratio optimization

*   **Enhanced Strategy Logic:**
    *   MA crossover with volume confirmation
    *   RSI divergence detection
    *   Price action patterns (engulfing patterns)
    *   Multiple timeframe analysis support

### Backtesting Framework:
*   **Comprehensive Performance Metrics:**
    *   Win rate and profit factor
    *   Maximum drawdown analysis
    *   Sharpe ratio calculation
    *   Monthly and pair-wise performance breakdown

*   **Visualization Suite:**
    *   Equity curve with drawdown visualization
    *   Trade distribution analysis
    *   Volume profile charts
    *   Performance heatmaps

*   **Historical Data Generation:**
    *   Realistic forex data simulation
    *   Session-based volatility patterns
    *   Trend and cycle incorporation

## Project Structure

```
pyfxtrader_simplema/
├── main.py                     # Real-time trading execution
├── config.ini                  # Configuration settings
├── data_handler.py            # Data fetching and management
├── strategy.py                # Basic MA crossover strategy
├── advanced_strategy.py       # Institutional trading strategy
├── broker_interface.py        # Broker interaction simulation
├── risk_manager.py           # Risk management logic
├── backtest_system.py        # Backtesting engine and metrics
├── backtest_forex.py         # Main backtesting script
├── utils.py                  # Utility functions
├── logs/                     # Log files directory
├── requirements.txt          # Python dependencies
├── backtest_results.png      # Generated backtest visualization
├── volume_profile_example.png # Volume profile visualization
└── trade_log.csv             # Detailed trade history
```

## Prerequisites

*   Python 3.8+
*   pip (Python package installer)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/PyFxTrader_SimpleMA.git
    cd PyFxTrader_SimpleMA
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

All settings are managed in the `config.ini` file. The default configuration is already set up, but you can modify it as needed:

```ini
[DEFAULT]
# List of Forex pairs to trade, comma-separated
TRADING_PAIRS = EUR/USD,USD/JPY,GBP/USD,AUD/USD,USD/CAD

# Trading Strategy Parameters
TIMEFRAME = H1       # e.g., M1, M5, M15, M30, H1, H4, D1
SHORT_MA_PERIOD = 50
LONG_MA_PERIOD = 200

# Risk Management
POSITION_SIZE = 1000 # Units per trade

# Broker API (Placeholders for now)
# For a real broker, you'd put API keys, account IDs here.
BROKER_API_KEY = YOUR_API_KEY_HERE
BROKER_ACCOUNT_ID = YOUR_ACCOUNT_ID_HERE
BROKER_ENVIRONMENT = practice # or live

# Main Loop
LOOP_INTERVAL_SECONDS = 3600 # Check every hour for H1 timeframe
```

## Usage

### Running Historical Backtest (Recommended First Step)

```bash
python backtest_forex.py
```

This will:
- Run a 1-year backtest on major forex pairs
- Generate detailed performance metrics
- Create visualization charts
- Save trade logs to CSV

Output files:
- `backtest_results.png` - Comprehensive performance visualization
- `volume_profile_example.png` - Sample volume profile analysis
- `trade_log.csv` - Detailed log of all trades

### Running Real-Time Trading (Simulation)

```bash
python main.py
```

This will:
- Connect to simulated broker
- Monitor configured pairs in real-time
- Execute trades based on signals
- Log all activities

## Strategy Details

### Institutional Trading Strategy (v0.2)

The enhanced strategy combines multiple institutional trading concepts:

#### Volume Profile Analysis:
- **Point of Control (POC):** Price level with highest volume
- **Value Area:** Price range containing 70% of trading volume
- **High/Low Volume Nodes:** Areas of significant or minimal trading activity

#### Entry Conditions (LONG):
1. MA crossover (fast > slow)
2. Price near Value Area Low or support level
3. RSI not overbought (< 70)
4. Positive trend strength
5. Adequate volume (> 70% of average)
6. Active trading session (London/NY)

#### Entry Conditions (SHORT):
1. MA crossover (fast < slow)
2. Price near Value Area High or resistance level
3. RSI not oversold (> 30)
4. Negative trend strength
5. Adequate volume (> 70% of average)
6. Active trading session (London/NY)

#### Risk Management:
- **Stop Loss:** 2 × ATR from entry
- **Take Profit:** Risk-reward ratio × stop distance
- **Position Sizing:** Based on 1% account risk
- **Trailing Stop:** Activated after 1% favorable movement
- **Max Daily Trades:** 3 per day (institutional discipline)

## Future Enhancements (Roadmap)

### v0.2 - Broker Integration:
- Connect to real Forex broker APIs (e.g., OANDA, IG, FXCM, Alpaca).
- Real-time data fetching.
- Live order execution.

### v0.3 - Advanced Strategy Components:
- Incorporate other indicators (RSI, MACD, Bollinger Bands).
- Implement Stop Loss (SL) and Take Profit (TP) orders.
- Trailing Stop Loss.

### v0.4 - Enhanced Risk Management:
- Position sizing based on account balance percentage.
- Max concurrent trades.
- Daily/Weekly loss limits.

### v0.5 - Backtesting Framework:
- Test strategies on historical data.
- Performance metrics (profit factor, Sharpe ratio, drawdown).

### v0.6 - Data Handling:
- Store historical data locally (e.g., CSV, SQLite, database).
- More robust live data stream handling.

### v0.7 - User Interface / Dashboard:
- Web interface (Flask/Django) or desktop GUI (Tkinter/PyQt) for monitoring.

### v0.8 - Notifications:
- Email, SMS, or push notifications for critical events.

### v0.9 - Error Handling & Resilience:
- More robust error handling and automatic reconnection.

### v1.0 - Unit & Integration Tests:
- Ensure code reliability with comprehensive testing.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` file for more information.

---

**Remember:** Always test on a demo account first. Forex trading carries substantial risk and is not suitable for all investors.# Forex
