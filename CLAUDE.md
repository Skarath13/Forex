# CLAUDE.md - Forex Trading Strategy Project

## Project Overview
This is an adaptive forex trading strategy system that analyzes market conditions and generates trading signals based on different market regimes. The system uses technical indicators, risk management, and multi-timeframe analysis to make trading decisions.

## Running the Project

### Prerequisites
- Python 3.7+
- Required dependencies (install using `pip3 install -r requirements.txt`):
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - yfinance
  - pandas-datareader
  - reportlab
  - lxml

### Commands
To run the strategy with generated sample data:
```
python3 -m src.main
```

To run the strategy with real historical forex data from Yahoo Finance:
```
python3 run_with_real_data.py
```

## Code Structure
- `src/main.py`: Entry point that runs the strategy with sample data
- `run_with_real_data.py`: Script to run strategy with real historical data
- `src/strategies/adaptive_strategy.py`: Core strategy implementation with market regime detection
- `src/utils/`: Utility modules
  - `indicators.py`: Technical indicator calculations
  - `risk_management.py`: Risk management functions
  - `market_analysis.py`: Market analysis utilities
  - `data_fetcher.py`: Module to fetch real historical forex data
  - `report_generator.py`: Module to generate detailed PDF reports

## Features
- Multi-timeframe analysis (1H, 4H, D)
- Market regime detection (Trending, Ranging, Volatile)
- Real-time signal generation based on current market conditions
- Comprehensive risk management
- Historical data backtesting
- Performance metrics and analysis
- PDF report generation
- Real market data from Yahoo Finance

## Data Sources
- Generated sample data (default)
- Real historical forex data from Yahoo Finance
- Supported forex pairs:
  - EUR/USD, GBP/USD, USD/JPY, AUD/USD (default)
  - Also supports: USD/CHF, NZD/USD, USD/CAD, EUR/JPY, GBP/JPY, EUR/GBP

## Maintenance Notes
1. There are FutureWarnings in pandas code that should be addressed:
   - Use `.iloc[]` instead of list index notation for Series objects
   - Update lines in adaptive_strategy.py that use indexing like `analysis['bb_width'][-1]`

2. Yahoo Finance API limitations:
   - The free API may have rate limits and data availability constraints
   - Data might not be available for very recent timeframes (especially intraday)
   - The API could change without notice, requiring updates to the data_fetcher.py module

## Development Tasks
- Fix pandas FutureWarnings
- Add unit tests for strategy components
- Implement additional data sources
- Improve visualization with more detailed plots
- Add machine learning predictions for market direction

## Notes for Claude
When asked to run the project, use:
```
python3 -m src.main                # For sample data
python3 run_with_real_data.py      # For real historical data
```

To fix FutureWarnings, replace Series index syntax like `series[-1]` with `series.iloc[-1]`.