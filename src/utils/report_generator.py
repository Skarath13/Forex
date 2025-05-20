import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
from enum import Enum

class MarketCycle(Enum):
    BULL = 'Bullish'
    BEAR = 'Bearish'
    SIDEWAYS = 'Sideways'
    VOLATILE = 'Volatile'
    RECOVERY = 'Recovery'
    UNKNOWN = 'Unknown'

def detect_market_cycle(data, window=20):
    """
    Detect the market cycle from price data
    
    Args:
        data: DataFrame with OHLCV data
        window: Window size for moving averages
        
    Returns:
        MarketCycle enum value and confidence score
    """
    if len(data) < window * 2:
        return MarketCycle.UNKNOWN, 0.0
        
    # Calculate some indicators for market cycle detection
    data = data.copy()
    data['ma_short'] = data['close'].rolling(window=window).mean()
    data['ma_long'] = data['close'].rolling(window=window*2).mean()
    data['volatility'] = data['close'].rolling(window=window).std() / data['close'].rolling(window=window).mean()
    
    # Get recent data
    recent = data.dropna().iloc[-window:]
    
    # Calculate trend metrics
    trend = (recent['ma_short'].iloc[-1] - recent['ma_short'].iloc[0]) / recent['ma_short'].iloc[0]
    ma_cross = recent['ma_short'].iloc[-1] > recent['ma_long'].iloc[-1]
    vol = recent['volatility'].mean()
    
    # Determine market cycle
    if trend > 0.03 and ma_cross:
        if vol > 0.015:
            return MarketCycle.VOLATILE, 0.7
        return MarketCycle.BULL, 0.8
    elif trend < -0.03:
        if vol > 0.015:
            return MarketCycle.VOLATILE, 0.7
        return MarketCycle.BEAR, 0.8
    elif -0.01 < trend < 0.01:
        return MarketCycle.SIDEWAYS, 0.6
    elif trend > 0 and not ma_cross:
        return MarketCycle.RECOVERY, 0.5
    else:
        return MarketCycle.UNKNOWN, 0.3

def calculate_advanced_metrics(trade_history):
    """
    Calculate advanced trading metrics
    
    Args:
        trade_history: List of trade dictionaries
        
    Returns:
        Dictionary of metrics
    """
    if not trade_history:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'win_loss_ratio': 0,
            'avg_trade_duration': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'expectancy': 0,
            'avg_return': 0,
            'std_return': 0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'profit_by_day': {},
            'profit_by_instrument': {}
        }
    
    # Convert to DataFrame for easier analysis
    trades_df = pd.DataFrame(trade_history)
    
    # Filter to closed trades
    closed_trades = trades_df[trades_df['status'] == 'closed'].copy()
    
    if len(closed_trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'win_loss_ratio': 0,
            'avg_trade_duration': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'expectancy': 0,
            'avg_return': 0,
            'std_return': 0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'profit_by_day': {},
            'profit_by_instrument': {}
        }
    
    # Add date columns if they don't exist
    if 'entry_time' in closed_trades.columns and not isinstance(closed_trades['entry_time'].iloc[0], datetime):
        closed_trades['entry_date'] = pd.to_datetime(closed_trades['entry_time'], unit='s')
    else:
        closed_trades['entry_date'] = pd.to_datetime(closed_trades['entry_time'])
        
    if 'exit_time' in closed_trades.columns and not isinstance(closed_trades['exit_time'].iloc[0], datetime):
        closed_trades['exit_date'] = pd.to_datetime(closed_trades['exit_time'], unit='s')
    else:
        closed_trades['exit_date'] = pd.to_datetime(closed_trades['exit_time'])
    
    # Basic metrics
    total_trades = len(closed_trades)
    winning_trades = closed_trades[closed_trades['profit_amount'] > 0]
    losing_trades = closed_trades[closed_trades['profit_amount'] <= 0]
    
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    
    win_rate = win_count / total_trades if total_trades > 0 else 0
    
    total_profit = winning_trades['profit_amount'].sum() if not winning_trades.empty else 0
    total_loss = abs(losing_trades['profit_amount'].sum()) if not losing_trades.empty else 0
    
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    avg_win = winning_trades['profit_amount'].mean() if not winning_trades.empty else 0
    avg_loss = abs(losing_trades['profit_amount'].mean()) if not losing_trades.empty else 0
    
    largest_win = winning_trades['profit_amount'].max() if not winning_trades.empty else 0
    largest_loss = abs(losing_trades['profit_amount'].min()) if not losing_trades.empty else 0
    
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    # Calculate trade durations
    closed_trades['duration'] = (closed_trades['exit_date'] - closed_trades['entry_date']).dt.total_seconds()
    avg_trade_duration = closed_trades['duration'].mean()
    
    # Calculate equity curve and drawdown
    closed_trades = closed_trades.sort_values('exit_date')
    closed_trades['cumulative_profit'] = closed_trades['profit_amount'].cumsum()
    
    # Calculate drawdown
    closed_trades['peak'] = closed_trades['cumulative_profit'].cummax()
    closed_trades['drawdown'] = closed_trades['peak'] - closed_trades['cumulative_profit']
    max_drawdown = closed_trades['drawdown'].max()
    max_drawdown_pct = max_drawdown / closed_trades['peak'].max() if closed_trades['peak'].max() > 0 else 0
    
    # Calculate Sharpe ratio (simplified)
    returns = closed_trades['profit_amount'].values
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = avg_return / std_return if std_return > 0 else 0
    
    # Calculate Sortino ratio (using only negative returns for denominator)
    negative_returns = returns[returns < 0]
    downside_dev = np.std(negative_returns) if len(negative_returns) > 0 else 1
    sortino_ratio = avg_return / downside_dev if downside_dev > 0 else 0
    
    # Calculate expectancy
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    # Find consecutive wins/losses
    if not closed_trades.empty:
        closed_trades['is_win'] = closed_trades['profit_amount'] > 0
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for is_win in closed_trades['is_win']:
            if is_win:
                win_streak += 1
                loss_streak = 0
            else:
                loss_streak += 1
                win_streak = 0
                
            max_win_streak = max(max_win_streak, win_streak)
            max_loss_streak = max(max_loss_streak, loss_streak)
    else:
        max_win_streak = 0
        max_loss_streak = 0
    
    # Profit by day
    if not closed_trades.empty:
        closed_trades['exit_day'] = closed_trades['exit_date'].dt.date
        profit_by_day = closed_trades.groupby('exit_day')['profit_amount'].sum().to_dict()
    else:
        profit_by_day = {}
        
    # Profit by instrument
    if not closed_trades.empty and 'instrument' in closed_trades.columns:
        profit_by_instrument = closed_trades.groupby('instrument')['profit_amount'].sum().to_dict()
    else:
        profit_by_instrument = {}
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'win_loss_ratio': win_loss_ratio,
        'avg_trade_duration': avg_trade_duration,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'expectancy': expectancy,
        'avg_return': avg_return,
        'std_return': std_return,
        'consecutive_wins': max_win_streak,
        'consecutive_losses': max_loss_streak,
        'profit_by_day': profit_by_day,
        'profit_by_instrument': profit_by_instrument
    }

def analyze_metrics_by_timeframe(trade_history):
    """
    Analyze metrics by timeframe
    
    Args:
        trade_history: List of trade dictionaries
        
    Returns:
        Dictionary of metrics by timeframe
    """
    # Group trades by timeframe
    timeframe_trades = {}
    
    for trade in trade_history:
        timeframe = trade.get('timeframe', 'unknown')
        if timeframe not in timeframe_trades:
            timeframe_trades[timeframe] = []
            
        timeframe_trades[timeframe].append(trade)
    
    # Calculate metrics for each timeframe
    timeframe_metrics = {}
    for timeframe, trades in timeframe_trades.items():
        timeframe_metrics[timeframe] = calculate_advanced_metrics(trades)
        
    return timeframe_metrics

def analyze_metrics_by_instrument(trade_history):
    """
    Analyze metrics by instrument
    
    Args:
        trade_history: List of trade dictionaries
        
    Returns:
        Dictionary of metrics by instrument
    """
    # Group trades by instrument
    instrument_trades = {}
    
    for trade in trade_history:
        instrument = trade.get('instrument', 'unknown')
        if instrument not in instrument_trades:
            instrument_trades[instrument] = []
            
        instrument_trades[instrument].append(trade)
    
    # Calculate metrics for each instrument
    instrument_metrics = {}
    for instrument, trades in instrument_trades.items():
        instrument_metrics[instrument] = calculate_advanced_metrics(trades)
        
    return instrument_metrics

def analyze_metrics_by_market_cycle(trade_history, market_data):
    """
    Analyze metrics by market cycle
    
    Args:
        trade_history: List of trade dictionaries
        market_data: Dictionary of market data
        
    Returns:
        Dictionary of metrics by market cycle
    """
    if not trade_history or not market_data:
        return {}
        
    # Detect market cycle for each trade
    for trade in trade_history:
        if 'instrument' in trade and 'timeframe' in trade and trade['instrument'] in market_data:
            instrument = trade['instrument']
            timeframe = trade['timeframe']
            
            if timeframe in market_data[instrument]:
                data = market_data[instrument][timeframe]
                exit_time = trade.get('exit_time')
                
                if exit_time:
                    # Find data up to exit time
                    if isinstance(exit_time, (int, float)):
                        exit_date = datetime.fromtimestamp(exit_time)
                    else:
                        exit_date = exit_time
                        
                    data_subset = data[data.index <= exit_date].copy()
                    
                    if len(data_subset) > 40:  # Need at least 40 bars for cycle detection
                        cycle, confidence = detect_market_cycle(data_subset)
                        trade['market_cycle'] = cycle.value
                    else:
                        trade['market_cycle'] = MarketCycle.UNKNOWN.value
                else:
                    trade['market_cycle'] = MarketCycle.UNKNOWN.value
            else:
                trade['market_cycle'] = MarketCycle.UNKNOWN.value
        else:
            trade['market_cycle'] = MarketCycle.UNKNOWN.value
    
    # Group trades by market cycle
    cycle_trades = {}
    
    for trade in trade_history:
        cycle = trade.get('market_cycle', MarketCycle.UNKNOWN.value)
        if cycle not in cycle_trades:
            cycle_trades[cycle] = []
            
        cycle_trades[cycle].append(trade)
    
    # Calculate metrics for each market cycle
    cycle_metrics = {}
    for cycle, trades in cycle_trades.items():
        cycle_metrics[cycle] = calculate_advanced_metrics(trades)
        
    return cycle_metrics

def create_equity_curve_chart(trade_history, filename='equity_curve.png'):
    """
    Create an equity curve chart
    
    Args:
        trade_history: List of trade dictionaries
        filename: Filename to save the chart to
        
    Returns:
        Path to saved chart
    """
    if not trade_history:
        # Create empty chart
        plt.figure(figsize=(10, 6))
        plt.title('Equity Curve - No trades')
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        return filename
    
    # Convert to DataFrame
    trades_df = pd.DataFrame([t for t in trade_history if t.get('status') == 'closed'])
    
    if trades_df.empty:
        # Create empty chart
        plt.figure(figsize=(10, 6))
        plt.title('Equity Curve - No closed trades')
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        return filename
    
    # Ensure we have datetime exit dates
    if 'exit_time' in trades_df.columns:
        if isinstance(trades_df['exit_time'].iloc[0], (int, float)):
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_time'], unit='s')
        else:
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_time'])
    
    # Sort by exit date
    trades_df = trades_df.sort_values('exit_date')
    
    # Calculate cumulative profit
    trades_df['cumulative_profit'] = trades_df['profit_amount'].cumsum()
    
    # Calculate drawdown
    trades_df['peak'] = trades_df['cumulative_profit'].cummax()
    trades_df['drawdown'] = trades_df['peak'] - trades_df['cumulative_profit']
    trades_df['drawdown_pct'] = trades_df['drawdown'] / trades_df['peak'].replace(0, np.nan)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot equity curve
    plt.subplot(2, 1, 1)
    plt.plot(trades_df['exit_date'], trades_df['cumulative_profit'], label='Equity Curve', color='blue')
    plt.plot(trades_df['exit_date'], trades_df['peak'], label='Equity Peak', color='green', alpha=0.5, linestyle='--')
    plt.fill_between(trades_df['exit_date'], trades_df['cumulative_profit'], trades_df['peak'], 
                     color='red', alpha=0.2, label='Drawdown')
    
    plt.title('Trading Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profit')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot drawdown
    plt.subplot(2, 1, 2)
    plt.plot(trades_df['exit_date'], trades_df['drawdown_pct'] * 100, label='Drawdown %', color='red')
    plt.axhline(y=0, color='green', linestyle='-', alpha=0.3)
    
    plt.title('Drawdown (%)')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # Invert y-axis to show drawdowns as negative
    
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filename

def create_win_loss_chart(trade_history, filename='win_loss.png'):
    """
    Create a win/loss histogram
    
    Args:
        trade_history: List of trade dictionaries
        filename: Filename to save the chart to
        
    Returns:
        Path to saved chart
    """
    if not trade_history:
        # Create empty chart
        plt.figure(figsize=(8, 6))
        plt.title('Win/Loss Distribution - No trades')
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        return filename
    
    # Convert to DataFrame
    trades_df = pd.DataFrame([t for t in trade_history if t.get('status') == 'closed'])
    
    if trades_df.empty:
        # Create empty chart
        plt.figure(figsize=(8, 6))
        plt.title('Win/Loss Distribution - No closed trades')
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        return filename
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Set Seaborn style
    sns.set_style('whitegrid')
    
    # Create histogram of profit amounts
    sns.histplot(trades_df['profit_amount'], kde=True, bins=20)
    
    plt.axvline(x=0, color='red', linestyle='--', label='Breakeven')
    
    plt.title('Win/Loss Distribution')
    plt.xlabel('Profit/Loss Amount')
    plt.ylabel('Number of Trades')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filename

def create_timeframe_comparison_chart(timeframe_metrics, filename='timeframe_comparison.png'):
    """
    Create a timeframe comparison chart
    
    Args:
        timeframe_metrics: Dictionary of metrics by timeframe
        filename: Filename to save the chart to
        
    Returns:
        Path to saved chart
    """
    if not timeframe_metrics:
        # Create empty chart
        plt.figure(figsize=(10, 6))
        plt.title('Timeframe Comparison - No data')
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        return filename
    
    # Prepare data for plotting
    timeframes = list(timeframe_metrics.keys())
    win_rates = [timeframe_metrics[tf]['win_rate'] * 100 for tf in timeframes]
    profit_factors = [min(timeframe_metrics[tf]['profit_factor'], 5) for tf in timeframes]  # Cap at 5 for visualization
    max_drawdowns = [timeframe_metrics[tf]['max_drawdown_pct'] * 100 for tf in timeframes]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot win rates
    axes[0].bar(timeframes, win_rates, color='green')
    axes[0].set_title('Win Rate by Timeframe')
    axes[0].set_ylabel('Win Rate (%)')
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3)
    
    # Plot profit factors
    axes[1].bar(timeframes, profit_factors, color='blue')
    axes[1].set_title('Profit Factor by Timeframe')
    axes[1].set_ylabel('Profit Factor')
    axes[1].grid(True, alpha=0.3)
    
    # Plot max drawdowns
    axes[2].bar(timeframes, max_drawdowns, color='red')
    axes[2].set_title('Max Drawdown by Timeframe')
    axes[2].set_ylabel('Max Drawdown (%)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filename

def create_market_cycle_chart(cycle_metrics, filename='market_cycle.png'):
    """
    Create a market cycle comparison chart
    
    Args:
        cycle_metrics: Dictionary of metrics by market cycle
        filename: Filename to save the chart to
        
    Returns:
        Path to saved chart
    """
    if not cycle_metrics:
        # Create empty chart
        plt.figure(figsize=(10, 6))
        plt.title('Market Cycle Comparison - No data')
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        return filename
    
    # Prepare data for plotting
    cycles = list(cycle_metrics.keys())
    win_rates = [cycle_metrics[c]['win_rate'] * 100 for c in cycles]
    profit_factors = [min(cycle_metrics[c]['profit_factor'], 5) for c in cycles]  # Cap at 5 for visualization
    profits = [sum(cycle_metrics[c]['profit_by_day'].values()) if cycle_metrics[c]['profit_by_day'] else 0 for c in cycles]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot win rates
    axes[0].bar(cycles, win_rates, color='green')
    axes[0].set_title('Win Rate by Market Cycle')
    axes[0].set_ylabel('Win Rate (%)')
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
    
    # Plot profit factors
    axes[1].bar(cycles, profit_factors, color='blue')
    axes[1].set_title('Profit Factor by Market Cycle')
    axes[1].set_ylabel('Profit Factor')
    axes[1].grid(True, alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
    
    # Plot total profits
    axes[2].bar(cycles, profits, color='purple')
    axes[2].set_title('Total Profit by Market Cycle')
    axes[2].set_ylabel('Total Profit')
    axes[2].grid(True, alpha=0.3)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filename

def create_instruments_chart(instrument_metrics, filename='instruments.png'):
    """
    Create an instruments comparison chart
    
    Args:
        instrument_metrics: Dictionary of metrics by instrument
        filename: Filename to save the chart to
        
    Returns:
        Path to saved chart
    """
    if not instrument_metrics:
        # Create empty chart
        plt.figure(figsize=(10, 6))
        plt.title('Instrument Comparison - No data')
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        return filename
    
    # Prepare data for plotting
    instruments = list(instrument_metrics.keys())
    win_rates = [instrument_metrics[i]['win_rate'] * 100 for i in instruments]
    total_profits = [sum(instrument_metrics[i]['profit_by_day'].values()) if instrument_metrics[i]['profit_by_day'] else 0 for i in instruments]
    trade_counts = [instrument_metrics[i]['total_trades'] for i in instruments]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot win rates
    axes[0].bar(instruments, win_rates, color='green')
    axes[0].set_title('Win Rate by Instrument')
    axes[0].set_ylabel('Win Rate (%)')
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3)
    
    # Plot total profits
    axes[1].bar(instruments, total_profits, color='blue')
    axes[1].set_title('Total Profit by Instrument')
    axes[1].set_ylabel('Total Profit')
    axes[1].grid(True, alpha=0.3)
    
    # Plot trade counts
    axes[2].bar(instruments, trade_counts, color='orange')
    axes[2].set_title('Number of Trades by Instrument')
    axes[2].set_ylabel('Number of Trades')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filename

def save_charts_to_memory(trade_history, timeframe_metrics, cycle_metrics, instrument_metrics):
    """
    Save all charts to memory for inclusion in the PDF
    
    Args:
        trade_history: List of trade dictionaries
        timeframe_metrics: Dictionary of metrics by timeframe
        cycle_metrics: Dictionary of metrics by market cycle
        instrument_metrics: Dictionary of metrics by instrument
        
    Returns:
        Dictionary of chart filenames to PIL Image objects
    """
    equity_curve_buffer = BytesIO()
    win_loss_buffer = BytesIO()
    timeframe_buffer = BytesIO()
    market_cycle_buffer = BytesIO()
    instruments_buffer = BytesIO()
    
    # Create and save charts
    create_equity_curve_chart(trade_history, equity_curve_buffer)
    create_win_loss_chart(trade_history, win_loss_buffer)
    create_timeframe_comparison_chart(timeframe_metrics, timeframe_buffer)
    create_market_cycle_chart(cycle_metrics, market_cycle_buffer)
    create_instruments_chart(instrument_metrics, instruments_buffer)
    
    # Reset buffer positions to beginning
    equity_curve_buffer.seek(0)
    win_loss_buffer.seek(0)
    timeframe_buffer.seek(0)
    market_cycle_buffer.seek(0)
    instruments_buffer.seek(0)
    
    # Create Image objects
    from PIL import Image
    images = {
        'equity_curve': Image.open(equity_curve_buffer),
        'win_loss': Image.open(win_loss_buffer),
        'timeframe': Image.open(timeframe_buffer),
        'market_cycle': Image.open(market_cycle_buffer),
        'instruments': Image.open(instruments_buffer)
    }
    
    return images

def generate_pdf_report(strategy, trade_history, market_data, output_file='forex_trading_report.pdf'):
    """
    Generate a detailed PDF report of trading performance
    
    Args:
        strategy: The trading strategy instance
        trade_history: List of trade dictionaries
        market_data: Dictionary of market data
        output_file: Output PDF filename
        
    Returns:
        Path to the generated PDF file
    """
    # Calculate metrics
    overall_metrics = calculate_advanced_metrics(trade_history)
    timeframe_metrics = analyze_metrics_by_timeframe(trade_history)
    cycle_metrics = analyze_metrics_by_market_cycle(trade_history, market_data)
    instrument_metrics = analyze_metrics_by_instrument(trade_history)
    
    # Set PDF document properties
    doc = SimpleDocTemplate(
        output_file,
        pagesize=landscape(letter),
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    heading2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Custom styles
    centered_style = ParagraphStyle(
        'centered',
        parent=styles['Normal'],
        alignment=1,
        spaceAfter=10
    )
    
    # Create content elements
    elements = []
    
    # Title
    elements.append(Paragraph('Forex Trading Strategy Performance Report', title_style))
    elements.append(Paragraph(f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', centered_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Overall metrics
    elements.append(Paragraph('Overall Performance Metrics', heading_style))
    
    # Create a table for the metrics
    overall_data = [
        ['Metric', 'Value', 'Metric', 'Value'],
        ['Total Trades', f"{overall_metrics['total_trades']:,}", 'Win Rate', f"{overall_metrics['win_rate']*100:.2f}%"],
        ['Profit Factor', f"{overall_metrics['profit_factor']:.2f}", 'Win/Loss Ratio', f"{overall_metrics['win_loss_ratio']:.2f}"],
        ['Average Win', f"{overall_metrics['avg_win']:.4f}", 'Average Loss', f"{overall_metrics['avg_loss']:.4f}"],
        ['Largest Win', f"{overall_metrics['largest_win']:.4f}", 'Largest Loss', f"{overall_metrics['largest_loss']:.4f}"],
        ['Max Drawdown', f"{overall_metrics['max_drawdown']:.4f}", 'Max Drawdown %', f"{overall_metrics['max_drawdown_pct']*100:.2f}%"],
        ['Sharpe Ratio', f"{overall_metrics['sharpe_ratio']:.2f}", 'Sortino Ratio', f"{overall_metrics['sortino_ratio']:.2f}"],
        ['Expectancy', f"{overall_metrics['expectancy']:.4f}", 'Avg Trade Duration', f"{overall_metrics['avg_trade_duration']/3600:.2f} hours"],
        ['Consecutive Wins', f"{overall_metrics['consecutive_wins']}", 'Consecutive Losses', f"{overall_metrics['consecutive_losses']}"],
    ]
    
    overall_table = Table(overall_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    overall_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (3, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (3, 0), colors.black),
        ('ALIGN', (0, 0), (3, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (3, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (3, 0), 12),
        ('BACKGROUND', (0, 1), (3, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    elements.append(overall_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Generate chart images and save to memory
    chart_images = {}
    
    # Equity curve
    temp_filename = os.path.join(os.path.dirname(output_file), 'temp_equity_curve.png')
    chart_images['equity_curve'] = create_equity_curve_chart(trade_history, temp_filename)
    elements.append(Paragraph('Equity Curve and Drawdown', heading2_style))
    elements.append(Image(temp_filename, width=9*inch, height=5*inch))
    elements.append(Spacer(1, 0.2*inch))
    
    # Win/Loss distribution
    temp_filename = os.path.join(os.path.dirname(output_file), 'temp_win_loss.png')
    chart_images['win_loss'] = create_win_loss_chart(trade_history, temp_filename)
    elements.append(Paragraph('Win/Loss Distribution', heading2_style))
    elements.append(Image(temp_filename, width=7*inch, height=4*inch))
    elements.append(Spacer(1, 0.2*inch))
    
    # Timeframe comparison
    temp_filename = os.path.join(os.path.dirname(output_file), 'temp_timeframe.png')
    chart_images['timeframe'] = create_timeframe_comparison_chart(timeframe_metrics, temp_filename)
    elements.append(Paragraph('Performance by Timeframe', heading_style))
    elements.append(Image(temp_filename, width=9*inch, height=3*inch))
    elements.append(Spacer(1, 0.2*inch))
    
    # Detailed timeframe metrics
    elements.append(Paragraph('Detailed Timeframe Metrics', heading2_style))
    
    for timeframe, metrics in timeframe_metrics.items():
        if metrics['total_trades'] > 0:
            elements.append(Paragraph(f'{timeframe} Timeframe', styles['Heading3']))
            
            timeframe_data = [
                ['Metric', 'Value', 'Metric', 'Value'],
                ['Total Trades', f"{metrics['total_trades']:,}", 'Win Rate', f"{metrics['win_rate']*100:.2f}%"],
                ['Profit Factor', f"{metrics['profit_factor']:.2f}", 'Max Drawdown %', f"{metrics['max_drawdown_pct']*100:.2f}%"],
                ['Average Win', f"{metrics['avg_win']:.4f}", 'Average Loss', f"{metrics['avg_loss']:.4f}"],
                ['Expectancy', f"{metrics['expectancy']:.4f}", 'Sharpe Ratio', f"{metrics['sharpe_ratio']:.2f}"],
            ]
            
            tf_table = Table(timeframe_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            tf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (3, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (3, 0), colors.black),
                ('ALIGN', (0, 0), (3, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (3, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (3, 0), 12),
                ('BACKGROUND', (0, 1), (3, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(tf_table)
            elements.append(Spacer(1, 0.2*inch))
    
    # Market cycle analysis
    temp_filename = os.path.join(os.path.dirname(output_file), 'temp_market_cycle.png')
    chart_images['market_cycle'] = create_market_cycle_chart(cycle_metrics, temp_filename)
    elements.append(Paragraph('Performance by Market Cycle', heading_style))
    elements.append(Image(temp_filename, width=9*inch, height=3*inch))
    elements.append(Spacer(1, 0.2*inch))
    
    # Detailed market cycle metrics
    elements.append(Paragraph('Detailed Market Cycle Metrics', heading2_style))
    
    for cycle, metrics in cycle_metrics.items():
        if metrics['total_trades'] > 0:
            elements.append(Paragraph(f'{cycle} Market Cycle', styles['Heading3']))
            
            cycle_data = [
                ['Metric', 'Value', 'Metric', 'Value'],
                ['Total Trades', f"{metrics['total_trades']:,}", 'Win Rate', f"{metrics['win_rate']*100:.2f}%"],
                ['Profit Factor', f"{metrics['profit_factor']:.2f}", 'Max Drawdown %', f"{metrics['max_drawdown_pct']*100:.2f}%"],
                ['Average Win', f"{metrics['avg_win']:.4f}", 'Average Loss', f"{metrics['avg_loss']:.4f}"],
                ['Total Profit', f"{sum(metrics['profit_by_day'].values()) if metrics['profit_by_day'] else 0:.4f}", 'Expectancy', f"{metrics['expectancy']:.4f}"],
            ]
            
            cycle_table = Table(cycle_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            cycle_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (3, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (3, 0), colors.black),
                ('ALIGN', (0, 0), (3, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (3, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (3, 0), 12),
                ('BACKGROUND', (0, 1), (3, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(cycle_table)
            elements.append(Spacer(1, 0.2*inch))
    
    # Instrument comparison
    temp_filename = os.path.join(os.path.dirname(output_file), 'temp_instruments.png')
    chart_images['instruments'] = create_instruments_chart(instrument_metrics, temp_filename)
    elements.append(Paragraph('Performance by Instrument', heading_style))
    elements.append(Image(temp_filename, width=9*inch, height=3*inch))
    elements.append(Spacer(1, 0.2*inch))
    
    # Strategy configuration
    if hasattr(strategy, 'config'):
        elements.append(Paragraph('Strategy Configuration', heading_style))
        
        # Convert config to a table
        config_data = [['Parameter', 'Value']]
        
        for key, value in strategy.config.items():
            if isinstance(value, (int, float, str, bool)):
                config_data.append([key, str(value)])
            elif isinstance(value, list):
                config_data.append([key, ', '.join(map(str, value))])
        
        config_table = Table(config_data, colWidths=[2*inch, 4*inch])
        config_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (1, 0), 12),
            ('BACKGROUND', (0, 1), (1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(config_table)
    
    # Build PDF
    doc.build(elements)
    
    # Cleanup temp files
    for filename in chart_images.values():
        try:
            os.remove(filename)
        except:
            pass
    
    # Return path to the generated PDF
    return output_file