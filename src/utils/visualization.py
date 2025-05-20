import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

def plot_order_flow_analysis(data, order_flow_results, title=None, figsize=(14, 10)):
    """
    Create a comprehensive order flow visualization with institutional activity
    
    Args:
        data: DataFrame with OHLCV price data
        order_flow_results: Results from analyze_order_flow function
        title: Optional title for the plot
        figsize: Size of the figure (width, height)
        
    Returns:
        Matplotlib figure
    """
    # Validate input
    if len(data) == 0 or not isinstance(order_flow_results, dict):
        return None
        
    # Create figure and subplots
    fig = plt.figure(figsize=figsize)
    
    # Calculate grid layout based on available data
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.15)
    
    # Get recent window of data for visualization
    window = min(50, len(data))
    recent_data = data.iloc[-window:].copy()
    
    # Price chart (with volume overlay)
    ax1 = fig.add_subplot(gs[0])
    
    # Create price candles
    plot_price_candles(ax1, recent_data)
    
    # Plot institutional signals if available
    if 'institutional_activity' in order_flow_results and 'signals' in order_flow_results['institutional_activity']:
        plot_institutional_signals(ax1, recent_data, order_flow_results['institutional_activity']['signals'], window)
    
    # Volume Delta (buying vs selling pressure)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Calculate candle colors for volume bars
    candle_colors = np.where(recent_data['close'] >= recent_data['open'], 'green', 'red')
    
    # Plot volume bars
    ax2.bar(range(len(recent_data)), recent_data['volume'], color=candle_colors, alpha=0.7)
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    # Plot cumulative delta
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # Calculate cumulative delta if not already done
    if 'volume_delta' not in recent_data.columns:
        recent_data['candle_type'] = np.where(recent_data['close'] > recent_data['open'], 1, 
                                             np.where(recent_data['close'] < recent_data['open'], -1, 0))
        recent_data['volume_delta'] = recent_data['volume'] * recent_data['candle_type']
        recent_data['cumulative_delta'] = recent_data['volume_delta'].cumsum()
    
    # Plot cumulative delta
    baseline = np.zeros(len(recent_data))
    delta_colors = np.where(recent_data['cumulative_delta'] >= 0, 'green', 'red')
    ax3.fill_between(range(len(recent_data)), recent_data['cumulative_delta'], baseline, 
                     where=recent_data['cumulative_delta'] >= baseline, 
                     color='green', alpha=0.5, interpolate=True)
    ax3.fill_between(range(len(recent_data)), recent_data['cumulative_delta'], baseline, 
                     where=recent_data['cumulative_delta'] < baseline, 
                     color='red', alpha=0.5, interpolate=True)
    ax3.plot(range(len(recent_data)), recent_data['cumulative_delta'], color='blue', linewidth=1.5)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax3.set_ylabel('Cum. Delta')
    ax3.grid(True, alpha=0.3)
    
    # Plot fingerprint (delta efficiency)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    
    # Calculate delta efficiency if available in results
    if 'fingerprint' in order_flow_results:
        periods = []
        efficiencies = []
        for key, value in order_flow_results['fingerprint'].items():
            if 'efficiency' in value:
                period = int(key.split('_')[0])
                periods.append(period)
                efficiencies.append(value['efficiency'])
        
        if periods:
            # Sort by period
            sorted_indices = np.argsort(periods)
            periods = [periods[i] for i in sorted_indices]
            efficiencies = [efficiencies[i] for i in sorted_indices]
            
            # Plot efficiency by period
            colors = ['#0343DF', '#00B7E0', '#95FD44', '#FF9300', '#FF0021']
            ax4.bar(periods, efficiencies, color=colors[:len(periods)], alpha=0.7, width=2)
            ax4.set_ylabel('Efficiency')
            ax4.set_xlabel('Period Length')
            ax4.grid(True, alpha=0.3)
            
            # Remove x-axis sharing for the last subplot
            plt.setp(ax4.get_xticklabels(), visible=True)
            ax4.set_xlim(0, max(periods) + 3)
            ax4.set_xticks(periods)
    else:
        # If no fingerprint data, show pressure imbalance
        imbalance = order_flow_results.get('imbalance', 0)
        ax4.bar(['Imbalance'], [imbalance], 
                color='green' if imbalance > 0 else 'red',
                alpha=0.7)
        ax4.set_ylabel('Buy/Sell Imbalance')
        ax4.set_ylim(-1, 1)
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax4.grid(True, alpha=0.3)
    
    # Format date axis for the price chart
    if 'datetime' in recent_data.columns:
        ax1.set_xlabel('Date')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        fig.autofmt_xdate()
    else:
        # If no datetime, use candle indices with reduced labels
        indices = range(len(recent_data))
        step = max(1, len(indices) // 10)  # Show roughly 10 labels
        visible_indices = indices[::step]
        ax1.set_xticks(visible_indices)
        ax1.set_xticklabels([str(i) for i in visible_indices])
        ax1.set_xlabel('Candle')
    
    # Hide x labels and tick labels for all but bottom subplot
    for ax in [ax1, ax2, ax3]:
        plt.setp(ax.get_xticklabels(), visible=False)
    
    # Add title with institutional activity summary
    if title:
        plt.suptitle(title, fontsize=16)
    else:
        inst_activity = order_flow_results.get('institutional_activity', {})
        if inst_activity.get('present', False):
            activity_type = inst_activity.get('type', 'neutral')
            confidence = inst_activity.get('confidence', 0)
            title = f"Order Flow Analysis - Institutional Activity: {activity_type.upper()} (Confidence: {confidence:.1f})"
        else:
            title = "Order Flow Analysis - No Clear Institutional Activity"
        plt.suptitle(title, fontsize=16)
    
    # Add annotations for key signals
    if 'institutional_activity' in order_flow_results and order_flow_results['institutional_activity'].get('present', False):
        signals = order_flow_results['institutional_activity'].get('signals', [])
        if signals:
            signal_list = ", ".join([s['type'].replace('_', ' ').title() for s in signals])
            fig.text(0.5, 0.01, f"Detected: {signal_list}", ha='center', fontsize=12)
    
    # Tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig

def plot_price_candles(ax, data):
    """Plot OHLC candlesticks on the provided axis"""
    # Plot candlesticks
    for i in range(len(data)):
        # Get OHLC values
        open_price = data['open'].iloc[i]
        high_price = data['high'].iloc[i]
        low_price = data['low'].iloc[i]
        close_price = data['close'].iloc[i]
        
        # Determine if bullish or bearish
        is_bullish = close_price >= open_price
        
        # Choose color
        color = 'green' if is_bullish else 'red'
        
        # Plot high-low line (wick)
        ax.plot([i, i], [low_price, high_price], color='black', linewidth=1)
        
        # Plot body
        body_height = abs(close_price - open_price)
        body_bottom = min(close_price, open_price)
        
        # Create rectangle for body
        rect = Rectangle((i - 0.4, body_bottom), 0.8, body_height, 
                        fill=True, color=color, alpha=0.8)
        ax.add_patch(rect)
    
    # Set y-axis to price
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)

def plot_institutional_signals(ax, data, signals, window):
    """Plot institutional signals on the price chart"""
    # Define colors and markers for different signal types
    signal_styles = {
        'bullish': {'color': 'green', 'marker': '^', 'size': 150},
        'bearish': {'color': 'red', 'marker': 'v', 'size': 150},
        'neutral': {'color': 'blue', 'marker': 'o', 'size': 120}
    }
    
    # Plot each signal
    for signal in signals:
        signal_type = signal['type']
        candle_index = signal['candle_index']
        strength = signal['strength']
        
        # Determine style category
        if 'bullish' in signal_type:
            style = signal_styles['bullish']
        elif 'bearish' in signal_type:
            style = signal_styles['bearish']
        else:
            style = signal_styles['neutral']
        
        # Adjust marker size based on strength
        marker_size = style['size'] * strength
        
        # Handle both single index and slice types
        if isinstance(candle_index, slice):
            # For range signals like accumulation/distribution
            start_idx = candle_index.start if candle_index.start is not None else 0
            if start_idx < 0:
                start_idx = len(data) + start_idx
            end_idx = candle_index.stop if candle_index.stop is not None else len(data)
            if end_idx < 0:
                end_idx = len(data) + end_idx
                
            # Draw shaded region
            mid_price = (data['high'].iloc[start_idx:end_idx].mean() + 
                        data['low'].iloc[start_idx:end_idx].mean()) / 2
            ax.axvspan(start_idx, end_idx-1, alpha=0.2, color=style['color'])
            
            # Add text label
            signal_name = signal_type.replace('_', ' ').title()
            ax.text(start_idx + (end_idx - start_idx) / 2, mid_price, 
                    signal_name, ha='center', va='center', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor=style['color']))
        else:
            # For single candle signals
            if candle_index >= 0 and candle_index < len(data):
                # Place marker at appropriate position
                if 'bullish' in signal_type:
                    # Place bullish markers below the candle
                    y_pos = data['low'].iloc[candle_index] * 0.999
                elif 'bearish' in signal_type:
                    # Place bearish markers above the candle
                    y_pos = data['high'].iloc[candle_index] * 1.001
                else:
                    # Place neutral markers at the close
                    y_pos = data['close'].iloc[candle_index]
                
                # Plot the marker
                ax.scatter(candle_index, y_pos, 
                          s=marker_size, 
                          c=style['color'], 
                          marker=style['marker'], 
                          alpha=0.8,
                          edgecolors='black', 
                          linewidths=1)
                
                # Add annotation for significant signals
                if strength > 0.7:
                    signal_name = signal_type.split('_')[0].title()
                    ax.annotate(signal_name, 
                              (candle_index, y_pos),
                              textcoords="offset points",
                              xytext=(0, 10 if 'bullish' in signal_type else -15),
                              ha='center',
                              fontsize=8,
                              bbox=dict(boxstyle="round,pad=0.3", 
                                        fc='white', 
                                        alpha=0.7,
                                        ec=style['color']))

def plot_volume_profile(data, volume_profile, figsize=(10, 8)):
    """
    Create a volume profile visualization
    
    Args:
        data: DataFrame with OHLCV price data
        volume_profile: Results from calculate_volume_profile function
        figsize: Size of the figure (width, height)
        
    Returns:
        Matplotlib figure
    """
    # Check for required data
    if not isinstance(volume_profile, dict) or 'volume_by_price' not in volume_profile:
        return None
    
    # Extract data
    volume_by_price = volume_profile['volume_by_price']
    poc = volume_profile.get('poc')
    vah = volume_profile.get('vah')
    val = volume_profile.get('val')
    institutional_levels = volume_profile.get('institutional_levels', [])
    
    # Create figure with price on y-axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract volumes and prices
    prices = [vbp['price'] for vbp in volume_by_price]
    volumes = [vbp['volume'] for vbp in volume_by_price]
    
    # Normalize volumes for better visualization
    max_volume = max(volumes) if volumes else 1
    normalized_volumes = [v / max_volume for v in volumes]
    
    # Create colormap for bull/bear volume
    bull_volumes = [vbp.get('bullish_volume', 0) for vbp in volume_by_price]
    bear_volumes = [vbp.get('bearish_volume', 0) for vbp in volume_by_price]
    
    # Plot the volume profile horizontally
    ax.barh(prices, normalized_volumes, height=prices[1]-prices[0] if len(prices) > 1 else 0.1,
           color='skyblue', alpha=0.6)
    
    # Highlight POC, VAH, VAL
    if poc is not None:
        ax.axhline(y=poc, color='blue', linestyle='-', linewidth=2, label='POC')
    if vah is not None:
        ax.axhline(y=vah, color='green', linestyle='--', linewidth=1.5, label='VAH')
    if val is not None:
        ax.axhline(y=val, color='green', linestyle='--', linewidth=1.5, label='VAL')
    
    # Add price range from data
    if len(data) > 0:
        current_price = data['close'].iloc[-1]
        ax.axhline(y=current_price, color='red', linestyle='-', linewidth=1.5, label='Current Price')
    
    # Add institutional levels
    for level in institutional_levels:
        level_price = level.get('price')
        level_type = level.get('type', '')
        level_strength = level.get('strength', 0.5)
        
        if level_price is not None:
            # Choose color and style based on level type
            if 'support' in level_type or level_type == 'val':
                color = 'green'
                linestyle = ':'
            elif 'resistance' in level_type or level_type == 'vah':
                color = 'red'
                linestyle = ':'
            elif level_type == 'poc':
                color = 'blue'
                linestyle = '-'
            elif 'acceleration' in level_type:
                color = 'purple'
                linestyle = '--'
                # For acceleration zones, draw a box
                if 'range' in level:
                    start, end = level['range']
                    height = end - start
                    rect = Rectangle((0, start), 1.1, height, 
                                    fill=True, color=color, alpha=0.2)
                    ax.add_patch(rect)
                    continue
            else:
                color = 'gray'
                linestyle = '-.'
            
            # Draw the line with thickness based on strength
            linewidth = 0.8 + level_strength
            ax.axhline(y=level_price, color=color, linestyle=linestyle, 
                      linewidth=linewidth, alpha=0.7,
                      label=f"{level_type.replace('_', ' ').title()}")
            
    # Format axes
    ax.set_ylabel('Price')
    ax.set_xlabel('Relative Volume')
    ax.set_title('Volume Profile with Institutional Levels')
    
    # Remove x-ticks as they're not meaningful for normalized volume
    ax.set_xticks([])
    
    # Add a legend
    ax.legend(loc='upper right')
    
    # Add current price OHLC data
    if len(data) > 0:
        text_data = (f"O: {data['open'].iloc[-1]:.5f}  H: {data['high'].iloc[-1]:.5f}\n"
                    f"L: {data['low'].iloc[-1]:.5f}  C: {data['close'].iloc[-1]:.5f}")
        fig.text(0.02, 0.02, text_data, fontsize=9)
    
    plt.tight_layout()
    return fig

def plot_order_flow_heatmap(data, order_flow_results, timeframes=['5m', '15m', '1h', '4h', 'D'], figsize=(12, 8)):
    """
    Create a multi-timeframe order flow heatmap visualization
    
    Args:
        data: Dictionary with DataFrames for different timeframes
        order_flow_results: Dictionary with order flow results for different timeframes
        timeframes: List of timeframes to include in the heatmap
        figsize: Size of the figure (width, height)
        
    Returns:
        Matplotlib figure
    """
    # Validate inputs
    if not isinstance(data, dict) or not isinstance(order_flow_results, dict):
        return None
    
    # Find common timeframes
    common_tfs = [tf for tf in timeframes if tf in data and tf in order_flow_results]
    if not common_tfs:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define metrics to extract
    metrics = [
        'imbalance',
        'delta.acceleration',
        'delta.cumulative',
        'buying_pressure',
        'selling_pressure'
    ]
    
    # Create data matrix
    matrix = []
    metric_labels = []
    
    for metric in metrics:
        row = []
        # Parse metric path (e.g., 'delta.acceleration')
        parts = metric.split('.')
        
        # Get friendly name for metric
        if len(parts) == 1:
            metric_labels.append(parts[0].replace('_', ' ').title())
        else:
            metric_labels.append(f"{parts[0].title()} {parts[1].replace('_', ' ').title()}")
        
        for tf in common_tfs:
            # Extract value based on metric path
            value = order_flow_results[tf]
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = 0
                    break
            
            # Scale and clip values
            if 'imbalance' in metric:
                # Range: -1 to 1
                value = max(-1, min(1, value))
            elif 'pressure' in metric:
                # Normalize pressure values
                value = max(0, min(1, value * 2))  # Scale to 0-1
            elif 'acceleration' in metric:
                # Normalize acceleration
                value = max(-1, min(1, value / (data[tf]['volume'].mean() * 0.1 + 1e-10)))
            elif 'cumulative' in metric:
                # Normalize cumulative delta
                value = max(-1, min(1, value / (data[tf]['volume'].sum() * 0.1 + 1e-10)))
            
            row.append(value)
        
        matrix.append(row)
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Intensity (Red = Bearish, Green = Bullish)')
    
    # Set labels
    ax.set_xticks(np.arange(len(common_tfs)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(common_tfs)
    ax.set_yticklabels(metric_labels)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add values to cells
    for i in range(len(metrics)):
        for j in range(len(common_tfs)):
            value = matrix[i][j]
            text_color = "black" if -0.3 <= value <= 0.3 else "white"
            text = f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", color=text_color)
    
    # Add institutional analysis if available
    institutional_summary = {}
    for tf in common_tfs:
        if tf in order_flow_results and 'institutional_activity' in order_flow_results[tf]:
            inst = order_flow_results[tf]['institutional_activity']
            if inst.get('present', False):
                institutional_summary[tf] = {
                    'type': inst.get('type', 'neutral'),
                    'confidence': inst.get('confidence', 0),
                    'signals': [s['type'] for s in inst.get('signals', [])]
                }
    
    # Add summary text
    if institutional_summary:
        summary_text = "Institutional Activity:\n"
        for tf, info in institutional_summary.items():
            signals_text = ", ".join(s.replace('_', ' ').title() for s in info['signals'][:2])
            if len(info['signals']) > 2:
                signals_text += f" + {len(info['signals']) - 2} more"
            summary_text += f"{tf}: {info['type'].upper()} ({info['confidence']:.1f}) - {signals_text}\n"
        
        fig.text(0.5, 0.01, summary_text, ha='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Title
    ax.set_title("Order Flow Analysis Across Timeframes")
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95] if institutional_summary else None)
    return fig