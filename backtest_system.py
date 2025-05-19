import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    pair: str
    entry_time: datetime
    exit_time: datetime = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    side: str = ''  # 'LONG' or 'SHORT'
    size: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    entry_reason: str = ''
    exit_reason: str = ''
    
class BacktestEngine:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades = []
        self.equity_curve = []
        self.open_positions = {}
        
    def calculate_metrics(self):
        """Calculate backtest performance metrics."""
        if not self.trades:
            return {}
        
        df_trades = pd.DataFrame([{
            'pair': t.pair,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'duration': (t.exit_time - t.entry_time).total_seconds() / 3600,
            'side': t.side,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct
        } for t in self.trades if t.exit_time])
        
        if df_trades.empty:
            return {}
        
        winning_trades = df_trades[df_trades['pnl'] > 0]
        losing_trades = df_trades[df_trades['pnl'] < 0]
        
        metrics = {
            'total_trades': len(df_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(df_trades) if len(df_trades) > 0 else 0,
            'avg_win': winning_trades['pnl'].mean() if not winning_trades.empty else 0,
            'avg_loss': losing_trades['pnl'].mean() if not losing_trades.empty else 0,
            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if not losing_trades.empty and losing_trades['pnl'].sum() != 0 else 0,
            'total_pnl': df_trades['pnl'].sum(),
            'total_return': (self.balance - self.initial_balance) / self.initial_balance,
            'avg_trade_duration': df_trades['duration'].mean(),
            'max_consecutive_wins': self._max_consecutive(df_trades['pnl'] > 0),
            'max_consecutive_losses': self._max_consecutive(df_trades['pnl'] < 0),
            'sharpe_ratio': self._calculate_sharpe_ratio()
        }
        
        # Calculate drawdown
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdown.min()
        
        return metrics
    
    def _max_consecutive(self, series):
        """Calculate maximum consecutive True values."""
        groups = (series != series.shift()).cumsum()
        return series.groupby(groups).sum().max()
    
    def _calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """Calculate Sharpe ratio."""
        if len(self.equity_curve) < 2:
            return 0
        
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        if returns.empty or returns.std() == 0:
            return 0
        
        # Annualized Sharpe ratio (assuming hourly data)
        hours_per_year = 24 * 252  # Trading days
        excess_returns = returns - (risk_free_rate / hours_per_year)
        return np.sqrt(hours_per_year) * excess_returns.mean() / returns.std()
    
    def plot_results(self, save_path=None):
        """Plot backtest results."""
        if not self.trades:
            logger.warning("No trades to plot")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Backtest Results', fontsize=16)
        
        # Equity curve
        ax1 = axes[0, 0]
        ax1.plot(self.equity_curve, label='Equity Curve')
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Account Value ($)')
        ax1.legend()
        ax1.grid(True)
        
        # Drawdown
        ax2 = axes[0, 1]
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        ax2.fill_between(range(len(drawdown)), drawdown, alpha=0.3, color='red')
        ax2.set_title('Drawdown %')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True)
        
        # Trade distribution
        ax3 = axes[1, 0]
        trade_pnls = [t.pnl for t in self.trades if t.exit_time]
        ax3.hist(trade_pnls, bins=30, alpha=0.7, color='blue')
        ax3.axvline(x=0, color='red', linestyle='--')
        ax3.set_title('Trade P&L Distribution')
        ax3.set_xlabel('P&L ($)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True)
        
        # Win rate by pair
        ax4 = axes[1, 1]
        pair_stats = {}
        for trade in self.trades:
            if trade.exit_time:
                if trade.pair not in pair_stats:
                    pair_stats[trade.pair] = {'wins': 0, 'total': 0}
                pair_stats[trade.pair]['total'] += 1
                if trade.pnl > 0:
                    pair_stats[trade.pair]['wins'] += 1
        
        pairs = list(pair_stats.keys())
        win_rates = [pair_stats[p]['wins'] / pair_stats[p]['total'] for p in pairs]
        ax4.bar(pairs, win_rates)
        ax4.set_title('Win Rate by Pair')
        ax4.set_xlabel('Currency Pair')
        ax4.set_ylabel('Win Rate')
        ax4.set_ylim(0, 1)
        ax4.grid(True)
        
        # Monthly returns
        ax5 = axes[2, 0]
        if self.trades:
            monthly_returns = self._calculate_monthly_returns()
            months = list(monthly_returns.keys())
            returns = list(monthly_returns.values())
            colors = ['green' if r > 0 else 'red' for r in returns]
            ax5.bar(range(len(months)), returns, color=colors)
            ax5.set_title('Monthly Returns')
            ax5.set_xlabel('Month')
            ax5.set_ylabel('Return %')
            ax5.set_xticklabels([m.strftime('%Y-%m') for m in months], rotation=45)
            ax5.grid(True)
        
        # Performance metrics
        ax6 = axes[2, 1]
        ax6.axis('off')
        metrics = self.calculate_metrics()
        metrics_text = f"""
        Total Trades: {metrics.get('total_trades', 0)}
        Win Rate: {metrics.get('win_rate', 0):.2%}
        Profit Factor: {metrics.get('profit_factor', 0):.2f}
        Total Return: {metrics.get('total_return', 0):.2%}
        Max Drawdown: {metrics.get('max_drawdown', 0):.2%}
        Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
        Avg Trade Duration: {metrics.get('avg_trade_duration', 0):.1f}h
        """
        ax6.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def _calculate_monthly_returns(self):
        """Calculate returns by month."""
        monthly_returns = {}
        
        for trade in self.trades:
            if trade.exit_time:
                month_key = trade.exit_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                if month_key not in monthly_returns:
                    monthly_returns[month_key] = 0
                monthly_returns[month_key] += trade.pnl
        
        # Convert to percentage returns
        for month in monthly_returns:
            monthly_returns[month] = (monthly_returns[month] / self.initial_balance) * 100
        
        return dict(sorted(monthly_returns.items()))