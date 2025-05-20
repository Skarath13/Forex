#!/usr/bin/env python3
"""
Simulate Order Flow Analysis Profitability

This script simulates trades to demonstrate the profitability
difference between trading with and without order flow analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import os

# Define parameters for simulation
NUM_TRADES = 200
SEED = 42  # For reproducibility
random.seed(SEED)
np.random.seed(SEED)

def simulate_trades():
    """
    Simulate trades with and without order flow analysis
    """
    print("Simulating trades to compare profitability with and without order flow analysis...")
    
    # Create empty trade list
    trades = []
    
    # Generate trade dates over 1 year period
    start_date = datetime.now() - timedelta(days=365)
    dates = [start_date + timedelta(days=i) for i in range(365)]
    
    # Define different market phases
    phases = [
        {"name": "Uptrend", "days": 60, "reg_win_rate": 0.55, "of_win_rate": 0.68},
        {"name": "Distribution", "days": 30, "reg_win_rate": 0.45, "of_win_rate": 0.67},
        {"name": "Downtrend", "days": 90, "reg_win_rate": 0.52, "of_win_rate": 0.65},
        {"name": "Accumulation", "days": 30, "reg_win_rate": 0.48, "of_win_rate": 0.72},
        {"name": "Uptrend", "days": 80, "reg_win_rate": 0.56, "of_win_rate": 0.70},
        {"name": "Volatile", "days": 75, "reg_win_rate": 0.42, "of_win_rate": 0.63}
    ]
    
    # Generate trades for each phase
    day_index = 0
    for phase in phases:
        for _ in range(phase["days"] // 5):  # Approximately one trade every 5 days
            if len(trades) >= NUM_TRADES:
                break
                
            # Create two similar trades - one with regular analysis, one with order flow
            trade_date = dates[day_index]
            
            # Regular trade
            is_win_reg = random.random() < phase["reg_win_rate"]
            profit_reg = random.uniform(0.5, 2.0) if is_win_reg else -random.uniform(0.5, 1.5)
            
            # Order flow trade - keep direction the same but improve win rate and R:R
            is_win_of = random.random() < phase["of_win_rate"]
            profit_of = random.uniform(0.6, 2.3) if is_win_of else -random.uniform(0.4, 1.3)
            
            # Add trades to list
            trades.append({
                'date': trade_date,
                'type': 'Regular',
                'phase': phase["name"],
                'profit': profit_reg,
                'win': is_win_reg
            })
            
            trades.append({
                'date': trade_date,
                'type': 'Order Flow',
                'phase': phase["name"],
                'profit': profit_of,
                'win': is_win_of
            })
            
            day_index += 5
    
    # Create dataframe
    df = pd.DataFrame(trades)
    
    # Calculate metrics
    print("\nTrade Performance Metrics:")
    print("-" * 50)
    
    # Calculate by trade type
    for trade_type in ['Regular', 'Order Flow']:
        type_df = df[df['type'] == trade_type]
        
        wins = type_df[type_df['win'] == True]
        losses = type_df[type_df['win'] == False]
        
        win_rate = len(wins) / len(type_df) if len(type_df) > 0 else 0
        avg_win = wins['profit'].mean() if len(wins) > 0 else 0
        avg_loss = losses['profit'].mean() if len(losses) > 0 else 0
        profit_factor = abs(wins['profit'].sum() / losses['profit'].sum()) if losses['profit'].sum() != 0 else float('inf')
        
        print(f"\n{trade_type} Analysis:")
        print(f"Total trades: {len(type_df)}")
        print(f"Win rate: {win_rate:.2%}")
        print(f"Average win: {avg_win:.2f}")
        print(f"Average loss: {avg_loss:.2f}")
        print(f"Profit factor: {profit_factor:.2f}")
        print(f"Total profit: {type_df['profit'].sum():.2f}")
    
    # Calculate improvement percentage
    regular_df = df[df['type'] == 'Regular']
    orderflow_df = df[df['type'] == 'Order Flow']
    
    regular_profit = regular_df['profit'].sum()
    orderflow_profit = orderflow_df['profit'].sum()
    
    improvement = ((orderflow_profit / regular_profit) - 1) * 100 if regular_profit > 0 else 0
    print(f"\nProfit improvement with order flow analysis: {improvement:.2f}%")
    
    # Calculate Sharpe ratio
    regular_sharpe = regular_df['profit'].mean() / regular_df['profit'].std() * np.sqrt(252 / len(regular_df))
    orderflow_sharpe = orderflow_df['profit'].mean() / orderflow_df['profit'].std() * np.sqrt(252 / len(orderflow_df))
    
    print(f"Regular Sharpe ratio: {regular_sharpe:.2f}")
    print(f"Order Flow Sharpe ratio: {orderflow_sharpe:.2f}")
    
    # Calculate profitability by market phase
    print("\nProfitability by Market Phase:")
    print("-" * 50)
    print(f"{'Phase':<15} {'Regular':<10} {'Order Flow':<15} {'Improvement':<15}")
    
    for phase in phases:
        phase_name = phase["name"]
        regular_phase = regular_df[regular_df['phase'] == phase_name]
        orderflow_phase = orderflow_df[orderflow_df['phase'] == phase_name]
        
        reg_profit = regular_phase['profit'].sum()
        of_profit = orderflow_phase['profit'].sum()
        
        phase_improvement = ((of_profit / reg_profit) - 1) * 100 if reg_profit > 0 else 0
        
        print(f"{phase_name:<15} {reg_profit:<10.2f} {of_profit:<15.2f} {phase_improvement:<15.2f}%")
    
    # Create equity curves
    reg_equity = [0]
    of_equity = [0]
    
    # Sort by date
    regular_df_sorted = regular_df.sort_values('date')
    orderflow_df_sorted = orderflow_df.sort_values('date')
    
    for profit in regular_df_sorted['profit']:
        reg_equity.append(reg_equity[-1] + profit)
        
    for profit in orderflow_df_sorted['profit']:
        of_equity.append(of_equity[-1] + profit)
    
    # Plot equity curves
    plt.figure(figsize=(12, 6))
    plt.plot(reg_equity, label='Regular Analysis', color='blue', alpha=0.7)
    plt.plot(of_equity, label='Order Flow Analysis', color='green')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.2)
    plt.title('Equity Curves: Regular vs Order Flow Analysis')
    plt.xlabel('Number of Trades')
    plt.ylabel('Cumulative Profit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save and show the plot
    plt.savefig('order_flow_profit_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nEquity curve comparison saved as 'order_flow_profit_comparison.png'")
    
    # Create win rate comparison across phases
    phases_list = [phase["name"] for phase in phases]
    reg_win_rates = []
    of_win_rates = []
    
    for phase_name in phases_list:
        regular_phase = regular_df[regular_df['phase'] == phase_name]
        orderflow_phase = orderflow_df[orderflow_df['phase'] == phase_name]
        
        reg_win_rate = len(regular_phase[regular_phase['win'] == True]) / len(regular_phase) if len(regular_phase) > 0 else 0
        of_win_rate = len(orderflow_phase[orderflow_phase['win'] == True]) / len(orderflow_phase) if len(orderflow_phase) > 0 else 0
        
        reg_win_rates.append(reg_win_rate)
        of_win_rates.append(of_win_rate)
    
    # Create bar chart for win rates
    plt.figure(figsize=(12, 6))
    x = np.arange(len(phases_list))
    width = 0.35
    
    plt.bar(x - width/2, reg_win_rates, width, label='Regular Analysis', color='blue', alpha=0.7)
    plt.bar(x + width/2, of_win_rates, width, label='Order Flow Analysis', color='green', alpha=0.7)
    
    plt.xlabel('Market Phase')
    plt.ylabel('Win Rate')
    plt.title('Win Rate Comparison Across Market Phases')
    plt.xticks(x, phases_list)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save and show the plot
    plt.savefig('order_flow_win_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Win rate comparison saved as 'order_flow_win_rate_comparison.png'")
    
    print("\nSimulation complete!")
    return df

if __name__ == "__main__":
    df = simulate_trades()