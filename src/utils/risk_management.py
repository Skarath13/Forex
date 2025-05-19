import numpy as np
import pandas as pd

def calculate_position_size(direction, entry_price, stop_loss, risk_per_trade, daily_risk_used, max_risk_per_day, instrument, account_balance=10000):
    """
    Calculate optimal position size based on risk parameters
    
    Args:
        direction: Trade direction (LONG/SHORT)
        entry_price: Entry price
        stop_loss: Stop loss price
        risk_per_trade: Risk per trade as percentage of account
        daily_risk_used: Risk already used today
        max_risk_per_day: Maximum risk per day
        instrument: Trading instrument
        account_balance: Account balance (default 10000)
        
    Returns:
        Dictionary with position details
    """
    # Calculate risk amount in account currency
    risk_amount = account_balance * risk_per_trade
    
    # Check if we're exceeding daily risk limit
    remaining_risk = max_risk_per_day - daily_risk_used
    if risk_amount > remaining_risk:
        risk_amount = remaining_risk
    
    # Calculate pip value and position size
    pip_value = calculate_pip_value(instrument, entry_price)
    
    # Calculate stop distance in pips
    if direction == 'LONG':
        stop_distance_pips = (entry_price - stop_loss) / pip_value
    else:  # SHORT
        stop_distance_pips = (stop_loss - entry_price) / pip_value
    
    # Calculate position size
    if stop_distance_pips > 0:
        position_size = risk_amount / (stop_distance_pips * pip_value)
    else:
        position_size = 0
        
    # Calculate potential loss with this position
    potential_loss = stop_distance_pips * pip_value * position_size
    
    # Store ATR value for possible trailing stop adjustments
    atr_value = abs(entry_price - stop_loss)
    
    return {
        'position_size': position_size,
        'risk_amount': potential_loss,
        'pip_value': pip_value,
        'stop_distance_pips': stop_distance_pips,
        'atr_value': atr_value
    }

def calculate_pip_value(instrument, price, account_currency='USD'):
    """
    Calculate pip value for a given instrument
    
    Args:
        instrument: Trading instrument (e.g., 'EURUSD')
        price: Current price
        account_currency: Account base currency
        
    Returns:
        Pip value in account currency
    """
    # Extract base and quote currencies
    if len(instrument) == 6:
        base_currency = instrument[:3]
        quote_currency = instrument[3:]
    else:
        # Default to standard pip value for non-forex instruments
        return 0.0001
    
    # Standard pip size for most forex pairs
    pip_size = 0.0001
    
    # Adjust for JPY pairs
    if 'JPY' in instrument:
        pip_size = 0.01
    
    # If account currency is the quote currency, calculation is simple
    if account_currency == quote_currency:
        return pip_size
    
    # If account currency is the base currency
    if account_currency == base_currency:
        return pip_size * price
    
    # More complex cases would require additional cross-rate calculations
    # This is a simplified implementation
    return pip_size

def calculate_risk_reward(direction, entry_price, stop_loss, take_profit):
    """
    Calculate risk-reward ratio for a trade
    
    Args:
        direction: Trade direction ('long' or 'short')
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        
    Returns:
        Risk-reward ratio
    """
    if direction == 'LONG':
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
    else:  # SHORT
        risk = stop_loss - entry_price
        reward = entry_price - take_profit
    
    # Avoid division by zero
    if risk == 0:
        return 0
        
    return reward / risk

def set_stop_loss(direction, entry_price, atr, multiplier=1.5):
    """
    Set stop loss based on ATR
    
    Args:
        direction: Trade direction ('long' or 'short')
        entry_price: Entry price
        atr: Average True Range value
        multiplier: ATR multiplier
        
    Returns:
        Stop loss price
    """
    if direction == 'long':
        return entry_price - (atr * multiplier)
    else:  # short
        return entry_price + (atr * multiplier)

def set_take_profit(direction, entry_price, stop_loss, risk_reward=2.0):
    """
    Set take profit based on risk-reward ratio
    
    Args:
        direction: Trade direction ('long' or 'short')
        entry_price: Entry price
        stop_loss: Stop loss price
        risk_reward: Target risk-reward ratio
        
    Returns:
        Take profit price
    """
    if direction == 'long':
        risk = entry_price - stop_loss
        return entry_price + (risk * risk_reward)
    else:  # short
        risk = stop_loss - entry_price
        return entry_price - (risk * risk_reward)

def adjust_position(position, adjustment_type, price, size=None, new_stop=None, new_target=None):
    """
    Adjust an existing position (add, reduce, move stops)
    
    Args:
        position: Current position details
        adjustment_type: Type of adjustment ('add', 'reduce', 'move_stop')
        price: Current price for adjustment
        size: Size to add or reduce (optional)
        new_stop: New stop loss price (optional)
        new_target: New take profit price (optional)
        
    Returns:
        Dictionary with adjustment details and updated position
    """
    updated_position = position.copy()
    
    if adjustment_type == 'add':
        # Add to position
        orig_size = position['size']
        new_total_size = orig_size + size
        
        # Calculate new average entry price
        orig_value = orig_size * position['entry_price']
        new_value = size * price
        updated_position['entry_price'] = (orig_value + new_value) / new_total_size
        updated_position['size'] = new_total_size
        
        # Update stop and target if provided
        if new_stop is not None:
            updated_position['stop_loss'] = new_stop
        if new_target is not None:
            updated_position['take_profit'] = new_target
            
        # Increment pyramid count
        updated_position['pyramid_count'] = position.get('pyramid_count', 0) + 1
        
        return {
            'type': 'add_position',
            'instrument': position['instrument'],
            'direction': position['direction'],
            'added_size': size,
            'added_price': price,
            'updated_position': updated_position,
            'time': position.get('entry_time', 0)
        }
        
    elif adjustment_type == 'reduce':
        # Reduce position
        orig_size = position['size']
        
        if size >= orig_size:
            # Full position closure
            remaining_size = 0
        else:
            remaining_size = orig_size - size
            
        updated_position['size'] = remaining_size
        
        return {
            'type': 'reduce_position',
            'instrument': position['instrument'],
            'direction': position['direction'],
            'reduced_size': size,
            'reduced_price': price,
            'remaining_size': remaining_size,
            'updated_position': updated_position,
            'time': position.get('entry_time', 0)
        }
        
    elif adjustment_type == 'move_stop':
        # Move stop loss
        old_stop = position['stop_loss']
        updated_position['stop_loss'] = new_stop
        
        return {
            'type': 'move_stop',
            'instrument': position['instrument'],
            'old_stop': old_stop,
            'new_stop': new_stop,
            'updated_position': updated_position,
            'time': position.get('entry_time', 0)
        }
    
    return {'type': 'no_adjustment', 'updated_position': position}