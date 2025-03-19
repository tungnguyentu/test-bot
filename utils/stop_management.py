"""
Stop Management Utilities - Functions for dynamic stop loss and take profit management.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, Tuple


def calculate_volatility_based_stop(df: pd.DataFrame, side: str, atr_multiplier: float = 2.0) -> float:
    """
    Calculate volatility-based stop loss level.
    
    Args:
        df: DataFrame with OHLCV data
        side: Trade direction ('BUY' or 'SELL')
        atr_multiplier: Multiplier for ATR to determine stop distance
        
    Returns:
        float: Stop loss price
    """
    if len(df) < 15:
        return 0.0
    
    # Calculate ATR using ta library
    atr_indicator = ta.volatility.AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14
    )
    
    df['atr'] = atr_indicator.average_true_range()
    
    # Get latest values
    latest = df.iloc[-1]
    entry_price = float(latest['close'])
    atr_value = float(latest['atr']) if not pd.isna(latest['atr']) else 0.0
    
    # Calculate stop distance
    stop_distance = atr_value * atr_multiplier
    
    # Set stop loss level based on side
    if side == 'BUY':
        stop_loss = entry_price - stop_distance
    else:  # SELL
        stop_loss = entry_price + stop_distance
    
    return float(stop_loss)


def calculate_volatility_based_target(df: pd.DataFrame, side: str, atr_multiplier: float = 2.0, 
                                    risk_reward_ratio: float = 1.5) -> float:
    """
    Calculate volatility-based take profit level.
    
    Args:
        df: DataFrame with OHLCV data
        side: Trade direction ('BUY' or 'SELL')
        atr_multiplier: Multiplier for ATR to determine stop distance
        risk_reward_ratio: Desired risk-reward ratio
        
    Returns:
        float: Take profit price
    """
    if len(df) < 15:
        return 0.0
    
    # Calculate ATR
    atr_indicator = ta.volatility.AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14
    )
    
    df['atr'] = atr_indicator.average_true_range()
    
    # Get latest values
    latest = df.iloc[-1]
    entry_price = float(latest['close'])
    atr_value = float(latest['atr']) if not pd.isna(latest['atr']) else 0.0
    
    # Calculate distances
    stop_distance = atr_value * atr_multiplier
    target_distance = stop_distance * risk_reward_ratio
    
    # Set take profit level based on side
    if side == 'BUY':
        take_profit = entry_price + target_distance
    else:  # SELL
        take_profit = entry_price - target_distance
    
    return float(take_profit)


def calculate_swing_based_stop(df: pd.DataFrame, side: str, lookback: int = 5) -> float:
    """
    Calculate stop loss based on recent swing levels.
    
    Args:
        df: DataFrame with OHLCV data
        side: Trade direction ('BUY' or 'SELL')
        lookback: Number of bars to look back for swing levels
        
    Returns:
        float: Stop loss price
    """
    if len(df) < lookback + 2:
        return 0.0
    
    # Get recent data
    recent_df = df.iloc[-lookback-1:-1].copy()
    
    # Find swing high and low
    swing_high = recent_df['high'].max()
    swing_low = recent_df['low'].min()
    
    # Get entry price
    entry_price = df['close'].iloc[-1]
    
    # Set stop based on side
    if side == 'BUY':
        stop_loss = float(swing_low) * 0.998  # Slightly below swing low
    else:  # SELL
        stop_loss = float(swing_high) * 1.002  # Slightly above swing high
    
    return float(stop_loss)


def calculate_swing_based_target(df: pd.DataFrame, side: str, lookback: int = 5) -> float:
    """
    Calculate take profit based on recent swing levels.
    
    Args:
        df: DataFrame with OHLCV data
        side: Trade direction ('BUY' or 'SELL')
        lookback: Number of bars to look back for swing levels
        
    Returns:
        float: Take profit price
    """
    if len(df) < lookback + 2:
        return 0.0
    
    # Get recent data
    recent_df = df.iloc[-lookback-1:-1].copy()
    
    # Find swing high and low
    swing_high = recent_df['high'].max()
    swing_low = recent_df['low'].min()
    
    # Get entry price
    entry_price = df['close'].iloc[-1]
    
    # Set target based on side
    if side == 'BUY':
        # Target is either a previous swing high or a projected target
        if swing_high > entry_price:
            take_profit = float(swing_high)
        else:
            # Project a target with same distance as the stop
            stop_distance = entry_price - float(swing_low) * 0.998
            take_profit = entry_price + stop_distance * 1.5
    else:  # SELL
        # Target is either a previous swing low or a projected target
        if swing_low < entry_price:
            take_profit = float(swing_low)
        else:
            # Project a target with same distance as the stop
            stop_distance = float(swing_high) * 1.002 - entry_price
            take_profit = entry_price - stop_distance * 1.5
    
    return float(take_profit)


def update_trailing_stop(current_trade: Dict, latest_price: float, 
                        trailing_callback: float = 0.008) -> Dict:
    """
    Update trailing stop for an open trade.
    
    Args:
        current_trade: Dictionary with current trade information
        latest_price: Latest price for the traded asset
        trailing_callback: Percentage to call back from the peak price
        
    Returns:
        Updated trade dictionary with new stop loss level
    """
    # Make a copy of the trade
    updated_trade = current_trade.copy()
    
    # Get required values
    side = current_trade.get('side', None)
    entry_price = current_trade.get('entry_price', 0.0)
    current_stop = current_trade.get('stop_loss', 0.0)
    
    # Skip if no side or entry price
    if not side or not entry_price:
        return updated_trade
    
    # Update trailing stop based on side
    if side == 'BUY':
        # For long trades, we trail as price goes up
        # Calculate minimum price that would trigger a stop update
        # (current stop + callback distance)
        callback_distance = entry_price * trailing_callback
        min_price_for_update = current_stop + callback_distance
        
        # Only update if price has moved up enough
        if latest_price > min_price_for_update:
            # New stop is (latest price - callback)
            new_stop = latest_price * (1 - trailing_callback)
            
            # Only update if new stop is higher
            if new_stop > current_stop:
                updated_trade['stop_loss'] = new_stop
                updated_trade['stop_updated'] = True
    
    else:  # SELL
        # For short trades, we trail as price goes down
        # Calculate maximum price that would trigger a stop update
        # (current stop - callback distance)
        callback_distance = entry_price * trailing_callback
        max_price_for_update = current_stop - callback_distance
        
        # Only update if price has moved down enough
        if latest_price < max_price_for_update:
            # New stop is (latest price + callback)
            new_stop = latest_price * (1 + trailing_callback)
            
            # Only update if new stop is lower
            if new_stop < current_stop:
                updated_trade['stop_loss'] = new_stop
                updated_trade['stop_updated'] = True
    
    return updated_trade
