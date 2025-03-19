"""
Advanced Filters - Smart entry and exit filters to improve trade quality and win rate.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, Tuple, List


def calculate_volatility_metrics(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Calculate volatility metrics like ATR and normalized ATR.
    
    Args:
        df: DataFrame with OHLCV data
        atr_period: Period for ATR calculation
        
    Returns:
        DataFrame with volatility metrics added
    """
    df = df.copy()
    
    # Calculate ATR using the ta library
    atr_indicator = ta.volatility.AverageTrueRange(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=atr_period
    )
    
    df['atr'] = atr_indicator.average_true_range()
    
    # Calculate normalized ATR (relative to price)
    if 'atr' in df:
        df['natr'] = df['atr'] / df['close'] * 100
    
    return df


def calculate_risk_reward(entry_price: float, stop_loss: float, take_profit: float) -> float:
    """
    Calculate risk-reward ratio for a trade.
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        
    Returns:
        Risk-reward ratio (reward divided by risk)
    """
    if entry_price == stop_loss or entry_price == take_profit:
        return 0.0
        
    risk = abs(entry_price - stop_loss)
    reward = abs(entry_price - take_profit)
    
    if risk == 0:
        return 0.0
    
    return reward / risk


def check_volume_confirmation(df: pd.DataFrame, lookback: int = 5) -> bool:
    """
    Check if recent volume confirms the price move.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Number of bars to look back
        
    Returns:
        Boolean indicating whether volume confirms the move
    """
    if len(df) < lookback + 10:
        return False
        
    recent_df = df.iloc[-lookback:].copy()
    
    # Determine if we're in an uptrend or downtrend
    price_change = recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]
    
    # Check if volume is increasing
    volume_avg = df['volume'].iloc[-(lookback+10):-lookback].mean()
    recent_volume_avg = recent_df['volume'].mean()
    
    # Volume should be increasing in the direction of the trend
    return recent_volume_avg > volume_avg * 1.2


def check_multi_timeframe_confirmation(df_short: pd.DataFrame, 
                                     df_medium: pd.DataFrame,
                                     df_long: pd.DataFrame,
                                     direction: str) -> bool:
    """
    Check for confirmation across multiple timeframes.
    
    Args:
        df_short: Short timeframe DataFrame (e.g., 5m)
        df_medium: Medium timeframe DataFrame (e.g., 15m or 1h)
        df_long: Long timeframe DataFrame (e.g., 4h or 1d)
        direction: 'buy' or 'sell'
        
    Returns:
        Boolean indicating whether multiple timeframes confirm the direction
    """
    # Ensure we have enough data
    if len(df_short) < 20 or len(df_medium) < 20 or len(df_long) < 20:
        return False
    
    # Calculate EMAs for each timeframe
    for df in [df_short, df_medium, df_long]:
        ema_indicator = ta.trend.EMAIndicator(close=df['close'], window=20)
        df['ema20'] = ema_indicator.ema_indicator()
        
        ema_indicator50 = ta.trend.EMAIndicator(close=df['close'], window=50)
        df['ema50'] = ema_indicator50.ema_indicator()
    
    # Get alignment status
    short_aligned = df_short['close'].iloc[-1] > df_short['ema20'].iloc[-1] if direction == 'buy' else df_short['close'].iloc[-1] < df_short['ema20'].iloc[-1]
    
    medium_aligned = df_medium['ema20'].iloc[-1] > df_medium['ema50'].iloc[-1] if direction == 'buy' else df_medium['ema20'].iloc[-1] < df_medium['ema50'].iloc[-1]
    
    long_aligned = df_long['ema20'].iloc[-1] > df_long['ema50'].iloc[-1] if direction == 'buy' else df_long['ema20'].iloc[-1] < df_long['ema50'].iloc[-1]
    
    # Check if at least 2 of 3 timeframes align
    alignment_count = sum([short_aligned, medium_aligned, long_aligned])
    
    return alignment_count >= 2


def filter_trade_setup(df: pd.DataFrame, setup: Dict, min_rr: float = 1.5) -> Dict:
    """
    Apply advanced filters to determine if a trade setup meets all criteria.
    
    Args:
        df: DataFrame with OHLCV and indicator data
        setup: Trade setup dictionary with entry, side, entry_price, stop_loss, take_profit
        min_rr: Minimum risk-reward ratio required
        
    Returns:
        Updated trade setup with additional filter results
    """
    # Make a copy of the setup
    filtered_setup = setup.copy()
    
    # Calculate risk-reward ratio
    rr_ratio = calculate_risk_reward(
        setup['entry_price'], setup['stop_loss'], setup['take_profit']
    )
    
    filtered_setup['risk_reward_ratio'] = rr_ratio
    
    # Check if risk-reward is acceptable
    filtered_setup['rr_filter_passed'] = rr_ratio >= min_rr
    
    # Check volume confirmation
    filtered_setup['volume_confirmed'] = check_volume_confirmation(df)
    
    # Final decision - all filters must pass
    filtered_setup['entry'] = (
        setup['entry'] and 
        filtered_setup['rr_filter_passed'] and 
        filtered_setup['volume_confirmed']
    )
    
    # Add reasoning
    if not filtered_setup['entry'] and setup['entry']:
        if not filtered_setup['rr_filter_passed']:
            filtered_setup['filter_reason'] = f"Risk-reward ratio {rr_ratio:.2f} below minimum {min_rr}"
        elif not filtered_setup['volume_confirmed']:
            filtered_setup['filter_reason'] = "Insufficient volume confirmation"
    
    return filtered_setup
