"""
Trend Strength Indicators - Advanced indicators to measure the strength and direction of market trends.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict
import warnings


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate the Average Directional Index (ADX) to determine trend strength.
    
    Args:
        df: DataFrame with OHLCV data
        period: Period for ADX calculation
        
    Returns:
        DataFrame with ADX indicators added
    """
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Filter warnings temporarily (to suppress the ta library's RuntimeWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        
        try:
            # Calculate ADX using ta library
            adx_indicator = ta.trend.ADXIndicator(
                high=df['high'], 
                low=df['low'], 
                close=df['close'], 
                window=period,
                fillna=True  # Fill NaN values
            )
            
            df['plus_di'] = adx_indicator.adx_pos()
            df['minus_di'] = adx_indicator.adx_neg()
            df['adx'] = adx_indicator.adx()
            
            # Handle potential NaN values
            df['plus_di'] = df['plus_di'].fillna(0)
            df['minus_di'] = df['minus_di'].fillna(0)
            df['adx'] = df['adx'].fillna(0)
        
        except Exception as e:
            # In case of errors, create empty columns
            df['plus_di'] = 0
            df['minus_di'] = 0
            df['adx'] = 0
            print(f"Error calculating ADX: {e}")
    
    # Calculate DI difference to determine the direction of the trend
    df['di_diff'] = df['plus_di'] - df['minus_di']
    
    return df


def calculate_slope(df: pd.DataFrame, column: str = 'close', period: int = 20) -> pd.DataFrame:
    """
    Calculate the slope of a given column to determine trend direction and strength.
    
    Args:
        df: DataFrame with OHLCV data
        column: Column to calculate slope for
        period: Period for slope calculation
        
    Returns:
        DataFrame with slope indicator added
    """
    # Make a copy of the dataframe
    df = df.copy()
    
    # Ensure we have enough data
    if len(df) < period:
        df['slope'] = np.nan
        return df
    
    # Calculate linear regression slope
    df['slope'] = df[column].rolling(window=period).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / x.mean(),
        raw=True
    ) * 100  # Convert to percentage for readability
    
    return df


def identify_strong_trend(df: pd.DataFrame, adx_threshold: float = 25.0,
                         slope_threshold: float = 0.5) -> Dict:
    """
    Identify whether there's a strong trend and its direction.
    
    Args:
        df: DataFrame with ADX and slope indicators
        adx_threshold: Minimum ADX value to consider a strong trend
        slope_threshold: Minimum slope value to consider a significant trend
        
    Returns:
        Dict with trend information: {'trend_strength': float, 'trend_direction': str}
    """
    # Get the last values
    last_row = df.iloc[-1]
    
    # Default values
    result = {
        'trend_strength': 0.0,
        'trend_direction': 'neutral',
        'is_strong_trend': False
    }
    
    # Check if we have the required indicators
    if 'adx' not in last_row or 'slope' not in last_row:
        return result
    
    # Get indicator values
    adx = last_row['adx']
    di_diff = last_row.get('di_diff', 0)
    slope = last_row['slope']
    
    # Determine trend strength and direction
    result['trend_strength'] = adx
    
    # Determine direction
    if di_diff > 0 and slope > 0:
        result['trend_direction'] = 'bullish'
    elif di_diff < 0 and slope < 0:
        result['trend_direction'] = 'bearish'
    else:
        # Mixed signals or weak trend
        result['trend_direction'] = 'neutral'
    
    # Is this a strong trend?
    result['is_strong_trend'] = adx > adx_threshold and abs(slope) > slope_threshold
    
    return result
