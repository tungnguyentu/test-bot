"""
Candlestick pattern detection for technical analysis.
"""

import pandas as pd
import numpy as np


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect various candlestick patterns in price data.
    
    Args:
        df: DataFrame containing OHLC price data
        
    Returns:
        DataFrame with pattern detection columns added
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Calculate some basic properties of each candle
    df_copy['body_size'] = abs(df_copy['close'] - df_copy['open'])
    df_copy['upper_shadow'] = df_copy['high'] - df_copy[['open', 'close']].max(axis=1)
    df_copy['lower_shadow'] = df_copy[['open', 'close']].min(axis=1) - df_copy['low']
    df_copy['is_bullish'] = df_copy['close'] > df_copy['open']
    df_copy['is_bearish'] = df_copy['close'] < df_copy['open']
    
    # Calculate average candle size (for relative comparisons)
    avg_body_size = df_copy['body_size'].rolling(window=14).mean()
    
    # 1. Doji Pattern (very small body)
    df_copy['doji'] = df_copy['body_size'] < (0.1 * avg_body_size)
    
    # 2. Hammer Pattern (bullish reversal)
    df_copy['hammer'] = (
        (df_copy['body_size'] > 0) &  # Must have a body
        (df_copy['lower_shadow'] > 2 * df_copy['body_size']) &  # Long lower shadow
        (df_copy['upper_shadow'] < 0.3 * df_copy['body_size'])  # Short upper shadow
    )
    
    # 3. Shooting Star (bearish reversal)
    df_copy['shooting_star'] = (
        (df_copy['body_size'] > 0) &  # Must have a body
        (df_copy['upper_shadow'] > 2 * df_copy['body_size']) &  # Long upper shadow
        (df_copy['lower_shadow'] < 0.3 * df_copy['body_size'])  # Short lower shadow
    )
    
    # 4. Engulfing Patterns
    # Bullish Engulfing
    df_copy['bullish_engulfing'] = (
        df_copy['is_bullish'] &  # Current candle is bullish
        df_copy['is_bearish'].shift(1) &  # Previous candle is bearish
        (df_copy['open'] < df_copy['close'].shift(1)) &  # Open below previous close
        (df_copy['close'] > df_copy['open'].shift(1))  # Close above previous open
    )
    
    # Bearish Engulfing
    df_copy['bearish_engulfing'] = (
        df_copy['is_bearish'] &  # Current candle is bearish
        df_copy['is_bullish'].shift(1) &  # Previous candle is bullish
        (df_copy['open'] > df_copy['close'].shift(1)) &  # Open above previous close
        (df_copy['close'] < df_copy['open'].shift(1))  # Close below previous open
    )
    
    # 5. Morning Star (bullish reversal)
    df_copy['morning_star'] = (
        df_copy['is_bearish'].shift(2) &  # First candle is bearish
        (df_copy['body_size'].shift(1) < 0.5 * df_copy['body_size'].shift(2)) &  # Second candle has a small body
        df_copy['is_bullish'] &  # Third candle is bullish
        (df_copy['open'] > df_copy['close'].shift(2))  # Gap between first and third
    )
    
    # 6. Evening Star (bearish reversal)
    df_copy['evening_star'] = (
        df_copy['is_bullish'].shift(2) &  # First candle is bullish
        (df_copy['body_size'].shift(1) < 0.5 * df_copy['body_size'].shift(2)) &  # Second candle has a small body
        df_copy['is_bearish'] &  # Third candle is bearish
        (df_copy['open'] < df_copy['close'].shift(2))  # Gap between first and third
    )
    
    # 7. Three White Soldiers (bullish continuation)
    df_copy['three_white_soldiers'] = (
        df_copy['is_bullish'] & 
        df_copy['is_bullish'].shift(1) & 
        df_copy['is_bullish'].shift(2) &
        (df_copy['close'] > df_copy['close'].shift(1)) & 
        (df_copy['close'].shift(1) > df_copy['close'].shift(2)) &
        (df_copy['open'] > df_copy['open'].shift(1)) & 
        (df_copy['open'].shift(1) > df_copy['open'].shift(2))
    )
    
    # 8. Three Black Crows (bearish continuation)
    df_copy['three_black_crows'] = (
        df_copy['is_bearish'] & 
        df_copy['is_bearish'].shift(1) & 
        df_copy['is_bearish'].shift(2) &
        (df_copy['close'] < df_copy['close'].shift(1)) & 
        (df_copy['close'].shift(1) < df_copy['close'].shift(2)) &
        (df_copy['open'] < df_copy['open'].shift(1)) & 
        (df_copy['open'].shift(1) < df_copy['open'].shift(2))
    )
    
    # 9. Harami Pattern (reversal)
    # Bullish Harami
    df_copy['bullish_harami'] = (
        df_copy['is_bearish'].shift(1) &  # Previous candle is bearish
        df_copy['is_bullish'] &  # Current candle is bullish
        (df_copy['high'] < df_copy['open'].shift(1)) &  # Current high below previous open
        (df_copy['low'] > df_copy['close'].shift(1))  # Current low above previous close
    )
    
    # Bearish Harami
    df_copy['bearish_harami'] = (
        df_copy['is_bullish'].shift(1) &  # Previous candle is bullish
        df_copy['is_bearish'] &  # Current candle is bearish
        (df_copy['high'] < df_copy['close'].shift(1)) &  # Current high below previous close
        (df_copy['low'] > df_copy['open'].shift(1))  # Current low above previous open
    )
    
    # Combine all bullish and bearish patterns
    df_copy['bullish_pattern'] = (
        df_copy['hammer'] | 
        df_copy['bullish_engulfing'] | 
        df_copy['morning_star'] | 
        df_copy['three_white_soldiers'] |
        df_copy['bullish_harami']
    )
    
    df_copy['bearish_pattern'] = (
        df_copy['shooting_star'] | 
        df_copy['bearish_engulfing'] | 
        df_copy['evening_star'] | 
        df_copy['three_black_crows'] |
        df_copy['bearish_harami']
    )
    
    return df_copy
