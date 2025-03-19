"""
Moving Averages indicator implementation.
"""

import pandas as pd
import numpy as np


def calculate_ma(df: pd.DataFrame, period: int, result_column: str, 
                column: str = 'close', ma_type: str = 'simple') -> pd.DataFrame:
    """
    Calculate a Moving Average for a given dataframe.
    
    Args:
        df: DataFrame containing price data
        period: Moving average period
        result_column: Column name for the resulting MA values
        column: Column name to use for calculations
        ma_type: Type of moving average ('simple', 'exponential', 'weighted')
        
    Returns:
        DataFrame with Moving Average values added
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Calculate the specified type of moving average
    if ma_type.lower() == 'simple':
        df_copy[result_column] = df_copy[column].rolling(window=period).mean()
    elif ma_type.lower() == 'exponential':
        df_copy[result_column] = df_copy[column].ewm(span=period, adjust=False).mean()
    elif ma_type.lower() == 'weighted':
        weights = np.arange(1, period + 1)
        df_copy[result_column] = df_copy[column].rolling(window=period).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True
        )
    else:
        raise ValueError(f"Unknown moving average type: {ma_type}")
    
    return df_copy


def calculate_ema(df: pd.DataFrame, period: int, result_column: str,
                 column: str = 'close') -> pd.DataFrame:
    """
    Calculate an Exponential Moving Average (EMA) for a given dataframe.
    
    Args:
        df: DataFrame containing price data
        period: EMA period
        result_column: Column name for the resulting EMA values
        column: Column name to use for calculations
        
    Returns:
        DataFrame with EMA values added
    """
    return calculate_ma(df, period, result_column, column, ma_type='exponential')


def calculate_cross(df: pd.DataFrame, fast_column: str, slow_column: str, 
                   result_column: str = 'cross') -> pd.DataFrame:
    """
    Calculate crossover points between two indicator lines.
    
    Args:
        df: DataFrame containing indicator data
        fast_column: Column name for the faster-moving line
        slow_column: Column name for the slower-moving line
        result_column: Column name for the resulting crossover values
        
    Returns:
        DataFrame with crossover values added (1 for bullish cross, -1 for bearish cross, 0 for no cross)
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Initialize the result column with zeros
    df_copy[result_column] = 0
    
    # Create shifted versions of the columns to detect changes
    df_copy['fast_prev'] = df_copy[fast_column].shift(1)
    df_copy['slow_prev'] = df_copy[slow_column].shift(1)
    
    # Bullish cross: fast line crosses above slow line
    bullish_cross = (df_copy['fast_prev'] <= df_copy['slow_prev']) & \
                    (df_copy[fast_column] > df_copy[slow_column])
    
    # Bearish cross: fast line crosses below slow line
    bearish_cross = (df_copy['fast_prev'] >= df_copy['slow_prev']) & \
                    (df_copy[fast_column] < df_copy[slow_column])
    
    # Set crossover values
    df_copy.loc[bullish_cross, result_column] = 1
    df_copy.loc[bearish_cross, result_column] = -1
    
    # Drop temporary columns
    df_copy.drop(['fast_prev', 'slow_prev'], axis=1, inplace=True)
    
    return df_copy
