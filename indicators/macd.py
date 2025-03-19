"""
MACD (Moving Average Convergence Divergence) indicator implementation.
"""

import pandas as pd
import numpy as np


def calculate_macd(df: pd.DataFrame, fast_length: int = 12, slow_length: int = 26,
                  signal_length: int = 9, column: str = 'close') -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence) for a given dataframe.
    
    Args:
        df: DataFrame containing price data
        fast_length: Fast EMA period
        slow_length: Slow EMA period
        signal_length: Signal line EMA period
        column: Column name to use for calculations
        
    Returns:
        DataFrame with MACD values added (macd, macd_signal, macd_histogram)
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Calculate Fast and Slow EMAs
    fast_ema = df_copy[column].ewm(span=fast_length, adjust=False).mean()
    slow_ema = df_copy[column].ewm(span=slow_length, adjust=False).mean()
    
    # Calculate MACD line
    df_copy['macd'] = fast_ema - slow_ema
    
    # Calculate Signal line
    df_copy['macd_signal'] = df_copy['macd'].ewm(span=signal_length, adjust=False).mean()
    
    # Calculate Histogram
    df_copy['macd_histogram'] = df_copy['macd'] - df_copy['macd_signal']
    
    return df_copy
