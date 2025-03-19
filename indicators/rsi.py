"""
RSI (Relative Strength Index) indicator implementation.
"""

import pandas as pd
import numpy as np


def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = 'close', 
                 result_column: str = 'rsi') -> pd.DataFrame:
    """
    Calculate the Relative Strength Index (RSI) for a given dataframe.
    
    Args:
        df: DataFrame containing price data
        period: RSI period length
        column: Column name to use for calculations
        result_column: Column name for the resulting RSI values
        
    Returns:
        DataFrame with RSI values added
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Calculate price changes
    delta = df_copy[column].diff()
    
    # Create gain and loss series
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over period
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    df_copy[result_column] = 100 - (100 / (1 + rs))
    
    # Handle cases where avg_loss is zero
    df_copy[result_column] = df_copy[result_column].fillna(100)
    
    return df_copy
