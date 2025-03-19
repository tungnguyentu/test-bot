"""
Bollinger Bands indicator implementation.
"""

import pandas as pd
import numpy as np


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0,
                             column: str = 'close') -> pd.DataFrame:
    """
    Calculate Bollinger Bands for a given dataframe.
    
    Args:
        df: DataFrame containing price data
        period: Moving average period
        std_dev: Number of standard deviations for the bands
        column: Column name to use for calculations
        
    Returns:
        DataFrame with Bollinger Bands values added (bb_middle, bb_upper, bb_lower)
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Calculate middle band (simple moving average)
    df_copy['bb_middle'] = df_copy[column].rolling(window=period).mean()
    
    # Calculate standard deviation
    rolling_std = df_copy[column].rolling(window=period).std()
    
    # Calculate upper and lower bands
    df_copy['bb_upper'] = df_copy['bb_middle'] + (rolling_std * std_dev)
    df_copy['bb_lower'] = df_copy['bb_middle'] - (rolling_std * std_dev)
    
    # Calculate bandwidth (not always used, but useful for additional analysis)
    df_copy['bb_bandwidth'] = (df_copy['bb_upper'] - df_copy['bb_lower']) / df_copy['bb_middle']
    
    # Calculate percent B (price position relative to the bands, 0-1 range)
    df_copy['bb_percent_b'] = (df_copy[column] - df_copy['bb_lower']) / (df_copy['bb_upper'] - df_copy['bb_lower'])
    
    return df_copy
