"""
Volume-based indicator implementations.
"""

import pandas as pd
import numpy as np


def calculate_volume_indicators(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Calculate various volume-based indicators for a given dataframe.
    
    Args:
        df: DataFrame containing price and volume data
        period: Period for volume indicators
        
    Returns:
        DataFrame with volume indicators added (volume_ma, volume_ratio, obv, and more)
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Calculate Volume Moving Average
    df_copy['volume_ma'] = df_copy['volume'].rolling(window=period).mean()
    
    # Calculate Volume Ratio (current volume / average volume)
    df_copy['volume_ratio'] = df_copy['volume'] / df_copy['volume_ma']
    
    # Calculate On-Balance Volume (OBV)
    df_copy['obv'] = 0
    # Initialize OBV with first row value as 0
    obv = 0
    obv_values = [obv]
    
    # Calculate OBV for each row without using vectorized operations that cause the error
    for i in range(1, len(df_copy)):
        if df_copy['close'].iloc[i] > df_copy['close'].iloc[i-1]:
            obv += df_copy['volume'].iloc[i]
        elif df_copy['close'].iloc[i] < df_copy['close'].iloc[i-1]:
            obv -= df_copy['volume'].iloc[i]
        # If prices are equal, OBV doesn't change
        obv_values.append(obv)
    
    df_copy['obv'] = obv_values
    
    # Calculate Chaikin Money Flow (CMF)
    money_flow_multiplier = ((df_copy['close'] - df_copy['low']) - (df_copy['high'] - df_copy['close'])) / (df_copy['high'] - df_copy['low'])
    money_flow_volume = money_flow_multiplier * df_copy['volume']
    df_copy['cmf'] = money_flow_volume.rolling(window=period).sum() / df_copy['volume'].rolling(window=period).sum()
    
    # Calculate Ease of Movement (EOM)
    distance_moved = ((df_copy['high'] + df_copy['low']) / 2) - ((df_copy['high'].shift(1) + df_copy['low'].shift(1)) / 2)
    box_ratio = (df_copy['volume'] / 100000000) / (df_copy['high'] - df_copy['low'])
    df_copy['eom'] = distance_moved / box_ratio
    df_copy['eom_ma'] = df_copy['eom'].rolling(window=period).mean()
    
    # Identify high volume bars (helpful for detecting breakouts)
    significant_volume_threshold = 1.5  # 50% above average
    df_copy['high_volume'] = df_copy['volume_ratio'] > significant_volume_threshold
    
    return df_copy
