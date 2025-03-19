"""
Ichimoku Cloud indicator implementation.
"""

import pandas as pd
import numpy as np


def calculate_ichimoku(df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26,
                      senkou_period: int = 52, displacement: int = 26) -> pd.DataFrame:
    """
    Calculate Ichimoku Cloud components for a given dataframe.
    
    Args:
        df: DataFrame containing price data (high, low, close)
        tenkan_period: Tenkan-sen (Conversion Line) period
        kijun_period: Kijun-sen (Base Line) period
        senkou_period: Senkou Span B period
        displacement: Displacement period for Senkou Span and Chikou Span
        
    Returns:
        DataFrame with Ichimoku Cloud values added (tenkan_sen, kijun_sen, 
        senkou_span_a, senkou_span_b, chikou_span)
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past tenkan_period
    high_tenkan = df_copy['high'].rolling(window=tenkan_period).max()
    low_tenkan = df_copy['low'].rolling(window=tenkan_period).min()
    df_copy['tenkan_sen'] = (high_tenkan + low_tenkan) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past kijun_period
    high_kijun = df_copy['high'].rolling(window=kijun_period).max()
    low_kijun = df_copy['low'].rolling(window=kijun_period).min()
    df_copy['kijun_sen'] = (high_kijun + low_kijun) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, displaced forward by displacement periods
    df_copy['senkou_span_a'] = ((df_copy['tenkan_sen'] + df_copy['kijun_sen']) / 2).shift(displacement)
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past senkou_period, displaced forward by displacement periods
    high_senkou = df_copy['high'].rolling(window=senkou_period).max()
    low_senkou = df_copy['low'].rolling(window=senkou_period).min()
    df_copy['senkou_span_b'] = ((high_senkou + low_senkou) / 2).shift(displacement)
    
    # Calculate Chikou Span (Lagging Span): Current closing price, displaced backwards by displacement periods
    df_copy['chikou_span'] = df_copy['close'].shift(-displacement)
    
    return df_copy
