"""
Strategy Selector - Determines which trading strategy to use based on market conditions.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union

import config


class StrategySelector:
    """
    Class that determines which trading strategy to use based on market conditions.
    Chooses between scalping and swing trading depending on volatility, volume, and trend strength.
    """
    
    def __init__(self):
        """Initialize the strategy selector."""
        self.logger = logging.getLogger('binance_bot')
        
        # Configuration
        self.volatility_threshold = config.STRATEGY_SELECTION['volatility_threshold']
        self.volume_increase_threshold = config.STRATEGY_SELECTION['volume_increase_threshold']
        self.trend_strength_threshold = config.STRATEGY_SELECTION['trend_strength_threshold']
    
    def select_strategy(self, scalping_data: pd.DataFrame, swing_data: pd.DataFrame) -> str:
        """
        Determine which strategy to use based on current market conditions.
        
        Args:
            scalping_data: Recent market data for scalping timeframe
            swing_data: Recent market data for swing timeframe
            
        Returns:
            str: Strategy to use ('scalping' or 'swing')
        """
        # Calculate metrics
        volatility = self._calculate_volatility(scalping_data)
        volume_increase = self._calculate_volume_increase(scalping_data)
        trend_strength = self._calculate_trend_strength(swing_data)
        
        self.logger.debug(f"Market conditions - Volatility: {volatility:.4f}, "
                         f"Volume Increase: {volume_increase:.2f}x, "
                         f"Trend Strength: {trend_strength:.2f}")
        
        # Decision logic
        # 1. High volatility and high volume spike -> Use scalping for quick profits
        if volatility > self.volatility_threshold and volume_increase > self.volume_increase_threshold:
            self.logger.debug("Selected scalping strategy due to high volatility and volume increase")
            return 'scalping'
        
        # 2. Strong trend -> Use swing trading to capture the larger move
        if trend_strength > self.trend_strength_threshold:
            self.logger.debug("Selected swing strategy due to strong trend")
            return 'swing'
        
        # 3. Low volatility, no strong trend -> Use scalping for range-bound markets
        if volatility < self.volatility_threshold / 2:
            self.logger.debug("Selected scalping strategy due to low volatility range-bound market")
            return 'scalping'
        
        # Default to scalping if unsure
        self.logger.debug("Selected scalping strategy as default")
        return 'scalping'
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """
        Calculate recent price volatility using True Range.
        
        Args:
            data: Recent market data
            
        Returns:
            float: Volatility measure as a decimal (e.g., 0.02 for 2%)
        """
        if len(data) < 20:
            return 0.0
        
        # Use the last 20 candles
        recent_data = data.iloc[-20:].copy()
        
        # Calculate True Range
        recent_data['high_low'] = recent_data['high'] - recent_data['low']
        recent_data['high_close'] = np.abs(recent_data['high'] - recent_data['close'].shift(1))
        recent_data['low_close'] = np.abs(recent_data['low'] - recent_data['close'].shift(1))
        
        recent_data['tr'] = recent_data[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Calculate Average True Range
        atr = recent_data['tr'].mean()
        
        # Normalize by current price
        volatility = atr / recent_data['close'].iloc[-1]
        
        return volatility
    
    def _calculate_volume_increase(self, data: pd.DataFrame) -> float:
        """
        Calculate recent volume increase compared to average.
        
        Args:
            data: Recent market data
            
        Returns:
            float: Volume increase as a ratio (e.g., 1.5 for 50% increase)
        """
        if len(data) < 20:
            return 1.0
        
        # Calculate average volume over last 20 candles
        avg_volume = data['volume'].iloc[-20:-1].mean()  # Exclude the most recent candle
        
        # Get the most recent volume
        recent_volume = data['volume'].iloc[-1]
        
        if avg_volume == 0:
            return 1.0
            
        # Calculate the ratio
        volume_increase = recent_volume / avg_volume
        
        return volume_increase
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate trend strength using Average Directional Index (ADX).
        
        Args:
            data: Recent market data
            
        Returns:
            float: ADX value (0-100, where >25 indicates a strong trend)
        """
        if len(data) < 30:
            return 0.0
        
        # Use the last 30 candles
        recent_data = data.iloc[-30:].copy()
        
        # Calculate +DM and -DM
        recent_data['+dm'] = np.where(
            (recent_data['high'] - recent_data['high'].shift(1)) > 
            (recent_data['low'].shift(1) - recent_data['low']),
            np.maximum(recent_data['high'] - recent_data['high'].shift(1), 0),
            0
        )
        
        recent_data['-dm'] = np.where(
            (recent_data['low'].shift(1) - recent_data['low']) > 
            (recent_data['high'] - recent_data['high'].shift(1)),
            np.maximum(recent_data['low'].shift(1) - recent_data['low'], 0),
            0
        )
        
        # Calculate True Range
        recent_data['high_low'] = recent_data['high'] - recent_data['low']
        recent_data['high_close'] = np.abs(recent_data['high'] - recent_data['close'].shift(1))
        recent_data['low_close'] = np.abs(recent_data['low'] - recent_data['close'].shift(1))
        recent_data['tr'] = recent_data[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Calculate smoothed values
        period = 14
        recent_data['smoothed_tr'] = recent_data['tr'].rolling(window=period).sum()
        recent_data['smoothed_+dm'] = recent_data['+dm'].rolling(window=period).sum()
        recent_data['smoothed_-dm'] = recent_data['-dm'].rolling(window=period).sum()
        
        # Calculate +DI and -DI
        recent_data['+di'] = 100 * recent_data['smoothed_+dm'] / recent_data['smoothed_tr']
        recent_data['-di'] = 100 * recent_data['smoothed_-dm'] / recent_data['smoothed_tr']
        
        # Calculate DX
        recent_data['dx'] = 100 * np.abs(recent_data['+di'] - recent_data['-di']) / \
                          (recent_data['+di'] + recent_data['-di'])
        
        # Calculate ADX (smoothed DX)
        adx = recent_data['dx'].iloc[-period:].mean()
        
        return adx
