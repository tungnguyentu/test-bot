"""
Swing Strategy - Implements medium-term swing trading strategy.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union

import config
from indicators.macd import calculate_macd
from indicators.ichimoku import calculate_ichimoku
from indicators.volume import calculate_volume_indicators


class SwingStrategy:
    """
    Implements a swing trading strategy using Ichimoku Cloud, MACD, and volume analysis.
    Designed for medium-term trades that can last hours to days.
    """
    
    def __init__(self):
        """Initialize the swing strategy with config parameters."""
        self.logger = logging.getLogger('binance_bot')
        
        # Load parameters from config
        self.ichimoku_params = config.SWING_STRATEGY['ichimoku']
        self.macd_params = config.SWING_STRATEGY['macd']
        self.volume_ma_period = config.SWING_STRATEGY['volume_ma_period']
        
        self.min_profit_target = config.SWING_STRATEGY['min_profit_target']
        self.max_stop_loss = config.SWING_STRATEGY['max_stop_loss']
        self.take_profit_ratio = config.SWING_STRATEGY['take_profit_ratio']
    
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """
        Generate trading signals based on the swing trading strategy.
        
        Args:
            data: Market data as a DataFrame with OHLCV data
            
        Returns:
            Dict: Trading signals including entry/exit points, stop loss, and take profit levels
        """
        # Make sure we have enough data
        min_data_points = max(
            self.ichimoku_params['displacement'] + self.ichimoku_params['lagging_span_period'],
            self.macd_params['slow_length'] + self.macd_params['signal_smoothing'],
            self.volume_ma_period
        ) + 10
        
        if len(data) < min_data_points:
            return self._get_empty_signal()
        
        # Create working copy of the data
        df = data.copy()
        
        # Calculate indicators
        df = calculate_ichimoku(
            df, 
            self.ichimoku_params['conversion_line_period'],
            self.ichimoku_params['base_line_period'],
            self.ichimoku_params['lagging_span_period'],
            self.ichimoku_params['displacement']
        )
        
        df = calculate_macd(
            df,
            self.macd_params['fast_length'],
            self.macd_params['slow_length'],
            self.macd_params['signal_smoothing']
        )
        
        df = calculate_volume_indicators(df, self.volume_ma_period)
        
        # Initialize signals
        signals = {
            'entry': False,
            'exit': False,
            'side': None,
            'entry_price': 0.0,
            'exit_price': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'exit_reason': '',
            'risk_reward_ratio': 0.0,
            
            # Additional data for reasoning
            'macd_signal': None,
            'ichimoku_signal': None,
            'volume_signal': None,
            'volume_percent': 0.0
        }
        
        # Check for buy signal
        buy_signal = self._check_buy_signal(df)
        sell_signal = self._check_sell_signal(df)
        
        # Generate entry signals
        if buy_signal:
            signals['entry'] = True
            signals['side'] = 'BUY'
            signals['entry_price'] = df.iloc[-1]['close']
            
            # Set signal values for reasoning
            signals['macd_signal'] = 'buy'
            signals['ichimoku_signal'] = 'bullish'
            signals['volume_signal'] = 'high' if df.iloc[-1]['volume_ratio'] > 1 else 'normal'
            signals['volume_percent'] = (df.iloc[-1]['volume_ratio'] - 1) * 100
            
            # Calculate stop loss and take profit
            signals['stop_loss'] = self._calculate_stop_loss(df, 'BUY')
            signals['take_profit'] = self._calculate_take_profit(
                signals['entry_price'], signals['stop_loss'], 'BUY'
            )
            
        elif sell_signal:
            signals['entry'] = True
            signals['side'] = 'SELL'
            signals['entry_price'] = df.iloc[-1]['close']
            
            # Set signal values for reasoning
            signals['macd_signal'] = 'sell'
            signals['ichimoku_signal'] = 'bearish'
            signals['volume_signal'] = 'high' if df.iloc[-1]['volume_ratio'] > 1 else 'normal'
            signals['volume_percent'] = (df.iloc[-1]['volume_ratio'] - 1) * 100
            
            # Calculate stop loss and take profit
            signals['stop_loss'] = self._calculate_stop_loss(df, 'SELL')
            signals['take_profit'] = self._calculate_take_profit(
                signals['entry_price'], signals['stop_loss'], 'SELL'
            )
        
        # Generate exit signals
        exit_signal, exit_price, exit_reason = self._check_exit_signal(df)
        
        if exit_signal:
            signals['exit'] = True
            signals['exit_price'] = exit_price
            signals['exit_reason'] = exit_reason
        
        # Calculate risk-reward ratio if entry signal exists
        if signals['entry']:
            risk = abs(signals['entry_price'] - signals['stop_loss'])
            reward = abs(signals['entry_price'] - signals['take_profit'])
            
            if risk > 0:
                signals['risk_reward_ratio'] = reward / risk
            
        return signals
    
    def _check_buy_signal(self, df: pd.DataFrame) -> bool:
        """
        Check for buy signal based on the swing strategy rules.
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            bool: True if buy signal is present, False otherwise
        """
        # Get recent data
        recent = df.iloc[-30:]  # Use more data for swing trading
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Check MACD bullish crossover
        macd_bullish = (previous['macd_histogram'] <= 0 and latest['macd_histogram'] > 0) or \
                       (previous['macd_histogram'] < latest['macd_histogram'] > 0 and 
                        latest['macd'] < 0 and latest['macd'] > latest['macd_signal'])
        
        # Check Ichimoku Cloud signals
        # 1. Price is above the cloud
        price_above_cloud = latest['close'] > max(latest['senkou_span_a'], latest['senkou_span_b'])
        
        # 2. Conversion line crosses above base line
        tenkan_cross_kijun = (previous['tenkan_sen'] <= previous['kijun_sen']) and \
                            (latest['tenkan_sen'] > latest['kijun_sen'])
        
        # 3. Lagging span is above price from 26 periods ago
        if len(df) > self.ichimoku_params['displacement'] * 2:
            lagging_span_bullish = latest['chikou_span'] > df.iloc[-(self.ichimoku_params['displacement']+1)]['close']
        else:
            lagging_span_bullish = False
        
        # Combine Ichimoku signals (need at least 2 out of 3)
        ichimoku_bullish = sum([price_above_cloud, tenkan_cross_kijun, lagging_span_bullish]) >= 2
        
        # Check volume
        volume_increasing = latest['volume_ratio'] > 1.2  # Volume 20% above average
        
        # Combined signal - Need at least 2 of the 3 main indicators
        conditions_met = sum([macd_bullish, ichimoku_bullish, volume_increasing])
        
        return conditions_met >= 2
    
    def _check_sell_signal(self, df: pd.DataFrame) -> bool:
        """
        Check for sell signal based on the swing strategy rules.
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            bool: True if sell signal is present, False otherwise
        """
        # Get recent data
        recent = df.iloc[-30:]  # Use more data for swing trading
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Check MACD bearish crossover
        macd_bearish = (previous['macd_histogram'] >= 0 and latest['macd_histogram'] < 0) or \
                       (previous['macd_histogram'] > latest['macd_histogram'] < 0 and 
                        latest['macd'] > 0 and latest['macd'] < latest['macd_signal'])
        
        # Check Ichimoku Cloud signals
        # 1. Price is below the cloud
        price_below_cloud = latest['close'] < min(latest['senkou_span_a'], latest['senkou_span_b'])
        
        # 2. Conversion line crosses below base line
        tenkan_cross_kijun = (previous['tenkan_sen'] >= previous['kijun_sen']) and \
                            (latest['tenkan_sen'] < latest['kijun_sen'])
        
        # 3. Lagging span is below price from 26 periods ago
        if len(df) > self.ichimoku_params['displacement'] * 2:
            lagging_span_bearish = latest['chikou_span'] < df.iloc[-(self.ichimoku_params['displacement']+1)]['close']
        else:
            lagging_span_bearish = False
        
        # Combine Ichimoku signals (need at least 2 out of 3)
        ichimoku_bearish = sum([price_below_cloud, tenkan_cross_kijun, lagging_span_bearish]) >= 2
        
        # Check volume
        volume_increasing = latest['volume_ratio'] > 1.2  # Volume 20% above average
        
        # Combined signal - Need at least 2 of the 3 main indicators
        conditions_met = sum([macd_bearish, ichimoku_bearish, volume_increasing])
        
        return conditions_met >= 2
    
    def _check_exit_signal(self, df: pd.DataFrame) -> tuple:
        """
        Check for exit signal for an existing position.
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            tuple: (exit_signal, exit_price, exit_reason)
        """
        # Get recent data
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Exit signal cases
        
        # 1. MACD crossover in the opposite direction
        if (previous['macd'] > previous['macd_signal'] and latest['macd'] <= latest['macd_signal']) or \
           (previous['macd'] < previous['macd_signal'] and latest['macd'] >= latest['macd_signal']):
            return True, latest['close'], "MACD Signal Crossover"
        
        # 2. Price crosses the cloud in the opposite direction
        # For long positions: price was above cloud but now enters or goes below cloud
        if previous['close'] > max(previous['senkou_span_a'], previous['senkou_span_b']) and \
           latest['close'] <= max(latest['senkou_span_a'], latest['senkou_span_b']):
            return True, latest['close'], "Price Entered Cloud"
        
        # For short positions: price was below cloud but now enters or goes above cloud
        if previous['close'] < min(previous['senkou_span_a'], previous['senkou_span_b']) and \
           latest['close'] >= min(latest['senkou_span_a'], latest['senkou_span_b']):
            return True, latest['close'], "Price Entered Cloud"
        
        # 3. Tenkan-Kijun cross in the opposite direction
        if (previous['tenkan_sen'] > previous['kijun_sen'] and latest['tenkan_sen'] <= latest['kijun_sen']) or \
           (previous['tenkan_sen'] < previous['kijun_sen'] and latest['tenkan_sen'] >= latest['kijun_sen']):
            return True, latest['close'], "Tenkan-Kijun Crossover"
        
        # 4. Significant volume drop after a spike (potential exhaustion)
        if previous['volume_ratio'] > 2.0 and latest['volume_ratio'] < 0.7:
            return True, latest['close'], "Volume Exhaustion"
        
        # No exit signal
        return False, 0.0, ""
    
    def _calculate_stop_loss(self, df: pd.DataFrame, side: str) -> float:
        """
        Calculate stop loss level based on Ichimoku Cloud and recent swing points.
        
        Args:
            df: DataFrame with calculated indicators
            side: Trade direction ('BUY' or 'SELL')
            
        Returns:
            float: Stop loss price
        """
        # Recent candles (more for swing trading)
        recent = df.iloc[-30:]
        latest = df.iloc[-1]
        
        # Calculate max stop loss amount
        max_stop_amount = latest['close'] * self.max_stop_loss
        
        # For buy orders, stop is below current price
        if side == 'BUY':
            # Find recent swing low over a longer period
            window_size = 5  # Look for local minimums in 5-bar windows
            swing_lows = []
            
            for i in range(len(recent) - window_size):
                window = recent.iloc[i:i+window_size]
                mid_idx = window_size // 2
                mid_point = window.iloc[mid_idx]
                
                if all(mid_point['low'] <= window.iloc[j]['low'] for j in range(window_size) if j != mid_idx):
                    swing_lows.append(mid_point['low'])
            
            # If we found swing lows, use the most recent one that's below current price
            if swing_lows and swing_lows[-1] < latest['close']:
                stop_price = swing_lows[-1]
            else:
                # Fallback to Ichimoku cloud bottom
                cloud_bottom = min(latest['senkou_span_a'], latest['senkou_span_b'])
                if cloud_bottom < latest['close']:
                    stop_price = cloud_bottom
                else:
                    # Last resort: fixed percentage
                    stop_price = latest['close'] * (1 - self.max_stop_loss)
            
            # Ensure stop loss is not too far from entry
            if latest['close'] - stop_price > max_stop_amount:
                stop_price = latest['close'] - max_stop_amount
        
        # For sell orders, stop is above current price
        else:
            # Find recent swing highs over a longer period
            window_size = 5  # Look for local maximums in 5-bar windows
            swing_highs = []
            
            for i in range(len(recent) - window_size):
                window = recent.iloc[i:i+window_size]
                mid_idx = window_size // 2
                mid_point = window.iloc[mid_idx]
                
                if all(mid_point['high'] >= window.iloc[j]['high'] for j in range(window_size) if j != mid_idx):
                    swing_highs.append(mid_point['high'])
            
            # If we found swing highs, use the most recent one that's above current price
            if swing_highs and swing_highs[-1] > latest['close']:
                stop_price = swing_highs[-1]
            else:
                # Fallback to Ichimoku cloud top
                cloud_top = max(latest['senkou_span_a'], latest['senkou_span_b'])
                if cloud_top > latest['close']:
                    stop_price = cloud_top
                else:
                    # Last resort: fixed percentage
                    stop_price = latest['close'] * (1 + self.max_stop_loss)
            
            # Ensure stop loss is not too far from entry
            if stop_price - latest['close'] > max_stop_amount:
                stop_price = latest['close'] + max_stop_amount
        
        return stop_price
    
    def _calculate_take_profit(self, entry_price: float, stop_loss: float, side: str) -> float:
        """
        Calculate take profit level based on risk-reward ratio.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            side: Trade direction ('BUY' or 'SELL')
            
        Returns:
            float: Take profit price
        """
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)
        
        # Calculate take profit distance using risk-reward ratio
        take_profit_distance = stop_distance * self.take_profit_ratio
        
        # Calculate minimum profit target
        min_profit_distance = entry_price * self.min_profit_target
        
        # Use the larger of the two
        target_distance = max(take_profit_distance, min_profit_distance)
        
        # For buy orders, take profit is above entry
        if side == 'BUY':
            take_profit = entry_price + target_distance
        # For sell orders, take profit is below entry
        else:
            take_profit = entry_price - target_distance
        
        return take_profit
    
    def _get_empty_signal(self) -> Dict:
        """
        Return an empty signal dictionary.
        
        Returns:
            Dict: Empty signal template
        """
        return {
            'entry': False,
            'exit': False,
            'side': None,
            'entry_price': 0.0,
            'exit_price': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'exit_reason': '',
            'risk_reward_ratio': 0.0,
            'macd_signal': None,
            'ichimoku_signal': None,
            'volume_signal': None,
            'volume_percent': 0.0
        }
