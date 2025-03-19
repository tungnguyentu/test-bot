"""
Scalping Strategy - Implements short-term scalping trading strategy.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union

import config
from indicators.rsi import calculate_rsi
from indicators.bollinger_bands import calculate_bollinger_bands
from indicators.moving_averages import calculate_ma
from indicators.trend_strength import calculate_adx, calculate_slope, identify_strong_trend
from indicators.advanced_filters import filter_trade_setup, check_volume_confirmation
from utils.stop_management import (
    calculate_volatility_based_stop, 
    calculate_volatility_based_target,
    calculate_swing_based_stop,
    calculate_swing_based_target,
    update_trailing_stop
)

class ScalpingStrategy:
    """
    Implements a scalping strategy using RSI, Bollinger Bands, and Moving Averages.
    Designed for quick trades with small profit targets.
    """
    
    def __init__(self):
        """Initialize the scalping strategy with config parameters."""
        self.logger = logging.getLogger('binance_bot')
        
        # Load parameters from config
        self.rsi_period = config.SCALPING_STRATEGY['rsi_period']
        self.rsi_overbought = config.SCALPING_STRATEGY['rsi_overbought']
        self.rsi_oversold = config.SCALPING_STRATEGY['rsi_oversold']
        
        self.bb_period = config.SCALPING_STRATEGY['bb_period']
        self.bb_std_dev = config.SCALPING_STRATEGY['bb_std_dev']
        
        self.ma_short_period = config.SCALPING_STRATEGY['ma_short_period']
        self.ma_long_period = config.SCALPING_STRATEGY['ma_long_period']
        
        self.min_profit_target = config.SCALPING_STRATEGY['min_profit_target']
        self.max_stop_loss = config.SCALPING_STRATEGY['max_stop_loss']
        self.take_profit_ratio = config.SCALPING_STRATEGY['take_profit_ratio']
        
        # Advanced parameters
        self.adx_period = config.SCALPING_STRATEGY.get('adx_period', 14)
        self.adx_threshold = config.SCALPING_STRATEGY.get('adx_threshold', 20)
        self.min_risk_reward = config.SCALPING_STRATEGY.get('min_risk_reward', 1.5)
        self.slope_period = config.SCALPING_STRATEGY.get('slope_period', 10)
        self.require_volume_confirmation = config.SCALPING_STRATEGY.get('require_volume_confirmation', True)
        self.use_volatility_stops = config.SCALPING_STRATEGY.get('use_volatility_stops', True)
        self.atr_multiplier = config.SCALPING_STRATEGY.get('atr_multiplier', 2.0)
        
        # Trailing stop settings
        self.use_trailing_stop = config.SCALPING_STRATEGY.get('use_trailing_stop', True)
        self.trailing_activation_pct = config.SCALPING_STRATEGY.get('trailing_activation_pct', 0.8)
        self.trailing_step_pct = config.SCALPING_STRATEGY.get('trailing_step_pct', 0.3)
    
    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """
        Generate trading signals based on the scalping strategy.
        
        Args:
            data: Market data as a DataFrame with OHLCV data
            
        Returns:
            Dict: Trading signals including entry/exit points, stop loss, and take profit levels
        """
        min_required_data = max(
            self.rsi_period, 
            self.bb_period, 
            self.ma_long_period, 
            self.adx_period,
            self.slope_period
        ) + 5
        
        if len(data) < min_required_data:
            return self._get_empty_signal()
        
        # Create working copy of the data
        df = data.copy()
        
        # Calculate basic indicators
        df = calculate_rsi(df, self.rsi_period)
        df = calculate_bollinger_bands(df, self.bb_period, self.bb_std_dev)
        df = calculate_ma(df, self.ma_short_period, 'ma_short')
        df = calculate_ma(df, self.ma_long_period, 'ma_long')
        
        # Calculate advanced indicators
        df = calculate_adx(df, self.adx_period)
        df = calculate_slope(df, 'close', self.slope_period)
        
        # Get trend information
        trend_info = identify_strong_trend(df, self.adx_threshold)
        
        # Initialize signals dictionary with empty values
        signals = self._get_empty_signal()
        
        # Get trend information for filtering signals
        trend_info = identify_strong_trend(df, self.adx_threshold)
        signals.update(trend_info)
        
        # Only generate buy signals in bullish trends or no strong trend
        if trend_info['trend_direction'] == 'bullish' or not trend_info['is_strong_trend']:
            buy_signal = self._check_buy_signal(df)
            if buy_signal:
                signals['entry'] = True
                signals['side'] = 'BUY'
                signals['entry_price'] = df['close'].iloc[-1]
                signals['bb_signal'] = 'buy'
                signals['ma_signal'] = 'buy'
                
                # Calculate optimized stop loss
                if self.use_volatility_stops:
                    signals['stop_loss'] = calculate_volatility_based_stop(df, 'BUY', self.atr_multiplier)
                    signals['take_profit'] = calculate_volatility_based_target(df, 'BUY', self.atr_multiplier)
                else:
                    signals['stop_loss'] = self._calculate_stop_loss(df, 'BUY')
                    # Calculate take profit with better risk-reward
                    signals['take_profit'] = self._calculate_take_profit(
                        signals['entry_price'], signals['stop_loss'], 'BUY'
                    )
                
                # Calculate risk-reward ratio
                risk = signals['entry_price'] - signals['stop_loss']
                reward = signals['take_profit'] - signals['entry_price']
                signals['risk_reward_ratio'] = reward / risk if risk > 0 else 0.0
                
                # Check volume confirmation
                signals['volume_confirmed'] = check_volume_confirmation(df) if self.require_volume_confirmation else True
                
                # Apply comprehensive filtering
                if signals['risk_reward_ratio'] < self.min_risk_reward:
                    signals['entry'] = False
                    signals['filter_reason'] = f"Risk-reward ratio {signals['risk_reward_ratio']:.2f} below minimum {self.min_risk_reward}"
                elif self.require_volume_confirmation and not signals['volume_confirmed']:
                    signals['entry'] = False
                    signals['filter_reason'] = "Insufficient volume confirmation"
        
        if trend_info['trend_direction'] == 'bearish' or not trend_info['is_strong_trend']:
            sell_signal = self._check_sell_signal(df)
            if sell_signal:
                signals['entry'] = True
                signals['side'] = 'SELL'
                signals['entry_price'] = df['close'].iloc[-1]
                signals['bb_signal'] = 'sell'
                signals['ma_signal'] = 'sell'
                
                # Calculate optimized stop loss
                if self.use_volatility_stops:
                    signals['stop_loss'] = calculate_volatility_based_stop(df, 'SELL', self.atr_multiplier)
                    signals['take_profit'] = calculate_volatility_based_target(df, 'SELL', self.atr_multiplier)
                else:
                    signals['stop_loss'] = self._calculate_stop_loss(df, 'SELL')
                    # Calculate take profit with better risk-reward
                    signals['take_profit'] = self._calculate_take_profit(
                        signals['entry_price'], signals['stop_loss'], 'SELL'
                    )
                
                # Calculate risk-reward ratio
                risk = signals['stop_loss'] - signals['entry_price']
                reward = signals['entry_price'] - signals['take_profit']
                signals['risk_reward_ratio'] = reward / risk if risk > 0 else 0.0
                
                # Check volume confirmation
                signals['volume_confirmed'] = check_volume_confirmation(df) if self.require_volume_confirmation else True
                
                # Apply comprehensive filtering
                if signals['risk_reward_ratio'] < self.min_risk_reward:
                    signals['entry'] = False
                    signals['filter_reason'] = f"Risk-reward ratio {signals['risk_reward_ratio']:.2f} below minimum {self.min_risk_reward}"
                elif self.require_volume_confirmation and not signals['volume_confirmed']:
                    signals['entry'] = False
                    signals['filter_reason'] = "Insufficient volume confirmation"
        
        return signals
    
    def _check_buy_signal(self, df: pd.DataFrame) -> bool:
        """
        Check for buy signal based on RSI, Bollinger Bands, and moving averages.
        
        Args:
            df: DataFrame with indicator data
            
        Returns:
            Boolean indicating buy signal
        """
        if len(df) < 3:
            return False
        
        # Get the latest and previous values
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # 1. RSI indicating oversold and starting to rise
        rsi_oversold = latest['rsi'] < self.rsi_oversold
        rsi_rising = latest['rsi'] > previous['rsi']
        
        # 2. Price near or below lower Bollinger Band
        bb_buy_signal = latest['close'] <= latest['bb_lower'] * 1.005  # Allow a small buffer (0.5%)
        
        # 3. Price crossing above short-term MA
        ma_cross_up = previous['close'] < previous['ma_short'] and latest['close'] > latest['ma_short']
        
        # 4. Both MAs trending up (slope)
        ma_short_up = df['ma_short'].iloc[-3] < df['ma_short'].iloc[-1]
        
        # 5. Check for bullish divergence (price making lower lows but RSI making higher lows)
        price_lower_low = (
            df['low'].iloc[-3] > df['low'].iloc[-5] and
            df['low'].iloc[-1] < df['low'].iloc[-3]
        )
        rsi_higher_low = (
            df['rsi'].iloc[-3] < df['rsi'].iloc[-5] and
            df['rsi'].iloc[-1] > df['rsi'].iloc[-3]
        )
        bullish_divergence = price_lower_low and rsi_higher_low
        
        # Combine signals - main signal
        main_buy_signal = (
            (rsi_oversold and rsi_rising) or  # RSI shows oversold and starting to rise
            (bb_buy_signal and rsi_rising) or  # Price at/below BB lower and RSI rising
            ma_cross_up or  # Price crossing above short MA
            bullish_divergence  # RSI showing bullish divergence
        )
        
        # Additional filters
        trend_aligned = latest['ma_short'] > latest['ma_long'] or ma_short_up  # Uptrend or potential reversal
        
        # Final buy signal
        return main_buy_signal and trend_aligned
    
    def _check_sell_signal(self, df: pd.DataFrame) -> bool:
        """
        Check for sell signal based on RSI, Bollinger Bands, and moving averages.
        
        Args:
            df: DataFrame with indicator data
            
        Returns:
            Boolean indicating sell signal
        """
        if len(df) < 3:
            return False
        
        # Get the latest and previous values
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # 1. RSI indicating overbought and starting to fall
        rsi_overbought = latest['rsi'] > self.rsi_overbought
        rsi_falling = latest['rsi'] < previous['rsi']
        
        # 2. Price near or above upper Bollinger Band
        bb_sell_signal = latest['close'] >= latest['bb_upper'] * 0.995  # Allow a small buffer (0.5%)
        
        # 3. Price crossing below short-term MA
        ma_cross_down = previous['close'] > previous['ma_short'] and latest['close'] < latest['ma_short']
        
        # 4. Both MAs trending down (slope)
        ma_short_down = df['ma_short'].iloc[-3] > df['ma_short'].iloc[-1]
        
        # 5. Check for bearish divergence (price making higher highs but RSI making lower highs)
        price_higher_high = (
            df['high'].iloc[-3] < df['high'].iloc[-5] and
            df['high'].iloc[-1] > df['high'].iloc[-3]
        )
        rsi_lower_high = (
            df['rsi'].iloc[-3] > df['rsi'].iloc[-5] and
            df['rsi'].iloc[-1] < df['rsi'].iloc[-3]
        )
        bearish_divergence = price_higher_high and rsi_lower_high
        
        # Combine signals - main signal
        main_sell_signal = (
            (rsi_overbought and rsi_falling) or  # RSI shows overbought and starting to fall
            (bb_sell_signal and rsi_falling) or  # Price at/above BB upper and RSI falling
            ma_cross_down or  # Price crossing below short MA
            bearish_divergence  # RSI showing bearish divergence
        )
        
        # Additional filters
        trend_aligned = latest['ma_short'] < latest['ma_long'] or ma_short_down  # Downtrend or potential reversal
        
        # Final sell signal
        return main_sell_signal and trend_aligned
    
    def _check_exit_signal(self, df: pd.DataFrame, current_trade: Dict = None) -> tuple:
        """
        Check for exit signals and implement trailing stop logic.
        
        Args:
            df: DataFrame with indicator data
            current_trade: Current active trade information (optional)
            
        Returns:
            Tuple of (exit_signal, exit_price, exit_reason)
        """
        if len(df) < 2:
            return False, 0.0, ""
        
        # Default values
        exit_signal = False
        exit_price = df['close'].iloc[-1]
        exit_reason = ""
        
        # If no current trade, no exit needed
        if not current_trade or not current_trade.get('entry', False):
            return exit_signal, exit_price, exit_reason
        
        # Get latest data
        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Get trade details
        side = current_trade.get('side', None)
        entry_price = current_trade.get('entry_price', 0.0)
        stop_loss = current_trade.get('stop_loss', 0.0)
        take_profit = current_trade.get('take_profit', 0.0)
        
        # Update trailing stop if needed
        if self.use_trailing_stop and side:
            # Check if trade is in profit enough to activate trailing stop
            if side == 'BUY':
                profit_pct = (latest['close'] - entry_price) / entry_price * 100
                # Activate trailing stop if profit exceeds threshold
                if profit_pct >= self.trailing_activation_pct and not current_trade.get('trailing_started', False):
                    # Start trailing
                    current_trade['trailing_started'] = True
                    current_trade['trailing_reference'] = latest['close']
                    # Set stop to lock in some profit
                    new_stop = entry_price + (latest['close'] - entry_price) * 0.5
                    if new_stop > stop_loss:
                        stop_loss = new_stop
                        current_trade['stop_loss'] = stop_loss
                
                # Update trailing stop if already activated
                elif current_trade.get('trailing_started', False):
                    trailing_ref = current_trade.get('trailing_reference', entry_price)
                    # If price moved up enough, adjust stop loss
                    if latest['close'] > trailing_ref * (1 + self.trailing_step_pct/100):
                        # Move stop up to lock in more profit
                        new_stop = stop_loss + (latest['close'] - trailing_ref) * 0.5
                        stop_loss = new_stop
                        current_trade['stop_loss'] = new_stop
                        current_trade['trailing_reference'] = latest['close']
                
            elif side == 'SELL':
                profit_pct = (entry_price - latest['close']) / entry_price * 100
                # Activate trailing stop if profit exceeds threshold
                if profit_pct >= self.trailing_activation_pct and not current_trade.get('trailing_started', False):
                    # Start trailing
                    current_trade['trailing_started'] = True
                    current_trade['trailing_reference'] = latest['close']
                    # Set stop to lock in some profit
                    new_stop = entry_price - (entry_price - latest['close']) * 0.5
                    if new_stop < stop_loss:
                        stop_loss = new_stop
                        current_trade['stop_loss'] = stop_loss
                
                # Update trailing stop if already activated
                elif current_trade.get('trailing_started', False):
                    trailing_ref = current_trade.get('trailing_reference', entry_price)
                    # If price moved down enough, adjust stop loss
                    if latest['close'] < trailing_ref * (1 - self.trailing_step_pct/100):
                        # Move stop down to lock in more profit
                        new_stop = stop_loss - (trailing_ref - latest['close']) * 0.5
                        stop_loss = new_stop
                        current_trade['stop_loss'] = new_stop
                        current_trade['trailing_reference'] = latest['close']
        
        # Check for stop loss hit
        if side == 'BUY' and latest['low'] <= stop_loss:
            exit_signal = True
            exit_price = max(latest['open'], stop_loss)  # Realistic exit assuming slippage
            exit_reason = "Stop loss hit"
            
        elif side == 'SELL' and latest['high'] >= stop_loss:
            exit_signal = True
            exit_price = min(latest['open'], stop_loss)  # Realistic exit assuming slippage
            exit_reason = "Stop loss hit"
            
        # Check for take profit hit
        elif side == 'BUY' and latest['high'] >= take_profit:
            exit_signal = True
            exit_price = min(latest['open'], take_profit)  # Realistic exit assuming slippage
            exit_reason = "Take profit hit"
            
        elif side == 'SELL' and latest['low'] <= take_profit:
            exit_signal = True
            exit_price = max(latest['open'], take_profit)  # Realistic exit assuming slippage
            exit_reason = "Take profit hit"
        
        # Check for exit based on RSI reversal
        elif side == 'BUY' and latest['rsi'] > 70 and latest['rsi'] < previous['rsi']:
            exit_signal = True
            exit_price = latest['close']
            exit_reason = "RSI overbought and starting to fall"
            
        elif side == 'SELL' and latest['rsi'] < 30 and latest['rsi'] > previous['rsi']:
            exit_signal = True
            exit_price = latest['close']
            exit_reason = "RSI oversold and starting to rise"
        
        # Check for MA crossover exit
        elif side == 'BUY' and previous['close'] > previous['ma_short'] and latest['close'] < latest['ma_short']:
            exit_signal = True
            exit_price = latest['close']
            exit_reason = "Price crossed below MA"
            
        elif side == 'SELL' and previous['close'] < previous['ma_short'] and latest['close'] > latest['ma_short']:
            exit_signal = True
            exit_price = latest['close']
            exit_reason = "Price crossed above MA"
        
        return exit_signal, exit_price, exit_reason
    
    def _calculate_stop_loss(self, df: pd.DataFrame, side: str) -> float:
        """
        Calculate stop loss level based on recent price action and volatility.
        
        Args:
            df: DataFrame with calculated indicators
            side: Trade direction ('BUY' or 'SELL')
            
        Returns:
            float: Stop loss price
        """
        # Recent candles
        recent = df.iloc[-10:]
        latest = df.iloc[-1]
        
        # Use ATR to gauge volatility
        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        recent_copy = recent.copy()
        recent_copy['tr'] = np.maximum(
            recent_copy['high'] - recent_copy['low'],
            np.maximum(
                abs(recent_copy['high'] - recent_copy['close'].shift(1)),
                abs(recent_copy['low'] - recent_copy['close'].shift(1))
            )
        )
        atr = recent_copy['tr'].mean()
        
        # Calculate max stop loss amount
        max_stop_amount = latest['close'] * self.max_stop_loss
        
        # For buy orders, stop is below current price
        if side == 'BUY':
            # Find recent swing low
            swing_low = recent['low'].min()
            
            # Stop loss is the higher of (current price - ATR) or swing low
            atr_stop = latest['close'] - atr
            stop_price = max(atr_stop, swing_low)
            
            # Ensure stop loss is not too far from entry
            if latest['close'] - stop_price > max_stop_amount:
                stop_price = latest['close'] - max_stop_amount
        
        # For sell orders, stop is above current price
        else:
            # Find recent swing high
            swing_high = recent['high'].max()
            
            # Stop loss is the lower of (current price + ATR) or swing high
            atr_stop = latest['close'] + atr
            stop_price = min(atr_stop, swing_high)
            
            # Ensure stop loss is not too far from entry
            if stop_price - latest['close'] > max_stop_amount:
                stop_price = latest['close'] + max_stop_amount
        
        return stop_price
    
    def _calculate_take_profit(self, entry_price: float, stop_loss: float, side: str) -> float:
        """
        Calculate take profit price based on risk-reward ratio.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            side: Trade direction ('BUY' or 'SELL')
            
        Returns:
            float: Take profit price
        """
        try:
            # Ensure we're working with scalar values
            entry_price = float(entry_price)
            stop_loss = float(stop_loss)
            
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
            
            return float(take_profit)
            
        except (TypeError, ValueError) as e:
            # In case of errors, use a default profit target
            print(f"Error calculating take profit: {e}")
            
            if side == 'BUY':
                return entry_price * 1.01  # Default 1% target
            else:
                return entry_price * 0.99  # Default 1% target
    
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
            
            # Additional data for reasoning
            'rsi_value': 0.0,
            'bb_signal': None,
            'ma_signal': None,
            'adx_value': 0.0,
            'trend_direction': 'neutral',
            'is_strong_trend': False,
            'volume_confirmed': False,
            'filter_reason': '',
            'trailing_stop_settings': {
                'enabled': self.use_trailing_stop,
                'activation_pct': self.trailing_activation_pct,
                'step_pct': self.trailing_step_pct,
                'trailing_started': False,
                'trailing_reference': 0.0
            }
        }
