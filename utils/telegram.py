"""
Telegram notification utilities for the trading bot.
"""

import logging
import os
import asyncio
from typing import Dict, List, Optional
import telegram


class TelegramNotifier:
    """
    Class for sending notifications to a Telegram chat about trading activities.
    """
    
    def __init__(self, token: str, chat_id: str, enabled: bool = True):
        """
        Initialize the TelegramNotifier.
        
        Args:
            token: Telegram bot token
            chat_id: Telegram chat ID to send messages to
            enabled: Whether notifications are enabled
        """
        self.logger = logging.getLogger('binance_bot')
        self.enabled = enabled
        self.token = token
        self.chat_id = chat_id
        
        if self.enabled:
            try:
                self.bot = telegram.Bot(token=self.token)
                self.logger.info("Telegram bot initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Telegram bot: {e}")
                self.enabled = False
    
    async def send_message(self, message: str) -> bool:
        """
        Send a text message to the Telegram chat.
        
        Args:
            message: Message text
            
        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=telegram.constants.ParseMode.MARKDOWN
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_message_sync(self, message: str) -> bool:
        """
        Synchronous wrapper for send_message.
        
        Args:
            message: Message text
            
        Returns:
            bool: Success status
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.send_message(message))
            return result
        finally:
            loop.close()
    
    def send_trade_entry(self, symbol: str, side: str, entry_price: float, 
                        stop_loss: float, take_profit: float, reasoning: str) -> bool:
        """
        Send notification about a new trade entry.
        
        Args:
            symbol: Trading pair symbol
            side: Trade direction (BUY/SELL)
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            reasoning: AI-generated reasoning for the trade
            
        Returns:
            bool: Success status
        """
        emoji = "🟢" if side == "BUY" else "🔴"
        risk_reward = abs(take_profit - entry_price) / abs(entry_price - stop_loss)
        
        message = (
            f"*{emoji} NEW TRADE: {side} {symbol} {emoji}*\n\n"
            f"Entry Price: `{entry_price:.8f}`\n"
            f"Stop Loss: `{stop_loss:.8f}`\n"
            f"Take Profit: `{take_profit:.8f}`\n"
            f"Risk/Reward: `{risk_reward:.2f}`\n\n"
            f"*Trade Reasoning:*\n{reasoning}"
        )
        
        return self.send_message_sync(message)
    
    def send_trade_exit(self, symbol: str, side: str, entry_price: float, 
                       exit_price: float, profit_loss: float, pnl_percent: float,
                       exit_reason: str) -> bool:
        """
        Send notification about a trade exit.
        
        Args:
            symbol: Trading pair symbol
            side: Original trade direction (BUY/SELL)
            entry_price: Entry price
            exit_price: Exit price
            profit_loss: Absolute P&L
            pnl_percent: P&L as percentage
            exit_reason: Reason for exit
            
        Returns:
            bool: Success status
        """
        emoji = "✅" if profit_loss > 0 else "❌"
        close_side = "SELL" if side == "BUY" else "BUY"
        
        message = (
            f"*{emoji} CLOSED TRADE: {close_side} {symbol} {emoji}*\n\n"
            f"Entry Price: `{entry_price:.8f}`\n"
            f"Exit Price: `{exit_price:.8f}`\n"
            f"P&L: `{profit_loss:.8f}` ({pnl_percent:.2f}%)\n"
            f"Exit Reason: {exit_reason}"
        )
        
        return self.send_message_sync(message)
    
    def send_error(self, error_message: str) -> bool:
        """
        Send error notification.
        
        Args:
            error_message: Error description
            
        Returns:
            bool: Success status
        """
        message = f"*⚠️ ERROR ⚠️*\n\n{error_message}"
        return self.send_message_sync(message)
    
    def send_strategy_switch(self, symbol: str, from_strategy: str, 
                           to_strategy: str, reason: str) -> bool:
        """
        Send notification about strategy switching.
        
        Args:
            symbol: Trading pair symbol
            from_strategy: Original strategy
            to_strategy: New strategy
            reason: Reason for switching
            
        Returns:
            bool: Success status
        """
        message = (
            f"*🔄 STRATEGY SWITCH: {symbol} 🔄*\n\n"
            f"From: `{from_strategy}`\n"
            f"To: `{to_strategy}`\n"
            f"Reason: {reason}"
        )
        
        return self.send_message_sync(message)
    
    def send_performance_report(self, total_trades: int, win_rate: float, 
                              profit_factor: float, total_pnl: float,
                              max_drawdown: float, sharpe_ratio: float) -> bool:
        """
        Send performance report with key metrics.
        
        Args:
            total_trades: Number of completed trades
            win_rate: Win rate percentage
            profit_factor: Profit factor
            total_pnl: Total P&L
            max_drawdown: Maximum drawdown percentage
            sharpe_ratio: Sharpe ratio
            
        Returns:
            bool: Success status
        """
        message = (
            f"*📊 PERFORMANCE REPORT 📊*\n\n"
            f"Total Trades: `{total_trades}`\n"
            f"Win Rate: `{win_rate:.2f}%`\n"
            f"Profit Factor: `{profit_factor:.2f}`\n"
            f"Total P&L: `{total_pnl:.8f}`\n"
            f"Max Drawdown: `{max_drawdown:.2f}%`\n"
            f"Sharpe Ratio: `{sharpe_ratio:.2f}`"
        )
        
        return self.send_message_sync(message)
    
    def generate_trade_reasoning(self, signals: Dict) -> str:
        """
        Generate detailed trade reasoning based on signal data.
        
        Args:
            signals: Dictionary containing signal details
            
        Returns:
            str: Formatted trade reasoning
        """
        if not signals:
            return "No signal data available for reasoning."
        
        side = signals.get('side')
        is_buy = side == 'BUY'
        
        # Build the base reasoning
        if is_buy:
            reasoning = "Bullish signal detected based on:"
        else:
            reasoning = "Bearish signal detected based on:"
        
        # Add MACD information if available
        if 'macd_signal' in signals and signals['macd_signal']:
            macd_signal = signals['macd_signal']
            if macd_signal == 'buy':
                reasoning += "\n- MACD showing bullish momentum with positive crossover"
            elif macd_signal == 'sell':
                reasoning += "\n- MACD showing bearish momentum with negative crossover"
        
        # Add RSI information if available
        if 'rsi' in signals and signals['rsi']:
            rsi = signals['rsi']
            if is_buy and rsi < 40:
                reasoning += f"\n- RSI is oversold at {rsi:.2f}, suggesting a potential reversal"
            elif not is_buy and rsi > 60:
                reasoning += f"\n- RSI is overbought at {rsi:.2f}, suggesting a potential reversal"
            else:
                reasoning += f"\n- RSI at {rsi:.2f} confirms the trend direction"
        
        # Add Bollinger Bands information if available
        if 'bollinger_signal' in signals and signals['bollinger_signal']:
            bb_signal = signals['bollinger_signal']
            if bb_signal == 'lower_band_touch' and is_buy:
                reasoning += "\n- Price touched the lower Bollinger Band, indicating oversold conditions"
            elif bb_signal == 'upper_band_touch' and not is_buy:
                reasoning += "\n- Price touched the upper Bollinger Band, indicating overbought conditions"
            elif bb_signal == 'middle_band_cross_up':
                reasoning += "\n- Price crossed above the middle Bollinger Band, confirming upward momentum"
            elif bb_signal == 'middle_band_cross_down':
                reasoning += "\n- Price crossed below the middle Bollinger Band, confirming downward momentum"
        
        # Add Ichimoku information if available
        if 'ichimoku_signal' in signals and signals['ichimoku_signal']:
            ichimoku_signal = signals['ichimoku_signal']
            if ichimoku_signal == 'bullish':
                reasoning += "\n- Ichimoku Cloud showing bullish signals (price above cloud, tenkan-sen > kijun-sen)"
            elif ichimoku_signal == 'bearish':
                reasoning += "\n- Ichimoku Cloud showing bearish signals (price below cloud, tenkan-sen < kijun-sen)"
        
        # Add volume information if available
        if 'volume_signal' in signals and signals['volume_signal']:
            volume_signal = signals['volume_signal']
            volume_percent = signals.get('volume_percent', 0)
            if volume_signal == 'high':
                reasoning += f"\n- High volume ({volume_percent:.2f}% above average) confirms the strength of the move"
            elif volume_signal == 'increasing':
                reasoning += "\n- Increasing volume confirms the trend direction"
            elif volume_signal == 'decreasing':
                reasoning += "\n- Decreasing volume suggests potential trend exhaustion"
        
        # Add candlestick pattern information if available
        if 'pattern' in signals and signals['pattern']:
            pattern = signals['pattern']
            if is_buy:
                reasoning += f"\n- Bullish candlestick pattern detected: {pattern}"
            else:
                reasoning += f"\n- Bearish candlestick pattern detected: {pattern}"
        
        # Add risk/reward information
        if 'risk_reward_ratio' in signals and signals['risk_reward_ratio'] > 0:
            reasoning += f"\n- Favorable risk-to-reward ratio of {signals['risk_reward_ratio']:.2f}"
        
        return reasoning
