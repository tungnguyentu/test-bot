"""
TradingBot class - Core functionality for the Binance Futures trading bot.
"""

import os
import time
import logging
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

from binance.client import Client
from binance.exceptions import BinanceAPIException

import config
from strategies.strategy_selector import StrategySelector
from strategies.scalping_strategy import ScalpingStrategy
from strategies.swing_strategy import SwingStrategy
from utils.trade_manager import TradeManager
from utils.risk_calculator import RiskCalculator
from utils.binance_client import BinanceClient


class TradingBot:
    """
    Main trading bot class that coordinates all trading activities.
    """
    
    def __init__(self, trading_pairs: List[str], mode: str = 'paper',
                 telegram_client: Any = None):
        """
        Initialize the trading bot.
        
        Args:
            trading_pairs: List of trading pairs to trade
            mode: Trading mode - 'live' or 'paper'
            telegram_client: Telegram client instance for notifications
        """
        self.logger = logging.getLogger('binance_bot')
        self.trading_pairs = trading_pairs
        self.mode = mode
        self.telegram = telegram_client
        
        # Initialize Binance client
        self.binance = BinanceClient(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET'),
            testnet=(mode == 'paper')
        )
        
        # Initialize strategies
        self.strategy_selector = StrategySelector()
        self.scalping_strategy = ScalpingStrategy()
        self.swing_strategy = SwingStrategy()
        
        # Initialize trade manager and risk calculator
        self.trade_manager = TradeManager(self.binance, self.telegram)
        self.risk_calculator = RiskCalculator()
        
        # Performance tracking
        self.initial_balance = None
        self.start_time = None
        
        # Flag to control the main loop
        self.is_running = False
        
        # Dictionary to track active strategies for each pair
        self.active_strategies = {}
        
        self.logger.info(f"Trading bot initialized in {mode} mode")
    
    def run(self):
        """Start the trading bot."""
        if self.is_running:
            self.logger.warning("Trading bot is already running")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # Get initial account balance
        self.initial_balance = self.binance.get_account_balance(config.QUOTE_ASSET)
        self.logger.info(f"Initial balance: {self.initial_balance} {config.QUOTE_ASSET}")
        
        if self.telegram:
            self.telegram.send_message(
                f"🏁 Trading bot started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Initial balance: {self.initial_balance} {config.QUOTE_ASSET}"
            )
        
        # Set up threads for each trading pair
        threads = []
        for pair in self.trading_pairs:
            thread = threading.Thread(
                target=self._trading_loop,
                args=(pair,),
                name=f"trading-{pair}"
            )
            thread.daemon = True
            thread.start()
            threads.append(thread)
            self.logger.info(f"Started trading thread for {pair}")
        
        # Main control thread
        try:
            self._monitor_loop()
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
        finally:
            self.is_running = False
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=5.0)
            
            self._report_performance()
    
    def _trading_loop(self, trading_pair: str):
        """
        Trading loop for a specific trading pair.
        
        Args:
            trading_pair: The trading pair to trade
        """
        self.logger.info(f"Starting trading loop for {trading_pair}")
        
        while self.is_running:
            try:
                # Check if we can trade (under max drawdown, etc.)
                if not self._can_continue_trading():
                    time.sleep(60)  # Check again in 1 minute
                    continue
                
                # Get market data
                scalping_data = self.binance.get_klines(
                    trading_pair, 
                    config.DEFAULT_TIMEFRAMES['scalping']
                )
                
                swing_data = self.binance.get_klines(
                    trading_pair, 
                    config.DEFAULT_TIMEFRAMES['swing']
                )
                
                # Determine which strategy to use based on market conditions
                strategy_name = self.strategy_selector.select_strategy(
                    scalping_data, 
                    swing_data
                )
                
                self.active_strategies[trading_pair] = strategy_name
                
                # Execute the selected strategy
                if strategy_name == 'scalping':
                    signals = self.scalping_strategy.generate_signals(scalping_data)
                    timeframe = config.DEFAULT_TIMEFRAMES['scalping']
                else:  # swing trading
                    signals = self.swing_strategy.generate_signals(swing_data)
                    timeframe = config.DEFAULT_TIMEFRAMES['swing']
                
                # Check if we have any signals
                if signals['entry'] and not self.trade_manager.has_open_position(trading_pair):
                    # Calculate position size based on risk
                    risk_params = self.risk_calculator.calculate_position_size(
                        trading_pair=trading_pair,
                        entry_price=signals['entry_price'],
                        stop_loss=signals['stop_loss'],
                        account_balance=self.binance.get_account_balance(config.QUOTE_ASSET)
                    )
                    
                    # Execute trade
                    trade_result = self.trade_manager.open_position(
                        trading_pair=trading_pair,
                        side=signals['side'],
                        entry_price=signals['entry_price'],
                        stop_loss=signals['stop_loss'],
                        take_profit=signals['take_profit'],
                        quantity=risk_params['quantity'],
                        strategy=strategy_name
                    )
                    
                    if trade_result and self.telegram:
                        reasoning = self._generate_trade_reasoning(
                            trading_pair, signals, strategy_name, timeframe
                        )
                        self.telegram.send_message(reasoning)
                
                # Check for exit signals on existing positions
                elif signals['exit'] and self.trade_manager.has_open_position(trading_pair):
                    self.trade_manager.close_position(
                        trading_pair=trading_pair,
                        exit_price=signals['exit_price'],
                        reason=signals['exit_reason']
                    )
                
                # Update trailing stops if needed
                self.trade_manager.update_trailing_stops(trading_pair)
                
                # Sleep to avoid excessive API calls
                time.sleep(10)  # Check every 10 seconds
                
            except BinanceAPIException as e:
                self.logger.error(f"Binance API error in trading loop for {trading_pair}: {str(e)}")
                if self.telegram:
                    self.telegram.send_message(f"⚠️ API Error: {str(e)}")
                time.sleep(60)  # Wait before retrying
                
            except Exception as e:
                self.logger.exception(f"Error in trading loop for {trading_pair}: {str(e)}")
                if self.telegram:
                    self.telegram.send_message(f"❌ Error in {trading_pair} trading loop: {str(e)}")
                time.sleep(60)  # Wait before retrying
    
    def _monitor_loop(self):
        """
        Monitor overall performance and system status.
        """
        while self.is_running:
            try:
                # Check overall account status
                current_balance = self.binance.get_account_balance(config.QUOTE_ASSET)
                open_positions = self.trade_manager.get_open_positions()
                
                # Check drawdown
                if self.initial_balance > 0:
                    drawdown = (self.initial_balance - current_balance) / self.initial_balance
                    if drawdown >= config.MAX_DRAWDOWN_PERCENTAGE:
                        self.logger.warning(
                            f"Maximum drawdown reached: {drawdown:.2%}. Pausing trading."
                        )
                        if self.telegram:
                            self.telegram.send_message(
                                f"⚠️ Maximum drawdown reached: {drawdown:.2%}\n"
                                f"Trading paused to protect capital."
                            )
                
                # Log current status every hour
                if datetime.now().minute == 0:
                    self._log_status(current_balance, open_positions)
                
                # Sleep for 5 minutes before checking again
                time.sleep(300)
                
            except Exception as e:
                self.logger.exception(f"Error in monitor loop: {str(e)}")
                time.sleep(300)  # Try again in 5 minutes
    
    def _can_continue_trading(self) -> bool:
        """
        Check if we can continue trading based on various conditions.
        
        Returns:
            bool: True if trading can continue, False otherwise
        """
        # Check if max drawdown has been exceeded
        current_balance = self.binance.get_account_balance(config.QUOTE_ASSET)
        
        if self.initial_balance > 0:
            drawdown = (self.initial_balance - current_balance) / self.initial_balance
            if drawdown >= config.MAX_DRAWDOWN_PERCENTAGE:
                return False
        
        # Check if we have too many open positions
        open_positions_count = len(self.trade_manager.get_open_positions())
        if open_positions_count >= config.MAX_OPEN_POSITIONS:
            return False
        
        return True
    
    def _log_status(self, current_balance: float, open_positions: List[Dict]):
        """
        Log current trading status.
        
        Args:
            current_balance: Current account balance
            open_positions: List of open positions
        """
        pnl = current_balance - self.initial_balance
        pnl_percentage = (pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        self.logger.info(f"Current balance: {current_balance} {config.QUOTE_ASSET}")
        self.logger.info(f"P&L: {pnl:.2f} {config.QUOTE_ASSET} ({pnl_percentage:.2f}%)")
        self.logger.info(f"Open positions: {len(open_positions)}")
        
        if self.telegram:
            status_message = (
                f"📊 Status Update\n"
                f"Current balance: {current_balance:.2f} {config.QUOTE_ASSET}\n"
                f"P&L: {pnl:.2f} {config.QUOTE_ASSET} ({pnl_percentage:.2f}%)\n"
                f"Open positions: {len(open_positions)}\n\n"
            )
            
            if open_positions:
                status_message += "Open Positions:\n"
                for pos in open_positions:
                    unrealized_pnl = pos.get('unrealized_pnl', 0)
                    status_message += (
                        f"- {pos['symbol']}: {pos['side']} {pos['quantity']}\n"
                        f"  Entry: {pos['entry_price']:.4f} | Current: {pos['current_price']:.4f}\n"
                        f"  PnL: {unrealized_pnl:.2f} {config.QUOTE_ASSET}\n"
                    )
            
            self.telegram.send_message(status_message)
    
    def _generate_trade_reasoning(self, trading_pair: str, signals: Dict,
                                 strategy_name: str, timeframe: str) -> str:
        """
        Generate natural language reasoning for a trade.
        
        Args:
            trading_pair: The trading pair being traded
            signals: Trading signals dictionary
            strategy_name: Name of the strategy used
            timeframe: Timeframe of the analysis
            
        Returns:
            str: Natural language explanation of the trade
        """
        # Current time for the message
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Basic trade information
        basic_info = (
            f"🔔 New Trade Alert - {trading_pair}\n"
            f"Time: {current_time}\n"
            f"Strategy: {strategy_name.capitalize()}\n"
            f"Timeframe: {timeframe}\n"
            f"Direction: {'Long' if signals['side'] == 'BUY' else 'Short'}\n"
        )
        
        # Trade details
        trade_details = (
            f"Entry Price: {signals['entry_price']:.4f}\n"
            f"Stop Loss: {signals['stop_loss']:.4f}\n"
            f"Take Profit: {signals['take_profit']:.4f}\n"
            f"Risk:Reward Ratio: {signals.get('risk_reward_ratio', 'N/A')}\n"
        )
        
        # Strategy-specific reasoning
        if strategy_name == 'scalping':
            reasoning = (
                f"RSI: {signals.get('rsi_value', 'N/A')} - "
                f"{'Oversold' if signals.get('rsi_value', 50) < 30 else 'Overbought' if signals.get('rsi_value', 50) > 70 else 'Neutral'}\n"
                f"Bollinger Bands: {'Price at lower band' if signals.get('bb_signal') == 'buy' else 'Price at upper band'}\n"
                f"MA Crossover: {'Bullish' if signals.get('ma_signal') == 'buy' else 'Bearish'}\n\n"
                f"Reasoning: "
            )
            
            if signals['side'] == 'BUY':
                reasoning += (
                    f"Entered a long position after price touched the lower Bollinger Band with "
                    f"RSI indicating oversold conditions at {signals.get('rsi_value', 'N/A')}. "
                    f"The fast MA crossed above the slow MA, confirming bullish momentum."
                )
            else:  # SELL
                reasoning += (
                    f"Entered a short position after price touched the upper Bollinger Band with "
                    f"RSI indicating overbought conditions at {signals.get('rsi_value', 'N/A')}. "
                    f"The fast MA crossed below the slow MA, confirming bearish momentum."
                )
                
        else:  # swing trading
            reasoning = (
                f"MACD: {'Bullish' if signals.get('macd_signal') == 'buy' else 'Bearish'}\n"
                f"Ichimoku: {signals.get('ichimoku_signal', 'N/A')}\n"
                f"Volume: {'Above average' if signals.get('volume_signal') == 'high' else 'Below average'}\n\n"
                f"Reasoning: "
            )
            
            if signals['side'] == 'BUY':
                reasoning += (
                    f"Entered a long position based on a bullish MACD crossover. "
                    f"Price moved above the Ichimoku cloud, indicating a strong trend shift. "
                    f"Volume is {signals.get('volume_percent', 0)}% above average, "
                    f"confirming strong buying pressure."
                )
            else:  # SELL
                reasoning += (
                    f"Entered a short position based on a bearish MACD crossover. "
                    f"Price dropped below the Ichimoku cloud, indicating a strong trend shift. "
                    f"Volume is {signals.get('volume_percent', 0)}% above average, "
                    f"confirming strong selling pressure."
                )
        
        # Combine all parts
        return basic_info + trade_details + reasoning
    
    def _report_performance(self):
        """Report overall performance when the bot is shutting down."""
        # Calculate runtime
        end_time = datetime.now()
        runtime = end_time - self.start_time
        
        # Get final balance
        final_balance = self.binance.get_account_balance(config.QUOTE_ASSET)
        
        # Calculate profit/loss
        pnl = final_balance - self.initial_balance
        pnl_percentage = (pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        # Log results
        self.logger.info(f"Bot runtime: {runtime}")
        self.logger.info(f"Initial balance: {self.initial_balance} {config.QUOTE_ASSET}")
        self.logger.info(f"Final balance: {final_balance} {config.QUOTE_ASSET}")
        self.logger.info(f"P&L: {pnl:.2f} {config.QUOTE_ASSET} ({pnl_percentage:.2f}%)")
        
        # Send report via Telegram
        if self.telegram:
            self.telegram.send_message(
                f"📈 Trading Bot Performance Report\n"
                f"Runtime: {runtime}\n"
                f"Initial balance: {self.initial_balance:.2f} {config.QUOTE_ASSET}\n"
                f"Final balance: {final_balance:.2f} {config.QUOTE_ASSET}\n"
                f"P&L: {pnl:.2f} {config.QUOTE_ASSET} ({pnl_percentage:.2f}%)\n\n"
                f"Bot has been shut down."
            )
