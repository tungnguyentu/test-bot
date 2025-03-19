"""
Backtester for the Binance Futures trading bot.
"""

import os
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

import config
from utils.binance_client import BinanceClient
from strategies.scalping_strategy import ScalpingStrategy
from strategies.swing_strategy import SwingStrategy
from strategies.strategy_selector import StrategySelector
from utils.metrics import calculate_performance_metrics


class Backtester:
    """
    Backtester class for evaluating strategies with historical data.
    """
    
    def __init__(self, trading_pairs: List[str], period: str = '30d',
                 telegram_client: Optional[Any] = None):
        """
        Initialize the backtester.
        
        Args:
            trading_pairs: List of trading pairs to backtest
            period: Period to backtest (e.g. '30d', '3m', '1y')
            telegram_client: Optional Telegram client for notifications
        """
        self.logger = logging.getLogger('binance_bot')
        self.trading_pairs = trading_pairs
        self.period = period
        self.telegram = telegram_client
        
        # Initialize Binance client for historical data
        self.binance = BinanceClient(testnet=False)
        
        # Initialize strategies
        self.scalping_strategy = ScalpingStrategy()
        self.swing_strategy = SwingStrategy()
        self.strategy_selector = StrategySelector()
        
        # Set up initial account balance for backtesting
        self.initial_balance = config.BACKTEST_SETTINGS['initial_balance']
        self.current_balance = self.initial_balance
        self.max_balance = self.initial_balance  # Track high watermark
        
        # Add backtest settings as class variables
        self.backtest_settings = config.BACKTEST_SETTINGS.copy()
        
        # Fees and slippage
        self.trading_fee = config.BACKTEST_SETTINGS['trading_fee']
        self.slippage = config.BACKTEST_SETTINGS['slippage']
        
        # Trade history
        self.trades = []
        self.equity_curve = []
        
        self.logger.info(f"Backtester initialized for {trading_pairs} over {period}")
        
    def run(self) -> Dict:
        """
        Run the backtest for all trading pairs.
        
        Returns:
            Dict: Backtest results
        """
        if self.telegram:
            self.telegram.send_message(
                f"🔄 Starting backtest for {', '.join(self.trading_pairs)} over {self.period}"
            )
        
        self.logger.info(f"Starting backtest for period: {self.period}")
        
        # Convert period to start and end dates
        end_date = datetime.now()
        start_date = self._parse_period(self.period, end_date)
        
        # Store results for each pair
        all_results = {
            'trades': [],
            'equity_curves': {},
            'metrics': {},
            'summary': {}
        }
        
        # Run backtest for each pair
        for pair in self.trading_pairs:
            self.logger.info(f"Backtesting {pair}...")
            
            # Reset balance for each pair (we'll calculate combined performance later)
            self.current_balance = self.initial_balance
            self.max_balance = self.initial_balance
            self.trades = []
            self.equity_curve = [(datetime.now(), self.current_balance)]  # Initial point
            
            # Get historical data
            scalping_data = self._get_historical_data(
                pair, 
                config.DEFAULT_TIMEFRAMES['scalping'], 
                start_date, 
                end_date
            )
            
            swing_data = self._get_historical_data(
                pair, 
                config.DEFAULT_TIMEFRAMES['swing'], 
                start_date, 
                end_date
            )
            
            if scalping_data.empty or swing_data.empty:
                self.logger.error(f"Failed to get historical data for {pair}")
                continue
            
            # Run backtest for this pair
            pair_results = self._backtest_pair(pair, scalping_data, swing_data)
            
            # Store results
            all_results['trades'].extend(pair_results['trades'])
            all_results['equity_curves'][pair] = pair_results['equity_curve']
            all_results['metrics'][pair] = pair_results['metrics']
        
        # Calculate overall performance
        all_results['summary'] = self._calculate_overall_performance(all_results)
        
        if self.telegram:
            # Send summary report
            summary_msg = self._format_telegram_summary(all_results['summary'])
            self.telegram.send_message(summary_msg)
        
        self.logger.info(f"Backtest completed for {len(self.trading_pairs)} trading pairs")
        return all_results
        
    def _backtest_pair(self, pair: str, scalping_data: pd.DataFrame, 
                      swing_data: pd.DataFrame) -> Dict:
        """
        Backtest a single trading pair.
        
        Args:
            pair: Trading pair to backtest
            scalping_data: Historical data for scalping strategy
            swing_data: Historical data for swing strategy
            
        Returns:
            Dict: Results for this pair
        """
        # Make copies to avoid modifying original data
        scalping_df = scalping_data.copy()
        swing_df = swing_data.copy()
        
        # Initialize tracking variables
        trades = []
        equity_curve = [(scalping_df.iloc[0]['timestamp'], self.initial_balance)]
        current_position = None
        
        # Iterate through scalping timeframes
        for i in range(len(scalping_df) - 1):
            # Skip if we're at the last candle
            if i >= len(scalping_df) - 1:
                break
                
            # Extract current market data (up to current candle)
            current_scalping_data = scalping_df.iloc[:i+1]
            
            # Find corresponding swing data period
            swing_end_time = current_scalping_data.iloc[-1]['timestamp']
            current_swing_data = swing_df[swing_df['timestamp'] <= swing_end_time]
            
            if len(current_swing_data) < 2:
                continue  # Not enough swing data yet
            
            # Determine which strategy to use
            strategy_name = self.strategy_selector.select_strategy(
                current_scalping_data, 
                current_swing_data
            )
            
            # Generate signals based on the selected strategy
            if strategy_name == 'scalping':
                signals = self.scalping_strategy.generate_signals(current_scalping_data)
                timeframe = config.DEFAULT_TIMEFRAMES['scalping']
            else:  # swing trading
                signals = self.swing_strategy.generate_signals(current_swing_data)
                timeframe = config.DEFAULT_TIMEFRAMES['swing']
            
            current_time = current_scalping_data.iloc[-1]['timestamp']
            current_price = current_scalping_data.iloc[-1]['close']
            
            # Check if we should open a position
            if current_position is None and signals['entry']:
                # Calculate position size based on risk
                max_risk_amount = self.current_balance * config.ACCOUNT_RISK_PER_TRADE
                stop_distance = abs(signals['entry_price'] - signals['stop_loss'])
                
                # Avoid division by zero and NaN values
                if stop_distance > 0 and not pd.isna(stop_distance):
                    # Calculate the number of contracts we can buy with our risk amount
                    # Formula: Risk Amount / (Entry Price - Stop Loss)
                    contracts = max_risk_amount / stop_distance
                    
                    # Calculate the position size in base currency
                    position_size_in_base_currency = contracts
                    
                    # Calculate the USD value and apply a leverage limit
                    position_value = position_size_in_base_currency * signals['entry_price']
                    
                    # Check if position value exceeds max leverage
                    max_position_value = self.current_balance * config.DEFAULT_LEVERAGE
                    
                    if position_value > max_position_value:
                        # Scale down to respect leverage limit
                        position_size_in_base_currency = max_position_value / signals['entry_price']
                        self.logger.warning(f"Position size for {pair} reduced from {contracts:.4f} to {position_size_in_base_currency:.4f} due to leverage limit")
                    
                    # Make sure position size is not too large relative to account
                    # Don't risk more than 5% of account on any single trade
                    max_safe_position_value = self.current_balance * 0.05 * config.DEFAULT_LEVERAGE
                    if position_value > max_safe_position_value:
                        position_size_in_base_currency = max_safe_position_value / signals['entry_price']
                        self.logger.warning(f"Position size for {pair} reduced to {position_size_in_base_currency:.4f} to limit risk")
                else:
                    # Default to a conservative position size if stop distance is zero/invalid
                    # Use 1% of account size with leverage as position value
                    position_value = self.current_balance * 0.01 * config.DEFAULT_LEVERAGE
                    position_size_in_base_currency = position_value / signals['entry_price']
                    self.logger.warning(f"Invalid stop distance detected for {pair}. Using default conservative position size.")
                
                # Apply slippage to entry
                entry_price = signals['entry_price'] * (1 + self.slippage if signals['side'] == 'BUY' else 1 - self.slippage)
                
                # Open position
                current_position = {
                    'pair': pair,
                    'side': signals['side'],
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'stop_loss': signals['stop_loss'],
                    'take_profit': signals['take_profit'],
                    'size': position_size_in_base_currency,
                    'strategy': strategy_name
                }
                
                # Apply trading fee
                self.current_balance -= position_size_in_base_currency * entry_price * self.trading_fee
                
                # Update max balance if current balance is higher
                self.max_balance = max(self.max_balance, self.current_balance)
                
                self.logger.debug(
                    f"Opened {signals['side']} position for {pair} at {entry_price} "
                    f"using {strategy_name} strategy"
                )
            
            # Check if we should close an existing position
            elif current_position is not None:
                exit_signal = False
                exit_price = current_price
                exit_reason = ""
                
                # Check for stop loss hit
                if (current_position['side'] == 'BUY' and current_price <= current_position['stop_loss']) or \
                   (current_position['side'] == 'SELL' and current_price >= current_position['stop_loss']):
                    exit_signal = True
                    exit_price = current_position['stop_loss']
                    exit_reason = "Stop Loss"
                
                # Check for take profit hit
                elif (current_position['side'] == 'BUY' and current_price >= current_position['take_profit']) or \
                     (current_position['side'] == 'SELL' and current_price <= current_position['take_profit']):
                    exit_signal = True
                    exit_price = current_position['take_profit']
                    exit_reason = "Take Profit"
                
                # Check for exit signal from strategy
                elif signals['exit']:
                    exit_signal = True
                    # Apply slippage to exit
                    exit_price = signals['exit_price'] * (1 - self.slippage if current_position['side'] == 'BUY' else 1 + self.slippage)
                    exit_reason = signals['exit_reason']
                
                # If we should exit, close the position
                if exit_signal:
                    # Calculate PnL more realistically
                    position_value = current_position['size'] * current_position['entry_price']
                    margin_used = position_value / config.DEFAULT_LEVERAGE
                    
                    # Calculate raw PnL based on the price difference and position size
                    # For futures trading, profit = price_diff * contracts
                    if current_position['side'] == 'BUY':
                        # Long position: profit when exit price > entry price
                        raw_profit = (exit_price - current_position['entry_price']) * current_position['size']
                    else:
                        # Short position: profit when exit price < entry price
                        raw_profit = (current_position['entry_price'] - exit_price) * current_position['size']
                    
                    # Apply trading fee (entry and exit fees)
                    entry_fee = position_value * self.trading_fee
                    exit_fee = current_position['size'] * exit_price * self.trading_fee
                    total_fees = entry_fee + exit_fee
                    
                    # Calculate final profit
                    profit = raw_profit - total_fees
                    
                    # Apply risk management constraints
                    # For futures, max loss is limited to margin used (liquidation)
                    # Max profit depends on price movement
                    if profit < -margin_used:
                        profit = -margin_used  # Can't lose more than margin (liquidation)
                        self.logger.warning(f"Position would have been liquidated - loss limited to margin: {margin_used:.2f}")
                    
                    # For very large profits, apply a more realistic cap
                    # In practice, it's unlikely to make more than 5x margin in a short trade
                    if profit > margin_used * 5:
                        old_profit = profit
                        profit = margin_used * 5
                        self.logger.warning(f"Unrealistic profit {old_profit:.2f} capped at {profit:.2f} (5x margin)")
                    
                    # Update account balance
                    self.current_balance += margin_used + profit  # Return margin + profit
                    
                    # Update max balance if current balance is higher
                    self.max_balance = max(self.max_balance, self.current_balance)
                    
                    # Calculate profit percentage with safety check
                    profit_pct = (profit / margin_used) * 100 if margin_used != 0 else 0
                    
                    # Record trade
                    trade = {
                        'pair': pair,
                        'side': current_position['side'],
                        'entry_time': current_position['entry_time'],
                        'entry_price': current_position['entry_price'],
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'strategy': current_position['strategy']
                    }
                    
                    trades.append(trade)
                    
                    # Format P&L values with protection against NaN for logging
                    profit_display = f"{profit:.2f}" if not pd.isna(profit) else "0.00"
                    pct_display = f"{trade['profit_pct']:.2f}" if not pd.isna(trade['profit_pct']) else "0.00"
                    
                    self.logger.debug(
                        f"Closed {current_position['side']} position for {pair} at {exit_price} "
                        f"with P&L: {profit_display} ({pct_display}%)"
                    )
                    
                    # Reset current position
                    current_position = None
            
            # Record equity at this point in time
            equity_curve.append((current_time, self.current_balance))
        
        # If we still have an open position at the end, close it at the last price
        if current_position is not None:
            last_price = scalping_df.iloc[-1]['close']
            
            # Calculate PnL more realistically
            position_value = current_position['size'] * current_position['entry_price']
            margin_used = position_value / config.DEFAULT_LEVERAGE
            
            # Calculate raw PnL based on the price difference and position size
            # For futures trading, profit = price_diff * contracts
            if current_position['side'] == 'BUY':
                # Long position: profit when exit price > entry price
                raw_profit = (last_price - current_position['entry_price']) * current_position['size']
            else:
                # Short position: profit when exit price < entry price
                raw_profit = (current_position['entry_price'] - last_price) * current_position['size']
            
            # Apply trading fee (entry and exit fees)
            entry_fee = position_value * self.trading_fee
            exit_fee = current_position['size'] * last_price * self.trading_fee
            total_fees = entry_fee + exit_fee
            
            # Calculate final profit
            profit = raw_profit - total_fees
            
            # Apply risk management constraints
            # For futures, max loss is limited to margin used (liquidation)
            # Max profit depends on price movement
            if profit < -margin_used:
                profit = -margin_used  # Can't lose more than margin (liquidation)
                self.logger.warning(f"Position would have been liquidated - loss limited to margin: {margin_used:.2f}")
            
            # For very large profits, apply a more realistic cap
            # In practice, it's unlikely to make more than 5x margin in a short trade
            if profit > margin_used * 5:
                old_profit = profit
                profit = margin_used * 5
                self.logger.warning(f"Unrealistic profit {old_profit:.2f} capped at {profit:.2f} (5x margin)")
            
            # Update account balance
            self.current_balance += margin_used + profit  # Return margin + profit
            
            # Update max balance if current balance is higher
            self.max_balance = max(self.max_balance, self.current_balance)
            
            # Calculate profit percentage with safety check
            profit_pct = (profit / margin_used) * 100 if margin_used != 0 else 0
            
            # Record trade
            trade = {
                'pair': pair,
                'side': current_position['side'],
                'entry_time': current_position['entry_time'],
                'entry_price': current_position['entry_price'],
                'exit_time': scalping_df.iloc[-1]['timestamp'],
                'exit_price': last_price,
                'exit_reason': "End of Backtest",
                'profit': profit,
                'profit_pct': profit_pct,
                'strategy': current_position['strategy']
            }
            
            trades.append(trade)
            
            # Format P&L values with protection against NaN for logging
            profit_display = f"{profit:.2f}" if not pd.isna(profit) else "0.00"
            pct_display = f"{trade['profit_pct']:.2f}" if not pd.isna(trade['profit_pct']) else "0.00"
            
            self.logger.debug(
                f"Closed {current_position['side']} position for {pair} at {last_price} "
                f"at end of backtest with P&L: {profit_display} ({pct_display}%)"
            )
            
            # Add final equity point
            equity_curve.append((scalping_df.iloc[-1]['timestamp'], self.current_balance))
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(trades, self.initial_balance, self.current_balance)
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': metrics
        }
    
    def _get_historical_data(self, pair: str, timeframe: str, 
                            start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get historical data for a trading pair.
        
        Args:
            pair: Trading pair to get data for
            timeframe: Timeframe for the data
            start_date: Start date for the data
            end_date: End date for the data
            
        Returns:
            pd.DataFrame: Historical data
        """
        try:
            # Convert datetime objects to millisecond timestamps for Binance API
            start_ms = int(start_date.timestamp() * 1000)
            end_ms = int(end_date.timestamp() * 1000)
            
            # Get klines from Binance
            klines = self.binance.get_historical_klines(pair, timeframe, start_ms, end_ms)
            
            if not klines:
                self.logger.error(f"No historical data returned for {pair} at {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
                df[col] = df[col].astype(float)
            
            return df
            
        except Exception as e:
            self.logger.exception(f"Error getting historical data: {str(e)}")
            return pd.DataFrame()
    
    def _parse_period(self, period: str, end_date: datetime) -> datetime:
        """
        Parse a period string and calculate the start date.
        
        Args:
            period: Period string (e.g. '30d', '3m', '1y')
            end_date: End date
            
        Returns:
            datetime: Start date
        """
        unit = period[-1].lower()
        value = int(period[:-1])
        
        if unit == 'd':
            return end_date - timedelta(days=value)
        elif unit == 'w':
            return end_date - timedelta(weeks=value)
        elif unit == 'm':
            return end_date - timedelta(days=value * 30)  # Approximate months
        elif unit == 'y':
            return end_date - timedelta(days=value * 365)  # Approximate years
        else:
            self.logger.warning(f"Unknown period unit: {unit}, defaulting to 30 days")
            return end_date - timedelta(days=30)
    
    def _calculate_overall_performance(self, results: Dict) -> Dict:
        """
        Calculate overall performance across all trading pairs.
        
        Args:
            results: Results from individual trading pairs
            
        Returns:
            Dict: Overall performance metrics
        """
        all_trades = results.get('trades', [])
        
        if not all_trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'avg_profit': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'scalping_trades': 0,
                'swing_trades': 0,
                'returns': 0.0,
                'max_balance': self.initial_balance
            }
        
        # Calculate cumulative metrics
        total_trades = len(all_trades)
        winning_trades = [t for t in all_trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in all_trades if t.get('profit', 0) <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        # Calculate win rate
        win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate profit metrics with safety checks
        # Filter out any extreme values, NaNs, or infinities
        valid_profits = [
            t.get('profit', 0) for t in all_trades 
            if not pd.isna(t.get('profit', 0)) and np.isfinite(t.get('profit', 0))
        ]
        
        total_profit = sum(valid_profits) if valid_profits else 0
        
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        # Calculate average win and loss with safety checks
        valid_wins = [
            t.get('profit', 0) for t in winning_trades 
            if not pd.isna(t.get('profit', 0)) and np.isfinite(t.get('profit', 0))
        ]
        
        valid_losses = [
            abs(t.get('profit', 0)) for t in losing_trades 
            if not pd.isna(t.get('profit', 0)) and np.isfinite(t.get('profit', 0))
        ]
        
        avg_win = sum(valid_wins) / len(valid_wins) if valid_wins else 0
        avg_loss = sum(valid_losses) / len(valid_losses) if valid_losses else 0
        
        # Calculate profit factor (gross profit / gross loss)
        gross_profit = sum(valid_wins) if valid_wins else 0
        gross_loss = sum(valid_losses) if valid_losses else 0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Calculate trades by strategy
        scalping_trades = len([t for t in all_trades if t.get('strategy', '') == 'scalping'])
        swing_trades = len([t for t in all_trades if t.get('strategy', '') == 'swing'])
        
        # Calculate return 
        initial_balance = self.initial_balance
        final_balance = self.current_balance
        returns = ((final_balance - initial_balance) / initial_balance) * 100 if initial_balance > 0 else 0
        
        # Make sure returns match direction of profit
        if (total_profit < 0 and returns > 0) or (total_profit > 0 and returns < 0):
            # If they don't match, use total_profit to calculate returns
            returns = (total_profit / initial_balance) * 100 if initial_balance > 0 else 0
        
        # Calculate max balance
        max_balance = self.max_balance
        if results.get('equity_curves'):
            try:
                # Try to get max balance from equity curves if available
                equity_values = [curve[-1][1] for pair, curve in results['equity_curves'].items() if curve]
                if equity_values:
                    max_balance = max(equity_values)
            except (KeyError, IndexError, TypeError) as e:
                self.logger.warning(f"Could not calculate max balance from equity curves: {e}")
                max_balance = self.max_balance
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'scalping_trades': scalping_trades,
            'swing_trades': swing_trades,
            'returns': returns,
            'max_balance': max_balance
        }
    
    def _format_telegram_summary(self, summary: Dict) -> str:
        """
        Format backtest summary for Telegram message.
        
        Args:
            summary: Summary of backtest results
            
        Returns:
            str: Formatted message
        """
        return (
            f"📊 Backtest Results Summary\n\n"
            f"Period: {self.period}\n"
            f"Pairs: {', '.join(self.trading_pairs)}\n\n"
            f"Total Trades: {summary['total_trades']}\n"
            f"Win Rate: {summary['win_rate']:.2%}\n"
            f"Total Profit: {summary['total_profit']:.2f} USDT\n"
            f"Return: {summary['returns']:.2f}%\n"
            f"Profit Factor: {summary['profit_factor']:.2f}\n\n"
            f"Strategy Usage:\n"
            f"  Scalping: {summary['scalping_trades']} trades\n"
            f"  Swing: {summary['swing_trades']} trades\n\n"
            f"Average Profit: {summary['avg_profit']:.2f} USDT\n"
            f"Average Win: {summary['avg_win']:.2f} USDT\n"
            f"Average Loss: {summary['avg_loss']:.2f} USDT\n"
            f"Max Balance: {summary['max_balance']:.2f} USDT\n"
        )
        
    def display_results(self, results: Dict):
        """
        Log the backtest results summary.
        
        Args:
            results (Dict): Dictionary containing backtest results
        """
        all_trades = results.get('trades', [])
        summary = results.get('summary', {})
        
        self.logger.info("=" * 50)
        self.logger.info("BACKTEST RESULTS SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Period: {self.period}")
        self.logger.info(f"Trading pairs: {self.trading_pairs}")
        
        # Display total trades
        total_trades = summary.get('total_trades', 0)
        self.logger.info(f"Total trades: {total_trades}")
        
        # Display win rate with proper formatting
        win_rate = summary.get('win_rate', 0.0)
        self.logger.info(f"Win rate: {win_rate:.2f}%")
        
        # Display total profit
        total_profit = summary.get('total_profit', 0.0)
        if np.isnan(total_profit):
            total_profit = 0.0
        self.logger.info(f"Total profit: {total_profit:.2f} USDT")
        
        # Display return
        returns = summary.get('returns', 0.0)
        if np.isnan(returns):
            returns = 0.0
        self.logger.info(f"Return: {returns:.2f}%")
        
        # Display profit factor
        profit_factor = summary.get('profit_factor', 0.0)
        if np.isnan(profit_factor) or np.isinf(profit_factor):
            self.logger.info("Profit factor: N/A (no losing trades)")
        else:
            self.logger.info(f"Profit factor: {profit_factor:.2f}")
            
        # Display max balance
        max_balance = summary.get('max_balance', 0.0)
        if np.isnan(max_balance):
            max_balance = 0.0
        self.logger.info(f"Max balance: {max_balance:.2f} USDT")
        
        self.logger.info("=" * 50)
        
        # Generate performance plots
        try:
            self._generate_equity_curve_plot(results)
            self._generate_trade_distribution_plot(all_trades)
            
            self.logger.info("Performance plots have been generated in the 'data' directory")
        except Exception as e:
            self.logger.error(f"Error generating performance plots: {str(e)}")
    
    def _generate_equity_curve_plot(self, results: Dict):
        """
        Generate and save equity curve plot.
        
        Args:
            results: Backtest results
        """
        plt.figure(figsize=(12, 6))
        
        # Plot equity curve for each pair
        for pair, equity_data in results['equity_curves'].items():
            times = [point[0] for point in equity_data]
            equity = [point[1] for point in equity_data]
            plt.plot(times, equity, label=pair)
        
        plt.axhline(y=self.initial_balance, color='r', linestyle='--', label='Initial Balance')
        
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Equity (USDT)')
        plt.legend()
        plt.grid(True)
        
        # Ensure the data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Save the plot
        plt.savefig('data/equity_curve.png')
        plt.close()
    
    def _generate_trade_distribution_plot(self, trades: List[Dict]):
        """
        Generate and save trade distribution plot.
        
        Args:
            trades: List of trades
        """
        profit_percentages = [trade['profit_pct'] for trade in trades]
        
        plt.figure(figsize=(12, 6))
        plt.hist(profit_percentages, bins=50, alpha=0.7, color='blue')
        plt.axvline(x=0, color='r', linestyle='--', label='Break Even')
        
        plt.title('Trade Profit/Loss Distribution')
        plt.xlabel('Profit/Loss (%)')
        plt.ylabel('Number of Trades')
        plt.legend()
        plt.grid(True)
        
        # Ensure the data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Save the plot
        plt.savefig('data/trade_distribution.png')
        plt.close()
