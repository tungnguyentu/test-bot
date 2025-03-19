"""
Trade Manager for tracking and managing active and historical trades.
"""

import logging
import time
import json
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime


class TradeManager:
    """
    Tracks and manages active trades and trade history.
    """
    
    def __init__(self, trades_dir: str = 'data/trades'):
        """
        Initialize the TradeManager.
        
        Args:
            trades_dir: Directory for storing trade data
        """
        self.logger = logging.getLogger('binance_bot')
        self.trades_dir = trades_dir
        
        # Create trades directory if it doesn't exist
        os.makedirs(self.trades_dir, exist_ok=True)
        
        # Active trades by symbol: symbol -> trade_info
        self.active_trades = {}
        
        # Historical trades
        self.trade_history = []
        
        # Load existing trade history if available
        self._load_trade_history()
    
    def _load_trade_history(self) -> None:
        """
        Load trade history from file.
        """
        history_file = os.path.join(self.trades_dir, 'trade_history.json')
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.trade_history = json.load(f)
                self.logger.info(f"Loaded {len(self.trade_history)} historical trades")
            except Exception as e:
                self.logger.error(f"Error loading trade history: {e}")
                self.trade_history = []
    
    def _save_trade_history(self) -> None:
        """
        Save trade history to file.
        """
        history_file = os.path.join(self.trades_dir, 'trade_history.json')
        
        try:
            with open(history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving trade history: {e}")
    
    def create_trade(self, symbol: str, side: str, entry_price: float, 
                   entry_time: Optional[int] = None, quantity: float = 0.0, 
                   stop_loss: float = 0.0, take_profit: float = 0.0,
                   strategy: str = "", trade_id: Optional[str] = None) -> Dict:
        """
        Create a new trade.
        
        Args:
            symbol: Trading pair symbol
            side: Trade direction (BUY/SELL)
            entry_price: Entry price
            entry_time: Entry timestamp (milliseconds)
            quantity: Trade quantity
            stop_loss: Stop loss price
            take_profit: Take profit price
            strategy: Strategy used for the trade
            trade_id: Custom trade ID (optional)
            
        Returns:
            Dict: Trade information
        """
        if entry_time is None:
            entry_time = int(time.time() * 1000)
            
        # Generate trade ID if not provided
        if trade_id is None:
            trade_id = f"{symbol}_{side}_{entry_time}"
            
        # Create trade object
        trade = {
            'id': trade_id,
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'entry_time_str': datetime.fromtimestamp(entry_time/1000).strftime('%Y-%m-%d %H:%M:%S'),
            'quantity': quantity,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strategy': strategy,
            'exit_price': 0.0,
            'exit_time': 0,
            'exit_time_str': '',
            'exit_reason': '',
            'profit_loss': 0.0,
            'profit_loss_percent': 0.0,
            'status': 'OPEN'
        }
        
        # Store in active trades
        self.active_trades[symbol] = trade
        
        self.logger.info(f"Created new trade: {trade_id} - {symbol} {side} at {entry_price}")
        
        return trade
    
    def update_trade(self, symbol: str, **kwargs) -> Optional[Dict]:
        """
        Update an existing trade.
        
        Args:
            symbol: Symbol of the trade to update
            **kwargs: Trade attributes to update
            
        Returns:
            Dict: Updated trade or None if not found
        """
        if symbol not in self.active_trades:
            self.logger.error(f"Cannot update trade for {symbol}: No active trade found")
            return None
            
        # Update trade attributes
        for key, value in kwargs.items():
            if key in self.active_trades[symbol]:
                self.active_trades[symbol][key] = value
                
        return self.active_trades[symbol]
    
    def close_trade(self, symbol: str, exit_price: float, exit_time: Optional[int] = None,
                  exit_reason: str = "") -> Optional[Dict]:
        """
        Close a trade.
        
        Args:
            symbol: Symbol of the trade to close
            exit_price: Exit price
            exit_time: Exit timestamp (milliseconds)
            exit_reason: Reason for closing the trade
            
        Returns:
            Dict: Closed trade or None if not found
        """
        if symbol not in self.active_trades:
            self.logger.error(f"Cannot close trade for {symbol}: No active trade found")
            return None
            
        # Get the active trade
        trade = self.active_trades[symbol]
        
        # Set exit information
        if exit_time is None:
            exit_time = int(time.time() * 1000)
            
        trade['exit_price'] = exit_price
        trade['exit_time'] = exit_time
        trade['exit_time_str'] = datetime.fromtimestamp(exit_time/1000).strftime('%Y-%m-%d %H:%M:%S')
        trade['exit_reason'] = exit_reason
        trade['status'] = 'CLOSED'
        
        # Calculate profit/loss
        if trade['side'] == 'BUY':
            profit_loss = (exit_price - trade['entry_price']) * trade['quantity']
            profit_loss_percent = ((exit_price / trade['entry_price']) - 1) * 100
        else:  # SELL
            profit_loss = (trade['entry_price'] - exit_price) * trade['quantity']
            profit_loss_percent = ((trade['entry_price'] / exit_price) - 1) * 100
            
        trade['profit_loss'] = profit_loss
        trade['profit_loss_percent'] = profit_loss_percent
        
        # Move to history
        self.trade_history.append(trade)
        
        # Save updated history
        self._save_trade_history()
        
        # Remove from active trades
        del self.active_trades[symbol]
        
        self.logger.info(
            f"Closed trade {trade['id']}: {profit_loss_percent:.2f}% P&L, reason: {exit_reason}"
        )
        
        return trade
    
    def get_active_trade(self, symbol: str) -> Optional[Dict]:
        """
        Get active trade for a symbol.
        
        Args:
            symbol: Symbol to get trade for
            
        Returns:
            Dict: Trade information or None if not found
        """
        return self.active_trades.get(symbol)
    
    def has_active_trade(self, symbol: str) -> bool:
        """
        Check if there's an active trade for a symbol.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            bool: True if there's an active trade, False otherwise
        """
        return symbol in self.active_trades
    
    def get_all_active_trades(self) -> Dict[str, Dict]:
        """
        Get all active trades.
        
        Returns:
            Dict[str, Dict]: All active trades
        """
        return self.active_trades
    
    def get_trade_history(self, symbol: Optional[str] = None, 
                        start_time: Optional[int] = None, 
                        end_time: Optional[int] = None) -> List[Dict]:
        """
        Get trade history, optionally filtered by symbol and time range.
        
        Args:
            symbol: Symbol to filter by (optional)
            start_time: Start time in milliseconds (optional)
            end_time: End time in milliseconds (optional)
            
        Returns:
            List[Dict]: Filtered trade history
        """
        # Filter by symbol if provided
        if symbol:
            filtered = [t for t in self.trade_history if t['symbol'] == symbol]
        else:
            filtered = self.trade_history.copy()
            
        # Filter by time range if provided
        if start_time:
            filtered = [t for t in filtered if t['entry_time'] >= start_time]
            
        if end_time:
            filtered = [t for t in filtered if t['entry_time'] <= end_time]
            
        return filtered
    
    def get_trade_statistics(self, symbol: Optional[str] = None, 
                           start_time: Optional[int] = None, 
                           end_time: Optional[int] = None) -> Dict:
        """
        Calculate trade statistics for a given set of trades.
        
        Args:
            symbol: Symbol to filter by (optional)
            start_time: Start time in milliseconds (optional)
            end_time: End time in milliseconds (optional)
            
        Returns:
            Dict: Trade statistics
        """
        # Get filtered trade history
        trades = self.get_trade_history(symbol, start_time, end_time)
        
        # Initialize statistics
        stats = {
            'total_trades': len(trades),
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'net_profit': 0.0,
            'avg_profit_percent': 0.0,
            'avg_loss_percent': 0.0,
            'max_profit_percent': 0.0,
            'max_loss_percent': 0.0,
            'avg_trade_duration': 0.0
        }
        
        if not trades:
            return stats
            
        # Calculate statistics
        winning_trades = [t for t in trades if t['profit_loss'] > 0]
        losing_trades = [t for t in trades if t['profit_loss'] <= 0]
        
        stats['winning_trades'] = len(winning_trades)
        stats['losing_trades'] = len(losing_trades)
        
        if stats['total_trades'] > 0:
            stats['win_rate'] = stats['winning_trades'] / stats['total_trades'] * 100
            
        # Calculate profit/loss metrics
        total_profit = sum(t['profit_loss'] for t in winning_trades)
        total_loss = sum(abs(t['profit_loss']) for t in losing_trades)
        
        stats['total_profit'] = total_profit
        stats['total_loss'] = total_loss
        stats['net_profit'] = total_profit - total_loss
        
        if total_loss > 0:
            stats['profit_factor'] = total_profit / total_loss
            
        # Calculate percentage metrics
        if winning_trades:
            stats['avg_profit_percent'] = sum(t['profit_loss_percent'] for t in winning_trades) / len(winning_trades)
            stats['max_profit_percent'] = max(t['profit_loss_percent'] for t in winning_trades)
            
        if losing_trades:
            stats['avg_loss_percent'] = sum(abs(t['profit_loss_percent']) for t in losing_trades) / len(losing_trades)
            stats['max_loss_percent'] = max(abs(t['profit_loss_percent']) for t in losing_trades)
            
        # Calculate average trade duration
        durations = [(t['exit_time'] - t['entry_time']) / (1000 * 60 * 60) for t in trades if t['exit_time'] > 0]
        if durations:
            stats['avg_trade_duration'] = sum(durations) / len(durations)  # in hours
            
        return stats
    
    def get_equity_curve(self, symbol: Optional[str] = None,
                        start_time: Optional[int] = None,
                        end_time: Optional[int] = None) -> pd.DataFrame:
        """
        Generate equity curve data for visualization.
        
        Args:
            symbol: Symbol to filter by (optional)
            start_time: Start time in milliseconds (optional)
            end_time: End time in milliseconds (optional)
            
        Returns:
            pd.DataFrame: Equity curve data
        """
        # Get filtered trade history
        trades = self.get_trade_history(symbol, start_time, end_time)
        
        if not trades:
            return pd.DataFrame()
            
        # Sort trades by entry time
        trades.sort(key=lambda t: t['entry_time'])
        
        # Create DataFrame with trade results
        data = []
        cumulative_pnl = 0.0
        
        for trade in trades:
            cumulative_pnl += trade['profit_loss']
            
            data.append({
                'timestamp': pd.to_datetime(trade['exit_time'], unit='ms'),
                'trade_pnl': trade['profit_loss'],
                'trade_pnl_percent': trade['profit_loss_percent'],
                'cumulative_pnl': cumulative_pnl,
                'symbol': trade['symbol'],
                'side': trade['side'],
                'strategy': trade['strategy']
            })
            
        df = pd.DataFrame(data)
        
        # Calculate drawdown
        if not df.empty:
            df['peak'] = df['cumulative_pnl'].cummax()
            df['drawdown'] = df['peak'] - df['cumulative_pnl']
            df['drawdown_percent'] = (df['drawdown'] / df['peak']) * 100
            df['drawdown_percent'] = df['drawdown_percent'].fillna(0)
            
        return df
