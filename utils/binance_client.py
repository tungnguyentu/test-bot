"""
Binance API client wrapper with additional error handling and utilities.
"""

import logging
import time
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException


class BinanceClient:
    """
    Wrapper for the Binance API client with enhanced functionality 
    and error handling for the trading bot.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = False):
        """
        Initialize the BinanceClient wrapper.
        
        Args:
            api_key: Binance API key (optional, will try to get from env vars if None)
            api_secret: Binance API secret (optional, will try to get from env vars if None)
            testnet: Whether to use the Binance testnet
        """
        self.logger = logging.getLogger('binance_bot')
        self.testnet = testnet
        
        # Get API credentials from environment if not provided
        api_key = api_key or os.environ.get('BINANCE_API_KEY')
        api_secret = api_secret or os.environ.get('BINANCE_API_SECRET')
        
        # For backtesting mode, we can use the Client without authentication
        if not api_key or not api_secret:
            self.logger.warning("No API credentials provided, using Client without auth (limited functionality)")
            self.client = Client()
        else:
            self.client = Client(api_key, api_secret, testnet=testnet)
        
        # Configure retry parameters
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Cache for exchange info to reduce API calls
        self.exchange_info = None
        self.exchange_info_timestamp = 0
        self.symbol_info_cache = {}
    
    def _handle_request(self, request_func, *args, **kwargs):
        """
        Execute a request with retry logic and error handling.
        
        Args:
            request_func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function call or None on failure
        """
        for attempt in range(self.max_retries):
            try:
                return request_func(*args, **kwargs)
            except BinanceAPIException as e:
                self.logger.error(f"BinanceAPIException: {e}")
                if e.code == -1021:  # Timestamp for this request is outside of the recvWindow
                    self.logger.info("Timestamp error, syncing time...")
                    self.client.synced = False  # Force time sync on next request
                    time.sleep(self.retry_delay)
                elif e.code == -1003:  # Too many requests
                    wait_time = 2 ** attempt
                    self.logger.info(f"Rate limit hit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif e.code == -2010 or e.code == -2011:  # Insufficient balance, order error
                    self.logger.error(f"Order error: {e}")
                    return None
                else:
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (attempt + 1)
                        self.logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"Failed after {self.max_retries} attempts: {e}")
                        return None
            except BinanceRequestException as e:
                self.logger.error(f"BinanceRequestException: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (attempt + 1)
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Failed after {self.max_retries} attempts: {e}")
                    return None
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                return None
        
        return None
    
    def get_exchange_info(self, force_refresh: bool = False) -> Dict:
        """
        Get Binance exchange info with caching.
        
        Args:
            force_refresh: Force refresh the cache
            
        Returns:
            Dict: Exchange information
        """
        current_time = time.time()
        cache_age = current_time - self.exchange_info_timestamp
        
        # Refresh if cache is older than 1 hour or forced
        if force_refresh or not self.exchange_info or cache_age > 3600:
            self.logger.info("Fetching exchange info...")
            self.exchange_info = self._handle_request(self.client.get_exchange_info)
            self.exchange_info_timestamp = current_time
            
            # Update symbol info cache
            if self.exchange_info:
                for symbol_info in self.exchange_info['symbols']:
                    self.symbol_info_cache[symbol_info['symbol']] = symbol_info
        
        return self.exchange_info
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get info for a specific symbol with caching.
        
        Args:
            symbol: Symbol name (e.g., 'BTCUSDT')
            
        Returns:
            Dict: Symbol information
        """
        # Try to get from cache first
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]
        
        # Ensure exchange info is loaded
        exchange_info = self.get_exchange_info()
        if not exchange_info:
            return None
        
        # Find symbol in exchange info
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                self.symbol_info_cache[symbol] = symbol_info
                return symbol_info
        
        self.logger.error(f"Symbol {symbol} not found in exchange info")
        return None
    
    def get_futures_account_balance(self) -> List[Dict]:
        """
        Get futures account balance.
        
        Returns:
            List[Dict]: List of balances for each asset
        """
        return self._handle_request(self.client.futures_account_balance)
    
    def get_futures_position_information(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get futures position information.
        
        Args:
            symbol: Symbol to get position for (optional)
            
        Returns:
            List[Dict]: Position information
        """
        if symbol:
            return self._handle_request(self.client.futures_position_information, symbol=symbol)
        else:
            return self._handle_request(self.client.futures_position_information)
    
    def get_historical_klines(self, symbol: str, interval: str, start_time: int, 
                             end_time: Optional[int] = None, limit: int = 1000) -> List:
        """
        Get historical klines (candlesticks) for a symbol.
        
        Args:
            symbol: Symbol name
            interval: Kline interval (e.g., '1m', '5m', '1h')
            start_time: Start time in milliseconds
            end_time: End time in milliseconds (optional)
            limit: Number of klines to get (max 1000)
            
        Returns:
            List: Kline data
        """
        return self._handle_request(
            self.client.get_historical_klines,
            symbol=symbol,
            interval=interval,
            start_str=start_time,
            end_str=end_time,
            limit=limit
        )
    
    def get_historical_klines_df(self, symbol: str, interval: str, 
                                start_time: int, end_time: Optional[int] = None, 
                                limit: int = 1000) -> pd.DataFrame:
        """
        Get historical klines as a DataFrame.
        
        Args:
            symbol: Symbol name
            interval: Kline interval
            start_time: Start time in milliseconds
            end_time: End time in milliseconds (optional)
            limit: Number of klines to get (max 1000)
            
        Returns:
            pd.DataFrame: Kline data as DataFrame
        """
        klines = self.get_historical_klines(symbol, interval, start_time, end_time, limit)
        
        if not klines:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                         'quote_asset_volume', 'taker_buy_base_asset_volume', 
                         'taker_buy_quote_asset_volume']
        
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column])
        
        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        return df
    
    def create_futures_order(self, symbol: str, side: str, order_type: str, 
                           quantity: float, price: Optional[float] = None,
                           stop_price: Optional[float] = None, close_position: bool = False,
                           reduce_only: bool = False) -> Dict:
        """
        Create a futures order.
        
        Args:
            symbol: Symbol name
            side: Order side (BUY or SELL)
            order_type: Order type (LIMIT, MARKET, STOP, etc.)
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            stop_price: Stop price (required for STOP/STOP_MARKET orders)
            close_position: Whether to close the position
            reduce_only: Whether the order is reduce-only
            
        Returns:
            Dict: Order information
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'reduceOnly': reduce_only
        }
        
        if price and order_type != 'MARKET':
            params['price'] = price
            
        if stop_price and order_type in ['STOP', 'STOP_MARKET', 'TAKE_PROFIT', 'TAKE_PROFIT_MARKET']:
            params['stopPrice'] = stop_price
            
        if close_position:
            params['closePosition'] = 'true'
            
        return self._handle_request(self.client.futures_create_order, **params)
    
    def cancel_futures_order(self, symbol: str, order_id: Optional[int] = None, 
                           orig_client_order_id: Optional[str] = None) -> Dict:
        """
        Cancel a futures order.
        
        Args:
            symbol: Symbol name
            order_id: Order ID (optional)
            orig_client_order_id: Original client order ID (optional)
            
        Returns:
            Dict: Cancellation information
        """
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = order_id
        elif orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
        else:
            self.logger.error("Either order_id or orig_client_order_id must be provided")
            return None
            
        return self._handle_request(self.client.futures_cancel_order, **params)
    
    def cancel_all_futures_orders(self, symbol: str) -> Dict:
        """
        Cancel all open futures orders for a symbol.
        
        Args:
            symbol: Symbol name
            
        Returns:
            Dict: Cancellation information
        """
        return self._handle_request(self.client.futures_cancel_all_open_orders, symbol=symbol)
    
    def get_futures_account_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """
        Get account trade history for a symbol.
        
        Args:
            symbol: Symbol name
            limit: Number of trades to get
            
        Returns:
            List[Dict]: Trade history
        """
        return self._handle_request(self.client.futures_account_trades, symbol=symbol, limit=limit)
    
    def change_leverage(self, symbol: str, leverage: int) -> Dict:
        """
        Change leverage for a symbol.
        
        Args:
            symbol: Symbol name
            leverage: Desired leverage (1-125)
            
        Returns:
            Dict: Leverage information
        """
        return self._handle_request(self.client.futures_change_leverage, symbol=symbol, leverage=leverage)
    
    def change_margin_type(self, symbol: str, margin_type: str) -> Dict:
        """
        Change margin type for a symbol.
        
        Args:
            symbol: Symbol name
            margin_type: 'ISOLATED' or 'CROSSED'
            
        Returns:
            Dict: Margin type information
        """
        try:
            return self._handle_request(self.client.futures_change_margin_type, 
                                       symbol=symbol, 
                                       marginType=margin_type)
        except BinanceAPIException as e:
            # Ignore error if margin type is already set
            if e.code == -4046:
                self.logger.info(f"Margin type for {symbol} already set to {margin_type}")
                return {"symbol": symbol, "marginType": margin_type}
            raise
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get current order book for a symbol.
        
        Args:
            symbol: Symbol name
            limit: Depth of the order book
            
        Returns:
            Dict: Order book data
        """
        return self._handle_request(self.client.get_order_book, symbol=symbol, limit=limit)
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        Get ticker information for a symbol.
        
        Args:
            symbol: Symbol name
            
        Returns:
            Dict: Ticker information
        """
        return self._handle_request(self.client.get_ticker, symbol=symbol)
