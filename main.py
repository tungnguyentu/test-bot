#!/usr/bin/env python3
"""
Binance Futures Trading Bot - Main Entry Point

This script initializes and runs the trading bot in either live trading,
paper trading, or backtesting mode.
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load local modules
from config import (
    TRADING_MODE, TRADING_PAIRS, BACKTEST_SETTINGS,
    LOG_LEVEL, DEBUG_MODE
)
from bot.trading_bot import TradingBot
from bot.backtester import Backtester
from bot.optimizer import StrategyOptimizer
from utils.telegram_client import TelegramClient
from utils.logger import setup_logger

# Load environment variables from .env file
load_dotenv()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Binance Futures Trading Bot')
    
    parser.add_argument('--mode', type=str, choices=['live', 'paper', 'backtest', 'optimize'],
                        default=TRADING_MODE, help='Trading mode')
    
    parser.add_argument('--pairs', type=str, nargs='+',
                        default=TRADING_PAIRS, help='Trading pairs')
    
    parser.add_argument('--period', type=str,
                        default=BACKTEST_SETTINGS['default_period'],
                        help='Backtesting period (e.g. 30d, 3m, 1y)')
    
    parser.add_argument('--debug', action='store_true',
                        default=DEBUG_MODE, help='Enable debug mode')
    
    return parser.parse_args()

def main():
    """Main entry point for the trading bot."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else getattr(logging, LOG_LEVEL)
    logger = setup_logger('binance_bot', log_level)
    
    logger.info(f"Starting Binance Futures Trading Bot in {args.mode.upper()} mode")
    logger.info(f"Trading pairs: {args.pairs}")
    
    # Initialize Telegram client for notifications
    telegram = TelegramClient()
    
    try:
        if args.mode in ('live', 'paper'):
            # Check for API keys
            if not os.getenv('BINANCE_API_KEY') or not os.getenv('BINANCE_API_SECRET'):
                logger.error("Binance API credentials not found in environment variables.")
                telegram.send_message("❌ Bot startup failed: API credentials missing")
                sys.exit(1)
            
            # Initialize and run trading bot
            bot = TradingBot(
                trading_pairs=args.pairs,
                mode=args.mode,
                telegram_client=telegram
            )
            
            telegram.send_message(f"🚀 Trading bot started in {args.mode.upper()} mode\n"
                                 f"Trading pairs: {', '.join(args.pairs)}")
            
            # Start the bot
            bot.run()
            
        elif args.mode == 'backtest':
            # Initialize and run backtester
            backtester = Backtester(
                trading_pairs=args.pairs,
                period=args.period,
                telegram_client=telegram
            )
            
            # Run backtest
            results = backtester.run()
            
            # Display results
            backtester.display_results(results)
            
        elif args.mode == 'optimize':
            # Initialize and run strategy optimizer
            optimizer = StrategyOptimizer(
                trading_pairs=args.pairs,
                period=args.period,
                telegram_client=telegram
            )
            
            # Run optimization
            optimal_params = optimizer.run()
            
            # Display optimal parameters
            optimizer.display_results(optimal_params)
            
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        telegram.send_message("⚠️ Bot stopped manually by user")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")
        telegram.send_message(f"❌ Bot crashed: {str(e)}")
    finally:
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    main()
