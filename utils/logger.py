"""
Logging setup for the trading bot.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logger(name: str = 'binance_bot', log_level: int = logging.INFO, log_dir: str = 'logs') -> logging.Logger:
    """
    Set up the logger for the trading bot.
    
    Args:
        name: Name of the logger
        log_level: Logging level (logging.DEBUG, logging.INFO, etc.)
        log_dir: Directory to store log files
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Get the root logger
    logger = logging.getLogger(name)
    
    # Set the log level
    logger.setLevel(log_level)
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create file handler for rotating log files
    file_handler = RotatingFileHandler(
        f"{log_dir}/{name}.log",
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    
    # Create formatters
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    
    # Set formatters
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        logging.Logger: Logger instance
    """
    if name:
        return logging.getLogger(f"binance_bot.{name}")
    else:
        return logging.getLogger("binance_bot")
