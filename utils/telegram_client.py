"""
Telegram client utility for sending notifications.
"""

import logging
import os
from typing import Dict, List, Optional
import telegram
from telegram import ParseMode  # Import ParseMode directly from telegram


class TelegramClient:
    """
    Class for sending notifications to a Telegram chat.
    """
    
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize the TelegramClient.
        
        Args:
            token: Telegram bot token (optional)
            chat_id: Telegram chat ID (optional)
        """
        self.logger = logging.getLogger('binance_bot')
        
        # Get token and chat_id from environment variables if not provided
        self.token = token or os.environ.get('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.environ.get('TELEGRAM_CHAT_ID')
        
        self.enabled = bool(self.token and self.chat_id)
        
        if self.enabled:
            try:
                self.bot = telegram.Bot(token=self.token)
                self.logger.info("Telegram bot initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Telegram bot: {e}")
                self.enabled = False
        else:
            self.logger.warning("Telegram notifications disabled. Missing token or chat_id.")
            
    def send_message(self, message: str) -> bool:
        """
        Send a message to Telegram.
        
        Args:
            message: Message text
            
        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False
            
        try:
            # Use the synchronous method directly
            self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_chat_action(self, action: str) -> bool:
        """
        Send a chat action to the Telegram chat.
        
        Args:
            action: Chat action type
            
        Returns:
            bool: Success status
        """
        if not self.enabled:
            return False
            
        try:
            # Use the synchronous method directly
            self.bot.send_chat_action(
                chat_id=self.chat_id,
                action=action
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to send Telegram chat action: {e}")
            return False
