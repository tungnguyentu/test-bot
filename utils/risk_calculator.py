"""
Risk Calculator for position sizing and risk management.
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple


class RiskCalculator:
    """
    Calculates position sizes and manages risk for trading.
    """
    
    def __init__(self, account_balance: float, max_risk_per_trade: float = 0.01,
                max_account_risk: float = 0.05, max_position_size: float = 0.2):
        """
        Initialize the RiskCalculator.
        
        Args:
            account_balance: Account balance in quote currency
            max_risk_per_trade: Maximum risk per trade as a decimal (default: 1%)
            max_account_risk: Maximum account risk across all positions (default: 5%)
            max_position_size: Maximum position size as a percentage of account (default: 20%)
        """
        self.logger = logging.getLogger('binance_bot')
        self.account_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.max_account_risk = max_account_risk
        self.max_position_size = max_position_size
        
        # Currently at risk
        self.current_risk = 0.0
        self.open_positions = {}
    
    def update_account_balance(self, balance: float) -> None:
        """
        Update the account balance.
        
        Args:
            balance: New account balance
        """
        self.account_balance = balance
        
    def update_position(self, symbol: str, quantity: float, entry_price: float, 
                      stop_loss: float) -> None:
        """
        Update an open position in the risk calculator.
        
        Args:
            symbol: Trading pair symbol
            quantity: Position size
            entry_price: Entry price
            stop_loss: Stop loss price
        """
        # Remove any existing position first
        if symbol in self.open_positions:
            self.current_risk -= self.open_positions[symbol]['risk_amount']
        
        # Calculate the risk for this position
        risk_per_unit = abs(entry_price - stop_loss)
        risk_amount = risk_per_unit * quantity
        risk_percent = risk_amount / self.account_balance
        
        # Update the position
        self.open_positions[symbol] = {
            'quantity': quantity,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent
        }
        
        # Update total risk
        self.current_risk += risk_amount
        
        self.logger.info(
            f"Updated position for {symbol}: {quantity} units at {entry_price}, "
            f"risk: {risk_percent:.2%} of account"
        )
    
    def remove_position(self, symbol: str) -> None:
        """
        Remove a position from the risk calculator.
        
        Args:
            symbol: Trading pair symbol
        """
        if symbol in self.open_positions:
            self.current_risk -= self.open_positions[symbol]['risk_amount']
            del self.open_positions[symbol]
            
            self.logger.info(f"Removed position for {symbol}, current total risk: {self.current_risk_percent:.2%}")
    
    @property
    def current_risk_percent(self) -> float:
        """
        Calculate current risk as a percentage of account balance.
        
        Returns:
            float: Current risk percentage
        """
        return self.current_risk / self.account_balance if self.account_balance > 0 else 0
    
    def can_take_new_trade(self, risk_amount: float) -> bool:
        """
        Check if a new trade can be taken given the risk amount.
        
        Args:
            risk_amount: Risk amount for the new trade
            
        Returns:
            bool: True if the trade can be taken, False otherwise
        """
        # Check if adding this trade would exceed max account risk
        new_total_risk = self.current_risk + risk_amount
        new_risk_percent = new_total_risk / self.account_balance
        
        return new_risk_percent <= self.max_account_risk
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                               symbol: str, leverage: int = 1,
                               override_max_risk: bool = False) -> Tuple[float, float]:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            symbol: Trading pair symbol
            leverage: Leverage multiplier (default: 1)
            override_max_risk: Whether to override the maximum risk check
            
        Returns:
            Tuple[float, float]: Position size and risk amount
        """
        if entry_price == stop_loss:
            self.logger.error("Entry price cannot equal stop loss price")
            return 0.0, 0.0
            
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        # Calculate maximum risk amount for this trade
        max_risk_amount = self.account_balance * self.max_risk_per_trade
        
        # Calculate position size based on risk
        position_size = max_risk_amount / risk_per_unit
        
        # Apply leverage effect (increases position size)
        position_size = position_size * leverage
        
        # Calculate actual risk amount with leverage
        risk_amount = max_risk_amount
        
        # Check for maximum position size constraint
        max_position_value = self.account_balance * self.max_position_size * leverage
        max_position_size = max_position_value / entry_price
        
        if position_size > max_position_size:
            position_size = max_position_size
            # Recalculate risk amount with constrained position size
            risk_amount = (position_size / leverage) * risk_per_unit
        
        # Check if this new trade would exceed max account risk
        if not override_max_risk and not self.can_take_new_trade(risk_amount):
            self.logger.warning(
                f"Trade for {symbol} would exceed maximum account risk. "
                f"Current risk: {self.current_risk_percent:.2%}, "
                f"Max allowed: {self.max_account_risk:.2%}"
            )
            return 0.0, 0.0
        
        # Round position size to appropriate precision
        # This would normally depend on the exchange's minimum quantity requirements
        # For now, we'll just round to 4 decimal places
        position_size = round(position_size, 4)
        
        self.logger.info(
            f"Calculated position size for {symbol}: {position_size} units at {entry_price}, "
            f"risk: {risk_amount / self.account_balance:.2%} of account"
        )
        
        return position_size, risk_amount
    
    def adjust_for_volatility(self, symbol: str, position_size: float, 
                            volatility: float, volatility_scale: float = 1.0) -> float:
        """
        Adjust position size based on market volatility.
        
        Args:
            symbol: Trading pair symbol
            position_size: Initial position size
            volatility: Market volatility measure (e.g. ATR %)
            volatility_scale: Scaling factor for volatility adjustment
            
        Returns:
            float: Adjusted position size
        """
        # For high volatility, reduce position size
        # For low volatility, increase position size (up to a limit)
        
        # Base volatility level (consider this "normal" volatility)
        base_volatility = 0.02  # 2% daily change as a baseline
        
        # Calculate adjustment factor
        volatility_ratio = base_volatility / max(volatility, 0.001)  # Prevent division by zero
        adjustment_factor = np.clip(volatility_ratio * volatility_scale, 0.5, 2.0)
        
        # Adjust position size
        adjusted_size = position_size * adjustment_factor
        
        self.logger.info(
            f"Adjusted position size for {symbol} based on volatility: "
            f"{position_size} -> {adjusted_size} (volatility: {volatility:.2%}, "
            f"adjustment factor: {adjustment_factor:.2f})"
        )
        
        return adjusted_size
    
    def apply_kelly_criterion(self, symbol: str, position_size: float, 
                            win_rate: float, risk_reward_ratio: float) -> float:
        """
        Apply Kelly Criterion to optimize position sizing based on historical performance.
        
        Args:
            symbol: Trading pair symbol
            position_size: Initial position size
            win_rate: Historical win rate (0.0 to 1.0)
            risk_reward_ratio: Risk-to-reward ratio
            
        Returns:
            float: Kelly-adjusted position size
        """
        # Kelly formula: f* = (bp - q) / b
        # where:
        # f* = optimal fraction of the bankroll to bet
        # b = odds received on the bet (risk-reward ratio)
        # p = probability of winning
        # q = probability of losing (1 - p)
        
        if win_rate <= 0 or win_rate >= 1:
            return position_size  # Invalid win rate, no adjustment
            
        if risk_reward_ratio <= 0:
            return position_size  # Invalid risk-reward ratio, no adjustment
            
        p = win_rate
        q = 1 - p
        b = risk_reward_ratio
        
        kelly_percent = (b * p - q) / b
        
        # Cap Kelly at 50% of calculated position size as a safety measure
        # This is a common practice called "half-Kelly"
        kelly_percent = min(max(kelly_percent, 0), 0.5)
        
        # Apply Kelly percentage to original position size
        kelly_adjusted_size = position_size * kelly_percent
        
        self.logger.info(
            f"Applied Kelly Criterion for {symbol}: Kelly %: {kelly_percent:.2%}, "
            f"Adjusted size: {position_size} -> {kelly_adjusted_size}"
        )
        
        return kelly_adjusted_size
    
    def calculate_drawdown(self) -> float:
        """
        Calculate current drawdown based on open positions.
        
        Returns:
            float: Current drawdown as a percentage of account balance
        """
        unrealized_pnl = 0.0
        
        # In a real implementation, we would need current market prices
        # For now, we'll assume that is handled elsewhere and this is a placeholder
        
        drawdown_percent = abs(unrealized_pnl) / self.account_balance if self.account_balance > 0 else 0
        
        return drawdown_percent
    
    def get_risk_summary(self) -> Dict:
        """
        Get a summary of current risk exposure.
        
        Returns:
            Dict: Risk summary
        """
        return {
            'account_balance': self.account_balance,
            'current_risk': self.current_risk,
            'current_risk_percent': self.current_risk_percent,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_account_risk': self.max_account_risk,
            'open_positions': len(self.open_positions),
            'position_details': self.open_positions,
        }
