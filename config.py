"""
Configuration settings for the Binance Futures Trading Bot.
"""

# Global settings
TRADING_MODE = "backtest"  # "backtest", "paper", "live"
DEBUG_MODE = True
LOG_LEVEL = "INFO"

# Trading parameters
TRADING_PAIRS = ["SOLUSDT"]
DEFAULT_TIMEFRAMES = {
    "scalping": "5m",    # 5-minute candles for scalping
    "swing": "4h"        # 4-hour candles for swing trading
}
QUOTE_ASSET = "USDT"
DEFAULT_LEVERAGE = 5     # 5x leverage by default

# Risk management
ACCOUNT_RISK_PER_TRADE = 0.02     # 2% risk per trade
MAX_DRAWDOWN_PERCENTAGE = 0.20    # Stop trading if 20% account drawdown reached
MAX_OPEN_POSITIONS = 3            # Maximum number of concurrent open positions
TRAILING_STOP_ACTIVATION = 0.02   # Activate trailing stop when 2% in profit
TRAILING_STOP_CALLBACK = 0.008    # 0.8% trailing stop callback

# Scalping strategy parameters
SCALPING_STRATEGY = {
    # Basic parameters
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "bb_period": 20,
    "bb_std_dev": 2,
    "ma_short_period": 9,
    "ma_long_period": 21,
    "min_profit_target": 0.005,   # 0.5% minimum profit target
    "max_stop_loss": 0.01,        # 1% maximum stop loss
    "take_profit_ratio": 1.5,     # TP:SL ratio (1.5:1)
    
    # Advanced parameters for improved win rate
    "adx_period": 14,
    "adx_threshold": 20,          # Minimum ADX to consider a trend strong
    "min_risk_reward": 1.5,       # Minimum risk-reward ratio for trades
    "slope_period": 10,           # Period for calculating price slope
    "require_volume_confirmation": True,  # Require volume to confirm signals
    "use_volatility_stops": True, # Use ATR-based stop loss
    "atr_multiplier": 2.0,        # Multiplier for ATR when calculating stops
    
    # Trailing stop settings
    "use_trailing_stop": True,
    "trailing_activation_pct": 0.8,  # Activate trailing at 0.8% profit
    "trailing_step_pct": 0.3         # Update trail every 0.3% move
}

# Swing trading strategy parameters
SWING_STRATEGY = {
    "ichimoku": {
        "conversion_line_period": 9,
        "base_line_period": 26,
        "lagging_span_period": 52,
        "displacement": 26
    },
    "macd": {
        "fast_length": 12,
        "slow_length": 26,
        "signal_smoothing": 9
    },
    "volume_ma_period": 20,
    "min_profit_target": 0.02,    # 2% minimum profit target
    "max_stop_loss": 0.03,        # 3% maximum stop loss
    "take_profit_ratio": 1.5      # TP:SL ratio (1.5:1)
}

# Strategy selection parameters
STRATEGY_SELECTION = {
    "volatility_threshold": 0.02,      # 2% volatility threshold
    "volume_increase_threshold": 1.5,  # 50% volume increase threshold
    "trend_strength_threshold": 25,    # ADX above 25 indicates trend
}

# Backtesting
BACKTEST_SETTINGS = {
    "default_period": "30d",    # 30-day period for backtesting
    "initial_balance": 10000,   # 10,000 USDT initial balance
    "trading_fee": 0.0004,      # 0.04% trading fee (maker or taker)
    "slippage": 0.0005          # 0.05% slippage estimate
}

# Telegram settings
TELEGRAM_SETTINGS = {
    "enabled": True,
    "notification_level": "ALL"  # "ALL", "TRADES_ONLY", "SUMMARY_ONLY"
}

# API settings
API_SETTINGS = {
    "retry_attempts": 3,
    "retry_delay": 5,  # seconds
    "rate_limit_margin": 0.8  # Stay under 80% of rate limits
}
