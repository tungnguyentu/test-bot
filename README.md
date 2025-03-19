# BinanceFuturesBot

A fully automated trading bot for Binance Futures that implements both scalping and swing trading strategies.

## Features

- **Dual Trading Strategies**:
  - Scalping: Using RSI, Bollinger Bands, and short-term Moving Averages
  - Swing Trading: Using Ichimoku Cloud, MACD, and volume analysis
  - Automatic strategy selection based on market conditions

- **Advanced Order Management**:
  - Support for limit, market, and stop orders
  - Dynamic position sizing based on risk parameters
  - Maximum drawdown protection

- **Comprehensive Technical Analysis**:
  - RSI, MACD, Bollinger Bands, Ichimoku Cloud
  - Candlestick pattern recognition
  - Volume analysis

- **Real-time Notifications**:
  - Telegram integration for trade alerts
  - Detailed reasoning for each trade decision
  - Performance reporting

- **Backtesting System**:
  - Validate strategies with historical data
  - Performance metrics: Sharpe Ratio, Profit Factor, Max Drawdown

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Create a `.env` file with your credentials:
```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

3. Run the bot:
```
python main.py
```

## Configuration

Edit `config.py` to adjust your trading parameters, including:
- Trading pairs
- Risk per trade
- Strategy parameters
- Backtesting options

## Usage

- Live trading mode: `python main.py --mode live`
- Backtesting mode: `python main.py --mode backtest --period 30d`
- Optimization mode: `python main.py --mode optimize`

## Win Rate Optimization Techniques

The trading bot implements several advanced techniques to improve win rate:

### 1. Enhanced Entry Filtering

- **Trend Strength Detection**: Uses ADX (Average Directional Index) to identify strong trends
- **Volume Confirmation**: Requires increased volume to validate entry signals
- **Multi-factor Confirmation**: Entry signals must satisfy multiple criteria
- **Price Action Analysis**: Detects bullish and bearish divergences between price and RSI

### 2. Dynamic Stop Loss Management

- **Volatility-Adjusted Stops**: Uses ATR (Average True Range) to set stops based on market volatility
- **Trailing Stop System**: Automatically adjusts stop loss to lock in profits as trade moves in favorable direction
- **Multi-tiered Take Profit**: Scales out of positions at different profit levels

### 3. Risk-Reward Filtering

- **Minimum R:R Ratio**: Only takes trades with favorable risk-reward ratios (default 1.5:1 or better)
- **Position Sizing**: Dynamically adjusts position size based on volatility and stop distance

### 4. Advanced Technical Analysis

- **Price Slope Analysis**: Measures momentum through linear regression of price
- **RSI Divergence Detection**: Identifies potential reversals through price-RSI divergence
- **Candlestick Pattern Recognition**: Detects significant reversal patterns

Configure these features in `config.py` under the strategy parameters sections.

## Disclaimer

This trading bot is for educational purposes only. Use at your own risk.
