"""
Performance metrics calculation for trading strategy evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def calculate_performance_metrics(trades: List[Dict], 
                                 initial_balance: Optional[float] = None,
                                 current_balance: Optional[float] = None,
                                 equity_curve: Optional[pd.DataFrame] = None) -> Dict:
    """
    Calculate trading performance metrics from a list of trades.
    
    Args:
        trades: List of trade dictionaries
        initial_balance: Optional initial account balance
        current_balance: Optional current account balance
        equity_curve: Optional equity curve DataFrame
        
    Returns:
        Dict: Performance metrics
    """
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'net_profit': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_percent': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'avg_trade': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_bars_in_trade': 0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'return_pct': 0.0,
        }
    
    # Calculate basic metrics
    total_trades = len(trades)
    
    # Check which profit field name is used in the trades dict
    profit_field = 'profit_loss' if 'profit_loss' in trades[0] else 'profit'
    
    winning_trades = [t for t in trades if t[profit_field] > 0]
    losing_trades = [t for t in trades if t[profit_field] <= 0]
    
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    # Profit metrics
    total_profit = sum(t[profit_field] for t in winning_trades) if winning_trades else 0
    total_loss = sum(abs(t[profit_field]) for t in losing_trades) if losing_trades else 0
    
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    net_profit = total_profit - total_loss
    
    # Return percentage if initial and current balance provided
    return_pct = 0.0
    if initial_balance is not None and current_balance is not None and initial_balance > 0:
        return_pct = ((current_balance - initial_balance) / initial_balance) * 100
    
    # Average metrics
    avg_trade = net_profit / total_trades if total_trades > 0 else 0
    avg_win = total_profit / len(winning_trades) if winning_trades else 0
    avg_loss = total_loss / len(losing_trades) if losing_trades else 0
    
    # Consecutive wins/losses
    if not trades:
        consecutive_wins = consecutive_losses = 0
    else:
        consecutive_wins = consecutive_losses = current_streak = 0
        prev_profit = None
        
        for trade in trades:
            profit = trade[profit_field]
            
            if prev_profit is None:
                current_streak = 1
            elif (profit > 0 and prev_profit > 0) or (profit <= 0 and prev_profit <= 0):
                current_streak += 1
            else:
                current_streak = 1
            
            if profit > 0:
                consecutive_wins = max(consecutive_wins, current_streak)
            else:
                consecutive_losses = max(consecutive_losses, current_streak)
                
            prev_profit = profit
    
    # Calculate drawdown if equity curve is provided
    max_drawdown = max_drawdown_percent = 0.0
    if equity_curve is not None and not equity_curve.empty:
        # If the equity curve already has drawdown metrics, use them
        if 'drawdown' in equity_curve.columns:
            max_drawdown = equity_curve['drawdown'].max()
        if 'drawdown_percent' in equity_curve.columns:
            max_drawdown_percent = equity_curve['drawdown_percent'].max()
        else:
            # Calculate drawdown
            equity = equity_curve['cumulative_pnl'].values if 'cumulative_pnl' in equity_curve.columns else None
            
            if equity is None and len(equity_curve) >= 2:
                # If it's a tuple of (timestamp, balance), convert to numpy array
                if isinstance(equity_curve[0], tuple) and len(equity_curve[0]) == 2:
                    equity = np.array([point[1] for point in equity_curve])
            
            if equity is not None:
                peak = np.maximum.accumulate(equity)
                drawdown = peak - equity
                max_drawdown = drawdown.max()
                
                # Calculate drawdown percentage
                max_drawdown_percent = ((peak - equity) / peak).max() * 100 if peak.max() > 0 else 0
    
    # Calculate Sharpe and Sortino ratios if equity curve is provided
    sharpe_ratio = sortino_ratio = 0.0
    if equity_curve is not None and isinstance(equity_curve, pd.DataFrame) and not equity_curve.empty and len(equity_curve) > 1:
        # Calculate daily returns
        if 'cumulative_pnl' in equity_curve.columns:
            equity_curve = equity_curve.sort_values('timestamp')
            equity_curve['daily_return'] = equity_curve['cumulative_pnl'].pct_change().fillna(0)
            
            # Sharpe Ratio (annualized)
            mean_return = equity_curve['daily_return'].mean()
            std_return = equity_curve['daily_return'].std()
            risk_free_rate = 0.0  # Assume zero risk-free rate for simplicity
            
            if std_return > 0:
                sharpe_ratio = ((mean_return - risk_free_rate) / std_return) * np.sqrt(252)  # Annualize
            
            # Sortino Ratio (annualized)
            downside_returns = equity_curve.loc[equity_curve['daily_return'] < 0, 'daily_return']
            downside_std = downside_returns.std()
            
            if downside_std > 0:
                sortino_ratio = ((mean_return - risk_free_rate) / downside_std) * np.sqrt(252)  # Annualize
    
    # Calculate average bars (candles) in trade
    avg_bars_in_trade = 0
    if trades and 'bars_in_trade' in trades[0]:
        total_bars = sum(t.get('bars_in_trade', 0) for t in trades)
        avg_bars_in_trade = total_bars / total_trades if total_trades > 0 else 0
    
    # Compile all metrics
    metrics = {
        'total_trades': total_trades,
        'win_rate': win_rate * 100,  # Convert to percentage
        'profit_factor': profit_factor,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'net_profit': net_profit,
        'return_pct': return_pct,
        'max_drawdown': max_drawdown,
        'max_drawdown_percent': max_drawdown_percent,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'avg_trade': avg_trade,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_bars_in_trade': avg_bars_in_trade,
        'consecutive_wins': consecutive_wins,
        'consecutive_losses': consecutive_losses,
    }
    
    return metrics


def calculate_trade_duration_stats(trades: List[Dict]) -> Dict:
    """
    Calculate statistics about trade durations.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Dict: Trade duration statistics
    """
    if not trades:
        return {
            'avg_duration_hours': 0,
            'min_duration_hours': 0,
            'max_duration_hours': 0,
            'median_duration_hours': 0
        }
    
    # Calculate durations in hours
    durations = []
    for trade in trades:
        if trade.get('entry_time') and trade.get('exit_time'):
            duration_hours = (trade['exit_time'] - trade['entry_time']) / (1000 * 60 * 60)
            durations.append(duration_hours)
    
    if not durations:
        return {
            'avg_duration_hours': 0,
            'min_duration_hours': 0,
            'max_duration_hours': 0,
            'median_duration_hours': 0
        }
    
    # Calculate statistics
    avg_duration = np.mean(durations)
    min_duration = np.min(durations)
    max_duration = np.max(durations)
    median_duration = np.median(durations)
    
    return {
        'avg_duration_hours': avg_duration,
        'min_duration_hours': min_duration,
        'max_duration_hours': max_duration,
        'median_duration_hours': median_duration
    }


def calculate_monthly_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly returns from an equity curve.
    
    Args:
        equity_curve: Equity curve DataFrame with 'timestamp' and 'cumulative_pnl' columns
        
    Returns:
        pd.DataFrame: Monthly returns
    """
    if equity_curve.empty or 'timestamp' not in equity_curve.columns or 'cumulative_pnl' not in equity_curve.columns:
        return pd.DataFrame()
    
    # Make sure the equity curve is sorted by timestamp
    equity_curve = equity_curve.sort_values('timestamp')
    
    # Extract month-end values
    equity_curve['year_month'] = equity_curve['timestamp'].dt.strftime('%Y-%m')
    monthly_equity = equity_curve.groupby('year_month').last().reset_index()
    
    # Calculate monthly returns
    monthly_equity['monthly_return'] = monthly_equity['cumulative_pnl'].pct_change()
    
    # Convert year_month to proper datetime for better plotting
    monthly_equity['date'] = pd.to_datetime(monthly_equity['year_month'] + '-01')
    
    return monthly_equity[['date', 'year_month', 'cumulative_pnl', 'monthly_return']]


def calculate_drawdowns(equity_curve: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Calculate and return the top N drawdowns.
    
    Args:
        equity_curve: Equity curve DataFrame with 'timestamp' and 'cumulative_pnl' columns
        top_n: Number of top drawdowns to return
        
    Returns:
        pd.DataFrame: Top drawdowns
    """
    if equity_curve.empty or 'timestamp' not in equity_curve.columns or 'cumulative_pnl' not in equity_curve.columns:
        return pd.DataFrame()
    
    # Make sure the equity curve is sorted by timestamp
    equity_curve = equity_curve.sort_values('timestamp')
    
    # Calculate running maximum
    equity_curve['peak'] = equity_curve['cumulative_pnl'].cummax()
    equity_curve['drawdown'] = equity_curve['peak'] - equity_curve['cumulative_pnl']
    equity_curve['drawdown_pct'] = (equity_curve['drawdown'] / equity_curve['peak']) * 100
    
    # Find drawdown periods
    in_drawdown = False
    drawdown_periods = []
    current_period = {}
    
    for idx, row in equity_curve.iterrows():
        if row['drawdown'] > 0 and not in_drawdown:
            # Start of a drawdown period
            in_drawdown = True
            current_period = {
                'start_date': row['timestamp'],
                'start_equity': row['peak'],
                'peak': row['peak']
            }
        elif row['drawdown'] == 0 and in_drawdown:
            # End of a drawdown period
            in_drawdown = False
            current_period['end_date'] = row['timestamp']
            current_period['end_equity'] = row['cumulative_pnl']
            current_period['drawdown'] = current_period['peak'] - current_period['end_equity']
            current_period['drawdown_pct'] = (current_period['drawdown'] / current_period['peak']) * 100
            current_period['duration_days'] = (current_period['end_date'] - current_period['start_date']).days
            
            drawdown_periods.append(current_period)
            current_period = {}
        elif row['drawdown'] > 0 and in_drawdown:
            # Update lowest point if needed
            if row['cumulative_pnl'] < current_period.get('lowest_equity', float('inf')):
                current_period['lowest_date'] = row['timestamp']
                current_period['lowest_equity'] = row['cumulative_pnl']
                current_period['max_drawdown'] = current_period['peak'] - row['cumulative_pnl']
                current_period['max_drawdown_pct'] = (current_period['max_drawdown'] / current_period['peak']) * 100
    
    # Handle the case where we're still in a drawdown at the end
    if in_drawdown:
        last_row = equity_curve.iloc[-1]
        current_period['end_date'] = last_row['timestamp']
        current_period['end_equity'] = last_row['cumulative_pnl']
        current_period['drawdown'] = current_period['peak'] - current_period['end_equity']
        current_period['drawdown_pct'] = (current_period['drawdown'] / current_period['peak']) * 100
        current_period['duration_days'] = (current_period['end_date'] - current_period['start_date']).days
        
        drawdown_periods.append(current_period)
    
    # Convert to DataFrame and sort by drawdown percentage
    if drawdown_periods:
        dd_df = pd.DataFrame(drawdown_periods)
        dd_df = dd_df.sort_values('max_drawdown_pct', ascending=False).head(top_n)
        return dd_df
    else:
        return pd.DataFrame()


def calculate_underwater_periods(equity_curve: pd.DataFrame) -> Dict:
    """
    Calculate underwater periods (when equity is below previous peak).
    
    Args:
        equity_curve: Equity curve DataFrame with 'timestamp' and 'cumulative_pnl' columns
        
    Returns:
        Dict: Underwater period statistics
    """
    if equity_curve.empty or 'timestamp' not in equity_curve.columns or 'cumulative_pnl' not in equity_curve.columns:
        return {
            'total_underwater_days': 0,
            'max_underwater_days': 0,
            'underwater_ratio': 0.0,
            'avg_underwater_days': 0.0
        }
    
    # Make sure the equity curve is sorted by timestamp
    equity_curve = equity_curve.sort_values('timestamp')
    
    # Calculate running maximum
    equity_curve['peak'] = equity_curve['cumulative_pnl'].cummax()
    equity_curve['underwater'] = equity_curve['cumulative_pnl'] < equity_curve['peak']
    
    # Count underwater days
    underwater_periods = []
    current_period_days = 0
    total_underwater_days = 0
    
    for idx, row in equity_curve.iterrows():
        if row['underwater']:
            current_period_days += 1
            total_underwater_days += 1
        elif current_period_days > 0:
            underwater_periods.append(current_period_days)
            current_period_days = 0
    
    # Add the last period if needed
    if current_period_days > 0:
        underwater_periods.append(current_period_days)
    
    # Calculate statistics
    max_underwater_days = max(underwater_periods) if underwater_periods else 0
    avg_underwater_days = np.mean(underwater_periods) if underwater_periods else 0
    underwater_ratio = total_underwater_days / len(equity_curve) if len(equity_curve) > 0 else 0
    
    return {
        'total_underwater_days': total_underwater_days,
        'max_underwater_days': max_underwater_days,
        'underwater_ratio': underwater_ratio,
        'avg_underwater_days': avg_underwater_days
    }
