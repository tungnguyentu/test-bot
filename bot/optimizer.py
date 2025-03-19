"""
Optimizer for finding optimal strategy parameters through grid search.
"""

import os
import logging
import itertools
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import json
import matplotlib.pyplot as plt

import config
from bot.backtester import Backtester
from utils.metrics import calculate_performance_metrics


class StrategyOptimizer:
    """
    Strategy optimizer that finds optimal parameters for trading strategies.
    """
    
    def __init__(self, trading_pairs: List[str], period: str = '30d',
                 telegram_client: Optional[Any] = None):
        """
        Initialize the strategy optimizer.
        
        Args:
            trading_pairs: List of trading pairs to optimize
            period: Period for optimization (e.g. '30d', '3m')
            telegram_client: Optional Telegram client for notifications
        """
        self.logger = logging.getLogger('binance_bot')
        self.trading_pairs = trading_pairs
        self.period = period
        self.telegram = telegram_client
        
        # Store original config values to restore later
        self.original_scalping_config = config.SCALPING_STRATEGY.copy()
        self.original_swing_config = config.SWING_STRATEGY.copy()
        
        # Define parameter ranges for optimization
        self.scalping_param_ranges = {
            'rsi_period': [7, 9, 14, 21],
            'rsi_overbought': [65, 70, 75, 80],
            'rsi_oversold': [20, 25, 30, 35],
            'bb_period': [10, 15, 20, 25],
            'bb_std_dev': [1.5, 2.0, 2.5],
            'ma_short_period': [5, 7, 9, 12],
            'ma_long_period': [15, 21, 25, 30],
            'take_profit_ratio': [1.0, 1.5, 2.0, 2.5]
        }
        
        self.swing_param_ranges = {
            'ichimoku': {
                'conversion_line_period': [7, 9, 11],
                'base_line_period': [22, 26, 30],
                'lagging_span_period': [44, 52, 60],
                'displacement': [22, 26, 30]
            },
            'macd': {
                'fast_length': [8, 12, 16],
                'slow_length': [21, 26, 30],
                'signal_smoothing': [7, 9, 11]
            },
            'volume_ma_period': [15, 20, 25],
            'take_profit_ratio': [1.0, 1.5, 2.0, 2.5]
        }
        
        # Results storage
        self.optimization_results = []
        
    def run(self) -> Dict:
        """
        Run the optimization process.
        
        Returns:
            Dict: Optimal parameters and corresponding performance metrics
        """
        if self.telegram:
            self.telegram.send_message(
                f"🔍 Starting strategy optimization for {', '.join(self.trading_pairs)} over {self.period}"
            )
        
        self.logger.info(f"Starting parameter optimization for {self.trading_pairs}")
        
        start_time = time.time()
        
        # Optimize scalping strategy
        self.logger.info("Optimizing Scalping Strategy parameters...")
        optimal_scalping = self._optimize_scalping_strategy()
        
        # Optimize swing strategy
        self.logger.info("Optimizing Swing Strategy parameters...")
        optimal_swing = self._optimize_swing_strategy()
        
        end_time = time.time()
        
        self.logger.info(f"Optimization completed in {end_time - start_time:.2f} seconds")
        
        # Save optimization results
        self._save_optimization_results()
        
        # Restore original config values
        config.SCALPING_STRATEGY = self.original_scalping_config
        config.SWING_STRATEGY = self.original_swing_config
        
        # Format results
        results = {
            'scalping': optimal_scalping,
            'swing': optimal_swing,
            'runtime_seconds': end_time - start_time,
            'trading_pairs': self.trading_pairs,
            'period': self.period
        }
        
        if self.telegram:
            # Send results summary
            self._send_optimization_summary(results)
        
        return results
    
    def _optimize_scalping_strategy(self) -> Dict:
        """
        Optimize parameters for the scalping strategy.
        
        Returns:
            Dict: Optimal parameters and corresponding performance
        """
        # Generate parameter combinations (use a subset to keep runtime reasonable)
        param_keys = list(self.scalping_param_ranges.keys())
        param_values = [self.scalping_param_ranges[key] for key in param_keys]
        
        # Limit the combinations to a reasonable number
        max_combinations = 50  # You can adjust this based on available time
        
        # Calculate total combinations
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
            
        # If too many combinations, select a subset
        if total_combinations > max_combinations:
            self.logger.info(f"Too many parameter combinations ({total_combinations}), "
                            f"limiting to {max_combinations} combinations")
            
            # Prioritize the most important parameters
            priority_params = ['rsi_period', 'bb_period', 'ma_short_period', 'ma_long_period']
            
            # Create a subset of combinations focusing on priority parameters
            param_list = []
            
            # Add variations of priority parameters
            for priority_param in priority_params:
                idx = param_keys.index(priority_param)
                for val in self.scalping_param_ranges[priority_param]:
                    # Create a baseline combination
                    combination = []
                    for i, key in enumerate(param_keys):
                        if i == idx:
                            combination.append(val)
                        else:
                            # Use middle value for non-priority params
                            values = self.scalping_param_ranges[key]
                            combination.append(values[len(values) // 2])
                    param_list.append(combination)
            
            # Add some random combinations if needed
            while len(param_list) < max_combinations:
                combination = []
                for values in param_values:
                    combination.append(np.random.choice(values))
                param_list.append(combination)
                
            # Remove duplicates
            param_list = [list(x) for x in set(tuple(x) for x in param_list)]
            
            # Limit to max_combinations
            param_list = param_list[:max_combinations]
        else:
            # Generate all combinations
            param_list = list(itertools.product(*param_values))
        
        self.logger.info(f"Testing {len(param_list)} parameter combinations for scalping strategy")
        
        # Track best performance
        best_params = None
        best_performance = {
            'profit_factor': 0,
            'total_profit': 0
        }
        
        # Test each parameter combination
        for i, param_combination in enumerate(param_list):
            # Update config with this parameter set
            param_dict = {param_keys[i]: param_combination[i] for i in range(len(param_keys))}
            
            # Apply parameters to config
            for key, value in param_dict.items():
                config.SCALPING_STRATEGY[key] = value
            
            # Reset profit target and stop loss based on take_profit_ratio
            tp_ratio = param_dict['take_profit_ratio']
            config.SCALPING_STRATEGY['min_profit_target'] = 0.005 * tp_ratio
            config.SCALPING_STRATEGY['max_stop_loss'] = 0.005 * tp_ratio / param_dict['take_profit_ratio']
            
            # Run backtest with these parameters
            backtester = Backtester(
                trading_pairs=self.trading_pairs,
                period=self.period
            )
            
            results = backtester.run()
            
            # Extract performance metrics
            summary = results['summary']
            
            # Store result
            optimization_result = {
                'params': param_dict,
                'performance': summary
            }
            self.optimization_results.append(optimization_result)
            
            # Log progress periodically
            if (i + 1) % 5 == 0 or (i + 1) == len(param_list):
                self.logger.info(f"Tested {i + 1}/{len(param_list)} scalping parameter combinations")
            
            # Check if this is the best so far
            if summary['profit_factor'] > best_performance['profit_factor'] or \
               (summary['profit_factor'] == best_performance['profit_factor'] and 
                summary['total_profit'] > best_performance['total_profit']):
                best_performance = summary
                best_params = param_dict
        
        self.logger.info(f"Best scalping parameters found: {best_params}")
        self.logger.info(f"Best performance: Profit Factor = {best_performance['profit_factor']:.2f}, "
                         f"Total Profit = {best_performance['total_profit']:.2f}")
        
        return {
            'parameters': best_params,
            'performance': best_performance
        }
    
    def _optimize_swing_strategy(self) -> Dict:
        """
        Optimize parameters for the swing strategy.
        
        Returns:
            Dict: Optimal parameters and corresponding performance
        """
        # For swing strategy, we'll handle the nested parameters differently
        
        # Ichimoku parameters
        ichimoku_keys = list(self.swing_param_ranges['ichimoku'].keys())
        ichimoku_values = [self.swing_param_ranges['ichimoku'][key] for key in ichimoku_keys]
        ichimoku_combinations = list(itertools.product(*ichimoku_values))
        
        # MACD parameters
        macd_keys = list(self.swing_param_ranges['macd'].keys())
        macd_values = [self.swing_param_ranges['macd'][key] for key in macd_keys]
        macd_combinations = list(itertools.product(*macd_values))
        
        # Other parameters
        volume_periods = self.swing_param_ranges['volume_ma_period']
        tp_ratios = self.swing_param_ranges['take_profit_ratio']
        
        # Create parameter combinations
        param_list = []
        for ichimoku_combo in ichimoku_combinations:
            for macd_combo in macd_combinations:
                for volume_period in volume_periods:
                    for tp_ratio in tp_ratios:
                        # Create parameter dictionary
                        param_dict = {
                            'ichimoku': {
                                ichimoku_keys[i]: ichimoku_combo[i] 
                                for i in range(len(ichimoku_keys))
                            },
                            'macd': {
                                macd_keys[i]: macd_combo[i] 
                                for i in range(len(macd_keys))
                            },
                            'volume_ma_period': volume_period,
                            'take_profit_ratio': tp_ratio
                        }
                        param_list.append(param_dict)
        
        # Limit the combinations to a reasonable number
        max_combinations = 30  # Adjust based on available time
        
        if len(param_list) > max_combinations:
            self.logger.info(f"Too many parameter combinations ({len(param_list)}), "
                            f"limiting to {max_combinations} combinations")
            
            # Take a random subset
            param_list = [param_list[i] for i in 
                          np.random.choice(len(param_list), max_combinations, replace=False)]
        
        self.logger.info(f"Testing {len(param_list)} parameter combinations for swing strategy")
        
        # Track best performance
        best_params = None
        best_performance = {
            'profit_factor': 0,
            'total_profit': 0
        }
        
        # Test each parameter combination
        for i, param_dict in enumerate(param_list):
            # Apply parameters to config
            config.SWING_STRATEGY['ichimoku'] = param_dict['ichimoku']
            config.SWING_STRATEGY['macd'] = param_dict['macd']
            config.SWING_STRATEGY['volume_ma_period'] = param_dict['volume_ma_period']
            config.SWING_STRATEGY['take_profit_ratio'] = param_dict['take_profit_ratio']
            
            # Reset profit target and stop loss based on take_profit_ratio
            tp_ratio = param_dict['take_profit_ratio']
            config.SWING_STRATEGY['min_profit_target'] = 0.02 * tp_ratio
            config.SWING_STRATEGY['max_stop_loss'] = 0.02 * tp_ratio / param_dict['take_profit_ratio']
            
            # Run backtest with these parameters
            backtester = Backtester(
                trading_pairs=self.trading_pairs,
                period=self.period
            )
            
            results = backtester.run()
            
            # Extract performance metrics
            summary = results['summary']
            
            # Store result
            optimization_result = {
                'params': param_dict,
                'performance': summary
            }
            self.optimization_results.append(optimization_result)
            
            # Log progress periodically
            if (i + 1) % 5 == 0 or (i + 1) == len(param_list):
                self.logger.info(f"Tested {i + 1}/{len(param_list)} swing parameter combinations")
            
            # Check if this is the best so far
            if summary['profit_factor'] > best_performance['profit_factor'] or \
               (summary['profit_factor'] == best_performance['profit_factor'] and 
                summary['total_profit'] > best_performance['total_profit']):
                best_performance = summary
                best_params = param_dict
        
        self.logger.info(f"Best swing parameters found: {best_params}")
        self.logger.info(f"Best performance: Profit Factor = {best_performance['profit_factor']:.2f}, "
                         f"Total Profit = {best_performance['total_profit']:.2f}")
        
        return {
            'parameters': best_params,
            'performance': best_performance
        }
    
    def _save_optimization_results(self):
        """Save optimization results to a file."""
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Format results as a DataFrqme
        results_list = []
        for result in self.optimization_results:
            # Flatten parameters to make them easier to store
            flat_params = self._flatten_params(result['params'])
            
            # Combine with performance metrics
            result_entry = {**flat_params, **result['performance']}
            results_list.append(result_entry)
        
        # Convert to DataFrame and save to CSV
        if results_list:
            df = pd.DataFrame(results_list)
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/optimization_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
            
            self.logger.info(f"Optimization results saved to {filename}")
    
    def _flatten_params(self, params: Dict) -> Dict:
        """
        Flatten nested parameters into a single level dictionary.
        
        Args:
            params: Nested parameter dictionary
            
        Returns:
            Dict: Flattened parameter dictionary
        """
        flat_params = {}
        
        for key, value in params.items():
            if isinstance(value, dict):
                # For nested dictionaries, prefix the keys with the parent key
                for sub_key, sub_value in value.items():
                    flat_params[f"{key}_{sub_key}"] = sub_value
            else:
                flat_params[key] = value
                
        return flat_params
    
    def _send_optimization_summary(self, results: Dict):
        """
        Send optimization summary via Telegram.
        
        Args:
            results: Optimization results
        """
        # Format scalping parameters
        scalping_params = results['scalping']['parameters']
        scalping_perf = results['scalping']['performance']
        
        # Format swing parameters
        swing_params = results['swing']['parameters']
        swing_perf = results['swing']['performance']
        
        # Create message
        message = (
            f"🔍 Strategy Optimization Results\n\n"
            f"Period: {self.period}\n"
            f"Pairs: {', '.join(self.trading_pairs)}\n"
            f"Runtime: {results['runtime_seconds']:.1f} seconds\n\n"
            
            f"📈 Scalping Strategy Optimal Parameters:\n"
            f"RSI Period: {scalping_params['rsi_period']}\n"
            f"RSI Levels: {scalping_params['rsi_oversold']}/{scalping_params['rsi_overbought']}\n"
            f"BB Period: {scalping_params['bb_period']}, StdDev: {scalping_params['bb_std_dev']}\n"
            f"MA Periods: {scalping_params['ma_short_period']}/{scalping_params['ma_long_period']}\n"
            f"TP Ratio: {scalping_params['take_profit_ratio']}\n"
            f"Performance: {scalping_perf['win_rate']:.1%} Win Rate, "
            f"{scalping_perf['profit_factor']:.2f} Profit Factor\n\n"
            
            f"📉 Swing Strategy Optimal Parameters:\n"
            f"Ichimoku: Conv={swing_params['ichimoku']['conversion_line_period']}, "
            f"Base={swing_params['ichimoku']['base_line_period']}\n"
            f"MACD: Fast={swing_params['macd']['fast_length']}, "
            f"Slow={swing_params['macd']['slow_length']}, "
            f"Signal={swing_params['macd']['signal_smoothing']}\n"
            f"Volume MA: {swing_params['volume_ma_period']}\n"
            f"TP Ratio: {swing_params['take_profit_ratio']}\n"
            f"Performance: {swing_perf['win_rate']:.1%} Win Rate, "
            f"{swing_perf['profit_factor']:.2f} Profit Factor\n\n"
            
            f"The optimized parameters have been saved to the data directory."
        )
        
        self.telegram.send_message(message)
        
    def display_results(self, results: Dict):
        """
        Display optimization results.
        
        Args:
            results: Optimization results
        """
        # Log results
        self.logger.info("=" * 50)
        self.logger.info("OPTIMIZATION RESULTS")
        self.logger.info("=" * 50)
        
        # Display scalping results
        self.logger.info("SCALPING STRATEGY OPTIMAL PARAMETERS:")
        for key, value in results['scalping']['parameters'].items():
            self.logger.info(f"  {key}: {value}")
        
        self.logger.info("Performance:")
        for key, value in results['scalping']['performance'].items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        # Display swing results
        self.logger.info("\nSWING STRATEGY OPTIMAL PARAMETERS:")
        
        # Ichimoku parameters
        self.logger.info("  Ichimoku:")
        for key, value in results['swing']['parameters']['ichimoku'].items():
            self.logger.info(f"    {key}: {value}")
        
        # MACD parameters
        self.logger.info("  MACD:")
        for key, value in results['swing']['parameters']['macd'].items():
            self.logger.info(f"    {key}: {value}")
        
        # Other parameters
        self.logger.info(f"  volume_ma_period: {results['swing']['parameters']['volume_ma_period']}")
        self.logger.info(f"  take_profit_ratio: {results['swing']['parameters']['take_profit_ratio']}")
        
        self.logger.info("Performance:")
        for key, value in results['swing']['performance'].items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        self.logger.info("=" * 50)
        
        # Generate optimization visualization
        try:
            self._generate_optimization_plots()
            self.logger.info("Optimization plots have been generated in the 'data' directory")
        except Exception as e:
            self.logger.error(f"Error generating optimization plots: {str(e)}")
    
    def _generate_optimization_plots(self):
        """Generate plots to visualize optimization results."""
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Prepare data
        scalping_results = [r for r in self.optimization_results 
                           if 'rsi_period' in r['params']]
        swing_results = [r for r in self.optimization_results 
                        if 'ichimoku' in r['params']]
        
        # Plot scalping parameters vs profit factor
        if scalping_results:
            # Extract data
            rsi_periods = [r['params']['rsi_period'] for r in scalping_results]
            bb_periods = [r['params']['bb_period'] for r in scalping_results]
            profit_factors = [r['performance']['profit_factor'] for r in scalping_results]
            total_profits = [r['performance']['total_profit'] for r in scalping_results]
            
            # Plot RSI period vs profit factor
            plt.figure(figsize=(10, 6))
            plt.scatter(rsi_periods, profit_factors, c=total_profits, cmap='viridis', alpha=0.7)
            plt.colorbar(label='Total Profit')
            plt.title('RSI Period vs Profit Factor (Scalping Strategy)')
            plt.xlabel('RSI Period')
            plt.ylabel('Profit Factor')
            plt.grid(True, alpha=0.3)
            plt.savefig('data/scalping_rsi_optimization.png')
            plt.close()
            
            # Plot BB period vs profit factor
            plt.figure(figsize=(10, 6))
            plt.scatter(bb_periods, profit_factors, c=total_profits, cmap='viridis', alpha=0.7)
            plt.colorbar(label='Total Profit')
            plt.title('Bollinger Band Period vs Profit Factor (Scalping Strategy)')
            plt.xlabel('BB Period')
            plt.ylabel('Profit Factor')
            plt.grid(True, alpha=0.3)
            plt.savefig('data/scalping_bb_optimization.png')
            plt.close()
        
        # Plot swing parameters vs profit factor
        if swing_results:
            # Extract data (using flattened parameters)
            flat_swing_results = []
            for r in swing_results:
                flat_params = self._flatten_params(r['params'])
                flat_swing_results.append({
                    'params': flat_params,
                    'performance': r['performance']
                })
            
            ichimoku_conversion = [r['params']['ichimoku_conversion_line_period'] 
                                 for r in flat_swing_results]
            macd_fast = [r['params']['macd_fast_length'] for r in flat_swing_results]
            profit_factors = [r['performance']['profit_factor'] for r in flat_swing_results]
            total_profits = [r['performance']['total_profit'] for r in flat_swing_results]
            
            # Plot Ichimoku conversion period vs profit factor
            plt.figure(figsize=(10, 6))
            plt.scatter(ichimoku_conversion, profit_factors, c=total_profits, cmap='viridis', alpha=0.7)
            plt.colorbar(label='Total Profit')
            plt.title('Ichimoku Conversion Period vs Profit Factor (Swing Strategy)')
            plt.xlabel('Ichimoku Conversion Period')
            plt.ylabel('Profit Factor')
            plt.grid(True, alpha=0.3)
            plt.savefig('data/swing_ichimoku_optimization.png')
            plt.close()
            
            # Plot MACD fast period vs profit factor
            plt.figure(figsize=(10, 6))
            plt.scatter(macd_fast, profit_factors, c=total_profits, cmap='viridis', alpha=0.7)
            plt.colorbar(label='Total Profit')
            plt.title('MACD Fast Length vs Profit Factor (Swing Strategy)')
            plt.xlabel('MACD Fast Length')
            plt.ylabel('Profit Factor')
            plt.grid(True, alpha=0.3)
            plt.savefig('data/swing_macd_optimization.png')
            plt.close()
