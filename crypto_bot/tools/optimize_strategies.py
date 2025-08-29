#!/usr/bin/env python3
"""
Strategy Optimization Script for Maximum Profit in Shortest Time

This script optimizes all trading strategies to maximize profit potential
while minimizing time to profit. It uses advanced optimization techniques
including genetic algorithms and machine learning.
"""

import asyncio
import json
import logging
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
import warnings

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import differential_evolution, minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import optuna

# Add the parent directory to the path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Now import from crypto_bot
try:
    from crypto_bot.utils.logger import LOG_DIR, setup_logger
except ImportError:
    # Fallback if crypto_bot is not available
    LOG_DIR = Path("crypto_bot/logs")
    def setup_logger(name, log_file):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

try:
    from crypto_bot.backtest.enhanced_backtester import EnhancedBacktestConfig
except ImportError:
    EnhancedBacktestConfig = None

try:
    from crypto_bot.strategy_router import strategy_for
except ImportError:
    strategy_for = None

warnings.filterwarnings('ignore')

logger = setup_logger(__name__, LOG_DIR / "optimize_strategies.log")


@dataclass
class OptimizationTarget:
    """Target metrics for optimization."""
    
    # Primary targets
    max_profit_pct: float = 0.50  # 50% profit target
    min_time_to_profit_hours: float = 2.0  # 2 hours max
    
    # Risk targets
    max_drawdown: float = 0.25  # 25% max drawdown
    max_volatility: float = 0.40  # 40% max volatility
    
    # Performance targets
    min_win_rate: float = 0.35  # 35% minimum win rate
    min_sharpe_ratio: float = 0.8  # 0.8 minimum Sharpe ratio
    min_profit_factor: float = 1.2  # 1.2 minimum profit factor
    
    # Speed targets
    max_signal_delay_seconds: float = 30.0  # 30 seconds max delay
    min_trades_per_hour: float = 5.0  # 5 trades per hour minimum


@dataclass
class StrategyOptimizationConfig:
    """Configuration for strategy optimization."""
    
    # Optimization parameters
    optimization_method: str = "genetic"  # genetic, bayesian, grid
    max_iterations: int = 1000
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # Backtesting parameters
    backtest_days: int = 30
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    top_pairs: int = 20
    
    # Parallel processing
    use_multiprocessing: bool = True
    max_workers: int = field(default_factory=lambda: min(multiprocessing.cpu_count(), 8))
    
    # Output
    results_dir: str = "crypto_bot/logs/optimization_results"
    save_optimized_config: bool = True


class StrategyOptimizer:
    """Optimizes trading strategies for maximum profit in shortest time."""
    
    def __init__(self, config: StrategyOptimizationConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load current configuration
        self.current_config = self._load_current_config()
        
        # Strategy parameter ranges
        self.parameter_ranges = self._define_parameter_ranges()
        
        # Optimization history
        self.optimization_history: List[Dict[str, Any]] = []
        
    def _load_current_config(self) -> Dict[str, Any]:
        """Load current configuration from config.yaml."""
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _define_parameter_ranges(self) -> Dict[str, Dict[str, List[float]]]:
        """Define parameter ranges for each strategy."""
        return {
            "ultra_scalp_bot": {
                "stop_loss_pct": [0.003, 0.005, 0.008, 0.01],
                "take_profit_pct": [0.015, 0.02, 0.025, 0.03],
                "min_score": [0.03, 0.04, 0.05, 0.06],
                "atr_window": [3, 5, 8, 10],
                "volume_mult": [1.2, 1.5, 2.0, 2.5]
            },
            "momentum_exploiter": {
                "stop_loss_pct": [0.005, 0.008, 0.01, 0.015],
                "take_profit_pct": [0.025, 0.035, 0.045, 0.06],
                "threshold": [0.008, 0.01, 0.015, 0.02],
                "momentum_window": [5, 8, 10, 12],
                "volume_zscore_threshold": [1.2, 1.5, 1.8, 2.0]
            },
            "volatility_harvester": {
                "stop_loss_pct": [0.008, 0.01, 0.015, 0.02],
                "take_profit_pct": [0.03, 0.04, 0.05, 0.07],
                "atr_threshold": [0.001, 0.002, 0.003, 0.004],
                "volume_spike": [1.5, 2.0, 2.5, 3.0],
                "atr_multiplier": [1.2, 1.5, 1.8, 2.0]
            },
            "bounce_scalper": {
                "stop_loss_atr_mult": [0.6, 0.8, 1.0, 1.2],
                "take_profit_atr_mult": [1.2, 1.5, 2.0, 2.5],
                "min_score": [0.06, 0.08, 0.1, 0.12],
                "rsi_window": [8, 10, 12, 14],
                "volume_multiple": [1.5, 2.0, 2.5, 3.0]
            },
            "sniper_bot": {
                "stop_loss_pct": [0.005, 0.008, 0.01, 0.015],
                "take_profit_pct": [0.02, 0.03, 0.04, 0.06],
                "breakout_pct": [0.01, 0.015, 0.02, 0.025],
                "volume_multiple": [1.2, 1.5, 2.0, 2.5],
                "atr_window": [5, 8, 10, 12]
            },
            "micro_scalp_bot": {
                "stop_loss_pct": [0.005, 0.008, 0.01, 0.015],
                "take_profit_pct": [0.015, 0.02, 0.025, 0.035],
                "min_score": [0.04, 0.06, 0.08, 0.1],
                "ema_fast": [1, 2, 3, 4],
                "ema_slow": [2, 3, 4, 5]
            }
        }
    
    def _evaluate_strategy_performance(
        self, 
        strategy_name: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate strategy performance with given parameters."""
        try:
            # This would integrate with your actual backtesting system
            # For now, we'll use a simplified evaluation
            performance = self._simulate_strategy_performance(strategy_name, parameters)
            return performance
        except Exception as e:
            logger.error(f"Error evaluating {strategy_name}: {e}")
            return {
                "profit_pct": 0.0,
                "time_to_profit_hours": float('inf'),
                "drawdown": 1.0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "signal_delay_seconds": float('inf'),
                "trades_per_hour": 0.0
            }
    
    def _simulate_strategy_performance(
        self, 
        strategy_name: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Simulate strategy performance based on parameters."""
        # This is a simplified simulation - replace with actual backtesting
        
        # Base performance characteristics for each strategy
        base_performance = {
            "ultra_scalp_bot": {
                "base_profit": 0.15,
                "base_time": 1.5,
                "base_drawdown": 0.12,
                "base_win_rate": 0.45,
                "base_sharpe": 1.2,
                "base_profit_factor": 1.4,
                "base_delay": 15.0,
                "base_trades": 8.0
            },
            "momentum_exploiter": {
                "base_profit": 0.25,
                "base_time": 2.5,
                "base_drawdown": 0.18,
                "base_win_rate": 0.42,
                "base_sharpe": 1.1,
                "base_profit_factor": 1.3,
                "base_delay": 25.0,
                "base_trades": 6.0
            },
            "volatility_harvester": {
                "base_profit": 0.30,
                "base_time": 3.0,
                "base_drawdown": 0.22,
                "base_win_rate": 0.38,
                "base_sharpe": 0.9,
                "base_profit_factor": 1.2,
                "base_delay": 30.0,
                "base_trades": 5.0
            }
        }
        
        if strategy_name not in base_performance:
            # Use generic base performance
            base = {
                "base_profit": 0.20,
                "base_time": 2.0,
                "base_drawdown": 0.15,
                "base_win_rate": 0.40,
                "base_sharpe": 1.0,
                "base_profit_factor": 1.25,
                "base_delay": 20.0,
                "base_trades": 7.0
            }
        else:
            base = base_performance[strategy_name]
        
        # Parameter adjustments
        profit_multiplier = 1.0
        time_multiplier = 1.0
        risk_multiplier = 1.0
        
        # Adjust based on stop loss and take profit
        if "stop_loss_pct" in parameters and "take_profit_pct" in parameters:
            sl = parameters["stop_loss_pct"]
            tp = parameters["take_profit_pct"]
            risk_reward_ratio = tp / sl
            
            if risk_reward_ratio > 3.0:
                profit_multiplier *= 1.3
                time_multiplier *= 0.8
            elif risk_reward_ratio > 2.0:
                profit_multiplier *= 1.1
                time_multiplier *= 0.9
            elif risk_reward_ratio < 1.5:
                profit_multiplier *= 0.8
                time_multiplier *= 1.2
        
        # Adjust based on score thresholds
        if "min_score" in parameters:
            min_score = parameters["min_score"]
            if min_score < 0.05:
                profit_multiplier *= 1.2
                time_multiplier *= 0.7
                risk_multiplier *= 1.3
            elif min_score > 0.1:
                profit_multiplier *= 0.9
                time_multiplier *= 1.1
                risk_multiplier *= 0.8
        
        # Calculate final performance metrics
        performance = {
            "profit_pct": base["base_profit"] * profit_multiplier,
            "time_to_profit_hours": base["base_time"] * time_multiplier,
            "drawdown": base["base_drawdown"] * risk_multiplier,
            "win_rate": base["base_win_rate"],
            "sharpe_ratio": base["base_sharpe"] * profit_multiplier / risk_multiplier,
            "profit_factor": base["base_profit_factor"],
            "signal_delay_seconds": base["base_delay"],
            "trades_per_hour": base["base_trades"]
        }
        
        # Add some randomness to simulate real market conditions
        for key in performance:
            if key != "win_rate" and key != "profit_factor":
                performance[key] *= np.random.uniform(0.8, 1.2)
        
        return performance
    
    def _calculate_fitness_score(
        self, 
        performance: Dict[str, float], 
        target: OptimizationTarget
    ) -> float:
        """Calculate fitness score based on performance and targets."""
        
        # Normalize each metric to 0-1 scale
        normalized_metrics = {}
        
        # Profit (higher is better)
        normalized_metrics["profit"] = min(performance["profit_pct"] / target.max_profit_pct, 1.0)
        
        # Time to profit (lower is better)
        normalized_metrics["time"] = max(0, 1 - (performance["time_to_profit_hours"] / target.min_time_to_profit_hours))
        
        # Risk metrics (lower is better)
        normalized_metrics["drawdown"] = max(0, 1 - (performance["drawdown"] / target.max_drawdown))
        normalized_metrics["volatility"] = max(0, 1 - (performance["drawdown"] / target.max_volatility))
        
        # Performance metrics (higher is better)
        normalized_metrics["win_rate"] = min(performance["win_rate"] / target.min_win_rate, 1.0)
        normalized_metrics["sharpe"] = min(performance["sharpe_ratio"] / target.min_sharpe_ratio, 1.0)
        normalized_metrics["profit_factor"] = min(performance["profit_factor"] / target.min_profit_factor, 1.0)
        
        # Speed metrics (lower is better)
        normalized_metrics["delay"] = max(0, 1 - (performance["signal_delay_seconds"] / target.max_signal_delay_seconds))
        normalized_metrics["trades"] = min(performance["trades_per_hour"] / target.min_trades_per_hour, 1.0)
        
        # Calculate weighted fitness score
        weights = {
            "profit": 0.25,      # 25% weight on profit
            "time": 0.20,        # 20% weight on speed
            "drawdown": 0.15,    # 15% weight on risk
            "volatility": 0.10,  # 10% weight on volatility
            "win_rate": 0.10,    # 10% weight on win rate
            "sharpe": 0.10,      # 10% weight on Sharpe ratio
            "profit_factor": 0.05, # 5% weight on profit factor
            "delay": 0.03,       # 3% weight on signal delay
            "trades": 0.02       # 2% weight on trade frequency
        }
        
        fitness_score = sum(
            normalized_metrics[metric] * weights[metric] 
            for metric in weights
        )
        
        return fitness_score
    
    def _optimize_strategy_genetic(
        self, 
        strategy_name: str, 
        target: OptimizationTarget
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize strategy using genetic algorithm."""
        
        param_ranges = self.parameter_ranges.get(strategy_name, {})
        if not param_ranges:
            logger.warning(f"No parameter ranges defined for {strategy_name}")
            return {}, 0.0
        
        # Convert parameter ranges to bounds for differential evolution
        bounds = []
        param_names = []
        
        for param_name, values in param_ranges.items():
            if isinstance(values, list) and len(values) > 1:
                bounds.append((min(values), max(values)))
                param_names.append(param_name)
        
        if not bounds:
            logger.warning(f"No valid parameter ranges for {strategy_name}")
            return {}, 0.0
        
        def objective_function(params):
            """Objective function for optimization."""
            parameters = dict(zip(param_names, params))
            
            try:
                performance = self._evaluate_strategy_performance(strategy_name, parameters)
                fitness = self._calculate_fitness_score(performance, target)
                
                # Store optimization history
                self.optimization_history.append({
                    "strategy": strategy_name,
                    "parameters": parameters,
                    "performance": performance,
                    "fitness": fitness,
                    "timestamp": pd.Timestamp.now().isoformat()
                })
                
                return -fitness  # Negative because we're minimizing
                
            except Exception as e:
                logger.error(f"Error in objective function: {e}")
                return 1.0  # Return high value (low fitness) on error
        
        # Run differential evolution
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                maxiter=self.config.max_iterations,
                popsize=self.config.population_size,
                mutation=self.config.mutation_rate,
                recombination=self.config.crossover_rate,
                seed=42
            )
            
            if result.success:
                best_params = dict(zip(param_names, result.x))
                best_fitness = -result.fun  # Convert back to positive
                
                logger.info(f"Genetic optimization successful for {strategy_name}")
                logger.info(f"Best fitness: {best_fitness:.4f}")
                logger.info(f"Best parameters: {best_params}")
                
                return best_params, best_fitness
            else:
                logger.warning(f"Genetic optimization failed for {strategy_name}")
                return {}, 0.0
                
        except Exception as e:
            logger.error(f"Error in genetic optimization for {strategy_name}: {e}")
            return {}, 0.0
    
    def _optimize_strategy_bayesian(
        self, 
        strategy_name: str, 
        target: OptimizationTarget
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize strategy using Bayesian optimization with Optuna."""
        
        param_ranges = self.parameter_ranges.get(strategy_name, {})
        if not param_ranges:
            logger.warning(f"No parameter ranges defined for {strategy_name}")
            return {}, 0.0
        
        def objective(trial):
            """Objective function for Optuna."""
            parameters = {}
            
            for param_name, values in param_ranges.items():
                if isinstance(values, list) and len(values) > 1:
                    if isinstance(values[0], int):
                        parameters[param_name] = trial.suggest_int(param_name, min(values), max(values))
                    else:
                        parameters[param_name] = trial.suggest_float(param_name, min(values), max(values))
            
            try:
                performance = self._evaluate_strategy_performance(strategy_name, parameters)
                fitness = self._calculate_fitness_score(performance, target)
                
                # Store optimization history
                self.optimization_history.append({
                    "strategy": strategy_name,
                    "parameters": parameters,
                    "performance": performance,
                    "fitness": fitness,
                    "timestamp": pd.Timestamp.now().isoformat()
                })
                
                return fitness
                
            except Exception as e:
                logger.error(f"Error in objective function: {e}")
                return 0.0
        
        try:
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.config.max_iterations)
            
            best_params = study.best_params
            best_fitness = study.best_value
            
            logger.info(f"Bayesian optimization successful for {strategy_name}")
            logger.info(f"Best fitness: {best_fitness:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
            return best_params, best_fitness
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization for {strategy_name}: {e}")
            return {}, 0.0
    
    def optimize_strategy(
        self, 
        strategy_name: str, 
        target: OptimizationTarget
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize a single strategy."""
        
        logger.info(f"Starting optimization for {strategy_name}")
        
        if self.config.optimization_method == "genetic":
            return self._optimize_strategy_genetic(strategy_name, target)
        elif self.config.optimization_method == "bayesian":
            return self._optimize_strategy_bayesian(strategy_name, target)
        else:
            logger.error(f"Unknown optimization method: {self.config.optimization_method}")
            return {}, 0.0
    
    def optimize_all_strategies(self, target: OptimizationTarget) -> Dict[str, Dict[str, Any]]:
        """Optimize all strategies."""
        
        logger.info("Starting optimization of all strategies")
        
        results = {}
        
        for strategy_name in self.parameter_ranges.keys():
            logger.info(f"Optimizing {strategy_name}...")
            
            try:
                best_params, best_fitness = self.optimize_strategy(strategy_name, target)
                
                if best_params:
                    results[strategy_name] = {
                        "parameters": best_params,
                        "fitness": best_fitness,
                        "status": "success"
                    }
                else:
                    results[strategy_name] = {
                        "parameters": {},
                        "fitness": 0.0,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error optimizing {strategy_name}: {e}")
                results[strategy_name] = {
                    "parameters": {},
                    "fitness": 0.0,
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    def save_results(self, results: Dict[str, Dict[str, Any]], target: OptimizationTarget):
        """Save optimization results."""
        
        # Save detailed results
        results_file = self.results_dir / f"optimization_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_data = {
            "target": asdict(target),
            "config": asdict(self.config),
            "results": results,
            "history": self.optimization_history,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save optimized configuration if requested
        if self.config.save_optimized_config:
            self._save_optimized_config(results)
    
    def _save_optimized_config(self, results: Dict[str, Dict[str, Any]]):
        """Save optimized configuration to config.yaml."""
        
        # Load current config
        config_path = Path(__file__).parent.parent / "config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update strategy configurations with optimized parameters
            for strategy_name, result in results.items():
                if result["status"] == "success" and result["parameters"]:
                    # Find the strategy section in config
                    for key in config.keys():
                        if key.endswith("_bot") or key.endswith("_scalper"):
                            if key == strategy_name:
                                config[key].update(result["parameters"])
                                logger.info(f"Updated {key} configuration with optimized parameters")
                                break
            
            # Save updated config
            backup_path = config_path.with_suffix('.yaml.backup')
            with open(backup_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Updated configuration saved. Backup created at {backup_path}")
            
        except Exception as e:
            logger.error(f"Error saving optimized configuration: {e}")
    
    def generate_optimization_report(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Generate a comprehensive optimization report."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("STRATEGY OPTIMIZATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary statistics
        successful_optimizations = sum(1 for r in results.values() if r["status"] == "success")
        total_strategies = len(results)
        
        report_lines.append(f"SUMMARY:")
        report_lines.append(f"  Total strategies: {total_strategies}")
        report_lines.append(f"  Successful optimizations: {successful_optimizations}")
        report_lines.append(f"  Success rate: {successful_optimizations/total_strategies*100:.1f}%")
        report_lines.append("")
        
        # Individual strategy results
        report_lines.append("STRATEGY RESULTS:")
        report_lines.append("-" * 80)
        
        for strategy_name, result in results.items():
            report_lines.append(f"{strategy_name.upper()}:")
            report_lines.append(f"  Status: {result['status']}")
            
            if result["status"] == "success":
                report_lines.append(f"  Fitness Score: {result['fitness']:.4f}")
                report_lines.append(f"  Optimized Parameters:")
                for param, value in result["parameters"].items():
                    report_lines.append(f"    {param}: {value}")
            elif result["status"] == "error":
                report_lines.append(f"  Error: {result.get('error', 'Unknown error')}")
            
            report_lines.append("")
        
        # Performance analysis
        if self.optimization_history:
            report_lines.append("PERFORMANCE ANALYSIS:")
            report_lines.append("-" * 80)
            
            # Find best overall performance
            best_performance = max(self.optimization_history, key=lambda x: x["fitness"])
            report_lines.append(f"Best overall fitness: {best_performance['fitness']:.4f}")
            report_lines.append(f"Strategy: {best_performance['strategy']}")
            report_lines.append(f"Parameters: {best_performance['parameters']}")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


async def main():
    """Main optimization function."""
    
    # Configuration
    config = StrategyOptimizationConfig(
        optimization_method="bayesian",  # Use Bayesian optimization
        max_iterations=500,
        use_multiprocessing=True,
        save_optimized_config=True
    )
    
    # Optimization targets
    target = OptimizationTarget(
        max_profit_pct=0.50,
        min_time_to_profit_hours=2.0,
        max_drawdown=0.25,
        max_volatility=0.40,
        min_win_rate=0.35,
        min_sharpe_ratio=0.8,
        min_profit_factor=1.2,
        max_signal_delay_seconds=30.0,
        min_trades_per_hour=5.0
    )
    
    # Create optimizer
    optimizer = StrategyOptimizer(config)
    
    # Run optimization
    logger.info("Starting strategy optimization...")
    results = optimizer.optimize_all_strategies(target)
    
    # Save results
    optimizer.save_results(results, target)
    
    # Generate report
    report = optimizer.generate_optimization_report(results)
    print(report)
    
    # Save report
    report_file = optimizer.results_dir / f"optimization_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Optimization complete. Report saved to {report_file}")
    
    return results


if __name__ == "__main__":
    # Run the optimization
    results = asyncio.run(main())
    
    print(f"\nOptimization completed. Check {LOG_DIR} for detailed logs.")
    print(f"Results saved to: {Path(__file__).parent.parent}/logs/optimization_results/")
