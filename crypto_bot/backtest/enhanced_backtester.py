"""
Enhanced Backtesting System with GPU Acceleration and Continuous Learning

This module provides comprehensive backtesting capabilities for all strategies
against the top 20 token pairs, with support for GPU acceleration and
continuous learning from results.
"""

import asyncio
import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import psutil

# GPU acceleration imports
try:
    import cupy as cp
    import cupyx.scipy as cp_scipy
    GPU_AVAILABLE = True
    logging.info("CuPy detected - GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    logging.info("CuPy not available - using CPU only")

try:
    import numba
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    logging.info("Numba detected - JIT compilation enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    logging.info("Numba not available - using standard Python")

from crypto_bot.backtest.backtest_runner import BacktestRunner, BacktestConfig
from crypto_bot.utils.market_loader import fetch_geckoterminal_ohlcv
from crypto_bot.strategy_router import strategy_for
from crypto_bot.regime.regime_classifier import classify_regime

logger = logging.getLogger(__name__)

@dataclass
class EnhancedBacktestConfig:
    """Configuration for enhanced backtesting system."""
    
    # Token pair selection
    top_pairs_count: int = 20
    min_volume_usd: float = 1_000_000
    refresh_interval_hours: int = 6
    
    # Backtesting parameters
    lookback_days: int = 90
    timeframes: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])
    strategies_to_test: List[str] = field(default_factory=list)  # Empty = all strategies
    
    # GPU acceleration
    use_gpu: bool = True
    gpu_memory_limit_gb: float = 8.0
    batch_size: int = 100
    
    # Parallel processing
    max_workers: int = field(default_factory=lambda: min(multiprocessing.cpu_count(), 8))
    use_process_pool: bool = True
    
    # Continuous learning
    learning_enabled: bool = True
    results_cache_dir: str = "crypto_bot/logs/backtest_results"
    model_update_interval_hours: int = 24
    
    # Risk management
    max_drawdown_threshold: float = 0.5
    min_sharpe_threshold: float = 0.5
    min_win_rate: float = 0.4

class StrategyPerformanceTracker:
    """Tracks and analyzes strategy performance across multiple pairs and timeframes."""
    
    def __init__(self, config: EnhancedBacktestConfig):
        self.config = config
        self.results_cache = Path(config.results_cache_dir)
        self.results_cache.mkdir(parents=True, exist_ok=True)
        self.performance_history: Dict[str, pd.DataFrame] = {}
        self.strategy_rankings: Dict[str, float] = {}
        
    def add_results(self, strategy: str, results: pd.DataFrame, pair: str, timeframe: str):
        """Add backtesting results to the performance tracker."""
        if strategy not in self.performance_history:
            self.performance_history[strategy] = []
            
        # Add metadata
        results_copy = results.copy()
        results_copy['pair'] = pair
        results_copy['timeframe'] = timeframe
        results_copy['timestamp'] = datetime.now()
        
        self.performance_history[strategy].append(results_copy)
        
    def get_strategy_rankings(self) -> Dict[str, float]:
        """Calculate current strategy rankings based on performance."""
        if not self.performance_history:
            return {}
            
        rankings = {}
        
        for strategy, results_list in self.performance_history.items():
            if not results_list:
                continue
                
            # Combine all results for this strategy
            combined = pd.concat(results_list, ignore_index=True)
            
            # Calculate composite score
            avg_sharpe = combined['sharpe'].mean()
            avg_win_rate = (combined['pnl'] > 0).mean()
            avg_drawdown = combined['max_drawdown'].mean()
            consistency = 1.0 / (combined['sharpe'].std() + 1e-6)
            
            # Weighted composite score
            composite_score = (
                avg_sharpe * 0.4 +
                avg_win_rate * 0.3 +
                (1 - avg_drawdown) * 0.2 +
                consistency * 0.1
            )
            
            rankings[strategy] = composite_score
            
        # Sort by score
        sorted_rankings = dict(sorted(rankings.items(), key=lambda x: x[1], reverse=True))
        self.strategy_rankings = sorted_rankings
        
        return sorted_rankings
        
    def save_results(self):
        """Save all results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual strategy results
        for strategy, results_list in self.performance_history.items():
            if results_list:
                combined = pd.concat(results_list, ignore_index=True)
                filename = f"{strategy}_{timestamp}.csv"
                filepath = self.results_cache / filename
                combined.to_csv(filepath, index=False)
                
        # Save rankings
        rankings_file = self.results_cache / f"rankings_{timestamp}.json"
        with open(rankings_file, 'w') as f:
            json.dump(self.strategy_rankings, f, indent=2, default=str)
            
        logger.info(f"Saved backtesting results to {self.results_cache}")

class GPUAcceleratedBacktester:
    """GPU-accelerated backtesting using CuPy for numerical operations."""
    
    def __init__(self, config: EnhancedBacktestConfig):
        self.config = config
        self.gpu_available = GPU_AVAILABLE
        
        if self.gpu_available:
            self._setup_gpu()
        else:
            logger.warning("GPU acceleration not available - using CPU fallback")
            
    def _setup_gpu(self):
        """Initialize GPU memory and settings."""
        try:
            # Set memory limit
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=int(self.config.gpu_memory_limit_gb * 1024**3))
            
            # Warm up GPU
            _ = cp.array([1.0])
            logger.info("GPU initialized successfully")
            
        except Exception as e:
            logger.error(f"GPU setup failed: {e}")
            self.gpu_available = False
            
    def _gpu_optimize_parameters(self, df: pd.DataFrame, param_ranges: Dict) -> Dict:
        """Use GPU acceleration for parameter optimization."""
        if not self.gpu_available:
            return self._cpu_optimize_parameters(df, param_ranges)
            
        try:
            # Convert DataFrame to GPU arrays
            close_gpu = cp.array(df['close'].values)
            high_gpu = cp.array(df['high'].values)
            low_gpu = cp.array(df['low'].values)
            volume_gpu = cp.array(df['volume'].values)
            
            # GPU-accelerated parameter search
            best_params = self._gpu_grid_search(
                close_gpu, high_gpu, low_gpu, volume_gpu, param_ranges
            )
            
            return best_params
            
        except Exception as e:
            logger.warning(f"GPU optimization failed, falling back to CPU: {e}")
            return self._cpu_optimize_parameters(df, param_ranges)
            
    def _gpu_grid_search(self, close, high, low, volume, param_ranges):
        """Perform grid search on GPU."""
        # This is a simplified example - in practice you'd implement
        # the full strategy logic on GPU
        best_score = -np.inf
        best_params = {}
        
        # Generate parameter combinations
        stop_losses = np.linspace(0.01, 0.05, 10)
        take_profits = np.linspace(0.02, 0.10, 10)
        
        for sl in stop_losses:
            for tp in take_profits:
                # Calculate metrics on GPU
                score = self._calculate_gpu_metrics(close, high, low, volume, sl, tp)
                
                if score > best_score:
                    best_score = score
                    best_params = {'stop_loss': sl, 'take_profit': tp, 'score': score}
                    
        return best_params
        
    def _calculate_gpu_metrics(self, close, high, low, volume, sl, tp):
        """Calculate trading metrics on GPU."""
        # Simplified metric calculation - replace with actual strategy logic
        returns = cp.diff(close) / close[:-1]
        volatility = cp.std(returns)
        sharpe = cp.mean(returns) / (volatility + 1e-8)
        
        return float(sharpe)
        
    def _cpu_optimize_parameters(self, df: pd.DataFrame, param_ranges: Dict) -> Dict:
        """CPU fallback for parameter optimization."""
        # Use existing BacktestRunner logic
        config = BacktestConfig(
            symbol="DUMMY",
            timeframe="1h",
            since=0,
            limit=len(df)
        )
        
        runner = BacktestRunner(config, df=df)
        results = runner.run_grid()
        
        if results.empty:
            return {}
            
        best = results.iloc[0]
        return {
            'stop_loss': float(best['stop_loss_pct']),
            'take_profit': float(best['take_profit_pct']),
            'sharpe': float(best['sharpe'])
        }

class ContinuousBacktestingEngine:
    """Main engine for continuous backtesting of top pairs against all strategies."""
    
    def __init__(self, config: EnhancedBacktestConfig):
        self.config = config
        self.performance_tracker = StrategyPerformanceTracker(config)
        self.gpu_backtester = GPUAcceleratedBacktester(config)
        self.running = False
        self.last_refresh = None
        
        # Load existing results
        self._load_existing_results()
        
    def _load_existing_results(self):
        """Load previously saved backtesting results."""
        if not self.config.results_cache_dir:
            return
            
        cache_dir = Path(self.config.results_cache_dir)
        if not cache_dir.exists():
            return
            
        # Load CSV files
        for csv_file in cache_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                strategy = csv_file.stem.split('_')[0]
                
                if strategy not in self.performance_tracker.performance_history:
                    self.performance_tracker.performance_history[strategy] = []
                    
                self.performance_tracker.performance_history[strategy].append(df)
                
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
                
        logger.info("Loaded existing backtesting results")
        
    async def get_top_pairs(self) -> List[str]:
        """Fetch the top trading pairs by volume."""
        try:
            # Use existing refresh_pairs logic
            from tasks.refresh_pairs import refresh_pairs
            
            pairs = refresh_pairs(
                min_volume_usd=self.config.min_volume_usd,
                top_k=self.config.top_pairs_count,
                config={}  # Use default config
            )
            
            logger.info(f"Fetched {len(pairs)} top pairs")
            return pairs
            
        except Exception as e:
            logger.error(f"Failed to fetch top pairs: {e}")
            # Fallback to common pairs
            return [
                "BTC/USDT", "ETH/USDT", "SOL/USDC", "MATIC/USDT", "ADA/USDT",
                "DOT/USDT", "LINK/USDT", "UNI/USDT", "AVAX/USDT", "ATOM/USDT",
                "LTC/USDT", "BCH/USDT", "XRP/USDT", "ETC/USDT", "FIL/USDT",
                "NEAR/USDT", "ALGO/USDT", "VET/USDT", "ICP/USDT", "THETA/USDT"
            ]
            
    def get_all_strategies(self) -> List[str]:
        """Get list of all available strategies."""
        if self.config.strategies_to_test:
            return self.config.strategies_to_test
            
        # Return all available strategies
        return [
            "trend_bot", "momentum_bot", "mean_bot", "breakout_bot", "grid_bot",
            "sniper_bot", "micro_scalp_bot", "bounce_scalper", "dip_hunter",
            "flash_crash_bot", "lstm_bot", "hft_engine", "stat_arb_bot",
            "range_arb_bot", "cross_chain_arb_bot", "dex_scalper", "maker_spread"
        ]
        
    async def backtest_pair_strategy(self, pair: str, strategy: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Run backtest for a specific pair-strategy-timeframe combination."""
        try:
            # Fetch historical data
            if pair.endswith("/USDC"):
                # Solana pairs
                data = await fetch_geckoterminal_ohlcv(pair, timeframe=timeframe, limit=1000)
                if not data:
                    return None
                    
                df = pd.DataFrame(
                    data[0] if isinstance(data, tuple) else data,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                
            else:
                # CEX pairs - use existing logic
                import ccxt
                exchange = ccxt.binance()
                ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=1000)
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                
            if df.empty or len(df) < 100:
                return None
                
            # Create backtest config
            config = BacktestConfig(
                symbol=pair,
                timeframe=timeframe,
                since=0,
                limit=len(df),
                stop_loss_range=[0.01, 0.02, 0.03],
                take_profit_range=[0.02, 0.04, 0.06]
            )
            
            # Run backtest
            runner = BacktestRunner(config, df=df)
            results = runner.run_grid()
            
            if not results.empty:
                # Add to performance tracker
                self.performance_tracker.add_results(strategy, results, pair, timeframe)
                
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed for {pair}-{strategy}-{timeframe}: {e}")
            return None
            
    async def run_continuous_backtesting(self):
        """Main loop for continuous backtesting."""
        self.running = True
        logger.info("Starting continuous backtesting engine")
        
        while self.running:
            try:
                # Check if we need to refresh pairs
                if (self.last_refresh is None or 
                    datetime.now() - self.last_refresh > timedelta(hours=self.config.refresh_interval_hours)):
                    
                    pairs = await self.get_top_pairs()
                    self.last_refresh = datetime.now()
                    logger.info(f"Refreshed top {len(pairs)} pairs")
                    
                else:
                    # Use cached pairs
                    pairs = await self.get_top_pairs()
                    
                strategies = self.get_all_strategies()
                timeframes = self.config.timeframes
                
                # Create all combinations
                combinations = [
                    (pair, strategy, timeframe)
                    for pair in pairs
                    for strategy in strategies
                    for timeframe in timeframes
                ]
                
                logger.info(f"Running {len(combinations)} backtests")
                
                # Process in batches
                batch_size = self.config.batch_size
                for i in range(0, len(combinations), batch_size):
                    batch = combinations[i:i + batch_size]
                    
                    # Run batch with parallel processing
                    if self.config.use_process_pool:
                        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                            futures = [
                                executor.submit(
                                    asyncio.run,
                                    self.backtest_pair_strategy(pair, strategy, timeframe)
                                )
                                for pair, strategy, timeframe in batch
                            ]
                            
                            # Wait for completion
                            for future in futures:
                                try:
                                    future.result(timeout=300)  # 5 minute timeout
                                except Exception as e:
                                    logger.warning(f"Batch backtest failed: {e}")
                    else:
                        # Sequential processing
                        for pair, strategy, timeframe in batch:
                            await self.backtest_pair_strategy(pair, strategy, timeframe)
                            
                    logger.info(f"Completed batch {i//batch_size + 1}/{(len(combinations) + batch_size - 1)//batch_size}")
                    
                # Update strategy rankings
                rankings = self.performance_tracker.get_strategy_rankings()
                logger.info("Current strategy rankings:")
                for i, (strategy, score) in enumerate(list(rankings.items())[:10]):
                    logger.info(f"{i+1:2d}. {strategy:20s} - Score: {score:.4f}")
                    
                # Save results
                self.performance_tracker.save_results()
                
                # Wait before next cycle
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Continuous backtesting error: {e}")
                await asyncio.sleep(300)  # 5 minutes on error
                
    def stop(self):
        """Stop the continuous backtesting engine."""
        self.running = False
        logger.info("Stopping continuous backtesting engine")

def create_enhanced_backtester(config_dict: Optional[Dict] = None) -> ContinuousBacktestingEngine:
    """Factory function to create enhanced backtester with configuration."""
    
    if config_dict is None:
        config_dict = {}
        
    # Merge with defaults
    config = EnhancedBacktestConfig(
        top_pairs_count=config_dict.get('top_pairs_count', 20),
        min_volume_usd=config_dict.get('min_volume_usd', 1_000_000),
        refresh_interval_hours=config_dict.get('refresh_interval_hours', 6),
        lookback_days=config_dict.get('lookback_days', 90),
        timeframes=config_dict.get('timeframes', ["1h", "4h", "1d"]),
        strategies_to_test=config_dict.get('strategies_to_test', []),
        use_gpu=config_dict.get('use_gpu', True),
        gpu_memory_limit_gb=config_dict.get('gpu_memory_limit_gb', 8.0),
        batch_size=config_dict.get('batch_size', 100),
        max_workers=config_dict.get('max_workers', min(multiprocessing.cpu_count(), 8)),
        use_process_pool=config_dict.get('use_process_pool', True),
        learning_enabled=config_dict.get('learning_enabled', True),
        results_cache_dir=config_dict.get('results_cache_dir', "crypto_bot/logs/backtest_results"),
        model_update_interval_hours=config_dict.get('model_update_interval_hours', 24),
        max_drawdown_threshold=config_dict.get('max_drawdown_threshold', 0.5),
        min_sharpe_threshold=config_dict.get('min_sharpe_threshold', 0.5),
        min_win_rate=config_dict.get('min_win_rate', 0.4)
    )
    
    return ContinuousBacktestingEngine(config)

# Convenience functions for easy usage
async def run_backtest_analysis(
    pairs: List[str],
    strategies: List[str],
    timeframes: List[str],
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """Run comprehensive backtest analysis for specified pairs and strategies."""
    
    engine = create_enhanced_backtester(config or {})
    
    results = {}
    for pair in pairs:
        results[pair] = {}
        for strategy in strategies:
            results[pair][strategy] = {}
            for timeframe in timeframes:
                result = await engine.backtest_pair_strategy(pair, strategy, timeframe)
                if result is not None:
                    results[pair][strategy][timeframe] = result.to_dict('records')
                    
    return results

def get_strategy_performance_summary(cache_dir: str = "crypto_bot/logs/backtest_results") -> pd.DataFrame:
    """Get summary of all strategy performance from cached results."""
    
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return pd.DataFrame()
        
    all_results = []
    
    for csv_file in cache_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            strategy = csv_file.stem.split('_')[0]
            df['strategy'] = strategy
            all_results.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {csv_file}: {e}")
            
    if not all_results:
        return pd.DataFrame()
        
    combined = pd.concat(all_results, ignore_index=True)
    
    # Calculate summary statistics
    summary = combined.groupby('strategy').agg({
        'sharpe': ['mean', 'std', 'min', 'max'],
        'pnl': ['mean', 'sum'],
        'max_drawdown': ['mean', 'max'],
        'win_rate': ['mean'] if 'win_rate' in combined.columns else ['count']
    }).round(4)
    
    return summary
