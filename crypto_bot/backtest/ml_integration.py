"""
ML Integration Module for Enhanced Backtesting System

This module integrates the enhanced backtesting system with the coinTrader_Trainer
ML system to create a unified trading intelligence platform.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
import yaml

from crypto_bot.backtest.enhanced_backtester import create_enhanced_backtester
from crypto_bot.backtest.gpu_accelerator import create_gpu_accelerator

logger = logging.getLogger(__name__)

class MLBacktestingIntegration:
    """
    Integration layer between enhanced backtesting and coinTrader_Trainer ML system.
    
    This class coordinates:
    1. Backtesting results → ML training data
    2. ML model predictions → Backtesting validation
    3. Continuous learning loop between both systems
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backtest_engine = create_enhanced_backtester(config)
        self.gpu_accelerator = create_gpu_accelerator(config)
        
        # ML system configuration
        self.ml_config = config.get('ml_integration', {})
        self.cointrainer_path = self.ml_config.get('cointrainer_path', 'cointrainer')
        self.supabase_config = self.ml_config.get('supabase', {})
        
        # Integration settings
        self.auto_retrain_enabled = self.ml_config.get('auto_retrain', True)
        self.retrain_interval_hours = self.ml_config.get('retrain_interval_hours', 24)
        self.last_retrain = None
        
        # Feature engineering settings
        self.feature_config = self.ml_config.get('features', {})
        self.horizon_minutes = self.feature_config.get('horizon_minutes', 15)
        self.hold_threshold = self.feature_config.get('hold_threshold', 0.0015)
        
    async def run_integrated_backtesting(self):
        """
        Run integrated backtesting that feeds results to ML training.
        """
        logger.info("Starting integrated backtesting with ML training")
        
        try:
            # Run enhanced backtesting
            await self.backtest_engine.run_continuous_backtesting()
            
        except Exception as e:
            logger.error(f"Integrated backtesting failed: {e}")
            raise
            
    async def prepare_ml_training_data(self, backtest_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert backtesting results into ML training data format.
        """
        logger.info("Preparing ML training data from backtesting results")
        
        try:
            # Extract performance metrics from backtesting results
            training_data = []
            
            for pair, pair_results in backtest_results.items():
                for strategy, strategy_results in pair_results.items():
                    for timeframe, timeframe_results in strategy_results.items():
                        if not timeframe_results:
                            continue
                            
                        # Convert backtest results to ML features
                        features = self._extract_ml_features(
                            pair, strategy, timeframe, timeframe_results
                        )
                        
                        if features:
                            training_data.append(features)
            
            if not training_data:
                logger.warning("No training data extracted from backtesting results")
                return pd.DataFrame()
                
            # Create DataFrame
            df = pd.DataFrame(training_data)
            
            # Add timestamp for ML training
            df['timestamp'] = datetime.now()
            df['data_source'] = 'enhanced_backtesting'
            
            logger.info(f"Prepared {len(df)} training samples")
            return df
            
        except Exception as e:
            logger.error(f"Failed to prepare ML training data: {e}")
            return pd.DataFrame()
            
    def _extract_ml_features(self, pair: str, strategy: str, timeframe: str, results: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Extract ML features from backtesting results.
        """
        try:
            if not results:
                return None
                
            # Aggregate results for this strategy-timeframe combination
            df_results = pd.DataFrame(results)
            
            # Calculate aggregate metrics
            avg_sharpe = df_results['sharpe'].mean()
            avg_pnl = df_results['pnl'].mean()
            max_dd = df_results['max_drawdown'].max()
            win_rate = (df_results['pnl'] > 0).mean()
            
            # Create feature vector
            features = {
                'pair': pair,
                'strategy': strategy,
                'timeframe': timeframe,
                'avg_sharpe': avg_sharpe,
                'avg_pnl': avg_pnl,
                'max_drawdown': max_dd,
                'win_rate': win_rate,
                'total_trades': len(results),
                'avg_stop_loss': df_results['stop_loss_pct'].mean(),
                'avg_take_profit': df_results['take_profit_pct'].mean(),
                'strategy_consistency': 1.0 / (df_results['sharpe'].std() + 1e-6),
                'risk_adjusted_return': avg_pnl / (max_dd + 1e-6)
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Failed to extract features for {pair}-{strategy}-{timeframe}: {e}")
            return None
            
    async def train_ml_models(self, training_data: pd.DataFrame, symbols: List[str] = None):
        """
        Train ML models using coinTrader_Trainer.
        """
        if training_data.empty:
            logger.warning("No training data available for ML training")
            return
            
        try:
            logger.info("Starting ML model training with coinTrader_Trainer")
            
            # Save training data to CSV for coinTrader_Trainer
            training_file = self._save_training_data(training_data)
            
            # Train models for each symbol
            if symbols is None:
                symbols = training_data['pair'].unique().tolist()
                
            for symbol in symbols:
                await self._train_symbol_model(symbol, training_file)
                
            logger.info("ML model training completed")
            
        except Exception as e:
            logger.error(f"ML model training failed: {e}")
            
    def _save_training_data(self, training_data: pd.DataFrame) -> str:
        """
        Save training data to CSV format compatible with coinTrader_Trainer.
        """
        try:
            # Create training data directory
            training_dir = Path("data/ml_training")
            training_dir.mkdir(parents=True, exist_ok=True)
            
            # Save aggregated data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_backtest_training_{timestamp}.csv"
            filepath = training_dir / filename
            
            training_data.to_csv(filepath, index=False)
            logger.info(f"Saved training data to {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
            return ""
            
    async def _train_symbol_model(self, symbol: str, training_file: str):
        """
        Train ML model for a specific symbol using coinTrader_Trainer.
        """
        try:
            # Prepare coinTrader_Trainer command
            cmd = [
                self.cointrainer_path,
                "csv-train",
                "--file", training_file,
                "--symbol", symbol,
                "--horizon", str(self.horizon_minutes),
                "--hold", str(self.hold_threshold),
                "--device-type", "gpu" if self.gpu_accelerator.gpu_available else "cpu"
            ]
            
            # Add GPU optimization if available
            if self.gpu_accelerator.gpu_available:
                cmd.extend(["--max-bin", "63", "--n-jobs", "0"])
                
            # Add publishing if configured
            if self.supabase_config:
                cmd.append("--publish")
                
            logger.info(f"Training ML model for {symbol}: {' '.join(cmd)}")
            
            # Execute training command
            result = await self._execute_cointrainer_command(cmd)
            
            if result['success']:
                logger.info(f"Successfully trained ML model for {symbol}")
                self.last_retrain = datetime.now()
            else:
                logger.error(f"Failed to train ML model for {symbol}: {result['error']}")
                
        except Exception as e:
            logger.error(f"Error training ML model for {symbol}: {e}")
            
    async def _execute_cointrainer_command(self, cmd: List[str]) -> Dict[str, Any]:
        """
        Execute coinTrader_Trainer command asynchronously.
        """
        try:
            # Run command in subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            success = process.returncode == 0
            
            return {
                'success': success,
                'returncode': process.returncode,
                'stdout': stdout.decode() if stdout else '',
                'stderr': stderr.decode() if stderr else '',
                'error': stderr.decode() if stderr and not success else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'returncode': -1
            }
            
    async def validate_ml_predictions(self, symbol: str, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate ML model predictions against backtesting results.
        """
        try:
            logger.info(f"Validating ML predictions for {symbol}")
            
            # Get ML predictions using coinTrader_Trainer runtime
            predictions = await self._get_ml_predictions(symbol, features_df)
            
            # Compare with backtesting results
            validation_results = self._validate_predictions(predictions, symbol)
            
            logger.info(f"ML validation completed for {symbol}")
            return validation_results
            
        except Exception as e:
            logger.error(f"ML validation failed for {symbol}: {e}")
            return {'valid': False, 'error': str(e)}
            
    async def _get_ml_predictions(self, symbol: str, features_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Get ML model predictions using coinTrader_Trainer runtime.
        """
        try:
            # This would integrate with coinTrader_Trainer's runtime API
            # For now, return placeholder predictions
            predictions = []
            
            for _, row in features_df.iterrows():
                # Simulate ML prediction
                prediction = {
                    'action': np.random.choice(['buy', 'sell', 'hold']),
                    'confidence': np.random.random(),
                    'timestamp': datetime.now(),
                    'features': row.to_dict()
                }
                predictions.append(prediction)
                
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to get ML predictions: {e}")
            return []
            
    def _validate_predictions(self, predictions: List[Dict], symbol: str) -> Dict[str, Any]:
        """
        Validate ML predictions against backtesting performance.
        """
        try:
            if not predictions:
                return {'valid': False, 'error': 'No predictions available'}
                
            # Get backtesting results for this symbol
            symbol_results = self._get_symbol_backtest_results(symbol)
            
            if not symbol_results:
                return {'valid': False, 'error': 'No backtesting results available'}
                
            # Calculate validation metrics
            total_predictions = len(predictions)
            buy_signals = sum(1 for p in predictions if p['action'] == 'buy')
            sell_signals = sum(1 for p in predictions if p['action'] == 'sell')
            hold_signals = sum(1 for p in predictions if p['action'] == 'hold')
            
            avg_confidence = np.mean([p['confidence'] for p in predictions])
            
            validation_results = {
                'valid': True,
                'symbol': symbol,
                'total_predictions': total_predictions,
                'signal_distribution': {
                    'buy': buy_signals,
                    'sell': sell_signals,
                    'hold': hold_signals
                },
                'avg_confidence': avg_confidence,
                'backtest_performance': symbol_results,
                'validation_timestamp': datetime.now()
            }
            
            return validation_results
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
            
    def _get_symbol_backtest_results(self, symbol: str) -> Dict[str, Any]:
        """
        Get backtesting results for a specific symbol.
        """
        try:
            # This would access the performance tracker from the backtesting engine
            # For now, return placeholder data
            return {
                'avg_sharpe': 0.5,
                'avg_pnl': 0.02,
                'max_drawdown': 0.15,
                'win_rate': 0.6
            }
            
        except Exception as e:
            logger.warning(f"Failed to get backtest results for {symbol}: {e}")
            return {}
            
    async def run_continuous_learning_loop(self):
        """
        Run continuous learning loop that integrates both systems.
        """
        logger.info("Starting continuous learning loop")
        
        try:
            while True:
                # Check if it's time to retrain
                if self._should_retrain():
                    logger.info("Starting ML model retraining cycle")
                    
                    # Get latest backtesting results
                    backtest_results = self._get_latest_backtest_results()
                    
                    # Prepare training data
                    training_data = await self.prepare_ml_training_data(backtest_results)
                    
                    # Train ML models
                    if not training_data.empty:
                        await self.train_ml_models(training_data)
                        
                    # Update last retrain time
                    self.last_retrain = datetime.now()
                    
                # Wait before next cycle
                await asyncio.sleep(3600)  # Check every hour
                
        except Exception as e:
            logger.error(f"Continuous learning loop failed: {e}")
            raise
            
    def _should_retrain(self) -> bool:
        """
        Check if ML models should be retrained.
        """
        if not self.auto_retrain_enabled:
            return False
            
        if self.last_retrain is None:
            return True
            
        time_since_retrain = datetime.now() - self.last_retrain
        return time_since_retrain >= timedelta(hours=self.retrain_interval_hours)
        
    def _get_latest_backtest_results(self) -> Dict[str, Any]:
        """
        Get latest backtesting results from the performance tracker.
        """
        try:
            # This would access the performance tracker from the backtesting engine
            # For now, return placeholder data
            return {
                'BTC/USDT': {
                    'trend_bot': {
                        '1h': [{'sharpe': 0.5, 'pnl': 0.02, 'max_drawdown': 0.15}]
                    }
                }
            }
            
        except Exception as e:
            logger.warning(f"Failed to get latest backtest results: {e}")
            return {}
            
    def get_integration_status(self) -> Dict[str, Any]:
        """
        Get status of the ML integration system.
        """
        return {
            'ml_integration_enabled': True,
            'auto_retrain_enabled': self.auto_retrain_enabled,
            'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None,
            'retrain_interval_hours': self.retrain_interval_hours,
            'gpu_acceleration': self.gpu_accelerator.get_performance_metrics(),
            'supabase_configured': bool(self.supabase_config),
            'cointrainer_available': self._check_cointrainer_availability()
        }
        
    def _check_cointrainer_availability(self) -> bool:
        """
        Check if coinTrader_Trainer is available.
        """
        try:
            result = subprocess.run(
                [self.cointrainer_path, "--help"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

def create_ml_integration(config: Dict[str, Any]) -> MLBacktestingIntegration:
    """
    Factory function to create ML integration instance.
    """
    return MLBacktestingIntegration(config)

# Convenience functions
async def run_integrated_system(config: Dict[str, Any]):
    """
    Run the complete integrated system.
    """
    integration = create_ml_integration(config)
    
    # Start continuous learning loop
    await integration.run_continuous_learning_loop()

def get_integration_config_template() -> Dict[str, Any]:
    """
    Get template configuration for ML integration.
    """
    return {
        'ml_integration': {
            'enabled': True,
            'auto_retrain': True,
            'retrain_interval_hours': 24,
            'cointrainer_path': 'cointrainer',
            'supabase': {
                'url': 'https://your-project.supabase.co',
                'key': 'your-service-role-key',
                'bucket': 'models'
            },
            'features': {
                'horizon_minutes': 15,
                'hold_threshold': 0.0015
            }
        }
    }
