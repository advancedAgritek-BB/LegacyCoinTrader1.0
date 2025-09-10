"""
Enhanced Logging Manager for Trade Pipeline

This module provides comprehensive logging capabilities for the entire trade pipeline,
with special handling for live vs paper trading modes and debug configurations.
"""

import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import traceback
from contextlib import contextmanager

from .logger import LOG_DIR, setup_logger

# Trade pipeline specific loggers
TRADE_PIPELINE_LOGGERS = {}

class TradePipelineLogger:
    """Enhanced logger for trade pipeline operations."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = self._setup_logger()
        self.execution_times = {}
        self.trade_counters = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'paper_trades': 0,
            'live_trades': 0
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Setup the logger with appropriate handlers and formatters."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        logger = logging.getLogger(f"trade_pipeline.{self.name}")
        logger.setLevel(log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler
        if log_config.get('file_logging', True):
            log_dir = Path(log_config.get('log_directory', LOG_DIR))
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create specific log file for this component
            log_file = log_dir / f"{self.name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setLevel(log_level)
            
            formatter = logging.Formatter(
                log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Console handler
        if log_config.get('console_logging', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
        return logger
    
    def log_signal_generation(self, symbol: str, strategy: str, score: float, 
                            direction: str, regime: str, **kwargs):
        """Log signal generation details."""
        if self.config.get('logging', {}).get('trade_pipeline', {}).get('log_signals', True):
            self.logger.info(
                f"Signal Generated: {symbol} | Strategy: {strategy} | "
                f"Score: {score:.4f} | Direction: {direction} | Regime: {regime} | "
                f"Details: {kwargs}"
            )
    
    def log_risk_check(self, symbol: str, strategy: str, allowed: bool, reason: str, 
                      risk_score: float, **kwargs):
        """Log risk management decisions."""
        if self.config.get('logging', {}).get('trade_pipeline', {}).get('log_risk_checks', True):
            status = "ALLOWED" if allowed else "BLOCKED"
            self.logger.info(
                f"Risk Check: {symbol} | Strategy: {strategy} | "
                f"Status: {status} | Reason: {reason} | Risk Score: {risk_score:.4f} | "
                f"Details: {kwargs}"
            )
    
    def log_position_sizing(self, symbol: str, strategy: str, base_size: float, 
                           final_size: float, sentiment_boost: float, **kwargs):
        """Log position sizing calculations."""
        if self.config.get('logging', {}).get('trade_pipeline', {}).get('log_position_sizing', True):
            self.logger.info(
                f"Position Sizing: {symbol} | Strategy: {strategy} | "
                f"Base Size: {base_size:.6f} | Final Size: {final_size:.6f} | "
                f"Sentiment Boost: {sentiment_boost:.4f} | Details: {kwargs}"
            )
    
    def log_execution_decision(self, symbol: str, strategy: str, side: str, 
                              size: float, price: float, execution_mode: str, **kwargs):
        """Log trade execution decisions."""
        if self.config.get('logging', {}).get('trade_pipeline', {}).get('log_execution_decisions', True):
            mode_emoji = "📄" if execution_mode == "dry_run" else "💰"
            self.logger.info(
                f"{mode_emoji} Execution Decision: {symbol} | Strategy: {strategy} | "
                f"Side: {side} | Size: {size:.6f} | Price: {price:.6f} | "
                f"Mode: {execution_mode} | Details: {kwargs}"
            )
    
    def log_trade_execution(self, symbol: str, strategy: str, side: str, size: float, 
                           price: float, execution_mode: str, order_id: str, 
                           execution_time: float, **kwargs):
        """Log trade execution details."""
        if self.config.get('logging', {}).get('trade_pipeline', {}).get('log_executions', True):
            mode_emoji = "📄" if execution_mode == "dry_run" else "💰"
            self.trade_counters['total_trades'] += 1
            if execution_mode == "dry_run":
                self.trade_counters['paper_trades'] += 1
            else:
                self.trade_counters['live_trades'] += 1
                
            self.logger.info(
                f"{mode_emoji} Trade Executed: {symbol} | Strategy: {strategy} | "
                f"Side: {side} | Size: {size:.6f} | Price: {price:.6f} | "
                f"Order ID: {order_id} | Execution Time: {execution_time:.3f}s | "
                f"Mode: {execution_mode} | Details: {kwargs}"
            )
    
    def log_position_update(self, symbol: str, side: str, entry_price: float, 
                           size: float, strategy: str, regime: str, **kwargs):
        """Log position updates."""
        if self.config.get('logging', {}).get('trade_pipeline', {}).get('log_position_updates', True):
            self.logger.info(
                f"Position Updated: {symbol} | Side: {side} | "
                f"Entry Price: {entry_price:.6f} | Size: {size:.6f} | "
                f"Strategy: {strategy} | Regime: {regime} | Details: {kwargs}"
            )
    
    def log_balance_change(self, previous_balance: float, new_balance: float, 
                          change_amount: float, change_pct: float, execution_mode: str, **kwargs):
        """Log balance changes."""
        if self.config.get('logging', {}).get('trade_pipeline', {}).get('log_balance_changes', True):
            mode_emoji = "📄" if execution_mode == "dry_run" else "💰"
            change_emoji = "📈" if change_amount > 0 else "📉"
            self.logger.info(
                f"{mode_emoji} Balance Change: ${previous_balance:.2f} → ${new_balance:.2f} | "
                f"{change_emoji} Change: ${change_amount:.2f} ({change_pct:+.2f}%) | "
                f"Mode: {execution_mode} | Details: {kwargs}"
            )
    
    def log_paper_trading_simulation(self, action: str, symbol: str, side: str, 
                                    size: float, price: float, **kwargs):
        """Log paper trading simulation details."""
        if self.config.get('logging', {}).get('paper_trading', {}).get('log_simulated_trades', True):
            self.logger.info(
                f"📄 Paper Trading Simulation: {action} | {symbol} | "
                f"Side: {side} | Size: {size:.6f} | Price: {price:.6f} | "
                f"Details: {kwargs}"
            )
    
    def log_live_trading_execution(self, action: str, symbol: str, side: str, 
                                  size: float, price: float, order_id: str, **kwargs):
        """Log live trading execution details."""
        if self.config.get('logging', {}).get('live_trading', {}).get('log_real_executions', True):
            self.logger.info(
                f"💰 Live Trading Execution: {action} | {symbol} | "
                f"Side: {side} | Size: {size:.6f} | Price: {price:.6f} | "
                f"Order ID: {order_id} | Details: {kwargs}"
            )
    
    @contextmanager
    def execution_timer(self, operation: str):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            self.execution_times[operation] = execution_time
            
            if self.config.get('logging', {}).get('performance', {}).get('log_execution_times', True):
                self.logger.debug(f"Operation '{operation}' took {execution_time:.3f}s")
    
    def log_error(self, error: Exception, context: str, **kwargs):
        """Log errors with context."""
        if self.config.get('logging', {}).get('error_tracking', {}).get('log_exceptions', True):
            self.logger.error(
                f"Error in {context}: {str(error)} | "
                f"Traceback: {traceback.format_exc()} | Details: {kwargs}"
            )
    
    def log_api_call(self, endpoint: str, method: str, params: Dict, response_time: float, 
                     success: bool, **kwargs):
        """Log API call details."""
        if self.config.get('logging', {}).get('live_trading', {}).get('log_api_calls', True):
            status = "✅" if success else "❌"
            self.logger.debug(
                f"{status} API Call: {method} {endpoint} | "
                f"Params: {params} | Response Time: {response_time:.3f}s | "
                f"Success: {success} | Details: {kwargs}"
            )
    
    def log_websocket_event(self, event_type: str, symbol: str, data: Dict, **kwargs):
        """Log WebSocket events."""
        if self.config.get('logging', {}).get('live_trading', {}).get('log_websocket_events', True):
            self.logger.debug(
                f"📡 WebSocket Event: {event_type} | {symbol} | "
                f"Data: {data} | Details: {kwargs}"
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            'trade_counters': self.trade_counters.copy(),
            'execution_times': self.execution_times.copy(),
            'logger_name': self.name,
            'config': self.config
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.trade_counters = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'paper_trades': 0,
            'live_trades': 0
        }
        self.execution_times = {}


class EnhancedLoggingManager:
    """Manager for all enhanced loggers in the trade pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loggers = {}
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Setup loggers for different components."""
        components = [
            'signal_generation',
            'risk_management', 
            'position_sizing',
            'trade_execution',
            'position_management',
            'balance_tracking',
            'performance_monitoring'
        ]
        
        for component in components:
            self.loggers[component] = TradePipelineLogger(component, self.config)
    
    def get_logger(self, component: str) -> TradePipelineLogger:
        """Get logger for a specific component."""
        return self.loggers.get(component, self.loggers['trade_execution'])
    
    def log_trade_pipeline_event(self, component: str, event_type: str, **kwargs):
        """Log a trade pipeline event."""
        logger = self.get_logger(component)
        
        if event_type == 'signal_generation':
            logger.log_signal_generation(**kwargs)
        elif event_type == 'risk_check':
            logger.log_risk_check(**kwargs)
        elif event_type == 'position_sizing':
            logger.log_position_sizing(**kwargs)
        elif event_type == 'execution_decision':
            logger.log_execution_decision(**kwargs)
        elif event_type == 'trade_execution':
            logger.log_trade_execution(**kwargs)
        elif event_type == 'position_update':
            logger.log_position_update(**kwargs)
        elif event_type == 'balance_change':
            logger.log_balance_change(**kwargs)
        elif event_type == 'paper_trading_simulation':
            logger.log_paper_trading_simulation(**kwargs)
        elif event_type == 'live_trading_execution':
            logger.log_live_trading_execution(**kwargs)
        elif event_type == 'error':
            logger.log_error(**kwargs)
        elif event_type == 'api_call':
            logger.log_api_call(**kwargs)
        elif event_type == 'websocket_event':
            logger.log_websocket_event(**kwargs)
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all loggers."""
        return {name: logger.get_statistics() for name, logger in self.loggers.items()}
    
    def reset_all_statistics(self):
        """Reset statistics for all loggers."""
        for logger in self.loggers.values():
            logger.reset_statistics()


# Global instance
_logging_manager = None

def get_enhanced_logging_manager(config: Dict[str, Any]) -> EnhancedLoggingManager:
    """Get or create the global enhanced logging manager."""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = EnhancedLoggingManager(config)
    return _logging_manager

def log_trade_pipeline_event(component: str, event_type: str, config: Dict[str, Any], **kwargs):
    """Convenience function to log trade pipeline events."""
    manager = get_enhanced_logging_manager(config)
    manager.log_trade_pipeline_event(component, event_type, **kwargs)
