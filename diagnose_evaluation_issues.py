#!/usr/bin/env python3
"""
Diagnostic script to identify specific evaluation pipeline issues.
This will help pinpoint exactly what's wrong with the evaluation system.
"""

import asyncio
import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.utils.logger import setup_logger, LOG_DIR

# Setup logging
logger = setup_logger("evaluation_diagnostic", LOG_DIR / "evaluation_diagnostic.log")

class EvaluationDiagnostic:
    """Diagnostic tool for evaluation pipeline issues."""
    
    def __init__(self):
        self.config = self.load_config()
        self.issues = []
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration."""
        config_path = project_root / "crypto_bot" / "config.yaml"
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded: {len(config)} keys")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def check_configuration(self):
        """Check configuration for potential issues."""
        logger.info("🔍 Checking configuration...")
        
        # Check critical configuration parameters
        critical_params = [
            "min_confidence_score",
            "timeframe", 
            "symbols",
            "execution_mode",
            "strategy_evaluation_mode"
        ]
        
        for param in critical_params:
            value = self.config.get(param)
            if value is None:
                self.issues.append(f"Missing configuration: {param}")
                logger.warning(f"❌ Missing: {param}")
            else:
                logger.info(f"✅ {param}: {value}")
        
        # Check API configurations
        api_configs = ["kraken", "coinbase", "pyth", "raydium"]
        for api in api_configs:
            api_config = self.config.get(api, {})
            if api_config.get("enabled", False):
                if not api_config.get("api_key"):
                    self.issues.append(f"API enabled but no key: {api}")
                    logger.warning(f"⚠️ {api}: enabled but no API key")
                else:
                    logger.info(f"✅ {api}: configured")
            else:
                logger.info(f"ℹ️ {api}: disabled")
    
    def check_logs_for_errors(self):
        """Analyze recent logs for error patterns."""
        logger.info("📋 Analyzing recent logs...")
        
        # Check bot.log for recent errors
        bot_log_path = LOG_DIR / "bot.log"
        if bot_log_path.exists():
            try:
                # Get last 1000 lines
                with open(bot_log_path, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-1000:] if len(lines) > 1000 else lines
                
                # Count error types
                error_counts = {}
                for line in recent_lines:
                    if "ERROR" in line:
                        # Extract error type
                        if "Strategy generate_signal failed" in line:
                            error_type = "strategy_execution"
                        elif "Circuit breaker is OPEN" in line:
                            error_type = "circuit_breaker"
                        elif "401 Unauthorized" in line:
                            error_type = "api_auth"
                        elif "Failed to load OHLCV" in line:
                            error_type = "data_fetch"
                        else:
                            error_type = "other"
                        
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
                logger.info("📊 Recent Error Analysis:")
                for error_type, count in error_counts.items():
                    logger.info(f"   {error_type}: {count} occurrences")
                    if count > 10:
                        self.issues.append(f"High {error_type} error rate: {count} occurrences")
                
            except Exception as e:
                logger.error(f"Error reading bot.log: {e}")
        else:
            logger.warning("bot.log not found")
    
    def check_circuit_breakers(self):
        """Check circuit breaker status."""
        logger.info("🔌 Checking circuit breakers...")
        
        circuit_breaker_log_path = LOG_DIR / "circuit_breaker.log"
        if circuit_breaker_log_path.exists():
            try:
                with open(circuit_breaker_log_path, 'r') as f:
                    lines = f.readlines()
                    recent_lines = lines[-50:] if len(lines) > 50 else lines
                
                open_breakers = []
                for line in recent_lines:
                    if "OPEN" in line:
                        # Extract breaker name
                        if "kraken" in line.lower():
                            open_breakers.append("kraken")
                        elif "coinbase" in line.lower():
                            open_breakers.append("coinbase")
                        elif "pyth" in line.lower():
                            open_breakers.append("pyth")
                
                if open_breakers:
                    logger.warning(f"⚠️ Open circuit breakers: {open_breakers}")
                    self.issues.append(f"Circuit breakers open: {open_breakers}")
                else:
                    logger.info("✅ All circuit breakers appear to be closed")
                    
            except Exception as e:
                logger.error(f"Error reading circuit_breaker.log: {e}")
    
    def check_data_quality(self):
        """Check data quality issues."""
        logger.info("📊 Checking data quality...")
        
        # Check if there are any recent successful data fetches
        try:
            # Look for successful OHLCV fetches in recent logs
            with open(LOG_DIR / "bot.log", 'r') as f:
                lines = f.readlines()
                recent_lines = lines[-500:] if len(lines) > 500 else lines
            
            successful_fetches = 0
            failed_fetches = 0
            
            for line in recent_lines:
                if "Fetched OHLCV" in line:
                    successful_fetches += 1
                elif "Failed to load OHLCV" in line:
                    failed_fetches += 1
            
            logger.info(f"📈 Data fetch success rate: {successful_fetches}/{successful_fetches + failed_fetches}")
            
            if failed_fetches > successful_fetches:
                self.issues.append("High data fetch failure rate")
                logger.warning("⚠️ More failed fetches than successful ones")
            elif successful_fetches == 0:
                self.issues.append("No successful data fetches")
                logger.error("❌ No successful data fetches detected")
            else:
                logger.info("✅ Data fetching appears to be working")
                
        except Exception as e:
            logger.error(f"Error checking data quality: {e}")
    
    def check_strategy_execution(self):
        """Check strategy execution issues."""
        logger.info("🎯 Checking strategy execution...")
        
        try:
            with open(LOG_DIR / "bot.log", 'r') as f:
                lines = f.readlines()
                recent_lines = lines[-1000:] if len(lines) > 1000 else lines
            
            strategy_errors = 0
            event_loop_errors = 0
            numpy_errors = 0
            
            for line in recent_lines:
                if "Strategy generate_signal failed" in line:
                    strategy_errors += 1
                    if "event loop" in line.lower():
                        event_loop_errors += 1
                    elif "numpy.ndarray" in line:
                        numpy_errors += 1
            
            logger.info(f"🔧 Strategy execution errors: {strategy_errors}")
            logger.info(f"   Event loop errors: {event_loop_errors}")
            logger.info(f"   Numpy array errors: {numpy_errors}")
            
            if strategy_errors > 0:
                self.issues.append(f"Strategy execution errors: {strategy_errors}")
                if event_loop_errors > 0:
                    self.issues.append("Event loop conflicts detected")
                if numpy_errors > 0:
                    self.issues.append("Data type issues with numpy arrays")
            
        except Exception as e:
            logger.error(f"Error checking strategy execution: {e}")
    
    def check_signal_generation(self):
        """Check signal generation status."""
        logger.info("📡 Checking signal generation...")
        
        try:
            with open(LOG_DIR / "bot.log", 'r') as f:
                lines = f.readlines()
                recent_lines = lines[-500:] if len(lines) > 500 else lines
            
            analysis_results = 0
            actionable_signals = 0
            
            for line in recent_lines:
                if "analysis completed" in line:
                    analysis_results += 1
                elif "actionable signals" in line:
                    # Extract number of signals
                    if "0 actionable signals" in line:
                        actionable_signals += 1
            
            logger.info(f"📊 Analysis cycles: {analysis_results}")
            logger.info(f"📊 Cycles with 0 signals: {actionable_signals}")
            
            if analysis_results > 0:
                signal_rate = (analysis_results - actionable_signals) / analysis_results * 100
                logger.info(f"📊 Signal generation rate: {signal_rate:.1f}%")
                
                if signal_rate == 0:
                    self.issues.append("No signals generated in recent cycles")
                    logger.warning("⚠️ No signals generated in recent analysis cycles")
                elif signal_rate < 5:
                    self.issues.append(f"Low signal generation rate: {signal_rate:.1f}%")
                    logger.warning(f"⚠️ Low signal generation rate: {signal_rate:.1f}%")
                else:
                    logger.info("✅ Signal generation appears normal")
            else:
                logger.warning("⚠️ No recent analysis cycles found")
                
        except Exception as e:
            logger.error(f"Error checking signal generation: {e}")
    
    def generate_report(self):
        """Generate comprehensive diagnostic report."""
        logger.info("📋 Generating diagnostic report...")
        
        print("\n" + "="*60)
        print("🔍 EVALUATION PIPELINE DIAGNOSTIC REPORT")
        print("="*60)
        
        # Run all checks
        self.check_configuration()
        self.check_logs_for_errors()
        self.check_circuit_breakers()
        self.check_data_quality()
        self.check_strategy_execution()
        self.check_signal_generation()
        
        # Summary
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        
        if not self.issues:
            print("✅ No critical issues detected!")
            print("🎉 Your evaluation pipeline appears to be working correctly.")
        else:
            print(f"❌ Found {len(self.issues)} issues:")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
            
            print("\n🔧 RECOMMENDED ACTIONS:")
            print("   1. Run fix_evaluation_pipeline.py to apply automatic fixes")
            print("   2. Check API keys and authentication")
            print("   3. Restart the trading bot")
            print("   4. Monitor logs for improvement")
        
        print("\n" + "="*60)
        
        return len(self.issues) == 0

def main():
    """Main diagnostic function."""
    diagnostic = EvaluationDiagnostic()
    success = diagnostic.generate_report()
    
    if success:
        print("✅ Diagnostic completed successfully!")
    else:
        print("⚠️ Issues detected - review the report above")

if __name__ == "__main__":
    main()
