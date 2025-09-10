#!/usr/bin/env python3
"""
Test script to verify strategy utilization fixes.

This script tests the updated regime detection and strategy routing to ensure
that bounce and scalp regimes are now being detected and all strategies are
being utilized properly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from crypto_bot.regime.regime_classifier import classify_regime, _ALL_REGIMES
from crypto_bot.strategy_router import route, get_strategies_for_regime
from crypto_bot.utils.strategy_performance_tracker import StrategyPerformanceTracker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(regime_type: str, num_candles: int = 200) -> pd.DataFrame:
    """Create test data that should trigger specific regime detection."""
    
    # Base data
    dates = pd.date_range(start='2024-01-01', periods=num_candles, freq='1h')
    
    if regime_type == "bounce":
        # Create oversold conditions with reversal
        close_prices = np.linspace(100, 80, num_candles//2)  # Downtrend
        close_prices = np.concatenate([close_prices, np.linspace(80, 95, num_candles//2)])  # Reversal
        volumes = np.random.uniform(1000, 2000, num_candles)
        volumes[-10:] *= 2  # Volume spike at end
        
    elif regime_type == "scalp":
        # Create low volatility, high frequency conditions
        close_prices = 100 + np.cumsum(np.random.normal(0, 0.1, num_candles))
        volumes = np.random.uniform(1500, 3000, num_candles)
        
    elif regime_type == "breakout":
        # Create breakout conditions
        close_prices = np.linspace(100, 110, num_candles)
        close_prices[-20:] += np.random.normal(0, 2, 20)  # Volatile breakout
        volumes = np.random.uniform(1000, 2000, num_candles)
        volumes[-10:] *= 3  # High volume
        
    else:
        # Default trending data
        close_prices = np.linspace(100, 120, num_candles)
        volumes = np.random.uniform(1000, 2000, num_candles)
    
    # Create OHLCV data
    high = close_prices + np.random.uniform(0, 2, num_candles)
    low = close_prices - np.random.uniform(0, 2, num_candles)
    open_prices = close_prices + np.random.uniform(-1, 1, num_candles)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close_prices,
        'volume': volumes
    })
    
    return df

def test_regime_detection():
    """Test that all regimes including bounce and scalp are being detected."""
    logger.info("Testing regime detection...")
    
    regimes_detected = set()
    
    for regime_type in ["bounce", "scalp", "breakout", "trending", "sideways", "mean-reverting", "volatile"]:
        logger.info(f"Testing {regime_type} regime detection...")
        
        # Create test data
        df = create_test_data(regime_type)
        
        # Classify regime
        try:
            regime, probabilities = classify_regime(df)
            regimes_detected.add(regime)
            logger.info(f"  Detected regime: {regime}")
            logger.info(f"  Probabilities: {probabilities}")
        except Exception as e:
            logger.error(f"  Error detecting {regime_type} regime: {e}")
    
    # Check if all expected regimes are detected
    expected_regimes = {"bounce", "scalp", "breakout", "trending", "sideways", "mean-reverting", "volatile"}
    missing_regimes = expected_regimes - regimes_detected
    
    if missing_regimes:
        logger.warning(f"Missing regimes: {missing_regimes}")
    else:
        logger.info("✅ All expected regimes detected!")
    
    return len(missing_regimes) == 0

def test_strategy_routing():
    """Test that strategies are being routed correctly for all regimes."""
    logger.info("Testing strategy routing...")
    
    strategies_used = set()
    
    for regime in ["bounce", "scalp", "breakout", "trending", "sideways", "mean-reverting", "volatile"]:
        logger.info(f"Testing strategy routing for {regime} regime...")
        
        try:
            # Get strategies for regime
            strategies = get_strategies_for_regime(regime, {})
            if strategies:
                for strategy in strategies:
                    strategies_used.add(strategy.__name__)
                logger.info(f"  Strategies: {[s.__name__ for s in strategies]}")
            else:
                logger.warning(f"  No strategies found for {regime} regime")
        except Exception as e:
            logger.error(f"  Error routing strategies for {regime}: {e}")
    
    logger.info(f"Total strategies used: {len(strategies_used)}")
    logger.info(f"Strategies: {sorted(strategies_used)}")
    
    return len(strategies_used) > 0

def test_strategy_rotation():
    """Test that strategy rotation is working."""
    logger.info("Testing strategy rotation...")
    
    # Create test data
    df = create_test_data("breakout")
    
    # Test multiple calls to see if different strategies are selected
    strategies_selected = []
    
    for i in range(10):
        try:
            strategy_fn = route("breakout", "auto", {}, df_map=df)
            strategy_name = strategy_fn.__name__ if hasattr(strategy_fn, '__name__') else str(strategy_fn)
            strategies_selected.append(strategy_name)
        except Exception as e:
            logger.error(f"Error in strategy rotation test {i}: {e}")
    
    unique_strategies = set(strategies_selected)
    logger.info(f"Selected strategies: {strategies_selected}")
    logger.info(f"Unique strategies: {unique_strategies}")
    
    # Check if multiple strategies were used
    if len(unique_strategies) > 1:
        logger.info("✅ Strategy rotation working - multiple strategies selected!")
        return True
    else:
        logger.warning("⚠️ Strategy rotation may not be working - only one strategy selected")
        return False

def test_performance_tracking():
    """Test the performance tracking system."""
    logger.info("Testing performance tracking...")
    
    tracker = StrategyPerformanceTracker()
    
    # Record some test usage
    tracker.record_strategy_usage("test_strategy", "bounce", 0.8, "long")
    tracker.record_strategy_usage("test_strategy", "scalp", 0.6, "short")
    tracker.record_strategy_usage("another_strategy", "breakout", 0.9, "long")
    
    # Record trade results
    tracker.record_trade_result("test_strategy", 0.05, True)
    tracker.record_trade_result("test_strategy", -0.02, False)
    tracker.record_trade_result("another_strategy", 0.08, True)
    
    # Get recommendations
    recommendations = tracker.get_strategy_recommendations()
    logger.info(f"Recommendations: {recommendations}")
    
    # Generate report
    report = tracker.generate_performance_report()
    logger.info("Performance Report:")
    logger.info(report)
    
    return True

def main():
    """Run all tests."""
    logger.info("Starting strategy utilization fix tests...")
    
    tests = [
        ("Regime Detection", test_regime_detection),
        ("Strategy Routing", test_strategy_routing),
        ("Strategy Rotation", test_strategy_rotation),
        ("Performance Tracking", test_performance_tracking),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("Test Results Summary:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("🎉 All tests passed! Strategy utilization fixes are working.")
    else:
        logger.warning("⚠️ Some tests failed. Please review the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
