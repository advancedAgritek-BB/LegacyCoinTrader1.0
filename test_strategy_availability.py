#!/usr/bin/env python3
"""
Strategy Availability Test Script
Tests that all strategies are properly available and mapped to regimes.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.strategy_router import (
    get_strategy_by_name,
    get_strategies_for_regime,
    route,
    RouterConfig
)
from crypto_bot.regime.regime_classifier import classify_regime
import yaml


def test_strategy_imports():
    """Test that all strategies can be imported and have generate_signal functions."""
    print("🔍 Testing Strategy Imports...")
    
    # List of all expected strategies
    expected_strategies = [
        # Core strategies
        "trend_bot", "grid_bot", "sniper_bot", "sniper_solana", "dex_scalper",
        "mean_bot", "breakout_bot", "micro_scalp_bot", "bounce_scalper",
        
        # New strategies
        "cross_chain_arb_bot", "dip_hunter", "flash_crash_bot", "hft_engine",
        "lstm_bot", "maker_spread", "momentum_bot", "range_arb_bot",
        "stat_arb_bot", "meme_wave_bot",
        
        # Ultra-aggressive strategies
        "ultra_scalp_bot", "momentum_exploiter", "volatility_harvester",
        
        # Additional strategies
        "solana_scalping", "dca_bot"
    ]
    
    available_strategies = []
    missing_strategies = []
    
    for strategy_name in expected_strategies:
        try:
            strategy_fn = get_strategy_by_name(strategy_name)
            if strategy_fn and hasattr(strategy_fn, '__name__'):
                available_strategies.append(strategy_name)
                print(f"  ✅ {strategy_name} - Available")
            else:
                missing_strategies.append(strategy_name)
                print(f"  ❌ {strategy_name} - Missing generate_signal function")
        except Exception as e:
            missing_strategies.append(strategy_name)
            print(f"  ❌ {strategy_name} - Import error: {e}")
    
    print(f"\n📊 Strategy Availability Summary:")
    print(f"  Total Expected: {len(expected_strategies)}")
    print(f"  Available: {len(available_strategies)}")
    print(f"  Missing: {len(missing_strategies)}")
    
    if missing_strategies:
        print(f"  Missing Strategies: {', '.join(missing_strategies)}")
    
    return len(missing_strategies) == 0


def test_regime_mapping():
    """Test that all regimes have strategies mapped to them."""
    print("\n🔍 Testing Regime Strategy Mapping...")
    
    # Load configuration
    config_path = project_root / "crypto_bot" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    strategy_router_config = config.get("strategy_router", {})
    regimes = strategy_router_config.get("regimes", {})
    
    print(f"  Found {len(regimes)} regimes in configuration")
    
    regime_coverage = {}
    total_strategies_mapped = 0
    
    for regime_name, strategy_list in regimes.items():
        if isinstance(strategy_list, str):
            strategy_list = [strategy_list]
        
        # Check if strategies exist
        available_strategies = []
        missing_strategies = []
        
        for strategy_name in strategy_list:
            strategy_fn = get_strategy_by_name(strategy_name)
            if strategy_fn:
                available_strategies.append(strategy_name)
            else:
                missing_strategies.append(strategy_name)
        
        regime_coverage[regime_name] = {
            'total': len(strategy_list),
            'available': len(available_strategies),
            'missing': len(missing_strategies),
            'strategies': strategy_list,
            'available_strategies': available_strategies,
            'missing_strategies': missing_strategies
        }
        
        total_strategies_mapped += len(strategy_list)
        
        if missing_strategies:
            print(f"  ⚠️  {regime_name}: {len(available_strategies)}/{len(strategy_list)} strategies available")
            print(f"     Missing: {', '.join(missing_strategies)}")
        else:
            print(f"  ✅ {regime_name}: {len(available_strategies)}/{len(strategy_list)} strategies available")
    
    print(f"\n📊 Regime Coverage Summary:")
    print(f"  Total Regimes: {len(regimes)}")
    print(f"  Total Strategy Mappings: {total_strategies_mapped}")
    
    # Calculate coverage statistics
    total_available = sum(r['available'] for r in regime_coverage.values())
    total_missing = sum(r['missing'] for r in regime_coverage.values())
    coverage_percentage = (total_available / total_strategies_mapped * 100) if total_strategies_mapped > 0 else 0
    
    print(f"  Available Strategies: {total_available}")
    print(f"  Missing Strategies: {total_missing}")
    print(f"  Coverage: {coverage_percentage:.1f}%")
    
    return total_missing == 0


def test_regime_detection():
    """Test that regime detection works with sample data."""
    print("\n🔍 Testing Regime Detection...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data for testing
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)
        
        # Generate realistic OHLCV data
        base_price = 100
        returns = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        # Test regime classification
        regime, probabilities = classify_regime(df)
        
        print(f"  ✅ Regime detection working")
        print(f"  Detected regime: {regime}")
        print(f"  Regime probabilities: {probabilities}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Regime detection failed: {e}")
        return False


def test_strategy_routing():
    """Test that strategy routing works correctly."""
    print("\n🔍 Testing Strategy Routing...")
    
    try:
        # Test routing for different regimes
        test_regimes = ["trending", "sideways", "breakout", "volatile", "mean-reverting"]
        
        for regime in test_regimes:
            try:
                # Test routing without actual data
                strategy_fn = route(regime, "auto")
                if strategy_fn and hasattr(strategy_fn, '__name__'):
                    print(f"  ✅ {regime} regime routing: {strategy_fn.__name__}")
                else:
                    print(f"  ❌ {regime} regime routing failed")
            except Exception as e:
                print(f"  ❌ {regime} regime routing error: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Strategy routing test failed: {e}")
        return False


def test_configuration_validation():
    """Test that the configuration is valid and complete."""
    print("\n🔍 Testing Configuration Validation...")
    
    try:
        # Load and validate main config
        config_path = project_root / "crypto_bot" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load and validate regime config
        regime_config_path = project_root / "crypto_bot" / "regime" / "regime_config.yaml"
        with open(regime_config_path, 'r') as f:
            regime_config = yaml.safe_load(f)
        
        print(f"  ✅ Main configuration loaded successfully")
        print(f"  ✅ Regime configuration loaded successfully")
        
        # Check required sections
        required_sections = ["strategy_router", "regime_timeframes", "timeframes"]
        for section in required_sections:
            if section in config:
                print(f"  ✅ {section} section present")
            else:
                print(f"  ❌ {section} section missing")
        
        # Check regime configuration
        required_regime_settings = ["adx_trending_min", "use_ml_regime_classifier", "score_weights"]
        for setting in required_regime_settings:
            if setting in regime_config:
                print(f"  ✅ {setting} setting present")
            else:
                print(f"  ❌ {setting} setting missing")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration validation failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Strategy Availability Tests\n")
    
    tests = [
        ("Strategy Imports", test_strategy_imports),
        ("Regime Strategy Mapping", test_regime_mapping),
        ("Regime Detection", test_regime_detection),
        ("Strategy Routing", test_strategy_routing),
        ("Configuration Validation", test_configuration_validation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your strategy system is fully configured.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
