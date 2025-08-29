#!/usr/bin/env python3
"""
Test script for the Enhanced Backtesting System

This script tests the basic functionality of the enhanced backtesting system
to ensure it's working correctly before running full backtests.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_gpu_detection():
    """Test GPU detection functionality."""
    print("Testing GPU detection...")
    
    try:
        from crypto_bot.backtest.gpu_accelerator import get_gpu_info, is_gpu_available
        
        gpu_available = is_gpu_available()
        gpu_info = get_gpu_info()
        
        print(f"GPU Available: {gpu_available}")
        print(f"GPU Info: {gpu_info}")
        
        if gpu_available:
            print("✅ GPU detection working")
        else:
            print("⚠️  No GPU detected - will use CPU fallback")
            
        return True
        
    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        return False

def test_enhanced_backtester():
    """Test enhanced backtester creation."""
    print("\nTesting enhanced backtester creation...")
    
    try:
        from crypto_bot.backtest.enhanced_backtester import create_enhanced_backtester
        
        config = {
            'top_pairs_count': 5,  # Small number for testing
            'use_gpu': True,
            'timeframes': ['1h'],
            'batch_size': 10
        }
        
        engine = create_enhanced_backtester(config)
        
        print(f"Engine created successfully")
        print(f"Available strategies: {len(engine.get_all_strategies())}")
        print(f"Configuration: {engine.config}")
        
        print("✅ Enhanced backtester creation working")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced backtester creation failed: {e}")
        return False

async def test_single_backtest():
    """Test single backtest functionality."""
    print("\nTesting single backtest...")
    
    try:
        from crypto_bot.backtest.enhanced_backtester import run_backtest_analysis
        
        # Test with a small set of pairs and strategies
        pairs = ["BTC/USDT", "ETH/USDT"]
        strategies = ["trend_bot", "momentum_bot"]
        timeframes = ["1h"]
        
        config = {
            'use_gpu': False,  # Disable GPU for testing
            'batch_size': 5
        }
        
        print(f"Running backtest for {len(pairs)} pairs, {len(strategies)} strategies, {len(timeframes)} timeframes")
        
        results = await run_backtest_analysis(pairs, strategies, timeframes, config)
        
        print(f"Backtest completed with {len(results)} pair results")
        
        # Print summary
        total_tests = 0
        successful_tests = 0
        
        for pair, pair_results in results.items():
            for strategy, strategy_results in pair_results.items():
                for timeframe, timeframe_results in strategy_results.items():
                    total_tests += 1
                    if timeframe_results:
                        successful_tests += 1
                        print(f"  {pair} - {strategy} ({timeframe}): {len(timeframe_results)} results")
                    else:
                        print(f"  {pair} - {strategy} ({timeframe}): No results")
        
        print(f"\nTotal tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
        
        if successful_tests > 0:
            print("✅ Single backtest working")
            return True
        else:
            print("⚠️  Backtest ran but no results generated")
            return False
            
    except Exception as e:
        print(f"❌ Single backtest failed: {e}")
        return False

def test_cli_imports():
    """Test CLI module imports."""
    print("\nTesting CLI imports...")
    
    try:
        from crypto_bot.backtest.cli import setup_logging, load_config, print_results_summary
        
        print("✅ CLI imports working")
        return True
        
    except Exception as e:
        print(f"❌ CLI imports failed: {e}")
        return False

def test_config_loading():
    """Test configuration file loading."""
    print("\nTesting configuration loading...")
    
    try:
        config_path = Path("config/backtest_config.yaml")
        
        if config_path.exists():
            from crypto_bot.backtest.cli import load_config
            config = load_config(str(config_path))
            
            print(f"Configuration loaded successfully")
            print(f"Top pairs count: {config.get('top_pairs_count', 'Not set')}")
            print(f"GPU enabled: {config.get('use_gpu', 'Not set')}")
            
            print("✅ Configuration loading working")
            return True
        else:
            print(f"⚠️  Configuration file not found at {config_path}")
            return False
            
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("Enhanced Backtesting System - Test Suite")
    print("=" * 50)
    
    tests = [
        ("GPU Detection", test_gpu_detection),
        ("Enhanced Backtester", test_enhanced_backtester),
        ("Single Backtest", test_single_backtest),
        ("CLI Imports", test_cli_imports),
        ("Config Loading", test_config_loading),
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
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25s}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced backtesting system is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest suite failed with exception: {e}")
        sys.exit(1)
