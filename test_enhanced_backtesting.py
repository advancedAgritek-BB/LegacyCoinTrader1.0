#!/usr/bin/env python3
"""
Test script for the Enhanced Backtesting System

This script tests the basic functionality of the enhanced backtesting system
to ensure it's working correctly before running full backtests.
"""

import asyncio
import logging
import sys
import pytest
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_gpu_detection():
    """Test GPU detection functionality."""

    from crypto_bot.backtest.gpu_accelerator import get_gpu_info, is_gpu_available

    gpu_available = is_gpu_available()
    gpu_info = get_gpu_info()

    assert isinstance(gpu_available, bool), "GPU availability should be boolean"
    assert isinstance(gpu_info, dict), "GPU info should be a dictionary"

    # GPU info should at least contain status or basic keys
    assert 'status' in gpu_info or 'gpu_count' in gpu_info, "GPU info should contain status or gpu_count"

def test_enhanced_backtester():
    """Test enhanced backtester creation."""

    from crypto_bot.backtest.enhanced_backtester import create_enhanced_backtester

    config = {
        'top_pairs_count': 5,  # Small number for testing
        'use_gpu': True,
        'timeframes': ['1h'],
        'batch_size': 10
    }

    engine = create_enhanced_backtester(config)

    assert engine is not None, "Enhanced backtester should be created successfully"
    assert hasattr(engine, 'get_all_strategies'), "Engine should have get_all_strategies method"
    assert hasattr(engine, 'config'), "Engine should have config attribute"
    assert engine.config is not None, "Engine config should not be None"

@pytest.mark.asyncio
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

    from crypto_bot.backtest.cli import setup_logging, load_config, print_results_summary

    # Verify the imported functions are callable
    assert callable(setup_logging), "setup_logging should be callable"
    assert callable(load_config), "load_config should be callable"
    assert callable(print_results_summary), "print_results_summary should be callable"

def test_config_loading():
    """Test configuration file loading."""

    config_path = Path("config/backtest_config.yaml")

    if config_path.exists():
        from crypto_bot.backtest.cli import load_config
        config = load_config(str(config_path))

        assert isinstance(config, dict), "Loaded config should be a dictionary"
        assert 'top_pairs_count' in config, "Config should contain top_pairs_count"
        assert 'use_gpu' in config, "Config should contain use_gpu"
    else:
        # If config file doesn't exist, skip the test
        pytest.skip(f"Configuration file not found at {config_path}")


