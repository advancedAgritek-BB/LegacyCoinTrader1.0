#!/usr/bin/env python3
"""
Offline Backtesting Test - Generate synthetic data and test backtesting functionality

This script tests backtesting using synthetic data to ensure the system works
without requiring external API calls.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crypto_bot.backtest.backtest_runner import BacktestConfig, BacktestRunner
from crypto_bot.utils.logger import setup_logger

def generate_synthetic_ohlcv_data(days=90, timeframe="1h"):
    """Generate realistic synthetic OHLCV data for testing."""
    
    # Calculate number of periods
    if timeframe == "1h":
        periods = days * 24
    elif timeframe == "4h":
        periods = days * 6
    elif timeframe == "1d":
        periods = days
    else:
        periods = days * 24  # Default to hourly
    
    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=periods)
    
    # Generate price data with random walk + trend
    np.random.seed(42)  # For reproducible results
    
    initial_price = 45000  # Starting price (roughly BTC price)
    returns = np.random.normal(0.0001, 0.02, periods)  # Small positive drift with volatility
    
    # Add some trend and cycles
    trend = np.linspace(0, 0.1, periods)  # 10% upward trend over period
    cycle = 0.05 * np.sin(np.linspace(0, 4*np.pi, periods))  # Some cyclical movement
    
    adjusted_returns = returns + trend/periods + cycle/periods
    
    # Calculate cumulative prices
    price_multipliers = np.cumprod(1 + adjusted_returns)
    prices = initial_price * price_multipliers
    
    # Generate OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        # Generate realistic OHLC from close price
        volatility = np.random.uniform(0.005, 0.03)  # 0.5% to 3% intraday volatility
        
        high = close * (1 + volatility * np.random.uniform(0.3, 1.0))
        low = close * (1 - volatility * np.random.uniform(0.3, 1.0))
        open_price = low + (high - low) * np.random.uniform(0.2, 0.8)
        
        # Generate volume (higher volume on larger price movements)
        price_change = abs(close - (prices[i-1] if i > 0 else close)) / close
        base_volume = np.random.uniform(100, 1000)
        volume = base_volume * (1 + price_change * 10)
        
        data.append([
            int(timestamp.timestamp() * 1000),  # Timestamp in milliseconds
            open_price,
            high,
            low,
            close,
            volume
        ])
    
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    return df

def test_single_backtest():
    """Test a single backtest with synthetic data."""
    print("🧪 Testing single backtest with synthetic data...")
    
    # Generate test data
    df = generate_synthetic_ohlcv_data(days=90, timeframe="1h")
    print(f"✅ Generated {len(df)} candles of synthetic data")
    
    # Create backtest config
    config = BacktestConfig(
        symbol="BTC/USDT",
        timeframe="1h",
        since=0,
        limit=len(df),
        stop_loss_range=[0.01, 0.02, 0.03],
        take_profit_range=[0.02, 0.04, 0.06],
        mode="cex"
    )
    
    try:
        # Run backtest with synthetic data
        runner = BacktestRunner(config, df=df)
        results = runner.run_grid()
        
        if not results.empty:
            print(f"✅ Backtest completed successfully!")
            print(f"📊 Generated {len(results)} parameter combinations")
            
            # Show best result
            best = results.iloc[0]
            print(f"🏆 Best result:")
            print(f"   Stop Loss: {best['stop_loss_pct']:.3f}")
            print(f"   Take Profit: {best['take_profit_pct']:.3f}")
            print(f"   Sharpe: {best['sharpe']:.3f}")
            print(f"   Max Drawdown: {best['max_drawdown']:.3f}")
            print(f"   Total PnL: {best['pnl']:.2f}%")
            
            return True
        else:
            print("❌ Backtest returned empty results")
            return False
            
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_strategy_optimization():
    """Test the full strategy optimization with synthetic data."""
    print("\n🧪 Testing strategy optimization with synthetic data...")
    
    try:
        # Import and patch the optimization to use synthetic data
        from crypto_bot.auto_optimizer import optimize_strategies, _load_config
        
        # Create synthetic data for the optimization
        df = generate_synthetic_ohlcv_data(days=90, timeframe="1h")
        
        # Temporarily patch the BacktestRunner to use our synthetic data
        original_init = BacktestRunner.__init__
        
        def patched_init(self, config, exchange=None, df_override=None):
            # Always use our synthetic data
            original_init(self, config, exchange=None, df=df)
        
        BacktestRunner.__init__ = patched_init
        
        try:
            results = optimize_strategies()
            
            if results:
                print(f"✅ Strategy optimization completed!")
                print(f"📊 Optimized {len(results)} strategies:")
                
                for strategy, params in results.items():
                    print(f"   {strategy}:")
                    print(f"     Stop Loss: {params.get('stop_loss_pct', 0):.3f}")
                    print(f"     Take Profit: {params.get('take_profit_pct', 0):.3f}")
                    print(f"     Sharpe: {params.get('sharpe', 0):.3f}")
                
                return True
            else:
                print("⚠️ Strategy optimization returned empty results")
                return False
                
        finally:
            # Restore original method
            BacktestRunner.__init__ = original_init
            
    except Exception as e:
        print(f"❌ Strategy optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all backtesting tests."""
    print("🤖 Offline Backtesting Test Suite")
    print("=" * 50)
    
    results = {
        "Single Backtest": test_single_backtest(),
        "Strategy Optimization": test_strategy_optimization(),
    }
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Backtesting system is working correctly.")
    else:
        print("❌ Some tests failed. Check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
