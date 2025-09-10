#!/usr/bin/env python3
"""
Diagnostic script to identify and fix trading issues.
"""

import asyncio
import json
import time
from pathlib import Path
import yaml
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from crypto_bot.utils.logger import setup_logger, LOG_DIR
from crypto_bot.utils.symbol_utils import get_filtered_symbols
from crypto_bot.utils.market_loader import load_kraken_symbols
from crypto_bot.execution.cex_executor import get_exchange

logger = setup_logger(__name__, LOG_DIR / "diagnostic.log")

async def diagnose_symbol_filtering():
    """Diagnose symbol filtering issues."""
    print("🔍 Diagnosing symbol filtering...")
    
    # Load config
    config_path = Path("crypto_bot/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"📊 Current symbol filter settings:")
    sf = config.get("symbol_filter", {})
    for key, value in sf.items():
        print(f"   {key}: {value}")
    
    # Test symbol filtering
    try:
        exchange = get_exchange(config)
        symbols = await get_filtered_symbols(exchange, config)
        print(f"✅ Symbol filtering working: {len(symbols)} symbols passed filtering")
        
        if symbols:
            print("📈 Sample symbols that passed filtering:")
            for i, (symbol, score) in enumerate(symbols[:10]):
                print(f"   {i+1}. {symbol} (score: {score:.3f})")
        else:
            print("❌ No symbols passed filtering - this is the main issue!")
            
    except Exception as e:
        print(f"❌ Symbol filtering failed: {e}")
        return False
    
    return True

async def diagnose_websocket_connection():
    """Diagnose WebSocket connection issues."""
    print("\n🔌 Diagnosing WebSocket connection...")
    
    try:
        from crypto_bot.execution.cex_executor import get_exchange
        exchange = get_exchange({"exchange": "kraken"})
        
        if hasattr(exchange, 'ws') and exchange.ws:
            print("✅ WebSocket connection available")
            return True
        else:
            print("⚠️ WebSocket not available, using REST API")
            return True
    except Exception as e:
        print(f"❌ WebSocket diagnosis failed: {e}")
        return False

async def diagnose_strategy_evaluation():
    """Diagnose strategy evaluation issues."""
    print("\n🧠 Diagnosing strategy evaluation...")
    
    try:
        from crypto_bot.strategy_router import get_strategies_for_regime
        from crypto_bot.utils.market_analyzer import analyze_symbol
        
        # Test strategy availability
        strategies = get_strategies_for_regime("trending")
        print(f"✅ Found {len(strategies)} strategies for trending regime")
        
        # Test with sample data
        import pandas as pd
        import numpy as np
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2025-01-01', periods=100, freq='15min')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(200, 300, 100),
            'low': np.random.uniform(50, 100, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        df_map = {"15m": sample_data}
        result = await analyze_symbol("BTC/USD", df_map, "cex", {})
        
        print(f"✅ Strategy evaluation working: {result.get('regime', 'unknown')} regime, confidence: {result.get('score', 0):.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Strategy evaluation failed: {e}")
        return False

async def diagnose_data_quality():
    """Diagnose data quality issues."""
    print("\n📊 Diagnosing data quality...")
    
    try:
        from crypto_bot.utils.market_loader import update_ohlcv_cache
        
        exchange = get_exchange({"exchange": "kraken"})
        cache = {}
        
        # Test with a few major symbols
        test_symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        
        for symbol in test_symbols:
            try:
                await update_ohlcv_cache(exchange, cache, [symbol], "15m")
                if symbol in cache:
                    df = cache[symbol]
                    print(f"✅ {symbol}: {len(df)} candles, latest: {df.index[-1]}")
                else:
                    print(f"❌ {symbol}: No data available")
            except Exception as e:
                print(f"❌ {symbol}: Failed to fetch data - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data quality diagnosis failed: {e}")
        return False

async def main():
    """Run all diagnostics."""
    print("🚀 Starting trading system diagnostics...\n")
    
    results = {}
    
    # Run diagnostics
    results['symbol_filtering'] = await diagnose_symbol_filtering()
    results['websocket'] = await diagnose_websocket_connection()
    results['strategy_evaluation'] = await diagnose_strategy_evaluation()
    results['data_quality'] = await diagnose_data_quality()
    
    # Summary
    print("\n" + "="*50)
    print("📋 DIAGNOSTIC SUMMARY")
    print("="*50)
    
    all_passed = True
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test.replace('_', ' ').title()}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All diagnostics passed! Your system should be working.")
        print("💡 If you're still not getting trades, try:")
        print("   1. Restart the bot with: ./start_bot_auto.py")
        print("   2. Check the logs for specific errors")
        print("   3. Monitor the frontend dashboard")
    else:
        print("⚠️ Some issues detected. Please check the specific failures above.")
        print("💡 Recommended fixes:")
        print("   1. Clear cache files: rm -f crypto_bot/logs/*.log")
        print("   2. Reset paper wallet: python3 reset_paper_wallet.py")
        print("   3. Restart the bot: ./start_bot_auto.py")

if __name__ == "__main__":
    asyncio.run(main())
