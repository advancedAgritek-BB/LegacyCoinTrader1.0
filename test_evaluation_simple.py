#!/usr/bin/env python3
"""
Simple test to verify evaluation pipeline works
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.utils.market_analyzer import analyze_symbol
import pandas as pd
import numpy as np

async def test_evaluation_pipeline():
    """Test the evaluation pipeline with simple data."""
    
    print("🧪 Testing evaluation pipeline...")
    
    # Create simple test data
    test_symbol = "BTC/USD"
    test_data = {
        "1m": pd.DataFrame({
            'timestamp': pd.date_range(start='2025-01-01', periods=100, freq='1min'),
            'open': np.random.uniform(40000, 50000, 100),
            'high': np.random.uniform(40000, 50000, 100),
            'low': np.random.uniform(40000, 50000, 100),
            'close': np.random.uniform(40000, 50000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        })
    }
    
    # Simple config
    config = {
        'min_confidence_score': 0.01,
        'execution_mode': 'dry_run',
        'testing_mode': True
    }
    
    try:
        # Test the analyze_symbol function
        result = await analyze_symbol(test_symbol, test_data, "live", config)
        
        print("✅ Evaluation pipeline test successful!")
        print(f"📊 Result: {result}")
        
        if 'signals' in result:
            print(f"📈 Generated {len(result['signals'])} signals")
            for signal in result['signals']:
                print(f"   - {signal}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_simple_bot():
    """Test a simple bot startup."""
    
    print("🤖 Testing simple bot startup...")
    
    try:
        # Import and test basic components
        from crypto_bot.utils.logger import setup_logger
        from crypto_bot.utils.market_loader import get_circuit_breaker
        
        logger = setup_logger("test", Path("crypto_bot/logs/test.log"))
        circuit_breaker = get_circuit_breaker()
        
        print("✅ Basic components loaded successfully")
        print(f"🔌 Circuit breaker state: {circuit_breaker.state}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple bot test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    
    print("🚀 Starting comprehensive evaluation pipeline tests...")
    
    # Test 1: Basic components
    basic_ok = await test_simple_bot()
    
    # Test 2: Evaluation pipeline
    eval_ok = await test_evaluation_pipeline()
    
    print("\n" + "="*50)
    print("📊 TEST RESULTS")
    print("="*50)
    print(f"Basic Components: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    print(f"Evaluation Pipeline: {'✅ PASS' if eval_ok else '❌ FAIL'}")
    
    if basic_ok and eval_ok:
        print("\n🎉 All tests passed! Evaluation pipeline is working.")
        print("📋 The issue is likely with symbol loading, not the pipeline itself.")
    else:
        print("\n❌ Some tests failed. There are deeper issues to resolve.")
    
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
