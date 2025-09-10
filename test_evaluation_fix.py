#!/usr/bin/env python3
"""
Test script to verify the evaluation pipeline fix.
This will help diagnose why no trades are being generated.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.utils.logger import setup_logger, LOG_DIR
from crypto_bot.utils.market_analyzer import analyze_symbol
from crypto_bot.phase_runner import BotContext
import yaml
import pandas as pd
import numpy as np

# Setup logging
logger = setup_logger("evaluation_fix_test", LOG_DIR / "evaluation_fix_test.log")

async def test_evaluation_pipeline():
    """Test the evaluation pipeline to see if signals are being generated."""
    
    # Load configuration
    config_path = project_root / "crypto_bot" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Testing evaluation pipeline fix...")
    logger.info(f"Configuration loaded: {len(config)} keys")
    
    # Check key configuration parameters
    logger.info(f"min_confidence_score: {config.get('min_confidence_score', 'NOT SET')}")
    logger.info(f"timeframe: {config.get('timeframe', 'NOT SET')}")
    logger.info(f"symbols count: {len(config.get('symbols', []))}")
    logger.info(f"execution_mode: {config.get('execution_mode', 'NOT SET')}")
    logger.info(f"use_enhanced_ohlcv_fetcher: {config.get('use_enhanced_ohlcv_fetcher', 'NOT SET')}")
    logger.info(f"max_concurrent_ohlcv: {config.get('max_concurrent_ohlcv', 'NOT SET')}")
    logger.info(f"cycle_delay_seconds: {config.get('cycle_delay_seconds', 'NOT SET')}")
    
    # Create a mock BotContext
    ctx = BotContext(
        positions={},
        df_cache={},
        regime_cache={},
        config=config
    )
    
    # Test with a few sample symbols
    test_symbols = ["BTC/USD", "ETH/USD", "ADA/USD"]
    
    for symbol in test_symbols:
        logger.info(f"\nTesting symbol: {symbol}")

        # Create mock OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic price data
        base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 0.5
        returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        ohlcv_data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Add some realistic OHLC variation
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.uniform(1000, 10000)
            
            ohlcv_data.append([
                int(date.timestamp() * 1000),  # timestamp in milliseconds
                open_price,
                high,
                low,
                price,
                volume
            ])
        
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['return'] = df['close'].pct_change()
        
        # Create timeframe mapping
        df_map = {'1h': df, '4h': df, '1d': df}
        
        try:
            # Test the analysis
            result = await analyze_symbol(symbol, df_map, "auto", config, None)
            
            logger.info(f"Analysis result for {symbol}:")
            logger.info(f"  Skip: {result.get('skip', 'No')}")
            logger.info(f"  Regime: {result.get('regime', 'Unknown')}")
            logger.info(f"  Score: {result.get('score', 0.0):.4f}")
            logger.info(f"  Direction: {result.get('direction', 'none')}")
            logger.info(f"  Confidence: {result.get('confidence', 0.0):.4f}")
            logger.info(f"  Min Confidence: {result.get('min_confidence', 0.0):.4f}")
            
            # Check if this would generate a signal
            if result.get('skip'):
                logger.info(f"  ❌ Skipped: {result.get('skip')}")
            elif result.get('direction') == 'none':
                logger.info(f"  ❌ No direction signal")
            elif result.get('score', 0.0) < config.get('min_confidence_score', 0.0):
                logger.info(f"  ❌ Score too low: {result.get('score', 0.0):.4f} < {config.get('min_confidence_score', 0.0)}")
            else:
                logger.info(f"  ✅ Would generate signal!")
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("\nEvaluation pipeline test completed.")

if __name__ == "__main__":
    asyncio.run(test_evaluation_pipeline())
