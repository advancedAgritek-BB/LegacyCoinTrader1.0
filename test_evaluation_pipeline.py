#!/usr/bin/env python3
"""
Test script to verify the evaluation pipeline is working correctly.
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

# Setup logging
logger = setup_logger("evaluation_test", LOG_DIR / "evaluation_test.log")

async def test_evaluation_pipeline():
    """Test the evaluation pipeline to see if signals are being generated."""
    
    # Load configuration
    config_path = project_root / "crypto_bot" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Testing evaluation pipeline...")
    logger.info(f"Configuration loaded: {len(config)} keys")
    
    # Check key configuration parameters
    logger.info(f"min_confidence_score: {config.get('min_confidence_score', 'NOT SET')}")
    logger.info(f"timeframe: {config.get('timeframe', 'NOT SET')}")
    logger.info(f"symbols count: {len(config.get('symbols', []))}")
    logger.info(f"execution_mode: {config.get('execution_mode', 'NOT SET')}")
    
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
        
        # Create mock data for testing
        import numpy as np
        dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')
        mock_data = {
            'timestamp': [int(d.timestamp() * 1000) for d in dates],  # Convert to milliseconds timestamp
            'open': np.random.uniform(40000, 50000, 100),
            'high': np.random.uniform(40000, 50000, 100),
            'low': np.random.uniform(40000, 50000, 100),
            'close': np.random.uniform(40000, 50000, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }
        
        # Add some trend to make it more realistic
        for i in range(1, 100):
            mock_data['close'][i] = mock_data['close'][i-1] * (1 + np.random.uniform(-0.02, 0.02))
            mock_data['high'][i] = max(mock_data['close'][i], mock_data['high'][i])
            mock_data['low'][i] = min(mock_data['close'][i], mock_data['low'][i])
            mock_data['open'][i] = mock_data['close'][i-1]
        
        df = pd.DataFrame(mock_data)
        df_map = {'15m': df, '1h': df, '4h': df, '1d': df}
        
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
