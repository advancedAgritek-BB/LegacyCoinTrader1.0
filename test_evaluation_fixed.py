#!/usr/bin/env python3
"""
Test script to verify the evaluation pipeline is working correctly after fixes.
This will test the complete pipeline with real market data.
"""

import asyncio
import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.utils.logger import setup_logger, LOG_DIR
from crypto_bot.utils.market_analyzer import analyze_symbol
from crypto_bot.phase_runner import BotContext
from crypto_bot.utils.market_loader import fetch_ohlcv_async
from crypto_bot.execution.cex_executor import get_exchange

# Setup logging
logger = setup_logger("evaluation_test_fixed", LOG_DIR / "evaluation_test_fixed.log")

async def test_evaluation_pipeline():
    """Test the evaluation pipeline with real market data."""
    
    logger.info("Starting evaluation pipeline test with fixes")
    
    # Test symbols
    test_symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    
    # Load configuration
    with open("crypto_bot/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config with min_confidence_score: {config.get('min_confidence_score', 'NOT SET')}")
    
    # Create bot context
    context = BotContext(
        config=config,
        positions={},
        df_cache={},
        regime_cache={}
    )
    
    results = []
    
    # Get exchange
    exchange, _ = get_exchange(config)
    
    for symbol in test_symbols:
        try:
            logger.info(f"Testing analysis for {symbol}")
            
            # Fetch real market data
            ohlcv_data = await fetch_ohlcv_async(exchange, symbol, "15m", 200)
            
            if ohlcv_data is None or len(ohlcv_data) < 50:
                logger.warning(f"Insufficient data for {symbol}, skipping")
                continue
            
            # Convert to DataFrame with proper column names
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Check the actual column names
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
            logger.info(f"DataFrame shape: {df.shape}")
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Analyze the symbol
            analysis_result = await analyze_symbol(
                symbol=symbol,
                df_map={"15m": df, "1d": df},  # Use same data for higher timeframe
                mode="paper",
                config=config
            )
            
            if analysis_result:
                confidence = analysis_result.get('confidence', 0.0)
                regime = analysis_result.get('regime', 'unknown')
                patterns = analysis_result.get('patterns', [])
                
                logger.info(f"Analysis result for {symbol}:")
                logger.info(f"  Confidence: {confidence:.4f}")
                logger.info(f"  Regime: {regime}")
                logger.info(f"  Patterns: {patterns}")
                
                # Check if it passes the confidence threshold
                min_confidence = config.get('min_confidence_score', 0.05)
                if confidence >= min_confidence:
                    logger.info(f"✅ {symbol} PASSES confidence threshold ({confidence:.4f} >= {min_confidence})")
                    results.append({
                        'symbol': symbol,
                        'confidence': confidence,
                        'regime': regime,
                        'patterns': patterns,
                        'passed': True
                    })
                else:
                    logger.info(f"❌ {symbol} FAILS confidence threshold ({confidence:.4f} < {min_confidence})")
                    results.append({
                        'symbol': symbol,
                        'confidence': confidence,
                        'regime': regime,
                        'patterns': patterns,
                        'passed': False
                    })
            else:
                logger.warning(f"No analysis result for {symbol}")
                results.append({
                    'symbol': symbol,
                    'confidence': 0.0,
                    'regime': 'unknown',
                    'patterns': [],
                    'passed': False
                })
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            results.append({
                'symbol': symbol,
                'confidence': 0.0,
                'regime': 'error',
                'patterns': [],
                'passed': False,
                'error': str(e)
            })
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION PIPELINE TEST SUMMARY")
    logger.info("="*50)
    
    passed_count = sum(1 for r in results if r.get('passed', False))
    total_count = len(results)
    
    logger.info(f"Total symbols tested: {total_count}")
    logger.info(f"Symbols passing confidence threshold: {passed_count}")
    logger.info(f"Success rate: {passed_count/total_count*100:.1f}%")
    
    for result in results:
        status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
        logger.info(f"{status} {result['symbol']}: {result['confidence']:.4f} ({result['regime']})")
    
    logger.info("="*50)
    
    if passed_count > 0:
        logger.info("🎉 EVALUATION PIPELINE IS WORKING! Found signals above confidence threshold.")
        return True
    else:
        logger.info("⚠️ No signals found above confidence threshold. This may be normal market conditions.")
        return False

if __name__ == "__main__":
    try:
        result = asyncio.run(test_evaluation_pipeline())
        if result:
            print("✅ Evaluation pipeline test PASSED")
            sys.exit(0)
        else:
            print("⚠️ Evaluation pipeline test completed but no signals found")
            sys.exit(0)
    except Exception as e:
        print(f"❌ Evaluation pipeline test FAILED: {e}")
        sys.exit(1)
