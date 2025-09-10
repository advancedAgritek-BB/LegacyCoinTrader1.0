#!/usr/bin/env python3
"""
Quick ETH/USD pump monitoring script.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.utils.logger import setup_logger, LOG_DIR
import yaml
import ccxt
import pandas as pd
import numpy as np

# Setup logging
logger = setup_logger("eth_pump_monitor", LOG_DIR / "eth_pump_monitor.log")

async def monitor_eth_pump():
    """Monitor ETH/USD pump and analyze why bot isn't catching it."""
    
    logger.info("🔍 Starting ETH/USD pump monitoring...")
    
    # Load config
    with open("crypto_bot/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize exchange
    exchange = ccxt.kraken({
        'apiKey': config.get('kraken', {}).get('api_key', ''),
        'secret': config.get('kraken', {}).get('secret', ''),
        'sandbox': config.get('kraken', {}).get('sandbox', False),
        'timeout': config.get('kraken', {}).get('timeout', 30)
    })
    
    try:
        # Fetch current ETH/USD data
        logger.info("📊 Fetching ETH/USD data...")
        
        # Get ticker
        ticker = await exchange.fetch_ticker('ETH/USD')
        logger.info(f"💰 ETH/USD Current Price: ${ticker['last']:,.2f}")
        logger.info(f"📈 24h Change: {ticker['percentage']:.2f}%")
        logger.info(f"📊 24h High: ${ticker['high']:,.2f}")
        logger.info(f"📉 24h Low: ${ticker['low']:,.2f}")
        logger.info(f"📈 24h Volume: {ticker['baseVolume']:,.2f} ETH")
        
        # Get OHLCV data
        ohlcv = await exchange.fetch_ohlcv('ETH/USD', '1h', limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate basic indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = calculate_rsi(df['close'], 14)
        df['volatility'] = df['close'].rolling(20).std()
        
        # Get latest values
        latest = df.iloc[-1]
        prev_20 = df.iloc[-20]
        
        logger.info(f"📊 Technical Analysis:")
        logger.info(f"   SMA 20: ${latest['sma_20']:,.2f}")
        logger.info(f"   SMA 50: ${latest['sma_50']:,.2f}")
        logger.info(f"   RSI: {latest['rsi']:.2f}")
        logger.info(f"   Volatility: {latest['volatility']:.2f}")
        
        # Check for pump signals
        price_change_1h = (latest['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close'] * 100
        price_change_24h = (latest['close'] - df.iloc[-24]['close']) / df.iloc[-24]['close'] * 100
        
        logger.info(f"🚀 Pump Analysis:")
        logger.info(f"   1h Change: {price_change_1h:.2f}%")
        logger.info(f"   24h Change: {price_change_24h:.2f}%")
        
        # Check if this should trigger a signal
        if price_change_1h > 2.0 or price_change_24h > 5.0:
            logger.info("🔥 ETH/USD PUMP DETECTED! Bot should be generating signals.")
        else:
            logger.info("📊 ETH/USD showing normal movement, not a significant pump.")
        
        # Test strategy analysis
        from crypto_bot.utils.market_analyzer import analyze_symbol
        
        logger.info("🧪 Testing strategy analysis...")
        
        # Create mock context
        class MockContext:
            def __init__(self):
                self.df_cache = {'1h': {'ETH/USD': df}}
        
        ctx = MockContext()
        
        # Analyze ETH/USD
        result = await analyze_symbol('ETH/USD', ctx, config)
        
        logger.info(f"📊 Strategy Analysis Results:")
        logger.info(f"   Regime: {result.get('regime', 'unknown')}")
        logger.info(f"   Confidence: {result.get('confidence', 0):.4f}")
        logger.info(f"   Score: {result.get('score', 0):.4f}")
        logger.info(f"   Direction: {result.get('direction', 'none')}")
        
        if result.get('score', 0) > 0:
            logger.info("✅ Strategy is generating signals!")
        else:
            logger.info("❌ Strategy score is 0 - no signals generated")
            
    except Exception as e:
        logger.error(f"❌ Error monitoring ETH/USD: {e}")
        raise

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

if __name__ == "__main__":
    asyncio.run(monitor_eth_pump())
