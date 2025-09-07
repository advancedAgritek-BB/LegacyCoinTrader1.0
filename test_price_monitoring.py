#!/usr/bin/env python3
"""
Test script for price monitoring functionality.

This script validates that the price monitoring service is working correctly
and that open positions are displaying current market prices.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.utils.price_monitor import get_price_monitor, start_price_monitoring, stop_price_monitoring
from crypto_bot.utils.trade_manager import get_trade_manager
from crypto_bot.execution.cex_executor import get_exchange
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_price_monitoring():
    """Test the price monitoring service functionality."""
    logger.info("Starting price monitoring test...")

    try:
        # Load configuration
        config_path = project_root / "crypto_bot" / "config.yaml"
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return False

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        logger.info("Configuration loaded successfully")

        # Initialize exchange
        exchange, ws_client = get_exchange(config)
        logger.info(f"Exchange initialized: {config.get('exchange', 'unknown')}")

        # Get TradeManager and initialize it
        tm = get_trade_manager()
        logger.info(f"TradeManager initialized with {len(tm.get_all_positions())} open positions")

        # Test price monitor initialization
        price_monitor = get_price_monitor(tm)
        logger.info("Price monitor created")

        # Test symbol subscription
        test_symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        for symbol in test_symbols:
            price_monitor.subscribe_symbol(symbol)
        logger.info(f"Subscribed to {len(test_symbols)} test symbols")

        # Start monitoring
        price_monitor.start_monitoring(exchange)
        logger.info("Price monitoring service started")

        # Wait a bit for prices to be fetched
        logger.info("Waiting 30 seconds for price updates...")
        await asyncio.sleep(30)

        # Check price status
        status = price_monitor.get_price_status()
        logger.info(f"Price status for {len(status)} symbols:")

        success_count = 0
        for symbol, info in status.items():
            if info['cached_price'] and info['is_fresh']:
                logger.info(f"✅ {symbol}: ${info['cached_price']:.2f} (fresh)")
                success_count += 1
            else:
                logger.warning(f"❌ {symbol}: No fresh price available")

        # Test force price update
        logger.info("Testing force price update for BTC/USD...")
        force_success = price_monitor.force_price_update("BTC/USD", exchange)
        if force_success:
            logger.info("✅ Force price update successful")
        else:
            logger.warning("❌ Force price update failed")

        # Stop monitoring
        price_monitor.stop_monitoring()
        logger.info("Price monitoring service stopped")

        # Summary
        total_symbols = len(status)
        success_rate = (success_count / total_symbols) * 100 if total_symbols > 0 else 0

        logger.info(f"Test Results:")
        logger.info(f"- Total symbols monitored: {total_symbols}")
        logger.info(f"- Successful price fetches: {success_count}")
        logger.info(f"- Success rate: {success_rate:.1f}%")

        return success_rate >= 70.0  # Consider test passed if 70%+ success rate

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trade_manager_integration():
    """Test TradeManager integration with price monitoring."""
    logger.info("Testing TradeManager integration...")

    try:
        tm = get_trade_manager()
        open_positions = tm.get_all_positions()

        logger.info(f"Found {len(open_positions)} open positions")

        for position in open_positions:
            cached_price = tm.price_cache.get(position.symbol)
            if cached_price:
                pnl, pnl_pct = position.calculate_unrealized_pnl(cached_price)
                logger.info(f"📊 {position.symbol}: {position.total_amount:.4f} @ ${position.average_price:.2f} "
                           f"(current: ${cached_price:.2f}) PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
            else:
                logger.warning(f"⚠️ No cached price for {position.symbol}")

        return True

    except Exception as e:
        logger.error(f"TradeManager integration test failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("=" * 50)
    logger.info("PRICE MONITORING SYSTEM TEST")
    logger.info("=" * 50)

    # Test TradeManager integration
    tm_success = test_trade_manager_integration()

    # Test price monitoring service
    pm_success = await test_price_monitoring()

    # Overall result
    logger.info("=" * 50)
    if tm_success and pm_success:
        logger.info("✅ ALL TESTS PASSED - Price monitoring system is working correctly")
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED - Check logs for details")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        stop_price_monitoring()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        stop_price_monitoring()
        sys.exit(1)
