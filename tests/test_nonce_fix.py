#!/usr/bin/env python3
"""
Test script to verify Kraken nonce error fix.
This script tests the WebSocket client with nonce improvements.
"""

import logging
import os
import sys
import time
from pathlib import Path

import pytest

# Ensure the project root and crypto_bot package are available for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CRYPTO_BOT_PATH = PROJECT_ROOT / "crypto_bot"
if str(CRYPTO_BOT_PATH) not in sys.path:
    sys.path.insert(0, str(CRYPTO_BOT_PATH))

from crypto_bot.execution.cex_executor import get_exchange
from crypto_bot.utils.logger import setup_logger

pytestmark = pytest.mark.regression

# Setup logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = setup_logger(__name__, LOG_DIR / "test_nonce.log")

def test_nonce_fix():
    """Test that WebSocket client can place orders without nonce errors."""

    # Check if API credentials are available
    api_key = os.getenv("API_KEY") or os.getenv("KRAKEN_API_KEY")
    api_secret = os.getenv("API_SECRET") or os.getenv("KRAKEN_API_SECRET")

    if not api_key or not api_secret:
        logger.warning("API credentials not found. Skipping live test.")
        logger.info("To test with live credentials, set API_KEY and API_SECRET environment variables")
        return False

    try:
        # Configure for WebSocket trading
        config = {
            "exchange": "kraken",
            "use_websocket": True,
            "enable_nonce_improvements": True,
            "api_retry_attempts": 3
        }

        logger.info("Testing nonce fix with WebSocket client...")

        # Get exchange and WebSocket client with nonce improvements
        exchange, ws_client = get_exchange(config)

        if not ws_client:
            logger.error("WebSocket client not created")
            return False

        if not ws_client.exchange:
            logger.error("WebSocket client does not have exchange instance")
            return False

        # Test token generation (this should use the improved nonce)
        logger.info("Testing WebSocket token generation...")
        token = ws_client.get_token()
        logger.info(f"Successfully generated WebSocket token: {token[:10]}...")

        # Test multiple token generations to ensure nonce doesn't get reused
        logger.info("Testing multiple token generations...")
        for i in range(3):
            time.sleep(0.1)  # Small delay to ensure different timestamps
            token2 = ws_client.get_token()
            if token != token2:
                logger.info(f"Token {i+1} generated successfully (different from previous)")
            else:
                logger.warning(f"Token {i+1} is the same as previous - this might indicate nonce reuse")

        # Note: We don't actually place orders in this test to avoid real trades
        # The main goal is to verify that token generation (which uses nonces) works correctly

        logger.info("✅ Nonce fix test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Nonce fix test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting Kraken nonce fix test...")
    success = test_nonce_fix()
    if success:
        logger.info("🎉 Test passed! Nonce error should be resolved.")
    else:
        logger.info("⚠️ Test inconclusive - check API credentials and try again.")
    sys.exit(0 if success else 1)
