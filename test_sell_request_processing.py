#!/usr/bin/env python3
"""
Test script to verify sell request processing functionality
"""

import json
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Add the crypto_bot directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'crypto_bot'))

from main import process_sell_requests
from paper_wallet import PaperWallet

class MockContext:
    def __init__(self):
        self.config = {"execution_mode": "dry_run"}
        self.positions = {}
        self.balance = 10000.0
        self.paper_wallet = PaperWallet(10000.0)
        self.df_cache = {}

def test_sell_request_processing():
    """Test the sell request processing function"""

    # Create mock context
    ctx = MockContext()

    # Create a mock position
    ctx.positions['GOAT/USD'] = {
        'symbol': 'GOAT/USD',
        'side': 'buy',
        'amount': 1000.0,
        'entry_price': 0.01,
        'size': 1000.0
    }

    # Add position to paper wallet
    ctx.paper_wallet.positions['GOAT/USD'] = {
        'symbol': 'GOAT/USD',
        'side': 'buy',
        'amount': 1000.0,
        'entry_price': 0.01,
        'size': 1000.0
    }

    # Create mock price data
    import pandas as pd
    from datetime import datetime

    price_data = pd.DataFrame({
        'timestamp': [datetime.now().timestamp()],
        'open': [0.012],
        'high': [0.012],
        'low': [0.012],
        'close': [0.012],  # Current price higher than entry (profit)
        'volume': [1000]
    })

    ctx.df_cache['GOAT/USD'] = price_data

    # Create sell requests file
    sell_requests_file = Path('crypto_bot/logs/sell_requests.json')
    sell_requests_file.parent.mkdir(exist_ok=True)

    sell_requests = [{
        'symbol': 'GOAT/USD',
        'amount': 1000.0,
        'timestamp': 1756742391.8169029
    }]

    with open(sell_requests_file, 'w') as f:
        json.dump(sell_requests, f)

    print("=== Test Setup Complete ===")
    print(f"Initial balance: ${ctx.balance:.2f}")
    print(f"Position: {ctx.positions['GOAT/USD']}")
    print(f"Sell requests: {sell_requests}")

    # Mock notifier
    notifier = Mock()

    async def run_test():
        try:
            # Process sell requests
            await process_sell_requests(ctx, notifier)

            print("\n=== Test Results ===")
            print(f"Final balance: ${ctx.balance:.2f}")
            print(f"Positions remaining: {list(ctx.positions.keys())}")
            print(f"Paper wallet positions: {list(ctx.paper_wallet.positions.keys())}")

            # Check if sell request was processed
            if sell_requests_file.exists():
                with open(sell_requests_file, 'r') as f:
                    remaining_requests = json.load(f)
                print(f"Remaining sell requests: {remaining_requests}")

                if len(remaining_requests) == 0:
                    print("✅ SUCCESS: Sell request was processed and removed from file")
                    return True
                else:
                    print("❌ FAILURE: Sell request was not processed")
                    return False
            else:
                print("✅ SUCCESS: Sell requests file was cleaned up")
                return True

        except Exception as e:
            print(f"❌ FAILURE: Exception during test: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Run the async test
    result = asyncio.run(run_test())

    # Clean up
    if sell_requests_file.exists():
        sell_requests_file.unlink()

    return result

if __name__ == '__main__':
    print("Testing Sell Request Processing Functionality")
    print("=" * 50)

    success = test_sell_request_processing()

    print("\n" + "=" * 50)
    if success:
        print("✅ All tests PASSED - Sell request processing is working")
    else:
        print("❌ Tests FAILED - Check the issues above")
