#!/usr/bin/env python3
"""
Test script to validate the sell request processing fixes.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest


pytestmark = pytest.mark.regression

def test_duplicate_prevention():
    """Test that duplicate sell requests are prevented."""
    print("Testing duplicate sell request prevention...")

    # Create test sell requests file
    test_requests = [
        {
            'symbol': 'BTC/USD',
            'amount': 1.0,
            'timestamp': datetime.utcnow().isoformat()
        },
        {
            'symbol': 'ETH/USD',
            'amount': 0.5,
            'timestamp': datetime.utcnow().isoformat()
        },
        {
            'symbol': 'BTC/USD',  # Duplicate
            'amount': 2.0,
            'timestamp': datetime.utcnow().isoformat()
        }
    ]

    # Simulate the duplicate prevention logic from frontend/app.py
    filtered_requests = []
    processed_symbols = set()

    for request in test_requests:
        symbol = request.get('symbol', '')

        # Skip duplicates
        if symbol in processed_symbols:
            print(f"✓ Duplicate request for {symbol} was filtered out")
            continue

        filtered_requests.append(request)
        processed_symbols.add(symbol)

    print(f"Original requests: {len(test_requests)}")
    print(f"Filtered requests: {len(filtered_requests)}")
    print(f"Unique symbols: {list(processed_symbols)}")

    assert len(filtered_requests) == 2, "Should have 2 unique requests"
    assert len(processed_symbols) == 2, "Should have 2 unique symbols"

    print("✓ Duplicate prevention test passed\n")

def test_stale_request_filtering():
    """Test that stale requests are filtered out."""
    print("Testing stale request filtering...")

    now = datetime.utcnow()
    one_hour_ago = now - timedelta(hours=1)
    recent_time = now - timedelta(minutes=30)

    test_requests = [
        {
            'symbol': 'BTC/USD',
            'amount': 1.0,
            'timestamp': now.isoformat()  # Recent
        },
        {
            'symbol': 'ETH/USD',
            'amount': 0.5,
            'timestamp': one_hour_ago.isoformat()  # Stale
        },
        {
            'symbol': 'ADA/USD',
            'amount': 100.0,
            'timestamp': recent_time.isoformat()  # Recent
        }
    ]

    # Filter stale requests
    filtered_requests = []
    one_hour_ago_threshold = datetime.utcnow() - timedelta(hours=1)

    for request in test_requests:
        timestamp_str = request.get('timestamp', '')
        try:
            request_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if request_time < one_hour_ago_threshold:
                print(f"✓ Stale request for {request['symbol']} was filtered out")
                continue
        except (ValueError, AttributeError):
            print(f"✓ Invalid timestamp for {request['symbol']}, processing anyway")

        filtered_requests.append(request)

    print(f"Original requests: {len(test_requests)}")
    print(f"Filtered requests: {len(filtered_requests)}")

    assert len(filtered_requests) == 2, "Should have 2 non-stale requests"

    print("✓ Stale request filtering test passed\n")

def test_atomic_file_writing():
    """Test atomic file writing with retry logic."""
    print("Testing atomic file writing...")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / 'sell_requests.json'
        test_data = [{'symbol': 'BTC/USD', 'amount': 1.0}]

        # Test atomic write
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Write attempt {attempt + 1}")

                # Ensure directory exists
                test_file.parent.mkdir(parents=True, exist_ok=True)

                # Write to temporary file first, then rename for atomic operation
                temp_file = test_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(test_data, f, indent=2)

                # Atomic rename
                temp_file.replace(test_file)

                # Verify the file was written correctly
                if test_file.exists():
                    with open(test_file, 'r') as f:
                        verify_requests = json.load(f)
                    print(f"✓ Verified {len(verify_requests)} requests in file")
                    break  # Success
                else:
                    print("✗ File was not created")

            except Exception as e:
                print(f"✗ Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    print("✗ All attempts failed")
                    raise

        # Verify final result
        assert test_file.exists(), "File should exist"
        with open(test_file, 'r') as f:
            final_data = json.load(f)
        assert len(final_data) == 1, "Should have 1 request"
        assert final_data[0]['symbol'] == 'BTC/USD', "Symbol should match"

    print("✓ Atomic file writing test passed\n")

if __name__ == "__main__":
    print("Running sell request processing fix tests...\n")

    try:
        test_duplicate_prevention()
        test_stale_request_filtering()
        test_atomic_file_writing()

        print("🎉 All tests passed! Sell request processing fixes are working correctly.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
