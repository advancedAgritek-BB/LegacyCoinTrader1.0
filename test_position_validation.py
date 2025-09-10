#!/usr/bin/env python3
"""
Test script to verify position validation is working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from crypto_bot.utils.trade_manager import is_test_position, validate_position_symbol, Trade, Position
from datetime import datetime
from decimal import Decimal

def test_validation_function():
    """Test the is_test_position function."""
    print("🧪 Testing is_test_position function...")

    # Test cases that should be flagged as test positions
    test_symbols = [
        'TEST/USD',
        'FAKE/BTC',
        'DUMMY/ETH',
        'MOCK/SOL',
        'SAMPLE/ADA',
        'EXAMPLE/DOT',
        'DEMO/LINK',
        'SANDBOX/UNI',
        'TESTEX/USD',
        'FAKEEX/BTC'
    ]

    # Test cases that should NOT be flagged
    real_symbols = [
        'BTC/USD',
        'ETH/USDT',
        'ADA/USD',
        'DOT/USD',
        'LINK/USD',
        'UNI/USD',
        'SOL/USD'
    ]

    print("Testing test symbols (should be flagged):")
    for symbol in test_symbols:
        result = is_test_position(symbol)
        status = "✅" if result else "❌"
        print(f"  {status} {symbol}: {result}")

    print("\nTesting real symbols (should NOT be flagged):")
    for symbol in real_symbols:
        result = is_test_position(symbol)
        status = "✅" if not result else "❌"
        print(f"  {status} {symbol}: {result}")

def test_trade_validation():
    """Test Trade class validation."""
    print("\n🧪 Testing Trade class validation...")

    try:
        # This should fail
        test_trade = Trade(
            id="test_123",
            symbol="TEST/USD",
            side="buy",
            amount=Decimal('1.0'),
            price=Decimal('100.0'),
            timestamp=datetime.utcnow()
        )
        print("❌ Trade with test symbol was not rejected!")
    except ValueError as e:
        print(f"✅ Trade with test symbol correctly rejected: {e}")

    try:
        # This should succeed
        real_trade = Trade(
            id="real_123",
            symbol="BTC/USD",
            side="buy",
            amount=Decimal('0.001'),
            price=Decimal('50000.0'),
            timestamp=datetime.utcnow()
        )
        print("✅ Trade with real symbol was accepted")
    except ValueError as e:
        print(f"❌ Trade with real symbol was incorrectly rejected: {e}")

def test_position_validation():
    """Test Position class validation."""
    print("\n🧪 Testing Position class validation...")

    try:
        # This should fail
        test_position = Position(
            symbol="TEST/USD",
            side="long",
            total_amount=Decimal('1.0'),
            average_price=Decimal('100.0')
        )
        print("❌ Position with test symbol was not rejected!")
    except ValueError as e:
        print(f"✅ Position with test symbol correctly rejected: {e}")

    try:
        # This should succeed
        real_position = Position(
            symbol="BTC/USD",
            side="long",
            total_amount=Decimal('0.001'),
            average_price=Decimal('50000.0')
        )
        print("✅ Position with real symbol was accepted")
    except ValueError as e:
        print(f"❌ Position with real symbol was incorrectly rejected: {e}")

def test_validate_position_symbol():
    """Test the validate_position_symbol function."""
    print("\n🧪 Testing validate_position_symbol function...")

    try:
        validate_position_symbol("TEST/USD", "test")
        print("❌ Test symbol was not rejected by validate_position_symbol!")
    except ValueError as e:
        print(f"✅ Test symbol correctly rejected by validate_position_symbol: {e}")

    try:
        validate_position_symbol("BTC/USD", "test")
        print("✅ Real symbol was accepted by validate_position_symbol")
    except ValueError as e:
        print(f"❌ Real symbol was incorrectly rejected by validate_position_symbol: {e}")

def main():
    """Run all validation tests."""
    print("🔍 Testing Position Validation System")
    print("=" * 50)

    test_validation_function()
    test_trade_validation()
    test_position_validation()
    test_validate_position_symbol()

    print("\n" + "=" * 50)
    print("🎉 Position validation testing completed!")

if __name__ == "__main__":
    main()
