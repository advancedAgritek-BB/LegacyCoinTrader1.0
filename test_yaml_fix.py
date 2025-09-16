#!/usr/bin/env python3
"""Test script to verify YAML serialization fix."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crypto_bot.paper_wallet import PaperWallet
from decimal import Decimal
import numpy as np

def test_yaml_serialization():
    """Test that YAML serialization works with numpy scalars and Decimal objects."""
    print("Testing YAML serialization fix...")

    # Create a paper wallet instance
    wallet = PaperWallet(balance=10000.0)

    # Add a position with Decimal values (simulating real usage)
    wallet.positions['TEST/USD'] = {
        'symbol': 'TEST/USD',
        'side': 'long',
        'amount': Decimal('100.0'),
        'entry_price': Decimal('50.0'),
        'current_price': Decimal('55.0'),
        'pnl': Decimal('0.1'),
        'fees_paid': 0.0,
        'timestamp': '2025-09-16T12:00:00.000000'
    }

    # Test save
    print("Saving state...")
    wallet.save_state()

    # Test load
    print("Loading state...")
    success = wallet.load_state()

    if success:
        print("✅ YAML serialization test passed!")
        print(f"Balance: {wallet.balance}")
        print(f"Positions: {len(wallet.positions)}")
        return True
    else:
        print("❌ YAML serialization test failed!")
        return False

if __name__ == "__main__":
    test_yaml_serialization()
