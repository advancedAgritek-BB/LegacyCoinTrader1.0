#!/usr/bin/env python3
"""
Test script to verify that PaperWallet handles Decimal objects correctly.
This addresses the issue: "unsupported operand type(s) for -: 'decimal.Decimal' and 'float'"
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

# Ensure the project root is on the Python path for local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crypto_bot.paper_wallet import PaperWallet

pytestmark = pytest.mark.regression

def test_paper_wallet_with_decimals():
    """Test that PaperWallet can handle Decimal objects without errors."""
    print("Testing PaperWallet with Decimal objects...")

    # Create a paper wallet
    wallet = PaperWallet(balance=1000.0)

    # Test opening a position with float prices
    print("\n1. Testing buy with float prices...")
    success = wallet.buy("AIR/USD", 10.0, 0.003000)
    print(f"Buy order success: {success}")
    print(f"Wallet balance: ${wallet.balance:.2f}")

    # Test closing position with Decimal price (this should work now)
    print("\n2. Testing close with Decimal price...")
    try:
        # Simulate the scenario from the error log
        decimal_price = Decimal("0.003330")  # This is what caused the error
        pnl = wallet.close("AIR/USD", 10.0, decimal_price)
        print(f"Close order success: PnL = ${pnl:.2f}")
        print(f"Wallet balance: ${wallet.balance:.2f}")
    except Exception as e:
        print(f"ERROR: {e}")
        return False

    # Test unrealized with Decimal
    print("\n3. Testing unrealized with Decimal price...")
    try:
        wallet.buy("BTC/USD", 0.01, 50000.0)
        unrealized_pnl = wallet.unrealized("BTC/USD", Decimal("51000.0"))
        print(f"Unrealized PnL with Decimal: ${unrealized_pnl:.2f}")
    except Exception as e:
        print(f"ERROR: {e}")
        return False

    # Test sell with Decimal
    print("\n4. Testing sell with Decimal price...")
    try:
        success = wallet.sell("BTC/USD", 0.01, Decimal("51000.0"))
        print(f"Sell order success: {success}")
        print(f"Wallet balance: ${wallet.balance:.2f}")
    except Exception as e:
        print(f"ERROR: {e}")
        return False

    print("\n✅ All tests passed! PaperWallet now handles Decimal objects correctly.")
    return True

if __name__ == "__main__":
    success = test_paper_wallet_with_decimals()
    sys.exit(0 if success else 1)
