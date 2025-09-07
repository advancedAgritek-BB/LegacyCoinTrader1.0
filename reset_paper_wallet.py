#!/usr/bin/env python3
"""
Script to reset the paper wallet to a clean state.
This will clear all positions and reset balance to initial value.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'crypto_bot'))

from paper_wallet import PaperWallet

def reset_paper_wallet():
    """Reset paper wallet to clean state."""
    print("Resetting paper wallet...")

    # Create wallet instance with initial balance
    wallet = PaperWallet(balance=10000.0, max_open_trades=10, allow_short=True)

    # Load current state (if it exists)
    wallet.load_state()

    print(f"Current state: balance=${wallet.balance:.2f}, positions={len(wallet.positions)}")

    # Reset to clean state
    wallet.reset(10000.0)

    # Force save the reset state
    wallet.save_state()

    print(f"Reset complete: balance=${wallet.balance:.2f}, positions={len(wallet.positions)}")
    print("Paper wallet has been reset to clean state.")

if __name__ == "__main__":
    reset_paper_wallet()
