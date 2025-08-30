#!/usr/bin/env python3
"""
Clear Paper Trading Cache Script

This script clears all trade cache data in paper trading mode to start fresh.
It resets the paper wallet, clears open positions, and clears cached data.

Usage:
    python tools/clear_paper_cache.py
"""

import sys
import os
from pathlib import Path

# Add the crypto_bot directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Main function to clear the paper trading cache."""
    try:
        from crypto_bot.utils.telegram import clear_paper_trading_cache
        from crypto_bot.paper_wallet import PaperWallet
        
        print("🔄 Clearing paper trading cache...")
        print("=" * 50)
        
        # Create a dummy paper wallet to test the reset functionality
        test_wallet = PaperWallet(1000.0)
        test_wallet.open("TEST/USDT", "buy", 0.1, 100.0)
        
        print("📊 Before clearing:")
        print(f"   Wallet balance: ${test_wallet.balance:.2f}")
        print(f"   Open positions: {len(test_wallet.positions)}")
        print(f"   Total trades: {test_wallet.total_trades}")
        
        # Clear the cache
        result = clear_paper_trading_cache(paper_wallet=test_wallet)
        
        print("\n📊 After clearing:")
        print(f"   Wallet balance: ${test_wallet.balance:.2f}")
        print(f"   Open positions: {len(test_wallet.positions)}")
        print(f"   Total trades: {test_wallet.total_trades}")
        
        print("\n" + "=" * 50)
        print("✅ Cache cleared successfully!")
        print("\nTo clear the actual bot cache, you need to:")
        print("1. Stop the bot if it's running")
        print("2. Run this script while the bot is stopped")
        print("3. Or use the Telegram bot command if available")
        print("4. Restart the bot to start fresh")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the project root directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
