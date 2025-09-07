#!/usr/bin/env python3
"""
Fix rate limiting configuration to prevent Kraken API rate limit errors.
This script reconfigures the global rate limiter with more conservative settings.
"""

import sys
import os
sys.path.append('/Users/brandonburnette/Downloads/LegacyCoinTrader1.0')

from crypto_bot.utils.market_loader import configure_rate_limiter

def fix_rate_limiting():
    """Reconfigure the rate limiter with more conservative settings."""
    try:
        # Configure rate limiter with more conservative settings
        configure_rate_limiter(
            max_requests_per_minute=20,  # Reduced from default 30
            base_delay=1.5,              # Increased from default 1.0
            max_delay=10.0,              # Increased from default 5.0
            error_backoff_multiplier=2.0,
            success_recovery_factor=0.9,
            window_size=1000
        )

        print("✅ Rate limiter reconfigured with conservative settings:")
        print("   Max requests per minute: 20")
        print("   Base delay: 1.5 seconds")
        print("   Max delay: 10.0 seconds")
        print("   Error backoff multiplier: 2.0")
        print("   Success recovery factor: 0.9")

        return True

    except Exception as e:
        print(f"❌ Failed to reconfigure rate limiter: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Fixing Rate Limiting Configuration...")
    success = fix_rate_limiting()
    if success:
        print("\n🎉 Rate limiting configuration updated!")
        print("This should reduce Kraken API rate limit errors.")
    else:
        print("\n💥 Failed to update rate limiting configuration!")
