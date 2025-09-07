#!/usr/bin/env python3
"""
Reset the circuit breaker to allow trading to resume.
This script manually resets the global circuit breaker state.
"""

import sys
import os
sys.path.append('/Users/brandonburnette/Downloads/LegacyCoinTrader1.0')

from crypto_bot.utils.market_loader import get_circuit_breaker

def reset_circuit_breaker():
    """Reset the circuit breaker to CLOSED state."""
    try:
        circuit_breaker = get_circuit_breaker()

        # Reset the circuit breaker state
        circuit_breaker.state = "CLOSED"
        circuit_breaker.failure_count = 0
        circuit_breaker.last_failure_time = None

        print("✅ Circuit breaker has been reset to CLOSED state")
        print(f"   State: {circuit_breaker.state}")
        print(f"   Failure count: {circuit_breaker.failure_count}")
        print(f"   Last failure time: {circuit_breaker.last_failure_time}")

        return True

    except Exception as e:
        print(f"❌ Failed to reset circuit breaker: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Resetting Circuit Breaker...")
    success = reset_circuit_breaker()
    if success:
        print("\n🎉 Circuit breaker reset complete!")
        print("The bot should now be able to fetch OHLCV data from Kraken.")
    else:
        print("\n💥 Failed to reset circuit breaker!")
