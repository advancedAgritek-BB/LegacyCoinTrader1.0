#!/usr/bin/env python3
"""Test script to verify Solana contract address processing."""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_bot.utils.market_loader import (
    normalize_symbol,
    is_solana_contract_address,
    map_solana_contract_to_symbol,
    should_use_kraken_for_symbol,
    KRAKEN_SUPPORTED_SOLANA_SYMBOLS
)

def test_solana_contract_addresses():
    """Test processing of Solana contract addresses from the user's log."""

    # Sample addresses from the user's warning messages
    test_addresses = [
        'So11111111111111111111111111111111111111112',  # SOL
        'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  # USDC
        'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',  # USDT
        '2b1kV6DkPAnxd5ixfnxCpjxmKwqjjaYmCZfHsFu24GXo',  # PYTH
        'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',  # WIF
        '7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr',  # BONK
        'J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn',  # JTO
        'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',  # JUP
        'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',  # DBR
        '27G8MtK7VtTcCHkpASjSDdkWWYfoqT6ggEuKidVJidD4',  # RAY
        '5LafQUrVco6o7KMz42eqVEJ9LW31StPyGjeeu5sKoMtA',  # HBAR
        'jupSoLaHXQiZZTSfEWMTRRgpnyFm8f6sZdosWBjx93v',  # JUPSOL
        'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So',  # MSOL
        '5mbK36SZ7J19An8jFochhQS4of8g6BwUjbeCSxBSoWdp',  # ORCA
        'hntyVP6YFm1Hg25TN9WGLqM12b8TQmcknKrdu1oxWux',  # HNT
        'GiG7Hr61RVm4CSUxJmgiCoySFQtdiwxtqf64MsRppump',  # PUMP token
        'GinNabffZL4fUj9Vactxha74GDAW8kDPGaHqMtMzps2f',  # Unknown token
        'GYKmdfcUmZVrqfcH1g579BGjuzSRijj3LBuwv79rpump',  # PUMP token
    ]

    print("Testing Solana Contract Address Processing")
    print("=" * 50)

    for address in test_addresses:
        print(f"\nTesting: {address}")

        # Test if it's detected as a Solana contract
        is_contract = is_solana_contract_address(address)
        print(f"  Is Solana contract: {is_contract}")

        if is_contract:
            # Test mapping to symbol
            symbol = map_solana_contract_to_symbol(address)
            print(f"  Mapped symbol: {symbol}")

            # Test normalization
            normalized = normalize_symbol(address)
            print(f"  Normalized symbol: {normalized}")

            # Test Kraken support
            use_kraken = should_use_kraken_for_symbol(address)
            print(f"  Should use Kraken: {use_kraken}")

            if use_kraken:
                kraken_supported = normalized in KRAKEN_SUPPORTED_SOLANA_SYMBOLS
                print(f"  Kraken supported: {kraken_supported}")
        else:
            print("  Not recognized as Solana contract address")

    print("\n" + "=" * 50)
    print("Kraken Supported Solana Symbols:")
    for symbol in sorted(KRAKEN_SUPPORTED_SOLANA_SYMBOLS):
        print(f"  {symbol}")

if __name__ == "__main__":
    test_solana_contract_addresses()
