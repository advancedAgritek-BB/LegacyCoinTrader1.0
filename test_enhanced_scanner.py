#!/usr/bin/env python3
"""
Test script for the enhanced Solana scanner.
"""

import asyncio
import sys
sys.path.append('/Users/brandonburnette/Downloads/LegacyCoinTrader1.0')

from crypto_bot.solana.enhanced_scanner import get_enhanced_scanner


async def test_enhanced_scanner() -> bool:
    """Test the enhanced scanner pipeline."""
    print("Testing Enhanced Solana Scanner...")

    # Load config
    config = {
        "solana_scanner": {
            "scan_interval_minutes": 30,
            "max_tokens_per_scan": 20,
            "min_score_threshold": 0.3,
            "enable_sentiment": False,
            "enable_pyth_prices": False,
            "min_volume_usd": 5000,
            "max_spread_pct": 2.0,
            "min_liquidity_score": 0.5,
            "min_strategy_fit": 0.6,
            "min_confidence": 0.5
        }
    }

    try:
        # Get scanner instance
        scanner = get_enhanced_scanner(config)
        print("✓ Scanner instance created successfully")

        # Test token discovery
        print("\nTesting token discovery...")
        tokens = await scanner._discover_tokens()
        print(f"✓ Discovered {len(tokens)} tokens: {tokens[:5]}...")

        if tokens:
            # Test token analysis
            print("\nTesting token analysis...")
            analyzed = await scanner._analyze_tokens(
                tokens[:5]  # Test with first 5
            )
            print(f"✓ Analyzed {len(analyzed)} tokens")

            if analyzed:
                # Test token scoring
                print("\nTesting token scoring...")
                scored = scanner._score_tokens(analyzed)
                print(f"✓ Scored {len(scored)} tokens")
                print("Top scored tokens:")
                for token, score, regime, data in scored[:3]:
                    print(f"  {token}: {score:.2f} ({regime})")

                # Test cache results
                print("\nTesting result caching...")
                await scanner._cache_results(scored)
                print("✓ Results cached successfully")

                # Test opportunity retrieval
                print("\nTesting opportunity retrieval...")
                opportunities = scanner.get_top_opportunities(limit=5)
                print(f"✓ Retrieved {len(opportunities)} opportunities")
                for opp in opportunities[:3]:
                    print(
                        f"  {opp.get('symbol', 'N/A')}: {opp.get('score', 0):.2f}"
                    )

        print("\n✅ Enhanced scanner test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Scanner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_enhanced_scanner())
    sys.exit(0 if success else 1)
