#!/usr/bin/env python3
"""
Test script for pump.fun wallet evaluation system.
"""

import asyncio
import os
import sys
import json
sys.path.append('.')

from crypto_bot.solana.scanner import evaluate_pump_fun_launches, evaluate_creator_wallet
import aiohttp

async def test_wallet_evaluation():
    """Test the wallet evaluation system with sample data."""

    # Sample pump.fun launch data (simulated)
    sample_launches = [
        {
            "mint": "9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump",
            "creator": "CreatorWallet12345678901234567890123456789012",
            "bonding_curve": {"progress": 85},
            "market_cap": 25000,
            "replies": 150,
            "created_timestamp": "2024-01-15T10:30:00Z"
        },
        {
            "mint": "SuspiciousToken98765432109876543210987654321098",
            "creator": "ScamWalletRugPull987654321098765432109876543210",
            "bonding_curve": {"progress": 15},
            "market_cap": 200,
            "replies": 5,
            "created_timestamp": "2024-01-15T10:25:00Z"  # Very new
        },
        {
            "mint": "CredibleToken45678901234567890123456789012345",
            "creator": "CredibleWallet45678901234567890123456789012pump",
            "bonding_curve": {"progress": 92},
            "market_cap": 45000,
            "replies": 300,
            "created_timestamp": "2024-01-14T15:20:00Z"  # 19 hours old
        }
    ]

    print("🧪 Testing pump.fun wallet evaluation system...")
    print(f"📊 Testing with {len(sample_launches)} sample launches\n")

    # Test wallet evaluation
    async with aiohttp.ClientSession() as session:
        print("🔍 Testing individual wallet evaluation:")
        for launch in sample_launches:
            creator = launch["creator"]
            score, factors = await evaluate_creator_wallet(creator, session)
            print(f"Wallet: {creator[:20]}...")
            print(f"Score: {score:+d}")
            print(f"Factors: {factors}")
            print()

        # Test full launch evaluation
        print("📈 Testing full launch evaluation:")
        evaluated_launches = await evaluate_pump_fun_launches(sample_launches, session)

        print("\n🏆 Results (sorted by credibility score):")
        for i, launch in enumerate(evaluated_launches, 1):
            print(f"{i}. {launch['mint'][:20]}...")
            print(f"   Score: {launch['credibility_score']}/100")
            print(f"   Reason: {launch['evaluation_reason']}")
            print(f"   Factors: {launch['evaluation_factors']}")
            print()

        # Show filtering results
        credible_threshold = 60
        credible_launches = [l for l in evaluated_launches if l.get("credibility_score", 0) >= credible_threshold]

        print(f"🎯 Filtering with credibility threshold >= {credible_threshold}:")
        print(f"Total launches: {len(evaluated_launches)}")
        print(f"Credible launches: {len(credible_launches)}")
        print(f"Filtered out: {len(evaluated_launches) - len(credible_launches)}")

        if credible_launches:
            print("\n✅ Tokens that would pass filtering:")
            for launch in credible_launches:
                print(f"   • {launch['mint']} (Score: {launch['credibility_score']})")

if __name__ == "__main__":
    asyncio.run(test_wallet_evaluation())
