#!/usr/bin/env python3
"""
Diagnostic script to investigate why the evaluation pipeline is not returning tokens.
This will help identify why the bot is not actively trading.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.evaluation_pipeline_integration import (
    get_evaluation_pipeline_integration,
    get_tokens_for_evaluation,
    get_pipeline_status
)
from crypto_bot.main import load_config

async def diagnose_pipeline():
    """Diagnose the evaluation pipeline to find why it's not returning tokens."""

    print("🔍 Diagnosing Evaluation Pipeline Issues")
    print("=" * 50)

    try:
        # Load configuration
        print("\n📄 Loading configuration...")
        config = load_config()
        print(f"✅ Config loaded. Mode: {config.get('mode', 'cex')}")

        # Get pipeline integration
        print("\n🔧 Initializing evaluation pipeline integration...")
        integration = get_evaluation_pipeline_integration(config)
        print(f"✅ Pipeline integration initialized: {type(integration).__name__}")

        # Check pipeline status
        print("\n📊 Checking pipeline status...")
        status = get_pipeline_status(config)
        print(f"Pipeline status: {status}")

        # Try to get tokens
        print("\n🎯 Testing token retrieval...")
        print("Requesting 5 tokens for evaluation...")

        tokens = await get_tokens_for_evaluation(config, 5)
        print(f"✅ Got {len(tokens)} tokens: {tokens}")

        if not tokens:
            print("❌ No tokens returned - this is the issue!")
            print("\n🔍 Let's check each source...")

            # Test each source individually
            print("\n1️⃣ Testing enhanced scanner...")
            try:
                scanner_tokens = await integration._get_scanner_tokens(5)
                print(f"   Scanner tokens: {len(scanner_tokens)} - {scanner_tokens}")
            except Exception as e:
                print(f"   ❌ Scanner failed: {e}")

            print("\n2️⃣ Testing Solana scanner...")
            try:
                solana_tokens = await integration._get_solana_tokens(5)
                print(f"   Solana tokens: {len(solana_tokens)} - {solana_tokens}")
            except Exception as e:
                print(f"   ❌ Solana scanner failed: {e}")

            print("\n3️⃣ Testing static config...")
            try:
                config_tokens = await integration._get_config_tokens(5)
                print(f"   Config tokens: {len(config_tokens)} - {config_tokens}")
            except Exception as e:
                print(f"   ❌ Config failed: {e}")

            print("\n4️⃣ Testing fallback...")
            try:
                fallback_tokens = await integration._get_fallback_tokens(5)
                print(f"   Fallback tokens: {len(fallback_tokens)} - {fallback_tokens}")
            except Exception as e:
                print(f"   ❌ Fallback failed: {e}")

        else:
            print("✅ Pipeline is working! Tokens found.")

        # Check metrics
        print("\n📈 Pipeline metrics:")
        print(f"   Tokens received: {integration.metrics.tokens_received}")
        print(f"   Tokens processed: {integration.metrics.tokens_processed}")
        print(f"   Error rate: {integration.metrics.error_rate}")
        print(f"   Consecutive failures: {integration.metrics.consecutive_failures}")

    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(diagnose_pipeline())
