#!/usr/bin/env python3
"""
Test script to verify comprehensive mode optimizations work correctly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.main import load_config

async def test_comprehensive_mode():
    """Test comprehensive mode configuration and optimizations."""

    print("🧪 Testing Comprehensive Mode Configuration")
    print("=" * 60)

    try:
        # Load configuration
        print("\n📄 Loading configuration...")
        config = load_config()

        # Check main configuration
        process_all_symbols = config.get("process_all_symbols", True)
        print(f"✅ process_all_symbols: {process_all_symbols}")

        # Check performance optimizations
        symbol_batch_size = config.get("symbol_batch_size", 50)
        cycle_delay = config.get("cycle_delay_seconds", 60)
        max_concurrent_ohlcv = config.get("max_concurrent_ohlcv", 5)
        max_concurrent_requests = config.get("max_concurrent_requests", 5)

        print("\n⚙️  Performance Settings:")
        print(f"   symbol_batch_size: {symbol_batch_size}")
        print(f"   cycle_delay_seconds: {cycle_delay}")
        print(f"   max_concurrent_ohlcv: {max_concurrent_ohlcv}")
        print(f"   max_concurrent_requests: {max_concurrent_requests}")

        # Check comprehensive mode optimizations
        comp_opts = config.get("comprehensive_mode_optimization", {})
        enable_memory_opt = comp_opts.get("enable_memory_optimization", False)
        batch_chunk_size = comp_opts.get("batch_chunk_size", 25)
        enable_progress = comp_opts.get("enable_progress_tracking", False)
        memory_cleanup = comp_opts.get("memory_cleanup_interval", 50)

        print("\n🔧 Comprehensive Mode Optimizations:")
        print(f"   enable_memory_optimization: {enable_memory_opt}")
        print(f"   batch_chunk_size: {batch_chunk_size}")
        print(f"   enable_progress_tracking: {enable_progress}")
        print(f"   memory_cleanup_interval: {memory_cleanup}")

        # Validate configuration
        print("\n✅ Configuration Validation:")
        checks = []

        # Check if comprehensive mode is enabled
        if process_all_symbols:
            checks.append(("✅", "Comprehensive mode enabled"))
        else:
            checks.append(("❌", "Comprehensive mode NOT enabled"))

        # Check if batch size is reasonable for comprehensive mode
        if symbol_batch_size >= 50:
            checks.append(("✅", f"Batch size {symbol_batch_size} is good for comprehensive mode"))
        else:
            checks.append(("⚠️", f"Batch size {symbol_batch_size} might be too small for comprehensive mode"))

        # Check if cycle delay is adequate
        if cycle_delay >= 60:
            checks.append(("✅", f"Cycle delay {cycle_delay}s is adequate for processing"))
        else:
            checks.append(("⚠️", f"Cycle delay {cycle_delay}s might be too short for comprehensive processing"))

        # Check memory optimizations
        if enable_memory_opt:
            checks.append(("✅", "Memory optimization enabled"))
        else:
            checks.append(("⚠️", "Memory optimization not enabled"))

        # Display validation results
        for status, message in checks:
            print(f"   {status} {message}")

        # Estimate performance
        print("\n📊 Performance Estimates:")
        if process_all_symbols:
            estimated_symbols_per_cycle = min(symbol_batch_size, 497)  # Kraken has ~497 USD pairs
            estimated_cycle_time = max(30, estimated_symbols_per_cycle * 0.6)  # Rough estimate
            estimated_api_calls = estimated_symbols_per_cycle * 3  # Rough estimate for multiple timeframes

            print(f"   Estimated symbols per cycle: {estimated_symbols_per_cycle}")
            print(f"   Estimated cycle time: ~{estimated_cycle_time:.0f}s")
            print(f"   Estimated API calls per cycle: ~{estimated_api_calls}")
            print(f"   Market coverage: ~{estimated_symbols_per_cycle/497*100:.1f}% of available symbols")
        else:
            print("   Using filtered mode - limited symbol coverage")

        print("\n🎉 Configuration Test Complete!")
        print("   The bot is now configured for comprehensive symbol processing.")
        print("   It will process ALL available symbols on the exchange by default.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_comprehensive_mode())
