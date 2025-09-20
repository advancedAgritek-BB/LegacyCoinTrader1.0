#!/usr/bin/env python3
"""
Demo script to showcase the new pipeline logging system.

This script demonstrates how the enhanced logging system provides
readable, meaningful logs throughout the scanning, evaluation, and
trading pipeline with proper context and performance metrics.
"""

import asyncio
import time
from crypto_bot.utils.pipeline_logger import get_pipeline_logger, pipeline_context
from crypto_bot.utils.logger import setup_logger

async def demo_pipeline_logging():
    """Demonstrate the enhanced pipeline logging system."""

    print("🚀 Starting Pipeline Logging Demo")
    print("=" * 50)

    # Get the pipeline logger
    pipeline_logger = get_pipeline_logger("demo_pipeline")

    async with pipeline_context(cycle_id=1) as logger:
        # Phase 1: Discovery
        logger.start_phase("discovery", "Scanning markets for trading opportunities")

        # Simulate market discovery
        await asyncio.sleep(0.1)  # Simulate work

        logger.log_discovery("BTC/USD", "exchange", confidence=0.95)
        logger.log_discovery("ETH/USD", "exchange", confidence=0.92)
        logger.log_discovery("SOL/USD", "dex_scanner", confidence=0.85)
        logger.log_discovery("ADA/USD", "cex_scanner", confidence=0.78)

        logger.end_phase("discovery")

        # Phase 2: Market Data Loading
        logger.start_phase("market_data", "Loading OHLCV data and indicators")

        await asyncio.sleep(0.2)  # Simulate data loading
        logger.log_performance("OHLCV data loading", 0.15)
        logger.log_performance("Technical analysis", 0.08)

        logger.end_phase("market_data")

        # Phase 3: Strategy Evaluation
        logger.start_phase("evaluation", "Evaluating strategies and generating signals")

        await asyncio.sleep(0.1)  # Simulate evaluation

        logger.log_evaluation("BTC/USD", "momentum", 0.87, "long")
        logger.log_evaluation("ETH/USD", "mean_reversion", 0.76, "short")
        logger.log_evaluation("SOL/USD", "breakout", 0.91, "long")
        logger.log_evaluation("ADA/USD", "trend_following", 0.65, "hold")

        logger.log_performance("Strategy evaluation", 0.12)

        logger.end_phase("evaluation")

        # Phase 4: Trade Execution
        logger.start_phase("execution", "Executing trades based on signals")

        await asyncio.sleep(0.05)  # Simulate execution

        # Simulate successful trades
        logger.log_trade_attempt("BTC/USD", "buy", 0.05, 45000.0, "High confidence momentum signal")
        logger.log_trade_execution("BTC/USD", "buy", 0.05, 45000.0, "ORDER_12345", 0.0)

        logger.log_trade_attempt("SOL/USD", "buy", 10.0, 120.0, "Breakout signal")
        logger.log_trade_execution("SOL/USD", "buy", 10.0, 120.0, "ORDER_12346", 0.0)

        # Simulate failed trade
        logger.log_trade_attempt("ETH/USD", "sell", 0.8, 2800.0, "Mean reversion signal")
        logger.log_error("Insufficient balance for ETH/USD trade", "ETH/USD", "execution")

        logger.log_performance("Trade execution", 0.08)

        logger.end_phase("execution")

        # Phase 5: Finalization
        logger.start_phase("finalization", "Finalizing cycle and calculating P&L")

        await asyncio.sleep(0.02)  # Simulate finalization

        # Calculate mock P&L
        total_volume = (0.05 * 45000.0) + (10.0 * 120.0)
        mock_pnl = total_volume * 0.002  # 0.2% profit

        logger.end_phase("finalization")

        # Complete the pipeline
        logger.complete_pipeline(success=True, final_pnl=mock_pnl)

    print("\n" + "=" * 50)
    print("✅ Pipeline Logging Demo Complete")
    print("\nKey improvements demonstrated:")
    print("• 📊 Phase-by-phase progress tracking")
    print("• 🎯 Readable trade execution logs")
    print("• ⚡ Performance timing for each operation")
    print("• 📈 Meaningful context with symbols and strategies")
    print("• 💰 Clear P&L reporting")
    print("• 🔍 Structured data for analysis while maintaining readability")

def demo_readable_vs_json_logging():
    """Compare readable vs JSON logging formats."""

    print("\n" + "=" * 60)
    print("📋 Readable vs JSON Logging Comparison")
    print("=" * 60)

    # Setup readable logger
    readable_logger = setup_logger("demo_readable", formatter="readable")

    # Setup JSON logger
    json_logger = setup_logger("demo_json", formatter="json")

    print("\n🎨 Readable Format:")
    print("-" * 30)
    readable_logger.info(
        "BTC/USD momentum signal generated",
        extra={
            "symbol": "BTC/USD",
            "strategy": "momentum",
            "score": 0.87,
            "direction": "long",
            "confidence": 0.85
        }
    )

    print("\n📊 JSON Format:")
    print("-" * 30)
    json_logger.info(
        "BTC/USD momentum signal generated",
        extra={
            "symbol": "BTC/USD",
            "strategy": "momentum",
            "score": 0.87,
            "direction": "long",
            "confidence": 0.85
        }
    )

    print("\n✨ Benefits of Readable Format:")
    print("• Human-friendly timestamps and messages")
    print("• Contextual information at a glance")
    print("• Clean formatting with emojis for quick identification")
    print("• Structured data still available for programmatic analysis")
    print("• Correlation IDs for tracking related operations")

if __name__ == "__main__":
    print("LegacyCoinTrader Enhanced Logging Demo")
    print("======================================")

    # Run the pipeline demo
    asyncio.run(demo_pipeline_logging())

    # Show format comparison
    demo_readable_vs_json_logging()

    print("\n" + "=" * 60)
    print("🎉 Demo complete! Check the logs above to see the improvements.")
    print("The new logging system provides:")
    print("• Clear pipeline flow visibility")
    print("• Meaningful context without ID clutter")
    print("• Performance metrics and timing")
    print("• Human-readable trade execution details")
    print("• Comprehensive cycle summaries")
    print("=" * 60)
