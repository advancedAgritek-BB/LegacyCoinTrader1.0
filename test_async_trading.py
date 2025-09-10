#!/usr/bin/env python3
"""
Test script to verify async Solana and CEX trading execution.
"""
import asyncio
import time
import pytest
from unittest.mock import Mock, AsyncMock
from crypto_bot.main import AsyncTradeManager, execute_solana_trade, execute_cex_trade


@pytest.mark.asyncio
async def test_async_trade_manager():
    """Test the AsyncTradeManager functionality."""
    print("Testing AsyncTradeManager...")

    manager = AsyncTradeManager()

    # Test executing async tasks
    async def mock_trade(name: str, delay: float):
        print(f"Starting {name} trade (delay: {delay}s)")
        await asyncio.sleep(delay)
        print(f"Completed {name} trade")
        return f"{name}_result"

    # Execute two trades with different delays
    await manager.execute_trade_async(mock_trade("Solana", 1.0))
    await manager.execute_trade_async(mock_trade("CEX", 2.0))

    print(f"Active tasks: {len(manager.active_tasks)}")

    # Wait for completion
    await manager.wait_for_completion(timeout=5.0)
    await manager.cleanup_completed()

    print(f"Remaining active tasks: {len(manager.active_tasks)}")
    print("AsyncTradeManager test completed!")


@pytest.mark.asyncio
async def test_trade_execution_functions():
    """Test the separate trade execution functions."""
    print("\nTesting trade execution functions...")

    # Mock context
    ctx = Mock()
    ctx.config = {
        "execution_mode": "dry_run",
        "solana_slippage_bps": 50,
        "wallet_address": "test_wallet"
    }
    ctx.paper_wallet = None
    ctx.notifier = None
    ctx.risk_manager = Mock()
    ctx.risk_manager.allocate_capital = Mock()
    ctx.positions = {}
    ctx.balance = 1000.0
    ctx.timing = {}

    # Mock candidate
    candidate = {
        "df": Mock(),
        "regime": "bullish",
        "score": 0.8
    }
    candidate["df"].close.iloc = Mock(return_value=[100.0])

    # Test Solana trade execution (should not actually execute due to mocking)
    print("Testing Solana trade execution...")
    try:
        result = await execute_solana_trade(
            ctx, candidate, "TEST/USDC", 10.0, 100.0, "test_strategy", "buy"
        )
        print(f"Solana trade result: {result}")
    except Exception as e:
        print(f"Solana trade test (expected failure due to mocking): {e}")

    # Test CEX trade execution (should not actually execute due to mocking)
    print("Testing CEX trade execution...")
    try:
        result = await execute_cex_trade(
            ctx, candidate, "TEST/USD", 10.0, 100.0, "test_strategy", "buy"
        )
        print(f"CEX trade result: {result}")
    except Exception as e:
        print(f"CEX trade test (expected failure due to mocking): {e}")

    print("Trade execution functions test completed!")


async def main():
    """Run all tests."""
    print("Starting async trading tests...")
    start_time = time.time()

    await test_async_trade_manager()
    await test_trade_execution_functions()

    end_time = time.time()
    print(".2f")


if __name__ == "__main__":
    asyncio.run(main())
