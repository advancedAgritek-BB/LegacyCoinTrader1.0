#!/usr/bin/env python3
"""
Test script to verify Solana trading features work in paper trading mode.

This script tests:
1. Paper wallet integration with Solana trading
2. RapidExecutor paper trading functionality
3. PumpSniperOrchestrator paper trading
4. SniperRiskManager paper trading integration
5. End-to-end paper trading workflow
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.solana.rapid_executor import RapidExecutor, ExecutionParams
from crypto_bot.solana.pump_sniper_orchestrator import PumpSniperOrchestrator
from crypto_bot.solana.sniper_risk_manager import SniperRiskManager
from crypto_bot.solana.pump_detector import PoolAnalysis
from crypto_bot.solana.pool_analyzer import PoolMetrics
from crypto_bot.solana.watcher import NewPoolEvent


class TestResults:
    """Track test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def test_passed(self, test_name: str):
        print(f"✅ {test_name} - PASSED")
        self.passed += 1

    def test_failed(self, test_name: str, error: str):
        print(f"❌ {test_name} - FAILED: {error}")
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        success_rate = (self.passed / total * 100) if total > 0 else 0
        print("\n📊 Test Summary:")
        print(f"   Total Tests: {total}")
        print(".1f")
        print(f"   Passed: {self.passed}")
        print(f"   Failed: {self.failed}")

        if self.errors:
            print("\n❌ Failed Tests:")
            for error in self.errors:
                print(f"   - {error}")

        return self.failed == 0


async def test_paper_wallet_integration():
    """Test basic paper wallet functionality."""
    print("\n🧪 Testing Paper Wallet Integration...")

    # Create paper wallet
    wallet = PaperWallet(balance=1000.0)

    # Test basic operations
    trade_id = wallet.open("SOL/USD", "buy", 10.0, 50.0)
    assert trade_id is not None, "Failed to open position"

    pnl = wallet.close("SOL/USD", 10.0, 55.0)
    assert pnl == 50.0, f"Expected PnL 50.0, got {pnl}"

    assert wallet.balance == 1050.0, f"Expected balance 1050.0, got {wallet.balance}"

    return True


async def test_rapid_executor_paper_trading(results: TestResults):
    """Test RapidExecutor paper trading functionality."""
    print("\n🧪 Testing RapidExecutor Paper Trading...")

    try:
        # Create paper wallet
        wallet = PaperWallet(balance=100.0)

        # Create executor in dry run mode
        config = {
            "rapid_executor": {
                "default_slippage_pct": 0.03,
                "base_position_size_sol": 0.1
            }
        }
        executor = RapidExecutor(config, dry_run=True, paper_wallet=wallet)

        # Test execution parameters
        params = ExecutionParams(
            token_mint="So11111111111111111111111111111111111111112",  # SOL
            side="buy",
            amount_sol=0.1,
            max_slippage_pct=0.05
        )

        # Mock route data
        route = {"price": 50.0}

        # Execute paper trade
        result = await executor._execute_paper_trade(params, route)

        # Verify result
        assert result.success, f"Trade failed: {result.error_message}"
        assert result.execution_mode == "paper", "Should be paper mode"
        assert result.transaction_hash.startswith("paper_tx_"), "Invalid tx hash format"
        assert result.executed_amount == 0.1, "Executed amount mismatch"

        # Check paper wallet was updated
        assert len(wallet.positions) > 0, "Paper wallet should have positions"

        results.test_passed("RapidExecutor Paper Trading")

    except Exception as e:
        results.test_failed("RapidExecutor Paper Trading", str(e))


async def test_sniper_risk_manager_paper_trading(results: TestResults):
    """Test SniperRiskManager paper trading integration."""
    print("\n🧪 Testing SniperRiskManager Paper Trading...")

    try:
        # Create paper wallet
        wallet = PaperWallet(balance=100.0)

        # Create risk manager in dry run mode
        config = {
            "sniper_risk_manager": {
                "default_profile": "moderate"
            }
        }
        risk_manager = SniperRiskManager(config, dry_run=True, paper_wallet=wallet)

        # Test balance retrieval - this is the key paper trading integration
        balance = risk_manager._get_account_balance()
        assert balance == 100.0, f"Expected balance 100.0, got {balance}"

        # Test that dry_run and paper_wallet are properly set
        assert risk_manager.dry_run == True, "Should be in dry run mode"
        assert risk_manager.paper_wallet == wallet, "Paper wallet should be set"

        results.test_passed("SniperRiskManager Paper Trading")

    except Exception as e:
        results.test_failed("SniperRiskManager Paper Trading", str(e))


async def test_pump_sniper_orchestrator_paper_trading(results: TestResults):
    """Test PumpSniperOrchestrator paper trading functionality."""
    print("\n🧪 Testing PumpSniperOrchestrator Paper Trading...")

    try:
        # Create paper wallet
        wallet = PaperWallet(balance=100.0)

        # Set environment variable to avoid API requirement
        import os
        original_helius_key = os.environ.get('HELIUS_KEY')
        os.environ['HELIUS_KEY'] = 'test_key'

        try:
            # Create orchestrator in dry run mode with minimal config
            config = {
                "pump_sniper_orchestrator": {
                    "enabled": True,
                    "min_decision_confidence": 0.7,
                    "decision_weights": {
                        "pump_probability": 0.25,
                        "pool_quality": 0.20,
                        "sentiment_score": 0.15,
                        "momentum_score": 0.15,
                        "risk_score": 0.15,
                        "timing_score": 0.10
                    }
                },
                "rapid_executor": {
                    "default_slippage_pct": 0.03,
                    "base_position_size_sol": 0.1
                },
                "sniper_risk_manager": {
                    "default_profile": "moderate"
                }
            }

            orchestrator = PumpSniperOrchestrator(config, dry_run=True, paper_wallet=wallet)

            # Test that components are initialized correctly
            assert orchestrator.dry_run == True, "Should be in dry run mode"
            assert orchestrator.paper_wallet == wallet, "Paper wallet not set correctly"
            assert orchestrator.rapid_executor.dry_run == True, "RapidExecutor should be in dry run"
            assert orchestrator.risk_manager.dry_run == True, "RiskManager should be in dry run"

            results.test_passed("PumpSniperOrchestrator Paper Trading")

        finally:
            # Restore original environment
            if original_helius_key:
                os.environ['HELIUS_KEY'] = original_helius_key
            elif 'HELIUS_KEY' in os.environ:
                del os.environ['HELIUS_KEY']

    except Exception as e:
        results.test_failed("PumpSniperOrchestrator Paper Trading", str(e))


async def test_end_to_end_paper_trading_workflow(results: TestResults):
    """Test end-to-end paper trading workflow."""
    print("\n🧪 Testing End-to-End Paper Trading Workflow...")

    try:
        # Create paper wallet with initial balance
        wallet = PaperWallet(balance=1000.0)

        # Setup configuration
        config = {
            "rapid_executor": {
                "default_slippage_pct": 0.03,
                "base_position_size_sol": 0.1
            },
            "sniper_risk_manager": {
                "default_profile": "moderate"
            }
        }

        # Create components
        executor = RapidExecutor(config, dry_run=True, paper_wallet=wallet)
        risk_manager = SniperRiskManager(config, dry_run=True, paper_wallet=wallet)

        # Simulate a trading scenario
        initial_balance = wallet.balance

        # Create execution parameters for a buy trade
        params = ExecutionParams(
            token_mint="So11111111111111111111111111111111111111112",
            side="buy",
            amount_sol=0.5,
            max_slippage_pct=0.05
        )

        # Mock analysis data with all required fields
        pool_analysis = PoolAnalysis(
            pool_address="test_pool_address",
            token_mint=params.token_mint,
            initial_liquidity=10000.0,
            current_liquidity=15000.0,
            liquidity_change_rate=0.5,
            liquidity_stability_score=0.8,
            transaction_velocity=100,
            unique_wallets=50,
            avg_transaction_size=0.1,
            whale_activity_score=0.7,
            price_momentum=0.8,
            volume_spike_factor=1.5,
            volume_consistency=0.9,
            price_stability=0.85,
            social_buzz_score=0.6,
            sentiment_score=0.7,
            influencer_mentions=10,
            rsi=65.0,
            bollinger_position=0.3,
            volume_profile_score=0.75,
            dev_activity_score=0.8,
            tokenomics_score=0.9,
            pump_probability=0.8,
            timing_score=0.9,
            rug_risk_score=0.1,
            risk_adjusted_score=0.75
        )

        pool_metrics = PoolMetrics(
            total_liquidity_usd=50000,
            sniping_viability=0.8,
            slippage_1pct=0.02
        )

        # Check risk validation
        position_size = risk_manager.calculate_position_size(pool_analysis, pool_metrics, initial_balance)
        is_valid, reason = risk_manager.validate_new_position(pool_analysis, pool_metrics, position_size)

        if is_valid:
            # Execute the trade
            route = {"price": 50.0}
            result = await executor._execute_paper_trade(params, route)

            # Verify the trade was successful
            assert result.success, f"Trade failed: {result.error_message}"

            # Check that balance was updated
            assert wallet.balance < initial_balance, "Balance should decrease after buying"

            # Check that position was created
            assert len(wallet.positions) > 0, "Should have open positions"

            # Simulate selling the position
            sell_params = ExecutionParams(
                token_mint=params.token_mint,
                side="sell",
                amount_sol=params.amount_sol,
                max_slippage_pct=0.05
            )

            sell_result = await executor._execute_paper_trade(sell_params, {"price": 55.0})
            assert sell_result.success, f"Sell trade failed: {sell_result.error_message}"

            # Check final balance (should be higher due to profit)
            final_balance = wallet.balance
            assert final_balance > initial_balance, f"Final balance {final_balance} should be > initial {initial_balance}"

            results.test_passed("End-to-End Paper Trading Workflow")
        else:
            results.test_failed("End-to-End Paper Trading Workflow", f"Risk validation failed: {reason}")

    except Exception as e:
        results.test_failed("End-to-End Paper Trading Workflow", str(e))


async def main():
    """Run all Solana paper trading tests."""
    print("🚀 Starting Solana Paper Trading Tests...")
    print("=" * 60)

    results = TestResults()

    # Test basic paper wallet functionality
    try:
        await test_paper_wallet_integration()
        results.test_passed("Paper Wallet Basic Functionality")
    except Exception as e:
        results.test_failed("Paper Wallet Basic Functionality", str(e))

    # Test individual components
    await test_rapid_executor_paper_trading(results)
    await test_sniper_risk_manager_paper_trading(results)
    await test_pump_sniper_orchestrator_paper_trading(results)

    # Test end-to-end workflow
    await test_end_to_end_paper_trading_workflow(results)

    print("\n" + "=" * 60)

    # Print summary
    success = results.summary()

    if success:
        print("\n🎉 All Solana paper trading tests PASSED!")
        return 0
    else:
        print("\n💥 Some Solana paper trading tests FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
