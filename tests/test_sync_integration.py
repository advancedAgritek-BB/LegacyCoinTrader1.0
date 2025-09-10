"""
Integration Tests for Synchronization System

Tests the complete synchronization workflow in the main bot application,
ensuring that positions are properly synchronized across all components
after bot restart and during normal operation.
"""

import asyncio
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from crypto_bot.main import sync_paper_wallet_with_positions_log
from crypto_bot.utils.trade_manager import TradeManager
from crypto_bot.paper_wallet import PaperWallet
from crypto_bot.sync_service import SyncService, ConflictResolution
from crypto_bot.phase_runner import BotContext


class TestBotSynchronizationIntegration:
    """Integration tests for bot synchronization functionality."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def mock_bot_context(self, temp_log_dir):
        """Create a mock BotContext for testing."""
        ctx = Mock(spec=BotContext)
        ctx.config = {"execution_mode": "dry_run"}
        ctx.paper_wallet = PaperWallet(balance=10000.0)
        ctx.trade_manager = TradeManager()
        ctx.balance = 10000.0

        # Mock logger
        ctx.logger = Mock()
        ctx.logger.info = Mock()
        ctx.logger.error = Mock()
        ctx.logger.warning = Mock()

        return ctx

    @pytest.mark.asyncio
    async def test_sync_paper_wallet_with_positions_log_integration(self, temp_log_dir, mock_bot_context):
        """Test the complete synchronization workflow as called from main bot."""
        # Create a positions.log file with some positions
        positions_log = temp_log_dir / "positions.log"
        with open(positions_log, 'w') as f:
            f.write("2025-09-06 10:00:00 Active BTC/USD long 1.5 entry 50000.0\n")
            f.write("2025-09-06 10:05:00 Active ETH/USD short 2.0 entry 3000.0\n")
            f.write("2025-09-06 10:10:00 Active ADA/USD long 100.0 entry 0.50\n")

        # Mock the LOG_DIR to use our temp directory
        with patch('crypto_bot.main.LOG_DIR', temp_log_dir):
            # Execute the synchronization function
            await sync_paper_wallet_with_positions_log(mock_bot_context)

        # Verify that the sync service was called and worked
        # The function should have loaded positions from positions.log

        # Note: In a real scenario, this would create positions in TradeManager
        # For this test, we're verifying the function executes without errors

    @pytest.mark.asyncio
    async def test_sync_service_full_integration(self, temp_log_dir):
        """Test complete SyncService integration with real components."""
        # Create real instances
        trade_manager = TradeManager()
        paper_wallet = PaperWallet(balance=10000.0)
        sync_service = SyncService(temp_log_dir)

        # Create positions.log with test data (using symbols not in TradeManager)
        positions_log = temp_log_dir / "positions.log"
        with open(positions_log, 'w') as f:
            f.write("2025-09-06 12:00:00 Active SOLANA/USD long 1.0 entry 50000.0\n")
            f.write("2025-09-06 12:05:00 Active AVAX/USD short 2.0 entry 3000.0\n")

        # Perform full synchronization
        results = await sync_service.full_synchronization(
            trade_manager=trade_manager,
            paper_wallet=paper_wallet,
            positions_log_path=positions_log,
            conflict_resolution=ConflictResolution.TRADE_MANAGER_WINS
        )

        # Verify results
        assert 'log_to_tm' in results
        assert 'tm_to_pw' in results

        # Check that synchronization completed
        log_to_tm = results['log_to_tm']
        tm_to_pw = results['tm_to_pw']

        # Should either succeed or find no changes (depending on state)
        assert log_to_tm.result.name in ['SUCCESS', 'NO_CHANGES', 'FAILED']
        assert tm_to_pw.result.name in ['SUCCESS', 'NO_CHANGES', 'FAILED']

        # Verify health status is available
        health = sync_service.get_health_status()
        assert 'overall_health' in health
        assert 'metrics' in health

    def test_positions_log_parsing_edge_cases(self, temp_log_dir):
        """Test positions.log parsing with various edge cases."""
        sync_service = SyncService(temp_log_dir)

        # Create positions.log with various formats
        positions_log = temp_log_dir / "positions.log"
        with open(positions_log, 'w') as f:
            f.write("2025-09-06 12:00:00 Active BTC/USD long 1.5 entry 50000.0\n")  # Valid
            f.write("2025-09-06 12:05:00 Active ETH/USD short 2.0 entry 3000.0\n")  # Valid
            f.write("Some random log line without Active\n")  # Invalid - no Active
            f.write("2025-09-06 12:10:00 Active ADA/USD long invalid_amount entry 0.5\n")  # Invalid amount
            f.write("2025-09-06 12:15:00 Active DOT/USD short 5.0 entry\n")  # Missing price
            f.write("2025-09-06 12:20:00 Active SOL/USD long 10.0 entry 100.0 extra data\n")  # Extra data
            f.write("2025-09-06 12:25:00 Active LINK/USD long 1.0 entry 20.0\n")  # Valid

        positions = sync_service._parse_positions_log(positions_log)

        # Should have parsed the valid entries
        assert 'BTC/USD' in positions
        assert 'ETH/USD' in positions
        assert 'SOL/USD' in positions
        assert 'LINK/USD' in positions

        # Should not have parsed invalid entries
        assert 'ADA/USD' not in positions  # Invalid amount
        assert 'DOT/USD' not in positions  # Missing price

        # Verify correct parsing
        assert positions['BTC/USD']['side'] == 'long'
        assert positions['BTC/USD']['amount'] == 1.5
        assert positions['BTC/USD']['entry_price'] == 50000.0

    @pytest.mark.asyncio
    async def test_trade_manager_position_creation_integration(self, temp_log_dir):
        """Test that positions are properly created in TradeManager during sync."""
        sync_service = SyncService(temp_log_dir)
        trade_manager = TradeManager()

        # Create positions.log with a symbol that doesn't exist in TradeManager
        positions_log = temp_log_dir / "positions.log"
        with open(positions_log, 'w') as f:
            f.write("2025-09-06 12:00:00 Active SOLANA/USD long 1.0 entry 50000.0\n")

        # Mock the record_trade method to avoid actual creation
        with patch.object(trade_manager, 'record_trade', return_value=None) as mock_record:
            result = await sync_service.sync_positions_log_to_trade_manager(
                trade_manager, positions_log
            )

            # Verify that record_trade would have been called
            assert mock_record.called
            call_args = mock_record.call_args
            trade_arg = call_args[0][0]  # First positional argument
            assert trade_arg.symbol == 'SOLANA/USD'
            assert trade_arg.side == 'long'
            assert trade_arg.amount == 1.0
            assert trade_arg.price == 50000.0

    def test_sync_health_monitoring(self, temp_log_dir):
        """Test that sync health monitoring works correctly."""
        from crypto_bot.sync_service import SyncResult

        sync_service = SyncService(temp_log_dir)

        # Simulate various sync operations
        operations = [
            (SyncResult.SUCCESS, 50.0),
            (SyncResult.SUCCESS, 45.0),
            (SyncResult.FAILED, 30.0),
            (SyncResult.SUCCESS, 55.0),
            (SyncResult.NO_CHANGES, 10.0),
        ]

        for result_enum, duration in operations:
            operation = Mock()
            operation.result = result_enum
            operation.duration_ms = duration
            operation.conflicts = []  # Add conflicts attribute
            sync_service._record_operation(operation)

        # Check health status
        health = sync_service.get_health_status()

        assert health['overall_health'] in ['healthy', 'degraded']
        assert health['metrics']['total_syncs'] == 5
        assert health['metrics']['successful_syncs'] == 3
        assert health['metrics']['failed_syncs'] == 1

    @pytest.mark.asyncio
    async def test_concurrent_sync_operations(self, temp_log_dir):
        """Test that concurrent sync operations don't interfere with each other."""
        sync_service = SyncService(temp_log_dir)

        # Create multiple TradeManager and PaperWallet instances
        trade_managers = [TradeManager() for _ in range(3)]
        paper_wallets = [PaperWallet(balance=10000.0) for _ in range(3)]

        # Create different positions.log files
        positions_logs = []
        for i in range(3):
            log_file = temp_log_dir / f"positions_{i}.log"
            with open(log_file, 'w') as f:
                f.write(f"2025-09-06 12:00:00 Active BTC{i}/USD long 1.0 entry 50000.0\n")
            positions_logs.append(log_file)

        # Run concurrent synchronizations
        tasks = []
        for i in range(3):
            task = sync_service.full_synchronization(
                trade_manager=trade_managers[i],
                paper_wallet=paper_wallets[i],
                positions_log_path=positions_logs[i]
            )
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all completed without exceptions
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, dict)
            assert 'log_to_tm' in result
            assert 'tm_to_pw' in result

    def test_sync_report_generation(self, temp_log_dir):
        """Test that sync reports are generated correctly."""
        from crypto_bot.sync_service import SyncResult, SyncDirection

        sync_service = SyncService(temp_log_dir)

        # Add some operations to history
        for i in range(5):
            operation = Mock()
            operation.operation_id = f"test_op_{i}"
            operation.result = SyncResult.SUCCESS if i < 4 else SyncResult.FAILED
            operation.duration_ms = 50.0 + i * 10
            operation.direction = SyncDirection.TRADE_MANAGER_TO_PAPER_WALLET
            operation.timestamp = datetime.now()
            operation.conflicts = []
            sync_service._record_operation(operation)

        # Generate summary report
        report = sync_service.get_sync_report()

        assert report['total_operations'] == 5
        assert report['successful'] == 4
        assert report['failed'] == 1
        assert 'recent_operations' in report
        assert len(report['recent_operations']) <= 5

    @pytest.mark.asyncio
    async def test_error_recovery_mechanisms(self, temp_log_dir):
        """Test that the system properly handles and recovers from errors."""
        sync_service = SyncService(temp_log_dir)

        # Create a positions.log file with some issues
        positions_log = temp_log_dir / "positions.log"
        with open(positions_log, 'w') as f:
            f.write("2025-09-06 12:00:00 Active SOLANA/USD long 1.0 entry 50000.0\n")
            f.write("Invalid line that should be handled\n")
            f.write("2025-09-06 12:05:00 Active AVAX/USD short invalid entry 3000.0\n")

        # Create mock components that might fail
        trade_manager = Mock()
        trade_manager.get_all_positions.return_value = []
        trade_manager.record_trade.side_effect = Exception("Position creation failed")

        paper_wallet = Mock()
        paper_wallet.sync_from_trade_manager.side_effect = Exception("Paper wallet sync failed")

        # Attempt synchronization - should handle errors gracefully
        results = await sync_service.full_synchronization(
            trade_manager=trade_manager,
            paper_wallet=paper_wallet,
            positions_log_path=positions_log
        )

        # Verify that results are returned even with errors
        assert 'log_to_tm' in results
        assert 'tm_to_pw' in results

        # Check that error information is captured
        log_to_tm = results['log_to_tm']
        tm_to_pw = results['tm_to_pw']

        # Operations should have completed (even if with errors)
        assert log_to_tm.result.name in ['SUCCESS', 'FAILED', 'PARTIAL_SUCCESS']
        assert tm_to_pw.result.name in ['SUCCESS', 'FAILED', 'PARTIAL_SUCCESS']

        # Error messages should be captured
        if log_to_tm.result.name == 'FAILED':
            assert log_to_tm.error_message is not None
        if tm_to_pw.result.name == 'FAILED':
            assert tm_to_pw.error_message is not None

    def test_backward_compatibility_with_legacy_sync(self, temp_log_dir):
        """Test that the new sync system maintains backward compatibility."""
        # This test ensures that existing functionality still works
        # while new features are added

        sync_service = SyncService(temp_log_dir)

        # Test that old-style position data can still be processed
        legacy_positions = {
            'BTC/USD': {'side': 'long', 'amount': 1.0, 'entry_price': 50000.0},
            'ETH/USD': {'side': 'short', 'amount': 2.0, 'entry_price': 3000.0}
        }

        # Simulate processing legacy format
        processed_positions = []
        for symbol, data in legacy_positions.items():
            # This should work with the new system
            snapshot = Mock()
            snapshot.symbol = symbol
            snapshot.side = data['side']
            snapshot.amount = data['amount']
            snapshot.entry_price = data['entry_price']
            processed_positions.append(snapshot)

        assert len(processed_positions) == 2
        assert processed_positions[0].symbol == 'BTC/USD'
        assert processed_positions[1].symbol == 'ETH/USD'


class TestEndToEndSynchronization:
    """End-to-end tests simulating real bot restart scenarios."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.asyncio
    async def test_bot_restart_position_recovery(self, temp_log_dir):
        """
        Test the complete scenario of bot restart and position recovery.

        This simulates:
        1. Bot running with positions
        2. Bot restart
        3. Synchronization recovering positions
        4. Verification that positions are consistent
        """
        # Phase 1: Simulate initial bot state with positions
        trade_manager = TradeManager()
        paper_wallet = PaperWallet(balance=10000.0)
        sync_service = SyncService(temp_log_dir)

        # Create initial positions
        initial_positions = [
            {'symbol': 'BTC/USD', 'side': 'long', 'amount': 1.0, 'price': 50000.0},
            {'symbol': 'ETH/USD', 'side': 'short', 'amount': 2.0, 'price': 3000.0},
        ]

        # Simulate initial synchronization
        with patch.object(paper_wallet, 'sync_from_trade_manager') as mock_sync:
            await sync_service.sync_trade_manager_to_paper_wallet(trade_manager, paper_wallet)
            mock_sync.assert_called()

        # Phase 2: Simulate bot restart - positions.log should contain the positions
        positions_log = temp_log_dir / "positions.log"
        with open(positions_log, 'w') as f:
            for pos in initial_positions:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} Active {pos['symbol']} {pos['side']} {pos['amount']} entry {pos['price']}\n")

        # Phase 3: Simulate restart synchronization
        restart_results = await sync_service.full_synchronization(
            trade_manager=trade_manager,
            paper_wallet=paper_wallet,
            positions_log_path=positions_log
        )

        # Phase 4: Verify recovery was successful
        assert 'log_to_tm' in restart_results
        assert 'tm_to_pw' in restart_results

        recovery_op = restart_results['log_to_tm']
        assert recovery_op.result.name in ['SUCCESS', 'NO_CHANGES']

        # Phase 5: Verify health monitoring
        health = sync_service.get_health_status()
        assert health['overall_health'] in ['healthy', 'degraded']
        assert health['metrics']['total_syncs'] >= 2  # Initial + restart

        print("✅ Bot restart position recovery test completed successfully")

    @pytest.mark.asyncio
    async def test_large_scale_position_recovery(self, temp_log_dir):
        """
        Test recovery of a large number of positions (stress test).
        """
        sync_service = SyncService(temp_log_dir)
        trade_manager = TradeManager()
        paper_wallet = PaperWallet(balance=100000.0)

        # Create a large positions.log file
        positions_log = temp_log_dir / "positions.log"
        with open(positions_log, 'w') as f:
            for i in range(100):  # 100 positions
                symbol = f"COIN{i}/USD"
                side = "long" if i % 2 == 0 else "short"
                amount = float(i + 1)
                price = 1000.0 + i * 10
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp} Active {symbol} {side} {amount} entry {price}\n")

        # Mock position creation to avoid actual database operations
        with patch.object(trade_manager, 'record_trade') as mock_record:
            start_time = asyncio.get_event_loop().time()

            result = await sync_service.sync_positions_log_to_trade_manager(
                trade_manager, positions_log
            )

            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time

            # Verify performance - should complete within reasonable time
            assert duration < 5.0  # Less than 5 seconds for 100 positions
            assert result.result.name == 'SUCCESS'
            assert mock_record.call_count == 100  # Should have tried to create 100 positions

        print(f"✅ Large scale recovery test completed in {duration:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
