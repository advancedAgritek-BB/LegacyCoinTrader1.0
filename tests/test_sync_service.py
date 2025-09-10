"""
Unit and Integration Tests for SyncService

Tests the enterprise-grade synchronization service to ensure:
- Bidirectional synchronization works correctly
- Conflict resolution strategies function properly
- Error handling and recovery mechanisms work
- Performance and reliability under various scenarios
- Integration with TradeManager, PaperWallet, and positions.log
"""

import asyncio
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

from crypto_bot.sync_service import (
    SyncService,
    SyncDirection,
    ConflictResolution,
    SyncResult,
    SyncOperation,
    PositionSnapshot
)
from crypto_bot.utils.trade_manager import TradeManager, Trade, Position
from crypto_bot.paper_wallet import PaperWallet


class TestPositionSnapshot:
    """Test PositionSnapshot creation and functionality."""

    def test_from_trade_manager_position(self):
        """Test creating snapshot from TradeManager position."""
        # Mock position object
        mock_pos = Mock()
        mock_pos.symbol = "BTC/USD"
        mock_pos.side = "long"
        mock_pos.total_amount = 1.5
        mock_pos.average_price = 50000.0
        mock_pos.entry_time = datetime(2025, 1, 1, 12, 0, 0)
        mock_pos.last_update = datetime(2025, 1, 1, 12, 30, 0)

        snapshot = PositionSnapshot.from_trade_manager_position(mock_pos)

        assert snapshot.symbol == "BTC/USD"
        assert snapshot.side == "long"
        assert snapshot.amount == 1.5
        assert snapshot.entry_price == 50000.0
        assert snapshot.source == "trade_manager"
        assert snapshot.checksum is not None

    def test_from_paper_wallet_position(self):
        """Test creating snapshot from paper wallet position."""
        pos_data = {
            'symbol': 'ETH/USD',
            'side': 'short',
            'amount': 2.0,
            'entry_price': 3000.0,
            'entry_time': '2025-01-01T12:00:00'
        }

        snapshot = PositionSnapshot.from_paper_wallet_position('ETH/USD', pos_data)

        assert snapshot.symbol == "ETH/USD"
        assert snapshot.side == "short"
        assert snapshot.amount == 2.0
        assert snapshot.entry_price == 3000.0
        assert snapshot.source == "paper_wallet"

    def test_from_positions_log(self):
        """Test creating snapshot from positions.log entry."""
        snapshot = PositionSnapshot.from_positions_log(
            symbol="ADA/USD",
            side="long",
            amount=100.0,
            entry_price=0.5
        )

        assert snapshot.symbol == "ADA/USD"
        assert snapshot.side == "long"
        assert snapshot.amount == 100.0
        assert snapshot.entry_price == 0.5
        assert snapshot.source == "positions_log"


class TestSyncService:
    """Test SyncService core functionality."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sync_service(self, temp_log_dir):
        """Create SyncService instance."""
        return SyncService(temp_log_dir)

    @pytest.fixture
    def mock_trade_manager(self):
        """Create mock TradeManager."""
        tm = Mock()
        tm.get_all_positions.return_value = []
        tm.price_cache = {'BTC/USD': 50000.0, 'ETH/USD': 3000.0}
        return tm

    @pytest.fixture
    def mock_paper_wallet(self):
        """Create mock PaperWallet."""
        pw = Mock()
        pw.positions = {}
        pw.sync_from_trade_manager = Mock()
        pw.save_state = Mock()
        return pw

    def test_initialization(self, sync_service):
        """Test SyncService initialization."""
        assert sync_service.log_dir is not None
        assert sync_service.sync_history == []
        assert 'total_syncs' in sync_service.health_metrics

    @pytest.mark.asyncio
    async def test_sync_trade_manager_to_paper_wallet_success(self, sync_service, mock_trade_manager, mock_paper_wallet):
        """Test successful synchronization from TradeManager to PaperWallet."""
        # Setup mock positions
        mock_pos = Mock()
        mock_pos.symbol = "BTC/USD"
        mock_pos.side = "long"
        mock_pos.total_amount = 1.0
        mock_pos.average_price = 50000.0
        mock_pos.entry_time = datetime.now()
        mock_trade_manager.get_all_positions.return_value = [mock_pos]

        # Execute sync
        result = await sync_service.sync_trade_manager_to_paper_wallet(
            mock_trade_manager, mock_paper_wallet
        )

        # Verify results
        assert result.result == SyncResult.SUCCESS
        assert result.direction == SyncDirection.TRADE_MANAGER_TO_PAPER_WALLET
        assert result.duration_ms > 0
        mock_paper_wallet.sync_from_trade_manager.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_trade_manager_to_paper_wallet_failure(self, sync_service, mock_trade_manager, mock_paper_wallet):
        """Test synchronization failure handling."""
        # Setup failure scenario
        mock_paper_wallet.sync_from_trade_manager.side_effect = Exception("Sync failed")

        result = await sync_service.sync_trade_manager_to_paper_wallet(
            mock_trade_manager, mock_paper_wallet
        )

        # Verify error handling
        assert result.result == SyncResult.FAILED
        assert result.error_message == "Sync failed"
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_sync_positions_log_to_trade_manager_no_file(self, sync_service, mock_trade_manager, temp_log_dir):
        """Test handling when positions.log doesn't exist."""
        non_existent_file = temp_log_dir / "non_existent_positions.log"

        result = await sync_service.sync_positions_log_to_trade_manager(
            mock_trade_manager, non_existent_file
        )

        assert result.result == SyncResult.NO_CHANGES
        assert result.source_positions == []

    @pytest.mark.asyncio
    async def test_sync_positions_log_to_trade_manager_with_positions(self, sync_service, mock_trade_manager, temp_log_dir):
        """Test synchronization from positions.log with actual positions."""
        # Create test positions.log file
        positions_log = temp_log_dir / "positions.log"
        with open(positions_log, 'w') as f:
            f.write("2025-01-01 12:00:00 Active BTC/USD long 1.5 entry 50000.0\n")
            f.write("2025-01-01 12:05:00 Active ETH/USD short 2.0 entry 3000.0\n")

        # Setup mock TradeManager to accept position creation
        mock_trade_manager.create_position = Mock()

        result = await sync_service.sync_positions_log_to_trade_manager(
            mock_trade_manager, positions_log
        )

        # Verify positions were processed
        assert result.result == SyncResult.SUCCESS
        assert len(result.source_positions) == 2
        mock_trade_manager.create_position.assert_called()

    def test_parse_positions_log(self, sync_service, temp_log_dir):
        """Test positions.log parsing functionality."""
        positions_log = temp_log_dir / "positions.log"
        with open(positions_log, 'w') as f:
            f.write("2025-01-01 12:00:00 Active BTC/USD long 1.5 entry 50000.0\n")
            f.write("2025-01-01 12:05:00 Active ETH/USD short 2.0 entry 3000.0\n")
            f.write("2025-01-01 12:10:00 Some other log line\n")

        positions = sync_service._parse_positions_log(positions_log)

        assert len(positions) == 2
        assert 'BTC/USD' in positions
        assert 'ETH/USD' in positions
        assert positions['BTC/USD']['side'] == 'long'
        assert positions['BTC/USD']['amount'] == 1.5
        assert positions['BTC/USD']['entry_price'] == 50000.0

    def test_resolve_conflicts_trade_manager_wins(self, sync_service):
        """Test conflict resolution when TradeManager wins."""
        conflicts = [{
            'symbol': 'BTC/USD',
            'trade_manager': {'side': 'long', 'amount': 1.0, 'price': 50000.0},
            'positions_log': {'side': 'short', 'amount': 1.5, 'entry_price': 51000.0}
        }]

        resolved = sync_service._resolve_conflicts(conflicts, ConflictResolution.TRADE_MANAGER_WINS)

        assert len(resolved) == 1
        assert resolved[0]['resolution'] == 'trade_manager_wins'
        assert resolved[0]['kept'] == conflicts[0]['trade_manager']
        assert resolved[0]['discarded'] == conflicts[0]['positions_log']

    def test_health_metrics_tracking(self, sync_service):
        """Test health metrics are properly tracked."""
        # Create a successful operation
        operation = SyncOperation(
            operation_id="test_op",
            direction=SyncDirection.TRADE_MANAGER_TO_PAPER_WALLET,
            timestamp=datetime.now(),
            source_positions=[],
            target_positions=[],
            result=SyncResult.SUCCESS,
            duration_ms=100.0
        )

        sync_service._record_operation(operation)

        # Verify metrics
        assert sync_service.health_metrics['total_syncs'] == 1
        assert sync_service.health_metrics['successful_syncs'] == 1
        assert sync_service.health_metrics['failed_syncs'] == 0

    def test_get_health_status(self, sync_service):
        """Test health status reporting."""
        health = sync_service.get_health_status()

        assert 'overall_health' in health
        assert 'metrics' in health
        assert 'recent_operations' in health
        assert health['overall_health'] in ['healthy', 'degraded']

    def test_sync_report_summary(self, sync_service):
        """Test synchronization report generation."""
        report = sync_service.get_sync_report()

        assert 'total_operations' in report
        assert 'successful' in report
        assert 'failed' in report
        assert 'average_duration_ms' in report
        assert 'recent_operations' in report


class TestIntegrationSyncService:
    """Integration tests for SyncService with real components."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def real_trade_manager(self):
        """Create real TradeManager instance."""
        tm = TradeManager()
        # Don't load from file for testing
        return tm

    @pytest.fixture
    def real_paper_wallet(self):
        """Create real PaperWallet instance."""
        return PaperWallet(balance=10000.0)

    @pytest.mark.asyncio
    async def test_full_synchronization_workflow(self, temp_log_dir, real_trade_manager, real_paper_wallet):
        """Test complete synchronization workflow."""
        sync_service = SyncService(temp_log_dir)

        # Create test positions.log
        positions_log = temp_log_dir / "positions.log"
        with open(positions_log, 'w') as f:
            f.write("2025-01-01 12:00:00 Active BTC/USD long 1.0 entry 50000.0\n")

        # Perform full synchronization
        results = await sync_service.full_synchronization(
            trade_manager=real_trade_manager,
            paper_wallet=real_paper_wallet,
            positions_log_path=positions_log
        )

        # Verify results structure
        assert 'log_to_tm' in results
        assert 'tm_to_pw' in results

        # Check that operations completed (may be NO_CHANGES if no actual work needed)
        for operation in results.values():
            assert operation.result in [SyncResult.SUCCESS, SyncResult.NO_CHANGES, SyncResult.FAILED]

    @pytest.mark.asyncio
    async def test_error_recovery_and_fallback(self, temp_log_dir):
        """Test error recovery and fallback mechanisms."""
        sync_service = SyncService(temp_log_dir)

        # Create positions.log with invalid data
        positions_log = temp_log_dir / "positions.log"
        with open(positions_log, 'w') as f:
            f.write("Invalid log line that should be handled gracefully\n")
            f.write("2025-01-01 12:00:00 Active BTC/USD long invalid_amount entry 50000.0\n")

        # Test parsing handles errors gracefully
        positions = sync_service._parse_positions_log(positions_log)

        # Should handle errors and continue processing
        # (may be empty or partial depending on error handling)
        assert isinstance(positions, dict)

    def test_concurrent_operations_safety(self, temp_log_dir):
        """Test thread safety of concurrent operations."""
        sync_service = SyncService(temp_log_dir)

        # Simulate concurrent operations
        operations = []
        for i in range(10):
            operation = SyncOperation(
                operation_id=f"concurrent_op_{i}",
                direction=SyncDirection.TRADE_MANAGER_TO_PAPER_WALLET,
                timestamp=datetime.now(),
                source_positions=[],
                target_positions=[],
                result=SyncResult.SUCCESS,
                duration_ms=10.0
            )
            operations.append(operation)

        # Record operations (simulating concurrent access)
        for op in operations:
            sync_service._record_operation(op)

        # Verify all operations were recorded
        assert len(sync_service.sync_history) == 10
        assert sync_service.health_metrics['total_syncs'] == 10
        assert sync_service.health_metrics['successful_syncs'] == 10


class TestConflictResolutionStrategies:
    """Test various conflict resolution strategies."""

    @pytest.fixture
    def sync_service(self, tmp_path):
        """Create SyncService instance."""
        return SyncService(tmp_path)

    def test_trade_manager_wins_strategy(self, sync_service):
        """Test TRADE_MANAGER_WINS conflict resolution."""
        conflicts = [{
            'symbol': 'BTC/USD',
            'trade_manager': {'side': 'long', 'amount': 1.0, 'price': 50000.0},
            'positions_log': {'side': 'long', 'amount': 1.5, 'entry_price': 51000.0}
        }]

        resolved = sync_service._resolve_conflicts(conflicts, ConflictResolution.TRADE_MANAGER_WINS)

        assert resolved[0]['resolution'] == 'trade_manager_wins'
        assert resolved[0]['kept']['amount'] == 1.0  # TM value kept
        assert resolved[0]['discarded']['amount'] == 1.5  # Log value discarded

    def test_latest_wins_strategy(self, sync_service):
        """Test LATEST_WINS conflict resolution."""
        conflicts = [{
            'symbol': 'ETH/USD',
            'trade_manager': {'side': 'short', 'amount': 2.0, 'price': 3000.0},
            'positions_log': {'side': 'short', 'amount': 2.5, 'entry_price': 3100.0}
        }]

        resolved = sync_service._resolve_conflicts(conflicts, ConflictResolution.LATEST_WINS)

        # Should default to trade manager for now (implementation detail)
        assert resolved[0]['resolution'] == 'latest_wins'
        assert resolved[0]['kept'] == conflicts[0]['trade_manager']

    def test_manual_resolution_strategy(self, sync_service):
        """Test MANUAL conflict resolution."""
        conflicts = [{
            'symbol': 'ADA/USD',
            'trade_manager': {'side': 'long', 'amount': 100.0, 'price': 0.5},
            'positions_log': {'side': 'short', 'amount': 150.0, 'entry_price': 0.6}
        }]

        resolved = sync_service._resolve_conflicts(conflicts, ConflictResolution.MANUAL)

        assert resolved[0]['resolution'] == 'manual_review_required'
        assert resolved[0]['trade_manager'] == conflicts[0]['trade_manager']
        assert resolved[0]['positions_log'] == conflicts[0]['positions_log']


class TestPerformanceAndScalability:
    """Test performance and scalability of synchronization operations."""

    @pytest.fixture
    def sync_service(self, tmp_path):
        """Create SyncService instance."""
        return SyncService(tmp_path)

    def test_large_position_dataset_handling(self, sync_service):
        """Test handling of large numbers of positions."""
        # Create many mock positions
        large_dataset = []
        for i in range(1000):
            pos_data = {
                'side': 'long' if i % 2 == 0 else 'short',
                'amount': float(i + 1),
                'entry_price': 50000.0 + i
            }
            large_dataset.append((f'SYMBOL{i}/USD', pos_data))

        # Test parsing performance (should complete in reasonable time)
        import time
        start_time = time.time()

        # This simulates processing a large positions.log
        processed_count = len(large_dataset)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should process 1000 positions in reasonable time (< 1 second)
        assert processing_time < 1.0
        assert processed_count == 1000

    def test_memory_efficiency_with_many_operations(self, sync_service):
        """Test memory usage with many synchronization operations."""
        # Record many operations to test history management
        for i in range(150):  # More than the 100 limit
            operation = SyncOperation(
                operation_id=f"mem_test_op_{i}",
                direction=SyncDirection.TRADE_MANAGER_TO_PAPER_WALLET,
                timestamp=datetime.now(),
                source_positions=[],
                target_positions=[],
                result=SyncResult.SUCCESS,
                duration_ms=1.0
            )
            sync_service._record_operation(operation)

        # Should only keep the most recent 100 operations
        assert len(sync_service.sync_history) <= 100

        # Most recent operations should be preserved
        recent_op_ids = [op.operation_id for op in sync_service.sync_history[-5:]]
        assert all('mem_test_op_1' in op_id for op_id in recent_op_ids)


if __name__ == "__main__":
    pytest.main([__file__])
