"""
Comprehensive tests for Unified Position Manager

This module tests all functionality of the Unified Position Manager to ensure
data consistency and proper conflict resolution.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

from crypto_bot.utils.unified_position_manager import (
    UnifiedPositionManager, 
    PositionConflict, 
    PositionSyncStats
)

class TestUnifiedPositionManager:
    """Test suite for Unified Position Manager."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for test logs."""
        temp_dir = tempfile.mkdtemp()
        log_dir = Path(temp_dir) / "logs"
        log_dir.mkdir()
        yield log_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_trade_manager(self):
        """Create mock TradeManager."""
        tm = Mock()
        tm.get_all_positions.return_value = []
        tm.update_position = Mock()
        tm.close_position = Mock()
        return tm
    
    @pytest.fixture
    def mock_paper_wallet(self):
        """Create mock paper wallet."""
        pw = Mock()
        pw.positions = {}
        return pw
    
    @pytest.fixture
    def sample_positions(self):
        """Sample position data for testing."""
        return {
            'BTC/USD': {
                'symbol': 'BTC/USD',
                'side': 'buy',
                'size': 0.1,
                'entry_price': 50000.0,
                'current_price': 51000.0,
                'pnl': 100.0,
                'timestamp': datetime.now().isoformat()
            },
            'ETH/USD': {
                'symbol': 'ETH/USD',
                'side': 'sell',
                'size': 1.0,
                'entry_price': 3000.0,
                'current_price': 2900.0,
                'pnl': 100.0,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    @pytest.fixture
    def upm(self, mock_trade_manager, mock_paper_wallet, temp_log_dir):
        """Create Unified Position Manager instance for testing."""
        config = {
            'position_sync_interval': 1,  # Fast sync for testing
            'max_conflict_history': 10
        }
        
        with patch('crypto_bot.utils.unified_position_manager.Path') as mock_path:
            mock_path.return_value = temp_log_dir / "positions.log"
            
            upm = UnifiedPositionManager(
                trade_manager=mock_trade_manager,
                paper_wallet=mock_paper_wallet,
                config=config
            )
            
            # Clear any existing data
            upm.position_cache.clear()
            upm.paper_wallet.positions.clear()
            upm.trade_manager.get_all_positions.return_value = []
            
            # Clear the log file
            log_file = temp_log_dir / "positions.log"
            if log_file.exists():
                log_file.unlink()
            
            return upm
    
    def test_initialization(self, upm):
        """Test Unified Position Manager initialization."""
        assert upm.trade_manager is not None
        assert upm.paper_wallet is not None
        assert upm.config is not None
        assert upm.sync_interval == 1
        assert upm.running == False
        assert upm.sync_task is None
        assert isinstance(upm.stats, PositionSyncStats)
        assert len(upm.conflict_history) == 0
    
    def test_get_trade_manager_positions(self, upm, sample_positions):
        """Test getting positions from TradeManager."""
        # Mock TradeManager to return sample positions
        mock_positions = []
        for symbol, pos_data in sample_positions.items():
            mock_pos = Mock()
            mock_pos.symbol = symbol
            mock_pos.to_dict.return_value = pos_data
            mock_positions.append(mock_pos)
        
        upm.trade_manager.get_all_positions.return_value = mock_positions
        
        positions = upm._get_trade_manager_positions()
        
        assert len(positions) == 2
        assert 'BTC/USD' in positions
        assert 'ETH/USD' in positions
        assert positions['BTC/USD']['side'] == 'buy'
        assert positions['ETH/USD']['side'] == 'sell'
    
    def test_get_paper_wallet_positions(self, upm, sample_positions):
        """Test getting positions from paper wallet."""
        upm.paper_wallet.positions = sample_positions.copy()
        
        positions = upm._get_paper_wallet_positions()
        
        assert len(positions) == 2
        assert positions == sample_positions
    
    def test_load_positions_from_log(self, upm, temp_log_dir, sample_positions):
        """Test loading positions from log file."""
        # Create test log file
        log_file = temp_log_dir / "positions.log"
        with open(log_file, 'w') as f:
            f.write("# timestamp,symbol,side,size,entry_price,current_price,pnl\n")
            for symbol, pos in sample_positions.items():
                f.write(f"{pos['timestamp']},{symbol},{pos['side']},{pos['size']},"
                       f"{pos['entry_price']},{pos['current_price']},{pos['pnl']}\n")
        
        with patch('crypto_bot.utils.unified_position_manager.Path') as mock_path:
            mock_path.return_value = log_file
            
            positions = upm._load_positions_from_log()
            
            assert len(positions) == 2
            assert 'BTC/USD' in positions
            assert 'ETH/USD' in positions
            assert positions['BTC/USD']['side'] == 'buy'
            assert positions['ETH/USD']['side'] == 'sell'
    
    def test_has_conflict_no_conflict(self, upm):
        """Test conflict detection when there's no conflict."""
        pos1 = {'symbol': 'BTC/USD', 'side': 'buy', 'size': 0.1, 'entry_price': 50000.0}
        pos2 = {'symbol': 'BTC/USD', 'side': 'buy', 'size': 0.1, 'entry_price': 50000.0}
        
        has_conflict = upm._has_conflict(pos1, pos2, None)
        assert has_conflict == False
    
    def test_has_conflict_size_mismatch(self, upm):
        """Test conflict detection when sizes don't match."""
        pos1 = {'symbol': 'BTC/USD', 'side': 'buy', 'size': 0.1, 'entry_price': 50000.0}
        pos2 = {'symbol': 'BTC/USD', 'side': 'buy', 'size': 0.2, 'entry_price': 50000.0}
        
        has_conflict = upm._has_conflict(pos1, pos2, None)
        assert has_conflict == True
    
    def test_has_conflict_side_mismatch(self, upm):
        """Test conflict detection when sides don't match."""
        pos1 = {'symbol': 'BTC/USD', 'side': 'buy', 'size': 0.1, 'entry_price': 50000.0}
        pos2 = {'symbol': 'BTC/USD', 'side': 'sell', 'size': 0.1, 'entry_price': 50000.0}
        
        has_conflict = upm._has_conflict(pos1, pos2, None)
        assert has_conflict == True
    
    def test_determine_conflict_type(self, upm):
        """Test conflict type determination."""
        pos1 = {'symbol': 'BTC/USD', 'side': 'buy', 'size': 0.1}
        pos2 = {'symbol': 'BTC/USD', 'side': 'sell', 'size': 0.1}
        pos3 = {'symbol': 'BTC/USD', 'side': 'buy', 'size': 0.2}
        
        # Three-way conflict
        conflict_type = upm._determine_conflict_type(pos1, pos2, pos3)
        assert conflict_type == "three_way_conflict"
        
        # Two-way conflict
        conflict_type = upm._determine_conflict_type(pos1, pos2, None)
        assert conflict_type == "tm_pw_conflict"
        
        # Single position
        conflict_type = upm._determine_conflict_type(pos1, None, None)
        assert conflict_type == "unknown_conflict"
    
    def test_determine_resolution_strategy(self, upm):
        """Test resolution strategy determination."""
        pos1 = {'symbol': 'BTC/USD', 'side': 'buy', 'size': 0.1}
        pos2 = {'symbol': 'BTC/USD', 'side': 'sell', 'size': 0.1}
        
        # TradeManager priority
        strategy = upm._determine_resolution_strategy(pos1, pos2, None)
        assert strategy == "trade_manager_priority"
        
        # Paper wallet priority
        strategy = upm._determine_resolution_strategy(None, pos2, None)
        assert strategy == "paper_wallet_priority"
        
        # Most recent
        strategy = upm._determine_resolution_strategy(None, None, pos1)
        assert strategy == "most_recent"
        
        # Emergency reset
        strategy = upm._determine_resolution_strategy(None, None, None)
        assert strategy == "emergency_reset"
    
    @pytest.mark.asyncio
    async def test_resolve_trade_manager_priority(self, upm, temp_log_dir):
        """Test resolving conflict by prioritizing TradeManager."""
        tm_pos = {'symbol': 'BTC/USD', 'side': 'buy', 'size': 0.1, 'entry_price': 50000.0}
        pw_pos = {'symbol': 'BTC/USD', 'side': 'sell', 'size': 0.1, 'entry_price': 50000.0}
        
        conflict = PositionConflict(
            symbol='BTC/USD',
            trade_manager_position=tm_pos,
            paper_wallet_position=pw_pos,
            resolution_strategy='trade_manager_priority'
        )
        
        # Create test log file
        log_file = temp_log_dir / "positions.log"
        log_file.touch()
        
        with patch('crypto_bot.utils.unified_position_manager.Path') as mock_path:
            mock_path.return_value = log_file
            
            await upm._resolve_trade_manager_priority(conflict)
            
            # Check that paper wallet was updated
            assert upm.paper_wallet.positions['BTC/USD'] == tm_pos
    
    @pytest.mark.asyncio
    async def test_resolve_paper_wallet_priority(self, upm, temp_log_dir):
        """Test resolving conflict by prioritizing paper wallet."""
        tm_pos = {'symbol': 'BTC/USD', 'side': 'buy', 'size': 0.1, 'entry_price': 50000.0}
        pw_pos = {'symbol': 'BTC/USD', 'side': 'sell', 'size': 0.1, 'entry_price': 50000.0}
        
        conflict = PositionConflict(
            symbol='BTC/USD',
            trade_manager_position=tm_pos,
            paper_wallet_position=pw_pos,
            resolution_strategy='paper_wallet_priority'
        )
        
        # Create test log file
        log_file = temp_log_dir / "positions.log"
        log_file.touch()
        
        with patch('crypto_bot.utils.unified_position_manager.Path') as mock_path:
            mock_path.return_value = log_file
            
            await upm._resolve_paper_wallet_priority(conflict)
            
            # Check that TradeManager was updated
            upm.trade_manager.update_position.assert_called_once_with('BTC/USD', pw_pos)
    
    @pytest.mark.asyncio
    async def test_resolve_emergency_reset(self, upm, temp_log_dir):
        """Test emergency reset resolution."""
        conflict = PositionConflict(
            symbol='BTC/USD',
            trade_manager_position={'symbol': 'BTC/USD', 'side': 'buy', 'size': 0.1},
            paper_wallet_position={'symbol': 'BTC/USD', 'side': 'sell', 'size': 0.1},
            resolution_strategy='emergency_reset'
        )
        
        # Add position to paper wallet
        upm.paper_wallet.positions['BTC/USD'] = {'symbol': 'BTC/USD', 'side': 'buy', 'size': 0.1}
        
        # Create test log file
        log_file = temp_log_dir / "positions.log"
        with open(log_file, 'w') as f:
            f.write("2024-01-01T00:00:00,BTC/USD,buy,0.1,50000,51000,100\n")
        
        with patch('crypto_bot.utils.unified_position_manager.Path') as mock_path:
            mock_path.return_value = log_file
            
            await upm._resolve_emergency_reset(conflict)
            
            # Check that position was removed from paper wallet
            assert 'BTC/USD' not in upm.paper_wallet.positions
            
            # Check that TradeManager close_position was called
            upm.trade_manager.close_position.assert_called_once_with('BTC/USD')
    
    def test_get_unified_positions(self, upm, sample_positions):
        """Test getting unified positions."""
        # Mock TradeManager to return sample positions
        mock_positions = []
        for symbol, pos_data in sample_positions.items():
            mock_pos = Mock()
            mock_pos.symbol = symbol
            mock_pos.to_dict.return_value = pos_data
            mock_positions.append(mock_pos)
        
        upm.trade_manager.get_all_positions.return_value = mock_positions
        
        positions = upm.get_unified_positions()
        
        assert len(positions) == 2
        assert 'BTC/USD' in positions
        assert 'ETH/USD' in positions
    
    def test_get_position(self, upm, sample_positions):
        """Test getting a specific position."""
        # Mock TradeManager to return sample positions
        mock_positions = []
        for symbol, pos_data in sample_positions.items():
            mock_pos = Mock()
            mock_pos.symbol = symbol
            mock_pos.to_dict.return_value = pos_data
            mock_positions.append(mock_pos)
        
        upm.trade_manager.get_all_positions.return_value = mock_positions
        
        position = upm.get_position('BTC/USD')
        
        assert position is not None
        assert position['symbol'] == 'BTC/USD'
        assert position['side'] == 'buy'
    
    def test_update_position(self, upm, temp_log_dir):
        """Test updating a position through unified manager."""
        position_data = {
            'symbol': 'BTC/USD',
            'side': 'buy',
            'size': 0.1,
            'entry_price': 50000.0,
            'current_price': 51000.0,
            'pnl': 100.0
        }
        
        # Create test log file
        log_file = temp_log_dir / "positions.log"
        log_file.touch()
        
        with patch('crypto_bot.utils.unified_position_manager.Path') as mock_path:
            mock_path.return_value = log_file
            
            upm.update_position('BTC/USD', position_data)
            
            # Check that TradeManager was updated
            upm.trade_manager.update_position.assert_called_once_with('BTC/USD', position_data)
            
            # Check that paper wallet was updated
            assert upm.paper_wallet.positions['BTC/USD'] == position_data
    
    @pytest.mark.asyncio
    async def test_sync_all_systems_no_conflicts(self, upm, sample_positions):
        """Test synchronization when there are no conflicts."""
        # Mock all systems to return consistent data
        mock_positions = []
        for symbol, pos_data in sample_positions.items():
            mock_pos = Mock()
            mock_pos.symbol = symbol
            mock_pos.to_dict.return_value = pos_data
            mock_positions.append(mock_pos)
        
        upm.trade_manager.get_all_positions.return_value = mock_positions
        upm.paper_wallet.positions = sample_positions.copy()
        
        # Mock log loading to return same data
        with patch.object(upm, '_load_positions_from_log', return_value=sample_positions):
            conflicts = await upm.sync_all_systems()
            
            assert len(conflicts) == 0
            assert upm.stats.total_syncs == 1
    
    @pytest.mark.asyncio
    async def test_sync_all_systems_with_conflicts(self, upm, temp_log_dir):
        """Test synchronization when there are conflicts."""
        # Create conflicting data
        tm_pos = {'symbol': 'BTC/USD', 'side': 'buy', 'size': 0.1, 'entry_price': 50000.0}
        pw_pos = {'symbol': 'BTC/USD', 'side': 'sell', 'size': 0.1, 'entry_price': 50000.0}
        
        mock_positions = []
        mock_pos = Mock()
        mock_pos.symbol = 'BTC/USD'
        mock_pos.to_dict.return_value = tm_pos
        mock_positions.append(mock_pos)
        
        upm.trade_manager.get_all_positions.return_value = mock_positions
        upm.paper_wallet.positions = {'BTC/USD': pw_pos}
        
        # Create test log file
        log_file = temp_log_dir / "positions.log"
        log_file.touch()
        
        with patch('crypto_bot.utils.unified_position_manager.Path') as mock_path:
            mock_path.return_value = log_file
            
            conflicts = await upm.sync_all_systems()
            
            assert len(conflicts) == 1
            assert conflicts[0].symbol == 'BTC/USD'
            assert conflicts[0].conflict_type == "tm_pw_conflict"
            assert conflicts[0].resolution_strategy == "trade_manager_priority"
    
    def test_validate_consistency(self, upm, sample_positions):
        """Test consistency validation."""
        # Mock TradeManager to return sample positions
        mock_positions = []
        for symbol, pos_data in sample_positions.items():
            mock_pos = Mock()
            mock_pos.symbol = symbol
            mock_pos.to_dict.return_value = pos_data
            mock_positions.append(mock_pos)
        
        upm.trade_manager.get_all_positions.return_value = mock_positions
        upm.paper_wallet.positions = sample_positions.copy()
        
        # Mock log loading to return same data
        with patch.object(upm, '_load_positions_from_log', return_value=sample_positions):
            is_consistent = upm.validate_consistency()
            assert is_consistent == True
    
    def test_get_sync_stats(self, upm):
        """Test getting synchronization statistics."""
        stats = upm.get_sync_stats()
        assert isinstance(stats, PositionSyncStats)
        assert stats.total_syncs == 0
        assert stats.conflicts_detected == 0
        assert stats.conflicts_resolved == 0
        assert stats.error_count == 0
    
    def test_get_conflict_history(self, upm):
        """Test getting conflict history."""
        # Add some conflicts to history
        conflict1 = PositionConflict(symbol='BTC/USD', conflict_type='test1')
        conflict2 = PositionConflict(symbol='ETH/USD', conflict_type='test2')
        
        upm.conflict_history = [conflict1, conflict2]
        
        history = upm.get_conflict_history()
        assert len(history) == 2
        assert history[0].symbol == 'BTC/USD'
        assert history[1].symbol == 'ETH/USD'
    
    @pytest.mark.asyncio
    async def test_start_stop_sync_monitoring(self, upm):
        """Test starting and stopping sync monitoring."""
        # Start monitoring
        await upm.start_sync_monitoring()
        assert upm.running == True
        assert upm.sync_task is not None
        
        # Stop monitoring
        await upm.stop_sync_monitoring()
        assert upm.running == False
        assert upm.sync_task is None
    
    def test_error_handling_in_get_positions(self, upm):
        """Test error handling when getting positions fails."""
        # Mock TradeManager to raise exception
        upm.trade_manager.get_all_positions.side_effect = Exception("Test error")
        
        positions = upm._get_trade_manager_positions()
        assert positions == {}
    
    def test_error_handling_in_paper_wallet(self, upm):
        """Test error handling when paper wallet access fails."""
        # Mock paper wallet to raise exception
        upm.paper_wallet.positions = None  # This will cause an error
        
        positions = upm._get_paper_wallet_positions()
        assert positions == {}
    
    def test_cache_functionality(self, upm, sample_positions):
        """Test position cache functionality."""
        # Mock TradeManager to return sample positions
        mock_positions = []
        for symbol, pos_data in sample_positions.items():
            mock_pos = Mock()
            mock_pos.symbol = symbol
            mock_pos.to_dict.return_value = pos_data
            mock_positions.append(mock_pos)
        
        upm.trade_manager.get_all_positions.return_value = mock_positions
        
        # First call should populate cache
        positions1 = upm.get_unified_positions()
        assert len(positions1) == 2
        
        # Second call should use cache
        positions2 = upm.get_unified_positions()
        assert positions1 == positions2
        
        # Verify cache timestamp is recent
        assert (datetime.now() - upm.cache_timestamp).total_seconds() < 30

class TestPositionConflict:
    """Test suite for PositionConflict dataclass."""
    
    def test_position_conflict_creation(self):
        """Test creating a PositionConflict instance."""
        conflict = PositionConflict(
            symbol='BTC/USD',
            trade_manager_position={'side': 'buy'},
            paper_wallet_position={'side': 'sell'},
            conflict_type='tm_pw_conflict',
            resolution_strategy='trade_manager_priority'
        )
        
        assert conflict.symbol == 'BTC/USD'
        assert conflict.conflict_type == 'tm_pw_conflict'
        assert conflict.resolution_strategy == 'trade_manager_priority'
        assert conflict.resolved == False
        assert isinstance(conflict.detected_at, datetime)

class TestPositionSyncStats:
    """Test suite for PositionSyncStats dataclass."""
    
    def test_position_sync_stats_creation(self):
        """Test creating a PositionSyncStats instance."""
        stats = PositionSyncStats(
            total_syncs=10,
            conflicts_detected=5,
            conflicts_resolved=4,
            sync_duration_avg=1.5,
            error_count=1
        )
        
        assert stats.total_syncs == 10
        assert stats.conflicts_detected == 5
        assert stats.conflicts_resolved == 4
        assert stats.sync_duration_avg == 1.5
        assert stats.error_count == 1
        assert stats.last_sync_time is None

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
