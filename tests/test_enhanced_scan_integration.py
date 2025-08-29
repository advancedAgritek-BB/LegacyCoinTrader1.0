"""Tests for enhanced scan integration module."""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from crypto_bot.enhanced_scan_integration import (
    EnhancedScanIntegration, 
    detect_meme_waves, 
    scan_new_pools,
    calculate_momentum_scores
)


class TestEnhancedScanIntegration:
    """Test suite for Enhanced Scan Integration."""

    @pytest.fixture
    def mock_config(self):
        return {
            'enhanced_scanning': {
                'enabled': True,
                'scan_interval': 30,
                'momentum_threshold': 0.7,
                'max_pools_per_scan': 50
            },
            'solana': {
                'enabled': True,
                'rpc_url': 'https://api.mainnet-beta.solana.com'
            }
        }

    @pytest.fixture  
    def scan_integration(self, mock_config):
        return EnhancedScanIntegration(mock_config)

    def test_init(self, scan_integration, mock_config):
        """Test initialization."""
        assert scan_integration.config == mock_config
        assert scan_integration.enabled == True
        assert scan_integration.scan_interval == 30

    @patch('crypto_bot.enhanced_scan_integration.detect_meme_waves')
    def test_detect_meme_waves_enabled(self, mock_detect, scan_integration):
        """Test meme wave detection when enabled."""
        mock_detect.return_value = ['PUMP/SOL', 'DOGE/SOL']
        result = scan_integration.detect_waves()
        assert result == ['PUMP/SOL', 'DOGE/SOL']
        mock_detect.assert_called_once()

    def test_detect_meme_waves_disabled(self, mock_config):
        """Test meme wave detection when disabled."""
        mock_config['enhanced_scanning']['enabled'] = False
        scan_integration = EnhancedScanIntegration(mock_config)
        result = scan_integration.detect_waves()
        assert result == []

    @patch('crypto_bot.enhanced_scan_integration.scan_new_pools')
    def test_scan_new_pools(self, mock_scan, scan_integration):
        """Test new pool scanning."""
        mock_scan.return_value = [{'pool_id': 'test123', 'score': 0.8}]
        result = scan_integration.scan_pools()
        assert len(result) == 1
        assert result[0]['score'] == 0.8

    def test_calculate_momentum_scores(self):
        """Test momentum score calculation."""
        pool_data = {
            'volume_24h': 100000,
            'price_change_5m': 0.15,
            'liquidity': 50000,
            'holders': 150
        }
        score = calculate_momentum_scores(pool_data)
        assert isinstance(score, float)
        assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_async_scan_integration(self, scan_integration):
        """Test async scanning functionality."""
        with patch.object(scan_integration, 'detect_waves', return_value=['TEST/SOL']):
            result = await scan_integration.async_scan()
            assert result == ['TEST/SOL']


def test_detect_meme_waves():
    """Test standalone meme wave detection function."""
    mock_data = [
        {'symbol': 'PUMP/SOL', 'volume_change': 5.0, 'price_change': 0.3},
        {'symbol': 'SLOW/SOL', 'volume_change': 1.1, 'price_change': 0.05}
    ]
    
    with patch('crypto_bot.enhanced_scan_integration.fetch_pool_data', return_value=mock_data):
        waves = detect_meme_waves()
        assert 'PUMP/SOL' in waves
        assert 'SLOW/SOL' not in waves


def test_scan_new_pools():
    """Test new pool scanning function."""
    mock_pools = [
        {'pool_id': 'pool1', 'created_at': '2024-01-01', 'liquidity': 100000},
        {'pool_id': 'pool2', 'created_at': '2024-01-01', 'liquidity': 50000}
    ]
    
    with patch('crypto_bot.enhanced_scan_integration.fetch_new_pools', return_value=mock_pools):
        pools = scan_new_pools(max_pools=10)
        assert len(pools) <= 10
        assert all('score' in pool for pool in pools)


def test_calculate_momentum_scores_edge_cases():
    """Test momentum score calculation edge cases."""
    # Test with zero values
    empty_data = {'volume_24h': 0, 'price_change_5m': 0, 'liquidity': 0, 'holders': 0}
    score = calculate_momentum_scores(empty_data)
    assert score == 0

    # Test with negative values
    negative_data = {'volume_24h': -100, 'price_change_5m': -0.5, 'liquidity': 1000, 'holders': 10}
    score = calculate_momentum_scores(negative_data)
    assert score >= 0

    # Test with very high values
    high_data = {'volume_24h': 10000000, 'price_change_5m': 2.0, 'liquidity': 1000000, 'holders': 10000}
    score = calculate_momentum_scores(high_data)
    assert score <= 1


@pytest.mark.integration
def test_integration_with_real_config():
    """Integration test with real configuration."""
    config = {
        'enhanced_scanning': {'enabled': True, 'scan_interval': 60},
        'solana': {'enabled': True, 'rpc_url': 'test_url'}
    }
    
    integration = EnhancedScanIntegration(config)
    assert integration.enabled == True
    assert integration.scan_interval == 60
