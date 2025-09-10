"""Tests for Solana momentum detector module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from crypto_bot.solana.momentum_detector import (
    MomentumDetector,
    calculate_momentum_score,
    detect_volume_spike,
    analyze_price_action
)


class TestMomentumDetector:
    """Test suite for Momentum Detector."""

    @pytest.fixture
    def mock_config(self):
        return {
            'momentum': {
                'volume_threshold': 2.0,
                'price_change_threshold': 0.1,
                'time_window': 300,  # 5 minutes
                'min_score': 0.6
            }
        }

    @pytest.fixture
    def detector(self, mock_config):
        return MomentumDetector(mock_config)

    @pytest.fixture
    def sample_pool_data(self):
        return {
            'pool_id': 'test_pool_123',
            'token_a': 'SOL',
            'token_b': 'USDC',
            'current_price': 100.5,
            'volume_5m': 50000,
            'volume_1h': 200000,
            'liquidity': 1000000,
            'price_history': [98.0, 99.2, 100.1, 100.5],
            'volume_history': [10000, 15000, 30000, 50000],
            'timestamp': 1640995200
        }

    def test_detector_init(self, detector, mock_config):
        """Test detector initialization."""
        assert detector.config == mock_config['momentum']
        assert detector.volume_threshold == 2.0
        assert detector.price_change_threshold == 0.1

    def test_calculate_momentum_score_high(self, sample_pool_data):
        """Test momentum score calculation for high momentum."""
        # Simulate high momentum conditions
        sample_pool_data['volume_5m'] = 100000
        sample_pool_data['price_history'] = [95.0, 98.0, 102.0, 105.0]
        
        score = calculate_momentum_score(sample_pool_data)
        assert isinstance(score, float)
        assert score > 0.7  # Should be high momentum

    def test_calculate_momentum_score_low(self, sample_pool_data):
        """Test momentum score calculation for low momentum."""
        # Simulate low momentum conditions
        sample_pool_data['volume_5m'] = 5000
        sample_pool_data['price_history'] = [100.0, 100.1, 100.0, 100.1]
        
        score = calculate_momentum_score(sample_pool_data)
        assert isinstance(score, float)
        assert score < 0.3  # Should be low momentum

    def test_detect_volume_spike_positive(self):
        """Test volume spike detection - positive case."""
        current_volume = 100000
        historical_volumes = [10000, 15000, 20000, 25000]
        
        spike_detected = detect_volume_spike(current_volume, historical_volumes)
        assert spike_detected == True

    def test_detect_volume_spike_negative(self):
        """Test volume spike detection - negative case."""
        current_volume = 20000
        historical_volumes = [18000, 19000, 21000, 22000]
        
        spike_detected = detect_volume_spike(current_volume, historical_volumes)
        assert spike_detected == False

    def test_analyze_price_action_uptrend(self):
        """Test price action analysis for uptrend."""
        price_history = [95.0, 97.0, 99.0, 101.0, 103.0]
        
        analysis = analyze_price_action(price_history)
        assert analysis['trend'] == 'bullish'
        assert analysis['strength'] > 0.5

    def test_analyze_price_action_downtrend(self):
        """Test price action analysis for downtrend."""
        price_history = [105.0, 103.0, 101.0, 99.0, 97.0]
        
        analysis = analyze_price_action(price_history)
        assert analysis['trend'] == 'bearish'
        assert analysis['strength'] > 0.5

    def test_analyze_price_action_sideways(self):
        """Test price action analysis for sideways movement."""
        price_history = [100.0, 100.5, 99.8, 100.2, 100.1]
        
        analysis = analyze_price_action(price_history)
        assert analysis['trend'] == 'sideways'
        assert analysis['strength'] < 0.3

    @patch('crypto_bot.solana.momentum_detector.fetch_pool_data')
    def test_detector_analyze_pool(self, mock_fetch, detector, sample_pool_data):
        """Test pool analysis through detector."""
        mock_fetch.return_value = sample_pool_data
        
        result = detector.analyze_pool('test_pool_123')
        assert 'momentum_score' in result
        assert 'trend_analysis' in result
        assert 'volume_spike' in result

    def test_detector_filter_by_score(self, detector):
        """Test filtering pools by minimum score."""
        pools = [
            {'pool_id': 'pool1', 'momentum_score': 0.8},
            {'pool_id': 'pool2', 'momentum_score': 0.4},
            {'pool_id': 'pool3', 'momentum_score': 0.7}
        ]
        
        filtered = detector.filter_by_score(pools, min_score=0.6)
        assert len(filtered) == 2
        assert all(pool['momentum_score'] >= 0.6 for pool in filtered)

    @pytest.mark.asyncio
    async def test_detector_async_scan(self, detector):
        """Test async scanning of multiple pools."""
        pool_ids = ['pool1', 'pool2', 'pool3']
        
        with patch.object(detector, 'analyze_pool') as mock_analyze:
            mock_analyze.return_value = {'momentum_score': 0.8, 'trend_analysis': {}}
            
            results = await detector.async_scan(pool_ids)
            assert len(results) == 3
            assert mock_analyze.call_count == 3

    def test_momentum_score_edge_cases(self):
        """Test momentum score calculation edge cases."""
        # Empty data
        empty_data = {}
        score = calculate_momentum_score(empty_data)
        assert score == 0.0

        # Missing required fields
        incomplete_data = {'pool_id': 'test'}
        score = calculate_momentum_score(incomplete_data)
        assert score == 0.0

        # Invalid numeric values
        invalid_data = {
            'volume_5m': 'invalid',
            'price_history': [None, None, None]
        }
        score = calculate_momentum_score(invalid_data)
        assert score == 0.0

    def test_volume_spike_edge_cases(self):
        """Test volume spike detection edge cases."""
        # Empty historical data
        assert detect_volume_spike(1000, []) == False
        
        # Single historical point
        assert detect_volume_spike(1000, [500]) == True
        
        # Zero volumes
        assert detect_volume_spike(0, [0, 0, 0]) == False

    def test_price_action_edge_cases(self):
        """Test price action analysis edge cases."""
        # Empty price history
        analysis = analyze_price_action([])
        assert analysis['trend'] == 'unknown'
        assert analysis['strength'] == 0.0

        # Single price point
        analysis = analyze_price_action([100.0])
        assert analysis['trend'] == 'unknown'
        assert analysis['strength'] == 0.0

        # All same prices
        analysis = analyze_price_action([100.0, 100.0, 100.0])
        assert analysis['trend'] == 'sideways'
        assert analysis['strength'] == 0.0


@pytest.mark.integration
class TestMomentumDetectorIntegration:
    """Integration tests for momentum detector."""

    def test_full_momentum_detection_workflow(self):
        """Test complete momentum detection workflow."""
        config = {
            'momentum': {
                'volume_threshold': 2.0,
                'price_change_threshold': 0.1,
                'time_window': 300,
                'min_score': 0.6
            }
        }
        
        detector = MomentumDetector(config)
        
        # Mock pool data with high momentum
        high_momentum_pool = {
            'pool_id': 'high_momentum',
            'volume_5m': 100000,
            'volume_1h': 200000,
            'price_history': [95.0, 98.0, 102.0, 105.0],
            'volume_history': [10000, 20000, 50000, 100000]
        }
        
        with patch('crypto_bot.solana.momentum_detector.fetch_pool_data', 
                   return_value=high_momentum_pool):
            result = detector.analyze_pool('high_momentum')
            
            assert result['momentum_score'] > 0.6
            assert 'trend_analysis' in result
            assert 'volume_spike' in result


@pytest.mark.solana
def test_solana_specific_momentum_features():
    """Test Solana-specific momentum detection features."""
    # Test with Solana token pairs
    solana_pool_data = {
        'pool_id': 'solana_pool',
        'token_a': 'SOL',
        'token_b': 'USDC',
        'program_id': 'raydium_v4',
        'volume_5m': 75000,
        'price_history': [20.0, 21.5, 23.0, 24.5],
        'liquidity': 500000
    }
    
    score = calculate_momentum_score(solana_pool_data)
    assert isinstance(score, float)
    assert 0 <= score <= 1
