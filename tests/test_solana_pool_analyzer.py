"""Tests for Solana pool analyzer module."""
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pandas as pd
import numpy as np
from crypto_bot.solana.pool_analyzer import (
    PoolAnalyzer,
    analyze_liquidity,
    calculate_pool_metrics,
    detect_manipulation
)


class TestPoolAnalyzer:
    """Test suite for Pool Analyzer."""

    @pytest.fixture
    def analyzer(self):
        return PoolAnalyzer()

    @pytest.fixture
    def sample_pool_data(self):
        """Generate sample pool data for testing."""
        return {
            'pool_id': 'test_pool_123',
            'token_a': 'SOL',
            'token_b': 'USDC',
            'reserve_a': 1000.0,
            'reserve_b': 50000.0,
            'fee_rate': 0.003,
            'volume_24h': 100000,
            'price_history': [20.0, 21.0, 22.0, 21.5, 22.5],
            'liquidity_providers': 150,
            'created_at': '2024-01-01T00:00:00Z'
        }

    def test_analyzer_init(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert hasattr(analyzer, 'min_liquidity_threshold')
        assert hasattr(analyzer, 'max_price_impact')

    def test_analyze_liquidity_sufficient(self, analyzer, sample_pool_data):
        """Test liquidity analysis with sufficient liquidity."""
        result = analyze_liquidity(sample_pool_data)
        
        assert isinstance(result, dict)
        assert 'liquidity_score' in result
        assert 'risk_level' in result
        assert result['liquidity_score'] > 0.5  # Should be good liquidity

    def test_analyze_liquidity_insufficient(self, analyzer):
        """Test liquidity analysis with insufficient liquidity."""
        low_liquidity_pool = {
            'reserve_a': 10.0,
            'reserve_b': 500.0,
            'volume_24h': 1000
        }
        
        result = analyze_liquidity(low_liquidity_pool)
        
        assert result['liquidity_score'] < 0.3
        assert result['risk_level'] == 'high'

    def test_calculate_pool_metrics(self, analyzer, sample_pool_data):
        """Test pool metrics calculation."""
        metrics = calculate_pool_metrics(sample_pool_data)
        
        assert 'price_volatility' in metrics
        assert 'volume_stability' in metrics
        assert 'liquidity_depth' in metrics
        assert 'fee_efficiency' in metrics
        
        # All metrics should be between 0 and 1
        for value in metrics.values():
            assert 0 <= value <= 1

    def test_detect_manipulation_no_manipulation(self, analyzer, sample_pool_data):
        """Test manipulation detection with normal pool behavior."""
        result = detect_manipulation(sample_pool_data)
        
        assert isinstance(result, dict)
        assert 'manipulation_score' in result
        assert 'suspicious_patterns' in result
        assert result['manipulation_score'] < 0.3  # Low manipulation score

    def test_detect_manipulation_suspicious(self, analyzer):
        """Test manipulation detection with suspicious patterns."""
        suspicious_pool = {
            'price_history': [100.0, 100.0, 100.0, 200.0, 100.0],  # Sudden spike
            'volume_24h': 1000000,  # Unusually high volume
            'liquidity_providers': 5,  # Very few LPs
            'reserve_a': 1000.0,
            'reserve_b': 100000.0
        }
        
        result = detect_manipulation(suspicious_pool)
        
        assert result['manipulation_score'] > 0.6
        assert len(result['suspicious_patterns']) > 0

    def test_price_impact_calculation(self, analyzer, sample_pool_data):
        """Test price impact calculation."""
        # Test small trade (low impact)
        small_trade_impact = analyzer._calculate_price_impact(
            sample_pool_data, 
            trade_size=1000.0
        )
        assert small_trade_impact < 0.01  # Less than 1%
        
        # Test large trade (high impact)
        large_trade_impact = analyzer._calculate_price_impact(
            sample_pool_data, 
            trade_size=100000.0
        )
        assert large_trade_impact > small_trade_impact

    def test_volume_analysis(self, analyzer, sample_pool_data):
        """Test volume analysis functionality."""
        volume_metrics = analyzer._analyze_volume(sample_pool_data)
        
        assert 'volume_stability' in volume_metrics
        assert 'volume_trend' in volume_metrics
        assert 'abnormal_volume' in volume_metrics
        
        # All metrics should be between 0 and 1
        for value in volume_metrics.values():
            assert 0 <= value <= 1

    def test_liquidity_depth_analysis(self, analyzer, sample_pool_data):
        """Test liquidity depth analysis."""
        depth_metrics = analyzer._analyze_liquidity_depth(sample_pool_data)
        
        assert 'depth_score' in depth_metrics
        assert 'concentration_risk' in depth_metrics
        
        # Test with concentrated liquidity
        concentrated_pool = sample_pool_data.copy()
        concentrated_pool['liquidity_providers'] = 3
        
        concentrated_depth = analyzer._analyze_liquidity_depth(concentrated_pool)
        assert concentrated_depth['concentration_risk'] > 0.7

    def test_fee_analysis(self, analyzer, sample_pool_data):
        """Test fee analysis functionality."""
        fee_metrics = analyzer._analyze_fees(sample_pool_data)
        
        assert 'fee_efficiency' in fee_metrics
        assert 'fee_competitiveness' in fee_metrics
        
        # Test with different fee rates
        high_fee_pool = sample_pool_data.copy()
        high_fee_pool['fee_rate'] = 0.01  # 1% fee
        
        high_fee_metrics = analyzer._analyze_fees(high_fee_pool)
        assert high_fee_metrics['fee_efficiency'] < 0.5

    def test_pool_health_scoring(self, analyzer, sample_pool_data):
        """Test overall pool health scoring."""
        health_score = analyzer._calculate_pool_health(sample_pool_data)
        
        assert isinstance(health_score, float)
        assert 0 <= health_score <= 1
        
        # Healthy pool should have good score
        assert health_score > 0.6

    def test_risk_assessment(self, analyzer, sample_pool_data):
        """Test risk assessment functionality."""
        risk_assessment = analyzer._assess_risk(sample_pool_data)
        
        assert 'overall_risk' in risk_assessment
        assert 'risk_factors' in risk_assessment
        assert 'recommendations' in risk_assessment
        
        assert isinstance(risk_assessment['overall_risk'], str)
        assert risk_assessment['overall_risk'] in ['low', 'medium', 'high']

    @pytest.mark.asyncio
    async def test_async_pool_analysis(self, analyzer, sample_pool_data):
        """Test async pool analysis functionality."""
        result = await analyzer.analyze_pool_async(sample_pool_data)
        
        assert isinstance(result, dict)
        assert 'analysis_summary' in result
        assert 'metrics' in result
        assert 'risk_assessment' in result

    def test_pool_comparison(self, analyzer):
        """Test pool comparison functionality."""
        pool1 = {
            'pool_id': 'pool_1',
            'liquidity_score': 0.8,
            'risk_level': 'low',
            'volume_24h': 100000
        }
        
        pool2 = {
            'pool_id': 'pool_2',
            'liquidity_score': 0.6,
            'risk_level': 'medium',
            'volume_24h': 80000
        }
        
        comparison = analyzer.compare_pools([pool1, pool2])
        
        assert 'rankings' in comparison
        assert 'recommendations' in comparison
        assert len(comparison['rankings']) == 2

    def test_edge_cases(self, analyzer):
        """Test edge cases and error handling."""
        # Empty pool data
        empty_pool = {}
        
        with pytest.raises(ValueError):
            analyze_liquidity(empty_pool)
        
        # Invalid data types
        invalid_pool = {
            'reserve_a': 'invalid',
            'reserve_b': 'invalid',
            'volume_24h': 'invalid'
        }
        
        with pytest.raises(ValueError):
            calculate_pool_metrics(invalid_pool)
        
        # Zero values
        zero_pool = {
            'reserve_a': 0.0,
            'reserve_b': 0.0,
            'volume_24h': 0
        }
        
        result = analyze_liquidity(zero_pool)
        assert result['liquidity_score'] == 0.0

    def test_performance_optimization(self, analyzer):
        """Test performance optimization features."""
        # Test with large dataset
        large_pool_data = []
        for i in range(1000):
            pool = {
                'pool_id': f'pool_{i}',
                'reserve_a': np.random.uniform(100, 10000),
                'reserve_b': np.random.uniform(5000, 500000),
                'volume_24h': np.random.uniform(1000, 100000),
                'price_history': list(np.random.uniform(10, 100, 100))
            }
            large_pool_data.append(pool)
        
        start_time = pd.Timestamp.now()
        
        # Should process large datasets efficiently
        for pool in large_pool_data[:100]:  # Test subset
            analyze_liquidity(pool)
        
        end_time = pd.Timestamp.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert processing_time < 5.0


@pytest.mark.integration
class TestPoolAnalyzerIntegration:
    """Integration tests for pool analyzer."""

    def test_full_analysis_workflow(self, analyzer):
        """Test complete pool analysis workflow."""
        # Create realistic pool data
        pool_data = {
            'pool_id': 'real_pool_123',
            'token_a': 'SOL',
            'token_b': 'USDC',
            'reserve_a': 5000.0,
            'reserve_b': 250000.0,
            'fee_rate': 0.0025,
            'volume_24h': 500000,
            'price_history': [25.0, 25.5, 26.0, 25.8, 26.2, 26.5, 26.8, 27.0],
            'liquidity_providers': 200,
            'created_at': '2024-01-01T00:00:00Z'
        }
        
        # Run full analysis
        liquidity_analysis = analyze_liquidity(pool_data)
        pool_metrics = calculate_pool_metrics(pool_data)
        manipulation_detection = detect_manipulation(pool_data)
        
        # Verify all analyses completed successfully
        assert 'liquidity_score' in liquidity_analysis
        assert 'price_volatility' in pool_metrics
        assert 'manipulation_score' in manipulation_detection
        
        # Verify data consistency
        assert 0 <= liquidity_analysis['liquidity_score'] <= 1
        assert 0 <= pool_metrics['price_volatility'] <= 1
        assert 0 <= manipulation_detection['manipulation_score'] <= 1

    def test_multiple_pool_analysis(self, analyzer):
        """Test analyzing multiple pools simultaneously."""
        pools = [
            {
                'pool_id': 'pool_1',
                'reserve_a': 1000.0,
                'reserve_b': 50000.0,
                'volume_24h': 100000,
                'price_history': [20.0, 21.0, 22.0]
            },
            {
                'pool_id': 'pool_2',
                'reserve_a': 2000.0,
                'reserve_b': 100000.0,
                'volume_24h': 200000,
                'price_history': [30.0, 31.0, 32.0]
            },
            {
                'pool_id': 'pool_3',
                'reserve_a': 500.0,
                'reserve_b': 25000.0,
                'volume_24h': 50000,
                'price_history': [15.0, 15.5, 16.0]
            }
        ]
        
        # Analyze all pools
        results = []
        for pool in pools:
            result = {
                'pool_id': pool['pool_id'],
                'liquidity': analyze_liquidity(pool),
                'metrics': calculate_pool_metrics(pool),
                'manipulation': detect_manipulation(pool)
            }
            results.append(result)
        
        # Verify all analyses completed
        assert len(results) == 3
        
        for result in results:
            assert 'liquidity' in result
            assert 'metrics' in result
            assert 'manipulation' in result


@pytest.mark.solana
def test_solana_specific_pool_features():
    """Test Solana-specific pool analysis features."""
    analyzer = PoolAnalyzer()
    
    # Test Solana-specific pool characteristics
    solana_pool = {
        'pool_id': 'solana_pool',
        'program_id': 'raydium_v4',
        'amm_id': 'test_amm_123',
        'reserve_a': 1000.0,
        'reserve_b': 50000.0,
        'volume_24h': 100000
    }
    
    # Should handle Solana-specific fields
    result = analyze_liquidity(solana_pool)
    assert 'liquidity_score' in result
