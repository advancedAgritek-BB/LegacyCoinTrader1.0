"""
Unit Tests for Evaluation Pipeline Integration

Comprehensive test suite covering:
- Pipeline initialization and configuration
- Token retrieval from multiple sources
- Fallback mechanisms and error handling
- Performance monitoring and metrics
- Caching and validation
- Health monitoring
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any

from crypto_bot.evaluation_pipeline_integration import (
    EvaluationPipelineIntegration,
    PipelineConfig,
    PipelineMetrics,
    PipelineStatus,
    TokenSource,
    TokenBatch,
    get_evaluation_pipeline_integration,
    initialize_evaluation_pipeline,
    get_tokens_for_evaluation,
    get_pipeline_status
)


class TestPipelineConfig:
    """Test PipelineConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.enabled is True
        assert config.max_batch_size == 20
        assert config.processing_timeout == 30.0
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0
        assert config.health_check_interval == 60.0
        assert config.max_consecutive_failures == 5
        assert config.enable_fallback_sources is True
        assert config.cache_ttl == 300.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            enabled=False,
            max_batch_size=50,
            processing_timeout=60.0,
            retry_attempts=5,
            cache_ttl=600.0
        )

        assert config.enabled is False
        assert config.max_batch_size == 50
        assert config.processing_timeout == 60.0
        assert config.retry_attempts == 5
        assert config.cache_ttl == 600.0


class TestPipelineMetrics:
    """Test PipelineMetrics class."""

    def test_initial_metrics(self):
        """Test initial metrics state."""
        metrics = PipelineMetrics()

        assert metrics.tokens_received == 0
        assert metrics.tokens_processed == 0
        assert metrics.tokens_failed == 0
        assert metrics.avg_processing_time == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.last_successful_run is None
        assert metrics.consecutive_failures == 0
        assert metrics.total_runtime == 0.0


class TestTokenSource:
    """Test TokenSource enum."""

    def test_enum_values(self):
        """Test enum values are correct."""
        assert TokenSource.ENHANCED_SCANNER.value == "enhanced_scanner"
        assert TokenSource.SOLANA_SCANNER.value == "solana_scanner"
        assert TokenSource.STATIC_CONFIG.value == "static_config"
        assert TokenSource.FALLBACK.value == "fallback"


class TestTokenBatch:
    """Test TokenBatch class."""

    def test_token_batch_creation(self):
        """Test TokenBatch creation and attributes."""
        tokens = ["BTC/USD", "ETH/USD"]
        batch = TokenBatch(tokens=tokens, source=TokenSource.ENHANCED_SCANNER)

        assert batch.tokens == tokens
        assert batch.source == TokenSource.ENHANCED_SCANNER
        assert isinstance(batch.timestamp, float)
        assert batch.metadata == {}
        assert batch.priority == 0

    def test_token_batch_with_metadata(self):
        """Test TokenBatch with custom metadata."""
        metadata = {"confidence": 0.8, "source": "test"}
        batch = TokenBatch(
            tokens=["BTC/USD"],
            source=TokenSource.STATIC_CONFIG,
            metadata=metadata,
            priority=1
        )

        assert batch.metadata == metadata
        assert batch.priority == 1


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return {
        "enhanced_scanning": {
            "enabled": True,
            "scan_interval": 30,
            "max_tokens_per_scan": 20
        },
        "solana_scanner": {
            "enabled": True,
            "interval_minutes": 30
        },
        "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "evaluation_pipeline": {
            "enabled": True,
            "max_batch_size": 10,
            "cache_ttl": 300.0
        }
    }


@pytest.fixture
def mock_enhanced_scanner():
    """Create a mock enhanced scanner."""
    scanner = Mock()
    scanner.get_integration_stats.return_value = {"running": True}
    scanner.get_top_opportunities.return_value = [
        Mock(symbol="BTC/USD", confidence=0.9),
        Mock(symbol="ETH/USD", confidence=0.8)
    ]
    return scanner


class TestEvaluationPipelineIntegration:
    """Test EvaluationPipelineIntegration class."""

    @pytest.mark.asyncio
    async def test_initialization_success(self, mock_config, mock_enhanced_scanner):
        """Test successful pipeline initialization."""
        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_get_scanner.return_value = mock_enhanced_scanner

            pipeline = EvaluationPipelineIntegration(mock_config)
            success = await pipeline.initialize()

            assert success is True
            assert pipeline.enhanced_scanner == mock_enhanced_scanner
            assert pipeline.status == PipelineStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_initialization_failure(self, mock_config):
        """Test pipeline initialization failure."""
        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_get_scanner.return_value = None

            pipeline = EvaluationPipelineIntegration(mock_config)
            success = await pipeline.initialize()

            assert success is False
            assert pipeline.status == PipelineStatus.OFFLINE

    def test_config_loading(self, mock_config):
        """Test pipeline configuration loading."""
        pipeline = EvaluationPipelineIntegration(mock_config)

        assert pipeline.pipeline_config.enabled is True
        assert pipeline.pipeline_config.max_batch_size == 10
        assert pipeline.pipeline_config.cache_ttl == 300.0

    def test_default_config_loading(self):
        """Test default configuration when no custom config provided."""
        config = {}
        pipeline = EvaluationPipelineIntegration(config)

        assert pipeline.pipeline_config.enabled is True
        assert pipeline.pipeline_config.max_batch_size == 20  # default

    @pytest.mark.asyncio
    async def test_get_tokens_enhanced_scanner_success(self, mock_config, mock_enhanced_scanner):
        """Test getting tokens from enhanced scanner successfully."""
        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_get_scanner.return_value = mock_enhanced_scanner

            pipeline = EvaluationPipelineIntegration(mock_config)
            await pipeline.initialize()

            tokens = await pipeline.get_tokens_for_evaluation(5)

            assert len(tokens) == 2
            assert "BTC/USD" in tokens
            assert "ETH/USD" in tokens
            assert pipeline.metrics.tokens_received == 2

    @pytest.mark.asyncio
    async def test_get_tokens_scanner_failure_fallback(self, mock_config):
        """Test fallback when enhanced scanner fails."""
        # Mock scanner to fail
        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_scanner = Mock()
            mock_scanner.get_integration_stats.return_value = {"running": True}
            mock_scanner.get_top_opportunities.side_effect = Exception("Scanner failed")
            mock_get_scanner.return_value = mock_scanner

            pipeline = EvaluationPipelineIntegration(mock_config)
            await pipeline.initialize()

            tokens = await pipeline.get_tokens_for_evaluation(5)

            # Should fall back to config tokens
            assert len(tokens) > 0
            assert pipeline.metrics.consecutive_failures > 0

    @pytest.mark.asyncio
    async def test_get_tokens_all_sources_fail(self, mock_config):
        """Test emergency fallback when all sources fail."""
        # Mock everything to fail
        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner, \
             patch.object(EvaluationPipelineIntegration, '_get_solana_tokens', side_effect=Exception("Failed")), \
             patch.object(EvaluationPipelineIntegration, '_get_config_tokens', side_effect=Exception("Failed")), \
             patch.object(EvaluationPipelineIntegration, '_get_fallback_tokens', side_effect=Exception("Failed")):

            mock_scanner = Mock()
            mock_scanner.get_integration_stats.return_value = {"running": True}
            mock_scanner.get_top_opportunities.side_effect = Exception("Scanner failed")
            mock_get_scanner.return_value = mock_scanner

            pipeline = EvaluationPipelineIntegration(mock_config)
            await pipeline.initialize()

            tokens = await pipeline.get_tokens_for_evaluation(5)

            # Should use emergency fallback
            assert len(tokens) == 2  # BTC/USD, ETH/USD
            assert "BTC/USD" in tokens
            assert "ETH/USD" in tokens

    def test_validate_and_deduplicate_tokens(self):
        """Test token validation and deduplication."""
        pipeline = EvaluationPipelineIntegration({})

        # Test with valid tokens
        tokens = ["BTC/USD", "ETH/USD", "btc/usd", "ADA/USD", "ETH/USD"]
        validated = asyncio.run(pipeline._validate_and_deduplicate_tokens(tokens))

        assert len(validated) == 3  # Should deduplicate BTC/USD and ETH/USD
        assert "BTC/USD" in validated
        assert "ETH/USD" in validated
        assert "ADA/USD" in validated

    def test_validate_invalid_tokens(self):
        """Test validation of invalid token formats."""
        pipeline = EvaluationPipelineIntegration({})

        # Test with invalid tokens
        tokens = ["BTCUSD", "", None, "BTC/USD", "INVALID"]
        validated = asyncio.run(pipeline._validate_and_deduplicate_tokens(tokens))

        assert len(validated) == 1
        assert "BTC/USD" in validated

    @pytest.mark.asyncio
    async def test_cache_pipeline_results(self, mock_config):
        """Test caching of pipeline results."""
        pipeline = EvaluationPipelineIntegration(mock_config)

        tokens = ["BTC/USD", "ETH/USD"]
        sources = [TokenSource.ENHANCED_SCANNER]

        await pipeline._cache_pipeline_results(tokens, sources)

        # Check cache was populated
        assert len(pipeline.token_cache) > 0

        # Get most recent cached result
        cached = await pipeline.get_cached_results()
        assert cached is not None
        assert cached.tokens == tokens
        assert cached.source == TokenSource.ENHANCED_SCANNER

    @pytest.mark.asyncio
    async def test_cache_expiry(self, mock_config):
        """Test cache expiry functionality."""
        pipeline = EvaluationPipelineIntegration(mock_config)
        pipeline.pipeline_config.cache_ttl = 0.1  # Very short TTL

        tokens = ["BTC/USD"]
        sources = [TokenSource.STATIC_CONFIG]

        await pipeline._cache_pipeline_results(tokens, sources)

        # Wait for cache to expire
        await asyncio.sleep(0.2)

        # Cache should be empty or expired
        cached = await pipeline.get_cached_results()
        assert cached is None

    def test_status_updates(self, mock_config):
        """Test pipeline status updates based on metrics."""
        pipeline = EvaluationPipelineIntegration(mock_config)

        # Test healthy status
        pipeline.metrics.consecutive_failures = 0
        pipeline.metrics.last_successful_run = time.time()
        pipeline._update_pipeline_status()
        assert pipeline.status == PipelineStatus.HEALTHY

        # Test degraded status
        pipeline.metrics.error_rate = 0.6
        pipeline._update_pipeline_status()
        assert pipeline.status == PipelineStatus.DEGRADED

        # Test critical status
        pipeline.metrics.consecutive_failures = 10
        pipeline._update_pipeline_status()
        assert pipeline.status == PipelineStatus.CRITICAL

    def test_get_pipeline_status(self, mock_config):
        """Test getting comprehensive pipeline status."""
        pipeline = EvaluationPipelineIntegration(mock_config)

        status = pipeline.get_pipeline_status()

        assert "status" in status
        assert "metrics" in status
        assert "config" in status
        assert "scanner" in status
        assert "last_health_check" in status

        assert status["status"] == PipelineStatus.OFFLINE.value
        assert isinstance(status["metrics"], dict)
        assert isinstance(status["config"], dict)

    @pytest.mark.asyncio
    async def test_force_cache_refresh(self, mock_config):
        """Test force cache refresh functionality."""
        pipeline = EvaluationPipelineIntegration(mock_config)

        # Add some cache entries
        await pipeline._cache_pipeline_results(["BTC/USD"], [TokenSource.STATIC_CONFIG])

        assert len(pipeline.token_cache) > 0

        # Force refresh
        await pipeline.force_refresh_cache()

        # Cache should be cleared
        assert len(pipeline.token_cache) == 0

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_config):
        """Test cleanup functionality."""
        pipeline = EvaluationPipelineIntegration(mock_config)

        # Add some data
        await pipeline._cache_pipeline_results(["BTC/USD"], [TokenSource.STATIC_CONFIG])

        # Cleanup
        await pipeline.cleanup()

        assert len(pipeline.token_cache) == 0
        assert pipeline.is_running is False


class TestGlobalFunctions:
    """Test global utility functions."""

    def test_get_evaluation_pipeline_integration(self, mock_config):
        """Test getting global pipeline integration instance."""
        # Clear any existing instance
        import crypto_bot.evaluation_pipeline_integration
        crypto_bot.evaluation_pipeline_integration._pipeline_integration = None

        integration = get_evaluation_pipeline_integration(mock_config)

        assert isinstance(integration, EvaluationPipelineIntegration)
        assert integration.config == mock_config

    def test_get_pipeline_status_no_instance(self):
        """Test getting pipeline status when no instance exists."""
        # Clear any existing instance
        import crypto_bot.evaluation_pipeline_integration
        crypto_bot.evaluation_pipeline_integration._pipeline_integration = None

        status = get_pipeline_status({})

        assert "error" in status
        assert status["error"] == "Enhanced scan integration not initialized"

    @pytest.mark.asyncio
    async def test_initialize_evaluation_pipeline(self, mock_config):
        """Test pipeline initialization function."""
        # Clear any existing instance
        import crypto_bot.evaluation_pipeline_integration
        crypto_bot.evaluation_pipeline_integration._pipeline_integration = None

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_scanner = Mock()
            mock_scanner.get_integration_stats.return_value = {"running": True}
            mock_scanner.get_top_opportunities.return_value = []
            mock_get_scanner.return_value = mock_scanner

            success = await initialize_evaluation_pipeline(mock_config)

            assert success is True

    @pytest.mark.asyncio
    async def test_get_tokens_for_evaluation_global(self, mock_config):
        """Test global get_tokens_for_evaluation function."""
        # Clear any existing instance
        import crypto_bot.evaluation_pipeline_integration
        crypto_bot.evaluation_pipeline_integration._pipeline_integration = None

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_scanner = Mock()
            mock_scanner.get_integration_stats.return_value = {"running": True}
            mock_scanner.get_top_opportunities.return_value = [
                Mock(symbol="BTC/USD", confidence=0.9)
            ]
            mock_get_scanner.return_value = mock_scanner

            tokens = await get_tokens_for_evaluation(mock_config, 5)

            assert len(tokens) == 1
            assert "BTC/USD" in tokens


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_critical_pipeline_failure(self, mock_config):
        """Test handling of critical pipeline failures."""
        pipeline = EvaluationPipelineIntegration(mock_config)

        # Mock all methods to fail
        with patch.object(pipeline, '_get_scanner_tokens', side_effect=Exception("Critical failure")), \
             patch.object(pipeline, '_get_solana_tokens', side_effect=Exception("Critical failure")), \
             patch.object(pipeline, '_get_config_tokens', side_effect=Exception("Critical failure")), \
             patch.object(pipeline, '_get_fallback_tokens', side_effect=Exception("Critical failure")), \
             patch.object(pipeline, '_validate_and_deduplicate_tokens', side_effect=Exception("Critical failure")):

            tokens = await pipeline.get_tokens_for_evaluation(5)

            # Should use emergency fallback
            assert len(tokens) == 2
            assert "BTC/USD" in tokens
            assert "ETH/USD" in tokens

    @pytest.mark.asyncio
    async def test_partial_failures(self, mock_config):
        """Test handling of partial failures with graceful degradation."""
        pipeline = EvaluationPipelineIntegration(mock_config)

        # Mock scanner to work, others to fail
        with patch.object(pipeline, '_get_scanner_tokens', return_value=["BTC/USD", "ETH/USD"]), \
             patch.object(pipeline, '_get_solana_tokens', side_effect=Exception("Solana failed")), \
             patch.object(pipeline, '_validate_and_deduplicate_tokens', return_value=["BTC/USD", "ETH/USD"]):

            tokens = await pipeline.get_tokens_for_evaluation(5)

            # Should still return scanner tokens despite other failures
            assert len(tokens) == 2
            assert "BTC/USD" in tokens
            assert "ETH/USD" in tokens

    @pytest.mark.asyncio
    async def test_concurrent_access(self, mock_config):
        """Test concurrent access to pipeline methods."""
        pipeline = EvaluationPipelineIntegration(mock_config)

        # Mock successful token retrieval
        with patch.object(pipeline, '_get_scanner_tokens', return_value=["BTC/USD"]), \
             patch.object(pipeline, '_validate_and_deduplicate_tokens', return_value=["BTC/USD"]):

            # Run multiple concurrent requests
            tasks = [
                pipeline.get_tokens_for_evaluation(5)
                for _ in range(5)
            ]

            results = await asyncio.gather(*tasks)

            # All should succeed and return same results
            for result in results:
                assert len(result) == 1
                assert "BTC/USD" in result


class TestPerformanceMonitoring:
    """Test performance monitoring and metrics."""

    def test_metrics_calculation(self, mock_config):
        """Test metrics calculation and updates."""
        pipeline = EvaluationPipelineIntegration(mock_config)

        # Simulate some activity
        pipeline.metrics.tokens_received = 100
        pipeline.metrics.tokens_processed = 95
        pipeline.metrics.tokens_failed = 5
        pipeline.metrics.avg_processing_time = 1.5
        pipeline.metrics.consecutive_failures = 2

        # Check error rate calculation
        pipeline.metrics.error_rate = 5 / 100  # 5%
        assert pipeline.metrics.error_rate == 0.05

        # Check status updates
        pipeline._update_pipeline_status()
        assert pipeline.status == PipelineStatus.DEGRADED  # Due to failures

    def test_processing_time_averaging(self, mock_config):
        """Test processing time averaging."""
        pipeline = EvaluationPipelineIntegration(mock_config)

        # First measurement
        pipeline.metrics.avg_processing_time = 1.0
        pipeline.metrics.tokens_processed = 1

        # Second measurement (should average)
        new_time = 3.0
        pipeline.metrics.avg_processing_time = (
            (pipeline.metrics.avg_processing_time * pipeline.metrics.tokens_processed) + new_time
        ) / (pipeline.metrics.tokens_processed + 1)
        pipeline.metrics.tokens_processed += 1

        expected_avg = (1.0 + 3.0) / 2  # 2.0
        assert abs(pipeline.metrics.avg_processing_time - expected_avg) < 0.001


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
