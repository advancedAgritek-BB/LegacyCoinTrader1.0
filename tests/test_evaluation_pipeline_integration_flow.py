"""
Integration Tests for Evaluation Pipeline Token Flow

End-to-end integration tests covering:
- Complete token flow from scanning to evaluation
- Real component interactions
- Production-like scenarios
- Performance and reliability testing
- Failure recovery and resilience
"""

import asyncio
import pytest
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional

from crypto_bot.evaluation_pipeline_integration import (
    EvaluationPipelineIntegration,
    PipelineStatus,
    TokenSource,
    initialize_evaluation_pipeline,
    get_tokens_for_evaluation,
    get_pipeline_status
)
from crypto_bot.main import fetch_candidates, BotContext
from crypto_bot.phase_runner import PhaseRunner


class MockEnhancedScanner:
    """Mock enhanced scanner for integration testing."""

    def __init__(self, tokens: List[str] = None, should_fail: bool = False):
        self.tokens = tokens or ["BTC/USD", "ETH/USD", "SOL/USD"]
        self.should_fail = should_fail
        self.call_count = 0

    def get_integration_stats(self) -> Dict[str, Any]:
        return {"running": not self.should_fail}

    def get_top_opportunities(self, limit: int = 20) -> List[Mock]:
        self.call_count += 1
        if self.should_fail:
            raise Exception("Mock scanner failure")

        return [
            Mock(symbol=token, confidence=0.8 + i * 0.05)
            for i, token in enumerate(self.tokens[:limit])
        ]


class MockSolanaScanner:
    """Mock Solana scanner for integration testing."""

    def __init__(self, tokens: List[str] = None, should_fail: bool = False):
        self.tokens = tokens or ["RAY/USD", "ORCA/USD"]
        self.should_fail = should_fail

    async def get_solana_new_tokens(self, config: Dict[str, Any]) -> List[str]:
        if self.should_fail:
            raise Exception("Mock Solana scanner failure")
        return self.tokens


@pytest.fixture
def integration_config():
    """Create integration test configuration."""
    return {
        "enhanced_scanning": {
            "enabled": True,
            "scan_interval": 30,
            "max_tokens_per_scan": 20,
            "min_score_threshold": 0.4
        },
        "solana_scanner": {
            "enabled": True,
            "interval_minutes": 30,
            "max_tokens_per_scan": 10
        },
        "symbols": ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD"],
        "evaluation_pipeline": {
            "enabled": True,
            "max_batch_size": 15,
            "processing_timeout": 10.0,
            "retry_attempts": 2,
            "cache_ttl": 60.0,  # Short for testing
            "enable_fallback_sources": True
        },
        "symbol_batch_size": 10,
        "timeframe": "1h"
    }


@pytest.fixture
def mock_exchange():
    """Create mock exchange for testing."""
    exchange = Mock()
    exchange.id = "kraken"
    exchange.load_markets = AsyncMock(return_value={})
    return exchange


@pytest.fixture
def mock_paper_wallet():
    """Create mock paper wallet for testing."""
    wallet = Mock()
    wallet.balance = 10000.0
    wallet.positions = {}
    return wallet


class TestEndToEndTokenFlow:
    """Test complete token flow from scanning to evaluation."""

    @pytest.mark.asyncio
    async def test_successful_token_flow_all_sources(self, integration_config):
        """Test successful token flow using all sources."""
        # Setup mocks
        mock_scanner = MockEnhancedScanner(["BTC/USD", "ETH/USD", "SOL/USD"])
        mock_solana = MockSolanaScanner(["RAY/USD", "ORCA/USD"])

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner, \
             patch('crypto_bot.evaluation_pipeline_integration.EvaluationPipelineIntegration._get_solana_tokens') as mock_get_solana, \
             patch('crypto_bot.evaluation_pipeline_integration.EvaluationPipelineIntegration._get_config_tokens') as mock_get_config:

            mock_get_scanner.return_value = mock_scanner
            mock_get_solana.return_value = mock_solana.tokens
            mock_get_config.return_value = ["ADA/USD", "DOT/USD"]

            # Initialize pipeline
            pipeline = EvaluationPipelineIntegration(integration_config)
            success = await pipeline.initialize()
            assert success is True

            # Get tokens
            tokens = await pipeline.get_tokens_for_evaluation(10)

            # Verify results
            assert len(tokens) >= 5  # Should get tokens from multiple sources
            assert pipeline.status == PipelineStatus.HEALTHY
            assert pipeline.metrics.tokens_received > 0
            assert pipeline.metrics.consecutive_failures == 0

            # Check that tokens came from expected sources
            expected_tokens = {"BTC/USD", "ETH/USD", "SOL/USD", "RAY/USD", "ORCA/USD", "ADA/USD", "DOT/USD"}
            actual_tokens = set(tokens)
            assert len(actual_tokens.intersection(expected_tokens)) > 0

    @pytest.mark.asyncio
    async def test_resilient_fallback_chain(self, integration_config):
        """Test resilient fallback when sources fail progressively."""
        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            # Scanner fails
            mock_scanner = MockEnhancedScanner(should_fail=True)
            mock_get_scanner.return_value = mock_scanner

            pipeline = EvaluationPipelineIntegration(integration_config)
            await pipeline.initialize()

            # Mock other sources to also fail initially
            with patch.object(pipeline, '_get_solana_tokens', side_effect=Exception("Solana failed")), \
                 patch.object(pipeline, '_get_config_tokens', side_effect=Exception("Config failed")):

                # Should still get fallback tokens
                tokens = await pipeline.get_tokens_for_evaluation(5)

                assert len(tokens) == 2  # Emergency fallback: BTC/USD, ETH/USD
                assert "BTC/USD" in tokens
                assert "ETH/USD" in tokens
                assert pipeline.status == PipelineStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_performance_under_load(self, integration_config):
        """Test pipeline performance under concurrent load."""
        mock_scanner = MockEnhancedScanner(["BTC/USD", "ETH/USD"] * 10)  # Many tokens

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_get_scanner.return_value = mock_scanner

            pipeline = EvaluationPipelineIntegration(integration_config)
            await pipeline.initialize()

            # Test concurrent requests
            start_time = time.time()

            tasks = [
                pipeline.get_tokens_for_evaluation(20)
                for _ in range(10)  # 10 concurrent requests
            ]

            results = await asyncio.gather(*tasks)
            end_time = time.time()

            # Verify all requests succeeded
            assert len(results) == 10
            for result in results:
                assert len(result) > 0
                assert isinstance(result, list)

            # Performance check (should complete within reasonable time)
            total_time = end_time - start_time
            avg_time_per_request = total_time / 10

            assert avg_time_per_request < 5.0  # Less than 5 seconds per request
            assert pipeline.metrics.tokens_processed >= 100  # At least 100 tokens processed

    @pytest.mark.asyncio
    async def test_caching_and_performance_optimization(self, integration_config):
        """Test caching reduces redundant operations."""
        mock_scanner = MockEnhancedScanner(["BTC/USD", "ETH/USD"])

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_get_scanner.return_value = mock_scanner

            pipeline = EvaluationPipelineIntegration(integration_config)
            await pipeline.initialize()

            # First request
            start_time = time.time()
            tokens1 = await pipeline.get_tokens_for_evaluation(5)
            first_request_time = time.time() - start_time

            # Second request (should use cache)
            start_time = time.time()
            tokens2 = await pipeline.get_tokens_for_evaluation(5)
            second_request_time = time.time() - start_time

            # Results should be identical
            assert tokens1 == tokens2

            # Second request should be faster (cache hit)
            assert second_request_time <= first_request_time

            # Verify cache was used
            assert len(pipeline.token_cache) > 0

    @pytest.mark.asyncio
    async def test_health_monitoring_and_recovery(self, integration_config):
        """Test health monitoring and automatic recovery."""
        mock_scanner = MockEnhancedScanner()

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_get_scanner.return_value = mock_scanner

            pipeline = EvaluationPipelineIntegration(integration_config)
            await pipeline.initialize()

            # Start health monitoring
            await pipeline.start_health_monitoring()

            # Wait for health check
            await asyncio.sleep(1)

            # Verify health monitoring is working
            assert pipeline.last_health_check > 0
            assert pipeline.status == PipelineStatus.HEALTHY

            # Simulate failures
            pipeline.metrics.consecutive_failures = 6  # Above threshold

            # Wait for health check to detect issues
            await asyncio.sleep(1)

            # Stop monitoring
            await pipeline.stop_health_monitoring()

    @pytest.mark.asyncio
    async def test_configuration_persistence(self, integration_config):
        """Test configuration changes persist and take effect."""
        pipeline = EvaluationPipelineIntegration(integration_config)

        # Test initial config
        assert pipeline.pipeline_config.max_batch_size == 15

        # Modify config
        integration_config["evaluation_pipeline"]["max_batch_size"] = 25

        # Create new pipeline instance
        new_pipeline = EvaluationPipelineIntegration(integration_config)

        # Verify config change took effect
        assert new_pipeline.pipeline_config.max_batch_size == 25

    @pytest.mark.asyncio
    async def test_token_validation_and_quality(self, integration_config):
        """Test token validation and quality assurance."""
        mock_scanner = MockEnhancedScanner([
            "BTC/USD",      # Valid
            "ETH/USD",      # Valid
            "INVALID",      # Invalid format
            "btc/usd",      # Valid (case insensitive)
            "",             # Empty
            "SOL/USD",      # Valid
            "INVALID/USD",  # Valid format but questionable
        ])

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_get_scanner.return_value = mock_scanner

            pipeline = EvaluationPipelineIntegration(integration_config)
            await pipeline.initialize()

            tokens = await pipeline.get_tokens_for_evaluation(10)

            # Verify validation worked
            assert "BTC/USD" in tokens
            assert "ETH/USD" in tokens
            assert "SOL/USD" in tokens
            assert "INVALID" not in tokens  # Should be filtered out
            assert "" not in tokens        # Should be filtered out

            # Verify deduplication worked
            assert tokens.count("BTC/USD") == 1  # No duplicates

    @pytest.mark.asyncio
    async def test_error_isolation_and_recovery(self, integration_config):
        """Test that errors in one source don't affect others."""
        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner, \
             patch('crypto_bot.evaluation_pipeline_integration.EvaluationPipelineIntegration._get_solana_tokens') as mock_solana, \
             patch('crypto_bot.evaluation_pipeline_integration.EvaluationPipelineIntegration._get_config_tokens') as mock_config:

            # Scanner works
            mock_scanner = MockEnhancedScanner(["BTC/USD", "ETH/USD"])
            mock_get_scanner.return_value = mock_scanner

            # Solana fails
            mock_solana.side_effect = Exception("Solana network error")

            # Config works
            mock_config.return_value = ["SOL/USD", "ADA/USD"]

            pipeline = EvaluationPipelineIntegration(integration_config)
            await pipeline.initialize()

            tokens = await pipeline.get_tokens_for_evaluation(10)

            # Should still get tokens from working sources
            assert len(tokens) >= 4  # BTC, ETH, SOL, ADA
            assert "BTC/USD" in tokens
            assert "ETH/USD" in tokens
            assert "SOL/USD" in tokens
            assert "ADA/USD" in tokens

            # Status should be degraded due to partial failure
            assert pipeline.status in [PipelineStatus.DEGRADED, PipelineStatus.HEALTHY]


class TestMainBotIntegration:
    """Test integration with main bot components."""

    @pytest.mark.asyncio
    async def test_fetch_candidates_integration(self, integration_config, mock_exchange, mock_paper_wallet):
        """Test fetch_candidates function integration with pipeline."""
        # Create mock context
        ctx = Mock()
        ctx.config = integration_config
        ctx.current_batch = []
        ctx.timing = {}
        ctx.volatility_factor = 1.0

        # Mock dependencies
        with patch('crypto_bot.main.get_filtered_symbols', return_value=[("BTC/USD", 1.0), ("ETH/USD", 1.0)]), \
             patch('crypto_bot.main.get_solana_new_tokens', return_value=[]), \
             patch('crypto_bot.main.compute_average_atr', return_value=0.02), \
             patch('crypto_bot.main.build_priority_queue', return_value=asyncio.Queue()), \
             patch('crypto_bot.main.QUEUE_LOCK'):

            # Mock the queue for testing
            mock_queue = Mock()
            mock_queue.popleft = Mock(side_effect=["BTC/USD", "ETH/USD"])

            with patch('crypto_bot.main.symbol_priority_queue', mock_queue):
                # Execute fetch_candidates
                await fetch_candidates(ctx)

                # Verify pipeline was used
                assert len(ctx.current_batch) == 2
                assert "BTC/USD" in ctx.current_batch
                assert "ETH/USD" in ctx.current_batch
                assert "timing" in ctx.__dict__
                assert "symbol_time" in ctx.timing

    @pytest.mark.asyncio
    async def test_phase_runner_integration(self, integration_config):
        """Test PhaseRunner integration with pipeline."""
        # Mock the phases
        mock_phases = [
            Mock(return_value=asyncio.Future()),
            Mock(return_value=asyncio.Future()),
        ]
        for phase in mock_phases:
            phase.return_value.set_result(None)

        runner = PhaseRunner(mock_phases)

        # Create mock context
        ctx = Mock()
        ctx.config = integration_config
        ctx.current_batch = ["BTC/USD", "ETH/USD"]
        ctx.timing = {}

        # Execute runner
        with patch('asyncio.gather', return_value=[None, None]):
            result = await runner.run(ctx)

            assert result is not None
            assert isinstance(result, dict)


class TestProductionScenarios:
    """Test production-like scenarios and edge cases."""

    @pytest.mark.asyncio
    async def test_high_frequency_trading_scenario(self, integration_config):
        """Test behavior under high-frequency trading conditions."""
        # Increase batch size for high-frequency scenario
        integration_config["evaluation_pipeline"]["max_batch_size"] = 50
        integration_config["symbol_batch_size"] = 30

        mock_scanner = MockEnhancedScanner(["TOKEN" + str(i) + "/USD" for i in range(100)])

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_get_scanner.return_value = mock_scanner

            pipeline = EvaluationPipelineIntegration(integration_config)
            await pipeline.initialize()

            # Test rapid successive requests
            start_time = time.time()

            for _ in range(20):
                tokens = await pipeline.get_tokens_for_evaluation(30)
                assert len(tokens) <= 30  # Respect max_batch_size
                assert len(tokens) > 0

            end_time = time.time()
            total_time = end_time - start_time

            # Should handle high frequency without issues
            assert total_time < 30  # Complete within 30 seconds
            assert pipeline.metrics.tokens_processed >= 400  # At least 400 tokens processed

    @pytest.mark.asyncio
    async def test_network_failure_recovery(self, integration_config):
        """Test recovery from network failures."""
        mock_scanner = MockEnhancedScanner()

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_get_scanner.return_value = mock_scanner

            pipeline = EvaluationPipelineIntegration(integration_config)
            await pipeline.initialize()

            # Simulate network failures followed by recovery
            failure_count = 0

            for i in range(10):
                if i < 3:  # First 3 requests fail
                    with patch.object(pipeline, '_get_scanner_tokens', side_effect=Exception("Network error")):
                        tokens = await pipeline.get_tokens_for_evaluation(5)
                        failure_count += 1
                else:  # Subsequent requests succeed
                    tokens = await pipeline.get_tokens_for_evaluation(5)
                    assert len(tokens) > 0

            # Verify recovery
            assert failure_count == 3
            assert pipeline.metrics.consecutive_failures < 5  # Should recover
            assert pipeline.status != PipelineStatus.CRITICAL

    @pytest.mark.asyncio
    async def test_memory_and_resource_management(self, integration_config):
        """Test memory usage and resource cleanup."""
        mock_scanner = MockEnhancedScanner(["TOKEN" + str(i) + "/USD" for i in range(1000)])

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_get_scanner.return_value = mock_scanner

            pipeline = EvaluationPipelineIntegration(integration_config)
            await pipeline.initialize()

            # Process many requests to test memory management
            for _ in range(50):
                tokens = await pipeline.get_tokens_for_evaluation(20)
                assert len(tokens) > 0

                # Verify cache doesn't grow unbounded
                assert len(pipeline.token_cache) <= 10  # Cache should be cleaned

            # Cleanup
            await pipeline.cleanup()

            # Verify cleanup
            assert len(pipeline.token_cache) == 0
            assert pipeline.is_running is False

    @pytest.mark.asyncio
    async def test_configuration_hot_reload(self, integration_config):
        """Test configuration changes without restart."""
        pipeline = EvaluationPipelineIntegration(integration_config)

        # Initial config
        assert pipeline.pipeline_config.max_batch_size == 15

        # Simulate config change
        integration_config["evaluation_pipeline"]["max_batch_size"] = 30

        # Create new pipeline instance (simulating reload)
        new_pipeline = EvaluationPipelineIntegration(integration_config)

        # Verify config was reloaded
        assert new_pipeline.pipeline_config.max_batch_size == 30

        # Old pipeline should still work
        assert pipeline.pipeline_config.max_batch_size == 15


class TestMonitoringAndObservability:
    """Test monitoring and observability features."""

    @pytest.mark.asyncio
    async def test_comprehensive_metrics_collection(self, integration_config):
        """Test comprehensive metrics collection."""
        mock_scanner = MockEnhancedScanner(["BTC/USD", "ETH/USD"])

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_get_scanner.return_value = mock_scanner

            pipeline = EvaluationPipelineIntegration(integration_config)
            await pipeline.initialize()

            # Generate some activity
            for _ in range(5):
                await pipeline.get_tokens_for_evaluation(5)

            # Check comprehensive status
            status = pipeline.get_pipeline_status()

            required_fields = [
                "status", "metrics", "config", "scanner", "last_health_check"
            ]

            for field in required_fields:
                assert field in status

            # Check metrics are populated
            metrics = status["metrics"]
            assert metrics["tokens_received"] >= 5
            assert metrics["tokens_processed"] >= 5
            assert "avg_processing_time" in metrics
            assert "error_rate" in metrics

    @pytest.mark.asyncio
    async def test_alert_generation(self, integration_config):
        """Test alert generation for critical conditions."""
        pipeline = EvaluationPipelineIntegration(integration_config)

        # Simulate critical failure
        pipeline.metrics.consecutive_failures = 10
        pipeline._update_pipeline_status()

        assert pipeline.status == PipelineStatus.CRITICAL

        # Status should reflect critical condition
        status = pipeline.get_pipeline_status()
        assert status["status"] == "critical"


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])
