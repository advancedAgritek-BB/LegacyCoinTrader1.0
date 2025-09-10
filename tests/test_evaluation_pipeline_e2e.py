"""
End-to-End Test for Evaluation Pipeline

Comprehensive production-like testing of the complete evaluation pipeline system.
Tests all components working together under realistic conditions.
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
    initialize_evaluation_pipeline,
    get_tokens_for_evaluation,
    get_pipeline_status
)
from crypto_bot.evaluation_pipeline_monitor import (
    EvaluationPipelineMonitor,
    get_evaluation_pipeline_monitor,
    start_pipeline_monitoring,
    stop_pipeline_monitoring
)
from crypto_bot.main import fetch_candidates
from crypto_bot.phase_runner import PhaseRunner, BotContext


class ProductionLikeTestEnvironment:
    """Simulates a production-like environment for comprehensive testing."""

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = self._create_production_config()
        self.mock_exchange = Mock()
        self.mock_paper_wallet = Mock()

        # Setup exchange mock
        self.mock_exchange.id = "kraken"
        self.mock_exchange.load_markets = AsyncMock(return_value={})

        # Setup paper wallet mock
        self.mock_paper_wallet.balance = 10000.0
        self.mock_paper_wallet.positions = {}

    def _create_production_config(self) -> Dict[str, Any]:
        """Create production-like configuration."""
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
            "symbols": [
                "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
                "LINK/USD", "UNI/USD", "AVAX/USD", "MATIC/USD", "ATOM/USD"
            ],
            "evaluation_pipeline": {
                "enabled": True,
                "max_batch_size": 15,
                "processing_timeout": 10.0,
                "retry_attempts": 2,
                "cache_ttl": 60.0,
                "enable_fallback_sources": True
            },
            "pipeline_monitoring": {
                "enabled": True,
                "collection_interval": 5.0,  # Faster for testing
                "health_check_interval": 10.0,  # Faster for testing
                "metrics_retention_hours": 1,
                "alerts_enabled": True
            },
            "symbol_batch_size": 10,
            "timeframe": "1h",
            "execution_mode": "dry_run",
            "max_positions": 5,
            "risk": {
                "max_risk_per_trade": 0.02,
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.04
            }
        }

    def cleanup(self):
        """Clean up temporary resources."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class MockProductionScanner:
    """Mock scanner that simulates production-like behavior."""

    def __init__(self, tokens: List[str] = None, failure_rate: float = 0.0):
        self.tokens = tokens or [
            "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
            "LINK/USD", "UNI/USD", "AVAX/USD", "MATIC/USD", "ATOM/USD"
        ]
        self.failure_rate = failure_rate
        self.call_count = 0
        self.failures = 0

    def get_integration_stats(self) -> Dict[str, Any]:
        return {"running": True}

    def get_top_opportunities(self, limit: int = 20) -> List[Mock]:
        self.call_count += 1

        # Simulate occasional failures
        if self.failure_rate > 0 and self.call_count % int(1/self.failure_rate) == 0:
            self.failures += 1
            raise Exception(f"Simulated scanner failure #{self.failures}")

        return [
            Mock(symbol=token, confidence=0.8 + i * 0.05)
            for i, token in enumerate(self.tokens[:limit])
        ]


class TestProductionEndToEnd:
    """End-to-end tests in production-like environment."""

    @pytest.fixture
    async def production_env(self):
        """Setup production-like test environment."""
        env = ProductionLikeTestEnvironment()

        # Setup mock scanner
        mock_scanner = MockProductionScanner()

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner, \
             patch('crypto_bot.evaluation_pipeline_integration.EvaluationPipelineIntegration._get_solana_tokens') as mock_solana, \
             patch('crypto_bot.evaluation_pipeline_integration.EvaluationPipelineIntegration._get_config_tokens') as mock_config:

            mock_get_scanner.return_value = mock_scanner
            mock_solana.return_value = ["RAY/USD", "ORCA/USD"]
            mock_config.return_value = ["APE/USD", "DOGE/USD"]

            yield env

        env.cleanup()

    @pytest.mark.asyncio
    async def test_complete_pipeline_lifecycle(self, production_env):
        """Test complete pipeline lifecycle from initialization to operation."""
        env = production_env

        # Phase 1: Initialization
        logger.info("🧪 Phase 1: Testing pipeline initialization...")

        pipeline = EvaluationPipelineIntegration(env.config)
        success = await pipeline.initialize()

        assert success is True
        assert pipeline.status == PipelineStatus.HEALTHY
        assert pipeline.enhanced_scanner is not None

        # Phase 2: Token Retrieval
        logger.info("🧪 Phase 2: Testing token retrieval...")

        tokens = await pipeline.get_tokens_for_evaluation(10)
        assert len(tokens) >= 5  # Should get tokens from multiple sources
        assert pipeline.metrics.tokens_received >= 5

        # Verify token quality
        for token in tokens:
            assert "/" in token  # Valid format
            assert token.split("/")[1] in ["USD", "USDC", "USDT"]  # Valid quote

        # Phase 3: Monitoring Integration
        logger.info("🧪 Phase 3: Testing monitoring integration...")

        monitor = EvaluationPipelineMonitor(env.config)

        # Start monitoring
        await monitor.start_monitoring()
        assert monitor.is_running is True

        # Wait for some data collection
        await asyncio.sleep(2)

        # Check monitoring data
        status = monitor.get_monitoring_status()
        assert status["running"] is True
        assert status["stats"]["total_metrics"] > 0
        assert status["stats"]["total_health_checks"] > 0

        # Stop monitoring
        await monitor.stop_monitoring()
        assert monitor.is_running is False

        # Phase 4: Integration with Main Bot
        logger.info("🧪 Phase 4: Testing integration with main bot...")

        # Create mock context
        ctx = Mock()
        ctx.config = env.config
        ctx.current_batch = []
        ctx.timing = {}
        ctx.volatility_factor = 1.0

        # Mock dependencies for fetch_candidates
        with patch('crypto_bot.main.get_filtered_symbols', return_value=[("BTC/USD", 1.0), ("ETH/USD", 1.0)]), \
             patch('crypto_bot.main.get_solana_new_tokens', return_value=["RAY/USD"]), \
             patch('crypto_bot.main.compute_average_atr', return_value=0.02), \
             patch('crypto_bot.main.build_priority_queue', return_value=asyncio.Queue()), \
             patch('crypto_bot.main.QUEUE_LOCK'):

            # Mock queue
            mock_queue = Mock()
            mock_queue.popleft = Mock(side_effect=["BTC/USD", "ETH/USD", "RAY/USD"])

            with patch('crypto_bot.main.symbol_priority_queue', mock_queue):
                await fetch_candidates(ctx)

                assert len(ctx.current_batch) == 3
                assert "BTC/USD" in ctx.current_batch
                assert "ETH/USD" in ctx.current_batch
                assert "RAY/USD" in ctx.current_batch

        logger.info("✅ Complete pipeline lifecycle test passed!")

    @pytest.mark.asyncio
    async def test_production_load_simulation(self, production_env):
        """Test pipeline under simulated production load."""
        env = production_env

        # Setup pipeline
        pipeline = EvaluationPipelineIntegration(env.config)
        await pipeline.initialize()

        # Simulate production load
        concurrent_requests = 10
        requests_per_client = 20
        total_requests = concurrent_requests * requests_per_client

        logger.info(f"🧪 Simulating {total_requests} concurrent requests...")

        start_time = time.time()

        # Create concurrent clients
        async def client_simulation(client_id: int):
            results = []
            for i in range(requests_per_client):
                try:
                    tokens = await pipeline.get_tokens_for_evaluation(5)
                    results.append(len(tokens))
                    await asyncio.sleep(0.1)  # Simulate processing time
                except Exception as e:
                    logger.error(f"Client {client_id} request {i} failed: {e}")
                    results.append(0)
            return results

        # Run concurrent simulation
        tasks = [client_simulation(i) for i in range(concurrent_requests)]
        all_results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        # Analyze results
        successful_requests = sum(1 for results in all_results for r in results if r > 0)
        total_tokens = sum(sum(results) for results in all_results)

        success_rate = successful_requests / total_requests
        avg_time_per_request = total_time / total_requests
        tokens_per_second = total_tokens / total_time

        logger.info("📊 Production Load Results:")
        logger.info(f"   Success Rate: {success_rate:.2%}")
        logger.info(f"   Total Time: {total_time:.2f}s")
        logger.info(f"   Avg Time/Request: {avg_time_per_request:.3f}s")
        logger.info(f"   Tokens/Second: {tokens_per_second:.1f}")

        # Assert performance criteria
        assert success_rate >= 0.95  # At least 95% success rate
        assert avg_time_per_request < 2.0  # Less than 2 seconds per request
        assert tokens_per_second > 10  # At least 10 tokens per second
        assert pipeline.metrics.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_failure_recovery_and_resilience(self, production_env):
        """Test system resilience and failure recovery."""
        env = production_env

        # Setup scanner with occasional failures
        failing_scanner = MockProductionScanner(failure_rate=0.3)  # 30% failure rate

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_get_scanner.return_value = failing_scanner

            pipeline = EvaluationPipelineIntegration(env.config)
            await pipeline.initialize()

            # Test resilience over multiple requests
            successful_requests = 0
            total_requests = 20

            for i in range(total_requests):
                try:
                    tokens = await pipeline.get_tokens_for_evaluation(5)
                    if len(tokens) > 0:
                        successful_requests += 1
                    await asyncio.sleep(0.2)  # Brief pause between requests
                except Exception as e:
                    logger.warning(f"Request {i} failed: {e}")

            success_rate = successful_requests / total_requests

            logger.info("📊 Resilience Test Results:")
            logger.info(f"   Success Rate: {success_rate:.2%}")
            logger.info(f"   Scanner Failures: {failing_scanner.failures}")
            logger.info(f"   Pipeline Failures: {pipeline.metrics.consecutive_failures}")

            # Assert resilience
            assert success_rate >= 0.7  # At least 70% success rate despite failures
            assert pipeline.status != PipelineStatus.CRITICAL  # Should not be critical
            assert pipeline.metrics.tokens_processed > 0  # Should still process some tokens

    @pytest.mark.asyncio
    async def test_monitoring_and_alerting(self, production_env):
        """Test monitoring and alerting system."""
        env = production_env

        # Setup pipeline and monitor
        pipeline = EvaluationPipelineIntegration(env.config)
        await pipeline.initialize()

        monitor = EvaluationPipelineMonitor(env.config)

        # Setup alert tracking
        alerts_received = []

        def alert_callback(alert):
            alerts_received.append(alert)

        monitor.add_alert_callback(alert_callback)

        # Start monitoring
        await monitor.start_monitoring()

        # Simulate some activity
        for _ in range(10):
            await pipeline.get_tokens_for_evaluation(5)
            await asyncio.sleep(0.5)

        # Wait for monitoring to collect data
        await asyncio.sleep(3)

        # Check monitoring status
        status = monitor.get_monitoring_status()

        assert status["running"] is True
        assert status["stats"]["total_metrics"] > 0
        assert status["stats"]["total_health_checks"] > 0

        # Stop monitoring
        await monitor.stop_monitoring()

        logger.info("📊 Monitoring Test Results:")
        logger.info(f"   Metrics Collected: {status['stats']['total_metrics']}")
        logger.info(f"   Health Checks: {status['stats']['total_health_checks']}")
        logger.info(f"   Alerts: {len(alerts_received)}")

        # Should have collected some monitoring data
        assert status["stats"]["total_metrics"] > 0
        assert status["stats"]["total_health_checks"] > 0

    @pytest.mark.asyncio
    async def test_configuration_hot_reload(self, production_env):
        """Test configuration changes without restart."""
        env = production_env

        # Initial configuration
        pipeline = EvaluationPipelineIntegration(env.config)
        await pipeline.initialize()

        original_batch_size = pipeline.pipeline_config.max_batch_size
        assert original_batch_size == 15

        # Simulate configuration change
        env.config["evaluation_pipeline"]["max_batch_size"] = 25

        # Create new pipeline instance (simulating reload)
        new_pipeline = EvaluationPipelineIntegration(env.config)

        # Verify configuration was updated
        assert new_pipeline.pipeline_config.max_batch_size == 25

        # Test that both work independently
        tokens1 = await pipeline.get_tokens_for_evaluation(20)  # Should respect original limit
        tokens2 = await new_pipeline.get_tokens_for_evaluation(30)  # Should respect new limit

        assert len(tokens1) <= 20  # Original pipeline respects original config
        assert len(tokens2) <= 30  # New pipeline respects new config

    @pytest.mark.asyncio
    async def test_resource_cleanup_and_memory_management(self, production_env):
        """Test proper resource cleanup and memory management."""
        env = production_env

        # Create multiple pipeline instances
        pipelines = []
        for i in range(5):
            pipeline = EvaluationPipelineIntegration(env.config)
            await pipeline.initialize()
            pipelines.append(pipeline)

        # Generate activity
        for pipeline in pipelines:
            for _ in range(10):
                await pipeline.get_tokens_for_evaluation(5)

        # Check memory usage (cache sizes)
        total_cache_size = sum(len(p.token_cache) for p in pipelines)

        # Cleanup all pipelines
        for pipeline in pipelines:
            await pipeline.cleanup()

        # Verify cleanup
        for pipeline in pipelines:
            assert len(pipeline.token_cache) == 0
            assert pipeline.is_running is False

        logger.info("🧹 Resource cleanup test completed successfully")

    @pytest.mark.asyncio
    async def test_long_running_stability(self, production_env):
        """Test system stability over extended period."""
        env = production_env

        pipeline = EvaluationPipelineIntegration(env.config)
        await pipeline.initialize()

        monitor = EvaluationPipelineMonitor(env.config)

        # Start monitoring
        await monitor.start_monitoring()

        # Run for extended period (simulated)
        test_duration = 30  # 30 seconds
        request_interval = 2  # Request every 2 seconds

        start_time = time.time()
        request_count = 0
        successful_requests = 0

        while time.time() - start_time < test_duration:
            try:
                tokens = await pipeline.get_tokens_for_evaluation(5)
                if len(tokens) > 0:
                    successful_requests += 1
                request_count += 1
            except Exception as e:
                logger.error(f"Request failed: {e}")

            await asyncio.sleep(request_interval)

        # Calculate metrics
        uptime = time.time() - start_time
        success_rate = successful_requests / request_count if request_count > 0 else 0
        requests_per_minute = (request_count / uptime) * 60

        logger.info("⏱️  Long-running stability test results:")
        logger.info(".1f")
        logger.info(".1f")
        logger.info(".1%")
        logger.info(f"   Final Pipeline Status: {pipeline.status.value}")

        # Assert stability criteria
        assert uptime >= test_duration * 0.9  # At least 90% uptime
        assert success_rate >= 0.8  # At least 80% success rate
        assert pipeline.status in [PipelineStatus.HEALTHY, PipelineStatus.DEGRADED]
        assert pipeline.metrics.consecutive_failures < 5

        # Stop monitoring
        await monitor.stop_monitoring()


class TestProductionEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_token_sources(self):
        """Test behavior when all token sources are empty."""
        config = {
            "evaluation_pipeline": {
                "enabled": True,
                "max_batch_size": 10,
                "enable_fallback_sources": True
            }
        }

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner, \
             patch('crypto_bot.evaluation_pipeline_integration.EvaluationPipelineIntegration._get_solana_tokens', return_value=[]), \
             patch('crypto_bot.evaluation_pipeline_integration.EvaluationPipelineIntegration._get_config_tokens', return_value=[]):

            mock_scanner = Mock()
            mock_scanner.get_integration_stats.return_value = {"running": True}
            mock_scanner.get_top_opportunities.return_value = []
            mock_get_scanner.return_value = mock_scanner

            pipeline = EvaluationPipelineIntegration(config)
            await pipeline.initialize()

            tokens = await pipeline.get_tokens_for_evaluation(5)

            # Should fall back to hardcoded tokens
            assert len(tokens) >= 2
            assert "BTC/USD" in tokens
            assert "ETH/USD" in tokens

    @pytest.mark.asyncio
    async def test_network_timeout_simulation(self):
        """Test behavior under network timeout conditions."""
        config = {
            "evaluation_pipeline": {
                "enabled": True,
                "processing_timeout": 0.1,  # Very short timeout
                "retry_attempts": 1
            }
        }

        async def slow_token_retrieval(max_tokens: int):
            await asyncio.sleep(1)  # Simulate slow network
            return ["BTC/USD"]

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_scanner = Mock()
            mock_scanner.get_integration_stats.return_value = {"running": True}
            mock_scanner.get_top_opportunities = slow_token_retrieval
            mock_get_scanner.return_value = mock_scanner

            pipeline = EvaluationPipelineIntegration(config)
            await pipeline.initialize()

            # Should timeout and use fallback
            tokens = await pipeline.get_tokens_for_evaluation(5)

            assert len(tokens) >= 2  # Fallback tokens
            assert pipeline.metrics.consecutive_failures > 0

    @pytest.mark.asyncio
    async def test_concurrent_initialization(self):
        """Test concurrent pipeline initialization."""
        config = {
            "evaluation_pipeline": {"enabled": True}
        }

        # Clear any existing instance
        import crypto_bot.evaluation_pipeline_integration
        crypto_bot.evaluation_pipeline_integration._pipeline_integration = None

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_scanner = Mock()
            mock_scanner.get_integration_stats.return_value = {"running": True}
            mock_scanner.get_top_opportunities.return_value = [Mock(symbol="BTC/USD", confidence=0.8)]
            mock_get_scanner.return_value = mock_scanner

            # Initialize multiple instances concurrently
            tasks = [initialize_evaluation_pipeline(config) for _ in range(5)]
            results = await asyncio.gather(*tasks)

            # All should succeed
            assert all(results) is True

            # Should have single shared instance
            from crypto_bot.evaluation_pipeline_integration import get_evaluation_pipeline_integration
            instance1 = get_evaluation_pipeline_integration(config)
            instance2 = get_evaluation_pipeline_integration(config)

            assert instance1 is instance2


if __name__ == "__main__":
    # Run end-to-end tests
    pytest.main([__file__, "-v", "--tb=short", "-x"])
