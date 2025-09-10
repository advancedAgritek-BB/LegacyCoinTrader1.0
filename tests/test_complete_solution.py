"""
Complete Solution Integration Test

Final integration test that verifies the entire bulletproof evaluation pipeline
solution works flawlessly from end-to-end.
"""

import asyncio
import pytest
import time
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from crypto_bot.evaluation_pipeline_integration import (
    EvaluationPipelineIntegration,
    initialize_evaluation_pipeline,
    get_tokens_for_evaluation,
    get_pipeline_status
)
from crypto_bot.evaluation_pipeline_monitor import (
    EvaluationPipelineMonitor,
    start_pipeline_monitoring,
    stop_pipeline_monitoring,
    get_monitoring_status
)
from crypto_bot.main import fetch_candidates, _main_impl
from crypto_bot.phase_runner import PhaseRunner


class CompleteSolutionTest:
    """Test the complete bulletproof solution."""

    @pytest.mark.asyncio
    async def test_solution_initialization(self):
        """Test complete solution initialization."""
        config = self._create_test_config()

        # Initialize pipeline
        pipeline_success = await initialize_evaluation_pipeline(config)
        assert pipeline_success is True

        # Initialize monitoring
        await start_pipeline_monitoring(config)

        # Verify initialization
        pipeline_status = get_pipeline_status(config)
        monitor_status = get_monitoring_status(config)

        assert pipeline_status["status"] == "healthy"
        assert monitor_status["running"] is True

        # Cleanup
        await stop_pipeline_monitoring()

    @pytest.mark.asyncio
    async def test_token_flow_end_to_end(self):
        """Test complete token flow from scanning to evaluation."""
        config = self._create_test_config()

        # Mock scanner
        mock_scanner = Mock()
        mock_scanner.get_integration_stats.return_value = {"running": True}
        mock_scanner.get_top_opportunities.return_value = [
            Mock(symbol="BTC/USD", confidence=0.9),
            Mock(symbol="ETH/USD", confidence=0.8),
            Mock(symbol="SOL/USD", confidence=0.7)
        ]

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_get_scanner.return_value = mock_scanner

            # Initialize
            await initialize_evaluation_pipeline(config)

            # Get tokens
            tokens = await get_tokens_for_evaluation(config, 10)

            # Verify tokens
            assert len(tokens) >= 3
            assert "BTC/USD" in tokens
            assert "ETH/USD" in tokens
            assert "SOL/USD" in tokens

    @pytest.mark.asyncio
    async def test_monitoring_integration(self):
        """Test monitoring integration with pipeline."""
        config = self._create_test_config()

        # Start monitoring
        await start_pipeline_monitoring(config)

        # Wait for data collection
        await asyncio.sleep(2)

        # Check monitoring status
        status = get_monitoring_status(config)

        assert status["running"] is True
        assert status["stats"]["total_metrics"] > 0
        assert status["stats"]["total_health_checks"] > 0

        # Stop monitoring
        await stop_pipeline_monitoring()

    @pytest.mark.asyncio
    async def test_main_bot_integration(self):
        """Test integration with main bot trading loop."""
        config = self._create_test_config()

        # Mock components
        mock_exchange = Mock()
        mock_exchange.id = "kraken"
        mock_exchange.load_markets = AsyncMock(return_value={})

        mock_paper_wallet = Mock()
        mock_paper_wallet.balance = 10000.0
        mock_paper_wallet.positions = {}

        # Mock pipeline tokens
        with patch('crypto_bot.evaluation_pipeline_integration.get_tokens_for_evaluation') as mock_get_tokens, \
             patch('crypto_bot.main.get_filtered_symbols', return_value=[("BTC/USD", 1.0)]), \
             patch('crypto_bot.main.get_solana_new_tokens', return_value=[]), \
             patch('crypto_bot.main.compute_average_atr', return_value=0.02), \
             patch('crypto_bot.main.build_priority_queue', return_value=asyncio.Queue()), \
             patch('crypto_bot.main.QUEUE_LOCK'):

            mock_get_tokens.return_value = ["BTC/USD", "ETH/USD", "SOL/USD"]

            # Mock queue
            mock_queue = Mock()
            mock_queue.popleft = Mock(side_effect=["BTC/USD", "ETH/USD", "SOL/USD"])

            with patch('crypto_bot.main.symbol_priority_queue', mock_queue):
                # Create mock context
                ctx = Mock()
                ctx.config = config
                ctx.current_batch = []
                ctx.timing = {}
                ctx.volatility_factor = 1.0

                # Execute fetch_candidates
                await fetch_candidates(ctx)

                # Verify integration
                assert len(ctx.current_batch) == 3
                assert "BTC/USD" in ctx.current_batch
                assert "ETH/USD" in ctx.current_batch
                assert "SOL/USD" in ctx.current_batch

    @pytest.mark.asyncio
    async def test_production_simulation(self):
        """Test complete solution under production-like conditions."""
        config = self._create_test_config()

        # Initialize complete solution
        await initialize_evaluation_pipeline(config)
        await start_pipeline_monitoring(config)

        # Simulate production workload
        successful_requests = 0
        total_requests = 50

        start_time = time.time()

        for i in range(total_requests):
            try:
                tokens = await get_tokens_for_evaluation(config, 5)
                if len(tokens) > 0:
                    successful_requests += 1
            except Exception as e:
                print(f"Request {i} failed: {e}")

        end_time = time.time()

        # Calculate metrics
        duration = end_time - start_time
        success_rate = successful_requests / total_requests
        requests_per_second = total_requests / duration

        print("📊 Production Simulation Results:")
        print(f"   Success Rate: {success_rate:.2%}")
        print(f"   Requests/Second: {requests_per_second:.1f}")

        # Verify performance
        assert success_rate >= 0.9  # 90% success rate
        assert requests_per_second >= 1.0  # At least 1 request per second
        assert duration < 60  # Complete within 1 minute

        # Check monitoring
        monitor_status = get_monitoring_status(config)
        assert monitor_status["stats"]["total_metrics"] > 0

        # Cleanup
        await stop_pipeline_monitoring()

    @pytest.mark.asyncio
    async def test_failure_recovery(self):
        """Test complete solution failure recovery."""
        config = self._create_test_config()

        # Initialize with failing scanner
        mock_scanner = Mock()
        mock_scanner.get_integration_stats.return_value = {"running": False}

        with patch('crypto_bot.evaluation_pipeline_integration.get_enhanced_scan_integration') as mock_get_scanner:
            mock_get_scanner.return_value = mock_scanner

            # Initialize (should handle failure gracefully)
            success = await initialize_evaluation_pipeline(config)
            assert success is False  # Should fail but not crash

            # Should still provide fallback tokens
            tokens = await get_tokens_for_evaluation(config, 5)
            assert len(tokens) >= 2  # Fallback tokens

    def _create_test_config(self):
        """Create test configuration."""
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
                "max_batch_size": 15,
                "processing_timeout": 10.0,
                "enable_fallback_sources": True
            },
            "pipeline_monitoring": {
                "enabled": True,
                "collection_interval": 5.0,
                "health_check_interval": 10.0,
                "alerts_enabled": True
            },
            "symbol_batch_size": 10,
            "timeframe": "1h"
        }


def test_solution_readme():
    """Test that solution documentation exists and is complete."""
    readme_path = Path(__file__).parent.parent / "EVALUATION_PIPELINE_README.md"

    if readme_path.exists():
        with open(readme_path, 'r') as f:
            content = f.read()

        # Check for key sections
        assert "Evaluation Pipeline" in content
        assert "Installation" in content
        assert "Configuration" in content
        assert "Usage" in content
        assert "Monitoring" in content
        assert "Testing" in content
    else:
        pytest.skip("README not found - documentation should be created")


if __name__ == "__main__":
    # Run complete solution tests
    pytest.main([__file__, "-v", "--tb=short"])
