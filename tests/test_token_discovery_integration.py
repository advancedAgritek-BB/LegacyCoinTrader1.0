"""
Integration test for the complete token discovery system.

This test demonstrates that the fixed token discovery system works end-to-end.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from crypto_bot.solana.scanner import get_solana_new_tokens
from crypto_bot.solana.enhanced_scanner import EnhancedSolanaScanner
from crypto_bot.utils.scan_cache_manager import get_scan_cache_manager


class TestTokenDiscoveryIntegration:
    """Integration tests for the complete token discovery system."""

    @pytest.fixture
    def working_config(self):
        """A working configuration for the token discovery system."""
        return {
            "solana_scanner": {
                "enabled": True,
                "limit": 50,
                "url": "https://api.helius.xyz/?api-key=test_key",
                "pump_fun_api_key": "test_pump_key",
                "max_tokens_per_scan": 100,
                "min_score_threshold": 0.3,
                "enable_sentiment": False,
                "enable_pyth_prices": False,
                "min_volume_usd": 10000,
                "max_spread_pct": 2.0,
                "min_liquidity_score": 0.5,
                "min_strategy_fit": 0.6,
                "min_confidence": 0.5
            }
        }

    @pytest.mark.asyncio
    async def test_end_to_end_token_discovery(self, working_config):
        """Test the complete token discovery pipeline."""
        # Mock all external dependencies
        with patch('crypto_bot.solana.scanner.aiohttp.ClientSession') as mock_session_class, \
             patch('crypto_bot.solana.enhanced_scanner.get_scan_cache_manager') as mock_cache:

            # Setup mock session
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Mock successful API responses
            mock_helius_response = {
                "result": [
                    {"tokenA": "TOKEN1", "tokenB": "TOKEN2"},
                    {"tokenA": "TOKEN3", "tokenB": "TOKEN4"}
                ]
            }

            # Mock all API calls to return successful responses
            mock_session.post.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_helius_response)
            mock_session.get.side_effect = [
                # Raydium fallback
                Exception("Raydium failed"),
                # Orca fallback
                Exception("Orca failed"),
                # Jupiter fallback
                Exception("Jupiter failed"),
                # pump.fun fallback
                Exception("pump.fun failed")
            ]

            # Setup mock cache
            mock_cache_manager = Mock()
            mock_cache_manager.get_execution_opportunities.return_value = []
            mock_cache_manager.add_scan_result.return_value = None
            mock_cache_manager.get_cache_stats.return_value = {"total_entries": 0}
            mock_cache.return_value = mock_cache_manager

            # Test 1: Basic token discovery works
            tokens = await get_solana_new_tokens(working_config["solana_scanner"])

            # Should get some tokens (fallback to static list if APIs fail)
            assert isinstance(tokens, list)
            assert len(tokens) > 0

            print(f"✓ Token discovery returned {len(tokens)} tokens")

            # Test 2: Enhanced scanner can be created and initialized
            scanner = EnhancedSolanaScanner(working_config)

            # Verify configuration is properly loaded
            assert scanner.max_tokens_per_scan == 100
            assert scanner.min_score_threshold == 0.3
            assert scanner.min_confidence == 0.5

            print("✓ Enhanced scanner initialized with proper configuration")

            # Test 3: Cache manager integration works
            opportunities = scanner.get_top_opportunities()
            assert isinstance(opportunities, list)

            cache_stats = scanner.get_cache_stats()
            assert isinstance(cache_stats, dict)

            print("✓ Cache manager integration working")

            # Test 4: Scanner can discover tokens (mocked)
            with patch.object(scanner, '_analyze_single_token') as mock_analyze:
                # Mock analysis to return valid market conditions
                from crypto_bot.solana.enhanced_scanner import MarketConditions
                mock_conditions = MarketConditions(
                    price=1.0,
                    volume_24h=50000.0,
                    volume_ma=45000.0,
                    price_change_24h=0.02,
                    price_change_7d=0.1,
                    atr=0.001,
                    atr_percent=0.05,
                    spread_pct=0.5,
                    liquidity_score=0.8,
                    volatility_score=0.6,
                    momentum_score=0.7
                )
                mock_analyze.return_value = mock_conditions

                discovered = await scanner._discover_tokens()
                assert isinstance(discovered, list)

                if discovered:  # If we got tokens to analyze
                    analyzed = await scanner._analyze_tokens(discovered)
                    assert isinstance(analyzed, dict)

                    if analyzed:  # If we got analyzed tokens
                        scored = scanner._score_tokens(analyzed)
                        assert isinstance(scored, list)

                        print(f"✓ Token analysis pipeline working: {len(discovered)} discovered, {len(analyzed)} analyzed, {len(scored)} scored")

            print("🎉 All integration tests passed!")

    def test_configuration_validation(self, working_config):
        """Test that configuration validation works properly."""
        # Test invalid configuration
        invalid_config = {
            "solana_scanner": {
                "enabled": True,
                "max_tokens_per_scan": 0,  # Invalid: should be > 0
                "min_score_threshold": 1.5,  # Invalid: should be <= 1.0
                "min_volume_usd": -1000,  # Invalid: should be >= 0
            }
        }

        with patch('crypto_bot.solana.enhanced_scanner.get_scan_cache_manager', return_value=Mock()):
            scanner = EnhancedSolanaScanner(invalid_config)

            # Should be corrected to valid values
            assert scanner.max_tokens_per_scan > 0
            assert 0.0 <= scanner.min_score_threshold <= 1.0
            assert scanner.min_volume_usd >= 0

            print("✓ Configuration validation working properly")

    def test_cache_size_limits(self):
        """Test that cache respects size limits."""
        from crypto_bot.utils.scan_cache_manager import AdaptiveCacheManager

        cache = AdaptiveCacheManager(initial_size=5, max_size=10)

        # Add entries up to the limit
        for i in range(15):
            cache.set("test_cache", f"key{i}", f"value{i}")

        # Should not exceed max_size
        assert len(cache.caches["test_cache"]) <= 10

        print("✓ Cache size limits working properly")

    def test_error_handling_and_fallbacks(self, working_config):
        """Test that the system handles errors gracefully with fallbacks."""
        with patch('crypto_bot.solana.scanner.aiohttp.ClientSession') as mock_session_class, \
             patch('crypto_bot.solana.enhanced_scanner.get_scan_cache_manager', return_value=Mock()):

            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Make all APIs fail
            mock_session.post.side_effect = Exception("All APIs failed")
            mock_session.get.side_effect = Exception("All APIs failed")

            # Should still return tokens (fallback to static list)
            tokens = asyncio.run(get_solana_new_tokens(working_config["solana_scanner"]))

            assert isinstance(tokens, list)
            assert len(tokens) > 0  # Should have fallback tokens

            print("✓ Error handling and fallbacks working properly")

    def test_system_components_integration(self, working_config):
        """Test that all system components work together."""
        with patch('crypto_bot.solana.enhanced_scanner.get_scan_cache_manager') as mock_cache:
            # Create a complete mock cache with all required methods
            mock_cache_instance = Mock()
            mock_cache_instance.get_execution_opportunities.return_value = [
                {
                    "symbol": "TEST_TOKEN",
                    "strategy": "momentum",
                    "direction": "long",
                    "confidence": 0.8,
                    "entry_price": 1.0,
                    "stop_loss": 0.95,
                    "take_profit": 1.1,
                    "risk_reward_ratio": 2.0,
                    "timestamp": 1234567890
                }
            ]
            mock_cache_instance.add_scan_result.return_value = None
            mock_cache_instance.get_cache_stats.return_value = {
                "total_entries": 1,
                "hit_rate": 0.8,
                "total_accesses": 10
            }
            mock_cache.return_value = mock_cache_instance

            # Create scanner
            scanner = EnhancedSolanaScanner(working_config)

            # Test all integration points
            opportunities = scanner.get_top_opportunities()
            assert len(opportunities) == 1
            assert opportunities[0]["symbol"] == "TEST_TOKEN"

            stats = scanner.get_cache_stats()
            assert stats["total_entries"] == 1

            print("✓ All system components integrated properly")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
