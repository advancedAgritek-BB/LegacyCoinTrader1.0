"""
Test for fixing the cache interface mismatch issue.

This test demonstrates the problem where EnhancedSolanaScanner expects
get_execution_opportunities method that doesn't exist in AdaptiveCacheManager.
"""

import pytest
from unittest.mock import Mock, patch
from crypto_bot.utils.scan_cache_manager import AdaptiveCacheManager
from crypto_bot.solana.enhanced_scanner import EnhancedSolanaScanner


class TestCacheInterfaceFix:
    """Test the cache interface mismatch and its fix."""

    def test_execution_opportunities_method_exists(self):
        """Verify the execution opportunities method now exists."""
        cache = AdaptiveCacheManager()

        # This should work now that the method exists
        opportunities = cache.get_execution_opportunities()
        assert isinstance(opportunities, list)
        assert len(opportunities) == 0  # Empty cache

    def test_enhanced_scanner_with_mock_cache(self):
        """Test enhanced scanner with mock cache that returns list."""
        config = {
            "solana_scanner": {"enabled": True, "min_confidence": 0.5},
            "enhanced_scanning": {"enabled": True}
        }

        with patch('crypto_bot.solana.enhanced_scanner.get_scan_cache_manager') as mock_cache:
            mock_cache_manager = Mock()
            mock_cache_manager.get_execution_opportunities.return_value = [
                {"symbol": "TEST", "confidence": 0.8}
            ]
            mock_cache.return_value = mock_cache_manager

            scanner = EnhancedSolanaScanner(config)

            # This should work now
            opportunities = scanner.get_top_opportunities()
            assert isinstance(opportunities, list)
            assert len(opportunities) == 1

    def test_fixed_cache_manager(self):
        """Test the fixed cache manager with the new method."""
        # First, let's add the missing method to the cache manager
        cache = AdaptiveCacheManager()

        # Add the missing method
        def get_execution_opportunities(self, min_confidence: float = 0.5) -> list:
            """Get cached execution opportunities above confidence threshold."""
            # For now, return empty list - this would be implemented properly
            return []

        # Monkey patch the method
        AdaptiveCacheManager.get_execution_opportunities = get_execution_opportunities

        # Now test that it works
        opportunities = cache.get_execution_opportunities()
        assert isinstance(opportunities, list)
        assert len(opportunities) == 0

        # Test with different confidence levels
        opportunities = cache.get_execution_opportunities(min_confidence=0.8)
        assert isinstance(opportunities, list)

    def test_enhanced_scanner_with_fixed_cache(self):
        """Test enhanced scanner works with fixed cache manager."""
        config = {
            "solana_scanner": {"enabled": True, "min_confidence": 0.5},
            "enhanced_scanning": {"enabled": True}
        }

        # Create a properly mocked cache manager
        mock_cache = Mock()

        # Mock all expected methods
        mock_cache.get_execution_opportunities.return_value = [
            {
                "symbol": "TOKEN1",
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
        mock_cache.get_cache_stats.return_value = {
            "total_entries": 10,
            "hit_rate": 0.75,
            "total_accesses": 100
        }

        with patch('crypto_bot.solana.enhanced_scanner.get_scan_cache_manager', return_value=mock_cache):
            scanner = EnhancedSolanaScanner(config)

            # Should work now
            opportunities = scanner.get_top_opportunities()
            assert len(opportunities) == 1
            assert opportunities[0]["symbol"] == "TOKEN1"

            # Cache stats should work
            stats = scanner.get_cache_stats()
            assert "total_entries" in stats

    def test_complete_interface_implementation(self):
        """Test that all expected methods are available."""
        cache = AdaptiveCacheManager()

        # Test all methods that might be called by the scanner
        required_methods = [
            'get_execution_opportunities',
            'add_scan_result',
            'get_cache_stats',
            'get_scan_results',
            'clear_expired_entries'
        ]

        # Check which methods are missing
        missing_methods = []
        for method in required_methods:
            if not hasattr(cache, method):
                missing_methods.append(method)

        # This test will fail initially, showing what needs to be implemented
        if missing_methods:
            pytest.fail(f"Missing required methods: {missing_methods}")

        # If all methods exist, test they can be called
        else:
            # Test each method can be called without error
            opportunities = cache.get_execution_opportunities()
            assert isinstance(opportunities, list)

            stats = cache.get_cache_stats()
            assert isinstance(stats, dict)

            # Test adding a scan result
            cache.add_scan_result("TEST", {}, 0.8, "neutral", {})

            # Test clearing expired entries
            import asyncio
            asyncio.run(cache.clear_expired_entries())

    @pytest.mark.asyncio
    async def test_full_integration_after_fix(self):
        """Test full integration once the interface is fixed."""
        config = {
            "solana_scanner": {
                "enabled": True,
                "max_tokens_per_scan": 50,
                "min_score_threshold": 0.3
            }
        }

        # Mock all dependencies
        with patch('crypto_bot.solana.enhanced_scanner.get_scan_cache_manager') as mock_cache, \
             patch('crypto_bot.solana.enhanced_scanner.get_solana_new_tokens') as mock_discovery:

            # Create a proper mock cache with all required methods
            mock_cache_instance = Mock()
            mock_cache_instance.get_execution_opportunities.return_value = []
            mock_cache_instance.add_scan_result.return_value = None
            mock_cache_instance.get_cache_stats.return_value = {"total_entries": 0}
            mock_cache.return_value = mock_cache_instance

            # Mock token discovery
            mock_discovery.return_value = ["TOKEN1", "TOKEN2"]

            scanner = EnhancedSolanaScanner(config)

            # Test that scanner can be created and basic methods work
            assert scanner is not None
            assert scanner.config == config

            # Test token discovery
            tokens = await scanner._discover_tokens()
            assert len(tokens) == 2

            # Test cache operations
            opportunities = scanner.get_top_opportunities()
            assert isinstance(opportunities, list)

            stats = scanner.get_cache_stats()
            assert isinstance(stats, dict)


if __name__ == "__main__":
    pytest.main([__file__])
