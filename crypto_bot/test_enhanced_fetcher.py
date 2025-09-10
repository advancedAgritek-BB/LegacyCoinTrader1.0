"""
Test for the enhanced OHLCV fetcher.
"""

import pytest
from unittest.mock import Mock
from crypto_bot.utils.enhanced_ohlcv_fetcher import EnhancedOHLCVFetcher
from crypto_bot.utils.config_validator import get_exchange_supported_timeframes


@pytest.mark.asyncio
async def test_enhanced_fetcher():
    """Test the enhanced OHLCV fetcher functionality."""
    
    # Mock exchange object for testing
    class MockExchange:
        def __init__(self):
            self.id = 'kraken'
            self.timeframes = get_exchange_supported_timeframes('kraken')
    
    # Mock config
    config = {
        'exchange': 'kraken',
        'timeframes': ['1m', '5m', '15m', '1h', '4h']
    }
    
    # Create fetcher
    exchange = MockExchange()
    fetcher = EnhancedOHLCVFetcher(exchange, config)

    # Test timeframe validation
    test_timeframes = ['1m', '5m', '10m', '15m', '30m', '1h', '4h', '45m']

    # Test that fetcher can be created
    assert fetcher is not None
    assert exchange.id == 'kraken'
    assert len(fetcher.supported_timeframes) > 0

    # Test timeframe validation
    for tf in test_timeframes:
        is_supported, message = fetcher.validate_timeframe_request(tf)
        assert isinstance(is_supported, bool)
        assert isinstance(message, str)

    # Test timeframe fallback
    for tf in test_timeframes:
        supported_tf = fetcher.get_supported_timeframe(tf)
        assert isinstance(supported_tf, str)
        assert supported_tf in fetcher.supported_timeframes

    # Test closest timeframe finding
    test_unsupported = ['10m', '30m', '6h', '2w']
    for tf in test_unsupported:
        closest = fetcher._find_closest_timeframe(tf)
        assert isinstance(closest, str)
        assert closest in fetcher.supported_timeframes
