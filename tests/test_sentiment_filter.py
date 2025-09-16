import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from crypto_bot.sentiment_filter import (
    too_bearish,
    boost_factor,
    get_sentiment_score,
    get_lunarcrush_sentiment_boost,
    check_sentiment_alignment,
    SentimentDirection,
    SentimentData,
    LunarCrushClient,
)


def test_too_bearish(monkeypatch):
    """Test too_bearish function with mock FNG value."""
    monkeypatch.setenv("MOCK_FNG_VALUE", "10")
    # Note: too_bearish now uses a default sentiment value since it's not async
    assert too_bearish(20, 0.4) is True


def test_boost_factor(monkeypatch):
    """Test boost_factor function with mock FNG value."""
    monkeypatch.setenv("MOCK_FNG_VALUE", "80")
    # Note: boost_factor now uses a default sentiment value since it's not async
    # With FNG=80 and threshold=70, and default sentiment=0.5 vs threshold=0.6
    # The function should return 1.0 (no boost) since sentiment doesn't meet threshold
    assert boost_factor(70, 0.6) == 1.0


@pytest.mark.asyncio
async def test_get_sentiment_score():
    """Test get_sentiment_score function."""
    with patch('crypto_bot.sentiment_filter.get_lunarcrush_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        
        # Mock sentiment data
        mock_sentiment_data = type('MockSentimentData', (), {
            'sentiment': 0.75
        })()
        mock_client.get_sentiment.return_value = mock_sentiment_data
        
        result = await get_sentiment_score("bitcoin")
        assert result == 0.75
        mock_client.get_sentiment.assert_called_once_with("bitcoin")


@pytest.mark.asyncio
async def test_get_sentiment_score_fallback():
    """Test get_sentiment_score function fallback on error."""
    with patch('crypto_bot.sentiment_filter.get_lunarcrush_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        
        # Mock error
        mock_client.get_sentiment.side_effect = Exception("API Error")
        
        result = await get_sentiment_score("bitcoin")
        assert result == 0.5  # Default neutral sentiment


@pytest.mark.asyncio
async def test_get_lunarcrush_sentiment_boost():
    """Test get_lunarcrush_sentiment_boost function."""
    with patch('crypto_bot.sentiment_filter.get_lunarcrush_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        
        # Mock bullish sentiment data
        mock_sentiment_data = type('MockSentimentData', (), {
            'galaxy_score': 80.0,
            'sentiment': 0.8,
            'sentiment_direction': SentimentDirection.BULLISH
        })()
        mock_client.get_sentiment.return_value = mock_sentiment_data
        
        result = await get_lunarcrush_sentiment_boost("bitcoin", "long")
        assert result > 1.0  # Should provide a boost
        mock_client.get_sentiment.assert_called_once_with("bitcoin")


@pytest.mark.asyncio
async def test_get_lunarcrush_sentiment_boost_no_boost():
    """Test get_lunarcrush_sentiment_boost function when no boost should be given."""
    with patch('crypto_bot.sentiment_filter.get_lunarcrush_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        
        # Mock bearish sentiment data
        mock_sentiment_data = type('MockSentimentData', (), {
            'galaxy_score': 30.0,
            'sentiment': 0.3,
            'sentiment_direction': SentimentDirection.BEARISH
        })()
        mock_client.get_sentiment.return_value = mock_sentiment_data
        
        result = await get_lunarcrush_sentiment_boost("bitcoin", "long")
        assert result == 1.0  # No boost for bearish sentiment


@pytest.mark.asyncio
async def test_check_sentiment_alignment():
    """Test check_sentiment_alignment function."""
    with patch('crypto_bot.sentiment_filter.get_lunarcrush_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        
        # Mock bullish sentiment data
        mock_sentiment_data = type('MockSentimentData', (), {
            'sentiment': 0.7,
            'sentiment_direction': SentimentDirection.BULLISH
        })()
        mock_client.get_sentiment.return_value = mock_sentiment_data
        
        result = await check_sentiment_alignment("bitcoin", "long", require_alignment=True)
        assert result is True  # Bullish trade aligns with bullish sentiment


@pytest.mark.asyncio
async def test_check_sentiment_alignment_misaligned():
    """Test check_sentiment_alignment function with misaligned sentiment."""
    with patch('crypto_bot.sentiment_filter.get_lunarcrush_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        
        # Mock bearish sentiment data
        mock_sentiment_data = type('MockSentimentData', (), {
            'sentiment': 0.3,
            'sentiment_direction': SentimentDirection.BEARISH
        })()
        mock_client.get_sentiment.return_value = mock_sentiment_data
        
        result = await check_sentiment_alignment("bitcoin", "long", require_alignment=True)
        assert result is False  # Bullish trade doesn't align with bearish sentiment


@pytest.mark.asyncio
async def test_check_sentiment_alignment_fallback():
    """Test check_sentiment_alignment function fallback on error."""
    with patch('crypto_bot.sentiment_filter.get_lunarcrush_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        
        # Mock error
        mock_client.get_sentiment.side_effect = Exception("API Error")
        
        result = await check_sentiment_alignment("bitcoin", "long", require_alignment=True)
        assert result is True  # Fail safely by allowing the trade


@pytest.mark.asyncio
async def test_lunarcrush_client_integration():
    """Test LunarCrush client integration with mocked API responses."""
    with patch('crypto_bot.sentiment_filter.requests.Session') as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "galaxy_score": 85.5,
                "alt_rank": 45,
                "sentiment": 78.2,
                "social_mentions": 1250,
                "social_volume": 456.7
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        
        # Create client and test sentiment fetch
        client = LunarCrushClient("test_api_key")
        sentiment_data = await client.get_sentiment("bitcoin")
        
        # Verify the data was parsed correctly
        assert sentiment_data.galaxy_score == 85.5
        assert sentiment_data.alt_rank == 45
        assert sentiment_data.sentiment == 0.782  # Normalized to 0-1 range
        assert sentiment_data.sentiment_direction == SentimentDirection.BULLISH
        assert sentiment_data.social_mentions == 1250
        assert sentiment_data.social_volume == 456.7


@pytest.mark.asyncio
async def test_lunarcrush_client_error_handling():
    """Test LunarCrush client error handling."""
    with patch('crypto_bot.sentiment_filter.requests.Session') as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        # Mock API error
        mock_session.get.side_effect = Exception("Network error")
        
        # Create client and test error handling
        client = LunarCrushClient("test_api_key")
        
        # The client should handle the error gracefully and return default data
        sentiment_data = await client.get_sentiment("bitcoin")
        assert sentiment_data.galaxy_score == 0.0  # Default value


@pytest.mark.asyncio
async def test_lunarcrush_client_caching():
    """Test LunarCrush client caching functionality."""
    with patch('crypto_bot.sentiment_filter.requests.Session') as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "galaxy_score": 75.0,
                "alt_rank": 100,
                "sentiment": 65.0,
                "social_mentions": 500,
                "social_volume": 200.0
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        
        # Create client
        client = LunarCrushClient("test_api_key")
        
        # First call should hit the API
        sentiment_data1 = await client.get_sentiment("ethereum")
        assert sentiment_data1.galaxy_score == 75.0
        
        # Second call should use cache (no additional API calls)
        sentiment_data2 = await client.get_sentiment("ethereum")
        assert sentiment_data2.galaxy_score == 75.0
        
        # Verify only one API call was made
        assert mock_session.get.call_count == 1


@pytest.mark.asyncio
async def test_get_trending_solana_tokens_parses_payload():
    """Ensure the trending helper normalizes v4 payload data."""
    with patch('crypto_bot.sentiment_filter.requests.Session') as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        trending_payload = {
            "data": {
                "coins": [
                    {
                        "symbol": "SOL",
                        "name": "Solana",
                        "chain": "solana",
                        "metrics": {
                            "galaxy_score": {"score": 75.5},
                            "alt_rank": {"rank": 12},
                            "average_sentiment": {"score": 68.0},
                            "social": {
                                "mentions": 1234,
                                "volume": 9876.5,
                            },
                        },
                    },
                    {
                        "symbol": "BONK",
                        "chain": "solana",
                        "metrics": {
                            "galaxyScore": 55.0,
                            "altRank": 200,
                            "sentiment": 42.0,
                            "socialMentions": 450,
                            "socialVolume": 1900.0,
                        },
                    },
                    {
                        "symbol": "ETH",
                        "chain": "ethereum",
                        "metrics": {
                            "galaxy_score": 90.0,
                            "sentiment": 80.0,
                        },
                    },
                ]
            }
        }

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = trending_payload
        mock_session.get.return_value = mock_response

        client = LunarCrushClient("test_api_key")
        results = await client.get_trending_solana_tokens(limit=3)

        assert len(results) == 2

        (symbol_sol, data_sol), (symbol_bonk, data_bonk) = results

        assert symbol_sol == "SOL"
        assert isinstance(data_sol, SentimentData)
        assert data_sol.galaxy_score == 75.5
        assert data_sol.alt_rank == 12
        assert data_sol.social_mentions == 1234
        assert data_sol.social_volume == pytest.approx(9876.5)
        assert data_sol.sentiment == pytest.approx(0.68)
        assert data_sol.sentiment_direction == SentimentDirection.BULLISH

        assert symbol_bonk == "BONK"
        assert data_bonk.sentiment_direction == SentimentDirection.NEUTRAL
        assert data_bonk.social_mentions == 450

        _, kwargs = mock_session.get.call_args
        assert kwargs["params"]["limit"] == 3
        assert kwargs["params"]["chains"] == "solana"


@pytest.mark.asyncio
async def test_get_trending_solana_tokens_uses_cache():
    """Verify that trending data responses are cached by the client."""
    with patch('crypto_bot.sentiment_filter.requests.Session') as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        trending_payload = {
            "data": {
                "coins": [
                    {
                        "symbol": "SOL",
                        "chain": "solana",
                        "metrics": {
                            "galaxy_score": 80.0,
                            "sentiment": 70.0,
                            "social_mentions": 300,
                            "social_volume": 1200,
                        },
                    },
                    {
                        "symbol": "BONK",
                        "chain": "solana",
                        "metrics": {
                            "galaxy_score": 60.0,
                            "sentiment": 55.0,
                            "social_mentions": 200,
                            "social_volume": 800,
                        },
                    },
                ]
            }
        }

        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = trending_payload
        mock_session.get.return_value = mock_response

        client = LunarCrushClient("test_api_key")
        first_results = await client.get_trending_solana_tokens(limit=2)
        assert len(first_results) == 2
        assert mock_session.get.call_count == 1

        second_results = await client.get_trending_solana_tokens(limit=1)
        assert len(second_results) == 1
        assert mock_session.get.call_count == 1  # No additional HTTP calls
        assert second_results[0][0] == first_results[0][0]


def test_sentiment_direction_enum():
    """Test SentimentDirection enum values."""
    assert SentimentDirection.BULLISH.value == "bullish"
    assert SentimentDirection.BEARISH.value == "bearish"
    assert SentimentDirection.NEUTRAL.value == "neutral"


def test_sentiment_data_properties():
    """Test SentimentData properties."""
    from crypto_bot.sentiment_filter import SentimentData
    
    # Test bullish strength
    bullish_data = SentimentData(
        sentiment=0.8,
        sentiment_direction=SentimentDirection.BULLISH
    )
    assert bullish_data.bullish_strength > 1.0
    
    # Test bearish strength
    bearish_data = SentimentData(
        sentiment=0.3,
        sentiment_direction=SentimentDirection.BEARISH
    )
    assert bearish_data.bullish_strength < 1.0
    
    # Test neutral strength
    neutral_data = SentimentData(
        sentiment=0.5,
        sentiment_direction=SentimentDirection.NEUTRAL
    )
    assert neutral_data.bullish_strength == 1.0

