import time
from unittest.mock import AsyncMock, patch

import pytest

from crypto_bot.sentiment_filter import SentimentData, SentimentDirection
from crypto_bot.solana.social_sentiment_analyzer import SocialSentimentAnalyzer


@pytest.mark.asyncio
async def test_get_lunarcrush_data_uses_client_and_records_call():
    """Ensure LunarCrush sentiment retrieval uses the shared client and logs calls."""
    analyzer = SocialSentimentAnalyzer({
        "social_sentiment": {"lunarcrush_api_key": "test-key"}
    })

    sentiment_data = SentimentData(
        galaxy_score=72.5,
        alt_rank=42,
        sentiment=0.68,
        sentiment_direction=SentimentDirection.BULLISH,
        social_mentions=320,
        social_volume=540.0,
        last_updated=time.time()
    )

    with patch("crypto_bot.solana.social_sentiment_analyzer.get_lunarcrush_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.get_sentiment.return_value = sentiment_data
        mock_get_client.return_value = mock_client

        result = await analyzer._get_lunarcrush_data("SOL")

    assert result is sentiment_data
    mock_get_client.assert_called_once()
    mock_client.get_sentiment.assert_called_once_with("SOL")
    assert len(analyzer.api_calls["lunarcrush"]) == 1
    assert analyzer.stats["api_calls_made"] == 1


def test_process_lunarcrush_data_maps_sentiment_fields():
    """Verify LunarCrush sentiment data is translated into social signals."""
    analyzer = SocialSentimentAnalyzer({
        "social_sentiment": {"lunarcrush_api_key": "test-key"}
    })

    sentiment_data = SentimentData(
        galaxy_score=80.0,
        alt_rank=15,
        sentiment=0.75,
        sentiment_direction=SentimentDirection.BULLISH,
        social_mentions=500,
        social_volume=1000.0,
        last_updated=time.time()
    )

    signals = analyzer._process_lunarcrush_data(sentiment_data, "SOL")
    assert len(signals) == 1

    signal = signals[0]
    assert signal.platform == "lunarcrush"
    assert signal.token_mentions == ["SOL"]
    assert signal.sentiment_score == pytest.approx((sentiment_data.sentiment * 2) - 1)
    assert signal.influence_score == pytest.approx(min(sentiment_data.galaxy_score / 100.0, 1.0))
    expected_engagement = min((sentiment_data.social_volume + sentiment_data.social_mentions) / 2000.0, 1.0)
    assert signal.engagement_score == pytest.approx(expected_engagement)
    assert f"galaxy_score={sentiment_data.galaxy_score:.2f}" in signal.content
    assert f"mentions:{sentiment_data.social_mentions}" in signal.keywords
    assert f"volume:{int(sentiment_data.social_volume)}" in signal.keywords
