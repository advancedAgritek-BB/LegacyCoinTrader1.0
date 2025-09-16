"""Tests for Kraken WebSocket ticker message handling."""

import json

import pytest

from crypto_bot.execution.kraken_ws import KrakenWSClient


def test_handle_ticker_message_normalizes_symbols_and_updates_cache():
    """Ticker updates should populate the price cache using CCXT symbols."""

    client = KrakenWSClient()

    # Use a deterministic object to simulate the public websocket connection.
    public_ws = object()
    client.public_ws = public_ws

    btc_message = json.dumps(
        {
            "channel": "ticker",
            "type": "update",
            "data": [
                {
                    "symbol": "XBT/USD",
                    "last": "50000.5",
                    "bid": "49999.1",
                    "ask": "50001.2",
                    "volume": "12.34",
                }
            ],
        }
    )

    client._handle_message(public_ws, btc_message)

    assert "BTC/USD" in client.price_cache
    assert client.price_cache["BTC/USD"]["last"] == pytest.approx(50000.5)
    assert client.get_current_price("BTC/USD") == pytest.approx(50000.5)
    # Normalization should also make the Kraken style lookup succeed.
    assert client.get_current_price("XBT/USD") == pytest.approx(50000.5)

    eth_message = json.dumps(
        {
            "channel": "ticker",
            "type": "update",
            "data": [
                {
                    "symbol": "ETH/USD",
                    "last": "2500.1",
                    "bid": "2499.9",
                    "ask": "2500.3",
                    "volume": "321.0",
                }
            ],
        }
    )

    client._handle_message(public_ws, eth_message)

    assert "ETH/USD" in client.price_cache
    assert "ETH/USD/USD" not in client.price_cache
    assert client.price_cache["ETH/USD"]["last"] == pytest.approx(2500.1)
    assert client.get_current_price("ETH/USD") == pytest.approx(2500.1)
