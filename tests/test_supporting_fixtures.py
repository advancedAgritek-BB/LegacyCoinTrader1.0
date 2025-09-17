"""Sanity tests for reusable fixtures introduced for integration support."""

import pytest

from services.api_gateway.rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_exchange_client_order_lifecycle(exchange_client):
    order = await exchange_client.create_order(
        symbol="BTC/USDT",
        order_type="limit",
        side="buy",
        amount=0.5,
        price=100.0,
    )
    fetched = await exchange_client.fetch_order(order["id"])
    assert fetched["status"] == "open"

    cancelled = await exchange_client.cancel_order(order["id"])
    assert cancelled is True
    updated = await exchange_client.fetch_order(order["id"])
    assert updated["status"] == "canceled"


@pytest.mark.asyncio
async def test_rate_limiter_uses_inmemory_redis(redis_client):
    limiter = RateLimiter(redis_client, default_limit=2, window_seconds=60)

    first = await limiter.check("client")
    second = await limiter.check("client")
    third = await limiter.check("client")

    assert first.allowed is True
    assert second.allowed is True
    assert third.allowed is False
    assert third.remaining == 0
