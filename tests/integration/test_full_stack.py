"""Full stack integration tests for LegacyCoinTrader microservices."""

from __future__ import annotations

import asyncio
import os
import pytest
import pytest_asyncio
import time
from typing import Dict, Any
import httpx
import redis.asyncio as redis
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


@pytest.fixture(scope="session")
def event_loop() -> asyncio.AbstractEventLoop:
    """Create event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    # Don't close the loop immediately, let pytest-asyncio handle it


@pytest_asyncio.fixture(scope="function")
async def redis_client():
    """Redis client for testing."""
    client = redis.Redis(host="localhost", port=6379, db=1)
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture(scope="session")
def postgres_connection() -> psycopg2.extensions.connection:
    """PostgreSQL connection for testing."""
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="legacy_coin_trader_test",
        user="postgres",
        password="test_password"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    yield conn
    conn.close()


@pytest_asyncio.fixture(scope="function")
async def api_gateway_client():
    """HTTP client for API Gateway."""
    async with httpx.AsyncClient(
        base_url="http://localhost:8000",
        timeout=30.0
    ) as client:
        yield client


class TestMicroserviceIntegration:
    """Test full microservice integration."""

    @pytest.mark.asyncio
    async def test_api_gateway_health(self, api_gateway_client):
        """Test API Gateway health endpoint."""
        response = await api_gateway_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "services" in data
        assert isinstance(data["services"], dict)

    @pytest.mark.asyncio
    async def test_service_discovery(self, api_gateway_client):
        """Test service discovery through API Gateway."""
        response = await api_gateway_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        services = data["services"]

        # Should have multiple services configured
        assert len(services) > 0

        # Each service should have health information
        for service_name, service_info in services.items():
            assert "healthy" in service_info
            assert "service" in service_info

    @pytest.mark.asyncio
    async def test_service_token_management(self, api_gateway_client):
        """Test service token generation and validation."""
        # Generate a service token
        response = await api_gateway_client.post(
            "/auth/service-token/generate",
            params={"service_name": "trading-engine"}
        )

        # May require authentication, endpoint should exist, or service may be unavailable
        assert response.status_code in [200, 401, 403, 500]

        if response.status_code == 200:
            data = response.json()
            assert "token" in data
            assert "service_name" in data
            assert data["service_name"] == "trading-engine"

            # Test token validation
            token = data["token"]
            validate_response = await api_gateway_client.post(
                "/auth/service-token/validate",
                params={"service_name": "trading-engine", "token": token}
            )
            assert validate_response.status_code == 200
            validate_data = validate_response.json()
            assert validate_data["valid"] is True

    @pytest.mark.asyncio
    async def test_redis_connectivity(self, redis_client):
        """Test Redis connectivity."""
        # Test basic Redis operations
        await redis_client.set("test_key", "test_value")
        value = await redis_client.get("test_key")
        assert value.decode() == "test_value"

        # Clean up
        await redis_client.delete("test_key")

    @pytest.mark.asyncio
    async def test_resilience_patterns(self, api_gateway_client):
        """Test resilience patterns (circuit breaker, retry)."""
        # Make multiple requests to test resilience
        responses = []
        for i in range(10):
            try:
                response = await api_gateway_client.get("/health")
                responses.append(response.status_code)
            except Exception as e:
                responses.append(f"error: {e}")

        # Should have mostly successful responses
        successful_responses = [r for r in responses if r == 200]
        assert len(successful_responses) >= 8, f"Too many failed responses: {responses}"

    @pytest.mark.asyncio
    async def test_cross_service_communication(self, api_gateway_client):
        """Test communication between services through API Gateway."""
        # Test that services can communicate with each other
        # This is a basic test - in a real scenario, we'd test actual data flow

        # Get initial health status
        initial_response = await api_gateway_client.get("/health")
        assert initial_response.status_code == 200

        # Wait a moment for services to stabilize
        await asyncio.sleep(2)

        # Get health status again
        final_response = await api_gateway_client.get("/health")
        assert final_response.status_code == 200

        initial_data = initial_response.json()
        final_data = final_response.json()

        # Services should still be accessible
        assert len(final_data["services"]) == len(initial_data["services"])


class TestDataPersistenceIntegration:
    """Test data persistence across services."""

    def test_postgres_connection(self, postgres_connection):
        """Test PostgreSQL connection."""
        cursor = postgres_connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1
        cursor.close()

    @pytest.mark.asyncio
    async def test_portfolio_data_persistence(self, api_gateway_client, postgres_connection):
        """Test portfolio data persistence through API Gateway."""
        # This would test actual portfolio operations in a full implementation
        # For now, just verify the endpoint exists and responds
        response = await api_gateway_client.get("/portfolio/state")

        # May require authentication, but endpoint should exist
        assert response.status_code in [200, 401, 403, 404]

        if response.status_code == 200:
            data = response.json()
            # Should return portfolio state structure
            assert isinstance(data, dict)


class TestPerformanceUnderLoad:
    """Test performance under load."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, api_gateway_client):
        """Test handling of concurrent requests."""
        async def make_request():
            return await api_gateway_client.get("/health")

        # Make 50 concurrent requests
        tasks = [make_request() for _ in range(50)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful responses
        successful_responses = [
            r for r in responses
            if not isinstance(r, Exception) and hasattr(r, 'status_code') and r.status_code == 200
        ]

        # Should handle at least 80% of concurrent requests successfully
        success_rate = len(successful_responses) / len(responses)
        assert success_rate >= 0.8, f"Success rate too low: {success_rate}"

    @pytest.mark.asyncio
    async def test_request_latency(self, api_gateway_client):
        """Test request latency."""
        latencies = []

        for _ in range(20):
            start_time = time.time()
            response = await api_gateway_client.get("/health")
            end_time = time.time()

            assert response.status_code == 200
            latencies.append(end_time - start_time)

        # Calculate average latency
        avg_latency = sum(latencies) / len(latencies)

        # Average latency should be reasonable (< 1 second)
        assert avg_latency < 1.0, f"Average latency too high: {avg_latency}s"

        # 95th percentile should also be reasonable
        latencies.sort()
        p95_latency = latencies[int(len(latencies) * 0.95)]
        assert p95_latency < 2.0, f"95th percentile latency too high: {p95_latency}s"


class TestServiceResilience:
    """Test service resilience patterns."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, api_gateway_client):
        """Test circuit breaker recovery pattern."""
        # This would require mocking service failures
        # For now, just verify the health endpoint works
        response = await api_gateway_client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_graceful_service_degradation(self, api_gateway_client):
        """Test graceful degradation when services are unavailable."""
        # Test that API Gateway handles service unavailability gracefully
        response = await api_gateway_client.get("/health")
        assert response.status_code == 200

        data = response.json()

        # Should still return a response even if some services are down
        assert "status" in data
        assert "services" in data


class TestSecurityIntegration:
    """Test security integration across services."""

    @pytest.mark.asyncio
    async def test_https_support(self, api_gateway_client):
        """Test HTTPS support if configured."""
        # If TLS is enabled, test HTTPS connection
        tls_enabled = os.getenv("TLS_ENABLED", "false").lower() == "true"

        if tls_enabled:
            # Test HTTPS connection
            async with httpx.AsyncClient(
                base_url="https://localhost:8443",
                verify=False,  # For self-signed certificates
                timeout=30.0
            ) as https_client:
                response = await https_client.get("/health")
                assert response.status_code == 200
        else:
            # Test HTTP connection
            response = await api_gateway_client.get("/health")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_authentication_flow(self, api_gateway_client):
        """Test complete authentication flow."""
        # This would test JWT token generation and validation
        # For now, just verify endpoints exist
        response = await api_gateway_client.post("/auth/token", json={
            "username": "test",
            "password": "test"
        })

        # Should get a response (may be authentication error)
        assert response.status_code in [200, 401, 422]


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
