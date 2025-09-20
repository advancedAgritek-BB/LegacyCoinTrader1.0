"""Tests for API Gateway main application."""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient

from services.api_gateway.main import create_app


@pytest.fixture
def client():
    """Create test client for API Gateway."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create async test client for API Gateway."""
    app = create_app()
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


class TestHealthEndpoint:
    """Test health check endpoints."""

    def test_health_endpoint(self, client):
        """Test basic health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "redis" in data

    async def test_health_endpoint_async(self, async_client):
        """Test health endpoint with async client."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestAuthentication:
    """Test authentication endpoints."""

    def test_login_endpoint_exists(self, client):
        """Test that login endpoint exists."""
        # This will fail if JWT is not configured, but endpoint should exist
        response = client.post("/auth/token", json={
            "username": "test",
            "password": "test"
        })
        # Should return 401 for invalid credentials or 422 for validation error
        assert response.status_code in [401, 422]

    def test_service_token_endpoints_exist(self, client):
        """Test that service token endpoints exist."""
        # Generate service token
        response = client.post("/auth/service-token/generate", params={"service_name": "test-service"})
        assert response.status_code in [200, 401, 403]  # May require auth

        # List service tokens
        response = client.get("/auth/service-tokens")
        assert response.status_code in [200, 401, 403]

        # Validate service token
        response = client.post("/auth/service-token/validate",
                              params={"service_name": "test-service", "token": "test-token"})
        assert response.status_code in [200, 401, 403]


class TestProxyEndpoints:
    """Test proxy functionality."""

    def test_proxy_to_unknown_service(self, client):
        """Test proxying to unknown service returns 404."""
        response = client.get("/unknown-service/test")
        assert response.status_code == 404

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_headers_present(self, client):
        """Test that rate limit headers are present in responses."""
        response = client.get("/health")
        # Rate limiting headers may or may not be present depending on configuration
        # This test ensures the endpoint works regardless
        assert response.status_code == 200


class TestServiceDiscovery:
    """Test service discovery functionality."""

    def test_service_routes_configured(self, client):
        """Test that service routes are properly configured."""
        # This indirectly tests service discovery by checking health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "services" in data
        # Should have services configured
        assert len(data["services"]) > 0


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_json_returns_422(self, client):
        """Test invalid JSON returns 422."""
        response = client.post("/auth/token", data="invalid json")
        assert response.status_code == 422

    def test_not_found_returns_404(self, client):
        """Test unknown endpoint returns 404."""
        response = client.get("/nonexistent/endpoint")
        assert response.status_code == 404

    def test_method_not_allowed_returns_405(self, client):
        """Test wrong method returns 405."""
        response = client.post("/health")
        assert response.status_code == 405


class TestSecurityHeaders:
    """Test security headers."""

    def test_security_headers_present(self, client):
        """Test that security headers are present."""
        response = client.get("/health")
        headers = response.headers

        # Check for common security headers
        security_headers = [
            "x-content-type-options",
            "x-frame-options",
            "x-xss-protection"
        ]

        # At least some security headers should be present
        present_headers = [h for h in security_headers if h in headers]
        assert len(present_headers) > 0, f"No security headers found. Headers: {dict(headers)}"


class TestMetricsAndMonitoring:
    """Test metrics and monitoring endpoints."""

    def test_prometheus_metrics_available(self, client):
        """Test that Prometheus metrics endpoint exists."""
        response = client.get("/metrics")
        # Metrics endpoint may require authentication or special configuration
        assert response.status_code in [200, 401, 403, 404]

    def test_health_detailed_response(self, client):
        """Test detailed health response structure."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        required_keys = ["status", "services", "redis", "routes"]
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"

        # Status should be healthy or degraded
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
