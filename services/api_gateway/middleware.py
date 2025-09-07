"""
Middleware for API Gateway service.
"""

import json
import logging
import time
from typing import Dict, Any, Callable
from aiohttp import web
import redis

logger = logging.getLogger(__name__)


@web.middleware
async def auth_middleware(request: web.Request, handler: Callable) -> web.Response:
    """
    Authentication middleware for API requests.

    This is a basic implementation - in production, you'd want more sophisticated
    authentication (JWT, API keys, OAuth, etc.)
    """
    # Skip auth for health checks and public endpoints
    public_paths = ['/health', '/docs', '/openapi.json']
    if any(request.path.startswith(path) for path in public_paths):
        return await handler(request)

    # Check for service-to-service authentication
    service_auth = request.headers.get('X-Service-Auth')
    if service_auth:
        # Verify service auth token
        expected_token = request.app['gateway'].config.service_auth_token
        if service_auth != expected_token:
            return web.HTTPUnauthorized(reason="Invalid service authentication")
        return await handler(request)

    # For external requests, you might want API key authentication
    api_key = request.headers.get('X-API-Key') or request.query.get('api_key')
    if not api_key:
        return web.HTTPUnauthorized(reason="API key required")

    # Verify API key (in production, check against database/redis)
    if not _verify_api_key(api_key, request.app):
        return web.HTTPUnauthorized(reason="Invalid API key")

    return await handler(request)


@web.middleware
async def rate_limit_middleware(request: web.Request, handler: Callable) -> web.Response:
    """
    Rate limiting middleware using Redis.

    This implements a simple sliding window rate limiter.
    """
    # Skip rate limiting for health checks
    if request.path.startswith('/health'):
        return await handler(request)

    client_ip = _get_client_ip(request)
    gateway = request.app['gateway']

    # Check rate limit
    is_allowed = await _check_rate_limit(
        client_ip, gateway.redis_client, gateway.config
    )

    if not is_allowed:
        return web.HTTPTooManyRequests(
            reason="Rate limit exceeded",
            headers={'Retry-After': str(gateway.config.rate_limit_window)}
        )

    return await handler(request)


@web.middleware
async def logging_middleware(request: web.Request, handler: Callable) -> web.Response:
    """
    Logging middleware for request/response tracking.
    """
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.path} from {request.remote}")

    try:
        # Process request
        response = await handler(request)

        # Log response
        duration = time.time() - start_time
        logger.info(
            f"Response: {response.status} for {request.method} {request.path} "
            f"in {duration:.3f}s"
        )

        return response

    except Exception as e:
        # Log error
        duration = time.time() - start_time
        logger.error(
            f"Error: {str(e)} for {request.method} {request.path} "
            f"in {duration:.3f}s"
        )
        raise


def _get_client_ip(request: web.Request) -> str:
    """Get client IP address from request."""
    # Check for forwarded headers first
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()

    # Fall back to remote address
    return request.remote or 'unknown'


def _verify_api_key(api_key: str, app: web.Application) -> bool:
    """
    Verify API key.

    In production, this should check against a database or Redis.
    For now, we'll accept any non-empty key.
    """
    return bool(api_key and len(api_key) > 10)


async def _check_rate_limit(client_ip: str, redis_client: redis.Redis,
                           config) -> bool:
    """
    Check if request is within rate limits.

    Uses Redis sorted sets for sliding window rate limiting.
    """
    try:
        current_time = time.time()
        window_start = current_time - config.rate_limit_window

        # Use Redis sorted set to track requests
        key = f"ratelimit:{client_ip}"

        # Remove old requests outside the window
        redis_client.zremrangebyscore(key, '-inf', window_start)

        # Count requests in current window
        request_count = redis_client.zcard(key)

        if request_count >= config.rate_limit_requests:
            return False

        # Add current request
        redis_client.zadd(key, {str(current_time): current_time})

        # Set expiry on the key
        redis_client.expire(key, config.rate_limit_window)

        return True

    except Exception as e:
        logger.error(f"Rate limit check error: {e}")
        # Allow request on error to avoid blocking legitimate traffic
        return True
