"""
API Gateway Service for LegacyCoinTrader Microservices Architecture

This service acts as the entry point for all external requests, routing them
to the appropriate microservices with authentication, rate limiting, and
request/response transformation.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import aiohttp
from aiohttp import web
import redis.asyncio as redis
import yaml

from .middleware import (
    auth_middleware,
    rate_limit_middleware,
    logging_middleware
)
from .routes import setup_routes
from .service_discovery import ServiceDiscovery
from .config import GatewayConfig

logger = logging.getLogger(__name__)


class APIGateway:
    """API Gateway for routing requests to microservices."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), 'config.yaml'
        )
        self.config = self._load_config()
        self.app = web.Application(middlewares=[
            auth_middleware,
            rate_limit_middleware,
            logging_middleware
        ])
        self.redis_client = None
        self.service_discovery = None
        self.http_client = None

    def _load_config(self) -> GatewayConfig:
        """Load gateway configuration."""
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return GatewayConfig(**config_data.get('api_gateway', {}))

    async def _init_redis(self) -> None:
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def _init_service_discovery(self) -> None:
        """Initialize service discovery."""
        self.service_discovery = ServiceDiscovery(
            redis_client=self.redis_client,
            config=self.config
        )
        await self.service_discovery.start()
        logger.info("Service discovery initialized")

    async def _init_http_client(self) -> None:
        """Initialize HTTP client for service communication."""
        timeout = aiohttp.ClientTimeout(total=30)
        self.http_client = aiohttp.ClientSession(timeout=timeout)

    async def setup(self) -> None:
        """Setup the API gateway."""
        # Initialize connections
        await self._init_redis()
        await self._init_service_discovery()
        await self._init_http_client()

        # Setup routes
        setup_routes(self.app, self)

        logger.info(f"API Gateway configured on port {self.config.port}")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.http_client:
            await self.http_client.close()
        if self.service_discovery:
            await self.service_discovery.stop()
        if self.redis_client:
            self.redis_client.close()

    async def proxy_request(
        self,
        service_name: str,
        path: str,
        method: str = 'GET',
        data: Dict = None,
        headers: Dict = None
    ) -> Dict[str, Any]:
        """
        Proxy request to a microservice.

        Args:
            service_name: Name of the target service
            path: API path to proxy
            method: HTTP method
            data: Request data
            headers: Request headers

        Returns:
            Response from the microservice
        """
        try:
            # Get service endpoint
            service_url = await self.service_discovery.get_service_url(
                service_name
            )
            if not service_url:
                raise web.HTTPServiceUnavailable(
                    reason=f"Service {service_name} not available"
                )

            # Build full URL
            url = f"{service_url}{path}"

            # Prepare request
            request_kwargs = {
                'method': method,
                'url': url,
                'headers': headers or {}
            }

            # Add authentication header for service-to-service communication
            request_kwargs['headers']['X-Service-Auth'] = (
                self.config.service_auth_token
            )

            if data:
                if isinstance(data, dict):
                    request_kwargs['json'] = data
                    request_kwargs['headers']['Content-Type'] = (
                        'application/json'
                    )
                else:
                    request_kwargs['data'] = data

            # Make request
            async with self.http_client.request(
                **request_kwargs
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(
                        f"Service error {response.status}: {error_text}"
                    )
                    raise web.HTTPBadGateway(
                        reason=f"Service error: {error_text}"
                    )

                # Return response
                if response.headers.get('Content-Type', '').startswith(
                    'application/json'
                ):
                    return await response.json()
                else:
                    return {'data': await response.text()}

        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {e}")
            raise web.HTTPBadGateway(reason=f"Service communication error: {str(e)}")
        except Exception as e:
            logger.error(f"Proxy request error: {e}")
            raise web.HTTPInternalServerError(reason=f"Gateway error: {str(e)}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of all services."""
        health_status = {
            'gateway': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'services': {}
        }

        services = ['trading_engine', 'market_data', 'portfolio',
                   'strategy_engine', 'token_discovery', 'execution', 'monitoring']

        for service in services:
            try:
                # Check if service is registered
                service_url = await self.service_discovery.get_service_url(service)
                if service_url:
                    # Try to ping the service
                    await self.proxy_request(service, '/health', method='GET')
                    health_status['services'][service] = 'healthy'
                else:
                    health_status['services'][service] = 'unregistered'
            except Exception as e:
                health_status['services'][service] = f'unhealthy: {str(e)}'

        return health_status


async def create_app(config_path: Optional[str] = None) -> web.Application:
    """Create and configure the API gateway application."""
    gateway = APIGateway(config_path)
    await gateway.setup()
    return gateway.app


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run app
    app = create_app()
    web.run_app(app, port=8000, access_log=None)
