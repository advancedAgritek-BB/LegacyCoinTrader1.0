"""
Trading Engine Service for LegacyCoinTrader Microservices Architecture

This service handles the core trading logic and orchestration, managing:
- Trading cycle execution
- Symbol batch processing
- Strategy evaluation and routing
- Position management
- Risk assessment
- Performance monitoring
"""

import asyncio
import json
import logging
import os
import signal
from datetime import datetime
from typing import Dict, Any, Optional
import aiohttp
from aiohttp import web
import redis.asyncio as redis
import yaml

from .trading_orchestrator import TradingOrchestrator
from .config import TradingEngineConfig
from .health import HealthChecker

logger = logging.getLogger(__name__)


class TradingEngineService:
    """Trading Engine service for orchestrating trading operations."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), 'config.yaml'
        )
        self.config = self._load_config()
        self.orchestrator = None
        self.redis_client = None
        self.http_client = None
        self.health_checker = None
        self.running = False

        # Service discovery
        self.service_urls: Dict[str, str] = {}

    def _load_config(self) -> TradingEngineConfig:
        """Load trading engine configuration."""
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return TradingEngineConfig(**config_data.get('trading_engine', {}))

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

    async def _init_http_client(self) -> None:
        """Initialize HTTP client for service communication."""
        timeout = aiohttp.ClientTimeout(total=30)
        self.http_client = aiohttp.ClientSession(timeout=timeout)

    async def _init_service_discovery(self) -> None:
        """Initialize service discovery."""
        # Register with API Gateway
        await self._register_with_gateway()

        # Discover other services
        await self._discover_services()

    async def _register_with_gateway(self) -> None:
        """Register this service with the API Gateway."""
        try:
            gateway_url = (
                f"http://{self.config.api_gateway_host}:"
                f"{self.config.api_gateway_port}"
            )
            service_data = {
                'name': 'trading_engine',
                'url': f"http://{self.config.host}:{self.config.port}",
                'status': 'healthy',
                'last_seen': datetime.utcnow().isoformat()
            }

            async with self.http_client.post(
                f"{gateway_url}/register",
                json=service_data,
                headers={'X-Service-Auth': self.config.service_auth_token}
            ) as response:
                if response.status == 200:
                    logger.info("Successfully registered with API Gateway")
                else:
                    logger.warning(
                        f"Failed to register with API Gateway: {response.status}"
                    )

        except Exception as e:
            logger.error(f"Failed to register with API Gateway: {e}")

    async def _discover_services(self) -> None:
        """Discover other microservices."""
        services = [
            'market_data', 'portfolio', 'strategy_engine',
            'execution', 'monitoring'
        ]

        for service in services:
            try:
                # Try to get service URL from Redis
                service_key = f"service:{service}"
                service_data = self.redis_client.get(service_key)

                if service_data:
                    service_info = json.loads(service_data)
                    self.service_urls[service] = service_info['url']
                    logger.info(
                        f"Discovered {service} at {service_info['url']}"
                    )
                else:
                    logger.warning(f"Service {service} not found in discovery")

            except Exception as e:
                logger.error(f"Failed to discover service {service}: {e}")

    async def setup(self) -> None:
        """Setup the trading engine service."""
        # Initialize connections
        await self._init_redis()
        await self._init_http_client()
        await self._init_service_discovery()

        # Initialize trading orchestrator
        self.orchestrator = TradingOrchestrator(
            config=self.config,
            redis_client=self.redis_client,
            service_urls=self.service_urls
        )

        # Initialize health checker
        self.health_checker = HealthChecker(self.config, self.orchestrator)

        logger.info(f"Trading Engine configured on port {self.config.port}")

    async def start_trading(self) -> Dict[str, Any]:
        """Start the trading engine."""
        try:
            if self.orchestrator:
                await self.orchestrator.start()
                self.running = True
                return {
                    'status': 'started',
                    'message': 'Trading engine started successfully'
                }
            else:
                return {'status': 'error', 'message': 'Trading orchestrator not initialized'}
        except Exception as e:
            logger.error(f"Failed to start trading engine: {e}")
            return {'status': 'error', 'message': str(e)}

    async def stop_trading(self) -> Dict[str, Any]:
        """Stop the trading engine."""
        try:
            if self.orchestrator:
                await self.orchestrator.stop()
                self.running = False
                return {
                    'status': 'stopped',
                    'message': 'Trading engine stopped successfully'
                }
            else:
                return {'status': 'error', 'message': 'Trading orchestrator not initialized'}
        except Exception as e:
            logger.error(f"Failed to stop trading engine: {e}")
            return {'status': 'error', 'message': str(e)}

    async def get_status(self) -> Dict[str, Any]:
        """Get trading engine status."""
        try:
            if self.orchestrator:
                status = await self.orchestrator.get_status()
                return {
                    'status': 'running' if self.running else 'stopped',
                    'orchestrator': status,
                    'health': await self.health_checker.check_health(),
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                return {'status': 'error', 'message': 'Trading orchestrator not initialized'}
        except Exception as e:
            logger.error(f"Failed to get trading status: {e}")
            return {'status': 'error', 'message': str(e)}

    async def execute_cycle(self) -> Dict[str, Any]:
        """Execute a single trading cycle."""
        try:
            if self.orchestrator:
                result = await self.orchestrator.execute_trading_cycle()
                return {'status': 'success', 'result': result}
            else:
                return {'status': 'error', 'message': 'Trading orchestrator not initialized'}
        except Exception as e:
            logger.error(f"Failed to execute trading cycle: {e}")
            return {'status': 'error', 'message': str(e)}

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.running = False

        if self.orchestrator:
            await self.orchestrator.cleanup()

        if self.http_client:
            await self.http_client.close()

        if self.redis_client:
            self.redis_client.close()

        logger.info("Trading Engine service cleaned up")


async def create_app(config_path: str = None) -> web.Application:
    """Create and configure the trading engine application."""
    app = web.Application()
    service = TradingEngineService(config_path)
    await service.setup()

    # Store service instance
    app['trading_engine'] = service

    # Add routes
    app.router.add_post('/start', start_trading_handler)
    app.router.add_post('/stop', stop_trading_handler)
    app.router.add_get('/status', get_status_handler)
    app.router.add_post('/cycle', execute_cycle_handler)
    app.router.add_get('/health', health_check_handler)

    return app


# Route handlers
async def start_trading_handler(request: web.Request) -> web.Response:
    """Start trading engine."""
    service = request.app['trading_engine']
    result = await service.start_trading()
    status_code = 200 if result['status'] == 'started' else 500
    return web.json_response(result, status=status_code)


async def stop_trading_handler(request: web.Request) -> web.Response:
    """Stop trading engine."""
    service = request.app['trading_engine']
    result = await service.stop_trading()
    return web.json_response(result)


async def get_status_handler(request: web.Request) -> web.Response:
    """Get trading engine status."""
    service = request.app['trading_engine']
    result = await service.get_status()
    return web.json_response(result)


async def execute_cycle_handler(request: web.Request) -> web.Response:
    """Execute trading cycle."""
    service = request.app['trading_engine']
    result = await service.execute_cycle()
    status_code = 200 if result['status'] == 'success' else 500
    return web.json_response(result, status=status_code)


async def health_check_handler(request: web.Request) -> web.Response:
    """Health check endpoint."""
    service = request.app['trading_engine']
    try:
        health_status = await service.health_checker.check_health()
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return web.json_response(health_status, status=status_code)
    except Exception as e:
        return web.json_response(
            {'status': 'error', 'message': str(e)},
            status=500
        )


async def main() -> None:
    """Main entry point for the trading engine service."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run app
    app = await create_app()

    # Setup graceful shutdown
    async def shutdown_handler() -> None:
        service = app['trading_engine']
        await service.cleanup()

    # Handle shutdown signals
    def signal_handler(signum: int, frame) -> None:
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(shutdown_handler())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        web.run_app(app, port=8001, access_log=None)
    except KeyboardInterrupt:
        logger.info("Trading Engine service stopped by user")
    finally:
        await shutdown_handler()


if __name__ == '__main__':
    asyncio.run(main())
