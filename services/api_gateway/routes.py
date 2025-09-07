"""
Route definitions for API Gateway service.
"""

import json
import logging
from typing import Dict, Any
from aiohttp import web

logger = logging.getLogger(__name__)


def setup_routes(app: web.Application, gateway) -> None:
    """Setup all routes for the API gateway."""

    # Store gateway instance in app
    app['gateway'] = gateway

    # Health check endpoint
    app.router.add_get('/health', health_check_handler)

    # Service registration endpoints
    app.router.add_post('/register', register_service_handler)
    app.router.add_post('/unregister', unregister_service_handler)

    # Trading Engine routes
    app.router.add_post('/api/trading/start', trading_start_handler)
    app.router.add_post('/api/trading/stop', trading_stop_handler)
    app.router.add_get('/api/trading/status', trading_status_handler)
    app.router.add_post('/api/trading/cycle', trading_cycle_handler)

    # Market Data routes
    app.router.add_get('/api/market/ohlcv', market_ohlcv_handler)
    app.router.add_get('/api/market/ticker', market_ticker_handler)
    app.router.add_get('/api/market/orderbook', market_orderbook_handler)
    app.router.add_get('/api/market/cache/stats', market_cache_stats_handler)

    # Portfolio routes
    app.router.add_get('/api/portfolio/positions', portfolio_positions_handler)
    app.router.add_get('/api/portfolio/balance', portfolio_balance_handler)
    app.router.add_get('/api/portfolio/pnl', portfolio_pnl_handler)
    app.router.add_post('/api/portfolio/sync', portfolio_sync_handler)

    # Strategy Engine routes
    app.router.add_post('/api/strategy/evaluate', strategy_evaluate_handler)
    app.router.add_get('/api/strategy/regime', strategy_regime_handler)
    app.router.add_get('/api/strategy/signals', strategy_signals_handler)

    # Token Discovery routes
    app.router.add_get('/api/tokens/discover', tokens_discover_handler)
    app.router.add_get('/api/tokens/pools', tokens_pools_handler)
    app.router.add_get('/api/tokens/scan/status', tokens_scan_status_handler)

    # Execution routes
    app.router.add_post('/api/execution/order', execution_order_handler)
    app.router.add_get('/api/execution/orders', execution_orders_handler)
    app.router.add_delete('/api/execution/order/{order_id}', execution_cancel_handler)

    # Monitoring routes
    app.router.add_get('/api/monitoring/health', monitoring_health_handler)
    app.router.add_get('/api/monitoring/metrics', monitoring_metrics_handler)
    app.router.add_get('/api/monitoring/logs', monitoring_logs_handler)

    logger.info("API Gateway routes configured")


# Health Check Handler
async def health_check_handler(request: web.Request) -> web.Response:
    """Handle health check requests."""
    gateway = request.app['gateway']
    try:
        health_status = await gateway.health_check()
        status_code = 200 if all(
            status == 'healthy' for status in health_status['services'].values()
        ) else 503
        return web.json_response(health_status, status=status_code)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return web.json_response(
            {'status': 'error', 'message': str(e)},
            status=500
        )


# Registration Handlers
async def register_service_handler(request: web.Request) -> web.Response:
    """Register a service in service discovery."""
    gateway = request.app['gateway']
    try:
        data = await request.json()
        name = data.get('name')
        url = data.get('url')
        metadata = {k: v for k, v in data.items() if k not in {'name', 'url'}}
        if not name or not url:
            return web.json_response({'error': 'name and url are required'}, status=400)

        ok = await gateway.service_discovery.register_service(name, url, metadata)
        return web.json_response({'registered': ok})
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def unregister_service_handler(request: web.Request) -> web.Response:
    """Unregister a service from service discovery."""
    gateway = request.app['gateway']
    try:
        data = await request.json()
        name = data.get('name')
        if not name:
            return web.json_response({'error': 'name is required'}, status=400)

        ok = await gateway.service_discovery.unregister_service(name)
        return web.json_response({'unregistered': ok})
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


# Trading Engine Handlers
async def trading_start_handler(request: web.Request) -> web.Response:
    """Start trading engine."""
    gateway = request.app['gateway']
    try:
        result = await gateway.proxy_request(
            'trading_engine', '/start', method='POST'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def trading_stop_handler(request: web.Request) -> web.Response:
    """Stop trading engine."""
    gateway = request.app['gateway']
    try:
        result = await gateway.proxy_request(
            'trading_engine', '/stop', method='POST'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def trading_status_handler(request: web.Request) -> web.Response:
    """Get trading engine status."""
    gateway = request.app['gateway']
    try:
        result = await gateway.proxy_request(
            'trading_engine', '/status', method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def trading_cycle_handler(request: web.Request) -> web.Response:
    """Trigger manual trading cycle."""
    gateway = request.app['gateway']
    try:
        result = await gateway.proxy_request(
            'trading_engine', '/cycle', method='POST'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


# Market Data Handlers
async def market_ohlcv_handler(request: web.Request) -> web.Response:
    """Get OHLCV data."""
    gateway = request.app['gateway']
    try:
        symbol = request.query.get('symbol')
        timeframe = request.query.get('timeframe', '1h')
        limit = request.query.get('limit', '100')

        result = await gateway.proxy_request(
            'market_data', f'/ohlcv?symbol={symbol}&timeframe={timeframe}&limit={limit}',
            method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def market_ticker_handler(request: web.Request) -> web.Response:
    """Get market ticker data."""
    gateway = request.app['gateway']
    try:
        symbol = request.query.get('symbol')
        result = await gateway.proxy_request(
            'market_data', f'/ticker?symbol={symbol}', method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def market_orderbook_handler(request: web.Request) -> web.Response:
    """Get order book data."""
    gateway = request.app['gateway']
    try:
        symbol = request.query.get('symbol')
        depth = request.query.get('depth', '10')
        result = await gateway.proxy_request(
            'market_data', f'/orderbook?symbol={symbol}&depth={depth}', method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def market_cache_stats_handler(request: web.Request) -> web.Response:
    """Get market data cache statistics."""
    gateway = request.app['gateway']
    try:
        result = await gateway.proxy_request(
            'market_data', '/cache/stats', method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


# Portfolio Handlers
async def portfolio_positions_handler(request: web.Request) -> web.Response:
    """Get current positions."""
    gateway = request.app['gateway']
    try:
        result = await gateway.proxy_request(
            'portfolio', '/positions', method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def portfolio_balance_handler(request: web.Request) -> web.Response:
    """Get account balance."""
    gateway = request.app['gateway']
    try:
        result = await gateway.proxy_request(
            'portfolio', '/balance', method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def portfolio_pnl_handler(request: web.Request) -> web.Response:
    """Get P&L information."""
    gateway = request.app['gateway']
    try:
        result = await gateway.proxy_request(
            'portfolio', '/pnl', method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def portfolio_sync_handler(request: web.Request) -> web.Response:
    """Sync portfolio data."""
    gateway = request.app['gateway']
    try:
        result = await gateway.proxy_request(
            'portfolio', '/sync', method='POST'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


# Strategy Engine Handlers
async def strategy_evaluate_handler(request: web.Request) -> web.Response:
    """Evaluate strategies for symbols."""
    gateway = request.app['gateway']
    try:
        data = await request.json()
        result = await gateway.proxy_request(
            'strategy_engine', '/evaluate', method='POST', data=data
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def strategy_regime_handler(request: web.Request) -> web.Response:
    """Get current market regime."""
    gateway = request.app['gateway']
    try:
        result = await gateway.proxy_request(
            'strategy_engine', '/regime', method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def strategy_signals_handler(request: web.Request) -> web.Response:
    """Get strategy signals."""
    gateway = request.app['gateway']
    try:
        symbol = request.query.get('symbol')
        result = await gateway.proxy_request(
            'strategy_engine', f'/signals?symbol={symbol}', method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


# Token Discovery Handlers
async def tokens_discover_handler(request: web.Request) -> web.Response:
    """Discover new tokens."""
    gateway = request.app['gateway']
    try:
        limit = request.query.get('limit', '20')
        result = await gateway.proxy_request(
            'token_discovery', f'/discover?limit={limit}', method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def tokens_pools_handler(request: web.Request) -> web.Response:
    """Get DEX pool information."""
    gateway = request.app['gateway']
    try:
        result = await gateway.proxy_request(
            'token_discovery', '/pools', method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def tokens_scan_status_handler(request: web.Request) -> web.Response:
    """Get token scanning status."""
    gateway = request.app['gateway']
    try:
        result = await gateway.proxy_request(
            'token_discovery', '/scan/status', method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


# Execution Handlers
async def execution_order_handler(request: web.Request) -> web.Response:
    """Place a new order."""
    gateway = request.app['gateway']
    try:
        data = await request.json()
        result = await gateway.proxy_request(
            'execution', '/order', method='POST', data=data
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def execution_orders_handler(request: web.Request) -> web.Response:
    """Get order status."""
    gateway = request.app['gateway']
    try:
        result = await gateway.proxy_request(
            'execution', '/orders', method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def execution_cancel_handler(request: web.Request) -> web.Response:
    """Cancel an order."""
    gateway = request.app['gateway']
    try:
        order_id = request.match_info['order_id']
        result = await gateway.proxy_request(
            'execution', f'/order/{order_id}', method='DELETE'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


# Monitoring Handlers
async def monitoring_health_handler(request: web.Request) -> web.Response:
    """Get detailed health status."""
    gateway = request.app['gateway']
    try:
        result = await gateway.proxy_request(
            'monitoring', '/health', method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def monitoring_metrics_handler(request: web.Request) -> web.Response:
    """Get system metrics."""
    gateway = request.app['gateway']
    try:
        result = await gateway.proxy_request(
            'monitoring', '/metrics', method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)


async def monitoring_logs_handler(request: web.Request) -> web.Response:
    """Get system logs."""
    gateway = request.app['gateway']
    try:
        level = request.query.get('level', 'INFO')
        limit = request.query.get('limit', '100')
        result = await gateway.proxy_request(
            'monitoring', f'/logs?level={level}&limit={limit}', method='GET'
        )
        return web.json_response(result)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)
