# API Gateway Service

The API Gateway service acts as the single entry point for all external requests to the LegacyCoinTrader microservices architecture.

## Features

- **Request Routing**: Routes requests to appropriate microservices
- **Authentication**: API key and service-to-service authentication
- **Rate Limiting**: Prevents abuse with configurable rate limits
- **Service Discovery**: Automatically discovers and routes to healthy services
- **Health Monitoring**: Provides health checks for all services
- **Logging**: Comprehensive request/response logging

## API Endpoints

### Health Check
- `GET /health` - Overall system health status

### Trading Engine
- `POST /api/trading/start` - Start trading engine
- `POST /api/trading/stop` - Stop trading engine
- `GET /api/trading/status` - Get trading status
- `POST /api/trading/cycle` - Trigger manual trading cycle

### Market Data
- `GET /api/market/ohlcv?symbol=BTC/USD&timeframe=1h&limit=100` - Get OHLCV data
- `GET /api/market/ticker?symbol=BTC/USD` - Get ticker data
- `GET /api/market/orderbook?symbol=BTC/USD&depth=10` - Get order book
- `GET /api/market/cache/stats` - Get cache statistics

### Portfolio
- `GET /api/portfolio/positions` - Get current positions
- `GET /api/portfolio/balance` - Get account balance
- `GET /api/portfolio/pnl` - Get P&L information
- `POST /api/portfolio/sync` - Sync portfolio data

### Strategy Engine
- `POST /api/strategy/evaluate` - Evaluate strategies
- `GET /api/strategy/regime` - Get current market regime
- `GET /api/strategy/signals?symbol=BTC/USD` - Get strategy signals

### Token Discovery
- `GET /api/tokens/discover?limit=20` - Discover new tokens
- `GET /api/tokens/pools` - Get DEX pool information
- `GET /api/tokens/scan/status` - Get scanning status

### Execution
- `POST /api/execution/order` - Place new order
- `GET /api/execution/orders` - Get order status
- `DELETE /api/execution/order/{order_id}` - Cancel order

### Monitoring
- `GET /api/monitoring/health` - Detailed health status
- `GET /api/monitoring/metrics` - System metrics
- `GET /api/monitoring/logs?level=INFO&limit=100` - System logs

## Configuration

The service is configured via `config.yaml`:

```yaml
api_gateway:
  port: 8000
  redis_host: "localhost"
  service_auth_token: "your-service-token"
  rate_limit_requests: 1000
  rate_limit_window: 60
```

## Running the Service

### Local Development
```bash
pip install -r requirements.txt
python app.py
```

### Docker
```bash
docker build -t api-gateway .
docker run -p 8000:8000 api-gateway
```

### Docker Compose (with Redis)
```yaml
version: '3.8'
services:
  api-gateway:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - redis
    environment:
      - REDIS_HOST=redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

## Authentication

### API Key Authentication
External clients must provide an API key:
```
X-API-Key: your-api-key-here
```

### Service-to-Service Authentication
Microservices authenticate using the service auth token:
```
X-Service-Auth: legacy-coin-trader-service-token-2024
```

## Service Discovery

The gateway uses Redis-based service discovery:

1. Services register themselves with their URLs
2. Gateway discovers available services automatically
3. Requests are routed to healthy service instances
4. Failed services are automatically removed from routing

## Monitoring

- Health checks every 30 seconds
- Request/response logging
- Rate limiting metrics
- Service availability monitoring

## Development

### Adding New Routes
1. Add route handler in `routes.py`
2. Implement business logic
3. Update service discovery if needed

### Testing
```bash
# Run health check
curl http://localhost:8000/health

# Test trading endpoints
curl -X POST http://localhost:8000/api/trading/status
```
