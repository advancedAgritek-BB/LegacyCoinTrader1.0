# Trading Engine Service

The Trading Engine service is the core orchestration service that manages the entire trading pipeline, coordinating between all other microservices.

## Features

- **Trading Cycle Orchestration**: Manages the main trading loop and cycle execution
- **Symbol Batch Processing**: Processes trading symbols in configurable batches
- **Strategy Integration**: Coordinates with strategy engine for signal generation
- **Risk Management**: Integrates with portfolio service for risk assessment
- **Trade Execution**: Coordinates order execution through execution service
- **Performance Monitoring**: Tracks and reports trading performance metrics
- **Health Monitoring**: Comprehensive health checks and service monitoring

## API Endpoints

### Trading Control
- `POST /start` - Start the trading engine
- `POST /stop` - Stop the trading engine
- `GET /status` - Get trading engine status
- `POST /cycle` - Execute a single trading cycle

### Health Check
- `GET /health` - Service health status

## Architecture

The Trading Engine orchestrates the following trading pipeline:

1. **Symbol Selection**: Get filtered symbols from market data service
2. **Market Data**: Fetch OHLCV data for symbols
3. **Strategy Evaluation**: Evaluate strategies for each symbol
4. **Risk Assessment**: Check portfolio and risk limits
5. **Trade Generation**: Generate trades from signals
6. **Order Execution**: Execute trades through execution service
7. **Performance Tracking**: Update metrics and logging

## Configuration

The service is configured via `config.yaml`:

```yaml
trading_engine:
  port: 8001
  cycle_interval: 120  # 2 minutes
  batch_size: 25
  max_risk_per_trade: 0.05
  default_symbols:
    - "BTC/USD"
    - "ETH/USD"
```

## Trading Pipeline

### 1. Symbol Processing
- Load symbols from market data service
- Filter symbols based on liquidity and volume
- Process in configurable batches to manage load

### 2. Market Data Integration
- Fetch OHLCV data for current batch
- Update local caches
- Validate data quality

### 3. Strategy Evaluation
- Send market data to strategy engine
- Receive strategy signals
- Filter signals by confidence

### 4. Risk Management
- Check current portfolio status
- Validate risk limits
- Calculate position sizes

### 5. Trade Execution
- Generate trade orders
- Send to execution service
- Track execution results

### 6. Performance Monitoring
- Update cycle metrics
- Track success rates
- Log performance data

## Service Dependencies

- **Market Data Service**: OHLCV data and symbol filtering
- **Strategy Engine**: Strategy evaluation and signals
- **Portfolio Service**: Position and risk management
- **Execution Service**: Order placement and execution
- **Monitoring Service**: Metrics and health monitoring

## Running the Service

### Local Development
```bash
pip install -r requirements.txt
python app.py
```

### Docker
```bash
docker build -t trading-engine .
docker run -p 8001:8001 trading-engine
```

### Docker Compose (with dependencies)
```yaml
version: '3.8'
services:
  trading-engine:
    build: .
    ports:
      - "8001:8001"
    depends_on:
      - redis
      - market-data
      - strategy-engine
      - portfolio
      - execution
    environment:
      - REDIS_HOST=redis
```

## Monitoring

### Health Checks
- Service availability
- Orchestrator status
- Dependency health
- Performance metrics

### Metrics
- Cycle execution time
- Symbols processed per cycle
- Trade success rate
- Error rates

### Logging
- Trading cycle events
- Error conditions
- Performance warnings
- Service coordination

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cycle_interval` | 120 | Seconds between trading cycles |
| `batch_size` | 25 | Symbols per processing batch |
| `max_risk_per_trade` | 0.05 | Maximum risk per trade (5%) |
| `max_position_size_pct` | 0.2 | Maximum position size (20%) |
| `default_timeframe` | "1h" | Default market data timeframe |

## Error Handling

- **Circuit Breaker**: Automatic failure handling for service dependencies
- **Graceful Degradation**: Continue operation with reduced functionality
- **Retry Logic**: Automatic retry for transient failures
- **Fallback Strategies**: Use cached data when services unavailable

## Development

### Adding New Features
1. Extend trading orchestrator with new pipeline steps
2. Add new API endpoints as needed
3. Update health checks for new components
4. Add configuration options

### Testing
```bash
# Run health check
curl http://localhost:8001/health

# Get status
curl http://localhost:8001/status

# Start trading
curl -X POST http://localhost:8001/start
```

## Performance Considerations

- **Batch Processing**: Process symbols in batches to manage memory
- **Async Operations**: All I/O operations are asynchronous
- **Caching**: Use Redis for frequently accessed data
- **Rate Limiting**: Respect API rate limits across services
- **Resource Monitoring**: Track memory and CPU usage
