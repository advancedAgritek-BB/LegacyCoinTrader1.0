# LegacyCoinTrader 1.0

A comprehensive, high-performance cryptocurrency trading bot supporting both centralized exchanges (Kraken, Coinbase Advanced Trade) and Solana DEX trading via Jupiter aggregator. Features advanced strategy routing, regime detection, real-time sentiment analysis, and automated optimization.

## 🚀 Quick Start

### Start the core services
```bash
chmod +x startup.sh
./startup.sh bootstrap   # optional: pull the latest images
./startup.sh start       # spins up Redis, the API gateway, trading engine, etc.
```

### Control the trading engine
```bash
# Inspect scheduler state
python -m crypto_bot.main status

# Start scheduled trading cycles immediately
python -m crypto_bot.main start --interval 60 --immediate

# Pause the engine or trigger an emergency liquidation
python -m crypto_bot.main pause
python -m crypto_bot.main emergency-stop
```

### Manual setup with Docker Compose
```bash
# Start the same stack without the helper script
docker compose up -d \
  redis api-gateway trading-engine market-data portfolio \
  strategy-engine execution monitoring frontend

# Use the CLI to manage trading cycles
python -m crypto_bot.main status
```

## 🧩 Tenant Configuration

Multi-tenant deployments are described in `config/tenants.yaml`. The API
gateway forwards the `X-Tenant-ID` header to downstream services so they can
load tenant-specific strategy, risk, and execution settings. The market data,
trading engine, and execution services expose tenant-aware health endpoints to
surface per-tenant workloads.

## 🕸 Microservice API Gateway

LegacyCoinTrader's microservice rollout is fronted by a FastAPI-powered API gateway
(`services/api_gateway/`). The gateway terminates client connections, validates
JWT or service-to-service tokens, applies Redis-backed rate limiting, and forwards
requests to the individual services defined in `microservice_architecture.yaml`.

### Route Map

| Path Prefix | Downstream Service | Default Target | Authentication | Rate Limit* |
|-------------|-------------------|----------------|----------------|-------------|
| `/api/v1/trading` | Trading Engine | `http://trading-engine:8001` | JWT & service tokens | 60 req/min |
| `/api/v1/market-data` | Market Data | `http://market-data:8002` | JWT & service tokens | 60 req/min |
| `/api/v1/portfolio` | Portfolio | `http://portfolio:8003` | JWT & service tokens | 60 req/min |
| `/api/v1/strategy` | Strategy Engine | `http://strategy-engine:8004` | JWT & service tokens | 60 req/min |
| `/api/v1/token-discovery` | Token Discovery | `http://token-discovery:8005` | JWT & service tokens | 60 req/min |
| `/api/v1/execution` | Execution | `http://execution:8006` | JWT & service tokens | 60 req/min |
| `/api/v1/monitoring` | Monitoring | `http://monitoring:8007` | Service tokens (optional JWT) | 60 req/min |

\*Defaults can be overridden with `GATEWAY_RATE_LIMIT_<SERVICE_NAME>` environment
variables or the global `GATEWAY_RATE_LIMIT_REQUESTS` setting.

### Authentication & Tokens

- **JWT:** Signed using `GATEWAY_JWT_SECRET` (default `change-me`) and the
  algorithm configured via `GATEWAY_JWT_ALGORITHM` (default `HS256`). Tokens must
  present a `sub` claim and optionally a `scopes` list.
- **Service Tokens:** Set `GATEWAY_SERVICE_TOKEN_<SERVICE_NAME>` for each
  microservice or provide a JSON map through `GATEWAY_SERVICE_TOKENS`. The code
  ships with insecure per-service defaults that should only be used in local
  development.
- Toggle enforcement by setting `GATEWAY_REQUIRE_AUTH=0` when running smoke tests.

### Rate Limiting

The gateway stores counters in Redis (`GATEWAY_REDIS_HOST/PORT/DB`). When Redis is
unavailable the limiter gracefully degrades to an in-memory bucket so local
development remains functional.

### Health & Telemetry

- `GET /health` aggregates Redis status and downstream service health checks.
- `GET /routes` returns the active route catalogue and rate-limit configuration.

### Reverse Proxy Configurations

Reference configurations for Envoy (`envoy.yaml`) and Nginx (`nginx.conf`) are
provided in `services/api_gateway/` for production deployments that require an
edge proxy with TLS termination or additional ingress policies.

### Running the Gateway

```bash
# Start the gateway and dependencies
docker-compose up -d api-gateway redis

# View health status
curl http://localhost:8000/health | jq
```

When using the full stack (`make dev` or `docker-compose up`), the frontend and
test harness consume the gateway endpoint at `http://localhost:8000`.

### Trading Engine CLI

The long-running `PhaseRunner` loop has moved into the dedicated trading engine
service. The local CLI now acts as a thin client that issues commands through
the gateway:

```bash
# Show available commands
python -m crypto_bot.main --help

# Trigger a single ad-hoc cycle
python -m crypto_bot.main run-once --metadata trigger=manual

# Pause scheduled cycles
python -m crypto_bot.main pause
```

Each command exits immediately after the gateway acknowledges the request. The
scheduler continues to run inside the trading engine microservice.

## ✨ Key Features

### 🎯 **Multi-Strategy Trading System**
- **25+ Trading Strategies**: From ultra-fast scalping to long-term trend following
- **Adaptive Strategy Router**: Automatically selects optimal strategies based on market conditions
- **Regime Detection**: Real-time market condition classification using EMA, ADX, RSI, and Bollinger Bands
- **Strategy Optimization**: Automated parameter tuning and performance optimization

### 🔄 **Exchange Support**
- **Kraken**: Full REST and WebSocket API support with advanced order types
- **Coinbase Advanced Trade**: Complete trading integration
- **Solana DEX**: Jupiter aggregator for best swap routes and meme token sniping

### 🧠 **Advanced Intelligence**
- **LunarCrush Integration**: Real-time sentiment analysis with Galaxy Score and AltRank
- **ML Signal Model**: Machine learning-based signal generation and validation
- **Portfolio Rotation**: Dynamic asset allocation based on momentum and performance
- **Risk Management**: Comprehensive position sizing, drawdown limits, and volatility filters

### 📊 **Monitoring & Control**
- **Web Dashboard**: Real-time monitoring at `http://localhost:8000`
- **Telegram Bot**: Interactive control with `/menu` commands and trade notifications
- **Performance Analytics**: Detailed metrics, backtesting, and optimization reports
- **Paper Trading**: Risk-free testing with simulated balance and positions

## 🔧 Configuration

### Environment Variables (`.env`)
Secrets are injected at deploy time from your secret manager (Vault, AWS Secrets
Manager, or Parameter Store) using the manifest defined in
`config/managed_secrets.yaml`. The deployment tooling now refuses to start if
required secrets are missing or stale according to the rotation policy.

Set `SECRETS_PROVIDER` to `vault`/`hashicorp` or `aws` to enable runtime secret
resolution. Depending on the provider, the following environment variables must
also be supplied:

* Hashicorp Vault: `VAULT_ADDR`, `VAULT_TOKEN`, `VAULT_SECRET_PATH`,
  `VAULT_VERIFY`, `VAULT_TIMEOUT`
* AWS Secrets Manager: `AWS_SECRET_NAME`, `AWS_REGION`, `AWS_PROFILE`

The frontend no longer generates a default Flask session secret; provision
`SESSION_SECRET_KEY` (or the aliases defined in `config/managed_secrets.yaml`)
through the configured store before starting any service. For automated API key
rotation expose `PORTFOLIO_NEW_API_KEY` (or `PORTFOLIO_ROTATION_API_KEY`) via
the same mechanism so that `PortfolioIdentityService.rotate_api_key` can pull
fresh credentials.

Use `python tools/manage_env.py consolidate` to generate a template populated
with `MANAGED:` placeholders. A minimal example:

```env
# Managed secrets (resolved by your deployment pipeline)
API_KEY=${MANAGED:API_KEY}
API_SECRET=${MANAGED:API_SECRET}
KRAKEN_API_KEY=${MANAGED:KRAKEN_API_KEY}
KRAKEN_API_SECRET=${MANAGED:KRAKEN_API_SECRET}
LUNARCRUSH_API_KEY=${MANAGED:LUNARCRUSH_API_KEY}

# Secret rotation metadata (ISO-8601 timestamp)
SECRETS_ROTATED_AT=2024-01-01T00:00:00Z

# Runtime overrides
EXCHANGE=kraken
EXECUTION_MODE=dry_run
```

Update `SECRETS_ROTATED_AT` whenever your secrets are rotated. The production
deployment script enforces a 30-day rotation window and will block rollout if
the timestamp is missing or too old.

### Configuration

Default settings now live in `crypto_bot/config/settings.py` and are powered by
[`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/).
At runtime the loader merges three sources in the following order:

1. **Code defaults** defined by the Pydantic models.
2. **YAML overrides** stored in `crypto_bot/config.yaml` (optional).
3. **Environment variables** prefixed with `BOT_` (nested fields use
   `__` as a delimiter, e.g. `BOT_RISK__MAX_POSITIONS=5`).

A generated reference document describing every available option lives at
`crypto_bot/config/REFERENCE.md`.

### Production Configuration (`production_config.yaml`)
- `production_config.yaml` is now the single source of truth for production deployments.
- It consolidates the advanced memory, error-handling, and strategy settings that were previously stored in `production_config_enhanced.yaml`.
- Tools such as `deploy_production.py` continue to load `production_config.yaml` by default, so update this file to tune production behaviour.

## 📈 Trading Strategies

### **Ultra-Fast Strategies**
- **`micro_scalp_bot`**: 1-minute timeframe scalping with EMA crossovers
- **`ultra_scalp_bot`**: 15-second ultra-fast scalping with volume spikes
- **`sniper_bot`**: News-like event detection using ATR and volume analysis
- **`flash_crash_bot`**: Rapid crash detection and recovery trading

### **Momentum & Trend Strategies**
- **`trend_bot`**: EMA-based trend following with dynamic stops
- **`momentum_bot`**: RSI and MACD momentum detection
- **`momentum_exploiter`**: Advanced momentum shift detection
- **`breakout_bot`**: Bollinger Band squeeze detection with volume confirmation

### **Mean Reversion Strategies**
- **`grid_bot`**: Dynamic grid trading with leverage (5x)
- **`mean_bot`**: RSI-based mean reversion with volatility adjustment
- **`bounce_scalper`**: Short-term reversal detection
- **`dip_hunter`**: Oversold condition detection and recovery

### **Arbitrage & Market Making**
- **`arbitrage_engine`**: Multi-exchange arbitrage detection
- **`cross_chain_arb_bot`**: Cross-chain arbitrage opportunities
- **`range_arb_bot`**: Range-bound market arbitrage
- **`market_making_bot`**: Automated market making with spread capture

### **Volatility & Specialized**
- **`volatility_harvester`**: ATR-based volatility exploitation
- **`meme_wave_bot`**: Social sentiment-driven meme token trading
- **`dex_scalper`**: Solana DEX arbitrage and scalping
- **`hft_engine`**: High-frequency trading engine

### **Advanced Strategies**
- **`lstm_bot`**: Machine learning-based price prediction
- **`stat_arb_bot`**: Statistical arbitrage with correlation analysis
- **`maker_spread`**: Spread capture market making
- **`dca_bot`**: Dollar-cost averaging automation

## 🌙 LunarCrush Sentiment Integration

Enhance trading decisions with real-time social intelligence:

```yaml
lunarcrush:
  sentiment_boost:
    enabled: true
    min_galaxy_score: 60.0
    min_sentiment: 0.6
    max_boost: 0.5  # 50% position size increase
```

## 🔗 Solana Integration

### **DEX Trading Features**
- Jupiter aggregator for optimal swap routes
- Real-time mempool monitoring
- Automatic slippage adjustment
- Support for 1000+ Solana tokens

### **Meme Token Sniper**
```yaml
meme_wave_sniper:
  enabled: true
  pool:
    url: https://mainnet.helius-rpc.com/v1/?api-key=${HELIUS_KEY}
  risk:
    max_concurrent: 20
    daily_loss_cap: 1.5
```

### **Solana Scanner**
- Real-time token discovery
- Volume and liquidity analysis
- Social sentiment integration
- Pump detection algorithms

## 📱 Telegram Bot

Interactive control via Telegram commands:
- `/menu` - Interactive button menu
- `/signals` - Latest scored assets
- `/balance` - Current holdings
- `/trades` - Trade history
- `/panic_sell` - Exit all positions
- `/status` - Bot performance metrics

## 🖥️ Web Dashboard

Access at `http://localhost:8000`:
- Real-time bot status and performance
- Live trade monitoring and P&L tracking
- Portfolio analytics and risk metrics
- Configuration management interface
- Log viewing and debugging tools
- Strategy performance comparison

## 🧪 Testing & Validation

```bash
# Run comprehensive test suite
pytest -q

# Test specific components
python test_macos_startup.py      # macOS compatibility
python test_integration.py        # Integration tests
python test_strategy_fixes.py     # Strategy validation
```

## 📚 Documentation

- **API Reference**: See `AGENTS.md` for Kraken and Helius API documentation
- **Performance Guide**: See `PERFORMANCE.md` for optimization details
- **Strategy Guide**: See `STRATEGY_OPTIMIZATION_README.md` for advanced configuration

## ⚡ Performance Features

### **Adaptive Systems**
- **Memory Management**: Dynamic cache sizing and garbage collection
- **Concurrency Control**: Adaptive request limits based on performance
- **Rate Limiting**: Smart backoff with exponential retry
- **Caching**: Intelligent TTL management with hit-rate optimization

### **Monitoring & Analytics**
- **Performance Metrics**: API latency, execution times, memory usage
- **Trade Analytics**: Win rate, profit factor, Sharpe ratio tracking
- **Risk Monitoring**: Real-time drawdown and exposure tracking
- **Strategy Analytics**: Individual strategy performance and optimization

## 🚨 Important Notes

- **Dry-run mode recommended first**: Test with paper trading before live execution
- **API rate limits**: Respect exchange rate limits to avoid account restrictions
- **Risk management**: Configure appropriate position sizes and stop losses
- **Backup configuration**: Keep backups of your configuration files
- **Performance monitoring**: Monitor system resources during high-frequency trading

## 🆘 Troubleshooting

### Common Issues
1. **API Connection Errors**: Verify API keys and network connectivity
2. **Nonce Errors**: Ensure system time is synchronized
3. **Memory Issues**: Reduce `max_concurrent_ohlcv` or enable adaptive memory management
4. **WebSocket Issues**: Check firewall settings and API token validity

### Performance Optimization
```yaml
performance:
  memory:
    max_memory_usage_pct: 75.0
    adaptive_cache_sizing: true
  concurrency:
    adaptive_concurrency: true
    max_concurrent_requests: 20
  rate_limiting:
    adaptive_rate_limiting: true
    max_requests_per_minute: 30
```

## 📄 License

This project is provided for educational purposes only. Use at your own risk. Nothing here constitutes financial advice.

---

For detailed API documentation, performance optimization, and advanced strategy configuration, see the respective documentation files in the repository.

