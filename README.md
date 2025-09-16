# LegacyCoinTrader 1.0

A comprehensive, high-performance cryptocurrency trading bot supporting both centralized exchanges (Kraken, Coinbase Advanced Trade) and Solana DEX trading via Jupiter aggregator. Features advanced strategy routing, regime detection, real-time sentiment analysis, and automated optimization.

## 🚀 Quick Start

### Automated Setup (Recommended)
```bash
chmod +x startup.sh
./startup.sh
```

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure API credentials
python crypto_bot/wallet_manager.py

# Start trading bot
python -m crypto_bot.main

# Launch web dashboard
python -m frontend.app
```

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
```env
# Exchange Configuration
EXCHANGE=kraken                    # or coinbase
API_KEY=your_kraken_api_key
API_SECRET=your_kraken_api_secret
API_PASSPHRASE=your_coinbase_passphrase_if_needed

# Telegram Configuration
TELEGRAM_TOKEN=your_telegram_token
TELEGRAM_CHAT_ID=your_chat_id

# Solana Configuration
HELIUS_KEY=your_helius_api_key
SOLANA_PRIVATE_KEY="[1,2,3,...]"
WALLET_ADDRESS=your_wallet_address

# Sentiment Analysis
LUNARCRUSH_API_KEY=your_lunarcrush_key

# Trading Mode
EXECUTION_MODE=dry_run            # or live
```

### Main Configuration (`crypto_bot/config.yaml`)
```yaml
# Core Settings
exchange: kraken
execution_mode: dry_run
use_websocket: false

# Risk Management
risk:
  trade_size_pct: 0.1
  max_open_trades: 10
  stop_loss_pct: 0.008
  take_profit_pct: 0.045
  max_drawdown: 0.25

# Strategy Allocation
strategy_allocation:
  micro_scalp_bot: 30
  sniper_bot: 25
  bounce_scalper: 15
  grid_bot: 15
  trend_bot: 15

# Performance Optimization
performance:
  adaptive_concurrency: true
  adaptive_rate_limiting: true
  adaptive_cache_sizing: true
  max_memory_usage_pct: 80.0
```

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

