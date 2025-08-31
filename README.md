# Crypto Trading Bot

A modular hybrid cryptocurrency trading bot for centralized exchanges (Coinbase Advanced Trade, Kraken) and Solana DEXes via Jupiter aggregator.

## 🚀 Quick Start

### Automated Setup (Recommended)
```bash
chmod +x startup.sh
./startup.sh
```

### Manual Setup
```bash
pip install -r requirements.txt
python crypto_bot/wallet_manager.py  # Configure API credentials
python -m crypto_bot.main            # Start trading bot
python -m frontend.app               # Launch web dashboard
```

## ✨ Key Features

- **Multi-Strategy Support**: Trend, grid, breakout, scalping, mean reversion, sniper, and DEX strategies
- **Regime Detection**: Automatic market condition classification using EMA, ADX, RSI, and Bollinger Bands
- **LunarCrush Integration**: Real-time sentiment analysis with Galaxy Score and AltRank
- **Solana Support**: DEX trading, meme-wave sniping, and token discovery
- **Risk Management**: Position sizing, drawdown limits, and volatility filters
- **Telegram Control**: Interactive bot management with `/menu` commands
- **Web Dashboard**: Real-time monitoring at `http://localhost:5000`
- **Performance Optimization**: Dynamic memory management, adaptive concurrency, and intelligent caching

## 🔧 Configuration

### Environment Variables (`crypto_bot/.env`)
```env
EXCHANGE=coinbase              # or kraken
API_KEY=your_key
API_SECRET=your_secret
API_PASSPHRASE=your_coinbase_passphrase_if_needed
TELEGRAM_TOKEN=your_telegram_token
SOLANA_PRIVATE_KEY="[1,2,3,...]"       # for Solana trades
HELIUS_KEY=your_helius_api_key         # for Helius RPC endpoints
LUNARCRUSH_API_KEY=your_lunarcrush_key  # for sentiment analysis
```

### Main Configuration (`crypto_bot/config.yaml`)
```yaml
exchange: coinbase
execution_mode: dry_run  # or live
use_websocket: true

risk:
  trade_size_pct: 1.5
  max_open_trades: 10
  sl_mult: 1.5
  tp_mult: 3.0

telegram:
  token: your_telegram_token
  chat_id: your_chat_id
  trade_updates: true
```

## 📊 Trading Strategies

### Core Strategies
- **trend_bot**: EMA-based trend following with dynamic stops
- **grid_bot**: Mean reversion grid trading with dynamic spacing
- **breakout_bot**: Bollinger Band squeeze detection with volume confirmation
- **bounce_scalper**: Short-term reversal detection with RSI and volume spikes
- **mean_bot**: RSI-based mean reversion with volatility-adjusted thresholds
- **sniper_bot**: News-like event detection using ATR and volume analysis
- **micro_scalp**: High-frequency scalping with tick-level aggregation
- **dex_scalper**: Solana DEX arbitrage and scalping strategies

### Strategy Configuration
```yaml
strategy_allocation:
  trend_bot: 25
  grid_bot: 20
  breakout_bot: 20
  bounce_scalper: 15
  mean_bot: 10
  sniper_bot: 10

strategy_router:
  trending: [trend_bot, grid_bot]
  sideways: [grid_bot, mean_bot]
  breakout: [breakout_bot, sniper_bot]
  volatile: [bounce_scalper, micro_scalp]
```

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

### DEX Trading
- Jupiter aggregator for best swap routes
- Real-time mempool monitoring
- Automatic slippage adjustment
- Support for multiple Solana tokens

### Meme-Wave Sniper
```yaml
meme_wave_sniper:
  enabled: true
  pool:
    url: https://mainnet.helius-rpc.com/v1/?api-key=${HELIUS_KEY}
  risk:
    max_concurrent: 20
    daily_loss_cap: 1.5
```

## 📱 Telegram Bot

Interactive control via Telegram commands:
- `/menu` - Interactive button menu
- `/signals` - Latest scored assets
- `/balance` - Current holdings
- `/trades` - Trade history
- `/panic_sell` - Exit all positions

## 🖥️ Web Dashboard

Access at `http://localhost:5000`:
- Real-time bot status and performance
- Live trade monitoring
- Portfolio analytics
- Configuration management
- Log viewing

## 🧪 Testing

```bash
pytest -q                    # Run test suite
python test_macos_startup.py # macOS compatibility test
```

## 📚 Documentation

- **API Reference**: See `AGENTS.md` for Kraken and Helius API documentation
- **Performance Guide**: See `PERFORMANCE.md` for optimization details
- **Strategy Guide**: See `STRATEGY_OPTIMIZATION_README.md` for advanced configuration

## 🚨 Important Notes

- **Dry-run mode recommended first**: Test with paper trading before live execution
- **API rate limits**: Respect exchange rate limits to avoid account restrictions
- **Risk management**: Configure appropriate position sizes and stop losses
- **Backup configuration**: Keep backups of your configuration files

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
    max_requests_per_minute: 10
```

## 📄 License

This project is provided for educational purposes only. Use at your own risk. Nothing here constitutes financial advice.

---

For detailed API documentation, performance optimization, and advanced strategy configuration, see the respective documentation files in the repository.

