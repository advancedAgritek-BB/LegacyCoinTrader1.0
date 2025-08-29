# 🚀 LegacyCoinTrader Startup Scripts

This directory contains comprehensive startup scripts for LegacyCoinTrader, inspired by the setup scripts from the [LegacyCoinTrader GitHub repository](https://github.com/advancedAgritek-BB/LegacyCoinTrader1.0-main).

## 📁 Scripts Overview

### 1. `startup.sh` - Full Setup & Launch Script
**Comprehensive script that handles everything from dependency installation to application launch.**

**Features:**
- ✅ OS detection (macOS/Linux)
- ✅ System dependency installation
- ✅ Python virtual environment setup
- ✅ Package installation from requirements.txt
- ✅ Environment configuration checking
- ✅ Test suite execution
- ✅ Multi-service application launch
- ✅ Process management and cleanup

**Usage:**
```bash
# Full setup and launch (default)
./startup.sh

# Only setup dependencies
./startup.sh setup

# Only run tests
./startup.sh test

# Only start application (assumes setup is complete)
./startup.sh start

# Show help
./startup.sh help
```

### 2. `launch.sh` - Quick Launcher
**Simple script for quick startup when environment is already configured.**

**Features:**
- ✅ Quick environment validation
- ✅ Multi-service startup
- ✅ Graceful shutdown handling
- ✅ Process monitoring

**Usage:**
```bash
./launch.sh
```

## 🛠️ Prerequisites

### macOS
- Homebrew (will be installed automatically if missing)
- Xcode Command Line Tools (may be required for some Python packages)

### Linux
- Python 3.11+
- Git
- Curl
- Package manager (apt, yum, or dnf)

## 🚀 Quick Start

### First Time Setup
```bash
# Make scripts executable (if not already done)
chmod +x startup.sh launch.sh

# Run full setup
./startup.sh
```

### Subsequent Launches
```bash
# Quick launch
./launch.sh
```

## 🔧 What Gets Installed

### System Dependencies
- **Python 3.11+** - Core runtime
- **Git** - Version control
- **Curl** - HTTP client
- **Homebrew** (macOS) - Package manager

### Python Packages
- **Core dependencies** from `requirements.txt`
- **GPU acceleration** from `requirements_gpu.txt` (if available)
- **Fallback packages** if requirements files are missing

## 🌐 Services Started

When you run the startup scripts, the following services are launched:

1. **Main Trading Bot** (`crypto_bot.main`)
   - Core trading logic and strategy execution
   - Market data processing
   - Order management

2. **Web Dashboard** (`frontend.app`)
   - Real-time trading interface
   - Portfolio monitoring
   - Strategy performance metrics
   - Available at: http://localhost:5000

3. **Telegram Bot** (`telegram_ctl.py`)
   - Mobile notifications
   - Trading commands
   - Status updates

## ⚙️ Configuration

### Environment File (.env)
The startup script will create a `.env` template if one doesn't exist. You'll need to configure:

- **Exchange API keys** (Kraken, Coinbase)
- **Telegram bot token**
- **Solana wallet credentials**
- **Supabase database credentials**
- **LunarCrush API key** (optional)

### Configuration Files
- `crypto_bot/config.yaml` - Main trading configuration
- `crypto_bot/config/lunarcrush_config.yaml` - Sentiment analysis settings
- `crypto_bot/user_config.yaml` - User-specific settings

## 🧪 Testing

The startup script includes automated testing:

```bash
# Run tests only
./startup.sh test

# Or manually
source venv/bin/activate
python -m pytest -q
```

## 🚨 Safety Features

### Dry Run Mode
- **Default execution mode** is `dry_run`
- **No real trades** are executed
- **Safe for testing** and development

### Live Trading
- **Change `EXECUTION_MODE=live`** in `.env` for real trading
- **⚠️ Use with caution** - real money is at risk
- **Test thoroughly** before enabling

## 📊 Monitoring

### Process Management
```bash
# View running processes
ps aux | grep python

# Check specific PIDs
ps -p $MAIN_PID,$FRONTEND_PID,$TELEGRAM_PID

# Stop all services
kill $MAIN_PID $FRONTEND_PID $TELEGRAM_PID
```

### Logs
- **Main bot logs**: `crypto_bot/logs/bot.log`
- **Execution logs**: `crypto_bot/logs/execution.log`
- **Telegram logs**: `crypto_bot/logs/telegram_ctl.log`

## 🔍 Troubleshooting

### Common Issues

1. **Virtual Environment Not Found**
   ```bash
   ./startup.sh setup
   ```

2. **Missing .env File**
   ```bash
   ./startup.sh setup
   # Then edit .env with your API keys
   ```

3. **Permission Denied**
   ```bash
   chmod +x startup.sh launch.sh
   ```

4. **Port Already in Use**
   ```bash
   # Check what's using port 5000
   lsof -i :5000
   
   # Kill conflicting process
   kill -9 <PID>
   
   # Or use a different port by setting environment variable
   export FLASK_RUN_PORT=8000
   ```

5. **Web Dashboard Port Configuration**
   The web dashboard defaults to port 8000 to avoid conflicts with macOS ControlCenter on port 5000.
   You can customize the port by setting the `FLASK_RUN_PORT` environment variable:
   
   ```bash
   # Set custom port
   export FLASK_RUN_PORT=3000
   
   # Or add to your .env file
   echo "FLASK_RUN_PORT=3000" >> .env
   ```

### Dependency Issues
```bash
# Reinstall Python packages
source venv/bin/activate
pip install --force-reinstall -r requirements.txt
```

## 📚 Additional Resources

- **Main Documentation**: [README.md](README.md)
- **API Reference**: [AGENTS.md](AGENTS.md)
- **Strategy Guide**: [STRATEGY_INTEGRATION_SUMMARY.md](STRATEGY_INTEGRATION_SUMMARY.md)
- **LunarCrush Integration**: [LUNARCRUSH_INTEGRATION.md](LUNARCRUSH_INTEGRATION.md)

## 🤝 Contributing

These startup scripts are designed to be:
- **Cross-platform compatible** (macOS/Linux)
- **Error-resistant** with comprehensive error handling
- **User-friendly** with colored output and clear messages
- **Maintainable** with modular function design

Feel free to submit improvements and bug reports!

---

**⚠️ Disclaimer**: This software is for educational purposes only. Use at your own risk. Nothing here constitutes financial advice.
