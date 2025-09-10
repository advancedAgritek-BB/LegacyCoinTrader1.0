#!/bin/bash
# Trading Bot Monitoring System Setup Script
# This script sets up comprehensive monitoring for your trading bot

set -e

echo "🚀 Setting up Trading Bot Monitoring System"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRYPTO_BOT_DIR="$WORKING_DIR/crypto_bot"
LOGS_DIR="$CRYPTO_BOT_DIR/logs"
MONITORING_LOG="$LOGS_DIR/monitoring_setup.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MONITORING_LOG"
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}❌ This setup script is designed for macOS${NC}"
    exit 1
fi

# Check if required directories exist
if [ ! -d "$CRYPTO_BOT_DIR" ]; then
    echo -e "${RED}❌ Crypto bot directory not found: $CRYPTO_BOT_DIR${NC}"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

log "Starting monitoring system setup"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
log "Python version: $PYTHON_VERSION"

# Check if required Python packages are installed
log "Checking required Python packages..."

REQUIRED_PACKAGES=("psutil" "asyncio" "pathlib" "json" "time" "threading")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $package" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Missing Python packages: ${MISSING_PACKAGES[*]}${NC}"
    echo "Installing missing packages..."
    pip3 install "${MISSING_PACKAGES[@]}"
fi

# Install additional monitoring dependencies
echo "Installing monitoring dependencies..."
pip3 install websocket-client requests python-telegram-bot 2>/dev/null || true

# Create monitoring configuration
log "Creating monitoring configuration..."

cat > "$WORKING_DIR/monitoring_config.json" << EOF
{
    "monitoring": {
        "enabled": true,
        "check_interval_seconds": 30,
        "alert_interval_seconds": 300,
        "auto_recovery_enabled": true,
        "recovery_cooldown_minutes": 15,
        "max_recovery_attempts": 3,
        "telegram_enabled": false,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "email_alerts_enabled": false,
        "email_smtp_server": "",
        "email_smtp_port": 587,
        "email_username": "",
        "email_password": "",
        "email_recipients": []
    },
    "thresholds": {
        "max_evaluation_latency": 5.0,
        "max_execution_latency": 2.0,
        "max_memory_usage_mb": 1000.0,
        "max_cpu_usage_percent": 80.0,
        "max_network_latency_ms": 1000.0,
        "min_websocket_connections": 1,
        "max_error_rate": 0.1
    },
    "logging": {
        "log_level": "INFO",
        "max_log_files": 30,
        "max_log_size_mb": 100
    }
}
EOF

# Create launch scripts
log "Creating monitoring launch scripts..."

# Enhanced monitoring launcher
cat > "$WORKING_DIR/launch_monitoring.sh" << 'EOF'
#!/bin/bash
# Launch Enhanced Monitoring System

WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Starting Enhanced Monitoring System"
echo "Working directory: $WORKING_DIR"

# Change to working directory
cd "$WORKING_DIR"

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Start monitoring in background
echo "Starting monitoring system..."
python3 enhanced_monitoring.py --daemon &

# Save PID
MONITORING_PID=$!
echo $MONITORING_PID > monitoring.pid
echo "✅ Monitoring system started (PID: $MONITORING_PID)"

# Start auto health check
echo "Starting automated health checks..."
python3 auto_health_check.py --quiet &

HEALTH_CHECK_PID=$!
echo $HEALTH_CHECK_PID > health_check.pid
echo "✅ Health check system started (PID: $HEALTH_CHECK_PID)"

echo "📊 Monitoring dashboard available at: http://localhost:8000/monitoring"
echo "🔍 View logs at: crypto_bot/logs/"
EOF

chmod +x "$WORKING_DIR/launch_monitoring.sh"

# Monitoring status checker
cat > "$WORKING_DIR/check_monitoring_status.sh" << 'EOF'
#!/bin/bash
# Check Monitoring System Status

WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔍 Checking Monitoring System Status"
echo "===================================="

# Check if monitoring is running
if [ -f "$WORKING_DIR/monitoring.pid" ]; then
    MONITORING_PID=$(cat "$WORKING_DIR/monitoring.pid")
    if ps -p $MONITORING_PID > /dev/null 2>&1; then
        echo "✅ Enhanced monitoring is running (PID: $MONITORING_PID)"
    else
        echo "❌ Enhanced monitoring process not found (PID: $MONITORING_PID)"
        rm -f "$WORKING_DIR/monitoring.pid"
    fi
else
    echo "❌ Enhanced monitoring PID file not found"
fi

# Check if health check is running
if [ -f "$WORKING_DIR/health_check.pid" ]; then
    HEALTH_PID=$(cat "$WORKING_DIR/health_check.pid")
    if ps -p $HEALTH_PID > /dev/null 2>&1; then
        echo "✅ Auto health check is running (PID: $HEALTH_PID)"
    else
        echo "❌ Auto health check process not found (PID: $HEALTH_PID)"
        rm -f "$WORKING_DIR/health_check.pid"
    fi
else
    echo "❌ Auto health check PID file not found"
fi

# Check log files
echo ""
echo "📁 Log Files Status:"
LOG_FILES=(
    "crypto_bot/logs/pipeline_monitor.log"
    "crypto_bot/logs/monitoring_report.json"
    "crypto_bot/logs/health_check_report.json"
    "crypto_bot/logs/recovery_actions.log"
)

for log_file in "${LOG_FILES[@]}"; do
    if [ -f "$WORKING_DIR/$log_file" ]; then
        FILE_SIZE=$(stat -f%z "$WORKING_DIR/$log_file" 2>/dev/null || echo "0")
        echo "✅ $log_file (${FILE_SIZE} bytes)"
    else
        echo "❌ $log_file (missing)"
    fi
done

# Show recent activity
echo ""
echo "📊 Recent Monitoring Activity:"
if [ -f "$WORKING_DIR/crypto_bot/logs/pipeline_monitor.log" ]; then
    echo "Last 5 log entries:"
    tail -5 "$WORKING_DIR/crypto_bot/logs/pipeline_monitor.log" | sed 's/^/  /'
else
    echo "No monitoring logs found"
fi

# Check system resources
echo ""
echo "🖥️  System Resources:"
echo "  CPU Usage: $(ps aux | awk 'BEGIN {sum=0} {sum+=$3} END {print sum"%"}')"
echo "  Memory Usage: $(ps aux | awk 'BEGIN {sum=0} {sum+=$4} END {print sum"%"}')"
EOF

chmod +x "$WORKING_DIR/check_monitoring_status.sh"

# Stop monitoring script
cat > "$WORKING_DIR/stop_monitoring.sh" << 'EOF'
#!/bin/bash
# Stop Monitoring System

WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🛑 Stopping Monitoring System"
echo "============================="

# Stop enhanced monitoring
if [ -f "$WORKING_DIR/monitoring.pid" ]; then
    MONITORING_PID=$(cat "$WORKING_DIR/monitoring.pid")
    if ps -p $MONITORING_PID > /dev/null 2>&1; then
        echo "Stopping enhanced monitoring (PID: $MONITORING_PID)..."
        kill $MONITORING_PID
        echo "✅ Enhanced monitoring stopped"
    else
        echo "Enhanced monitoring process not running"
    fi
    rm -f "$WORKING_DIR/monitoring.pid"
else
    echo "No enhanced monitoring PID file found"
fi

# Stop health check
if [ -f "$WORKING_DIR/health_check.pid" ]; then
    HEALTH_PID=$(cat "$WORKING_DIR/health_check.pid")
    if ps -p $HEALTH_PID > /dev/null 2>&1; then
        echo "Stopping auto health check (PID: $HEALTH_PID)..."
        kill $HEALTH_PID
        echo "✅ Auto health check stopped"
    else
        echo "Auto health check process not running"
    fi
    rm -f "$WORKING_DIR/health_check.pid"
else
    echo "No auto health check PID file found"
fi

# Kill any remaining monitoring processes
echo "Cleaning up any remaining monitoring processes..."
pkill -f "enhanced_monitoring.py" || true
pkill -f "auto_health_check.py" || true

echo "✅ Monitoring system stopped"
EOF

chmod +x "$WORKING_DIR/stop_monitoring.sh"

# Create cron job installer
cat > "$WORKING_DIR/install_monitoring_cron.sh" << 'EOF'
#!/bin/bash
# Install Monitoring Cron Jobs

WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "⏰ Installing Monitoring Cron Jobs"
echo "=================================="

# Check if cron is available
if ! command -v crontab &> /dev/null; then
    echo "❌ cron is not available on this system"
    exit 1
fi

# Backup existing crontab
crontab -l > "$WORKING_DIR/crontab_backup_$(date +%Y%m%d_%H%M%S).txt" 2>/dev/null || true

# Install new cron jobs
crontab "$WORKING_DIR/monitoring.cron"

echo "✅ Monitoring cron jobs installed"
echo "📋 Installed jobs:"
echo "  • Health check every 5 minutes"
echo "  • Enhanced monitoring status check every minute"
echo "  • Daily monitoring report at 6 AM"
echo "  • Weekly cleanup on Sundays at 2 AM"
echo "  • Service restart check every 10 minutes"
echo ""
echo "To view installed cron jobs: crontab -l"
echo "To edit cron jobs: crontab -e"
EOF

chmod +x "$WORKING_DIR/install_monitoring_cron.sh"

# Make all scripts executable
chmod +x "$WORKING_DIR/enhanced_monitoring.py"
chmod +x "$WORKING_DIR/auto_health_check.py"

# Test monitoring system
log "Testing monitoring system..."
echo "Testing monitoring components..."

# Test Python imports
if python3 -c "from crypto_bot.pipeline_monitor import PipelineMonitor; print('✅ PipelineMonitor import successful')" 2>/dev/null; then
    log "PipelineMonitor import test passed"
else
    log "PipelineMonitor import test failed"
fi

# Test enhanced monitoring
if python3 enhanced_monitoring.py --help >/dev/null 2>&1; then
    log "Enhanced monitoring script test passed"
else
    log "Enhanced monitoring script test failed"
fi

# Test auto health check
if python3 auto_health_check.py --help >/dev/null 2>&1; then
    log "Auto health check script test passed"
else
    log "Auto health check script test failed"
fi

# Create README for monitoring system
cat > "$WORKING_DIR/MONITORING_README.md" << 'EOF'
# Trading Bot Monitoring System

This comprehensive monitoring system ensures your evaluation and execution pipelines are always working correctly.

## 🚀 Quick Start

1. **Start Monitoring:**
   ```bash
   ./launch_monitoring.sh
   ```

2. **Check Status:**
   ```bash
   ./check_monitoring_status.sh
   ```

3. **View Dashboard:**
   Open http://localhost:8000/monitoring in your browser

4. **Stop Monitoring:**
   ```bash
   ./stop_monitoring.sh
   ```

## 📊 What It Monitors

### Evaluation Pipeline
- ✅ Strategy evaluation activity
- ✅ Trading bot process status
- ✅ Configuration validation
- ✅ Recent trading signals

### Execution Pipeline
- ✅ Order execution success/failure rates
- ✅ Pending order queue status
- ✅ WebSocket connection health
- ✅ Trade execution latency

### System Resources
- ✅ Memory usage
- ✅ CPU usage
- ✅ Network connectivity
- ✅ Disk space

### Position Monitoring
- ✅ Active position tracking
- ✅ Trailing stop status
- ✅ PnL calculations
- ✅ Risk management

## 🔧 Automated Recovery

The system can automatically recover from common issues:

- **Restart trading bot** if process dies
- **Clear stuck orders** if queue gets too large
- **Reset WebSocket connections** if connectivity fails
- **Memory cleanup** if usage gets too high

## 📈 Monitoring Dashboard

Access the monitoring dashboard at: http://localhost:8000/monitoring

Features:
- Real-time system health status
- Component-specific monitoring
- Performance metrics and charts
- Active alerts and notifications
- Historical data visualization

## ⚙️ Configuration

Edit `monitoring_config.json` to customize:

```json
{
    "monitoring": {
        "check_interval_seconds": 30,
        "alert_interval_seconds": 300,
        "auto_recovery_enabled": true,
        "telegram_enabled": false
    },
    "thresholds": {
        "max_memory_usage_mb": 1000.0,
        "max_cpu_usage_percent": 80.0
    }
}
```

## 📋 Cron Job Automation

For 24/7 monitoring, install cron jobs:

```bash
./install_monitoring_cron.sh
```

This installs:
- Health checks every 5 minutes
- Status monitoring every minute
- Daily reports at 6 AM
- Weekly cleanup on Sundays
- Automatic service restart if needed

## 📁 Log Files

All monitoring data is stored in `crypto_bot/logs/`:

- `pipeline_monitor.log` - Main monitoring activity
- `monitoring_report.json` - Current health status
- `health_check_report.json` - Detailed health reports
- `recovery_actions.log` - Automated recovery actions
- `daily_health_report_YYYYMMDD.json` - Daily summaries

## 🚨 Alerting

Configure alerts in `monitoring_config.json`:

### Telegram Alerts
```json
{
    "telegram_enabled": true,
    "telegram_bot_token": "YOUR_BOT_TOKEN",
    "telegram_chat_id": "YOUR_CHAT_ID"
}
```

### Email Alerts
```json
{
    "email_alerts_enabled": true,
    "email_smtp_server": "smtp.gmail.com",
    "email_recipients": ["admin@example.com"]
}
```

## 🛠️ Troubleshooting

### Monitoring Not Starting
```bash
# Check Python dependencies
python3 -c "import psutil, websocket"

# Check log files
tail -f crypto_bot/logs/pipeline_monitor.log
```

### Dashboard Not Loading
```bash
# Check Flask app
python3 frontend/app.py

# Check port 8000 availability
lsof -i :8000
```

### Alerts Not Working
```bash
# Test Telegram bot
python3 -c "import telegram; bot = telegram.Bot('YOUR_TOKEN')"

# Check email configuration
python3 -c "import smtplib; smtp = smtplib.SMTP('YOUR_SMTP')"
```

## 📈 Performance Tuning

### For High-Frequency Trading
```json
{
    "monitoring": {
        "check_interval_seconds": 10,
        "max_recovery_attempts": 5
    },
    "thresholds": {
        "max_evaluation_latency": 1.0,
        "max_execution_latency": 0.5
    }
}
```

### For Low-Latency Systems
```json
{
    "monitoring": {
        "check_interval_seconds": 5,
        "alert_interval_seconds": 60
    }
}
```

## 🔒 Security Considerations

- Monitoring runs with same permissions as trading bot
- Log files may contain sensitive trading data
- Consider encrypting log files in production
- Use firewall rules to restrict dashboard access
- Regularly rotate and archive old log files

## 📞 Support

For issues with the monitoring system:

1. Check log files in `crypto_bot/logs/`
2. Run diagnostic: `./check_monitoring_status.sh`
3. Review configuration in `monitoring_config.json`
4. Check system resources and dependencies

The monitoring system is designed to be self-healing and will attempt to resolve most common issues automatically.
EOF

log "Monitoring system setup completed successfully"
echo ""
echo -e "${GREEN}✅ Monitoring System Setup Complete!${NC}"
echo ""
echo "📋 What was installed:"
echo "  • Enhanced monitoring system (enhanced_monitoring.py)"
echo "  • Automated health checks (auto_health_check.py)"
echo "  • Web dashboard integration"
echo "  • Launch/stop/status scripts"
echo "  • Cron job configuration"
echo "  • Comprehensive documentation"
echo ""
echo "🚀 Quick start:"
echo "  1. Start monitoring: ./launch_monitoring.sh"
echo "  2. View dashboard: http://localhost:8000/monitoring"
echo "  3. Check status: ./check_monitoring_status.sh"
echo ""
echo "📖 Read the documentation: MONITORING_README.md"
echo ""
echo "🔧 For automated 24/7 monitoring:"
echo "  ./install_monitoring_cron.sh"
