# LegacyCoinTrader Auto-Start & Monitoring Setup

This guide explains how to set up automatic startup and continuous monitoring for your LegacyCoinTrader trading system to prevent the issues you experienced.

## 🚀 Quick Setup (Recommended)

### Option 1: Complete Automated Setup (Linux/macOS)
```bash
cd /Users/brandonburnette/Downloads/LegacyCoinTrader1.0-1
./setup-autostart.sh --all
```

This installs:
- ✅ Systemd service (automatic startup on boot)
- ✅ Cron job (health monitoring every 5 minutes)
- ✅ Supervisor configuration (process management)
- ✅ All necessary scripts and permissions

### Option 2: Manual Setup by Component

#### Systemd Service (Automatic Boot Startup)
```bash
./setup-autostart.sh --systemd
```

#### Health Monitoring (Cron Job)
```bash
./setup-autostart.sh --cron
```

#### Process Supervisor (Advanced Management)
```bash
./setup-autostart.sh --supervisor
```

## 📊 What Gets Installed

### 1. Systemd Service (`monitoring.service`)
- **Purpose**: Starts LegacyCoinTrader on system boot
- **Location**: `~/.config/systemd/user/monitoring.service`
- **Commands**:
  ```bash
  systemctl --user status monitoring.service  # Check status
  systemctl --user restart monitoring.service # Restart
  systemctl --user stop monitoring.service    # Stop
  ```

### 2. Cron Job (Health Monitoring)
- **Purpose**: Checks service health every 5 minutes
- **Location**: Your user crontab
- **Function**: Restarts failed services automatically
- **Logs**: `health-check.log`

### 3. Supervisor Configuration
- **Purpose**: Advanced process management
- **Location**: `~/.supervisor/legacycointrader.conf`
- **Commands**:
  ```bash
  supervisord -c ~/.supervisor/legacycointrader.conf
  supervisorctl status
  ```

## 🔧 Manual Operation

### Start Services
```bash
# Recommended: Auto-startup with monitoring
./auto-startup.sh

# Basic startup (no monitoring)
./startup.sh start

# Manual Docker Compose (your current method)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Advanced management
./docker-manager.sh start dev
```

### Stop Services
```bash
# Graceful shutdown
./shutdown.sh

# Manual Docker Compose
docker-compose -f docker-compose.yml -f docker-compose.dev.yml down

# Force shutdown
./shutdown.sh force

# Quick stop
./docker-manager.sh stop
```

### Health Monitoring
```bash
# Manual health check
./health-check-cron.sh

# Continuous monitoring
./docker-manager.sh watch

# Status overview
./docker-manager.sh status
```

## 📈 Monitoring & Alerts

### Health Checks Include:
- ✅ Docker daemon status
- ✅ All microservices (11 services)
- ✅ Trading engine status
- ✅ Database connectivity
- ✅ System resources (disk, memory)
- ✅ Automatic restarts for failed services

### Alert System:
- **Email Alerts**: Set `ALERT_EMAIL` in `.env`
- **System Logs**: All events logged to system journal
- **Log Files**:
  - `startup.log` - Startup events
  - `health-check.log` - Health monitoring
  - `logs/supervisor.log` - Process management
  - `logs/health-monitor.log` - Detailed health checks

## 🔍 Troubleshooting

### Check Service Status
```bash
# Systemd
systemctl --user status monitoring.service

# Manual Docker Compose
docker-compose -f docker-compose.yml -f docker-compose.dev.yml ps

# Trading engine
python3 -m crypto_bot.main status

# Health overview
./docker-manager.sh health

# Quick status dashboard
./status-dashboard.sh
```

### View Logs
```bash
# Systemd logs
journalctl --user -u monitoring.service -f

# Manual Docker Compose logs
docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

# Application logs
tail -f logs/*.log

# Health check logs
tail -f health-check.log
```

### Restart Everything
```bash
# Stop everything
./shutdown.sh force

# Manual Docker Compose restart
docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Start fresh with monitoring
./auto-startup.sh
```

## 🛠️ Configuration Files

### Environment Variables (`.env`)
```bash
# API Keys (Required)
BITQUERY_KEY=your_bitquery_key_here
MORALIS_KEY=your_moralis_key_here

# Email alerts
ALERT_EMAIL=your_email@example.com

# Logging
LOG_LEVEL=INFO
```

### Systemd Service (`monitoring.service`)
- Automatic startup on system boot
- Proper shutdown handling
- Resource limits and security settings

### Cron Job
- Runs every 5 minutes
- Checks all services
- Restarts failed services
- Sends alerts on failures

## 🚨 Emergency Procedures

### If Services Won't Start:
1. Check Docker: `docker info`
2. Restart Docker: `sudo systemctl restart docker`
3. Clean restart: `./shutdown.sh force && ./auto-startup.sh`

### If Services Keep Crashing:
1. Check logs: `tail -f logs/*.log`
2. Check resources: `df -h` and `free -h`
3. Restart with clean state: `./docker-manager.sh clean && ./auto-startup.sh`

### Disable Auto-Start Temporarily:
```bash
# Disable systemd
systemctl --user disable monitoring.service

# Remove cron job
crontab -e  # Remove the health check line
```

## 📋 System Requirements

- **Docker & Docker Compose**: Latest versions
- **Python 3.11+**: For monitoring scripts
- **Systemd**: For service management (Linux)
- **Cron**: For scheduled health checks
- **Email**: Optional, for alerts

## 🔒 Security Considerations

- Services run as your user (not root)
- No privileged Docker access required
- Environment variables for sensitive data
- Resource limits prevent system overload
- Proper signal handling for graceful shutdowns

## 🎯 What This Prevents

✅ **No more forgotten startups** - Services start automatically
✅ **No more crashed services** - Automatic restart on failure
✅ **No more missed alerts** - Email notifications for issues
✅ **No more manual monitoring** - Automated health checks
✅ **No more data loss** - Proper shutdown procedures
✅ **No more resource issues** - Monitoring prevents overload

Your trading system will now be **highly reliable** and **self-managing**! 🚀
