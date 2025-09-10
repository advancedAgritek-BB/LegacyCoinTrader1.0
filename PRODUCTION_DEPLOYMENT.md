# Production Deployment Guide

## Overview
LegacyCoinTrader now supports production deployment using Gunicorn instead of Flask's development server.

## Production Mode

### Automatic Detection
The system automatically detects production mode when:
- `FLASK_ENV=production` environment variable is set
- `PRODUCTION=true` environment variable is set

### Manual Configuration
Set environment variables before starting:
```bash
export FLASK_ENV=production
# or
export PRODUCTION=true
```

### Starting in Production Mode
```bash
# Method 1: Using environment variable
FLASK_ENV=production ./startup.sh

# Method 2: Using PRODUCTION variable
PRODUCTION=true ./startup.sh
```

## Production Features

### Gunicorn Configuration
- **Workers**: `CPU cores * 2 + 1` (optimal for most systems)
- **Worker Class**: `gevent` (async support)
- **Timeout**: 120 seconds (handles long-running requests)
- **Max Requests**: 1000 per worker (automatic restart)

### Logging
Production logs are stored in:
- Access log: `logs/gunicorn_access.log`
- Error log: `logs/gunicorn_error.log`
- PID file: `gunicorn.pid`

### Performance Benefits
- Multiple worker processes for concurrent requests
- Automatic worker recycling to prevent memory leaks
- Proper request queuing and load balancing
- Better error handling and recovery

## Development Mode (Default)

### Automatic Detection
If neither `FLASK_ENV=production` nor `PRODUCTION=true` is set, the system uses Flask development server.

### Starting in Development Mode
```bash
./startup.sh  # Uses Flask development server by default
```

## Configuration Files

### Gunicorn Configuration
Located at: `gunicorn.conf.py`
- Configurable worker count
- Logging configuration
- Timeout settings
- SSL support (commented out)

### Requirements
Added to `requirements.txt`:
- `gunicorn` - WSGI server
- `gevent` - Async worker class

## Troubleshooting

### Gunicorn Won't Start
1. Check if port 8000 is available
2. Verify Python environment has required packages
3. Check `logs/gunicorn_error.log` for detailed errors

### Port Conflicts
- Gunicorn binds to `0.0.0.0:8000`
- Flask development server finds available port starting from 8000

### Memory Issues
- Reduce worker count in `gunicorn.conf.py`
- Monitor memory usage with `htop` or `ps aux`

## Migration from Development

### Existing Deployments
No changes needed for existing development deployments - they continue to work as before.

### New Production Deployments
1. Install new requirements: `pip install -r requirements.txt`
2. Set production environment variables
3. Start with: `PRODUCTION=true ./startup.sh`

## Monitoring Production

### Check Running Processes
```bash
# Check Gunicorn processes
ps aux | grep gunicorn

# Check specific PID
cat gunicorn.pid
```

### View Logs
```bash
# Access logs
tail -f logs/gunicorn_access.log

# Error logs
tail -f logs/gunicorn_error.log
```

### Restart Services
```bash
# Kill existing processes
pkill -f gunicorn
pkill -f "frontend.app"

# Restart
PRODUCTION=true ./startup.sh
```
