# LegacyCoinTrader Shutdown System

This document describes the comprehensive shutdown system implemented for LegacyCoinTrader to safely terminate all application processes.

## Overview

The shutdown system provides:
- ✅ **Safe process termination** with graceful shutdown and fallback to force kill
- 🔍 **Process discovery** across all application components
- 💾 **Data persistence** and state saving before shutdown
- 🧹 **Resource cleanup** including PID files and temporary files
- 📊 **Shutdown reporting** with detailed logs and status
- 🛡️ **Signal handling** for proper response to termination signals

## Quick Usage

### Simple Shutdown
```bash
# Stop all processes safely
./shutdown.sh

# Or use the Python script directly
python3 shutdown_system.py
```

### Management Script (Recommended)
```bash
# Stop all services
./manage.py stop

# Force stop if graceful shutdown fails
./manage.py stop --force

# Check status before shutdown
./manage.py status

# View logs
./manage.py logs
```

## Components

### 1. Comprehensive Shutdown System (`shutdown_system.py`)

The main shutdown script that:
- Discovers all running application processes
- Performs pre-shutdown health checks
- Safely terminates processes in priority order
- Cleans up resources and PID files
- Generates detailed shutdown reports

**Features:**
- Process discovery via PID files and pattern matching
- Graceful shutdown with configurable timeouts
- Force kill as fallback
- Pre-shutdown checks for active trading
- Data backup and state saving
- Comprehensive error handling and reporting

### 2. System Status Checker (`system_status_checker.py`)

Provides detailed status information about all system components:
- Process health monitoring
- Resource usage tracking
- Component-specific health checks
- JSON output for automation
- Watch mode for real-time monitoring

### 3. Signal Handlers (`signal_handlers.py`)

Standardized signal handling for all application components:
- Graceful shutdown on SIGTERM/SIGINT
- Component-specific cleanup callbacks
- PID file management
- Multi-component process support

### 4. Management Interface (`manage.py`)

Unified command-line interface for all system operations:
- Start/stop/restart services
- Status checking and health monitoring
- Log viewing and following
- System backup functionality

### 5. Shell Wrapper (`shutdown.sh`)

Simple shell script wrapper for easy access to the shutdown system.

## Process Discovery

The shutdown system automatically discovers processes using:

1. **PID Files**: Checks for standard PID files:
   - `bot_pid.txt` - Trading Bot
   - `frontend.pid` - Web Frontend
   - `monitoring.pid` - Monitoring System
   - `health_check.pid` - Health Check
   - `telegram.pid` - Telegram Bot
   - And more...

2. **Pattern Matching**: Searches for processes by command line patterns:
   - `crypto_bot.main` - Trading Bot
   - `frontend.app` - Web Frontend
   - `enhanced_scanner.py` - Enhanced Scanner
   - `websocket_monitor.py` - WebSocket Monitor
   - And more...

## Shutdown Priority

Processes are shut down in priority order (lower number = higher priority):

1. **Trading Bot** - Stop trading operations first
2. **Web Frontend** - Stop user interface access
3. **Enhanced Scanner** - Stop data collection
4. **Strategy Router** - Stop strategy processing
5. **WebSocket Monitor** - Stop real-time data feeds
6. **Monitoring System** - Stop monitoring
7. **Health Check** - Stop health checks
8. **Telegram Bot** - Stop notifications
9. **Production Monitor** - Stop production monitoring
10. **API Server** - Stop API access last

## Pre-Shutdown Checks

Before shutting down, the system performs:

- ✅ **Trading Activity Check** - Detects active trading and attempts to close positions
- 💾 **State Saving** - Saves current application state
- ⏳ **Pending Operations** - Waits for pending operations to complete
- 🗂️ **Data Backup** - Backs up critical data files

## Usage Examples

### Basic Shutdown
```bash
# Simple shutdown
./shutdown.sh

# Dry run (show what would be shut down)
./shutdown.sh --dry-run

# Force shutdown (skip checks)
./shutdown.sh --force

# Skip pre-shutdown checks
./shutdown.sh --skip-checks
```

### Status Checking
```bash
# Check system status
python3 system_status_checker.py

# Detailed status
python3 system_status_checker.py --detailed

# JSON output
python3 system_status_checker.py --json

# Watch mode (refresh every 5 seconds)
python3 system_status_checker.py --watch 5

# Save status report
python3 system_status_checker.py --save
```

### Management Interface
```bash
# Start all services
./manage.py start

# Start specific service
./manage.py start --service bot

# Stop all services
./manage.py stop

# Force stop
./manage.py stop --force

# Check status
./manage.py status --detailed

# View logs
./manage.py logs

# Follow logs in real-time
./manage.py logs --follow

# Restart system
./manage.py restart

# Health check
./manage.py health

# Create backup
./manage.py backup
```

## Integration with Existing Scripts

The shutdown system integrates with existing startup scripts:

- `start_all_services.py` - Enhanced with comprehensive shutdown
- `stop_integrated.sh` - Falls back to comprehensive shutdown
- `stop_monitoring.sh` - Falls back to comprehensive shutdown

## Logging and Reporting

### Shutdown Logs
- Written to `logs/shutdown.log`
- Includes detailed process information
- Records all cleanup operations
- Notes any errors or warnings

### Status Reports
- Saved to `logs/shutdown_report.json`
- Contains process discovery results
- Lists all cleanup operations performed
- Includes timing and error information

### System Status
- Real-time process monitoring
- Health status for each component
- Resource usage tracking
- Saved to `logs/system_status_*.json`

## Error Handling

The shutdown system handles various error scenarios:

- **Stale PID Files** - Automatically cleaned up
- **Unresponsive Processes** - Force killed after timeout
- **Missing Components** - Gracefully skipped
- **Permission Errors** - Logged and continued
- **Unexpected Errors** - Caught and reported

## Customization

### Adding New Processes

To add monitoring for new processes:

1. **Update `shutdown_system.py`**:
   ```python
   self.process_patterns.append(
       ("New Process", "new_process_pattern", priority)
   )
   ```

2. **Add PID file mapping**:
   ```python
   self.pid_files["new_process.pid"] = ("New Process", "new_process_pattern")
   ```

3. **Update status checker** in `system_status_checker.py`

### Custom Cleanup Callbacks

Add component-specific cleanup:

```python
from signal_handlers import setup_basic_signal_handlers

def my_cleanup():
    # Custom cleanup logic
    pass

handler = setup_basic_signal_handlers(
    "My Component",
    Path("my_component.pid"),
    my_cleanup
)
```

## Troubleshooting

### Common Issues

1. **Processes Won't Stop**
   ```bash
   # Check what's running
   ./manage.py status --detailed
   
   # Force stop
   ./manage.py stop --force
   ```

2. **Stale PID Files**
   ```bash
   # Clean up manually
   rm -f *.pid
   
   # Or use the status checker
   python3 system_status_checker.py
   ```

3. **Permission Errors**
   ```bash
   # Make sure scripts are executable
   chmod +x *.sh *.py
   
   # Check file ownership
   ls -la *.pid
   ```

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

### Manual Process Cleanup

If all else fails:
```bash
# Find all related processes
ps aux | grep -E "(crypto_bot|frontend|telegram)"

# Kill specific processes
pkill -f crypto_bot.main
pkill -f frontend.app

# Clean up PID files
rm -f *.pid
```

## Security Considerations

- PID files are checked for validity before use
- Process ownership is verified before termination
- Signal handlers prevent unauthorized shutdown
- Backup data is protected with appropriate permissions

## Performance

- Process discovery is optimized for speed
- Parallel shutdown where possible
- Configurable timeouts to prevent hanging
- Minimal resource usage during operation

## Future Enhancements

Planned improvements:
- Integration with systemd/launchd
- Remote shutdown capabilities
- Scheduled maintenance shutdowns
- Enhanced health monitoring
- Automatic restart on failure

## Support

For issues with the shutdown system:
1. Check the logs in `logs/shutdown.log`
2. Run status check: `./manage.py status --detailed`
3. Try force shutdown: `./manage.py stop --force`
4. Review this documentation
5. Check process ownership and permissions
