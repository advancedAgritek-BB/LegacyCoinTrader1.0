# 🚀 LegacyCoinTrader Launcher Options

This document explains the different ways to launch LegacyCoinTrader with automatic browser opening.

## 🌐 Available Launchers

### 1. **launch.sh** (Cross-platform Bash)
- **Usage**: `./launch.sh`
- **Features**: 
  - Cross-platform compatibility (macOS, Linux, Windows)
  - Automatic browser detection and opening
  - 3-second delay to ensure server is running
  - Graceful cleanup on exit

### 2. **launch_macos.sh** (macOS Optimized)
- **Usage**: `./launch_macos.sh`
- **Features**:
  - Optimized for macOS
  - Uses native `open` command
  - Faster startup (macOS-specific optimizations)
  - 3-second delay for server startup

### 3. **launch_with_browser.py** (Python Launcher)
- **Usage**: `python launch_with_browser.py` or `./launch_with_browser.py`
- **Features**:
  - Cross-platform Python implementation
  - Uses Python's `webbrowser` module
  - Better error handling
  - More robust process management

### 4. **startup.sh** (Full Setup + Launch)
- **Usage**: `./startup.sh start` or `./startup.sh full`
- **Features**:
  - Full environment setup and dependency installation
  - Automatic browser opening
  - Comprehensive error checking
  - Best for first-time users

## 🔧 Prerequisites

Before using any launcher, ensure you have:

1. **Virtual Environment**: Run `./startup.sh setup` first
2. **Environment File**: Valid `.env` file with real API keys
3. **Dependencies**: All Python packages installed

## 🚀 Quick Start

### For First-Time Users:
```bash
./startup.sh full
```

### For Regular Use:
```bash
# macOS users (recommended)
./launch_macos.sh

# Cross-platform users
./launch.sh

# Python users
python launch_with_browser.py
```

## 🌍 Browser Opening Details

All launchers automatically open your default browser after a 3-second delay to ensure the Flask server is running.

### Supported Platforms:
- **macOS**: Uses `open` command
- **Linux**: Uses `xdg-open`, `gnome-open`, or `kde-open`
- **Windows**: Uses `start` command
- **Fallback**: Manual navigation instructions if automatic opening fails

### Port Configuration:
- **Frontend**: Runs on port 8000 (configurable via `FLASK_RUN_PORT` environment variable)
- **URL**: `http://localhost:8000`

## 🛠️ Troubleshooting

### Browser Doesn't Open:
1. Check if the frontend is running: `ps aux | grep frontend.app`
2. Verify the port: `lsof -i :8000`
3. Try manual navigation: `http://localhost:8000`

### Port Already in Use:
1. Check what's using port 8000: `lsof -i :8000`
2. Kill conflicting processes: `kill -9 <PID>`
3. Or use a different port: `FLASK_RUN_PORT=8001 ./launch.sh`

### Permission Denied:
```bash
chmod +x launch.sh launch_macos.sh
```

## 📱 Alternative Launch Methods

### Manual Launch:
```bash
# Start services manually
source venv/bin/activate
python -m crypto_bot.main &
python -m frontend.app &
python telegram_ctl.py &

# Open browser manually
open http://localhost:8000  # macOS
xdg-open http://localhost:8000  # Linux
start http://localhost:8000  # Windows
```

### Using Python Directly:
```bash
source venv/bin/activate
python -m frontend.app
# Then manually open browser to http://localhost:8000
```

## 🔄 Updating Launchers

If you modify the launcher scripts, make them executable again:
```bash
chmod +x launch.sh launch_macos.sh
```

## 📝 Notes

- All launchers use a 3-second delay to ensure the Flask server is fully started
- The frontend runs on port 8000 by default to avoid conflicts with macOS ControlCenter on port 5000
- Browser opening is done in a background thread to avoid blocking the main application
- All launchers provide graceful cleanup when stopped with Ctrl+C

