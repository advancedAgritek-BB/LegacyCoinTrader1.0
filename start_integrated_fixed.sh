#!/bin/bash
# Fixed Integrated LegacyCoinTrader Startup Script
# Starts bot, monitoring, and web server in one unified process

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}🚀 Starting LegacyCoinTrader - Integrated Edition${NC}"
echo -e "${BLUE}$(printf '%.0s=' {1..60})${NC}"
echo -e "${GREEN}🤖 Trading Bot + 📊 Monitoring Dashboard + 🌐 Web Server${NC}"
echo -e "${BLUE}$(printf '%.0s=' {1..60})${NC}"

# Check if we're already running
if pgrep -f "start_bot_auto.py" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Integrated system appears to already be running${NC}"
    echo "   To stop: pkill -f 'start_bot_auto.py'"
    echo "   To restart: pkill -f 'start_bot_auto.py' && ./start_integrated.sh"
    exit 1
fi

# Clean up any stale PID files
rm -f *.pid 2>/dev/null || true

# Start the integrated system in background
echo "🎯 Launching integrated system..."
echo ""

# Run the bot in background and capture output
python3 start_bot_auto.py > bot_output.log 2>&1 &
BOT_PID=$!

echo "✅ Bot started with PID: $BOT_PID"
echo "📋 Logs are being written to: bot_output.log"
echo ""

# Wait a moment for the web server to start
echo "⏳ Waiting for web server to initialize..."
sleep 10

# Check if the bot is still running
if ! kill -0 $BOT_PID 2>/dev/null; then
    echo -e "${RED}❌ Bot failed to start. Check bot_output.log for details.${NC}"
    exit 1
fi

# Try to find the web server port
PORT=$(lsof -p $BOT_PID 2>/dev/null | grep LISTEN | head -1 | awk '{print $9}' | sed 's/.*://' || echo "8000")

# Validate port is numeric
if [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "🌐 Web server detected on port $PORT"
    echo "📊 Dashboard: http://localhost:$PORT"
    echo "📋 Monitoring: http://localhost:$PORT/monitoring"
    echo ""
    
    # Try to open browser
    echo "🌐 Opening browser..."
    if command -v open >/dev/null 2>&1; then
        open "http://localhost:$PORT"
        echo "✅ Browser opened successfully"
    else
        echo "⚠️ Could not open browser automatically"
        echo "🌐 Please manually navigate to: http://localhost:$PORT"
    fi
else
    echo "⚠️ Could not detect web server port"
    echo "🌐 Try accessing: http://localhost:8000 or http://localhost:8001"
fi

echo ""
echo -e "${GREEN}✅ Integrated system is running!${NC}"
echo "📋 To view logs: tail -f bot_output.log"
echo "🛑 To stop: pkill -f 'start_bot_auto.py'"
echo "🔄 To restart: ./start_integrated.sh"
echo ""
echo "The terminal is now free for other commands."
echo "The bot will continue running in the background."
