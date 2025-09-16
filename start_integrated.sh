#!/bin/bash
# Comprehensive LegacyCoinTrader Startup Script
# Starts everything: OHLCV fetching, trading bot, web dashboard, and browser

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}🚀 Starting LegacyCoinTrader - Complete System${NC}"
echo -e "${BLUE}$(printf '%.0s=' {1..60})${NC}"
echo -e "${GREEN}📈 OHLCV Fetching + 🤖 Trading Bot + 🌐 Web Dashboard + 🖥️ Browser${NC}"
echo -e "${BLUE}$(printf '%.0s=' {1..60})${NC}"

# Check if we're already running
if pgrep -f "start_bot.py\|crypto_bot.main\|frontend.app" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  System appears to already be running${NC}"
    echo "   To stop: Ctrl+C or use './stop_integrated.sh'"
    echo "   Check processes: ps aux | grep -E '(start_bot.py|crypto_bot|frontend)'"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down all services...${NC}"
    pkill -f "start_bot.py" || true
    pkill -f "crypto_bot.main" || true
    pkill -f "frontend.app" || true
    pkill -f "telegram_ctl.py" || true
    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Clean up any stale PID files
rm -f *.pid 2>/dev/null || true

# Verify .env file exists
if [[ ! -f ".env" ]]; then
    echo -e "${RED}❌ Error: .env file not found!${NC}"
    echo "   Please create a .env file with your API keys"
    echo "   See .env.example for the required format"
    exit 1
fi

# Start the complete system
echo "🎯 Launching complete LegacyCoinTrader system..."
echo ""

# Set environment variables for proper startup
export AUTO_START_TRADING=1
export NON_INTERACTIVE=1

# Start the integrated system
python3 start_bot.py auto

echo ""
echo -e "${GREEN}✅ Complete system has stopped${NC}"
