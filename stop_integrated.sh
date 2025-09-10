#!/bin/bash
# Stop LegacyCoinTrader Integrated System
# Gracefully stops all running services

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 Stopping LegacyCoinTrader - Complete System${NC}"
echo -e "${BLUE}$(printf '%.0s=' {1..50})${NC}"

# Function to check if process is running
is_running() {
    pgrep -f "$1" > /dev/null 2>&1
    return $?
}

# Function to stop process gracefully
stop_process() {
    local pattern="$1"
    local name="$2"

    if is_running "$pattern"; then
        echo -e "${YELLOW}Stopping $name...${NC}"
        pkill -f "$pattern" || true

        # Wait for process to stop
        local count=0
        while is_running "$pattern" && [ $count -lt 10 ]; do
            sleep 1
            count=$((count + 1))
        done

        if is_running "$pattern"; then
            echo -e "${RED}Force stopping $name...${NC}"
            pkill -9 -f "$pattern" || true
            sleep 1
        fi

        if is_running "$pattern"; then
            echo -e "${RED}❌ Failed to stop $name${NC}"
            return 1
        else
            echo -e "${GREEN}✅ $name stopped${NC}"
            return 0
        fi
    else
        echo -e "${BLUE}ℹ️  $name not running${NC}"
        return 0
    fi
}

# Stop services in reverse order
echo "Stopping services..."

# Stop Telegram bot
stop_process "telegram_ctl.py" "Telegram notification bot"

# Stop frontend/dashboard
stop_process "frontend.app" "Web dashboard"

# Stop main trading bot
stop_process "crypto_bot.main" "Trading bot"

# Stop integrated startup script
stop_process "start_bot_auto.py" "Integrated startup script"

# Stop any remaining Python processes related to the bot
if pgrep -f "python.*crypto_bot\|python.*frontend\|python.*telegram" > /dev/null 2>&1; then
    echo -e "${YELLOW}Stopping remaining bot processes...${NC}"
    pkill -f "python.*crypto_bot" || true
    pkill -f "python.*frontend" || true
    pkill -f "python.*telegram" || true
    sleep 2
fi

# Clean up PID files
echo -e "${BLUE}Cleaning up PID files...${NC}"
rm -f *.pid 2>/dev/null || true

# Verify all services are stopped
echo ""
echo -e "${BLUE}Verifying shutdown...${NC}"

SERVICES_STOPPED=true
if is_running "crypto_bot.main"; then
    echo -e "${RED}❌ Trading bot still running${NC}"
    SERVICES_STOPPED=false
fi

if is_running "frontend.app"; then
    echo -e "${RED}❌ Web dashboard still running${NC}"
    SERVICES_STOPPED=false
fi

if is_running "telegram_ctl.py"; then
    echo -e "${RED}❌ Telegram bot still running${NC}"
    SERVICES_STOPPED=false
fi

if is_running "start_bot_auto.py"; then
    echo -e "${RED}❌ Startup script still running${NC}"
    SERVICES_STOPPED=false
fi

if $SERVICES_STOPPED; then
    echo -e "${GREEN}✅ All services stopped successfully${NC}"
    echo ""
    echo -e "${GREEN}LegacyCoinTrader system shutdown complete${NC}"
    echo "To restart: ./start_integrated.sh"
else
    echo -e "${RED}❌ Some services may still be running${NC}"
    echo "Check with: ps aux | grep -E '(crypto_bot|frontend|telegram)'"
    echo "Or force kill: ./stop_integrated.sh"
    exit 1
fi