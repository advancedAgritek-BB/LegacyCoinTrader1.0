#!/bin/bash
# LegacyCoinTrader System Status Checker
# Shows the current status of all system components

# Note: Not using set -e since we want to continue checking even if services are stopped

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}📊 LegacyCoinTrader System Status${NC}"
echo -e "${BLUE}$(printf '%.0s=' {1..40})${NC}"

# Function to check if process is running
check_process() {
    local pattern="$1"
    local name="$2"
    local description="$3"

    if pgrep -f "$pattern" >/dev/null 2>&1; then
        local pid=$(pgrep -f "$pattern" | head -1)
        echo -e "${GREEN}✅ $name: RUNNING${NC} (PID: $pid) - $description"
        return 0
    else
        echo -e "${RED}❌ $name: STOPPED${NC} - $description"
        return 1
    fi
}

# Function to check web service
check_web_service() {
    local port="$1"
    local name="$2"
    local url="$3"

    if curl -s --max-time 3 "http://localhost:$port" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ $name: RESPONDING${NC} - $url"
        return 0
    else
        echo -e "${YELLOW}⚠️  $name: NOT RESPONDING${NC} - $url"
        return 1
    fi
}

echo ""
echo -e "${BLUE}🤖 Core Services:${NC}"

# Check trading bot (running as module)
if pgrep -f "python.*crypto_bot.main" >/dev/null 2>&1; then
    local pid=$(pgrep -f "python.*crypto_bot.main" | head -1)
    echo -e "${GREEN}✅ Trading Bot: RUNNING${NC} (PID: $pid) - Main trading engine with OHLCV fetching"
else
    echo -e "${RED}❌ Trading Bot: STOPPED${NC} - Main trading engine with OHLCV fetching"
fi

# Check web dashboard (running as module)
if pgrep -f "python.*frontend.app" >/dev/null 2>&1; then
    local pid=$(pgrep -f "python.*frontend.app" | head -1)
    echo -e "${GREEN}✅ Web Dashboard: RUNNING${NC} (PID: $pid) - Flask web server for dashboard"
else
    echo -e "${RED}❌ Web Dashboard: STOPPED${NC} - Flask web server for dashboard"
fi

# Check Telegram bot
if pgrep -f "python.*telegram" >/dev/null 2>&1; then
    local pid=$(pgrep -f "python.*telegram" | head -1)
    echo -e "${GREEN}✅ Telegram Bot: RUNNING${NC} (PID: $pid) - Notification and control bot"
else
    echo -e "${RED}❌ Telegram Bot: STOPPED${NC} - Notification and control bot"
fi

# Check startup script
check_process "start_bot_auto.py" "Startup Script" "Integrated system launcher"

echo ""
echo -e "${BLUE}🌐 Web Services:${NC}"

# Check main dashboard (common ports)
DASHBOARD_FOUND=false
for port in 8000 5000 8080; do
    if check_web_service "$port" "Dashboard (Port $port)" "http://localhost:$port"; then
        DASHBOARD_FOUND=true
        break
    fi
done

if ! $DASHBOARD_FOUND; then
    echo -e "${RED}❌ Dashboard: NOT FOUND${NC} - Check if web server is running"
fi

# Check for Flask port in logs if available
if [[ -f "frontend.log" ]]; then
    FLASK_PORT=$(grep -o "Running on http://[^:]*:\([0-9]*\)" frontend.log 2>/dev/null | tail -1 | grep -o "[0-9]*")
    if [[ -n "$FLASK_PORT" ]]; then
        check_web_service "$FLASK_PORT" "Dashboard (Log Port $FLASK_PORT)" "http://localhost:$FLASK_PORT"
    fi
fi

echo ""
echo -e "${BLUE}💾 System Resources:${NC}"

# Show Python processes
PYTHON_PROCESSES=$(pgrep -f "python.*crypto_bot\|python.*frontend\|python.*telegram" | wc -l)
if [[ $PYTHON_PROCESSES -gt 0 ]]; then
    echo -e "${GREEN}🐍 Python processes: $PYTHON_PROCESSES running${NC}"
    echo "   Details:"
    ps aux | head -1
    ps aux | grep -E "(crypto_bot|frontend|telegram)" | grep -v grep
else
    echo -e "${YELLOW}🐍 Python processes: None running${NC}"
fi

echo ""
echo -e "${BLUE}📁 Configuration:${NC}"

# Check .env file
if [[ -f ".env" ]]; then
    echo -e "${GREEN}✅ .env file: PRESENT${NC}"
    # Check for API keys (basic check)
    if grep -q "your_kraken_api_key_here\|your_telegram_token_here" .env 2>/dev/null; then
        echo -e "${YELLOW}⚠️  .env file: CONTAINS PLACEHOLDER VALUES${NC}"
    else
        echo -e "${GREEN}✅ .env file: CONFIGURED${NC}"
    fi
else
    echo -e "${RED}❌ .env file: MISSING${NC}"
fi

echo ""
echo -e "${BLUE}🔧 Quick Actions:${NC}"
echo "   Start system:  ./start_integrated.sh"
echo "   Stop system:   ./stop_integrated.sh"
echo "   Check logs:    tail -f bot_output.log"
echo "   View dashboard: Open http://localhost:8000 in browser"

echo ""
echo -e "${BLUE}$(printf '%.0s=' {1..40})${NC}"