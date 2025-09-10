#!/bin/bash
# Stop Monitoring System and all related processes

WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 Stopping LegacyCoinTrader Monitoring System${NC}"
echo -e "${BLUE}$(printf '%.0s=' {1..50})${NC}"

# Function to stop process by PID file
stop_process() {
    local pid_file="$1"
    local process_name="$2"

    echo -n "Stopping $process_name: "

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo -n "Terminating PID $pid... "
            kill $pid 2>/dev/null || true
            sleep 2

            # Check if it's still running
            if ps -p $pid > /dev/null 2>&1; then
                echo -n "Force killing... "
                kill -9 $pid 2>/dev/null || true
                sleep 1
            fi

            if ps -p $pid > /dev/null 2>&1; then
                echo -e "${RED}❌ Failed to stop${NC}"
                return 1
            else
                echo -e "${GREEN}✅ Stopped${NC}"
                rm -f "$pid_file"
                return 0
            fi
        else
            echo -e "${YELLOW}⚠️  Process not running (removing stale PID file)${NC}"
            rm -f "$pid_file"
            return 0
        fi
    else
        echo -e "${YELLOW}⚠️  PID file not found${NC}"
        return 0
    fi
}

# Stop all processes
echo "Stopping processes..."
stop_process "bot_pid.txt" "Trading Bot"
stop_process "monitoring.pid" "Monitoring System"
stop_process "health_check.pid" "Health Check"
stop_process "frontend.pid" "Web Frontend"

echo ""
echo "🧹 Cleaning up any remaining processes..."

# Kill any remaining monitoring processes
pkill -f "enhanced_monitoring.py" 2>/dev/null || true
pkill -f "auto_health_check.py" 2>/dev/null || true
pkill -f "frontend.app" 2>/dev/null || true
pkill -f "crypto_bot.main" 2>/dev/null || true

# Clean up any stale PID files
rm -f bot_pid.txt monitoring.pid health_check.pid frontend.pid

echo ""
echo -e "${GREEN}✅ All systems stopped successfully${NC}"
echo -e "${GREEN}To restart: ./start_with_monitoring.sh${NC}"
