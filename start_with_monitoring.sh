#!/bin/bash
# Complete startup script for LegacyCoinTrader with monitoring
# This script starts the bot, monitoring system, and frontend dashboard

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HYBRID_MODE=${HYBRID_MODE_ENABLED:-false}
READ_STRATEGY=${HYBRID_READ_STRATEGY:-compare}
WRITE_STRATEGY=${DUAL_WRITE_STRATEGY:-mirror}
CUTOVER_FLAG=${CUTOVER_COMPLETED:-false}
DRIFT_TOLERANCE=${MIGRATION_DRIFT_TOLERANCE:-0.0}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hybrid)
            HYBRID_MODE=true
            ;;
        --read-strategy)
            READ_STRATEGY="$2"
            shift || true
            ;;
        --write-strategy)
            WRITE_STRATEGY="$2"
            shift || true
            ;;
        --cutover-ready)
            CUTOVER_FLAG=true
            ;;
        --drift-tolerance)
            DRIFT_TOLERANCE="$2"
            shift || true
            ;;
    esac
    shift
done

if [[ "$HYBRID_MODE" == "true" ]]; then
    export HYBRID_MODE_ENABLED=true
    export HYBRID_READ_STRATEGY="$READ_STRATEGY"
    export DUAL_WRITE_STRATEGY="$WRITE_STRATEGY"
    export CUTOVER_COMPLETED=$([[ "$CUTOVER_FLAG" == "true" ]] && echo "true" || echo "false")
    export CUTOVER_GUARDRAIL_ENABLED=true
    export MIGRATION_DRIFT_TOLERANCE="$DRIFT_TOLERANCE"
    if [[ "$READ_STRATEGY" == "prefer-legacy" ]]; then
        export LEGACY_READ_ENABLED=true
    elif [[ "$READ_STRATEGY" == "prefer-modern" ]]; then
        export LEGACY_READ_ENABLED=false
    fi
    export HYBRID_FEATURE_FLAG_SOURCE="start_with_monitoring.sh"
fi

echo -e "${BLUE}🚀 Starting LegacyCoinTrader with Complete Monitoring${NC}"
echo -e "${BLUE}$(printf '%.0s=' {1..60})${NC}"
if [[ "$HYBRID_MODE" == "true" ]]; then
    echo -e "${GREEN}🤝 Hybrid telemetry mode enabled${NC}"
fi

# Function to check if process is running
check_process() {
    local pid_file="$1"
    local process_name="$2"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $process_name is running (PID: $pid)${NC}"
            return 0
        else
            echo -e "${RED}❌ $process_name process not found (stale PID: $pid)${NC}"
            rm -f "$pid_file"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️  $process_name PID file not found${NC}"
        return 1
    fi
}

# Check if we're already running
echo "🔍 Checking for existing processes..."

if check_process "bot_pid.txt" "Trading Bot"; then
    echo -e "${YELLOW}⚠️  Trading bot appears to already be running${NC}"
    echo "   Use './stop_monitoring.sh' to stop all processes first"
    exit 1
fi

# Start the complete system
echo ""
echo -e "${GREEN}🚀 Starting all systems...${NC}"

# Start bot with monitoring (this will start both bot and monitoring)
echo "🤖 Starting trading bot with monitoring..."
python3 start_bot.py auto &
BOT_PID=$!

# Wait for bot to initialize
sleep 3

# Start the frontend Flask app
echo "🌐 Starting web frontend..."
python3 -m frontend.app &
FRONTEND_PID=$!

# Save frontend PID
echo $FRONTEND_PID > frontend.pid

# Wait a moment for systems to initialize
sleep 5

echo ""
echo -e "${GREEN}📊 System Status:${NC}"

# Check all processes
check_process "bot_pid.txt" "Trading Bot" || true
check_process "monitoring.pid" "Monitoring System" || true
check_process "health_check.pid" "Health Check" || true
check_process "frontend.pid" "Web Frontend" || true

echo ""
echo -e "${GREEN}🌐 Access Points:${NC}"
echo -e "  📊 Monitoring Dashboard: ${BLUE}http://localhost:8000/monitoring${NC}"
echo -e "  📋 System Logs: ${BLUE}http://localhost:8000/system_logs${NC}"
echo -e "  🏠 Main Dashboard: ${BLUE}http://localhost:8000${NC}"

echo ""
echo -e "${GREEN}📁 Log Files:${NC}"
echo "  • Bot logs: crypto_bot/logs/bot.log"
echo "  • Monitoring: crypto_bot/logs/pipeline_monitor.log"
echo "  • Health checks: crypto_bot/logs/health_check.log"
echo "  • Recovery actions: crypto_bot/logs/recovery_actions.log"

echo ""
echo -e "${YELLOW}💡 Management Commands:${NC}"
echo "  • Check status: ./check_monitoring_status.sh"
echo "  • Stop all: ./stop_monitoring.sh"
echo "  • View logs: tail -f crypto_bot/logs/bot.log"

echo ""
echo -e "${GREEN}✅ LegacyCoinTrader with monitoring is now running!${NC}"
echo -e "${BLUE}$(printf '%.0s=' {1..60})${NC}"

# Keep running and monitor
echo ""
echo -e "${YELLOW}🔄 Monitoring system health...${NC}"
echo "Press Ctrl+C to stop all systems"

# Monitor loop
while true; do
    # Check if main bot process is still running
    if ! ps -p $BOT_PID > /dev/null 2>&1; then
        echo -e "${RED}❌ Main bot process has stopped${NC}"
        break
    fi

    # Check if frontend process is still running
    if ! ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo -e "${RED}❌ Frontend process has stopped${NC}"
        break
    fi

    # Brief status update every 30 seconds
    sleep 30
    echo -e "${BLUE}🔄 System check at $(date '+%H:%M:%S')${NC}"

    # Quick health check
    if [ -f "crypto_bot/logs/health_check_report.json" ]; then
        last_check=$(stat -f%B "crypto_bot/logs/health_check_report.json" 2>/dev/null || echo "0")
        seconds_ago=$(( $(date +%s) - last_check ))

        if [ $seconds_ago -gt 300 ]; then
            echo -e "${YELLOW}⚠️  Health check is ${seconds_ago}s old${NC}"
        fi
    fi
done

echo ""
echo -e "${YELLOW}🛑 Main bot process stopped, shutting down monitoring...${NC}"

# Cleanup
./stop_monitoring.sh

echo -e "${GREEN}✅ All systems stopped${NC}"
