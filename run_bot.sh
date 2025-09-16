#!/bin/bash
# Interactive Bot Runner for LegacyCoinTrader
# This script starts the bot and allows you to stop it with Ctrl+C or Enter

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_PID=""
SHUTDOWN_REQUESTED=false

# Function to handle shutdown
shutdown_bot() {
    local reason="$1"
    
    if [ "$SHUTDOWN_REQUESTED" = true ]; then
        return
    fi
    
    SHUTDOWN_REQUESTED=true
    
    echo -e "\n${YELLOW}🛑 Shutdown requested: $reason${NC}"
    echo -e "${BLUE}🔄 Initiating safe shutdown...${NC}"
    
    if [ -n "$BOT_PID" ] && kill -0 "$BOT_PID" 2>/dev/null; then
        echo -e "${BLUE}📤 Sending SIGTERM to bot (PID: $BOT_PID)...${NC}"
        kill -TERM "$BOT_PID" 2>/dev/null || true
        
        # Wait for graceful shutdown
        local count=0
        while [ $count -lt 10 ] && kill -0 "$BOT_PID" 2>/dev/null; do
            sleep 1
            count=$((count + 1))
            echo -ne "${BLUE}⏳ Waiting for graceful shutdown... ($count/10)\r${NC}"
        done
        echo ""
        
        if kill -0 "$BOT_PID" 2>/dev/null; then
            echo -e "${YELLOW}⏰ Graceful shutdown timeout, force killing...${NC}"
            kill -KILL "$BOT_PID" 2>/dev/null || true
            sleep 1
            echo -e "${RED}💀 Bot force killed${NC}"
        else
            echo -e "${GREEN}✅ Bot stopped gracefully${NC}"
        fi
    fi
    
    # Clean up PID file
    if [ -f "$WORKING_DIR/bot_pid.txt" ]; then
        rm -f "$WORKING_DIR/bot_pid.txt"
        echo -e "${BLUE}🧹 PID file cleaned up${NC}"
    fi
    
    echo -e "${GREEN}👋 Shutdown complete${NC}"
    exit 0
}

# Signal handlers
trap 'shutdown_bot "Ctrl+C"' INT
trap 'shutdown_bot "SIGTERM"' TERM
trap 'shutdown_bot "SIGHUP"' HUP 2>/dev/null || true

# Function to start input monitor in background
start_input_monitor() {
    (
        while [ "$SHUTDOWN_REQUESTED" != true ]; do
            if read -r user_input; then
                case "${user_input,,}" in
                    ""|"quit"|"exit"|"stop"|"shutdown")
                        if [ -z "$user_input" ]; then
                            echo "🛑 Enter key detected - requesting shutdown..."
                        else
                            echo "🛑 Command '$user_input' detected - requesting shutdown..."
                        fi
                        kill -USR1 $$ 2>/dev/null || kill -TERM $$ 2>/dev/null
                        break
                        ;;
                    "help"|"h"|"?")
                        echo ""
                        echo "📖 Available commands:"
                        echo "  <Enter>           - Safe shutdown"
                        echo "  quit, exit, stop  - Safe shutdown"
                        echo "  status            - Show bot status"
                        echo "  help              - Show this help"
                        echo "  Ctrl+C            - Emergency shutdown"
                        echo ""
                        ;;
                    "status")
                        if [ -n "$BOT_PID" ] && kill -0 "$BOT_PID" 2>/dev/null; then
                            echo "🟢 Bot Status: Running (PID: $BOT_PID)"
                        else
                            echo "🔴 Bot Status: Not running"
                        fi
                        ;;
                    *)
                        if [ -n "$user_input" ]; then
                            echo "❓ Unknown command: $user_input"
                            echo "💡 Press Enter to shutdown, or type 'help' for commands"
                        fi
                        ;;
                esac
            fi
        done
    ) &
    INPUT_MONITOR_PID=$!
}

# Handle custom signals for shutdown
trap 'shutdown_bot "Enter/Command"' USR1 2>/dev/null || true

echo -e "${BLUE}🎮 LegacyCoinTrader Interactive Runner${NC}"
echo -e "${BLUE}$(printf '%.0s=' {1..45})${NC}"
echo -e "${GREEN}💡 Interactive controls:${NC}"
echo -e "${GREEN}   • Press Ctrl+C for emergency shutdown${NC}"
echo -e "${GREEN}   • Press Enter for safe shutdown${NC}"
echo -e "${GREEN}   • Type 'help' for more commands${NC}"
echo -e "${BLUE}$(printf '%.0s=' {1..45})${NC}"

# Find Python executable
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Python not found. Please install Python 3.6+${NC}"
    exit 1
fi

# Check for virtual environment
if [ -d "$WORKING_DIR/venv" ]; then
    echo -e "${BLUE}🐍 Using virtual environment: venv${NC}"
    source "$WORKING_DIR/venv/bin/activate"
elif [ -d "$WORKING_DIR/modern_trader_env" ]; then
    echo -e "${BLUE}🐍 Using virtual environment: modern_trader_env${NC}"
    source "$WORKING_DIR/modern_trader_env/bin/activate"
fi

# Check if bot script exists
BOT_SCRIPT="$WORKING_DIR/crypto_bot/main.py"
if [ ! -f "$BOT_SCRIPT" ]; then
    echo -e "${RED}❌ Bot script not found: $BOT_SCRIPT${NC}"
    exit 1
fi

echo -e "${YELLOW}🚀 Starting LegacyCoinTrader bot...${NC}"

# Set environment
export PYTHONPATH="$WORKING_DIR/crypto_bot"

# Start the bot in background
"$PYTHON_CMD" "$BOT_SCRIPT" &
BOT_PID=$!

echo -e "${GREEN}✅ Bot started (PID: $BOT_PID)${NC}"
echo -e "${BLUE}📊 Bot output:${NC}"
echo -e "${BLUE}$(printf '%.0s-' {1..45})${NC}"

# Start input monitor
start_input_monitor

# Wait for bot to finish
wait $BOT_PID
BOT_EXIT_CODE=$?

# Clean up input monitor
if [ -n "$INPUT_MONITOR_PID" ]; then
    kill $INPUT_MONITOR_PID 2>/dev/null || true
fi

if [ "$BOT_EXIT_CODE" -eq 0 ]; then
    echo -e "\n${GREEN}🏁 Bot exited normally${NC}"
else
    echo -e "\n${YELLOW}🏁 Bot exited with code: $BOT_EXIT_CODE${NC}"
fi

# Final cleanup
shutdown_bot "Bot finished"
