#!/bin/bash
# Simple wrapper for the comprehensive shutdown system

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/safe_shutdown.py"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 LegacyCoinTrader Comprehensive Shutdown${NC}"
echo -e "${BLUE}$(printf '%.0s=' {1..45})${NC}"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}❌ Shutdown script not found: $PYTHON_SCRIPT${NC}"
    exit 1
fi

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
if [ -d "$SCRIPT_DIR/venv" ]; then
    echo -e "${BLUE}🐍 Using virtual environment${NC}"
    source "$SCRIPT_DIR/venv/bin/activate"
elif [ -d "$SCRIPT_DIR/modern_trader_env" ]; then
    echo -e "${BLUE}🐍 Using virtual environment${NC}"
    source "$SCRIPT_DIR/modern_trader_env/bin/activate"
fi

# Execute the Python shutdown script with all passed arguments
echo -e "${YELLOW}🚀 Executing shutdown...${NC}"
exec "$PYTHON_CMD" "$PYTHON_SCRIPT" "$@"
