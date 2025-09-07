#!/bin/bash

# Unified Browser Opening Utility for LegacyCoinTrader
# This script provides consistent browser opening functionality across all startup scripts

# Colors for output (matching other scripts)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[BROWSER]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[BROWSER]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[BROWSER]${NC} $1"
}

print_error() {
    echo -e "${RED}[BROWSER]${NC} $1"
}

# Function to open browser (cross-platform)
open_browser() {
    local url="$1"
    local delay="${2:-2}"

    print_status "Waiting $delay seconds for services to fully start..."
    sleep "$delay"

    # Verify services are running by checking if URL is accessible
    print_status "Verifying dashboard is accessible..."
    if command -v curl >/dev/null 2>&1; then
        if curl -s --max-time 5 "$url" > /dev/null 2>&1; then
            print_success "Web dashboard is responding at $url"
        else
            print_warning "Web dashboard may not be fully ready yet at $url"
        fi
    fi

    # Detect OS and open appropriate browser
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        print_status "Opening browser on macOS..."
        if command -v open >/dev/null 2>&1; then
            open "$url" 2>/dev/null && print_success "Browser opened successfully"
        else
            print_warning "Could not automatically open browser. Please manually navigate to: $url"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        print_status "Opening browser on Linux..."
        local browser_opened=false
        if command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$url" 2>/dev/null && browser_opened=true
        elif command -v gnome-open >/dev/null 2>&1; then
            gnome-open "$url" 2>/dev/null && browser_opened=true
        elif command -v kde-open >/dev/null 2>&1; then
            kde-open "$url" 2>/dev/null && browser_opened=true
        fi

        if [[ "$browser_opened" == "true" ]]; then
            print_success "Browser opened successfully"
        else
            print_warning "Could not automatically open browser. Please manually navigate to: $url"
        fi
    elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        print_status "Opening browser on Windows..."
        if command -v start >/dev/null 2>&1; then
            start "$url" 2>/dev/null && print_success "Browser opened successfully"
        else
            print_warning "Could not automatically open browser. Please manually navigate to: $url"
        fi
    else
        # Other/Unknown OS
        print_warning "Unknown operating system: $OSTYPE"
        print_warning "Please manually navigate to: $url"
    fi
}

# Function to start Flask app and detect port
start_flask_and_detect_port() {
    local temp_file
    temp_file=$(mktemp)

    print_status "Starting Flask web dashboard..."
    python3 -m frontend.app > "$temp_file" 2>&1 &
    local frontend_pid=$!

    # Wait for Flask to start and write port info
    print_status "Waiting for Flask to initialize..."
    sleep 3

    # Extract the port from the Flask output
    local flask_port=8000  # default fallback
    if grep -q "FLASK_PORT=" "$temp_file"; then
        flask_port=$(grep "FLASK_PORT=" "$temp_file" | cut -d'=' -f2)
        print_success "Detected Flask port: $flask_port"
    else
        print_warning "Could not detect Flask port from output, using default: $flask_port"
    fi

    # Clean up temp file
    rm -f "$temp_file"

    # Return values
    echo "$frontend_pid:$flask_port"
}

# If this script is called directly, show usage
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Unified Browser Opening Utility for LegacyCoinTrader"
    echo ""
    echo "Usage:"
    echo "  source $0  # Source this file to use the functions"
    echo ""
    echo "Available functions:"
    echo "  open_browser <url> [delay_seconds]  # Open browser to URL after delay"
    echo "  start_flask_and_detect_port         # Start Flask and return pid:port"
    echo ""
    echo "Example:"
    echo "  source browser_util.sh"
    echo "  result=\$(start_flask_and_detect_port)"
    echo "  IFS=':' read -r frontend_pid flask_port <<< \"\$result\""
    echo "  open_browser \"http://localhost:\$flask_port\" 2"
fi
