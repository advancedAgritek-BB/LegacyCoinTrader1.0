#!/bin/bash

# LegacyCoinTrader Startup Script
# This script sets up and launches the LegacyCoinTrader application
# Compatible with macOS (darwin) and Linux systems

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check OS
check_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        print_status "Detected macOS"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        print_status "Detected Linux"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Function to install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    if [[ "$OS" == "macos" ]]; then
        # Check if Homebrew is installed
        if ! command_exists brew; then
            print_status "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        else
            print_status "Homebrew already installed"
        fi
        
        # Install Python and other dependencies via Homebrew
        print_status "Installing Python and dependencies via Homebrew..."
        brew install python@3.11 git curl
        
    elif [[ "$OS" == "linux" ]]; then
        if command_exists apt-get; then
            print_status "Installing dependencies via apt..."
            sudo apt-get update -y
            sudo apt-get install -y --no-install-recommends git curl python3 python3-pip python3-venv
        elif command_exists yum; then
            print_status "Installing dependencies via yum..."
            sudo yum update -y
            sudo yum install -y git curl python3 python3-pip python3-venv
        else
            print_warning "Package manager not detected. Please install Python 3.11+ manually."
        fi
    fi
}

# Function to setup Python virtual environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    if [[ ! -d "venv" ]]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    else
        print_status "Virtual environment already exists"
    fi
    
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    print_status "Upgrading pip..."
    python -m pip install --upgrade pip
    
    print_status "Installing Python dependencies..."
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    else
        print_warning "requirements.txt not found, installing basic dependencies..."
        pip install ccxt pandas numpy requests websockets asyncio-mqtt python-telegram-bot
    fi
    
    if [[ -f "requirements_gpu.txt" ]]; then
        if [[ "$OS" == "macos" ]]; then
            print_warning "Skipping GPU dependencies on macOS (CUDA not supported)"
            print_status "GPU acceleration will not be available on this platform"
            
            # Install macOS-optimized dependencies instead
            if [[ -f "requirements_macos.txt" ]]; then
                print_status "Installing macOS-optimized performance dependencies..."
                pip install -r requirements_macos.txt
            fi
        else
            print_status "Installing GPU-accelerated dependencies..."
            pip install -r requirements_gpu.txt
        fi
    fi
}

# Function to check environment configuration
check_env() {
    print_status "Checking environment configuration..."
    
    # Use the Python utility to manage environment files
    if command_exists python3; then
        print_status "Using Python utility to manage environment files..."
        if python3 tools/manage_env.py consolidate; then
            print_success "Environment files consolidated successfully"
            return 0
        else
            print_error "Failed to consolidate environment files"
            exit 1
        fi
    else
        print_warning "Python3 not found, using fallback environment check..."
        # Fallback to the old method if Python is not available
        _fallback_env_check
    fi
}

# Fallback environment check method
_fallback_env_check() {
    # Check for .env files in multiple locations
    ENV_LOCATIONS=(".env" "crypto_bot/.env")
    EXISTING_ENV=""
    
    for env_path in "${ENV_LOCATIONS[@]}"; do
        if [[ -f "$env_path" ]]; then
            print_status "Found .env file at: $env_path"
            
            # Check if this .env file contains real API keys (not template values)
            if grep -q "your_kraken_api_key_here\|your_telegram_token_here\|your_helius_key_here" "$env_path"; then
                print_warning "Found template .env file at $env_path (contains placeholder values)"
            else
                print_success "Found .env file with real API keys at $env_path"
                EXISTING_ENV="$env_path"
                break
            fi
        fi
    done
    
    # If we found a real .env file, we're good to go
    if [[ -n "$EXISTING_ENV" ]]; then
        print_success "Using existing .env file: $EXISTING_ENV"
        return 0
    fi
    
    # If we found only template files, ask user what to do
    if [[ -f ".env" ]] || [[ -f "crypto_bot/.env" ]]; then
        print_warning "Found .env file(s) but they contain template values!"
        echo
        echo "You have .env file(s) with placeholder values. You need to:"
        echo "1. Edit the .env file(s) with your real API keys"
        echo "2. Run this script again"
        echo
        echo "Current .env locations:"
        for env_path in "${ENV_LOCATIONS[@]}"; do
            if [[ -f "$env_path" ]]; then
                echo "  - $env_path"
            fi
        done
        echo
        read -p "Press Enter to exit so you can edit your .env file(s)..."
        exit 1
    fi
    
    # No .env files found anywhere, create a new one in the root directory
    print_warning "No .env file found anywhere!"
    print_status "Creating .env file from template in root directory..."
    
    cat > .env << 'EOF'
# .env File for LegacyCoinTrader
# Updated with actual API keys and configuration

# Exchange Configuration
EXCHANGE=kraken
API_KEY=your_kraken_api_key_here
API_SECRET=your_kraken_api_secret_here
KRAKEN_API_KEY=your_kraken_api_key_here
KRAKEN_API_SECRET=your_kraken_api_secret_here

# Alternative Exchange (Coinbase)
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_API_SECRET=your_coinbase_api_secret_here
# COINBASE_PASSPHRASE=your_coinbase_passphrase_here

# Telegram Configuration
TELEGRAM_TOKEN=your_telegram_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TELE_CHAT_ADMINS=your_admin_chat_id_here

# Solana Configuration
HELIUS_KEY=your_helius_key_here
WALLET_ADDRESS=your_wallet_address_here
SOLANA_PRIVATE_KEY=your_solana_private_key_here

# Supabase Configuration
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here

# LunarCrush Sentiment Analysis (Optional)
LUNARCRUSH_API_KEY=your_lunarcrush_api_key_here

# Trading Mode
MODE=cex
EXECUTION_MODE=dry_run

# CloudTrader Configuration
CT_MODELS_BUCKET=models
CT_REGIME_PREFIX=
CT_SYMBOL=XRPUSD
EOF
        
        print_warning "Please edit .env file with your actual API keys before running the bot!"
        print_status "You can now edit the .env file and run this script again."
        print_status "Note: The application expects .env files in either the root directory or crypto_bot/ directory."
        exit 1
}

# Function to run tests
run_tests() {
    print_status "Running test suite..."
    
    if command_exists pytest; then
        print_status "Running pytest..."
        python -m pytest -q
    else
        print_warning "pytest not found, skipping tests"
    fi
}

# Function to start the application
start_application() {
    print_status "Starting LegacyCoinTrader..."
    
    # Check if we're in dry run mode
    if grep -q "EXECUTION_MODE=dry_run" .env; then
        print_warning "Running in DRY RUN mode - no real trades will be executed"
    else
        print_warning "Running in LIVE mode - real trades will be executed!"
    fi
    
    print_status "Starting main application..."
    python -m crypto_bot.main &
    MAIN_PID=$!
    
    print_status "Starting web frontend..."
    python -m frontend.app &
    FRONTEND_PID=$!
    
    print_status "Starting Telegram bot..."
    python telegram_ctl.py &
    TELEGRAM_PID=$!
    
    print_success "LegacyCoinTrader started successfully!"
    print_status "Main application PID: $MAIN_PID"
    print_status "Frontend PID: $FRONTEND_PID"
    print_status "Telegram bot PID: $TELEGRAM_PID"
    print_status "Web dashboard available at: http://localhost:5000"
    print_status "Use 'ps aux | grep python' to see running processes"
    print_status "Use 'kill $MAIN_PID $FRONTEND_PID $TELEGRAM_PID' to stop all services"
    
    # Wait for user input to stop
    echo
    read -p "Press Enter to stop all services..."
    
    print_status "Stopping all services..."
    kill $MAIN_PID $FRONTEND_PID $TELEGRAM_PID 2>/dev/null || true
    print_success "All services stopped"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTION]"
    echo
    echo "Options:"
    echo "  setup     - Install dependencies and setup environment"
    echo "  test      - Run test suite"
    echo "  start     - Start the application"
    echo "  full      - Full setup and start (default)"
    echo "  help      - Show this help message"
    echo
    echo "Examples:"
    echo "  $0 setup    # Only setup dependencies"
    echo "  $0 test     # Only run tests"
    echo "  $0 start    # Only start application (assumes setup is complete)"
    echo "  $0          # Full setup and start"
}

# Main execution
main() {
    print_status "LegacyCoinTrader Startup Script"
    print_status "================================="
    
    # Parse command line arguments
    case "${1:-full}" in
        "setup")
            check_os
            install_system_deps
            setup_python_env
            check_env
            print_success "Setup completed successfully!"
            ;;
        "test")
            source venv/bin/activate 2>/dev/null || print_error "Virtual environment not found. Run setup first."
            run_tests
            ;;
        "start")
            source venv/bin/activate 2>/dev/null || print_error "Virtual environment not found. Run setup first."
            check_env
            start_application
            ;;
        "full")
            check_os
            install_system_deps
            setup_python_env
            check_env
            run_tests
            start_application
            ;;
        "help"|"-h"|"--help")
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
