#!/usr/bin/env bash

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$PROJECT_ROOT"

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root"
        exit 1
    fi
}

install_systemd_service() {
    info "Installing systemd service..."

    local service_file="$PROJECT_ROOT/monitoring.service"
    local systemd_dir="$HOME/.config/systemd/user"

    # Create systemd user directory if it doesn't exist
    mkdir -p "$systemd_dir"

    # Copy service file
    cp "$service_file" "$systemd_dir/"

    # Enable and start service
    systemctl --user daemon-reload
    systemctl --user enable monitoring.service
    systemctl --user start monitoring.service

    success "Systemd service installed and started"
    info "Service status: systemctl --user status monitoring.service"
}

install_cron_job() {
    info "Installing cron job for health monitoring..."

    local cron_job="*/5 * * * * /bin/bash $PROJECT_ROOT/health-check-cron.sh"
    local current_cron

    # Get current crontab
    current_cron=$(crontab -l 2>/dev/null || true)

    # Check if our job is already installed
    if echo "$current_cron" | grep -q "$PROJECT_ROOT/health-check-cron.sh"; then
        warn "Cron job already exists"
        return 0
    fi

    # Add our job to crontab
    if [[ -z "$current_cron" ]]; then
        echo "$cron_job" | crontab -
    else
        (echo "$current_cron"; echo "$cron_job") | crontab -
    fi

    success "Cron job installed (runs every 5 minutes)"
}

install_supervisor_config() {
    info "Setting up supervisor configuration..."

    if ! command -v supervisord >/dev/null 2>&1; then
        warn "Supervisor not installed. Install with: pip install supervisor"
        return 1
    fi

    local supervisor_dir="$HOME/.supervisor"
    local config_file="$supervisor_dir/legacycointrader.conf"

    mkdir -p "$supervisor_dir"
    cp "$PROJECT_ROOT/legacycointrader.conf" "$config_file"

    # Update paths in config file
    sed -i.bak "s|/Users/brandonburnette/Downloads/LegacyCoinTrader1.0|$PROJECT_ROOT|g" "$config_file"

    info "Supervisor config created at: $config_file"
    info "Start with: supervisord -c $config_file"
    success "Supervisor configuration ready"
}

make_scripts_executable() {
    info "Making scripts executable..."

    chmod +x "$PROJECT_ROOT/auto-startup.sh"
    chmod +x "$PROJECT_ROOT/shutdown.sh"
    chmod +x "$PROJECT_ROOT/health-check-cron.sh"
    chmod +x "$PROJECT_ROOT/startup.sh"

    success "Scripts are now executable"
}

setup_logging() {
    info "Setting up logging directories..."

    mkdir -p "$PROJECT_ROOT/logs"

    # Create log files if they don't exist
    touch "$PROJECT_ROOT/logs/supervisor.log"
    touch "$PROJECT_ROOT/logs/health-monitor.log"
    touch "$PROJECT_ROOT/startup.log"
    touch "$PROJECT_ROOT/health-check.log"

    success "Logging directories and files created"
}

create_environment_file() {
    info "Checking for environment configuration..."

    local env_file="$PROJECT_ROOT/.env"

    if [[ ! -f "$env_file" ]]; then
        warn "No .env file found. Creating template..."

        cat > "$env_file" << 'EOF'
# LegacyCoinTrader Environment Configuration
# Copy this file and update with your actual values

# API Keys (Required)
BITQUERY_KEY=your_bitquery_key_here
MORALIS_KEY=your_moralis_key_here
PYTH_API_KEY=your_pyth_key_here

# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here
POSTGRES_DB=legacycointrader

# Redis
REDIS_URL=redis://localhost:6379

# Email alerts (Optional)
ALERT_EMAIL=your_email@example.com

# Service URLs (usually don't need to change)
MARKET_DATA_SERVICE_URL=http://localhost:8002
TRADING_ENGINE_SERVICE_URL=http://localhost:8001
PORTFOLIO_SERVICE_URL=http://localhost:8003
EXECUTION_SERVICE_URL=http://localhost:8006

# Logging
LOG_LEVEL=INFO
LOG_FILE=/Users/brandonburnette/Downloads/LegacyCoinTrader1.0-1/logs/app.log
EOF

        warn "Please edit $env_file with your actual API keys and configuration"
    else
        success "Environment file already exists"
    fi
}

test_services() {
    info "Testing service availability..."

    # Test Docker
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker is not installed or not in PATH"
        return 1
    fi

    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon is not running"
        return 1
    fi

    success "Docker is available"

    # Test Docker Compose
    if docker compose version >/dev/null 2>&1; then
        success "Docker Compose v2 is available"
    elif command -v docker-compose >/dev/null 2>&1; then
        success "Docker Compose v1 is available"
    else
        error "Docker Compose is not available"
        return 1
    fi

    return 0
}

show_usage() {
    cat << 'EOF'
LegacyCoinTrader Auto-Start Setup
================================

This script sets up automatic startup and monitoring for LegacyCoinTrader services.

OPTIONS:
  --systemd     Install systemd user service (recommended for Linux)
  --cron        Install cron job for health monitoring
  --supervisor  Setup supervisor configuration
  --all         Install all available options
  --test        Test service availability
  --help        Show this help

USAGE EXAMPLES:
  ./setup-autostart.sh --all          # Install everything
  ./setup-autostart.sh --systemd      # Just systemd service
  ./setup-autostart.sh --cron         # Just cron monitoring
  ./setup-autostart.sh --test         # Test setup

MANUAL STARTUP:
  ./auto-startup.sh                   # Start with health monitoring
  ./startup.sh start                  # Basic startup
  ./docker-manager.sh start dev       # Advanced management

MONITORING:
  ./health-check-cron.sh              # Manual health check
  ./docker-manager.sh health          # Comprehensive health check

EOF
}

main() {
    check_root

    case "${1:-help}" in
        --systemd)
            make_scripts_executable
            setup_logging
            create_environment_file
            install_systemd_service
            ;;
        --cron)
            make_scripts_executable
            setup_logging
            create_environment_file
            install_cron_job
            ;;
        --supervisor)
            make_scripts_executable
            setup_logging
            create_environment_file
            install_supervisor_config
            ;;
        --all)
            make_scripts_executable
            setup_logging
            create_environment_file
            test_services

            if command -v systemctl >/dev/null 2>&1; then
                install_systemd_service
            else
                warn "Systemd not available, skipping service installation"
            fi

            install_cron_job
            install_supervisor_config
            ;;
        --test)
            test_services
            ;;
        --help|help|-h)
            show_usage
            ;;
        *)
            error "Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac

    if [[ $# -eq 0 ]]; then
        show_usage
    fi
}

main "$@"
