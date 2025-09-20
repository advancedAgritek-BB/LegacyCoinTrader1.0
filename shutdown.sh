#!/usr/bin/env bash

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$PROJECT_ROOT/docker-compose.yml}"

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

ensure_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        warn "Docker not available - assuming services already stopped"
        return 0
    fi
}

compose() {
    # Use the specific Docker Compose command format for this project
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml "$@"
    else
        warn "docker-compose not available"
        return 1
    fi
}

stop_trading_engine() {
    info "Stopping trading engine..."
    if command -v python3 >/dev/null 2>&1; then
        PYTHONWARNINGS="ignore:urllib3 v2 only supports OpenSSL 1.1.1+" \
            python3 -m crypto_bot.main stop 2>/dev/null || true
    fi
}

graceful_shutdown() {
    ensure_docker

    info "Initiating graceful shutdown of LegacyCoinTrader services..."

    # Stop trading engine first
    stop_trading_engine

    # Give services time to finish current operations
    info "Waiting for services to complete current operations..."
    sleep 5

    # Stop Docker services
    if compose -f "$COMPOSE_FILE" ps -q | grep -q . 2>/dev/null; then
        info "Stopping Docker services..."
        compose -f "$COMPOSE_FILE" down --timeout 30
        success "Docker services stopped"
    else
        info "No Docker services running"
    fi

    # Clean up any orphaned containers
    if command -v docker >/dev/null 2>&1; then
        orphaned=$(docker ps -aq --filter "label=com.docker.compose.project=legacycointrader10-1" 2>/dev/null || true)
        if [[ -n "$orphaned" ]]; then
            warn "Cleaning up orphaned containers..."
            docker rm -f $orphaned 2>/dev/null || true
        fi
    fi

    success "LegacyCoinTrader shutdown complete"
}

# Handle SIGTERM gracefully
trap graceful_shutdown SIGTERM SIGINT

# Main shutdown logic
case "${1:-}" in
    force)
        info "Force shutdown requested"
        ensure_docker
        compose -f "$COMPOSE_FILE" down --timeout 5 --remove-orphans 2>/dev/null || true
        success "Force shutdown complete"
        ;;
    *)
        graceful_shutdown
        ;;
esac