#!/usr/bin/env bash

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$PROJECT_ROOT/docker-compose.yml}"
STACK_SERVICES=(redis api-gateway trading-engine market-data portfolio strategy-engine execution monitoring frontend)

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
        error "Docker is required to run this script."
        exit 1
    fi
}

compose() {
    if docker compose version >/dev/null 2>&1; then
        docker compose "$@"
    elif command -v docker-compose >/dev/null 2>&1; then
        docker-compose "$@"
    else
        error "Docker Compose is not available. Install Docker Compose v2 or docker-compose."
        exit 1
    fi
}

bootstrap_stack() {
    ensure_docker
    info "Pulling images for trading stack (${STACK_SERVICES[*]})"
    compose -f "$COMPOSE_FILE" pull "${STACK_SERVICES[@]}"
    success "Images pulled"
}

start_stack() {
    ensure_docker
    info "Starting LegacyCoinTrader services via Docker Compose"
    compose -f "$COMPOSE_FILE" up -d "${STACK_SERVICES[@]}"
    success "Services started"
    info "Gateway: http://localhost:8000"
    info "Frontend dashboard (optional): http://localhost:5000"
    info "Use 'python -m crypto_bot.main status' to inspect the trading engine scheduler."
    info "Trigger cycles with 'python -m crypto_bot.main start --interval 60'."
}

stop_stack() {
    ensure_docker
    info "Stopping LegacyCoinTrader services"
    compose -f "$COMPOSE_FILE" down
    success "Services stopped"
}

restart_stack() {
    stop_stack
    start_stack
}

stack_status() {
    ensure_docker
    info "Service status"
    compose -f "$COMPOSE_FILE" ps
}

stack_logs() {
    ensure_docker
    local services=("${STACK_SERVICES[@]}")
    if [[ $# -gt 0 ]]; then
        services=("$@")
    fi
    info "Tailing logs for: ${services[*]}"
    compose -f "$COMPOSE_FILE" logs -f "${services[@]}"
}

run_cli() {
    info "Executing trading engine CLI: python -m crypto_bot.main $*"
    PYTHONWARNINGS="ignore:urllib3 v2 only supports OpenSSL 1.1.1+" \
        python -m crypto_bot.main "$@"
}

usage() {
    cat <<USAGE
Usage: ./startup.sh <command>

Commands:
  bootstrap   Pull container images defined in $COMPOSE_FILE
  start       Start the microservices stack (Docker Compose)
  stop        Stop and remove stack containers
  restart     Restart the stack
  status      Show docker compose service status
  logs [svc]  Tail logs for services (default: ${STACK_SERVICES[*]})
  cli [args]  Run the trading engine CLI (python -m crypto_bot.main)
  help        Show this message

Examples:
  ./startup.sh start
  ./startup.sh cli status
  ./startup.sh cli start --interval 60 --no-immediate
  ./startup.sh logs trading-engine
USAGE
}

main() {
    local command="${1:-help}"
    shift || true

    case "$command" in
        bootstrap)
            bootstrap_stack
            ;;
        start)
            start_stack
            ;;
        stop)
            stop_stack
            ;;
        restart)
            restart_stack
            ;;
        status)
            stack_status
            ;;
        logs)
            stack_logs "$@"
            ;;
        cli)
            run_cli "$@"
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

main "$@"
