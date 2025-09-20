#!/usr/bin/env bash

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$PROJECT_ROOT/docker-compose.yml}"
LOG_FILE="${PROJECT_ROOT}/startup.log"
HEALTH_CHECK_INTERVAL=30
MAX_RETRIES=3

# Service dependencies and startup order
SERVICES_ORDER=(
    "redis postgres"
    "api-gateway"
    "market-data portfolio"
    "trading-engine strategy-engine token-discovery"
    "execution monitoring"
    "frontend"
)

info() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') ${RED}[ERROR]${NC} $1" >&2 | tee -a "$LOG_FILE"
}

ensure_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker is required to run this script."
        exit 1
    fi

    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
}

compose() {
    # Use the specific Docker Compose command format for this project
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml "$@"
    else
        error "docker-compose is not available."
        exit 1
    fi
}

check_service_health() {
    local service=$1
    local port=$2
    local path=${3:-/health}
    local timeout=${4:-10}

    if [[ "$service" == "redis" ]]; then
        # Special handling for Redis
        if docker exec "$(compose ps -q "$service" 2>/dev/null)" redis-cli ping 2>/dev/null | grep -q PONG; then
            return 0
        fi
    elif [[ "$service" == "postgres" ]]; then
        # Special handling for PostgreSQL
        if docker exec "$(compose ps -q "$service" 2>/dev/null)" pg_isready -U postgres 2>/dev/null; then
            return 0
        fi
    else
        # HTTP health check
        if curl -s --max-time "$timeout" "http://localhost:${port}${path}" >/dev/null 2>&1; then
            return 0
        fi
    fi

    return 1
}

wait_for_service() {
    local service=$1
    local port=$2
    local path=${3:-/health}
    local timeout=${4:-30}

    info "Waiting for $service to be healthy (port $port)..."

    local start_time=$(date +%s)
    while true; do
        if check_service_health "$service" "$port" "$path"; then
            success "$service is healthy"
            return 0
        fi

        local elapsed=$(( $(date +%s) - start_time ))
        if [[ $elapsed -gt $timeout ]]; then
            error "$service failed to become healthy within ${timeout}s"
            return 1
        fi

        sleep 2
    done
}

start_services_with_retry() {
    local phase=$1
    local services=$2
    local retry_count=0

    while [[ $retry_count -lt $MAX_RETRIES ]]; do
        info "Starting phase '$phase': $services (attempt $((retry_count + 1))/$MAX_RETRIES)"

        if compose -f "$COMPOSE_FILE" up -d $services; then
            # Wait for services to be healthy
            local all_healthy=true

            for service in $services; do
                case $service in
                    redis) wait_for_service "$service" 6379 "" 30 ;;
                    postgres) wait_for_service "$service" 5432 "" 30 ;;
                    api-gateway) wait_for_service "$service" 8000 "/health" 30 ;;
                    trading-engine) wait_for_service "$service" 8001 "/health" 30 ;;
                    market-data) wait_for_service "$service" 8002 "/health" 30 ;;
                    portfolio) wait_for_service "$service" 8003 "/health" 30 ;;
                    strategy-engine) wait_for_service "$service" 8004 "/health" 30 ;;
                    token-discovery) wait_for_service "$service" 8005 "/health" 30 ;;
                    execution) wait_for_service "$service" 8006 "/health" 30 ;;
                    monitoring) wait_for_service "$service" 8007 "/health" 30 ;;
                    frontend) wait_for_service "$service" 5000 "/health" 30 ;;
                esac

                if [[ $? -ne 0 ]]; then
                    all_healthy=false
                    break
                fi
            done

            if [[ $all_healthy == true ]]; then
                success "Phase '$phase' started successfully"
                return 0
            fi
        fi

        retry_count=$((retry_count + 1))
        if [[ $retry_count -lt $MAX_RETRIES ]]; then
            warn "Phase '$phase' failed, retrying in 10s..."
            compose -f "$COMPOSE_FILE" down $services 2>/dev/null || true
            sleep 10
        fi
    done

    error "Phase '$phase' failed after $MAX_RETRIES attempts"
    return 1
}

start_trading_engine() {
    info "Starting trading engine..."

    # Give services a moment to fully initialize
    sleep 5

    local retry_count=0
    while [[ $retry_count -lt $MAX_RETRIES ]]; do
        if python3 -m crypto_bot.main start --interval 60 >/dev/null 2>&1; then
            success "Trading engine started successfully"
            return 0
        fi

        retry_count=$((retry_count + 1))
        warn "Trading engine start failed (attempt $retry_count/$MAX_RETRIES)"
        sleep 5
    done

    error "Failed to start trading engine after $MAX_RETRIES attempts"
    return 1
}

health_monitor_loop() {
    info "Starting health monitoring loop (interval: ${HEALTH_CHECK_INTERVAL}s)"

    while true; do
        local unhealthy_services=()

        # Check all services
        for service_info in "${SERVICES_ORDER[@]}"; do
            for service in $service_info; do
                case $service in
                    redis) check_service_health "$service" 6379 "" || unhealthy_services+=("$service") ;;
                    postgres) check_service_health "$service" 5432 "" || unhealthy_services+=("$service") ;;
                    api-gateway) check_service_health "$service" 8000 "/health" || unhealthy_services+=("$service") ;;
                    trading-engine) check_service_health "$service" 8001 "/health" || unhealthy_services+=("$service") ;;
                    market-data) check_service_health "$service" 8002 "/health" || unhealthy_services+=("$service") ;;
                    portfolio) check_service_health "$service" 8003 "/health" || unhealthy_services+=("$service") ;;
                    strategy-engine) check_service_health "$service" 8004 "/health" || unhealthy_services+=("$service") ;;
                    token-discovery) check_service_health "$service" 8005 "/health" || unhealthy_services+=("$service") ;;
                    execution) check_service_health "$service" 8006 "/health" || unhealthy_services+=("$service") ;;
                    monitoring) check_service_health "$service" 8007 "/health" || unhealthy_services+=("$service") ;;
                    frontend) check_service_health "$service" 5000 "/health" || unhealthy_services+=("$service") ;;
                esac
            done
        done

        if [[ ${#unhealthy_services[@]} -gt 0 ]]; then
            warn "Unhealthy services detected: ${unhealthy_services[*]}"

            # Restart unhealthy services
            for service in "${unhealthy_services[@]}"; do
                warn "Restarting $service..."
                compose -f "$COMPOSE_FILE" restart "$service" || error "Failed to restart $service"
            done

            # Wait a bit before checking again
            sleep 30
        fi

        sleep $HEALTH_CHECK_INTERVAL
    done
}

main() {
    info "LegacyCoinTrader Auto-Startup Script"
    info "=================================="

    # Ensure Docker is available
    ensure_docker

    # Pull latest images
    info "Pulling latest Docker images..."
    compose -f "$COMPOSE_FILE" pull || warn "Failed to pull some images"

    # Start services in phases
    local phase_index=0
    for phase_services in "${SERVICES_ORDER[@]}"; do
        phase_index=$((phase_index + 1))
        if ! start_services_with_retry "phase_$phase_index" "$phase_services"; then
            error "Failed to start services in phase $phase_index"
            exit 1
        fi
    done

    # Start trading engine
    if ! start_trading_engine; then
        warn "Trading engine failed to start, but services are running"
    fi

    success "All services started successfully!"
    success "Dashboard: http://localhost:5000"
    success "API Gateway: http://localhost:8000"

    # Start health monitoring in background
    info "Starting background health monitoring..."
    health_monitor_loop &

    # Wait for termination signal
    info "System is running. Press Ctrl+C to stop."

    # Wait for background processes
    wait
}

# Handle signals gracefully
trap 'error "Received signal, shutting down..."; exit 0' INT TERM

# Run main function
main "$@"
