#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${PROJECT_ROOT}/health-check.log"
ALERT_EMAIL="${ALERT_EMAIL:-}"

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG_FILE"
}

info() {
    log "[INFO] $*"
    echo -e "${BLUE}[INFO]${NC} $*"
}

success() {
    log "[SUCCESS] $*"
}

warn() {
    log "[WARN] $*"
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    log "[ERROR] $*"
    echo -e "${RED}[ERROR]${NC} $*"
}

send_alert() {
    local subject="$1"
    local message="$2"

    log "Sending alert: $subject"

    if [[ -n "$ALERT_EMAIL" ]]; then
        echo "$message" | mail -s "$subject" "$ALERT_EMAIL" 2>/dev/null || true
    fi

    # Also send to system journal if available
    if command -v logger >/dev/null 2>&1; then
        logger -t legacycointrader -p user.warning "$subject: $message"
    fi
}

check_docker_running() {
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker command not found"
        return 1
    fi

    if ! docker info >/dev/null 2>&1; then
        error "Docker daemon not running"
        return 1
    fi

    return 0
}

check_service_health() {
    local service=$1
    local port=$2
    local path=${3:-/health}
    local timeout=${4:-5}

    case $service in
        redis)
            if docker exec "$(docker-compose -f docker-compose.yml -f docker-compose.dev.yml ps -q "$service" 2>/dev/null)" redis-cli ping 2>/dev/null | grep -q PONG; then
                return 0
            fi
            ;;
        postgres)
            if docker exec "$(docker-compose -f docker-compose.yml -f docker-compose.dev.yml ps -q "$service" 2>/dev/null)" pg_isready -U postgres 2>/dev/null; then
                return 0
            fi
            ;;
        *)
            if curl -s --max-time "$timeout" "http://localhost:${port}${path}" >/dev/null 2>&1; then
                return 0
            fi
            ;;
    esac

    return 1
}

restart_service() {
    local service=$1

    warn "Restarting $service..."
    log "Restarting $service"

    if docker-compose -f docker-compose.yml -f docker-compose.dev.yml restart "$service" 2>/dev/null; then
        success "Successfully restarted $service"

        # Wait a bit and verify it's healthy
        sleep 10
        case $service in
            redis) check_service_health "$service" 6379 "" ;;
            postgres) check_service_health "$service" 5432 "" ;;
            api-gateway) check_service_health "$service" 8000 "/health" ;;
            trading-engine) check_service_health "$service" 8001 "/health" ;;
            market-data) check_service_health "$service" 8002 "/health" ;;
            portfolio) check_service_health "$service" 8003 "/health" ;;
            strategy-engine) check_service_health "$service" 8004 "/health" ;;
            token-discovery) check_service_health "$service" 8005 "/health" ;;
            execution) check_service_health "$service" 8006 "/health" ;;
            monitoring) check_service_health "$service" 8007 "/health" ;;
            frontend) check_service_health "$service" 5000 "/health" ;;
        esac

        if [[ $? -eq 0 ]]; then
            success "$service is healthy after restart"
        else
            error "$service failed to become healthy after restart"
            send_alert "LegacyCoinTrader: $service Restart Failed" "$service was restarted but failed health check"
        fi

        return 0
    else
        error "Failed to restart $service"
        send_alert "LegacyCoinTrader: $service Restart Failed" "Failed to restart $service"
        return 1
    fi
}

check_system_resources() {
    # Check disk space
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [[ $disk_usage -gt 90 ]]; then
        warn "High disk usage: ${disk_usage}%"
        send_alert "LegacyCoinTrader: High Disk Usage" "Disk usage is at ${disk_usage}%"
    fi

    # Check memory usage
    local mem_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [[ $mem_usage -gt 90 ]]; then
        warn "High memory usage: ${mem_usage}%"
        send_alert "LegacyCoinTrader: High Memory Usage" "Memory usage is at ${mem_usage}%"
    fi
}

main() {
    info "Starting LegacyCoinTrader health check"

    # Check if Docker is running
    if ! check_docker_running; then
        send_alert "LegacyCoinTrader: Docker Not Running" "Docker daemon is not running or not accessible"
        exit 1
    fi

    # Check system resources
    check_system_resources

    # Define services to check
    declare -A services=(
        ["redis"]="6379"
        ["postgres"]="5432"
        ["api-gateway"]="8000"
        ["trading-engine"]="8001"
        ["market-data"]="8002"
        ["portfolio"]="8003"
        ["strategy-engine"]="8004"
        ["token-discovery"]="8005"
        ["execution"]="8006"
        ["monitoring"]="8007"
        ["frontend"]="5000"
    )

    local failed_services=()
    local restarted_services=()

    # Check each service
    for service in "${!services[@]}"; do
        local port=${services[$service]}
        local path="/health"

        if [[ "$service" == "redis" ]] || [[ "$service" == "postgres" ]]; then
            path=""
        fi

        if ! check_service_health "$service" "$port" "$path" 5; then
            error "$service is unhealthy"
            failed_services+=("$service")

            # Attempt to restart
            if restart_service "$service"; then
                restarted_services+=("$service")
            fi
        else
            success "$service is healthy"
        fi
    done

    # Check trading engine status
    if command -v python3 >/dev/null 2>&1; then
        if ! python3 -m crypto_bot.main status >/dev/null 2>&1; then
            warn "Trading engine status check failed"
            # Try to restart trading engine
            if python3 -m crypto_bot.main start --interval 60 >/dev/null 2>&1; then
                success "Trading engine restarted"
            else
                error "Failed to restart trading engine"
                send_alert "LegacyCoinTrader: Trading Engine Down" "Trading engine is down and could not be restarted"
            fi
        fi
    fi

    # Summary
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        local failed_list=$(IFS=, ; echo "${failed_services[*]}")
        local restarted_list=$(IFS=, ; echo "${restarted_services[*]}")

        if [[ ${#restarted_services[@]} -gt 0 ]]; then
            info "Health check complete: ${#failed_services[@]} failed, ${#restarted_services[@]} restarted ($failed_list)"
            send_alert "LegacyCoinTrader: Services Recovered" "Failed services: $failed_list. Restarted: $restarted_list"
        else
            error "Health check complete: ${#failed_services[@]} failed, 0 restarted ($failed_list)"
            send_alert "LegacyCoinTrader: Services Failed" "Failed services: $failed_list. Could not restart any services."
        fi
    else
        success "Health check complete: All services healthy"
    fi
}

# Run main function
main "$@"
