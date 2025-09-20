#!/usr/bin/env bash

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

info() {
    echo -e "${BLUE}$1${NC}"
}

success() {
    echo -e "${GREEN}$1${NC}"
}

warn() {
    echo -e "${YELLOW}$1${NC}"
}

error() {
    echo -e "${RED}$1${NC}"
}

check_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        error "❌ Docker not installed"
        return 1
    fi

    if ! docker info >/dev/null 2>&1; then
        error "❌ Docker daemon not running"
        return 1
    fi

    success "✅ Docker running"
    return 0
}

check_service() {
    local service=$1
    local port=$2
    local path=${3:-/health}

    if [[ "$service" == "redis" ]]; then
        if docker exec "$(docker-compose -f docker-compose.yml -f docker-compose.dev.yml ps -q "$service" 2>/dev/null)" redis-cli ping 2>/dev/null | grep -q PONG; then
            success "✅ $service healthy"
            return 0
        fi
    elif [[ "$service" == "postgres" ]]; then
        if docker exec "$(docker-compose -f docker-compose.yml -f docker-compose.dev.yml ps -q "$service" 2>/dev/null)" pg_isready -U postgres 2>/dev/null; then
            success "✅ $service healthy"
            return 0
        fi
    else
        if curl -s --max-time 3 "http://localhost:${port}${path}" >/dev/null 2>&1; then
            success "✅ $service healthy"
            return 0
        fi
    fi

    error "❌ $service unhealthy"
    return 1
}

check_trading_engine() {
    if command -v python3 >/dev/null 2>&1; then
        if python3 -c "
import asyncio
from crypto_bot.main import TradingEngineClient

async def check():
    try:
        client = TradingEngineClient()
        status = await client._request('GET', '/api/v1/trading/cycles/status')
        if status.get('running'):
            print('RUNNING')
        else:
            print('STOPPED')
    except:
        print('ERROR')

result = asyncio.run(check())
print(result, end='')
" 2>/dev/null | grep -q RUNNING; then
            success "✅ Trading engine running"
            return 0
        fi
    fi

    error "❌ Trading engine not running"
    return 1
}

check_systemd_service() {
    if command -v systemctl >/dev/null 2>&1; then
        if systemctl --user is-active --quiet monitoring.service 2>/dev/null; then
            success "✅ Systemd service active"
            return 0
        else
            warn "⚠️  Systemd service inactive"
            return 1
        fi
    else
        info "ℹ️  Systemd not available"
        return 0
    fi
}

check_cron_job() {
    if crontab -l 2>/dev/null | grep -q "health-check-cron.sh"; then
        success "✅ Cron job active"
        return 0
    else
        warn "⚠️  Cron job not configured"
        return 1
    fi
}

check_disk_space() {
    local usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [[ $usage -gt 90 ]]; then
        error "❌ High disk usage: ${usage}%"
        return 1
    elif [[ $usage -gt 80 ]]; then
        warn "⚠️  Disk usage: ${usage}%"
        return 0
    else
        success "✅ Disk usage: ${usage}%"
        return 0
    fi
}

check_memory() {
    # Check memory usage (works on both Linux and macOS)
    if command -v free >/dev/null 2>&1; then
        # Linux
        local usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        local usage=$(vm_stat | awk '/Pages active/ {print int($3/1024/1024*100)}')
        if [[ -z "$usage" ]]; then
            usage=0
        fi
    else
        # Fallback
        success "✅ Memory check not available on this OS"
        return 0
    fi

    if [[ $usage -gt 90 ]]; then
        error "❌ High memory usage: ${usage}%"
        return 1
    elif [[ $usage -gt 80 ]]; then
        warn "⚠️  Memory usage: ${usage}%"
        return 0
    else
        success "✅ Memory usage: ${usage}%"
        return 0
    fi
}

show_recent_trades() {
    info "📊 Recent Trading Activity:"
    if [[ -f "$PROJECT_ROOT/logs/execution.log" ]]; then
        tail -5 "$PROJECT_ROOT/logs/execution.log" 2>/dev/null | grep -E "(BUY|SELL|TRADE|EXECUTED)" | tail -3 || echo "   No recent trades"
    else
        echo "   No execution logs available"
    fi
}

show_positions() {
    info "📈 Current Positions:"
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "
import asyncio
from crypto_bot.main import TradingEngineClient

async def get_positions():
    try:
        client = TradingEngineClient()
        positions = await client._request('GET', '/api/v1/portfolio/positions')
        if isinstance(positions, list) and len(positions) > 0:
            for pos in positions[:3]:  # Show first 3
                symbol = pos.get('symbol', 'Unknown')
                pnl = pos.get('unrealized_pnl', 0)
                print(f'   {symbol}: \${pnl:+.2f} P&L')
            if len(positions) > 3:
                print(f'   ... and {len(positions)-3} more positions')
        else:
            print('   No open positions')
    except Exception as e:
        print(f'   Error getting positions: {e}')

asyncio.run(get_positions())
" 2>/dev/null || echo "   Could not retrieve positions"
    fi
}

show_service_status() {
    info "🐳 Docker Services:"
    if command -v docker-compose >/dev/null 2>&1; then
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null | while read -r line; do
            if echo "$line" | grep -q "Up"; then
                success "   $line"
            elif echo "$line" | grep -q "Down\|Exit"; then
                error "   $line"
            else
                info "   $line"
            fi
        done
    else
        warn "   docker-compose not available"
    fi
}

main() {
    echo "=========================================="
    echo "🧭 LegacyCoinTrader Status Dashboard"
    echo "=========================================="
    echo ""

    # System checks
    echo "🔧 System Status:"
    check_docker
    check_disk_space
    check_memory
    echo ""

    # Auto-start checks
    echo "🔄 Auto-Start Status:"
    check_systemd_service
    check_cron_job
    echo ""

    # Service checks
    echo "🌐 Service Health:"
    check_service "redis" "6379" ""
    check_service "postgres" "5432" ""
    check_service "api-gateway" "8000" "/health"
    check_service "trading-engine" "8001" "/health"
    check_service "market-data" "8002" "/health"
    check_service "portfolio" "8003" "/health"
    check_service "execution" "8006" "/health"
    check_service "frontend" "5000" "/health"
    check_trading_engine
    echo ""

    # Trading activity
    show_positions
    echo ""
    show_recent_trades
    echo ""

    # Docker services
    show_service_status
    echo ""

    echo "=========================================="
    echo "📋 Quick Actions:"
    echo "  Start services:    ./auto-startup.sh"
    echo "  Stop services:     ./shutdown.sh"
    echo "  Health check:      ./health-check-cron.sh"
    echo "  View logs:         ./docker-manager.sh logs"
    echo "  Restart all:       ./docker-manager.sh restart dev"
    echo "=========================================="
}

main "$@"
