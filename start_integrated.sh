#!/bin/bash
# Comprehensive LegacyCoinTrader Startup Script
# Starts everything: OHLCV fetching, trading bot, web dashboard, and browser

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --hybrid                Enable dual-run mode (legacy + microservices)
  --legacy-only           Start only the legacy stack (default behaviour)
  --microservices-only    Start only the microservices via docker-compose
  --compose               Launch docker-compose stack before starting legacy services
  --read-strategy <mode>  Dual read strategy (compare, prefer-legacy, prefer-modern)
  --write-strategy <mode> Dual write strategy (mirror, legacy-primary, modern-primary, legacy-only, modern-only)
  --cutover-ready         Mark the deployment as post-cutover (disables legacy writes)
  --drift-tolerance <n>   Numeric drift tolerance (default: 0.0)
  -h, --help              Show this help message
USAGE
}

HYBRID_MODE=false
MICROSERVICE_ONLY=false
START_LEGACY=true
START_COMPOSE=false
READ_STRATEGY="compare"
WRITE_STRATEGY="mirror"
CUTOVER_READY=false
DRIFT_TOLERANCE="0.0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hybrid)
            HYBRID_MODE=true
            START_LEGACY=true
            MICROSERVICE_ONLY=false
            START_COMPOSE=true
            ;;
        --legacy-only)
            START_LEGACY=true
            HYBRID_MODE=false
            MICROSERVICE_ONLY=false
            ;;
        --microservices-only)
            MICROSERVICE_ONLY=true
            START_LEGACY=false
            START_COMPOSE=true
            ;;
        --compose)
            START_COMPOSE=true
            ;;
        --read-strategy)
            READ_STRATEGY="$2"
            shift || true
            ;;
        --write-strategy)
            WRITE_STRATEGY="$2"
            shift || true
            ;;
        --cutover-ready)
            CUTOVER_READY=true
            ;;
        --drift-tolerance)
            DRIFT_TOLERANCE="$2"
            shift || true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
    shift
done

echo -e "${BLUE}🚀 Starting LegacyCoinTrader - Complete System${NC}"
echo -e "${BLUE}$(printf '%.0s=' {1..60})${NC}"
if [[ "$HYBRID_MODE" == "true" ]]; then
    echo -e "${GREEN}🤝 Hybrid mode enabled - orchestrating legacy + microservices${NC}"
else
    echo -e "${GREEN}📈 OHLCV Fetching + 🤖 Trading Bot + 🌐 Web Dashboard${NC}"
fi
echo -e "${BLUE}$(printf '%.0s=' {1..60})${NC}"

# Check if we're already running
if pgrep -f "start_bot.py\|crypto_bot.main\|frontend.app" > /dev/null 2>&1 && [[ "$START_LEGACY" == "true" ]]; then
    echo -e "${YELLOW}⚠️  Legacy services appear to already be running${NC}"
    echo "   To stop: Ctrl+C or use './stop_integrated.sh'"
    echo "   Check processes: ps aux | grep -E '(start_bot.py|crypto_bot|frontend)'"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down all services...${NC}"
    if [[ "$MICROSERVICE_ONLY" == "true" || "$HYBRID_MODE" == "true" || "$START_COMPOSE" == "true" ]]; then
        docker-compose down >/dev/null 2>&1 || true
    fi
    if [[ "$START_LEGACY" == "true" ]]; then
        pkill -f "start_bot.py" || true
        pkill -f "crypto_bot.main" || true
        pkill -f "frontend.app" || true
        pkill -f "telegram_ctl.py" || true
    fi
    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Clean up any stale PID files
rm -f *.pid 2>/dev/null || true

# Configure hybrid feature flags
if [[ "$HYBRID_MODE" == "true" || "$START_COMPOSE" == "true" ]]; then
    export HYBRID_MODE_ENABLED=$([[ "$HYBRID_MODE" == "true" ]] && echo "true" || echo "false")
    if [[ "$READ_STRATEGY" == "prefer-legacy" ]]; then
        export LEGACY_READ_ENABLED=true
    elif [[ "$READ_STRATEGY" == "prefer-modern" ]]; then
        export LEGACY_READ_ENABLED=false
    else
        export LEGACY_READ_ENABLED=$([[ "$HYBRID_MODE" == "true" ]] && echo "true" || echo "false")
    fi
    export HYBRID_READ_STRATEGY="$READ_STRATEGY"
    export DUAL_WRITE_STRATEGY="$WRITE_STRATEGY"
    export CUTOVER_COMPLETED=$([[ "$CUTOVER_READY" == "true" ]] && echo "true" || echo "false")
    export CUTOVER_GUARDRAIL_ENABLED=true
    export MIGRATION_DRIFT_TOLERANCE="$DRIFT_TOLERANCE"
    export HYBRID_FEATURE_FLAG_SOURCE="start_integrated.sh"
fi

# Verify .env file exists when legacy services are needed
if [[ "$START_LEGACY" == "true" ]] && [[ ! -f ".env" ]]; then
    echo -e "${RED}❌ Error: .env file not found!${NC}"
    echo "   Please create a .env file with your API keys"
    echo "   See .env.example for the required format"
    exit 1
fi

if [[ "$START_COMPOSE" == "true" ]]; then
    echo -e "${BLUE}🔧 Launching docker-compose stack...${NC}"
    docker-compose up -d
fi

if [[ "$MICROSERVICE_ONLY" == "true" ]]; then
    echo -e "${GREEN}✅ Microservices stack started in isolation${NC}"
    echo -e "${YELLOW}ℹ️  Legacy services were not started (microservices-only mode)${NC}"
    exit 0
fi

if [[ "$START_LEGACY" == "true" ]]; then
    echo "🎯 Launching legacy LegacyCoinTrader system..."
    echo ""

    export AUTO_START_TRADING=1
    export NON_INTERACTIVE=1

    python3 start_bot.py auto

    echo ""
    echo -e "${GREEN}✅ Legacy system has stopped${NC}"
fi

if [[ "$HYBRID_MODE" == "true" ]]; then
    echo -e "${GREEN}✅ Hybrid runtime finished. Compose services remain running.${NC}"
fi
