#!/bin/bash
# LegacyCoinTrader Docker Management Script
# Provides easy access to enhanced Docker startup and monitoring

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to show usage
show_usage() {
    echo -e "${BLUE}🐳 LegacyCoinTrader Docker Manager${NC}"
    echo -e "${BLUE}$(printf '%.0s=' {1..40})${NC}"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start [env]     - Start services with orchestration (dev/prod/test)"
    echo "  stop            - Stop all services"
    echo "  restart [env]   - Restart services with orchestration"
    echo "  status          - Show detailed service status"
    echo "  health          - Check service health"
    echo "  watch           - Watch service health in real-time"
    echo "  validate        - Validate environment"
    echo "  logs [service]  - Show logs (optionally for specific service)"
    echo "  shell [service] - Open shell in service container"
    echo "  clean           - Clean up containers and volumes"
    echo "  help            - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 start dev    - Start in development mode"
    echo "  $0 start prod   - Start in production mode"
    echo "  $0 health       - Check all service health"
    echo "  $0 watch        - Monitor health in real-time"
    echo "  $0 logs api-gateway - Show API gateway logs"
    echo ""
}

# Function to start services
start_services() {
    local env=${1:-dev}
    echo -e "${GREEN}🚀 Starting LegacyCoinTrader ($env environment)${NC}"
    
    # Validate environment first
    echo -e "${BLUE}🔍 Validating environment...${NC}"
    if ! python3 docker-startup-orchestrator.py --validate-only; then
        echo -e "${RED}❌ Environment validation failed${NC}"
        exit 1
    fi
    
    # Start services with orchestration
    python3 docker-startup-orchestrator.py --env "$env" --report
    
    echo -e "${GREEN}✅ Services started successfully!${NC}"
    echo -e "${BLUE}🌐 Dashboard: http://localhost:5000${NC}"
    echo -e "${BLUE}🔧 API Gateway: http://localhost:8000${NC}"
}

# Function to stop services
stop_services() {
    echo -e "${YELLOW}🛑 Stopping all services...${NC}"
    docker-compose down
    echo -e "${GREEN}✅ All services stopped${NC}"
}

# Function to restart services
restart_services() {
    local env=${1:-dev}
    echo -e "${YELLOW}🔄 Restarting services...${NC}"
    stop_services
    sleep 2
    start_services "$env"
}

# Function to show status
show_status() {
    echo -e "${BLUE}📊 Service Status${NC}"
    echo -e "${BLUE}$(printf '%.0s=' {1..30})${NC}"
    docker-compose ps
    echo ""
    python3 docker-health-monitor.py
}

# Function to check health
check_health() {
    echo -e "${BLUE}🏥 Health Check${NC}"
    echo -e "${BLUE}$(printf '%.0s=' {1..30})${NC}"
    python3 docker-health-monitor.py --report
}

# Function to watch health
watch_health() {
    echo -e "${BLUE}👁️ Watching service health (Press Ctrl+C to exit)${NC}"
    python3 docker-health-monitor.py --watch 10
}

# Function to validate environment
validate_environment() {
    echo -e "${BLUE}🔍 Validating environment${NC}"
    python3 docker-startup-orchestrator.py --validate-only
}

# Function to show logs
show_logs() {
    local service=${1:-}
    if [[ -n "$service" ]]; then
        echo -e "${BLUE}📄 Logs for $service${NC}"
        docker-compose logs -f "$service"
    else
        echo -e "${BLUE}📄 Logs for all services${NC}"
        docker-compose logs -f
    fi
}

# Function to open shell
open_shell() {
    local service=${1:-}
    if [[ -z "$service" ]]; then
        echo -e "${RED}❌ Please specify a service name${NC}"
        echo "Available services: $(docker-compose config --services | tr '\n' ' ')"
        exit 1
    fi
    
    echo -e "${BLUE}🐚 Opening shell in $service${NC}"
    docker-compose exec "$service" /bin/bash || docker-compose exec "$service" /bin/sh
}

# Function to clean up
clean_up() {
    echo -e "${YELLOW}🧹 Cleaning up containers and volumes...${NC}"
    read -p "This will remove all containers and volumes. Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down -v --rmi all
        docker system prune -f
        echo -e "${GREEN}✅ Cleanup completed${NC}"
    else
        echo -e "${BLUE}ℹ️ Cleanup cancelled${NC}"
    fi
}

# Main command handling
case "${1:-help}" in
    start)
        start_services "${2:-dev}"
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services "${2:-dev}"
        ;;
    status)
        show_status
        ;;
    health)
        check_health
        ;;
    watch)
        watch_health
        ;;
    validate)
        validate_environment
        ;;
    logs)
        show_logs "$2"
        ;;
    shell)
        open_shell "$2"
        ;;
    clean)
        clean_up
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo ""
        show_usage
        exit 1
        ;;
esac
