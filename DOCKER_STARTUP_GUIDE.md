# 🐳 Enhanced Docker Startup Guide

## Overview

Your LegacyCoinTrader system now has an **enhanced Docker startup process** with intelligent orchestration, comprehensive health monitoring, and proper service sequencing. This guide explains how to use the improved system.

## 🚀 Quick Start

### Recommended Startup (New Method)
```bash
# Easy startup with intelligent orchestration
./docker-manager.sh start dev

# Or using Make
make smart-start
```

### Legacy Startup (Still Available)
```bash
# Traditional Docker Compose
make dev

# Or directly
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

## 🔧 Available Tools

### 1. Docker Manager Script (`./docker-manager.sh`)
**Main interface for managing your Docker services**

```bash
./docker-manager.sh start dev     # Start in development mode
./docker-manager.sh start prod    # Start in production mode  
./docker-manager.sh status        # Show detailed status
./docker-manager.sh health        # Check service health
./docker-manager.sh watch         # Monitor health real-time
./docker-manager.sh stop          # Stop all services
./docker-manager.sh clean         # Clean up everything
```

### 2. Health Monitor (`docker-health-monitor.py`)
**Comprehensive health checking and monitoring**

```bash
python docker-health-monitor.py                    # Check current status
python docker-health-monitor.py --watch 10         # Watch every 10 seconds
python docker-health-monitor.py --wait --timeout 5 # Wait for healthy startup
python docker-health-monitor.py --report           # Generate health report
```

### 3. Startup Orchestrator (`docker-startup-orchestrator.py`)
**Intelligent service startup with proper sequencing**

```bash
python docker-startup-orchestrator.py --env dev    # Start development
python docker-startup-orchestrator.py --env prod   # Start production
python docker-startup-orchestrator.py --validate-only  # Just validate environment
```

### 4. Enhanced Makefile Commands
```bash
make smart-start          # Intelligent development startup
make smart-start-prod     # Intelligent production startup
make validate-env         # Validate environment
make health-detailed      # Detailed health check
make health-watch         # Watch health in real-time
make status-detailed      # Detailed status
```

## 📊 Service Architecture

### Service Startup Phases
The orchestrator starts services in the following phases:

1. **Infrastructure** (Redis, PostgreSQL)
2. **Core Services** (API Gateway)
3. **Data Services** (Market Data, Portfolio)
4. **Business Logic** (Trading Engine, Strategy Engine, Token Discovery)
5. **Execution** (Execution Service)
6. **Monitoring** (Monitoring Service)
7. **Frontend** (Web Dashboard)

### Service Dependencies
```
Frontend ← API Gateway ← Core Services ← Infrastructure
    ↑           ↑            ↑
Monitoring ← Execution ← Business Logic
```

## 🏥 Health Monitoring

### What's Monitored
- **Container Status**: Running, stopped, failed
- **HTTP Health Endpoints**: Response time and status codes
- **Service Dependencies**: Dependency readiness
- **Database Connectivity**: PostgreSQL and Redis
- **Service-Specific Health**: Custom validation per service

### Health Status Levels
- 🟢 **Healthy**: Service is fully operational
- 🟡 **Warning**: Service running but with issues
- 🔴 **Unhealthy**: Service has problems
- 💀 **Failed**: Service completely failed

## 🔍 Environment Validation

Before startup, the system validates:
- ✅ Docker and Docker Compose availability
- ✅ `.env` file presence and configuration
- ✅ API key availability (warns if missing)
- ✅ Disk space (requires 5GB minimum)
- ✅ Port availability (warns of conflicts)

## 📈 Monitoring & Observability

### Real-time Monitoring
```bash
# Watch all services
./docker-manager.sh watch

# Watch specific service logs
./docker-manager.sh logs trading-engine
```

### Health Reports
Health reports are automatically saved to `logs/` directory:
- `docker_health_report_TIMESTAMP.json`
- `startup_report_TIMESTAMP.json`

### Service Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api-gateway
```

## 🔧 Configuration

### Environment Files
- **Development**: `docker-compose.dev.yml`
- **Production**: `docker-compose.prod.yml`
- **Testing**: `docker-compose.test.yml`

### Service Ports
| Service | Port | Health Endpoint |
|---------|------|----------------|
| Frontend | 5000 | `/health` |
| API Gateway | 8000 | `/health` |
| Trading Engine | 8001 | `/health` |
| Market Data | 8002 | `/health` |
| Portfolio | 8003 | `/health` |
| Strategy Engine | 8004 | `/health` |
| Token Discovery | 8005 | `/health` |
| Execution | 8006 | `/health` |
| Monitoring | 8007 | `/health` |
| Redis | 6379 | - |
| PostgreSQL | 5432 | - |

## 🚨 Troubleshooting

### Common Issues

#### Services Won't Start
```bash
# Check environment
./docker-manager.sh validate

# Check Docker status
docker version
docker-compose version

# Check port conflicts
netstat -tulpn | grep :8000
```

#### Services Unhealthy
```bash
# Detailed health check
./docker-manager.sh health

# Check specific service logs
./docker-manager.sh logs [service-name]

# Restart problematic service
docker-compose restart [service-name]
```

#### Database Issues
```bash
# Check PostgreSQL
docker-compose exec postgres pg_isready -U legacy_user

# Check Redis
docker-compose exec redis redis-cli ping

# Reset databases (CAUTION: Data loss)
docker-compose down -v
```

#### API Key Issues
```bash
# Check .env file
cat .env | grep -E "(KRAKEN|HELIUS|TELEGRAM)"

# Validate API keys (if implemented)
./docker-manager.sh validate
```

### Emergency Commands
```bash
# Stop everything immediately
docker stop $(docker ps -q)

# Complete cleanup (CAUTION: Removes everything)
./docker-manager.sh clean

# Emergency reset
make emergency-stop
make emergency-clean
```

## 📚 Advanced Usage

### Custom Health Checks
You can extend health checks by modifying `docker-health-monitor.py`:

```python
async def validate_service_specific_health(self, service_name: str, response: requests.Response) -> bool:
    # Add custom validation logic
    pass
```

### Custom Startup Phases
Modify startup phases in `docker-startup-orchestrator.py`:

```python
def _define_startup_phases(self) -> Dict[str, List[str]]:
    return {
        'custom_phase': ['your-service'],
        # ... other phases
    }
```

### Integration with CI/CD
```bash
# In CI pipeline
make validate-env
make smart-start
make health-wait
make test
```

## 🎯 Best Practices

1. **Always validate environment** before starting services
2. **Use health monitoring** to ensure services are truly ready
3. **Check logs** when services fail to start
4. **Use phased startup** for complex dependencies
5. **Monitor resource usage** in production
6. **Keep health reports** for troubleshooting

## 📋 Comparison: Old vs New Process

| Feature | Old Process | New Process |
|---------|-------------|-------------|
| Startup Method | `docker-compose up -d` | Phased orchestration |
| Health Checking | Basic HTTP checks | Comprehensive validation |
| Dependency Management | Docker depends_on | Intelligent sequencing |
| Monitoring | Manual log checking | Real-time health monitoring |
| Error Handling | Manual troubleshooting | Automated validation |
| Environment Validation | None | Pre-startup checks |
| Reporting | None | Automated health reports |

## 🔗 Related Files

- `docker-compose.yml` - Main service definitions
- `docker-compose.dev.yml` - Development overrides
- `docker-compose.prod.yml` - Production configuration
- `Makefile` - Build and management commands
- `.env` - Environment variables
- `logs/` - Health reports and logs

## 🆘 Support

If you encounter issues:

1. **Check this guide** for common solutions
2. **Run validation**: `./docker-manager.sh validate`
3. **Check health**: `./docker-manager.sh health`
4. **Review logs**: `./docker-manager.sh logs`
5. **Try clean restart**: `./docker-manager.sh clean` then `./docker-manager.sh start dev`

The enhanced Docker startup process provides much better visibility into what's happening during startup and makes it easier to identify and resolve issues when they occur.
