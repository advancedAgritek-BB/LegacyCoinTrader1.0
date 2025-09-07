# LegacyCoinTrader Microservices Architecture

This document outlines the complete microservices architecture for the LegacyCoinTrader application, which has been refactored from a monolithic structure into a scalable, maintainable microservices system.

## 🏗️ Architecture Overview

The LegacyCoinTrader has been broken down into the following microservices:

### Core Services

1. **API Gateway** (Port 8000) - Entry point for all external requests
2. **Trading Engine** (Port 8001) - Core orchestration and trading logic
3. **Market Data** (Port 8002) - OHLCV data acquisition and caching
4. **Portfolio** (Port 8003) - Position management and risk assessment
5. **Strategy Engine** (Port 8004) - Strategy evaluation and signal generation
6. **Token Discovery** (Port 8005) - DEX scanning and new token discovery
7. **Execution** (Port 8006) - Order placement and exchange integration
8. **Monitoring** (Port 8007) - System monitoring and health checks

### Supporting Infrastructure

- **Redis** (Port 6379) - Caching, service discovery, and pub/sub messaging
- **PostgreSQL** (Port 5432) - Persistent data storage (production)
- **Frontend** (Port 5000) - Web dashboard and user interface

## 📁 Project Structure

```
LegacyCoinTrader1.0/
├── services/                          # Microservices directory
│   ├── api_gateway/                   # API Gateway service
│   │   ├── app.py                     # Main application
│   │   ├── config.py                  # Configuration
│   │   ├── middleware.py              # Authentication & rate limiting
│   │   ├── routes.py                  # Route definitions
│   │   ├── service_discovery.py       # Service discovery logic
│   │   ├── config.yaml               # Service configuration
│   │   ├── requirements.txt           # Python dependencies
│   │   └── Dockerfile                 # Docker configuration
│   ├── trading_engine/                # Trading Engine service
│   │   ├── app.py                     # Main application
│   │   ├── trading_orchestrator.py    # Core trading logic
│   │   ├── config.py                  # Configuration
│   │   ├── health.py                  # Health checks
│   │   └── ...
│   ├── market_data/                   # Market Data service
│   ├── portfolio/                     # Portfolio service
│   ├── strategy_engine/               # Strategy Engine service
│   ├── token_discovery/               # Token Discovery service
│   ├── execution/                     # Execution service
│   └── monitoring/                    # Monitoring service
├── docker-compose.yml                 # Base Docker Compose
├── docker-compose.dev.yml            # Development overrides
├── docker-compose.prod.yml           # Production configuration
├── microservice_architecture.yaml    # Architecture documentation
└── MICROSERVICES_README.md           # This file
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Git
- Python 3.11+ (for local development)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd LegacyCoinTrader1.0
```

### 2. Environment Configuration

Create environment files for each service or use the provided defaults.

### 3. Start with Docker Compose

#### Development Environment
```bash
# Start all services in development mode
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Production Environment
```bash
# Start all services in production mode
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale specific services
docker-compose up -d --scale trading-engine=3
```

### 4. Verify Installation

```bash
# Check service health
curl http://localhost:8000/health

# Check individual services
curl http://localhost:8001/health  # Trading Engine
curl http://localhost:8002/health  # Market Data
curl http://localhost:5000/        # Frontend Dashboard
```

## 🔧 Service Details

### API Gateway Service

**Purpose**: Single entry point for all external requests
**Port**: 8000
**Key Features**:
- Request routing to appropriate services
- Authentication and authorization
- Rate limiting
- Service discovery integration

**API Endpoints**:
- `GET /health` - System health check
- `POST /api/trading/start` - Start trading
- `GET /api/market/ohlcv` - Get market data
- `GET /api/portfolio/positions` - Get positions

### Trading Engine Service

**Purpose**: Core trading orchestration and cycle management
**Port**: 8001
**Key Features**:
- Trading cycle execution
- Symbol batch processing
- Strategy coordination
- Performance monitoring

**Configuration**:
```yaml
trading_engine:
  cycle_interval: 120  # 2 minutes
  batch_size: 25
  max_risk_per_trade: 0.05
```

### Market Data Service

**Purpose**: Data acquisition, caching, and WebSocket management
**Port**: 8002
**Key Features**:
- OHLCV data fetching
- Real-time price feeds
- Data validation and caching
- Multiple data sources support

### Portfolio Service

**Purpose**: Position tracking and risk management
**Port**: 8003
**Key Features**:
- Position lifecycle management
- P&L calculation
- Risk assessment
- Balance synchronization

### Strategy Engine Service

**Purpose**: Strategy evaluation and signal generation
**Port**: 8004
**Key Features**:
- Multiple strategy support
- Regime classification
- Signal generation
- Strategy optimization

### Token Discovery Service

**Purpose**: DEX scanning and new token discovery
**Port**: 8005
**Key Features**:
- Solana DEX monitoring
- New token detection
- Liquidity analysis
- Pool scanning

### Execution Service

**Purpose**: Order execution and exchange integration
**Port**: 8006
**Key Features**:
- Multi-exchange support
- Order lifecycle management
- Execution monitoring
- Error handling

### Monitoring Service

**Purpose**: System monitoring and alerting
**Port**: 8007
**Key Features**:
- Health checks
- Performance metrics
- Alerting system
- Log aggregation

## 🔗 Inter-Service Communication

### Service Discovery

Services automatically discover each other using Redis-based service discovery:

```python
# Register service
await redis.setex(f"service:{service_name}", ttl, service_info)

# Discover service
service_url = await redis.get(f"service:{service_name}")
```

### API Communication

Services communicate via HTTP REST APIs with authentication:

```python
headers = {'X-Service-Auth': 'service-token'}
async with session.post(f"{service_url}/endpoint", headers=headers, json=data)
```

### Message Patterns

- **Request/Response**: Synchronous API calls
- **Pub/Sub**: Real-time events via Redis
- **Health Checks**: Periodic status monitoring

## 🛠️ Development Workflow

### 1. Local Development

```bash
# Start only required services for development
docker-compose up redis api-gateway trading-engine

# Run service locally with hot reload
cd services/trading_engine
python app.py
```

### 2. Testing

```bash
# Run service-specific tests
cd services/trading_engine
python -m pytest tests/

# Test inter-service communication
curl -X POST http://localhost:8001/cycle
```

### 3. Adding New Services

1. Create service directory structure
2. Implement service logic
3. Add to docker-compose.yml
4. Update API Gateway routes
5. Add health checks

### 4. Configuration Management

Each service has its own configuration:

```yaml
# services/trading_engine/config.yaml
trading_engine:
  port: 8001
  cycle_interval: 120
  batch_size: 25
```

## 📊 Monitoring and Observability

### Health Checks

All services expose health endpoints:
```bash
curl http://localhost:8001/health
```

### Metrics

- **Trading Engine**: Cycle execution time, success rate
- **Market Data**: Cache hit rate, data freshness
- **Portfolio**: Position count, risk metrics
- **Execution**: Order success rate, latency

### Logging

Centralized logging with structured JSON:

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "service": "trading_engine",
  "level": "INFO",
  "message": "Trading cycle completed",
  "cycle_time": 45.2,
  "symbols_processed": 25
}
```

## 🔒 Security

### Authentication

- **API Key**: External client authentication
- **Service Tokens**: Inter-service authentication
- **JWT**: Session management

### Network Security

- Service-to-service communication over internal network
- External access through API Gateway only
- Rate limiting and request validation

### Data Protection

- Encrypted communication (HTTPS/TLS)
- Secure credential management
- Audit logging

## 🚀 Deployment Strategies

### Development

- Hot reload enabled
- Debug logging
- Local database
- Single instance per service

### Staging

- Production-like configuration
- Multi-instance testing
- Integration testing
- Performance benchmarking

### Production

- Load balancing
- Auto-scaling
- High availability
- Comprehensive monitoring

## 🔄 Migration from Monolith

### Phase 1: Core Services (✅ Complete)
- [x] API Gateway
- [x] Trading Engine
- [x] Docker setup

### Phase 2: Data Services (🔄 In Progress)
- [ ] Market Data Service
- [ ] Portfolio Service
- [ ] Strategy Engine Service

### Phase 3: Specialized Services
- [ ] Token Discovery Service
- [ ] Execution Service
- [ ] Monitoring Service

### Phase 4: Integration
- [ ] Update Frontend
- [ ] Testing Strategy
- [ ] Production Deployment

## 🐛 Troubleshooting

### Common Issues

1. **Service Discovery Failures**
   ```bash
   # Check Redis connectivity
   docker-compose exec redis redis-cli ping

   # Verify service registration
   docker-compose logs api-gateway
   ```

2. **Inter-Service Communication**
   ```bash
   # Test service endpoints
   curl http://localhost:8001/health
   curl http://localhost:8002/health
   ```

3. **Resource Constraints**
   ```bash
   # Check container resources
   docker stats

   # Scale services if needed
   docker-compose up -d --scale trading-engine=2
   ```

### Logs and Debugging

```bash
# View all service logs
docker-compose logs

# View specific service logs
docker-compose logs trading-engine

# Follow logs in real-time
docker-compose logs -f trading-engine
```

## 📈 Performance Optimization

### Caching Strategy

- **Redis**: Session data, service discovery, metrics
- **In-memory**: Frequently accessed market data
- **Database**: Persistent trading history

### Scaling Considerations

- **Horizontal Scaling**: Multiple instances per service
- **Load Balancing**: Distribute requests across instances
- **Database Sharding**: Split data across multiple databases

### Resource Management

- **Memory Limits**: Prevent memory leaks
- **CPU Allocation**: Optimize for compute-intensive tasks
- **Network Optimization**: Efficient inter-service communication

## 🤝 Contributing

### Code Standards

- Follow PEP 8 style guidelines
- Add comprehensive tests
- Update documentation
- Use type hints

### Testing Strategy

- Unit tests for individual components
- Integration tests for service communication
- End-to-end tests for complete workflows
- Performance tests for scaling validation

### Documentation

- Update service README files
- Maintain API documentation
- Keep deployment guides current
- Document configuration options

## 📚 Additional Resources

- [Microservice Architecture Patterns](https://microservices.io/patterns/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Redis Documentation](https://redis.io/documentation)

---

## 🎯 Benefits of Microservices Architecture

### Scalability
- Scale individual services based on load
- Optimize resource allocation
- Handle peak loads efficiently

### Maintainability
- Smaller, focused codebases
- Independent deployments
- Easier testing and debugging

### Reliability
- Service isolation prevents cascade failures
- Independent scaling and updates
- Better fault tolerance

### Technology Diversity
- Choose best technology for each service
- Gradual modernization
- Experiment with new technologies

### Team Productivity
- Independent development teams
- Faster deployment cycles
- Clear service ownership

This microservices architecture provides a solid foundation for scaling the LegacyCoinTrader application while maintaining reliability and ease of development.
