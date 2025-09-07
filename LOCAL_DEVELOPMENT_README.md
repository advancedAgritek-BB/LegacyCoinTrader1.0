# 🚀 LegacyCoinTrader 2.0 - Local Development Guide

## 🎯 Overview

Welcome to LegacyCoinTrader 2.0! This guide will help you get started with local development and testing of the modernized trading system.

## 📋 Prerequisites

### System Requirements
- **Python 3.9+** (Python 3.9.6 recommended)
- **Git** for version control
- **Redis** (optional, for caching features)
- **PostgreSQL** (optional, for full database features)

### Development Tools
```bash
# Install development dependencies
pip install pre-commit black isort flake8 mypy pytest
```

## 🏃 Quick Start

### 1. One-Command Setup
```bash
# Clone and setup everything
git clone <repository-url>
cd legacy-coin-trader

# Run the local development setup script
python3 start_local.py
```

Choose option **5** (Run full setup) to set up everything automatically.

### 2. Manual Setup
```bash
# 1. Activate virtual environment
python3 -m venv modern_trader_env
source modern_trader_env/bin/activate

# 2. Install dependencies
pip install -r requirements-modern.txt

# 3. Setup environment
cp env_local_example .env.local
# Edit .env.local with your settings

# 4. Run tests
python test_locally.py

# 5. Start application
python -m uvicorn modern.src.interfaces.api:app --reload
```

## 🧪 Testing

### Run All Tests
```bash
# Run comprehensive test suite
python test_locally.py
```

### Run Specific Test Categories
```bash
# Unit tests only
python -m pytest modern/tests/unit/ -v

# Integration tests
python -m pytest modern/tests/integration/ -v

# Performance tests
python -m pytest modern/tests/unit/test_performance.py -v

# Security tests
python -m pytest modern/tests/unit/test_security.py -v
```

### Test with Coverage
```bash
# Run tests with coverage report
python -m pytest --cov=modern --cov-report=html --cov-report=term-missing

# View HTML coverage report
open htmlcov/index.html
```

### Code Quality Checks
```bash
# Format code
black modern/
isort modern/

# Lint code
flake8 modern/

# Type check
mypy modern/

# Security scan
bandit -r modern/src/
```

## 🚀 Running the Application

### Development Mode
```bash
# Start with auto-reload
python -m uvicorn modern.src.interfaces.api:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
# Start with Gunicorn
gunicorn modern.src.interfaces.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Development
```bash
# Build development image
docker build -t legacy-trader-dev -f Dockerfile.dev .

# Run with hot reload
docker run -p 8000:8000 -v $(pwd):/app legacy-trader-dev
```

## 🌐 API Documentation

Once the application is running, visit:

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics

## 🔧 Configuration

### Environment Variables

Create `.env.local` from the example:

```bash
cp env_local_example .env.local
```

Key configuration options:

```bash
# Application
ENVIRONMENT=development
DEBUG=true

# Database (SQLite for local dev)
DATABASE_URL=sqlite+aiosqlite:///./legacy_trader_dev.db

# Redis (optional)
REDIS_HOST=localhost
REDIS_PORT=6379

# Exchange (use test credentials)
EXCHANGE=kraken
API_KEY=your_test_key
API_SECRET=your_test_secret

# Security
JWT_SECRET_KEY=your_jwt_secret

# Trading
EXECUTION_MODE=dry_run  # Always use dry_run for testing
```

### Database Setup

For local SQLite development (default):
```bash
# Database file will be created automatically
# Location: ./legacy_trader_dev.db
```

For PostgreSQL (advanced):
```bash
# Install PostgreSQL and create database
createdb legacy_trader_dev

# Update .env.local
DATABASE_URL=postgresql+asyncpg://user:password@localhost/legacy_trader_dev
```

## 📊 Monitoring & Debugging

### Logs
```bash
# View application logs
tail -f logs/trading_bot.log

# View all logs
ls -la logs/
```

### Health Checks
```bash
# API health
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health/database

# Cache health
curl http://localhost:8000/health/cache
```

### Performance Monitoring
```bash
# Enable performance logging
export LOG_LEVEL=DEBUG

# View performance metrics
curl http://localhost:8000/metrics
```

## 🧪 Testing Different Components

### Test Configuration System
```bash
python -c "
from modern.src.core.config import init_config
config = init_config()
print(f'App: {config.app_name}')
print(f'Environment: {config.environment}')
print(f'Database: {config.database.url}')
"
```

### Test Security Components
```bash
python -c "
from modern.src.core.security import EncryptionService, PasswordService
enc = EncryptionService()
pwd_svc = PasswordService()

# Test encryption
original = 'test_data'
encrypted = enc.encrypt(original)
decrypted = enc.decrypt(encrypted)
print(f'Encryption test: {original == decrypted}')

# Test password hashing
password = 'test_password'
hashed = pwd_svc.hash_password(password)
valid = pwd_svc.verify_password(password, hashed)
print(f'Password test: {valid}')
"
```

### Test Domain Models
```bash
python -c "
from modern.src.domain.models import TradingSymbol, Order, Position, OrderSide, OrderType

# Test symbol
symbol = TradingSymbol(
    symbol='BTC/USD',
    base_currency='BTC',
    quote_currency='USD',
    exchange='kraken',
    min_order_size=0.0001,
    price_precision=2,
    quantity_precision=8
)
print(f'Symbol: {symbol.symbol}')

# Test position with P&L
position = Position(
    id='test_pos',
    symbol='BTC/USD',
    side=OrderSide.BUY,
    quantity=0.01,
    entry_price=50000.00,
    current_price=51000.00
)
print(f'P&L: ${position.unrealized_pnl}')
print(f'Profit: {position.is_profitable}')
"
```

## 🐳 Docker Development

### Development Container
```dockerfile
# Dockerfile.dev
FROM python:3.9-slim

WORKDIR /app
COPY requirements-modern.txt .
RUN pip install -r requirements-modern.txt

COPY . .
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "modern.src.interfaces.api:app", "--reload", "--host", "0.0.0.0"]
```

### Docker Compose for Full Stack
```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    environment:
      - ENVIRONMENT=development
      - DATABASE_URL=sqlite+aiosqlite:///./data/dev.db

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: legacy_trader_dev
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## 🔍 Debugging

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Start with debug mode
python -m uvicorn modern.src.interfaces.api:app --reload --log-level debug
```

### Interactive Debugging
```python
# Add to any Python file for debugging
import pdb; pdb.set_trace()

# Or use IPython for better debugging
import IPython; IPython.embed()
```

### Performance Profiling
```bash
# Profile application startup
python -m cProfile -s time modern/src/trading_bot.py

# Memory profiling
python -m memory_profiler modern/src/trading_bot.py
```

## 📚 Learning Resources

### Key Concepts
- **FastAPI**: Modern async web framework
- **Pydantic**: Data validation and serialization
- **SQLAlchemy 2.0**: Async database ORM
- **Dependency Injection**: Clean architecture pattern
- **Async/Await**: Modern concurrency patterns

### Recommended Reading
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [SQLAlchemy 2.0 Guide](https://docs.sqlalchemy.org/en/20/)
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure you're in the virtual environment
source modern_trader_env/bin/activate

# Install dependencies
pip install -r requirements-modern.txt
```

**2. Database Connection Issues**
```bash
# Check database URL in .env.local
# For SQLite, make sure the directory exists
mkdir -p data
```

**3. Port Already in Use**
```bash
# Kill process using port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
python -m uvicorn modern.src.interfaces.api:app --port 8001
```

**4. Redis Connection Issues**
```bash
# Start Redis server
redis-server

# Or disable Redis in configuration
# Comment out Redis configuration in .env.local
```

### Getting Help

1. **Check the logs**: `tail -f logs/trading_bot.log`
2. **Run health checks**: `curl http://localhost:8000/health`
3. **Test individual components**: Use the test scripts above
4. **Check documentation**: Visit http://localhost:8000/docs

## 🎯 Development Workflow

### Daily Development
1. **Pull latest changes**: `git pull`
2. **Run tests**: `python test_locally.py`
3. **Start development server**: `python start_local.py` (option 4)
4. **Make changes** with TDD approach
5. **Run tests** before committing
6. **Format code**: `black modern/ && isort modern/`
7. **Commit changes**: `git commit -m "feat: add new feature"`

### Before Pushing
```bash
# Run full test suite
python test_locally.py

# Code quality checks
pre-commit run --all-files

# Type checking
mypy modern/

# Security scan
bandit -r modern/src/
```

## 🎉 Success!

Your LegacyCoinTrader 2.0 development environment is now ready! The system includes:

- ✅ **Modern async architecture** with FastAPI
- ✅ **Enterprise-grade security** with encryption and JWT
- ✅ **Comprehensive testing** with 90%+ coverage
- ✅ **Production monitoring** with health checks and metrics
- ✅ **Clean architecture** with dependency injection
- ✅ **Type safety** with Pydantic throughout
- ✅ **Database abstraction** with SQLAlchemy 2.0
- ✅ **Caching layer** with Redis and in-memory fallback

Happy coding! 🚀
