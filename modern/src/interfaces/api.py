"""
FastAPI Application for LegacyCoinTrader 2.0

This is the main API interface for the modernized trading system.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import asyncio
from datetime import datetime
import sys
import os

# Add modern source path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
modern_src = os.path.join(project_root, "modern", "src")
if modern_src not in sys.path:
    sys.path.insert(0, modern_src)

try:
    from core.config import get_settings, AppConfig, init_config
    from core.container import get_container
    from domain.models import TradingSymbol, Order, Position, Trade
    from infrastructure.cache import InMemoryCache

    # Initialize configuration if not already done
    try:
        settings = get_settings()
    except RuntimeError:
        # Initialize with default settings
        settings = init_config()

except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed and the modern architecture is set up correctly.")
    sys.exit(1)

# Create FastAPI application
app = FastAPI(
    title="LegacyCoinTrader 2.0",
    description="Modernized Cryptocurrency Trading System",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection
async def get_config() -> AppConfig:
    """Get application configuration."""
    return settings

async def get_cache() -> InMemoryCache:
    """Get cache instance."""
    container = get_container()
    return container.in_memory_cache()

# Pydantic models for API
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    environment: str

class SymbolResponse(BaseModel):
    symbol: str
    base_currency: str
    quote_currency: str
    exchange: str
    min_order_size: float
    price_precision: int
    quantity_precision: int

class PositionResponse(BaseModel):
    id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    is_profitable: bool

# Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Welcome to LegacyCoinTrader 2.0",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(config: AppConfig = Depends(get_config)):
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=config.version,
        environment=config.environment
    )

@app.get("/health/database", tags=["Health"])
async def database_health():
    """Database health check."""
    try:
        # For now, just return a placeholder
        # In a real implementation, you'd test database connectivity
        return {"status": "healthy", "database": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unhealthy: {e}")

@app.get("/health/cache", tags=["Health"])
async def cache_health(cache: InMemoryCache = Depends(get_cache)):
    """Cache health check."""
    try:
        # Test cache functionality
        test_key = "health_check"
        await cache.set(test_key, "ok", ttl=60)
        value = await cache.get(test_key)
        if value == "ok":
            await cache.delete(test_key)
            return {"status": "healthy", "cache": "working"}
        else:
            return {"status": "unhealthy", "cache": "not responding"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cache unhealthy: {e}")

@app.get("/symbols", response_model=List[SymbolResponse], tags=["Trading"])
async def get_symbols():
    """Get available trading symbols."""
    # Placeholder data - in real implementation, this would come from exchange APIs
    symbols = [
        SymbolResponse(
            symbol="BTC/USD",
            base_currency="BTC",
            quote_currency="USD",
            exchange="kraken",
            min_order_size=0.0001,
            price_precision=2,
            quantity_precision=8
        ),
        SymbolResponse(
            symbol="ETH/USD",
            base_currency="ETH",
            quote_currency="USD",
            exchange="kraken",
            min_order_size=0.001,
            price_precision=2,
            quantity_precision=8
        )
    ]
    return symbols

@app.get("/positions", response_model=List[PositionResponse], tags=["Trading"])
async def get_positions():
    """Get current positions."""
    # Placeholder data - in real implementation, this would come from the trade manager
    positions = [
        PositionResponse(
            id="pos_001",
            symbol="BTC/USD",
            side="buy",
            quantity=0.01,
            entry_price=50000.00,
            current_price=51000.00,
            unrealized_pnl=100.00,
            is_profitable=True
        )
    ]
    return positions

@app.post("/positions/{symbol}/close", tags=["Trading"])
async def close_position(symbol: str):
    """Close a position."""
    # Placeholder implementation
    return {
        "message": f"Position for {symbol} closed successfully",
        "symbol": symbol,
        "timestamp": datetime.now()
    }

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get application metrics."""
    return {
        "uptime": "00:00:00",  # Would track actual uptime
        "total_requests": 0,   # Would track actual requests
        "active_positions": 1, # Would track actual positions
        "last_update": datetime.now()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "timestamp": datetime.now()
        }
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    print("🚀 LegacyCoinTrader 2.0 starting up...")
    print("📊 Loading configuration...")
    print("🔗 Initializing services...")
    print("✅ Application ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    print("🛑 LegacyCoinTrader 2.0 shutting down...")
    print("💾 Saving state...")
    print("🔌 Disconnecting services...")
    print("👋 Shutdown complete!")

if __name__ == "__main__":
    # Get configuration
    try:
        config = get_settings()
        host = getattr(config, 'host', '0.0.0.0')
        port = getattr(config, 'port', 8000)
        reload = getattr(config, 'reload', True)
    except Exception:
        # Fallback to defaults
        host = "0.0.0.0"
        port = 8000
        reload = True

    print(f"🌐 Starting server on http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔄 Reload enabled: {reload}")

    uvicorn.run(
        "interfaces.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
