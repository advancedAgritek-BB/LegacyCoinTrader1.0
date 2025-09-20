"""
Monitoring Service - Metrics collection and exposure
"""

import logging
import time
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
import uvicorn

from services.monitoring.config import get_monitoring_settings
from services.monitoring.instrumentation import instrument_fastapi_app
from services.monitoring.logging_utils import configure_logging

# Configure logging
settings = get_monitoring_settings().for_service("monitoring", environment="production")
configure_logging(settings)

# Create FastAPI app
app = FastAPI(
    title="LegacyCoinTrader Monitoring Service",
    description="Centralized metrics collection and monitoring",
    version="1.0.0"
)

# Instrument the app
instrument_fastapi_app(app, settings=settings)

logger = logging.getLogger(__name__)

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "monitoring",
        "timestamp": int(time.time())
    }

@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus metrics endpoint"""
    # In a real implementation, this would collect metrics from all services
    # For now, return basic service health metrics
    metrics_content = """# HELP monitoring_service_up Service uptime
# TYPE monitoring_service_up gauge
monitoring_service_up 1

# HELP monitoring_service_health Service health status
# TYPE monitoring_service_health gauge
monitoring_service_health 1
"""
    return PlainTextResponse(metrics_content, media_type="text/plain; charset=utf-8")

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint"""
    return {
        "service": "LegacyCoinTrader Monitoring",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics"
        }
    }

if __name__ == "__main__":
    logger.info("Starting Monitoring Service")
    uvicorn.run(
        "services.monitoring.main:app",
        host="0.0.0.0",
        port=8007,
        reload=False,
        log_level="info"
    )
