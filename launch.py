#!/usr/bin/env python3
"""
Simple Launcher for LegacyCoinTrader 2.0

This script provides easy startup for the modernized trading system.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def setup_environment():
    """Setup basic environment variables."""
    # Set minimal environment for development
    os.environ.setdefault("PYTHONPATH", str(Path(__file__).parent / "modern" / "src"))
    os.environ.setdefault("ENVIRONMENT", "development")
    os.environ.setdefault("APP_NAME", "LegacyCoinTrader")
    os.environ.setdefault("VERSION", "2.0.0")

    # Database configuration
    os.environ.setdefault("DB_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    os.environ.setdefault("DB_POOL_SIZE", "10")
    os.environ.setdefault("DB_MAX_OVERFLOW", "20")

    # Redis configuration
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
    os.environ.setdefault("REDIS_CACHE_TTL", "300")
    os.environ.setdefault("REDIS_MAX_CONNECTIONS", "20")

    # Exchange configuration
    os.environ.setdefault("EXCHANGE_NAME", "kraken")
    os.environ.setdefault("EXCHANGE_API_KEY", "demo_key")
    os.environ.setdefault("EXCHANGE_API_SECRET", "demo_secret")
    os.environ.setdefault("EXCHANGE_SANDBOX", "false")

    # Security configuration
    os.environ.setdefault("SECURITY_JWT_SECRET_KEY", "demo_jwt_secret_for_development_only")
    os.environ.setdefault("SECURITY_PASSWORD_MIN_LENGTH", "8")
    os.environ.setdefault("SECURITY_ENCRYPTION_KEY", "demo_encryption_key_1234567890123456")

    # Monitoring configuration
    os.environ.setdefault("MONITORING_ENABLED", "false")
    os.environ.setdefault("MONITORING_LOG_LEVEL", "INFO")
    os.environ.setdefault("MONITORING_METRICS_INTERVAL", "60")

    # Trading configuration
    os.environ.setdefault("TRADING_EXECUTION_MODE", "dry_run")
    os.environ.setdefault("TRADING_MAX_POSITION_SIZE", "1.0")
    os.environ.setdefault("TRADING_RISK_PER_TRADE", "0.02")

def check_requirements():
    """Check if basic requirements are met."""
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✅ Dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements-modern.txt")
        return False

def launch_server():
    """Launch the FastAPI server."""
    print("🚀 Launching LegacyCoinTrader 2.0...")
    print("=" * 50)

    # Add modern source to path
    project_root = Path(__file__).parent
    modern_src = project_root / "modern" / "src"
    sys.path.insert(0, str(modern_src))

    try:
        # Simple approach - create a basic FastAPI app directly
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        from datetime import datetime

        app = FastAPI(
            title="LegacyCoinTrader 2.0",
            description="Modernized Cryptocurrency Trading System",
            version="2.0.0"
        )

        # Add CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/")
        async def root():
            return {
                "message": "Welcome to LegacyCoinTrader 2.0",
                "status": "running",
                "version": "2.0.0"
            }

        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "timestamp": datetime.now(),
                "environment": "development"
            }

        @app.get("/symbols")
        async def get_symbols():
            return [
                {"symbol": "BTC/USD", "exchange": "kraken"},
                {"symbol": "ETH/USD", "exchange": "kraken"}
            ]

        import uvicorn

        print("🌐 Starting FastAPI server...")
        print("📚 API Docs: http://localhost:8000/docs")
        print("🏥 Health Check: http://localhost:8000/health")
        print("🔄 Press Ctrl+C to stop")
        print()

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        print("Make sure you're in the virtual environment:")
        print("source modern_trader_env/bin/activate")
        return False

    return True

def launch_with_uvicorn():
    """Launch using uvicorn command directly."""
    print("🚀 Launching with uvicorn...")

    # Set PYTHONPATH
    env = os.environ.copy()
    project_root = Path(__file__).parent
    modern_src = project_root / "modern" / "src"
    env["PYTHONPATH"] = str(modern_src)

    try:
        cmd = [
            sys.executable, "-m", "uvicorn",
            "interfaces.api:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ]

        print("🌐 Starting server...")
        print("📚 API Docs: http://localhost:8000/docs")
        print("🏥 Health Check: http://localhost:8000/health")
        print("🔄 Press Ctrl+C to stop")
        print()

        subprocess.run(cmd, env=env)

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

    return True

def main():
    """Main launcher function."""
    print("🏠 LegacyCoinTrader 2.0 Launcher")
    print("=" * 40)

    # Setup environment
    setup_environment()

    # Check requirements
    if not check_requirements():
        return

    # Try to launch
    try:
        # First try direct import method
        success = launch_server()
        if not success:
            # Fallback to uvicorn command
            print("\n🔄 Trying alternative launch method...")
            launch_with_uvicorn()
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the virtual environment:")
        print("   source modern_trader_env/bin/activate")
        print("2. Install dependencies:")
        print("   pip install -r requirements-modern.txt")
        print("3. Check Python path:")
        print(f"   PYTHONPATH={os.environ.get('PYTHONPATH', 'not set')}")

if __name__ == "__main__":
    main()
