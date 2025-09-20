from __future__ import annotations

import os
import ssl
import uvicorn
from pathlib import Path
from typing import Optional

from .main import app


def create_ssl_context(cert_file: str, key_file: str, ca_file: Optional[str] = None) -> ssl.SSLContext:
    """Create SSL context for HTTPS support."""
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)

    # Load server certificate and private key
    ssl_context.load_cert_chain(cert_file, key_file)

    # Load CA certificate if provided
    if ca_file and Path(ca_file).exists():
        ssl_context.load_verify_locations(ca_file)

    # Configure SSL settings - only disable verification in development
    dev_mode = os.getenv("TLS_DEV_MODE", "false").lower() == "true"
    if dev_mode:
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        print("⚠️  TLS running in development mode - certificate verification disabled")

    return ssl_context


if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("GATEWAY_HOST", "0.0.0.0")
    port = int(os.getenv("GATEWAY_PORT", "8000"))
    tls_enabled = os.getenv("TLS_ENABLED", "false").lower() == "true"
    cert_file = os.getenv("TLS_CERT_FILE")
    key_file = os.getenv("TLS_KEY_FILE")
    ca_file = os.getenv("TLS_CA_FILE")

    # Configure SSL if TLS is enabled
    ssl_context = None
    if tls_enabled and cert_file and key_file:
        try:
            ssl_context = create_ssl_context(cert_file, key_file, ca_file)
            print(f"🔐 TLS/HTTPS enabled - Certificate: {cert_file}")
            # Use HTTPS port if TLS is enabled
            port = int(os.getenv("TLS_PORT", "8443"))
        except Exception as e:
            print(f"❌ Failed to configure TLS: {e}")
            print("   Falling back to HTTP...")
            tls_enabled = False

    if tls_enabled:
        print(f"🚀 Starting API Gateway with HTTPS on {host}:{port}")
        uvicorn.run(
            "services.api_gateway.main:app",
            host=host,
            port=port,
            ssl=ssl_context,
            log_level="info"
        )
    else:
        print(f"🚀 Starting API Gateway with HTTP on {host}:{port}")
        uvicorn.run(
            "services.api_gateway.main:app",
            host=host,
            port=port,
            log_level="info"
        )
