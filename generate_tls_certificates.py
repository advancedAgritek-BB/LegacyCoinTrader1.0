#!/usr/bin/env python3
"""
TLS Certificate Generation Script for LegacyCoinTrader API Gateway

This script generates self-signed TLS certificates for HTTPS support
in development and testing environments.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path


def generate_tls_certificates(
    cert_dir: str = "certs",
    cert_file: str = "server.crt",
    key_file: str = "server.key",
    ca_file: str = "ca.crt",
    validity_days: int = 365,
    common_name: str = "localhost",
    organization: str = "LegacyCoinTrader",
    country: str = "US",
    state: str = "CA",
    city: str = "San Francisco"
) -> dict[str, str]:
    """
    Generate TLS certificates for HTTPS support.

    Args:
        cert_dir: Directory to store certificates
        cert_file: Server certificate filename
        key_file: Private key filename
        ca_file: CA certificate filename
        validity_days: Certificate validity period
        common_name: Certificate common name (domain)
        organization: Organization name
        country: Country code
        state: State/province
        city: City

    Returns:
        Dictionary with certificate file paths
    """

    # Create certificates directory
    cert_path = Path(cert_dir)
    cert_path.mkdir(parents=True, exist_ok=True)

    cert_path_full = cert_path / cert_file
    key_path_full = cert_path / key_file
    ca_path_full = cert_path / ca_file

    print(f"🔐 Generating TLS certificates in {cert_path.absolute()}")

    # Generate CA private key
    print("📝 Generating CA private key...")
    ca_key_cmd = [
        "openssl", "genrsa",
        "-out", str(cert_path / "ca.key"),
        "2048"
    ]
    subprocess.run(ca_key_cmd, check=True, capture_output=True)

    # Generate CA certificate
    print("🏛️  Generating CA certificate...")
    ca_cert_cmd = [
        "openssl", "req",
        "-new", "-x509",
        "-days", str(validity_days),
        "-key", str(cert_path / "ca.key"),
        "-sha256",
        "-out", str(ca_path_full),
        "-subj", f"/C={country}/ST={state}/L={city}/O={organization}/CN={organization} CA"
    ]
    subprocess.run(ca_cert_cmd, check=True, capture_output=True)

    # Generate server private key
    print("🔑 Generating server private key...")
    server_key_cmd = [
        "openssl", "genrsa",
        "-out", str(key_path_full),
        "2048"
    ]
    subprocess.run(server_key_cmd, check=True, capture_output=True)

    # Generate certificate signing request
    print("📄 Generating certificate signing request...")
    csr_cmd = [
        "openssl", "req",
        "-subj", f"/C={country}/ST={state}/L={city}/O={organization}/CN={common_name}",
        "-new",
        "-key", str(key_path_full),
        "-out", str(cert_path / "server.csr")
    ]
    subprocess.run(csr_cmd, check=True, capture_output=True)

    # Generate server certificate
    print("📜 Generating server certificate...")
    server_cert_cmd = [
        "openssl", "x509",
        "-req",
        "-days", str(validity_days),
        "-in", str(cert_path / "server.csr"),
        "-CA", str(ca_path_full),
        "-CAkey", str(cert_path / "ca.key"),
        "-CAcreateserial",
        "-out", str(cert_path_full),
        "-sha256",
        "-extfile", "/dev/stdin",
        "-extensions", "v3_req"
    ]

    # Extension configuration for server certificate
    extensions = """
[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = api-gateway
IP.1 = 127.0.0.1
IP.2 = 0.0.0.0
"""

    process = subprocess.run(
        server_cert_cmd,
        input=extensions,
        text=True,
        check=True,
        capture_output=True
    )

    # Set appropriate permissions
    os.chmod(key_path_full, 0o600)
    os.chmod(cert_path_full, 0o644)
    os.chmod(ca_path_full, 0o644)

    # Clean up temporary files
    csr_file = cert_path / "server.csr"
    ca_key_file = cert_path / "ca.key"
    ca_serial_file = cert_path / "ca.srl"

    for temp_file in [csr_file, ca_key_file, ca_serial_file]:
        if temp_file.exists():
            temp_file.unlink()

    certificate_info = {
        "cert_file": str(cert_path_full),
        "key_file": str(key_path_full),
        "ca_file": str(ca_path_full),
        "cert_dir": str(cert_path),
        "valid_until": (datetime.now() + timedelta(days=validity_days)).isoformat(),
        "common_name": common_name,
        "organization": organization
    }

    print("✅ TLS certificates generated successfully!")
    print(f"   📜 Certificate: {cert_path_full}")
    print(f"   🔑 Private Key: {key_path_full}")
    print(f"   🏛️  CA Certificate: {ca_path_full}")
    print(f"   ⏰ Valid until: {certificate_info['valid_until']}")
    print()
    print("⚠️  WARNING: These are self-signed certificates for development only!")
    print("   For production, use certificates from a trusted Certificate Authority.")

    return certificate_info


def create_docker_compose_tls_config(cert_info: dict[str, str]) -> str:
    """Generate Docker Compose TLS configuration snippet."""

    config = f"""
    # TLS Configuration for API Gateway
    environment:
      - TLS_ENABLED=true
      - TLS_CERT_FILE=/app/{Path(cert_info['cert_file']).name}
      - TLS_KEY_FILE=/app/{Path(cert_info['key_file']).name}
      - TLS_CA_FILE=/app/{Path(cert_info['ca_file']).name}
    volumes:
      - {cert_info['cert_dir']}:/app/certs:ro
    ports:
      - "443:8443"  # HTTPS port
      - "80:8000"   # HTTP redirect to HTTPS
"""

    return config


def main():
    """Main function to generate certificates."""

    import argparse

    parser = argparse.ArgumentParser(description="Generate TLS certificates for LegacyCoinTrader")
    parser.add_argument(
        "--cert-dir",
        default="certs",
        help="Directory to store certificates"
    )
    parser.add_argument(
        "--common-name",
        default="localhost",
        help="Certificate common name (domain)"
    )
    parser.add_argument(
        "--organization",
        default="LegacyCoinTrader",
        help="Organization name"
    )
    parser.add_argument(
        "--validity-days",
        type=int,
        default=365,
        help="Certificate validity period in days"
    )
    parser.add_argument(
        "--generate-docker-config",
        action="store_true",
        help="Generate Docker Compose TLS configuration"
    )

    args = parser.parse_args()

    try:
        # Check if OpenSSL is available
        subprocess.run(["openssl", "version"], check=True, capture_output=True)
        print("✅ OpenSSL found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ OpenSSL not found. Please install OpenSSL to generate certificates.")
        print("   macOS: brew install openssl")
        print("   Ubuntu: apt-get install openssl")
        return 1

    # Generate certificates
    cert_info = generate_tls_certificates(
        cert_dir=args.cert_dir,
        common_name=args.common_name,
        organization=args.organization,
        validity_days=args.validity_days
    )

    # Generate Docker Compose configuration if requested
    if args.generate_docker_config:
        print("\n🐳 Docker Compose TLS Configuration:")
        print("=" * 50)
        docker_config = create_docker_compose_tls_config(cert_info)
        print(docker_config)

        # Save to file
        config_file = Path(args.cert_dir) / "docker-compose-tls.yml"
        with open(config_file, 'w') as f:
            f.write(docker_config)
        print(f"💾 Configuration saved to: {config_file}")

    print("\n🎯 Next Steps:")
    print("1. Add certificates to your Docker Compose configuration")
    print("2. Update API gateway settings to enable TLS")
    print("3. Restart the API gateway service")
    print("4. Test HTTPS connection: https://localhost")

    return 0


if __name__ == "__main__":
    exit(main())
