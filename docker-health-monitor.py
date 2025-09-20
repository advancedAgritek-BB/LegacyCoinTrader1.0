#!/usr/bin/env python3
"""
Enhanced Docker Health Monitor for LegacyCoinTrader
Provides comprehensive health checking and startup validation for Docker services.
"""

import asyncio
import json
import time
import sys
import subprocess
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ServiceHealth:
    """Health status for a service."""
    name: str
    status: str  # starting, healthy, unhealthy, failed
    http_status: Optional[int] = None
    response_time_ms: Optional[float] = None
    dependencies_ready: bool = False
    startup_time_seconds: Optional[float] = None
    last_check: Optional[str] = None
    error_message: Optional[str] = None
    container_id: Optional[str] = None
    container_status: Optional[str] = None

class DockerHealthMonitor:
    """Monitor Docker services health and startup."""
    
    def __init__(self):
        self.services = {
            'redis': {'port': 6379, 'path': None, 'dependencies': []},
            'postgres': {'port': 5432, 'path': None, 'dependencies': []},
            'api-gateway': {'port': 8000, 'path': '/health', 'dependencies': ['redis', 'postgres']},
            'trading-engine': {'port': 8001, 'path': '/health', 'dependencies': ['redis', 'api-gateway']},
            'market-data': {'port': 8002, 'path': '/health', 'dependencies': ['redis', 'api-gateway']},
            'portfolio': {'port': 8003, 'path': '/health', 'dependencies': ['redis', 'api-gateway']},
            'strategy-engine': {'port': 8004, 'path': '/health', 'dependencies': ['redis', 'api-gateway', 'market-data']},
            'token-discovery': {'port': 8005, 'path': '/health', 'dependencies': ['redis', 'api-gateway', 'market-data']},
            'execution': {'port': 8006, 'path': '/health', 'dependencies': ['redis', 'api-gateway']},
            'monitoring': {'port': 8007, 'path': '/health', 'dependencies': ['redis', 'api-gateway']},
            'frontend': {'port': 5000, 'path': '/health', 'dependencies': ['api-gateway']},
        }
        self.startup_times = {}
        self.service_status = {}
        
    def get_docker_container_info(self, service_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Get Docker container ID and status."""
        try:
            # Get container info using docker-compose
            result = subprocess.run([
                'docker-compose', 'ps', '-q', service_name
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0 or not result.stdout.strip():
                return None, None
                
            container_id = result.stdout.strip()
            
            # Get container status
            result = subprocess.run([
                'docker', 'inspect', container_id, '--format', '{{.State.Status}}'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return container_id, result.stdout.strip()
            
            return container_id, None
            
        except Exception as e:
            logger.debug(f"Failed to get container info for {service_name}: {e}")
            return None, None
    
    async def check_service_health(self, service_name: str, config: Dict) -> ServiceHealth:
        """Check health of a specific service."""
        start_time = time.time()
        container_id, container_status = self.get_docker_container_info(service_name)
        
        health = ServiceHealth(
            name=service_name,
            status='starting',
            container_id=container_id,
            container_status=container_status,
            last_check=datetime.now().isoformat()
        )
        
        # Check if container is running
        if container_status != 'running':
            health.status = 'failed'
            health.error_message = f"Container not running: {container_status}"
            return health
        
        # Check dependencies first
        health.dependencies_ready = await self.check_dependencies(service_name)
        if not health.dependencies_ready:
            health.status = 'unhealthy'
            health.error_message = "Dependencies not ready"
            return health
        
        # For database services, use special checks
        if service_name == 'redis':
            return await self.check_redis_health(health)
        elif service_name == 'postgres':
            return await self.check_postgres_health(health)
        
        # For HTTP services, check endpoint
        if config['path']:
            try:
                url = f"http://localhost:{config['port']}{config['path']}"
                response = requests.get(url, timeout=5)
                
                response_time = (time.time() - start_time) * 1000
                health.response_time_ms = round(response_time, 2)
                health.http_status = response.status_code
                
                if response.status_code == 200:
                    # Additional health validation for specific services
                    if await self.validate_service_specific_health(service_name, response):
                        health.status = 'healthy'
                    else:
                        health.status = 'unhealthy'
                        health.error_message = "Service-specific health check failed"
                else:
                    health.status = 'unhealthy'
                    health.error_message = f"HTTP {response.status_code}"
                    
            except requests.exceptions.RequestException as e:
                health.status = 'unhealthy'
                health.error_message = f"HTTP request failed: {str(e)}"
                
        else:
            # For services without HTTP endpoints
            health.status = 'healthy' if container_status == 'running' else 'unhealthy'
        
        return health
    
    async def check_redis_health(self, health: ServiceHealth) -> ServiceHealth:
        """Check Redis-specific health."""
        try:
            result = subprocess.run([
                'docker-compose', 'exec', '-T', 'redis', 'redis-cli', 'ping'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and 'PONG' in result.stdout:
                health.status = 'healthy'
            else:
                health.status = 'unhealthy'
                health.error_message = "Redis ping failed"
                
        except Exception as e:
            health.status = 'unhealthy'
            health.error_message = f"Redis check failed: {e}"
            
        return health
    
    async def check_postgres_health(self, health: ServiceHealth) -> ServiceHealth:
        """Check PostgreSQL-specific health."""
        try:
            result = subprocess.run([
                'docker-compose', 'exec', '-T', 'postgres', 'pg_isready', 
                '-U', 'legacy_user', '-d', 'legacy_coin_trader'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                health.status = 'healthy'
            else:
                health.status = 'unhealthy'
                health.error_message = "PostgreSQL not ready"
                
        except Exception as e:
            health.status = 'unhealthy'
            health.error_message = f"PostgreSQL check failed: {e}"
            
        return health
    
    async def validate_service_specific_health(self, service_name: str, response: requests.Response) -> bool:
        """Validate service-specific health beyond HTTP 200."""
        try:
            data = response.json()

            status_value = str(data.get('status', '')).lower()
            is_basic_healthy = status_value in {'healthy', 'ok', 'ready'}

            if service_name == 'api-gateway':
                # Gateway should report downstream service summary when it is up
                return is_basic_healthy and 'services' in data

            elif service_name == 'market-data':
                # Treat missing connectivity flag as "unknown but healthy"
                if 'kraken_connected' in data:
                    return is_basic_healthy and bool(data.get('kraken_connected'))
                return is_basic_healthy

            elif service_name == 'trading-engine':
                if 'scanner_active' in data:
                    return is_basic_healthy and bool(data.get('scanner_active'))
                return is_basic_healthy

            elif service_name == 'portfolio':
                if 'database_connected' in data:
                    return is_basic_healthy and bool(data.get('database_connected'))
                return is_basic_healthy

            # Default validation - fall back to interpreted status flag
            return is_basic_healthy

        except (ValueError, KeyError):
            # If response isn't JSON or doesn't have expected structure, 
            # fall back to HTTP status code
            return True
    
    async def check_dependencies(self, service_name: str) -> bool:
        """Check if service dependencies are ready."""
        dependencies = self.services[service_name]['dependencies']
        
        for dep in dependencies:
            if dep not in self.service_status:
                return False
            if self.service_status[dep].status != 'healthy':
                return False
                
        return True
    
    async def check_all_services(self) -> Dict[str, ServiceHealth]:
        """Check health of all services."""
        results = {}
        
        # Check services in dependency order
        dependency_order = self.get_startup_order()
        
        for service_name in dependency_order:
            config = self.services[service_name]
            health = await self.check_service_health(service_name, config)
            results[service_name] = health
            self.service_status[service_name] = health
            
        return results
    
    def get_startup_order(self) -> List[str]:
        """Get services in startup dependency order."""
        # Topological sort of dependencies
        order = []
        visited = set()
        temp_visited = set()
        
        def visit(service):
            if service in temp_visited:
                raise ValueError(f"Circular dependency detected involving {service}")
            if service in visited:
                return
                
            temp_visited.add(service)
            
            for dep in self.services[service]['dependencies']:
                visit(dep)
                
            temp_visited.remove(service)
            visited.add(service)
            order.append(service)
        
        for service in self.services:
            if service not in visited:
                visit(service)
                
        return order
    
    async def wait_for_healthy_startup(self, timeout_minutes: int = 10) -> bool:
        """Wait for all services to be healthy."""
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        logger.info(f"🚀 Waiting for services to be healthy (timeout: {timeout_minutes}m)")
        
        while time.time() - start_time < timeout_seconds:
            results = await self.check_all_services()
            
            healthy_count = sum(1 for h in results.values() if h.status == 'healthy')
            total_count = len(results)
            
            logger.info(f"📊 Services: {healthy_count}/{total_count} healthy")
            
            if healthy_count == total_count:
                logger.info("✅ All services are healthy!")
                return True
            
            # Show status of unhealthy services
            for name, health in results.items():
                if health.status != 'healthy':
                    logger.info(f"  ⚠️ {name}: {health.status} - {health.error_message or 'No error'}")
            
            await asyncio.sleep(10)  # Check every 10 seconds
        
        logger.error(f"❌ Timeout after {timeout_minutes} minutes")
        return False
    
    def generate_status_report(self, results: Dict[str, ServiceHealth]) -> str:
        """Generate a formatted status report."""
        report = []
        report.append("🐳 Docker Services Health Report")
        report.append("=" * 50)
        report.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        healthy = sum(1 for h in results.values() if h.status == 'healthy')
        total = len(results)
        report.append(f"📊 Overall Status: {healthy}/{total} services healthy")
        report.append("")
        
        # Individual service status
        report.append("🔍 Service Details:")
        report.append("-" * 30)
        
        for service_name in self.get_startup_order():
            health = results.get(service_name)
            if not health:
                continue
                
            status_icon = {
                'healthy': '🟢',
                'unhealthy': '🔴', 
                'starting': '🟡',
                'failed': '💀'
            }.get(health.status, '❓')
            
            report.append(f"{status_icon} {service_name}")
            report.append(f"   Status: {health.status}")
            report.append(f"   Container: {health.container_status or 'unknown'}")
            
            if health.http_status:
                report.append(f"   HTTP: {health.http_status} ({health.response_time_ms}ms)")
            
            if health.error_message:
                report.append(f"   Error: {health.error_message}")
                
            if not health.dependencies_ready:
                deps = self.services[service_name]['dependencies']
                report.append(f"   Dependencies: {', '.join(deps)} not ready")
                
            report.append("")
        
        return "\n".join(report)
    
    async def save_health_report(self, results: Dict[str, ServiceHealth], filename: str = None):
        """Save health report to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"docker_health_report_{timestamp}.json"
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'services': {name: asdict(health) for name, health in results.items()},
            'summary': {
                'total_services': len(results),
                'healthy_services': sum(1 for h in results.values() if h.status == 'healthy'),
                'unhealthy_services': sum(1 for h in results.values() if h.status != 'healthy')
            }
        }
        
        report_path = Path('logs') / filename
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"📄 Health report saved to: {report_path}")

async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Docker Health Monitor")
    parser.add_argument("--wait", "-w", action="store_true", help="Wait for all services to be healthy")
    parser.add_argument("--timeout", "-t", type=int, default=10, help="Timeout in minutes (default: 10)")
    parser.add_argument("--report", "-r", action="store_true", help="Generate and save health report")
    parser.add_argument("--watch", type=int, metavar="SECONDS", help="Watch mode - refresh every N seconds")
    
    args = parser.parse_args()
    
    monitor = DockerHealthMonitor()
    
    try:
        if args.wait:
            success = await monitor.wait_for_healthy_startup(args.timeout)
            sys.exit(0 if success else 1)
            
        elif args.watch:
            logger.info(f"👁️ Watch mode - refreshing every {args.watch} seconds")
            while True:
                results = await monitor.check_all_services()
                print("\033[2J\033[H")  # Clear screen
                print(monitor.generate_status_report(results))
                
                if args.report:
                    await monitor.save_health_report(results)
                
                await asyncio.sleep(args.watch)
                
        else:
            results = await monitor.check_all_services()
            print(monitor.generate_status_report(results))
            
            if args.report:
                await monitor.save_health_report(results)
    
    except KeyboardInterrupt:
        logger.info("👋 Health monitoring stopped")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
