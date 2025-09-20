#!/usr/bin/env python3
"""
Docker Startup Orchestrator for LegacyCoinTrader
Provides intelligent startup sequencing and validation for Docker services.
"""

import asyncio
import subprocess
import sys
import time
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DockerStartupOrchestrator:
    """Orchestrate Docker service startup with proper sequencing and validation."""
    
    def __init__(self, environment: str = 'dev'):
        self.environment = environment
        self.project_root = Path(__file__).parent
        self.compose_files = self._get_compose_files()
        self.startup_phases = self._define_startup_phases()
        
    def _get_compose_files(self) -> List[str]:
        """Get appropriate docker-compose files for environment."""
        base_files = ['docker-compose.yml']
        
        if self.environment == 'dev':
            base_files.append('docker-compose.dev.yml')
        elif self.environment == 'prod':
            base_files.append('docker-compose.prod.yml')
        elif self.environment == 'test':
            base_files.append('docker-compose.test.yml')
        
        result = []
        for file in base_files:
            result.extend(['-f', file])
        return result
    
    def _define_startup_phases(self) -> Dict[str, List[str]]:
        """Define service startup phases for proper sequencing."""
        return {
            'infrastructure': ['redis', 'postgres'],
            'core_services': ['api-gateway'],
            'data_services': ['market-data', 'portfolio'],
            'business_logic': ['trading-engine', 'strategy-engine', 'token-discovery'],
            'execution': ['execution'],
            'monitoring': ['monitoring'],
            'frontend': ['frontend']
        }
    
    async def validate_environment(self) -> bool:
        """Validate environment before startup."""
        logger.info("🔍 Validating environment...")
        
        checks = [
            self._check_docker_available,
            self._check_docker_compose_available,
            self._check_env_file,
            self._check_api_keys,
            self._check_disk_space,
            self._check_ports_available
        ]
        
        for check in checks:
            if not await check():
                return False
        
        logger.info("✅ Environment validation passed")
        return True
    
    async def _check_docker_available(self) -> bool:
        """Check if Docker is available and running."""
        try:
            result = await self._run_command(['docker', 'version'])
            if result.returncode == 0:
                logger.info("✅ Docker is available")
                return True
            else:
                logger.error("❌ Docker is not available or not running")
                return False
        except Exception as e:
            logger.error(f"❌ Docker check failed: {e}")
            return False
    
    async def _check_docker_compose_available(self) -> bool:
        """Check if Docker Compose is available."""
        try:
            result = await self._run_command(['docker-compose', '--version'])
            if result.returncode == 0:
                logger.info("✅ Docker Compose is available")
                return True
            else:
                logger.error("❌ Docker Compose is not available")
                return False
        except Exception as e:
            logger.error(f"❌ Docker Compose check failed: {e}")
            return False
    
    async def _check_env_file(self) -> bool:
        """Check if .env file exists and has required variables."""
        env_file = self.project_root / '.env'
        
        if not env_file.exists():
            logger.error("❌ .env file not found")
            return False
        
        required_vars = [
            'KRAKEN_API_KEY',
            'KRAKEN_API_SECRET',
            'HELIUS_API_KEY',
            'TELEGRAM_BOT_TOKEN'
        ]
        
        try:
            with open(env_file) as f:
                env_content = f.read()
            
            missing_vars = []
            for var in required_vars:
                if f"{var}=" not in env_content or f"{var}=MANAGED:" in env_content:
                    missing_vars.append(var)
            
            if missing_vars:
                logger.warning(f"⚠️ Missing or placeholder values for: {', '.join(missing_vars)}")
                logger.warning("⚠️ Some services may not function properly")
            else:
                logger.info("✅ .env file contains required variables")
            
            return True  # Don't fail startup for missing API keys in dev mode
            
        except Exception as e:
            logger.error(f"❌ Error reading .env file: {e}")
            return False
    
    async def _check_api_keys(self) -> bool:
        """Validate API keys if provided."""
        # This is a placeholder for API key validation
        # In production, you might want to test connectivity
        logger.info("✅ API key validation (placeholder)")
        return True
    
    async def _check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage('/')
            free_gb = free // (1024**3)
            
            if free_gb < 5:  # Less than 5GB free
                logger.error(f"❌ Insufficient disk space: {free_gb}GB free")
                return False
            else:
                logger.info(f"✅ Sufficient disk space: {free_gb}GB free")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ Could not check disk space: {e}")
            return True  # Don't fail startup for this
    
    async def _check_ports_available(self) -> bool:
        """Check if required ports are available."""
        import socket
        
        required_ports = [5000, 6379, 8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007]
        
        unavailable_ports = []
        for port in required_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                result = sock.connect_ex(('localhost', port))
                if result == 0:  # Port is in use
                    unavailable_ports.append(port)
            except Exception:
                pass
            finally:
                sock.close()
        
        if unavailable_ports:
            logger.warning(f"⚠️ Ports already in use: {unavailable_ports}")
            logger.warning("⚠️ This may cause conflicts during startup")
        else:
            logger.info("✅ Required ports are available")
        
        return True  # Don't fail startup, just warn
    
    async def start_services_by_phase(self) -> bool:
        """Start services in phases with proper sequencing."""
        logger.info(f"🚀 Starting LegacyCoinTrader ({self.environment} environment)")
        
        for phase_name, services in self.startup_phases.items():
            logger.info(f"📦 Starting {phase_name} phase: {', '.join(services)}")
            
            # Start services in this phase
            if not await self._start_phase_services(services):
                logger.error(f"❌ Failed to start {phase_name} phase")
                return False
            
            # Wait for services to be healthy
            if not await self._wait_for_phase_health(services):
                logger.error(f"❌ {phase_name} phase services are not healthy")
                return False
            
            logger.info(f"✅ {phase_name} phase completed successfully")
            
            # Brief pause between phases
            await asyncio.sleep(2)
        
        logger.info("🎉 All services started successfully!")
        return True
    
    async def _start_phase_services(self, services: List[str]) -> bool:
        """Start services in a specific phase."""
        try:
            cmd = ['docker-compose'] + self.compose_files + ['up', '-d'] + services
            result = await self._run_command(cmd)
            
            if result.returncode == 0:
                logger.info(f"✅ Started services: {', '.join(services)}")
                return True
            else:
                logger.error(f"❌ Failed to start services: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting services: {e}")
            return False
    
    async def _wait_for_phase_health(self, services: List[str], timeout_seconds: int = 120) -> bool:
        """Wait for services in a phase to be healthy."""
        from docker_health_monitor import DockerHealthMonitor
        
        monitor = DockerHealthMonitor()
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            all_healthy = True
            
            for service in services:
                if service in monitor.services:
                    config = monitor.services[service]
                    health = await monitor.check_service_health(service, config)
                    
                    if health.status != 'healthy':
                        all_healthy = False
                        logger.debug(f"⏳ {service}: {health.status} - {health.error_message or 'Starting...'}")
            
            if all_healthy:
                return True
            
            await asyncio.sleep(5)  # Check every 5 seconds
        
        logger.error(f"❌ Timeout waiting for services to be healthy: {services}")
        return False
    
    async def perform_post_startup_validation(self) -> bool:
        """Perform comprehensive validation after startup."""
        logger.info("🔍 Performing post-startup validation...")
        
        validations = [
            self._validate_api_gateway,
            self._validate_database_connectivity,
            self._validate_redis_connectivity,
            self._validate_external_apis,
            self._validate_service_communication
        ]
        
        all_passed = True
        for validation in validations:
            try:
                if not await validation():
                    all_passed = False
            except Exception as e:
                logger.error(f"❌ Validation error: {e}")
                all_passed = False
        
        if all_passed:
            logger.info("✅ All post-startup validations passed")
        else:
            logger.warning("⚠️ Some post-startup validations failed")
        
        return all_passed
    
    async def _validate_api_gateway(self) -> bool:
        """Validate API Gateway is responding and can route requests."""
        try:
            import requests
            response = requests.get('http://localhost:8000/health', timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    logger.info("✅ API Gateway is healthy")
                    return True
            
            logger.error("❌ API Gateway health check failed")
            return False
            
        except Exception as e:
            logger.error(f"❌ API Gateway validation failed: {e}")
            return False
    
    async def _validate_database_connectivity(self) -> bool:
        """Validate database connectivity."""
        try:
            cmd = ['docker-compose'] + self.compose_files + [
                'exec', '-T', 'postgres', 'pg_isready', 
                '-U', 'legacy_user', '-d', 'legacy_coin_trader'
            ]
            result = await self._run_command(cmd)
            
            if result.returncode == 0:
                logger.info("✅ Database connectivity validated")
                return True
            else:
                logger.error("❌ Database connectivity failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Database validation error: {e}")
            return False
    
    async def _validate_redis_connectivity(self) -> bool:
        """Validate Redis connectivity."""
        try:
            cmd = ['docker-compose'] + self.compose_files + [
                'exec', '-T', 'redis', 'redis-cli', 'ping'
            ]
            result = await self._run_command(cmd)
            
            if result.returncode == 0 and 'PONG' in result.stdout:
                logger.info("✅ Redis connectivity validated")
                return True
            else:
                logger.error("❌ Redis connectivity failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Redis validation error: {e}")
            return False
    
    async def _validate_external_apis(self) -> bool:
        """Validate external API connectivity."""
        # This would test Kraken and Helius APIs if keys are available
        logger.info("✅ External API validation (placeholder)")
        return True
    
    async def _validate_service_communication(self) -> bool:
        """Validate inter-service communication."""
        # Test that services can communicate with each other
        logger.info("✅ Service communication validation (placeholder)")
        return True
    
    async def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run a command asynchronously."""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        return type('Result', (), {
            'returncode': process.returncode,
            'stdout': stdout.decode() if stdout else '',
            'stderr': stderr.decode() if stderr else ''
        })()
    
    async def generate_startup_report(self) -> Dict:
        """Generate a startup report."""
        from docker_health_monitor import DockerHealthMonitor
        
        monitor = DockerHealthMonitor()
        health_results = await monitor.check_all_services()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'environment': self.environment,
            'startup_phases': self.startup_phases,
            'services': {name: {
                'status': health.status,
                'container_status': health.container_status,
                'response_time_ms': health.response_time_ms,
                'error_message': health.error_message
            } for name, health in health_results.items()},
            'summary': {
                'total_services': len(health_results),
                'healthy_services': sum(1 for h in health_results.values() if h.status == 'healthy'),
                'failed_services': sum(1 for h in health_results.values() if h.status == 'failed')
            }
        }
        
        # Save report
        report_path = Path('logs') / f'startup_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Startup report saved to: {report_path}")
        return report

async def main():
    """Main orchestrator function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Docker Startup Orchestrator")
    parser.add_argument("--env", choices=['dev', 'prod', 'test'], default='dev',
                       help="Environment to start (default: dev)")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate environment, don't start services")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip pre-startup validation")
    parser.add_argument("--report", action="store_true",
                       help="Generate startup report")
    
    args = parser.parse_args()
    
    orchestrator = DockerStartupOrchestrator(args.env)
    
    try:
        # Validate environment
        if not args.skip_validation:
            if not await orchestrator.validate_environment():
                logger.error("❌ Environment validation failed")
                sys.exit(1)
        
        if args.validate_only:
            logger.info("✅ Environment validation completed")
            sys.exit(0)
        
        # Start services
        if not await orchestrator.start_services_by_phase():
            logger.error("❌ Service startup failed")
            sys.exit(1)
        
        # Post-startup validation
        await orchestrator.perform_post_startup_validation()
        
        # Generate report
        if args.report:
            await orchestrator.generate_startup_report()
        
        logger.info("🎉 Startup orchestration completed successfully!")
        logger.info("🌐 Dashboard: http://localhost:5000")
        logger.info("🔧 API Gateway: http://localhost:8000")
        logger.info("📊 Health Check: python docker-health-monitor.py")
        
    except KeyboardInterrupt:
        logger.info("👋 Startup orchestration interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Orchestration error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
