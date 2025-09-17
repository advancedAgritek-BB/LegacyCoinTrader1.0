#!/usr/bin/env python3
"""
Production Deployment Script for LegacyCoinTrader

Provides comprehensive deployment, monitoring, and management capabilities
for production trading operations.
"""

import asyncio
import subprocess
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from services.configuration import (
    DEFAULT_MANIFEST_PATH,
    ManagedConfigService,
    load_manifest,
)

logger = setup_logger(__name__, LOG_DIR / "production_deployment.log")


class ProductionDeployment:
    """
    Production deployment manager for LegacyCoinTrader.

    Features:
    - Automated deployment and startup
    - Health monitoring and alerting
    - Configuration management
    - Backup and recovery
    - Performance optimization
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("production_config.yaml")
        self.config = self._load_config()
        self.project_root = Path(__file__).parent

        self.manifest_path = self._resolve_manifest_path()
        self._managed_manifest = load_manifest(self.manifest_path)
        self._managed_config_service = ManagedConfigService(
            manifest=self._managed_manifest
        )
        logger.debug("Using managed secrets manifest at %s", self.manifest_path)

        # Process management
        self.bot_process = None
        self.monitor_process = None
        self.web_process = None

        # Deployment state
        self.is_deployed = False
        self.deployment_time = None

        logger.info("Production deployment manager initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load production configuration."""
        if self.config_path.exists():
            try:
                import yaml
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Failed to load config: {e}")

        # Default configuration
        return {
            "deployment": {
                "auto_start": True,
                "health_checks": True,
                "monitoring": True,
                "backup_enabled": True
            },
            "processes": {
                "bot": {"enabled": True, "script": "start_bot.py auto"},
                "monitor": {"enabled": True, "script": "production_monitor.py"},
                "web": {"enabled": True, "script": "start_frontend.py"}
            },
            "health_checks": {
                "interval": 60,
                "timeout": 30,
                "retries": 3
            },
            "monitoring": {
                "alerts": True,
                "metrics": True,
                "log_rotation": True
            }
        }

    def _resolve_manifest_path(self) -> Path:
        """Return the managed secrets manifest path for this deployment."""

        manifest_setting = (
            self.config.get("secret_management", {}).get("manifest")
            or self.config.get("environment", {}).get("manifest")
        )
        if manifest_setting:
            candidate = Path(manifest_setting)
            if not candidate.is_absolute():
                candidate = (self.project_root / candidate).resolve()
            return candidate
        return DEFAULT_MANIFEST_PATH

    async def deploy(self) -> bool:
        """
        Deploy the production system.

        Returns:
            True if deployment successful, False otherwise
        """
        logger.info("Starting production deployment...")

        try:
            # Pre-deployment checks
            if not await self._pre_deployment_checks():
                logger.error("Pre-deployment checks failed")
                return False

            # Create production environment
            await self._setup_production_environment()

            # Start services
            success = await self._start_services()

            if success:
                self.is_deployed = True
                self.deployment_time = datetime.now()
                logger.info("Production deployment completed successfully")

                # Start health monitoring
                if self.config.get("deployment", {}).get("health_checks", True):
                    asyncio.create_task(self._health_monitoring_loop())

                return True
            else:
                logger.error("Service startup failed")
                await self.rollback()
                return False

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            await self.rollback()
            return False

    async def _pre_deployment_checks(self) -> bool:
        """Perform pre-deployment health checks."""
        logger.info("Performing pre-deployment checks...")

        checks = [
            self._check_dependencies,
            self._check_configuration,
            self._check_environment,
            self._check_ports,
            self._check_permissions
        ]

        for check in checks:
            try:
                if not await check():
                    return False
            except Exception as e:
                logger.error(f"Check failed: {e}")
                return False

        logger.info("All pre-deployment checks passed")
        return True

    async def _check_dependencies(self) -> bool:
        """Check if all required dependencies are installed."""
        required_modules = [
            'ccxt', 'pandas', 'numpy', 'aiohttp', 'psutil',
            'cryptography', 'python-dotenv', 'pyyaml'
        ]

        missing = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)

        if missing:
            logger.error(f"Missing dependencies: {', '.join(missing)}")
            logger.error("Install with: pip install " + " ".join(missing))
            return False

        logger.info("All dependencies available")
        return True

    async def _check_configuration(self) -> bool:
        """Check if configuration is valid."""
        missing_env = self._managed_config_service.missing_environment()
        if missing_env:
            logger.error(
                "Missing managed secrets: %s",
                ", ".join(sorted(missing_env)),
            )
            logger.error("Populate required secrets via the configured secret manager.")
            return False

        if not self._check_secret_rotation_policy():
            return False

        if not self.config_path.exists():
            logger.warning(f"Configuration file not found: {self.config_path}")
            logger.info("Using default configuration")

        logger.info("Configuration check passed")
        return True

    def _check_secret_rotation_policy(self) -> bool:
        """Validate that secrets comply with the configured rotation policy."""

        policy = self.config.get("secret_management", {}).get("rotation_policy", {})
        if not policy.get("enabled", False):
            return True

        timestamp_env = policy.get("timestamp_env")
        rotate_every = policy.get("rotate_every_days")
        try:
            rotate_every_days = int(rotate_every) if rotate_every is not None else None
        except (TypeError, ValueError):
            rotate_every_days = None

        if not timestamp_env:
            logger.warning(
                "Secret rotation policy is enabled but no timestamp_env is configured."
            )
            return True

        timestamp_value = os.getenv(timestamp_env)
        if not timestamp_value:
            logger.error(
                "Environment variable %s is required to enforce the secret rotation policy.",
                timestamp_env,
            )
            return False

        try:
            rotated_at = datetime.fromisoformat(timestamp_value)
        except ValueError:
            logger.error(
                "Environment variable %s must be an ISO-8601 timestamp (current value: %s)",
                timestamp_env,
                timestamp_value,
            )
            return False

        if rotated_at.tzinfo is not None:
            rotated_at = rotated_at.astimezone(timezone.utc).replace(tzinfo=None)

        if rotate_every_days is None or rotate_every_days <= 0:
            return True

        age = datetime.utcnow() - rotated_at
        if age.days > rotate_every_days:
            logger.error(
                "Managed secrets were rotated %s days ago on %s; rotation cadence is %s days.",
                age.days,
                rotated_at.isoformat(),
                rotate_every_days,
            )
            return False

        return True

    async def _check_environment(self) -> bool:
        """Check if environment is suitable for production."""
        # Check Python version
        if sys.version_info < (3, 9):
            logger.error("Python 3.9+ required for production")
            return False

        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.total < 4 * 1024 * 1024 * 1024:  # 4GB
                logger.warning("Less than 4GB RAM available - performance may be degraded")
        except ImportError:
            logger.warning("psutil not available - cannot check memory")

        # Check disk space
        try:
            stat = os.statvfs('/')
            free_space = stat.f_bavail * stat.f_frsize
            if free_space < 1 * 1024 * 1024 * 1024:  # 1GB
                logger.error("Less than 1GB free disk space")
                return False
        except:
            logger.warning("Cannot check disk space")

        logger.info("Environment check passed")
        return True

    async def _check_ports(self) -> bool:
        """Check if required ports are available."""
        import socket

        ports_to_check = [8000, 5000]  # Common ports used by the system

        for port in ports_to_check:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    result = s.connect_ex(('localhost', port))
                    if result == 0:
                        logger.warning(f"Port {port} is already in use")
                        # Don't fail - ports might be used by our own services
            except:
                pass

        logger.info("Port availability check completed")
        return True

    async def _check_permissions(self) -> bool:
        """Check if we have necessary file permissions."""
        paths_to_check = [
            LOG_DIR,
            Path("cache"),
            Path("temp"),
            self.config_path
        ]

        for path in paths_to_check:
            try:
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)

                # Try to write a test file
                test_file = path / ".permission_test"
                test_file.write_text("test")
                test_file.unlink()

            except Exception as e:
                logger.error(f"Permission check failed for {path}: {e}")
                return False

        logger.info("Permission check passed")
        return True

    async def _setup_production_environment(self):
        """Set up production environment variables and configuration."""
        logger.info("Setting up production environment...")

        # Set production environment variables
        os.environ['PRODUCTION'] = 'true'
        os.environ['FLASK_ENV'] = 'production'
        os.environ['LOG_LEVEL'] = 'INFO'

        # Production-specific settings
        os.environ['ENABLE_METRICS'] = 'true'
        os.environ['ENABLE_POSITION_SYNC'] = 'true'
        os.environ['ENABLE_MEMORY_MANAGEMENT'] = 'true'

        # Configure Python path
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        logger.info("Production environment configured")

    async def _start_services(self) -> bool:
        """Start all production services."""
        logger.info("Starting production services...")

        services = self.config.get("processes", {})

        # Start trading bot
        if services.get("bot", {}).get("enabled", True):
            if not await self._start_bot():
                return False

        # Start monitoring
        if services.get("monitor", {}).get("enabled", True):
            if not await self._start_monitor():
                return False

        # Start web interface
        if services.get("web", {}).get("enabled", True):
            if not await self._start_web():
                return False

        logger.info("All services started successfully")
        return True

    async def _start_bot(self) -> bool:
        """Start the trading bot."""
        try:
            logger.info("Starting trading bot...")

            script_path = self.project_root / "start_bot.py"
            if not script_path.exists():
                logger.error(f"Bot script not found: {script_path}")
                return False

            # Start bot process
            self.bot_process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_path),
                "auto",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )

            # Wait a bit for startup
            await asyncio.sleep(5)

            if self.bot_process.returncode is None:
                logger.info("Trading bot started successfully")
                return True
            else:
                logger.error("Trading bot failed to start")
                return False

        except Exception as e:
            logger.error(f"Failed to start trading bot: {e}")
            return False

    async def _start_monitor(self) -> bool:
        """Start the production monitor."""
        try:
            logger.info("Starting production monitor...")

            script_path = self.project_root / "production_monitor.py"
            if not script_path.exists():
                logger.error(f"Monitor script not found: {script_path}")
                return False

            # Start monitor process
            self.monitor_process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )

            logger.info("Production monitor started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start production monitor: {e}")
            return False

    async def _start_web(self) -> bool:
        """Start the web interface."""
        try:
            logger.info("Starting web interface...")

            script_path = self.project_root / "start_frontend.py"
            if not script_path.exists():
                logger.error(f"Web script not found: {script_path}")
                return False

            # Start web process
            self.web_process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )

            logger.info("Web interface started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start web interface: {e}")
            return False

    async def _health_monitoring_loop(self):
        """Continuous health monitoring loop."""
        logger.info("Starting health monitoring loop...")

        while self.is_deployed:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.get("health_checks", {}).get("interval", 60))
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)

    async def _perform_health_checks(self):
        """Perform health checks on all services."""
        checks = [
            ("bot", self._check_bot_health),
            ("monitor", self._check_monitor_health),
            ("web", self._check_web_health)
        ]

        for service_name, check_func in checks:
            try:
                health = await check_func()
                if not health:
                    logger.warning(f"Health check failed for {service_name}")
                    await self._handle_service_failure(service_name)
            except Exception as e:
                logger.error(f"Health check error for {service_name}: {e}")

    async def _check_bot_health(self) -> bool:
        """Check if trading bot is healthy."""
        if not self.bot_process or self.bot_process.returncode is not None:
            return False

        # Additional health checks could go here
        return True

    async def _check_monitor_health(self) -> bool:
        """Check if monitor is healthy."""
        if not self.monitor_process or self.monitor_process.returncode is not None:
            return False

        return True

    async def _check_web_health(self) -> bool:
        """Check if web interface is healthy."""
        if not self.web_process or self.web_process.returncode is not None:
            return False

        return True

    async def _handle_service_failure(self, service_name: str):
        """Handle service failure."""
        logger.warning(f"Handling failure for service: {service_name}")

        # Implement recovery logic here
        if service_name == "bot":
            logger.info("Attempting to restart trading bot...")
            await self._start_bot()
        elif service_name == "monitor":
            logger.info("Attempting to restart monitor...")
            await self._start_monitor()
        elif service_name == "web":
            logger.info("Attempting to restart web interface...")
            await self._start_web()

    async def stop(self):
        """Stop the production deployment."""
        logger.info("Stopping production deployment...")

        processes = [
            ("bot", self.bot_process),
            ("monitor", self.monitor_process),
            ("web", self.web_process)
        ]

        for service_name, process in processes:
            if process and process.returncode is None:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=10)
                    logger.info(f"Stopped {service_name} service")
                except Exception as e:
                    logger.error(f"Error stopping {service_name}: {e}")
                    try:
                        process.kill()
                    except:
                        pass

        self.is_deployed = False
        logger.info("Production deployment stopped")

    async def rollback(self):
        """Rollback deployment in case of failure."""
        logger.info("Rolling back deployment...")

        await self.stop()

        # Clean up any temporary files or configurations
        # This could include removing PID files, cleaning caches, etc.

        logger.info("Deployment rollback completed")

    def get_status(self) -> Dict[str, Any]:
        """Get deployment status."""
        return {
            "deployed": self.is_deployed,
            "deployment_time": self.deployment_time.isoformat() if self.deployment_time else None,
            "services": {
                "bot": {
                    "running": self.bot_process and self.bot_process.returncode is None,
                    "pid": self.bot_process.pid if self.bot_process else None
                },
                "monitor": {
                    "running": self.monitor_process and self.monitor_process.returncode is None,
                    "pid": self.monitor_process.pid if self.monitor_process else None
                },
                "web": {
                    "running": self.web_process and self.web_process.returncode is None,
                    "pid": self.web_process.pid if self.web_process else None
                }
            },
            "config": self.config
        }

    async def backup(self, backup_path: Optional[str] = None):
        """Create a backup of the current deployment state."""
        if not self.config.get("deployment", {}).get("backup_enabled", True):
            logger.info("Backups disabled in configuration")
            return

        backup_dir = Path(backup_path) if backup_path else self.project_root / "backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"production_backup_{timestamp}.json"

        try:
            status = self.get_status()
            with open(backup_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)

            logger.info(f"Backup created: {backup_file}")

        except Exception as e:
            logger.error(f"Backup failed: {e}")


async def main():
    """Main deployment function."""
    import argparse

    parser = argparse.ArgumentParser(description="Production Deployment for LegacyCoinTrader")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--deploy", action="store_true", help="Deploy production system")
    parser.add_argument("--stop", action="store_true", help="Stop production system")
    parser.add_argument("--status", action="store_true", help="Show deployment status")
    parser.add_argument("--backup", help="Create backup (optional path)")

    args = parser.parse_args()

    deployment = ProductionDeployment(args.config)

    if args.deploy:
        success = await deployment.deploy()
        if success:
            print("✅ Production deployment successful")
            print("\n📊 Services Status:")
            status = deployment.get_status()
            for service, info in status["services"].items():
                status_icon = "🟢" if info["running"] else "🔴"
                print(f"  {status_icon} {service}: {info}")
        else:
            print("❌ Production deployment failed")
            sys.exit(1)

    elif args.stop:
        await deployment.stop()
        print("✅ Production system stopped")

    elif args.status:
        status = deployment.get_status()
        print(json.dumps(status, indent=2, default=str))

    elif args.backup:
        await deployment.backup(args.backup)
        print("✅ Backup completed")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
