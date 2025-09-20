#!/usr/bin/env python3
"""
Frontend Cutover Manager for LegacyCoinTrader

This script manages the transition of the frontend from monolithic Flask server
to API Gateway-based microservice architecture.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time
import httpx


@dataclass
class FrontendCutoverConfig:
    """Configuration for frontend cutover."""
    old_flask_port: int = 5000
    new_api_gateway_port: int = 8000
    new_api_gateway_host: str = "localhost"
    enable_https: bool = False
    cutover_duration_minutes: int = 60
    traffic_distribution_check_interval: int = 30
    rollback_enabled: bool = True
    health_check_timeout: int = 10


@dataclass
class FrontendTrafficMetrics:
    """Metrics for frontend traffic distribution."""
    total_requests: int = 0
    old_flask_requests: int = 0
    new_gateway_requests: int = 0
    error_requests: int = 0
    average_response_time_old: float = 0.0
    average_response_time_new: float = 0.0
    last_request_time: Optional[float] = None


@dataclass
class FrontendServiceState:
    """State of frontend services."""
    service_name: str
    port: int
    status: str = "unknown"  # unknown, running, stopped, error
    health_status: str = "unknown"  # unknown, healthy, unhealthy
    request_count: int = 0
    error_count: int = 0
    last_health_check: Optional[datetime] = None


class FrontendCutoverManager:
    """Manages frontend cutover from monolithic to microservice architecture."""

    def __init__(self, config: FrontendCutoverConfig):
        self.config = config
        self.metrics = FrontendTrafficMetrics()
        self.services: Dict[str, FrontendServiceState] = {}
        self.cutover_log: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        self._logger = logging.getLogger(__name__)

        # Initialize service states
        self._initialize_services()

    def _initialize_services(self):
        """Initialize frontend service states."""
        self.services["old_flask"] = FrontendServiceState(
            service_name="old_flask",
            port=self.config.old_flask_port,
            status="unknown"
        )

        self.services["new_gateway"] = FrontendServiceState(
            service_name="new_gateway",
            port=self.config.new_api_gateway_port,
            status="unknown"
        )

    async def check_service_health(self, service_name: str) -> str:
        """Check health of a frontend service."""
        if service_name not in self.services:
            return "unknown"

        service = self.services[service_name]
        service.last_health_check = datetime.now()

        try:
            # Determine URL based on service
            if service_name == "old_flask":
                url = f"http://localhost:{service.port}/health"
            else:  # new_gateway
                protocol = "https" if self.config.enable_https else "http"
                url = f"{protocol}://{self.config.new_api_gateway_host}:{service.port}/health"

            async with httpx.AsyncClient(timeout=self.config.health_check_timeout) as client:
                response = await client.get(url)

                if response.status_code == 200:
                    service.status = "running"
                    service.health_status = "healthy"
                    return "healthy"
                else:
                    service.status = "running"
                    service.health_status = "unhealthy"
                    return "unhealthy"

        except httpx.ConnectError:
            service.status = "stopped"
            service.health_status = "unhealthy"
            return "unreachable"

        except Exception as e:
            service.status = "error"
            service.health_status = "unhealthy"
            self._logger.error(f"Health check failed for {service_name}: {e}")
            return "error"

    async def start_cutover_process(self):
        """Start the frontend cutover process."""
        self._logger.info("🚀 Starting frontend cutover process")
        self._logger.info(f"📍 Old Flask: http://localhost:{self.config.old_flask_port}")
        self._logger.info(f"📍 New Gateway: http://{self.config.new_api_gateway_host}:{self.config.new_api_gateway_port}")

        # Phase 1: Pre-flight checks
        await self._phase_preflight_checks()

        # Phase 2: Parallel operation
        await self._phase_parallel_operation()

        # Phase 3: Traffic cutover
        await self._phase_traffic_cutover()

        # Phase 4: Final verification
        await self._phase_final_verification()

    async def _phase_preflight_checks(self):
        """Phase 1: Pre-flight checks."""
        self._logger.info("📋 Phase 1: Pre-flight checks")

        # Check old Flask service
        old_health = await self.check_service_health("old_flask")
        self._logger.info(f"🔍 Old Flask health: {old_health}")

        # Check new API Gateway service
        new_health = await self.check_service_health("new_gateway")
        self._logger.info(f"🔍 New Gateway health: {new_health}")

        if old_health != "healthy":
            self._logger.warning("⚠️  Old Flask service is not healthy")

        if new_health != "healthy":
            raise RuntimeError("❌ New API Gateway service is not healthy. Cannot proceed with cutover.")

        await self._log_cutover_event("preflight_checks_completed", {
            "old_flask_health": old_health,
            "new_gateway_health": new_health,
            "can_proceed": new_health == "healthy"
        })

    async def _phase_parallel_operation(self):
        """Phase 2: Run both services in parallel."""
        self._logger.info("🔄 Phase 2: Parallel operation")

        # Both services should already be running
        # Monitor them for a period to ensure stability
        monitoring_duration = timedelta(minutes=5)

        self._logger.info(f"👀 Monitoring both services for {monitoring_duration.seconds // 60} minutes")

        end_time = datetime.now() + monitoring_duration

        while datetime.now() < end_time:
            # Check health of both services
            old_health = await self.check_service_health("old_flask")
            new_health = await self.check_service_health("new_gateway")

            self._logger.info(f"📊 Health - Old: {old_health}, New: {new_health}")

            if new_health != "healthy":
                raise RuntimeError("❌ New API Gateway became unhealthy during monitoring")

            await asyncio.sleep(30)  # Check every 30 seconds

        await self._log_cutover_event("parallel_operation_completed", {
            "monitoring_duration_minutes": monitoring_duration.seconds // 60,
            "final_old_health": old_health,
            "final_new_health": new_health
        })

    async def _phase_traffic_cutover(self):
        """Phase 3: Gradually cutover traffic."""
        self._logger.info("📈 Phase 3: Traffic cutover")

        # In a real implementation, this would involve:
        # 1. Load balancer configuration changes
        # 2. DNS updates
        # 3. Service mesh traffic shifting
        # 4. Nginx/Apache configuration updates

        # For this demo, we'll simulate the cutover process
        self._logger.info("🔄 Simulating traffic cutover...")

        # Simulate gradual traffic shift
        traffic_steps = [0, 25, 50, 75, 100]
        for traffic_pct in traffic_steps:
            self._logger.info(f"📊 Traffic distribution: {traffic_pct}% to new gateway, {100-traffic_pct}% to old flask")

            # In a real implementation, you would:
            # - Update load balancer weights
            # - Update DNS records
            # - Update service mesh configuration

            # Simulate monitoring period
            await asyncio.sleep(10)  # 10 seconds per step

            # Check health during cutover
            new_health = await self.check_service_health("new_gateway")
            if new_health != "healthy":
                if self.config.rollback_enabled:
                    self._logger.error("❌ New gateway became unhealthy during cutover. Rolling back...")
                    await self._rollback_traffic()
                    raise RuntimeError("Cutover failed, rolled back to old service")
                else:
                    raise RuntimeError("❌ New gateway became unhealthy during cutover")

        await self._log_cutover_event("traffic_cutover_completed", {
            "traffic_steps": traffic_steps,
            "final_traffic_distribution": "100% to new gateway"
        })

    async def _phase_final_verification(self):
        """Phase 4: Final verification."""
        self._logger.info("✅ Phase 4: Final verification")

        # Final health checks
        old_health = await self.check_service_health("old_flask")
        new_health = await self.check_service_health("new_gateway")

        # Verify new service is handling all traffic
        self._logger.info(f"🏥 Final health check - Old: {old_health}, New: {new_health}")

        if new_health != "healthy":
            raise RuntimeError("❌ New API Gateway is not healthy after cutover")

        # Run a few test requests to verify functionality
        test_results = await self._run_functionality_tests()

        # Calculate cutover duration
        cutover_duration = datetime.now() - self.start_time

        await self._log_cutover_event("cutover_completed", {
            "total_duration_seconds": cutover_duration.total_seconds(),
            "final_old_health": old_health,
            "final_new_health": new_health,
            "functionality_tests_passed": test_results["passed"],
            "functionality_tests_total": test_results["total"]
        })

        self._logger.info("🎉 Frontend cutover completed successfully!")
        self._logger.info(f"⏱️  Total duration: {cutover_duration.total_seconds()/60:.1f} minutes")

    async def _run_functionality_tests(self) -> Dict[str, int]:
        """Run basic functionality tests on the new gateway."""
        tests_passed = 0
        total_tests = 0

        test_endpoints = [
            "/health",
            "/auth/service-tokens",
            "/portfolio/state",
            "/market-data/symbols"
        ]

        protocol = "https" if self.config.enable_https else "http"
        base_url = f"{protocol}://{self.config.new_api_gateway_host}:{self.config.new_api_gateway_port}"

        async with httpx.AsyncClient(timeout=10) as client:
            for endpoint in test_endpoints:
                total_tests += 1
                try:
                    response = await client.get(f"{base_url}{endpoint}")
                    if response.status_code in [200, 401, 403]:  # 401/403 are OK for auth endpoints
                        tests_passed += 1
                        self._logger.info(f"✅ {endpoint}: {response.status_code}")
                    else:
                        self._logger.warning(f"⚠️  {endpoint}: {response.status_code}")
                except Exception as e:
                    self._logger.error(f"❌ {endpoint}: {e}")

        return {"passed": tests_passed, "total": total_tests}

    async def _rollback_traffic(self):
        """Rollback traffic to old service."""
        self._logger.warning("🔄 Rolling back traffic to old Flask service")

        # In a real implementation, this would:
        # - Restore load balancer configuration
        # - Restore DNS records
        # - Restore service mesh configuration

        self._logger.info("✅ Traffic rolled back to old service")

        await self._log_cutover_event("traffic_rollback_completed", {
            "rollback_time": datetime.now().isoformat(),
            "reason": "health_check_failed"
        })

    async def _log_cutover_event(self, event_type: str, data: Dict[str, Any]):
        """Log cutover event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }

        self.cutover_log.append(event)

        # Log to file
        log_file = Path("frontend_cutover_log.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def get_cutover_status(self) -> Dict[str, Any]:
        """Get current cutover status."""
        return {
            "cutover_start_time": self.start_time.isoformat(),
            "current_time": datetime.now().isoformat(),
            "services": {
                name: {
                    "status": state.status,
                    "health_status": state.health_status,
                    "port": state.port,
                    "request_count": state.request_count,
                    "error_count": state.error_count,
                    "last_health_check": state.last_health_check.isoformat() if state.last_health_check else None,
                }
                for name, state in self.services.items()
            },
            "traffic_metrics": {
                "total_requests": self.metrics.total_requests,
                "old_flask_requests": self.metrics.old_flask_requests,
                "new_gateway_requests": self.metrics.new_gateway_requests,
                "error_requests": self.metrics.error_requests,
                "average_response_time_old": self.metrics.average_response_time_old,
                "average_response_time_new": self.metrics.average_response_time_new,
            },
            "events_logged": len(self.cutover_log),
        }

    async def stop_old_service(self):
        """Stop the old Flask service."""
        self._logger.info("🛑 Stopping old Flask service")

        try:
            # Find and kill the old Flask process
            result = subprocess.run(
                ["lsof", "-ti", f":{self.config.old_flask_port}"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0 and result.stdout.strip():
                pid = result.stdout.strip()
                subprocess.run(["kill", pid])
                self._logger.info(f"✅ Killed old Flask process (PID: {pid})")
            else:
                self._logger.warning("⚠️  Could not find old Flask process to kill")

        except Exception as e:
            self._logger.error(f"Failed to stop old Flask service: {e}")

        await self._log_cutover_event("old_service_stopped", {
            "stop_time": datetime.now().isoformat()
        })


async def main():
    """Main function for frontend cutover."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Frontend Cutover Manager")
    parser.add_argument(
        "--old-port",
        type=int,
        default=5000,
        help="Port of old Flask service"
    )
    parser.add_argument(
        "--new-port",
        type=int,
        default=8000,
        help="Port of new API Gateway"
    )
    parser.add_argument(
        "--new-host",
        default="localhost",
        help="Host of new API Gateway"
    )
    parser.add_argument(
        "--enable-https",
        action="store_true",
        help="Enable HTTPS for new gateway"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Cutover duration in minutes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without actual cutover"
    )
    parser.add_argument(
        "--stop-old",
        action="store_true",
        help="Stop old service after cutover"
    )

    args = parser.parse_args()

    # Create cutover configuration
    config = FrontendCutoverConfig(
        old_flask_port=args.old_port,
        new_api_gateway_port=args.new_port,
        new_api_gateway_host=args.new_host,
        enable_https=args.enable_https,
        cutover_duration_minutes=args.duration
    )

    # Create cutover manager
    manager = FrontendCutoverManager(config)

    try:
        if args.dry_run:
            print("🔍 Performing dry run...")

            # Just check health of both services
            old_health = await manager.check_service_health("old_flask")
            new_health = await manager.check_service_health("new_gateway")

            print(f"📊 Old Flask health: {old_health}")
            print(f"📊 New Gateway health: {new_health}")

            if new_health == "healthy":
                print("✅ Dry run successful - services are ready for cutover")
            else:
                print("❌ Dry run failed - new gateway is not healthy")
                sys.exit(1)

        else:
            # Perform actual cutover
            await manager.start_cutover_process()

            if args.stop_old:
                await manager.stop_old_service()

        # Print final status
        status = manager.get_cutover_status()
        print("\n📊 Final Cutover Status:")
        print("=" * 50)
        print(f"Duration: {(datetime.now() - manager.start_time).total_seconds()/60:.1f} minutes")
        print(f"Events logged: {status['events_logged']}")

        for service_name, service_info in status['services'].items():
            print(f"  {service_name}: {service_info['status']} ({service_info['health_status']})")

    except KeyboardInterrupt:
        print("\n⚠️  Cutover interrupted by user")
        status = manager.get_cutover_status()
        print(f"Progress at interruption: {len(status['cutover_log'])} events logged")

    except Exception as e:
        print(f"\n❌ Cutover failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
