#!/usr/bin/env python3
"""
Migration Cutover Manager for LegacyCoinTrader Microservice Migration

This script manages the gradual cutover from monolithic to microservice architecture
using dual-write patterns, feature flags, and progressive migration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time

from libs.dual_write_manager import (
    DualWriteConfig,
    DualWriteManager,
    execute_with_dual_write,
    dual_write_registry,
)


@dataclass
class ServiceMigrationState:
    """Migration state for a service."""
    service_name: str
    migration_phase: str = "preparation"  # preparation, dual_write, cutover, completed
    dual_write_enabled: bool = False
    primary_system: str = "monolith"  # monolith or microservice
    traffic_percentage: float = 0.0  # 0.0 to 1.0
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"  # unknown, healthy, degraded, unhealthy
    error_count: int = 0
    success_count: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationConfig:
    """Configuration for the overall migration."""
    project_name: str = "legacy-coin-trader"
    migration_start_date: Optional[datetime] = None
    target_completion_date: Optional[datetime] = None
    enable_dual_write: bool = True
    enable_feature_flags: bool = True
    enable_monitoring: bool = True
    rollback_enabled: bool = True
    auto_rollback_threshold: float = 0.1  # 10% error rate triggers rollback
    health_check_interval: int = 60  # seconds
    gradual_traffic_increase: float = 0.1  # 10% traffic increase per step


class MigrationCutoverManager:
    """Manages the cutover from monolithic to microservice architecture."""

    def __init__(self, config: MigrationConfig):
        self.config = config
        self.services: Dict[str, ServiceMigrationState] = {}
        self.migration_log: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
        self._logger = logging.getLogger(__name__)

        # Initialize service states
        self._initialize_services()

    def _initialize_services(self):
        """Initialize migration state for all services."""
        services = [
            "api-gateway",
            "trading-engine",
            "market-data",
            "strategy-engine",
            "portfolio",
            "execution",
            "token-discovery",
            "monitoring"
        ]

        for service_name in services:
            self.services[service_name] = ServiceMigrationState(
                service_name=service_name,
                migration_phase="preparation",
                dual_write_enabled=False,
                primary_system="monolith",
                traffic_percentage=0.0
            )

    async def start_migration(self):
        """Start the migration process."""
        self._logger.info("🚀 Starting LegacyCoinTrader microservice migration")
        self.config.migration_start_date = datetime.now()

        # Log migration start
        await self._log_migration_event("migration_started", {
            "timestamp": datetime.now().isoformat(),
            "services": list(self.services.keys()),
            "config": {
                "dual_write_enabled": self.config.enable_dual_write,
                "feature_flags_enabled": self.config.enable_feature_flags,
                "monitoring_enabled": self.config.enable_monitoring,
            }
        })

        # Phase 1: Preparation
        await self._phase_preparation()

        # Phase 2: Dual Write
        if self.config.enable_dual_write:
            await self._phase_dual_write()

        # Phase 3: Gradual Cutover
        await self._phase_gradual_cutover()

        # Phase 4: Completion
        await self._phase_completion()

    async def _phase_preparation(self):
        """Phase 1: Preparation - Ensure all services are ready."""
        self._logger.info("📋 Phase 1: Preparation")

        for service_name, state in self.services.items():
            # Check service health
            health_status = await self._check_service_health(service_name)

            if health_status == "healthy":
                state.migration_phase = "ready_for_dual_write"
                state.health_status = "healthy"
                self._logger.info(f"✅ {service_name}: Ready for dual write")
            else:
                state.migration_phase = "preparation_failed"
                state.health_status = health_status
                self._logger.error(f"❌ {service_name}: Health check failed - {health_status}")

        await self._log_migration_event("phase_preparation_completed", {
            "services_ready": len([s for s in self.services.values() if s.migration_phase == "ready_for_dual_write"]),
            "services_failed": len([s for s in self.services.values() if s.migration_phase == "preparation_failed"]),
        })

    async def _phase_dual_write(self):
        """Phase 2: Dual Write - Enable dual write for all services."""
        self._logger.info("🔄 Phase 2: Dual Write")

        for service_name, state in self.services.items():
            if state.migration_phase == "ready_for_dual_write":
                # Enable dual write
                success = await self._enable_dual_write(service_name)
                if success:
                    state.migration_phase = "dual_write_active"
                    state.dual_write_enabled = True
                    self._logger.info(f"✅ {service_name}: Dual write enabled")
                else:
                    state.migration_phase = "dual_write_failed"
                    self._logger.error(f"❌ {service_name}: Failed to enable dual write")

        await self._log_migration_event("phase_dual_write_completed", {
            "services_dual_write": len([s for s in self.services.values() if s.migration_phase == "dual_write_active"]),
        })

        # Monitor dual write for a period
        await self._monitor_dual_write_phase(duration_minutes=30)

    async def _phase_gradual_cutover(self):
        """Phase 3: Gradual Cutover - Gradually shift traffic to microservices."""
        self._logger.info("📈 Phase 3: Gradual Cutover")

        # Cutover services in order of dependency
        cutover_order = [
            "monitoring",      # Start with monitoring
            "market-data",     # Then market data (independent)
            "token-discovery", # Token discovery
            "strategy-engine", # Strategy engine (depends on market data)
            "portfolio",       # Portfolio (depends on market data)
            "execution",       # Execution (depends on portfolio)
            "trading-engine",  # Trading engine (depends on all above)
            "api-gateway"      # Finally API gateway
        ]

        for service_name in cutover_order:
            if service_name in self.services:
                await self._cutover_service(service_name)

        await self._log_migration_event("phase_gradual_cutover_completed", {
            "services_cutover": len([s for s in self.services.values() if s.migration_phase == "cutover_completed"]),
        })

    async def _phase_completion(self):
        """Phase 4: Completion - Finalize migration."""
        self._logger.info("🎉 Phase 4: Completion")

        # Disable dual write for all services
        for service_name, state in self.services.items():
            if state.dual_write_enabled:
                await self._disable_dual_write(service_name)
                state.dual_write_enabled = False

        # Final health check
        all_healthy = True
        for service_name, state in self.services.items():
            health_status = await self._check_service_health(service_name)
            if health_status != "healthy":
                all_healthy = False
                self._logger.warning(f"⚠️  {service_name}: {health_status}")

        migration_duration = datetime.now() - self.start_time

        await self._log_migration_event("migration_completed", {
            "total_duration_seconds": migration_duration.total_seconds(),
            "all_services_healthy": all_healthy,
            "final_service_states": {
                name: {
                    "phase": state.migration_phase,
                    "health": state.health_status,
                    "traffic_percentage": state.traffic_percentage
                }
                for name, state in self.services.items()
            }
        })

        if all_healthy:
            self._logger.info("🎉 Migration completed successfully!")
        else:
            self._logger.warning("⚠️  Migration completed with some services unhealthy")

    async def _enable_dual_write(self, service_name: str) -> bool:
        """Enable dual write for a service."""
        try:
            # Create dual write configuration
            config = DualWriteConfig(
                service_name=service_name,
                enabled=True,
                primary_system="monolith",
                secondary_system="microservice",
                compare_results=True,
                fail_on_mismatch=False,  # Don't fail on mismatches during migration
                sampling_rate=0.1  # Sample 10% of requests initially
            )

            # Get or create dual write manager
            manager = await dual_write_registry.get_or_create(service_name, config)

            # Set up callables (these would be actual service callables in production)
            manager.set_primary_callable(lambda *args, **kwargs: self._mock_monolith_call(service_name, *args, **kwargs))
            manager.set_secondary_callable(lambda *args, **kwargs: self._mock_microservice_call(service_name, *args, **kwargs))

            self._logger.info(f"🔄 Dual write enabled for {service_name}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to enable dual write for {service_name}: {e}")
            return False

    async def _disable_dual_write(self, service_name: str):
        """Disable dual write for a service."""
        await dual_write_registry.disable_dual_write(service_name)
        self._logger.info(f"🔄 Dual write disabled for {service_name}")

    async def _cutover_service(self, service_name: str):
        """Cutover a single service."""
        state = self.services[service_name]

        self._logger.info(f"📈 Starting cutover for {service_name}")

        # Gradually increase traffic to microservice
        traffic_steps = [0.1, 0.25, 0.5, 0.75, 1.0]

        for traffic_pct in traffic_steps:
            # Update traffic percentage
            state.traffic_percentage = traffic_pct

            # Monitor for a short period
            await asyncio.sleep(10)  # 10 seconds per step

            # Check health
            health_status = await self._check_service_health(service_name)
            state.health_status = health_status

            if health_status != "healthy":
                self._logger.warning(f"⚠️  {service_name}: Health degraded at {traffic_pct*100}% traffic")
                # Could implement rollback logic here

        # Complete cutover
        state.migration_phase = "cutover_completed"
        state.primary_system = "microservice"
        self._logger.info(f"✅ {service_name}: Cutover completed")

    async def _check_service_health(self, service_name: str) -> str:
        """Check health of a service."""
        try:
            # In a real implementation, this would make actual health check calls
            # For now, simulate health checks
            await asyncio.sleep(0.1)  # Simulate network call

            # Simulate occasional health issues
            import random
            if random.random() < 0.05:  # 5% chance of being unhealthy
                return "unhealthy"
            elif random.random() < 0.1:  # 10% chance of being degraded
                return "degraded"
            else:
                return "healthy"

        except Exception as e:
            self._logger.error(f"Health check failed for {service_name}: {e}")
            return "unhealthy"

    async def _monitor_dual_write_phase(self, duration_minutes: int):
        """Monitor dual write phase for specified duration."""
        self._logger.info(f"👀 Monitoring dual write phase for {duration_minutes} minutes")

        end_time = datetime.now() + timedelta(minutes=duration_minutes)

        while datetime.now() < end_time:
            # Get metrics from all dual write managers
            all_metrics = dual_write_registry.get_all_metrics()

            # Log summary
            total_operations = sum(m.get("total_operations", 0) for m in all_metrics.values())
            total_mismatches = sum(m.get("mismatches", 0) for m in all_metrics.values())

            self._logger.info(f"📊 Dual write metrics: {total_operations} operations, {total_mismatches} mismatches")

            await asyncio.sleep(60)  # Check every minute

    async def _mock_monolith_call(self, service_name: str, *args, **kwargs):
        """Mock monolith service call."""
        await asyncio.sleep(0.01)  # Simulate processing time
        return f"monolith_response_{service_name}"

    async def _mock_microservice_call(self, service_name: str, *args, **kwargs):
        """Mock microservice call."""
        await asyncio.sleep(0.005)  # Simulate faster microservice response
        return f"microservice_response_{service_name}"

    async def _log_migration_event(self, event_type: str, data: Dict[str, Any]):
        """Log migration event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }

        self.migration_log.append(event)

        # Log to file
        log_file = Path("migration_log.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        return {
            "migration_start_time": self.start_time.isoformat(),
            "current_time": datetime.now().isoformat(),
            "services": {
                name: {
                    "phase": state.migration_phase,
                    "dual_write_enabled": state.dual_write_enabled,
                    "primary_system": state.primary_system,
                    "traffic_percentage": state.traffic_percentage,
                    "health_status": state.health_status,
                    "error_count": state.error_count,
                    "success_count": state.success_count,
                }
                for name, state in self.services.items()
            },
            "overall_progress": self._calculate_overall_progress(),
            "events_logged": len(self.migration_log),
        }

    def _calculate_overall_progress(self) -> float:
        """Calculate overall migration progress (0.0 to 1.0)."""
        total_services = len(self.services)
        completed_services = len([
            s for s in self.services.values()
            if s.migration_phase == "cutover_completed"
        ])

        return completed_services / total_services if total_services > 0 else 0.0

    async def rollback_service(self, service_name: str):
        """Rollback a service to monolith."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not found")

        state = self.services[service_name]

        # Disable dual write
        await self._disable_dual_write(service_name)
        state.dual_write_enabled = False

        # Reset traffic to 0%
        state.traffic_percentage = 0.0

        # Switch back to monolith
        state.primary_system = "monolith"
        state.migration_phase = "rolled_back"

        self._logger.warning(f"🔄 {service_name}: Rolled back to monolith")

        await self._log_migration_event("service_rolled_back", {
            "service_name": service_name,
            "reason": "manual_rollback"
        })


async def main():
    """Main function for migration cutover."""
    logging.basicConfig(level=logging.INFO)

    # Create migration configuration
    config = MigrationConfig(
        enable_dual_write=True,
        enable_feature_flags=True,
        enable_monitoring=True,
        rollback_enabled=True,
        auto_rollback_threshold=0.1,
        health_check_interval=60,
        gradual_traffic_increase=0.1
    )

    # Create migration manager
    manager = MigrationCutoverManager(config)

    try:
        # Start migration
        await manager.start_migration()

        # Print final status
        status = manager.get_migration_status()
        print("\n🎉 Migration completed!")
        print(f"📊 Overall Progress: {status['overall_progress']*100:.1f}%")
        print(f"⏱️  Duration: {(datetime.now() - manager.start_time).total_seconds()/60:.1f} minutes")

        for service_name, service_status in status['services'].items():
            phase = service_status['phase']
            health = service_status['health_status']
            traffic = service_status['traffic_percentage']

            if phase == "cutover_completed":
                print(f"✅ {service_name}: {phase} ({traffic*100:.0f}% traffic)")
            else:
                print(f"⚠️  {service_name}: {phase} - {health}")

    except KeyboardInterrupt:
        print("\n⚠️  Migration interrupted by user")
        status = manager.get_migration_status()
        print(f"📊 Progress at interruption: {status['overall_progress']*100:.1f}%")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
