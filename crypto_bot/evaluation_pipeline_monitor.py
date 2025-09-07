"""
Evaluation Pipeline Monitor

Comprehensive monitoring and health check system for the evaluation pipeline.
Provides real-time monitoring, alerting, and performance tracking.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading

from .evaluation_pipeline_integration import get_evaluation_pipeline_integration
from .utils.logger import setup_logger, LOG_DIR

logger = setup_logger(__name__, LOG_DIR / "evaluation_pipeline_monitor.log")


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics being monitored."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    level: AlertLevel
    message: str
    cooldown_seconds: float = 300.0  # Don't spam alerts
    last_triggered: float = 0.0


@dataclass
class Metric:
    """Metric data structure."""
    name: str
    value: Any
    type: MetricType
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: bool
    message: str
    timestamp: float
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)


class EvaluationPipelineMonitor:
    """
    Comprehensive monitoring system for the evaluation pipeline.

    Features:
    - Real-time metrics collection
    - Health checks with configurable thresholds
    - Alert system with cooldowns
    - Performance monitoring
    - Dashboard data export
    - Historical data retention
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline_integration = None

        # Monitoring configuration
        self.monitor_config = self.config.get("pipeline_monitoring", {})
        self.enabled = self.monitor_config.get("enabled", True)
        self.collection_interval = self.monitor_config.get("collection_interval", 30.0)
        self.health_check_interval = self.monitor_config.get("health_check_interval", 60.0)
        self.metrics_retention_hours = self.monitor_config.get("metrics_retention_hours", 24)
        self.alerts_enabled = self.monitor_config.get("alerts_enabled", True)

        # Storage
        self.metrics: List[Metric] = []
        self.health_checks: List[HealthCheck] = []
        self.alerts: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False

        # Alert rules
        self.alert_rules = self._setup_alert_rules()

        # Callbacks
        self.alert_callbacks: List[Callable] = []
        self.health_callbacks: List[Callable] = []

        logger.info("Evaluation Pipeline Monitor initialized")

    def _setup_alert_rules(self) -> List[AlertRule]:
        """Setup default alert rules."""
        return [
            AlertRule(
                name="pipeline_offline",
                condition=lambda status: status.get("status") == "offline",
                level=AlertLevel.CRITICAL,
                message="Evaluation pipeline is offline",
                cooldown_seconds=300.0
            ),
            AlertRule(
                name="high_error_rate",
                condition=lambda status: status.get("metrics", {}).get("error_rate", 0) > 0.5,
                level=AlertLevel.ERROR,
                message="High error rate detected in evaluation pipeline",
                cooldown_seconds=600.0
            ),
            AlertRule(
                name="consecutive_failures",
                condition=lambda status: status.get("metrics", {}).get("consecutive_failures", 0) >= 5,
                level=AlertLevel.WARNING,
                message="Multiple consecutive failures in evaluation pipeline",
                cooldown_seconds=300.0
            ),
            AlertRule(
                name="scanner_unhealthy",
                condition=lambda status: not status.get("scanner", {}).get("healthy", True),
                level=AlertLevel.WARNING,
                message="Enhanced scanner is unhealthy",
                cooldown_seconds=600.0
            ),
            AlertRule(
                name="low_token_throughput",
                condition=lambda status: status.get("metrics", {}).get("tokens_processed", 0) < 10,
                level=AlertLevel.INFO,
                message="Low token processing throughput detected",
                cooldown_seconds=1800.0  # 30 minutes
            )
        ]

    async def start_monitoring(self):
        """Start the monitoring system."""
        if not self.enabled or self.is_running:
            return

        logger.info("Starting evaluation pipeline monitoring...")

        try:
            # Get pipeline integration instance
            self.pipeline_integration = get_evaluation_pipeline_integration(self.config)

            self.is_running = True

            # Start background tasks
            self.monitoring_task = asyncio.create_task(self._metrics_collection_loop())
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

            logger.info("✅ Evaluation pipeline monitoring started")

        except Exception as e:
            logger.error(f"❌ Failed to start monitoring: {e}")
            self.is_running = False

    async def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self.is_running:
            return

        logger.info("Stopping evaluation pipeline monitoring...")

        self.is_running = False

        # Cancel background tasks
        tasks = [self.monitoring_task, self.health_check_task, self.cleanup_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("✅ Evaluation pipeline monitoring stopped")

    async def _metrics_collection_loop(self):
        """Main metrics collection loop."""
        while self.is_running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(30)  # Retry sooner on error

    async def _health_check_loop(self):
        """Health check loop."""
        while self.is_running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)

    async def _cleanup_loop(self):
        """Cleanup old data loop."""
        while self.is_running:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Clean up every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)

    async def _collect_metrics(self):
        """Collect current metrics from the pipeline."""
        try:
            if not self.pipeline_integration:
                return

            # Get pipeline status
            status = self.pipeline_integration.get_pipeline_status()

            # Extract metrics
            metrics_data = status.get("metrics", {})

            # Create metric objects
            timestamp = time.time()

            metrics = [
                Metric(
                    name="tokens_received",
                    value=metrics_data.get("tokens_received", 0),
                    type=MetricType.COUNTER,
                    timestamp=timestamp,
                    tags={"source": "pipeline"}
                ),
                Metric(
                    name="tokens_processed",
                    value=metrics_data.get("tokens_processed", 0),
                    type=MetricType.COUNTER,
                    timestamp=timestamp,
                    tags={"source": "pipeline"}
                ),
                Metric(
                    name="tokens_failed",
                    value=metrics_data.get("tokens_failed", 0),
                    type=MetricType.COUNTER,
                    timestamp=timestamp,
                    tags={"source": "pipeline"}
                ),
                Metric(
                    name="avg_processing_time",
                    value=metrics_data.get("avg_processing_time", 0.0),
                    type=MetricType.GAUGE,
                    timestamp=timestamp,
                    tags={"source": "pipeline", "unit": "seconds"}
                ),
                Metric(
                    name="error_rate",
                    value=metrics_data.get("error_rate", 0.0),
                    type=MetricType.GAUGE,
                    timestamp=timestamp,
                    tags={"source": "pipeline", "unit": "ratio"}
                ),
                Metric(
                    name="consecutive_failures",
                    value=metrics_data.get("consecutive_failures", 0),
                    type=MetricType.GAUGE,
                    timestamp=timestamp,
                    tags={"source": "pipeline"}
                ),
                Metric(
                    name="pipeline_status",
                    value=1 if status.get("status") == "healthy" else 0,
                    type=MetricType.GAUGE,
                    timestamp=timestamp,
                    tags={"source": "pipeline", "status": status.get("status", "unknown")}
                )
            ]

            # Store metrics
            with self._lock:
                self.metrics.extend(metrics)

            # Check for alerts
            await self._check_alerts(status)

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

    async def _perform_health_checks(self):
        """Perform comprehensive health checks."""
        try:
            health_checks = []

            # Pipeline connectivity check
            pipeline_check = await self._check_pipeline_connectivity()
            health_checks.append(pipeline_check)

            # Scanner health check
            scanner_check = await self._check_scanner_health()
            health_checks.append(scanner_check)

            # Token flow check
            flow_check = await self._check_token_flow()
            health_checks.append(flow_check)

            # Performance check
            perf_check = await self._check_performance()
            health_checks.append(perf_check)

            # Store health checks
            with self._lock:
                self.health_checks.extend(health_checks)

            # Trigger health callbacks
            for callback in self.health_callbacks:
                try:
                    await callback(health_checks)
                except Exception as e:
                    logger.error(f"Error in health callback: {e}")

        except Exception as e:
            logger.error(f"Error performing health checks: {e}")

    async def _check_pipeline_connectivity(self) -> HealthCheck:
        """Check pipeline connectivity and basic functionality."""
        start_time = time.time()

        try:
            if not self.pipeline_integration:
                return HealthCheck(
                    name="pipeline_connectivity",
                    status=False,
                    message="Pipeline integration not available",
                    timestamp=start_time,
                    duration=0.0
                )

            # Try to get status
            status = self.pipeline_integration.get_pipeline_status()

            duration = time.time() - start_time

            return HealthCheck(
                name="pipeline_connectivity",
                status=True,
                message="Pipeline connectivity OK",
                timestamp=start_time,
                duration=duration,
                details={"status": status.get("status")}
            )

        except Exception as e:
            duration = time.time() - start_time
            return HealthCheck(
                name="pipeline_connectivity",
                status=False,
                message=f"Pipeline connectivity failed: {e}",
                timestamp=start_time,
                duration=duration
            )

    async def _check_scanner_health(self) -> HealthCheck:
        """Check enhanced scanner health."""
        start_time = time.time()

        try:
            if not self.pipeline_integration:
                return HealthCheck(
                    name="scanner_health",
                    status=False,
                    message="Pipeline integration not available",
                    timestamp=start_time,
                    duration=0.0
                )

            status = self.pipeline_integration.get_pipeline_status()
            scanner_info = status.get("scanner", {})

            duration = time.time() - start_time

            return HealthCheck(
                name="scanner_health",
                status=scanner_info.get("healthy", False),
                message="Scanner health OK" if scanner_info.get("healthy") else "Scanner unhealthy",
                timestamp=start_time,
                duration=duration,
                details=scanner_info
            )

        except Exception as e:
            duration = time.time() - start_time
            return HealthCheck(
                name="scanner_health",
                status=False,
                message=f"Scanner health check failed: {e}",
                timestamp=start_time,
                duration=duration
            )

    async def _check_token_flow(self) -> HealthCheck:
        """Check token flow through the pipeline."""
        start_time = time.time()

        try:
            if not self.pipeline_integration:
                return HealthCheck(
                    name="token_flow",
                    status=False,
                    message="Pipeline integration not available",
                    timestamp=start_time,
                    duration=0.0
                )

            # Try to get a small batch of tokens
            tokens = await self.pipeline_integration.get_tokens_for_evaluation(3)

            duration = time.time() - start_time

            success = len(tokens) > 0

            return HealthCheck(
                name="token_flow",
                status=success,
                message="Token flow OK" if success else "Token flow failed",
                timestamp=start_time,
                duration=duration,
                details={"tokens_returned": len(tokens), "sample_tokens": tokens[:3]}
            )

        except Exception as e:
            duration = time.time() - start_time
            return HealthCheck(
                name="token_flow",
                status=False,
                message=f"Token flow check failed: {e}",
                timestamp=start_time,
                duration=duration
            )

    async def _check_performance(self) -> HealthCheck:
        """Check pipeline performance metrics."""
        start_time = time.time()

        try:
            if not self.pipeline_integration:
                return HealthCheck(
                    name="performance",
                    status=False,
                    message="Pipeline integration not available",
                    timestamp=start_time,
                    duration=0.0
                )

            status = self.pipeline_integration.get_pipeline_status()
            metrics = status.get("metrics", {})

            duration = time.time() - start_time

            # Check performance thresholds
            avg_time = metrics.get("avg_processing_time", 0)
            error_rate = metrics.get("error_rate", 0)
            tokens_processed = metrics.get("tokens_processed", 0)

            # Performance criteria
            time_ok = avg_time < 5.0  # Less than 5 seconds average
            error_ok = error_rate < 0.3  # Less than 30% error rate
            throughput_ok = tokens_processed > 0  # Some tokens processed

            overall_status = time_ok and error_ok and throughput_ok

            return HealthCheck(
                name="performance",
                status=overall_status,
                message="Performance OK" if overall_status else "Performance issues detected",
                timestamp=start_time,
                duration=duration,
                details={
                    "avg_processing_time": avg_time,
                    "error_rate": error_rate,
                    "tokens_processed": tokens_processed,
                    "time_ok": time_ok,
                    "error_ok": error_ok,
                    "throughput_ok": throughput_ok
                }
            )

        except Exception as e:
            duration = time.time() - start_time
            return HealthCheck(
                name="performance",
                status=False,
                message=f"Performance check failed: {e}",
                timestamp=start_time,
                duration=duration
            )

    async def _check_alerts(self, status: Dict[str, Any]):
        """Check alert rules against current status."""
        if not self.alerts_enabled:
            return

        current_time = time.time()

        for rule in self.alert_rules:
            try:
                # Check cooldown
                if current_time - rule.last_triggered < rule.cooldown_seconds:
                    continue

                # Check condition
                if rule.condition(status):
                    # Trigger alert
                    alert = {
                        "rule_name": rule.name,
                        "level": rule.level.value,
                        "message": rule.message,
                        "timestamp": current_time,
                        "status": status
                    }

                    with self._lock:
                        self.alerts.append(alert)

                    # Update last triggered
                    rule.last_triggered = current_time

                    # Log alert
                    logger.warning(f"🚨 ALERT [{rule.level.value.upper()}]: {rule.message}")

                    # Trigger alert callbacks
                    for callback in self.alert_callbacks:
                        try:
                            await callback(alert)
                        except Exception as e:
                            logger.error(f"Error in alert callback: {e}")

            except Exception as e:
                logger.error(f"Error checking alert rule {rule.name}: {e}")

    async def _cleanup_old_data(self):
        """Clean up old metrics and health checks."""
        try:
            cutoff_time = time.time() - (self.metrics_retention_hours * 3600)

            with self._lock:
                # Clean old metrics
                self.metrics = [
                    m for m in self.metrics
                    if m.timestamp > cutoff_time
                ]

                # Clean old health checks
                self.health_checks = [
                    h for h in self.health_checks
                    if h.timestamp > cutoff_time
                ]

                # Clean old alerts (keep more alerts for longer)
                alert_cutoff = time.time() - (self.metrics_retention_hours * 2 * 3600)
                self.alerts = [
                    a for a in self.alerts
                    if a["timestamp"] > alert_cutoff
                ]

            logger.debug(f"Cleaned up old monitoring data (cutoff: {cutoff_time})")

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

    def add_alert_callback(self, callback: Callable):
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)

    def add_health_callback(self, callback: Callable):
        """Add a health check callback function."""
        self.health_callbacks.append(callback)

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "running": self.is_running,
                "config": {
                    "collection_interval": self.collection_interval,
                    "health_check_interval": self.health_check_interval,
                    "metrics_retention_hours": self.metrics_retention_hours,
                    "alerts_enabled": self.alerts_enabled
                },
                "stats": {
                    "total_metrics": len(self.metrics),
                    "total_health_checks": len(self.health_checks),
                    "total_alerts": len(self.alerts),
                    "active_alerts": len([a for a in self.alerts if time.time() - a["timestamp"] < 3600])
                },
                "latest_metrics": self._get_latest_metrics(),
                "latest_health_checks": self._get_latest_health_checks(),
                "active_alerts": self._get_active_alerts()
            }

    def _get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest values for each metric type."""
        latest = {}

        with self._lock:
            for metric in reversed(self.metrics):
                if metric.name not in latest:
                    latest[metric.name] = {
                        "value": metric.value,
                        "timestamp": metric.timestamp,
                        "tags": metric.tags
                    }

        return latest

    def _get_latest_health_checks(self) -> List[Dict[str, Any]]:
        """Get latest health check results."""
        latest_checks = {}

        with self._lock:
            for check in reversed(self.health_checks):
                if check.name not in latest_checks:
                    latest_checks[check.name] = {
                        "status": check.status,
                        "message": check.message,
                        "timestamp": check.timestamp,
                        "duration": check.duration,
                        "details": check.details
                    }

        return list(latest_checks.values())

    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts (last hour)."""
        cutoff_time = time.time() - 3600

        with self._lock:
            return [
                alert for alert in self.alerts
                if alert["timestamp"] > cutoff_time
            ]

    def export_monitoring_data(self, filepath: Optional[str] = None) -> str:
        """Export monitoring data to JSON file."""
        try:
            if not filepath:
                timestamp = int(time.time())
                filepath = f"evaluation_pipeline_monitoring_{timestamp}.json"

            data = self.get_monitoring_status()

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Monitoring data exported to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to export monitoring data: {e}")
            return ""


# Global monitor instance
_monitor: Optional[EvaluationPipelineMonitor] = None


def get_evaluation_pipeline_monitor(config: Dict[str, Any]) -> EvaluationPipelineMonitor:
    """Get or create the global monitor instance."""
    global _monitor

    if _monitor is None:
        _monitor = EvaluationPipelineMonitor(config)

    return _monitor


async def start_pipeline_monitoring(config: Dict[str, Any]):
    """Start the pipeline monitoring system."""
    monitor = get_evaluation_pipeline_monitor(config)
    await monitor.start_monitoring()


async def stop_pipeline_monitoring():
    """Stop the pipeline monitoring system."""
    global _monitor
    if _monitor:
        await _monitor.stop_monitoring()


def get_monitoring_status(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get monitoring status."""
    monitor = get_evaluation_pipeline_monitor(config)
    return monitor.get_monitoring_status()
