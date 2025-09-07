"""Comprehensive monitoring system for evaluation and execution pipelines."""

import asyncio
import time
import threading
import psutil
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys

from .utils.logger import LOG_DIR, setup_logger
from .utils.metrics_logger import log_metrics_to_csv
from .utils.telegram import TelegramNotifier

logger = setup_logger(__name__, LOG_DIR / "pipeline_monitor.log")


@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance monitoring."""
    timestamp: datetime
    evaluation_latency: float = 0.0
    execution_latency: float = 0.0
    strategy_evaluation_count: int = 0
    order_execution_count: int = 0
    position_updates: int = 0
    websocket_connections: int = 0
    api_calls: int = 0
    errors: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    network_latency_ms: float = 0.0


@dataclass
class HealthStatus:
    """System health status."""
    component: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    last_check: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)


class PipelineMonitor:
    """Comprehensive monitoring system for trading pipelines."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history: List[PipelineMetrics] = []
        self.health_status: Dict[str, HealthStatus] = {}
        self.alerts_sent: set = set()
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Initialize components
        self.telegram = TelegramNotifier() if config.get('telegram_enabled') else None
        self.metrics_file = LOG_DIR / "pipeline_metrics.csv"
        self.health_file = LOG_DIR / "health_status.json"

        # Monitoring intervals
        self.check_interval = config.get('monitoring', {}).get('check_interval_seconds', 30)
        self.alert_interval = config.get('monitoring', {}).get('alert_interval_seconds', 300)

        # Thresholds
        self.thresholds = {
            'max_evaluation_latency': 5.0,
            'max_execution_latency': 2.0,
            'max_memory_usage_mb': 1000.0,
            'max_cpu_usage_percent': 80.0,
            'max_network_latency_ms': 1000.0,
            'min_websocket_connections': 1,
            'max_error_rate': 0.1
        }

        # Callbacks for component health checks
        self.health_checks: Dict[str, Callable] = {
            'evaluation_pipeline': self._check_evaluation_pipeline,
            'execution_pipeline': self._check_execution_pipeline,
            'websocket_connections': self._check_websocket_connections,
            'system_resources': self._check_system_resources,
            'position_monitoring': self._check_position_monitoring,
            'strategy_router': self._check_strategy_router
        }

    async def start_monitoring(self) -> None:
        """Start the monitoring system."""
        logger.info("Starting comprehensive pipeline monitoring")
        self.monitoring_active = True

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        # Initial health check
        await self.perform_health_check()

    async def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        logger.info("Stopping pipeline monitoring")
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        while self.monitoring_active:
            try:
                # Create new event loop for async operations
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Run health check
                loop.run_until_complete(self.perform_health_check())

                # Collect metrics
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)

                # Save metrics
                self._save_metrics(metrics)

                # Check for alerts
                loop.run_until_complete(self._check_alerts())

                # Cleanup old metrics (keep last 1000 entries)
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            time.sleep(self.check_interval)

    async def perform_health_check(self) -> Dict[str, HealthStatus]:
        """Perform comprehensive health check of all components."""
        logger.debug("Performing health check")

        for component_name, check_func in self.health_checks.items():
            try:
                status = await check_func()
                self.health_status[component_name] = status
            except Exception as e:
                logger.error(f"Health check failed for {component_name}: {e}")
                self.health_status[component_name] = HealthStatus(
                    component=component_name,
                    status='critical',
                    message=f"Health check failed: {e}",
                    last_check=datetime.now()
                )

        # Save health status
        self._save_health_status()

        # Also save a simplified status file for frontend API
        self._save_frontend_status()

        return self.health_status

    async def _check_evaluation_pipeline(self) -> HealthStatus:
        """Check evaluation pipeline health."""
        try:
            # Check if strategy evaluation is running
            process_running = (
                self._is_process_running("start_bot_noninteractive") or
                self._is_process_running("crypto_bot.main") or
                self._is_process_running("start_bot") or
                self._is_process_running("start_bot_final") or
                self._is_process_running("start_bot_auto") or
                self._is_process_running("start_bot_clean")
            )

            # Check recent strategy evaluations
            strategy_log = LOG_DIR / "bot.log"
            recent_evaluations = 0

            if strategy_log.exists():
                with open(strategy_log, 'r') as f:
                    lines = f.readlines()[-100:]  # Last 100 lines
                    for line in lines:
                        if any(keyword in line.lower() for keyword in [
                            'strategy evaluation', 'evaluating strategy', 'signal generated',
                            'analysis result', 'actionable signals', 'execute_signals',
                            'queueing cex trade', 'queueing solana trade'
                        ]):
                            recent_evaluations += 1

            # Determine status
            if not process_running:
                return HealthStatus(
                    component='evaluation_pipeline',
                    status='critical',
                    message='Trading bot process not running',
                    last_check=datetime.now(),
                    metrics={'process_running': False, 'recent_evaluations': recent_evaluations}
                )

            if recent_evaluations == 0:
                return HealthStatus(
                    component='evaluation_pipeline',
                    status='warning',
                    message='No recent strategy evaluations detected',
                    last_check=datetime.now(),
                    metrics={'process_running': True, 'recent_evaluations': recent_evaluations}
                )

            return HealthStatus(
                component='evaluation_pipeline',
                status='healthy',
                message=f'Evaluation pipeline healthy ({recent_evaluations} recent evaluations)',
                last_check=datetime.now(),
                metrics={'process_running': True, 'recent_evaluations': recent_evaluations}
            )

        except Exception as e:
            return HealthStatus(
                component='evaluation_pipeline',
                status='critical',
                message=f'Evaluation check failed: {e}',
                last_check=datetime.now()
            )

    async def _check_execution_pipeline(self) -> HealthStatus:
        """Check execution pipeline health."""
        try:
            # Check execution log for recent activity
            execution_log = LOG_DIR / "execution.log"
            recent_executions = 0
            recent_errors = 0

            if execution_log.exists():
                with open(execution_log, 'r') as f:
                    lines = f.readlines()[-100:]
                    for line in lines:
                        if any(keyword in line.lower() for keyword in [
                            'order placed', 'execution successful', 'trade executed'
                        ]):
                            recent_executions += 1
                        if 'error' in line.lower() or 'failed' in line.lower():
                            recent_errors += 1

            # Check for stuck orders or pending executions
            pending_orders = 0
            if (LOG_DIR / "sell_requests.json").exists():
                try:
                    with open(LOG_DIR / "sell_requests.json", 'r') as f:
                        sell_requests = json.load(f)
                        pending_orders = len(sell_requests) if isinstance(sell_requests, list) else 0
                except:
                    pass

            status = 'healthy'
            message = f'Execution pipeline healthy ({recent_executions} executions)'

            if pending_orders > 5:
                status = 'warning'
                message = f'High pending orders: {pending_orders}'
            elif recent_errors > recent_executions * 0.5:
                status = 'warning'
                message = f'High error rate: {recent_errors} errors vs {recent_executions} executions'
            elif recent_executions == 0 and recent_errors > 0:
                status = 'critical'
                message = f'Execution failures detected: {recent_errors} errors'

            return HealthStatus(
                component='execution_pipeline',
                status=status,
                message=message,
                last_check=datetime.now(),
                metrics={
                    'recent_executions': recent_executions,
                    'recent_errors': recent_errors,
                    'pending_orders': pending_orders
                }
            )

        except Exception as e:
            return HealthStatus(
                component='execution_pipeline',
                status='critical',
                message=f'Execution check failed: {e}',
                last_check=datetime.now()
            )

    async def _check_websocket_connections(self) -> HealthStatus:
        """Check WebSocket connection health."""
        try:
            # Check if Kraken WebSocket client is active
            ws_active = False
            ws_connections = 0

            # Look for WebSocket activity in logs
            bot_log = LOG_DIR / "bot.log"
            if bot_log.exists():
                with open(bot_log, 'r') as f:
                    lines = f.readlines()[-50:]
                    for line in lines:
                        if 'websocket' in line.lower():
                            ws_active = True
                            if 'connected' in line.lower() or 'subscribed' in line.lower():
                                ws_connections += 1

            # Test basic connectivity
            try:
                import websocket
                ws = websocket.create_connection(
                    "wss://ws.kraken.com/v2",
                    timeout=5
                )
                ws.close()
                connectivity_ok = True
            except:
                connectivity_ok = False

            if not connectivity_ok:
                return HealthStatus(
                    component='websocket_connections',
                    status='critical',
                    message='Cannot connect to Kraken WebSocket',
                    last_check=datetime.now(),
                    metrics={'connectivity_ok': False, 'ws_active': ws_active}
                )

            if not ws_active:
                return HealthStatus(
                    component='websocket_connections',
                    status='warning',
                    message='WebSocket client not active',
                    last_check=datetime.now(),
                    metrics={'connectivity_ok': True, 'ws_active': False, 'connections': ws_connections}
                )

            return HealthStatus(
                component='websocket_connections',
                status='healthy',
                message=f'WebSocket connections healthy ({ws_connections} active)',
                last_check=datetime.now(),
                metrics={'connectivity_ok': True, 'ws_active': True, 'connections': ws_connections}
            )

        except Exception as e:
            return HealthStatus(
                component='websocket_connections',
                status='critical',
                message=f'WebSocket check failed: {e}',
                last_check=datetime.now()
            )

    async def _check_system_resources(self) -> HealthStatus:
        """Check system resource usage."""
        try:
            # Get current process
            current_process = psutil.Process()
            memory_usage = current_process.memory_info().rss / 1024 / 1024  # MB
            cpu_usage = current_process.cpu_percent()

            # Get system memory
            system_memory = psutil.virtual_memory()
            system_memory_percent = system_memory.percent

            # Check thresholds
            status = 'healthy'
            message = '.1f'

            if memory_usage > self.thresholds['max_memory_usage_mb']:
                status = 'warning'
                message = '.1f'
            elif cpu_usage > self.thresholds['max_cpu_usage_percent']:
                status = 'warning'
                message = '.1f'
            elif system_memory_percent > 90:
                status = 'critical'
                message = '.1f'

            return HealthStatus(
                component='system_resources',
                status=status,
                message=message,
                last_check=datetime.now(),
                metrics={
                    'memory_usage_mb': memory_usage,
                    'cpu_usage_percent': cpu_usage,
                    'system_memory_percent': system_memory_percent
                }
            )

        except Exception as e:
            return HealthStatus(
                component='system_resources',
                status='critical',
                message=f'System resource check failed: {e}',
                last_check=datetime.now()
            )

    async def _check_position_monitoring(self) -> HealthStatus:
        """Check position monitoring health."""
        try:
            positions_log = LOG_DIR / "positions.log"
            recent_updates = 0

            if positions_log.exists():
                stat = positions_log.stat()
                age_seconds = time.time() - stat.st_mtime

                if age_seconds > 300:  # 5 minutes
                    return HealthStatus(
                        component='position_monitoring',
                        status='warning',
                        message='.1f',
                        last_check=datetime.now(),
                        metrics={'age_seconds': age_seconds, 'recent_updates': recent_updates}
                    )

                # Check for recent position updates
                with open(positions_log, 'r') as f:
                    lines = f.readlines()[-20:]
                    for line in lines:
                        if any(keyword in line.lower() for keyword in [
                            'position update', 'pnl', 'trailing stop'
                        ]):
                            recent_updates += 1

            return HealthStatus(
                component='position_monitoring',
                status='healthy',
                message=f'Position monitoring healthy ({recent_updates} recent updates)',
                last_check=datetime.now(),
                metrics={'age_seconds': age_seconds if 'age_seconds' in locals() else 0, 'recent_updates': recent_updates}
            )

        except Exception as e:
            return HealthStatus(
                component='position_monitoring',
                status='critical',
                message=f'Position monitoring check failed: {e}',
                last_check=datetime.now()
            )

    async def _check_strategy_router(self) -> HealthStatus:
        """Check strategy router health."""
        try:
            # Check if strategy router is processing symbols
            bot_log = LOG_DIR / "bot.log"
            recent_routing = 0

            if bot_log.exists():
                with open(bot_log, 'r') as f:
                    lines = f.readlines()[-50:]
                    for line in lines:
                        if any(keyword in line.lower() for keyword in [
                            'routing to strategy', 'strategy selected', 'evaluating',
                            'analysis result', 'actionable signals', 'execute_signals',
                            'queueing cex trade', 'queueing solana trade'
                        ]):
                            recent_routing += 1

            if recent_routing == 0:
                return HealthStatus(
                    component='strategy_router',
                    status='warning',
                    message='No recent strategy routing activity',
                    last_check=datetime.now(),
                    metrics={'recent_routing': recent_routing}
                )

            return HealthStatus(
                component='strategy_router',
                status='healthy',
                message=f'Strategy router healthy ({recent_routing} recent routings)',
                last_check=datetime.now(),
                metrics={'recent_routing': recent_routing}
            )

        except Exception as e:
            return HealthStatus(
                component='strategy_router',
                status='critical',
                message=f'Strategy router check failed: {e}',
                last_check=datetime.now()
            )

    def _collect_current_metrics(self) -> PipelineMetrics:
        """Collect current pipeline metrics."""
        try:
            # Get system metrics
            current_process = psutil.Process()
            memory_usage = current_process.memory_info().rss / 1024 / 1024
            cpu_usage = current_process.cpu_percent()

            # Count recent activities from logs
            evaluation_count = self._count_recent_log_entries([
                'strategy evaluation', 'evaluating strategy', 'signal generated'
            ])

            execution_count = self._count_recent_log_entries([
                'order placed', 'execution successful', 'trade executed'
            ])

            return PipelineMetrics(
                timestamp=datetime.now(),
                strategy_evaluation_count=evaluation_count,
                order_execution_count=execution_count,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage
            )

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return PipelineMetrics(timestamp=datetime.now())

    def _count_recent_log_entries(self, keywords: List[str]) -> int:
        """Count recent log entries containing specific keywords."""
        try:
            bot_log = LOG_DIR / "bot.log"
            if not bot_log.exists():
                return 0

            count = 0
            with open(bot_log, 'r') as f:
                lines = f.readlines()[-100:]  # Last 100 lines
                for line in lines:
                    if any(keyword in line.lower() for keyword in keywords):
                        count += 1
            return count

        except Exception:
            return 0

    async def _check_alerts(self) -> None:
        """Check for alert conditions and send notifications."""
        try:
            for component, status in self.health_status.items():
                alert_key = f"{component}_{status.status}"

                # Only send alert if we haven't sent one recently
                if status.status in ['warning', 'critical'] and alert_key not in self.alerts_sent:
                    await self._send_alert(status)
                    self.alerts_sent.add(alert_key)

                # Clear resolved alerts
                elif status.status == 'healthy' and alert_key in self.alerts_sent:
                    self.alerts_sent.discard(alert_key)

        except Exception as e:
            logger.error(f"Error checking alerts: {e}")

    async def _send_alert(self, status: HealthStatus) -> None:
        """Send alert notification."""
        try:
            emoji = "🟢" if status.status == "healthy" else "🟡" if status.status == "warning" else "🔴"
            message = f"{emoji} {status.component.upper()}\nStatus: {status.status.upper()}\n{status.message}"

            if self.telegram:
                await self.telegram.send_message(message)
            else:
                logger.warning(f"Alert (no Telegram): {message}")

        except Exception as e:
            logger.error(f"Error sending alert: {e}")

    def _save_metrics(self, metrics: PipelineMetrics) -> None:
        """Save metrics to CSV file."""
        try:
            if not self.config.get("metrics_enabled"):
                return

            metrics_dict = {
                "timestamp": metrics.timestamp.isoformat(),
                "evaluation_latency": metrics.evaluation_latency,
                "execution_latency": metrics.execution_latency,
                "strategy_evaluation_count": metrics.strategy_evaluation_count,
                "order_execution_count": metrics.order_execution_count,
                "position_updates": metrics.position_updates,
                "websocket_connections": metrics.websocket_connections,
                "api_calls": metrics.api_calls,
                "errors": metrics.errors,
                "memory_usage_mb": metrics.memory_usage_mb,
                "cpu_usage_percent": metrics.cpu_usage_percent,
                "network_latency_ms": metrics.network_latency_ms
            }

            log_metrics_to_csv(metrics_dict, str(self.metrics_file))

        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def _save_health_status(self) -> None:
        """Save health status to JSON file."""
        try:
            health_dict = {}
            for component, status in self.health_status.items():
                health_dict[component] = {
                    'status': status.status,
                    'message': status.message,
                    'last_check': status.last_check.isoformat(),
                    'metrics': status.metrics
                }

            with open(self.health_file, 'w') as f:
                json.dump(health_dict, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving health status: {e}")

    def _save_frontend_status(self) -> None:
        """Save simplified status for frontend API."""
        try:
            # Create frontend-friendly status data
            frontend_data = {
                'overall_status': self._calculate_overall_status(),
                'components': {},
                'alerts_active': list(self.alerts_sent),
                'last_update': datetime.now().isoformat()
            }

            # Add component status in frontend format
            for component, status in self.health_status.items():
                frontend_data['components'][component] = {
                    'status': status.status,
                    'message': status.message,
                    'last_check': status.last_check.isoformat(),
                    'metrics': status.metrics
                }

            # Save to frontend status file
            frontend_file = self.health_file.parent / 'frontend_monitoring_status.json'
            with open(frontend_file, 'w') as f:
                json.dump(frontend_data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving frontend status: {e}")

    def _is_process_running(self, process_name: str) -> bool:
        """Check if a process is running."""
        try:
            result = subprocess.run(
                ['pgrep', '-f', process_name],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        return {
            'overall_status': self._calculate_overall_status(),
            'components': {
                name: {
                    'status': status.status,
                    'message': status.message,
                    'last_check': status.last_check.isoformat(),
                    'metrics': status.metrics
                }
                for name, status in self.health_status.items()
            },
            'recent_metrics': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'evaluation_count': m.strategy_evaluation_count,
                    'execution_count': m.order_execution_count,
                    'memory_mb': m.memory_usage_mb,
                    'cpu_percent': m.cpu_usage_percent
                }
                for m in self.metrics_history[-10:]  # Last 10 metrics
            ],
            'alerts_active': list(self.alerts_sent)
        }

    def _calculate_overall_status(self) -> str:
        """Calculate overall system status."""
        if any(status.status == 'critical' for status in self.health_status.values()):
            return 'critical'
        elif any(status.status == 'warning' for status in self.health_status.values()):
            return 'warning'
        else:
            return 'healthy'
