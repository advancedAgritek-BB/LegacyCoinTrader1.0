#!/usr/bin/env python3
"""
Production Monitoring and Alerting System for LegacyCoinTrader

Provides comprehensive monitoring, alerting, and health checks for production deployment.
"""

import asyncio
import time
import json
import psutil
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.memory_manager import get_memory_manager
from crypto_bot.utils.position_sync_manager import get_position_sync_manager

logger = setup_logger(__name__, LOG_DIR / "production_monitor.log")


class ProductionMonitor:
    """
    Production monitoring system with alerting and health checks.

    Features:
    - System resource monitoring
    - Trading bot health checks
    - Position consistency validation
    - Memory usage tracking
    - Alert generation and notification
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()

        # Monitoring thresholds
        self.memory_threshold = self.config.get("memory_threshold", 85)
        self.cpu_threshold = self.config.get("cpu_threshold", 90)
        self.disk_threshold = self.config.get("disk_threshold", 90)
        self.error_rate_threshold = self.config.get("error_rate_threshold", 5)

        # Monitoring intervals
        self.health_check_interval = self.config.get("health_check_interval", 60)
        self.alert_check_interval = self.config.get("alert_check_interval", 30)

        # Alert system
        self.alerts_enabled = self.config.get("alerts_enabled", True)
        self.alert_cooldown = self.config.get("alert_cooldown", 300)  # 5 minutes
        self.last_alert_time = {}

        # Health tracking
        self.health_history = []
        self.max_history_size = self.config.get("max_history_size", 100)

        # Component references
        self.memory_manager = None
        self.position_sync_manager = None

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task = None

        logger.info("Production monitor initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            "memory_threshold": 85,
            "cpu_threshold": 90,
            "disk_threshold": 90,
            "error_rate_threshold": 5,
            "health_check_interval": 60,
            "alert_check_interval": 30,
            "alerts_enabled": True,
            "alert_cooldown": 300,
            "max_history_size": 100,
            "enable_telegram_alerts": True,
            "enable_email_alerts": False
        }

    def start_monitoring(self):
        """Start production monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Production monitoring started")

    def stop_monitoring(self):
        """Stop production monitoring."""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info("Production monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Perform health checks
                await self._perform_health_checks()

                # Check for alerts
                await self._check_alerts()

                # Wait for next check
                await asyncio.sleep(min(self.health_check_interval, self.alert_check_interval))

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)  # Wait before retry

    async def _perform_health_checks(self):
        """Perform comprehensive health checks."""
        health_data = {
            "timestamp": datetime.now().isoformat(),
            "system": self._check_system_health(),
            "trading_bot": await self._check_trading_bot_health(),
            "memory": self._check_memory_health(),
            "positions": await self._check_position_health(),
            "alerts": self._get_active_alerts()
        }

        # Store in history
        self.health_history.append(health_data)
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]

        # Log health status
        self._log_health_status(health_data)

    def _check_system_health(self) -> Dict[str, Any]:
        """Check system resource health."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # Network connections (for trading bot)
            connections = len(psutil.net_connections())

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "network_connections": connections,
                "status": "healthy" if all([
                    cpu_percent < self.cpu_threshold,
                    memory_percent < self.memory_threshold,
                    disk_percent < self.disk_threshold
                ]) else "warning"
            }

        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return {"error": str(e), "status": "error"}

    async def _check_trading_bot_health(self) -> Dict[str, Any]:
        """Check trading bot health."""
        try:
            # Check if bot process is running
            bot_running = False
            bot_pid = None

            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'crypto_bot' in ' '.join(proc.info['cmdline'] or []):
                        bot_running = True
                        bot_pid = proc.info['pid']
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Check bot logs for recent activity
            log_files = [
                LOG_DIR / "bot.log",
                LOG_DIR / "execution.log",
                LOG_DIR / "bot_controller.log"
            ]

            recent_activity = False
            for log_file in log_files:
                if log_file.exists():
                    mtime = log_file.stat().st_mtime
                    if time.time() - mtime < 300:  # 5 minutes
                        recent_activity = True
                        break

            return {
                "running": bot_running,
                "pid": bot_pid,
                "recent_activity": recent_activity,
                "status": "healthy" if bot_running and recent_activity else "warning"
            }

        except Exception as e:
            logger.error(f"Trading bot health check failed: {e}")
            return {"error": str(e), "status": "error"}

    def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory management health."""
        try:
            if not self.memory_manager:
                self.memory_manager = get_memory_manager()

            memory_stats = self.memory_manager.get_memory_stats()

            # Check for memory pressure
            memory_pressure = memory_stats.get("system_memory_percent", 0) > self.memory_threshold

            return {
                "memory_stats": memory_stats,
                "memory_pressure": memory_pressure,
                "status": "warning" if memory_pressure else "healthy"
            }

        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return {"error": str(e), "status": "error"}

    async def _check_position_health(self) -> Dict[str, Any]:
        """Check position synchronization health."""
        try:
            if not self.position_sync_manager:
                self.position_sync_manager = get_position_sync_manager()

            # Get consistency stats
            consistency_stats = self.position_sync_manager.get_consistency_stats()

            # Check for recent inconsistencies
            recent_inconsistencies = consistency_stats.get("recent_avg_inconsistencies", 0) > 0

            return {
                "consistency_stats": consistency_stats,
                "has_inconsistencies": recent_inconsistencies,
                "status": "warning" if recent_inconsistencies else "healthy"
            }

        except Exception as e:
            logger.error(f"Position health check failed: {e}")
            return {"error": str(e), "status": "error"}

    async def _check_alerts(self):
        """Check for alert conditions and send notifications."""
        if not self.alerts_enabled:
            return

        alerts = []

        # Get latest health data
        if self.health_history:
            latest_health = self.health_history[-1]

            # System alerts
            system = latest_health.get("system", {})
            if system.get("status") == "warning":
                alerts.append({
                    "type": "system",
                    "severity": "warning",
                    "message": ".1f",
                    "data": system
                })

            # Trading bot alerts
            bot = latest_health.get("trading_bot", {})
            if bot.get("status") == "warning":
                alerts.append({
                    "type": "trading_bot",
                    "severity": "warning",
                    "message": f"Trading bot health check failed: running={bot.get('running', False)}",
                    "data": bot
                })

            # Memory alerts
            memory = latest_health.get("memory", {})
            if memory.get("status") == "warning":
                alerts.append({
                    "type": "memory",
                    "severity": "warning",
                    "message": "Memory pressure detected",
                    "data": memory
                })

            # Position alerts
            positions = latest_health.get("positions", {})
            if positions.get("status") == "warning":
                alerts.append({
                    "type": "positions",
                    "severity": "warning",
                    "message": "Position synchronization issues detected",
                    "data": positions
                })

        # Send alerts (with cooldown)
        for alert in alerts:
            await self._send_alert(alert)

    async def _send_alert(self, alert: Dict[str, Any]):
        """Send alert notification."""
        alert_key = f"{alert['type']}_{alert['severity']}"

        # Check cooldown
        current_time = time.time()
        last_alert = self.last_alert_time.get(alert_key, 0)

        if current_time - last_alert < self.alert_cooldown:
            return  # Still in cooldown

        self.last_alert_time[alert_key] = current_time

        try:
            # Telegram alerts
            if self.config.get("enable_telegram_alerts", True):
                await self._send_telegram_alert(alert)

            # Email alerts
            if self.config.get("enable_email_alerts", False):
                await self._send_email_alert(alert)

            # Log alert
            logger.warning(f"ALERT: {alert['message']}")

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    async def _send_telegram_alert(self, alert: Dict[str, Any]):
        """Send alert via Telegram."""
        try:
            from crypto_bot.utils.telegram import TelegramNotifier

            # Get telegram config
            telegram_config = self.config.get("telegram", {})
            if not telegram_config.get("token") or not telegram_config.get("chat_id"):
                return

            notifier = TelegramNotifier(
                token=telegram_config["token"],
                chat_id=telegram_config["chat_id"]
            )

            emoji_map = {
                "error": "🚨",
                "warning": "⚠️",
                "info": "ℹ️"
            }

            emoji = emoji_map.get(alert.get("severity", "info"), "ℹ️")
            message = f"{emoji} **PRODUCTION ALERT**\n\n{alert['message']}"

            notifier.notify(message)

        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")

    async def _send_email_alert(self, alert: Dict[str, Any]):
        """Send alert via email."""
        # Email implementation would go here
        logger.info(f"Email alert: {alert['message']}")

    def _log_health_status(self, health_data: Dict[str, Any]):
        """Log health status summary."""
        system = health_data.get("system", {})
        bot = health_data.get("trading_bot", {})
        memory = health_data.get("memory", {})
        positions = health_data.get("positions", {})

        # Determine overall status
        statuses = [system.get("status", "unknown"),
                   bot.get("status", "unknown"),
                   memory.get("status", "unknown"),
                   positions.get("status", "unknown")]

        if "error" in statuses:
            overall_status = "ERROR"
        elif "warning" in statuses:
            overall_status = "WARNING"
        else:
            overall_status = "HEALTHY"

        logger.info(f"Health Check - Overall: {overall_status} | "
                   f"System: {system.get('status', 'unknown')} | "
                   f"Bot: {bot.get('status', 'unknown')} | "
                   f"Memory: {memory.get('status', 'unknown')} | "
                   f"Positions: {positions.get('status', 'unknown')}")

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        if not self.health_history:
            return {"status": "no_data", "message": "No health data available"}

        latest = self.health_history[-1]

        # Calculate trends
        if len(self.health_history) >= 2:
            previous = self.health_history[-2]
            trends = self._calculate_trends(latest, previous)
        else:
            trends = {}

        return {
            "timestamp": latest["timestamp"],
            "overall_status": self._calculate_overall_status(latest),
            "latest_health": latest,
            "trends": trends,
            "alert_history": list(self.last_alert_time.keys()),
            "monitoring_active": self.is_monitoring
        }

    def _calculate_overall_status(self, health_data: Dict[str, Any]) -> str:
        """Calculate overall system status."""
        components = ["system", "trading_bot", "memory", "positions"]
        statuses = []

        for component in components:
            status = health_data.get(component, {}).get("status", "unknown")
            statuses.append(status)

        if "error" in statuses:
            return "error"
        elif "warning" in statuses:
            return "warning"
        else:
            return "healthy"

    def _calculate_trends(self, latest: Dict[str, Any], previous: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate health trends."""
        trends = {}

        # Memory trend
        latest_mem = latest.get("system", {}).get("memory_percent", 0)
        prev_mem = previous.get("system", {}).get("memory_percent", 0)
        trends["memory_change"] = latest_mem - prev_mem

        # CPU trend
        latest_cpu = latest.get("system", {}).get("cpu_percent", 0)
        prev_cpu = previous.get("system", {}).get("cpu_percent", 0)
        trends["cpu_change"] = latest_cpu - prev_cpu

        return trends

    def export_health_data(self, filepath: str):
        """Export health data to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "health_history": self.health_history,
                    "alert_history": self.last_alert_time,
                    "config": self.config
                }, f, indent=2, default=str)

            logger.info(f"Health data exported to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export health data: {e}")


# Global monitor instance
_monitor = None

def get_production_monitor(config: Optional[Dict[str, Any]] = None) -> ProductionMonitor:
    """Get or create global production monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = ProductionMonitor(config)
    return _monitor


async def main():
    """Main function for running production monitor."""
    import argparse

    parser = argparse.ArgumentParser(description="Production Monitor for LegacyCoinTrader")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--export", help="Export health data to file")
    parser.add_argument("--report", action="store_true", help="Generate health report")

    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    # Create monitor
    monitor = get_production_monitor(config)

    if args.report:
        # Generate and print report
        report = monitor.get_health_report()
        print(json.dumps(report, indent=2, default=str))
        return

    if args.export:
        # Export health data
        monitor.export_health_data(args.export)
        return

    # Start monitoring
    monitor.start_monitoring()

    try:
        print("Production monitoring started. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(60)
            report = monitor.get_health_report()
            print(f"Status: {report['overall_status']} | "
                  f"Checks: {len(monitor.health_history)}")

    except KeyboardInterrupt:
        print("\nStopping production monitor...")
        monitor.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
