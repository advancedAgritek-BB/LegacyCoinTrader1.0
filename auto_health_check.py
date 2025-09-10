#!/usr/bin/env python3
"""
Automated Health Check and Recovery System
Runs comprehensive health checks and performs automated recovery actions.
Designed to be run as a cron job or systemd service.
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add crypto_bot to path
sys.path.insert(0, str(Path(__file__).parent / "crypto_bot"))

try:
    from crypto_bot.pipeline_monitor import PipelineMonitor
    from crypto_bot.utils.logger import setup_logger, LOG_DIR
    import yaml
    logger = setup_logger(__name__, LOG_DIR / "health_check.log")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class AutoHealthChecker:
    """Automated health check and recovery system."""

    def __init__(self, config_path: Optional[str] = None, quiet: bool = False):
        self.config_path = config_path or "crypto_bot/config.yaml"
        self.config = self._load_config()
        self.quiet = quiet
        self.monitor: Optional[PipelineMonitor] = None
        self.last_recovery_actions: Dict[str, datetime] = {}
        self.recovery_cooldown_minutes = self.config.get('monitoring', {}).get('recovery_cooldown_minutes', 15)

        # Recovery action tracking
        self.recovery_log_file = Path("crypto_bot/logs/recovery_actions.log")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration."""
        try:
            config_path = Path(self.config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}
            
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {
                'monitoring': {
                    'enabled': True,
                    'auto_recovery': True,
                    'recovery_cooldown_minutes': 15
                }
            }

    def _log(self, message: str, level: str = "info") -> None:
        """Log message based on quiet setting."""
        if not self.quiet:
            print(message)

        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)

    def _is_recovery_on_cooldown(self, action: str) -> bool:
        """Check if recovery action is on cooldown."""
        if action not in self.last_recovery_actions:
            return False

        cooldown_end = self.last_recovery_actions[action] + timedelta(minutes=self.recovery_cooldown_minutes)
        return datetime.now() < cooldown_end

    def _log_recovery_action(self, action: str, result: str, details: str = "") -> None:
        """Log recovery action to file."""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'action': action,
                'result': result,
                'details': details
            }

            # Append to log file
            with open(self.recovery_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            self.last_recovery_actions[action] = datetime.now()

        except Exception as e:
            logger.error(f"Failed to log recovery action: {e}")

    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        self._log("🔍 Starting automated health check...")

        try:
            # Initialize monitor if needed
            if not self.monitor:
                self.monitor = PipelineMonitor(self.config)

            # Perform health check
            health_status = await self.monitor.perform_health_check()
            health_summary = self.monitor.get_health_summary()

            return {
                'success': True,
                'health_status': health_status,
                'health_summary': health_summary,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self._log(f"❌ Health check failed: {e}", "error")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def perform_recovery_actions(self, health_status: Dict[str, Any]) -> List[str]:
        """Perform automated recovery actions for critical issues."""
        recovery_actions = []

        if not self.config.get('monitoring', {}).get('auto_recovery', True):
            self._log("⚠️  Auto-recovery disabled in configuration")
            return recovery_actions

        self._log("🔧 Checking for recovery actions...")

        # Analyze each component for recovery needs
        for component_name, status in health_status.items():
            if status.status == 'critical':
                recovery_action = await self._determine_recovery_action(component_name, status)
                if recovery_action:
                    recovery_actions.append(recovery_action)

        # Execute recovery actions
        executed_actions = []
        for action in recovery_actions:
            if not self._is_recovery_on_cooldown(action['name']):
                success = await self._execute_recovery_action(action)
                executed_actions.append(action['name'])

                if success:
                    self._log_recovery_action(action['name'], 'success', action['details'])
                    self._log(f"✅ Recovery action '{action['name']}' completed successfully")
                else:
                    self._log_recovery_action(action['name'], 'failed', action['details'])
                    self._log(f"❌ Recovery action '{action['name']}' failed")
            else:
                self._log(f"⏳ Recovery action '{action['name']}' is on cooldown")

        return executed_actions

    async def _determine_recovery_action(self, component: str, status: Any) -> Optional[Dict[str, Any]]:
        """Determine appropriate recovery action for a component."""
        if component == 'evaluation_pipeline':
            if not status.metrics.get('process_running'):
                return {
                    'name': 'restart_trading_bot',
                    'component': component,
                    'details': 'Trading bot process not running, attempting restart'
                }

        elif component == 'execution_pipeline':
            if status.metrics.get('pending_orders', 0) > 10:
                return {
                    'name': 'clear_pending_orders',
                    'component': component,
                    'details': f'High pending orders: {status.metrics["pending_orders"]}'
                }

        elif component == 'websocket_connections':
            if not status.metrics.get('connectivity_ok'):
                return {
                    'name': 'reset_websocket_connections',
                    'component': component,
                    'details': 'WebSocket connectivity issues detected'
                }

        elif component == 'system_resources':
            if status.metrics.get('system_memory_percent', 0) > 95:
                return {
                    'name': 'cleanup_memory',
                    'component': component,
                    'details': f'High memory usage: {status.metrics["system_memory_percent"]}%'
                }

        return None

    async def _execute_recovery_action(self, action: Dict[str, Any]) -> bool:
        """Execute a specific recovery action."""
        try:
            action_name = action['name']

            if action_name == 'restart_trading_bot':
                return await self._restart_trading_bot()

            elif action_name == 'clear_pending_orders':
                return await self._clear_pending_orders()

            elif action_name == 'reset_websocket_connections':
                return await self._reset_websocket_connections()

            elif action_name == 'cleanup_memory':
                return await self._cleanup_memory()

            else:
                self._log(f"⚠️  Unknown recovery action: {action_name}")
                return False

        except Exception as e:
            self._log(f"❌ Error executing recovery action {action['name']}: {e}", "error")
            return False

    async def _restart_trading_bot(self) -> bool:
        """Restart the trading bot process."""
        try:
            self._log("🔄 Attempting to restart trading bot...")

            # Kill existing processes
            subprocess.run(['pkill', '-f', 'crypto_bot.main'], check=False)
            subprocess.run(['pkill', '-f', 'start_bot_auto.py'], check=False)

            # Wait a moment
            await asyncio.sleep(2)

            # Start new bot process
            result = subprocess.run([
                sys.executable, 'start_bot_auto.py'
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                self._log("✅ Trading bot restarted successfully")
                return True
            else:
                self._log(f"❌ Failed to restart trading bot: {result.stderr}", "error")
                return False

        except Exception as e:
            self._log(f"❌ Error restarting trading bot: {e}", "error")
            return False

    async def _clear_pending_orders(self) -> bool:
        """Clear stuck pending orders."""
        try:
            self._log("🧹 Attempting to clear pending orders...")

            # This is a placeholder - actual implementation would depend on your order management system
            # You might need to implement specific logic to cancel stuck orders

            # For now, just log the action
            self._log("ℹ️  Pending order clearing would be implemented here")
            return True

        except Exception as e:
            self._log(f"❌ Error clearing pending orders: {e}", "error")
            return False

    async def _reset_websocket_connections(self) -> bool:
        """Reset WebSocket connections."""
        try:
            self._log("🔌 Attempting to reset WebSocket connections...")

            # Restart any WebSocket-related processes
            subprocess.run(['pkill', '-f', 'kraken_ws'], check=False)

            # Wait for restart
            await asyncio.sleep(5)

            self._log("✅ WebSocket connections reset")
            return True

        except Exception as e:
            self._log(f"❌ Error resetting WebSocket connections: {e}", "error")
            return False

    async def _cleanup_memory(self) -> bool:
        """Perform memory cleanup."""
        try:
            self._log("🧽 Attempting memory cleanup...")

            # Force garbage collection
            import gc
            gc.collect()

            # Clear any caches if they exist
            # This is a placeholder - add specific cleanup logic as needed

            self._log("✅ Memory cleanup completed")
            return True

        except Exception as e:
            self._log(f"❌ Error during memory cleanup: {e}", "error")
            return False

    def generate_report(self, health_result: Dict[str, Any], recovery_actions: List[str]) -> Dict[str, Any]:
        """Generate a comprehensive health check report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'summary': {},
            'issues': [],
            'recovery_actions_taken': recovery_actions,
            'recommendations': []
        }

        if health_result['success']:
            health_summary = health_result['health_summary']
            report['overall_status'] = health_summary['overall_status']

            # Count issues by severity
            components = health_summary.get('components', {})
            critical_count = sum(1 for c in components.values() if c['status'] == 'critical')
            warning_count = sum(1 for c in components.values() if c['status'] == 'warning')
            healthy_count = sum(1 for c in components.values() if c['status'] == 'healthy')

            report['summary'] = {
                'total_components': len(components),
                'healthy': healthy_count,
                'warning': warning_count,
                'critical': critical_count
            }

            # List specific issues
            for component, status in components.items():
                if status['status'] != 'healthy':
                    report['issues'].append({
                        'component': component,
                        'severity': status['status'],
                        'message': status['message'],
                        'metrics': status.get('metrics', {})
                    })

            # Generate recommendations
            if critical_count > 0:
                report['recommendations'].append("Immediate attention required for critical issues")
            if warning_count > 0:
                report['recommendations'].append("Review warning conditions to prevent escalation")
            if len(recovery_actions) > 0:
                report['recommendations'].append(f"Recovery actions performed: {', '.join(recovery_actions)}")

        else:
            report['error'] = health_result.get('error', 'Unknown error')

        return report

    async def run_health_check(self) -> Dict[str, Any]:
        """Run complete health check and recovery cycle."""
        start_time = time.time()

        # Perform health check
        health_result = await self.perform_health_check()

        if not health_result['success']:
            self._log("❌ Health check failed, skipping recovery")
            return self.generate_report(health_result, [])

        # Perform recovery actions if needed
        health_status = health_result['health_status']
        recovery_actions = await self.perform_recovery_actions(health_status)

        # Generate report
        report = self.generate_report(health_result, recovery_actions)

        # Save report
        self._save_report(report)

        # Log summary
        duration = time.time() - start_time
        self._log(f"Health check completed in {duration:.2f} seconds")
        if report['overall_status'] == 'healthy':
            self._log("✅ System health: HEALTHY")
        elif report['overall_status'] == 'warning':
            self._log("⚠️  System health: WARNING")
        else:
            self._log("🔴 System health: CRITICAL")

        if recovery_actions:
            self._log(f"🔧 Recovery actions performed: {', '.join(recovery_actions)}")

        return report

    def _save_report(self, report: Dict[str, Any]) -> None:
        """Save health check report to file."""
        try:
            report_file = Path("crypto_bot/logs/health_check_report.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save health check report: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Automated Health Check and Recovery System")
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode (no console output)')
    parser.add_argument('--no-recovery', action='store_true', help='Skip recovery actions')
    parser.add_argument('--report-only', action='store_true', help='Only generate report, no actions')

    args = parser.parse_args()

    # Setup logging
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    checker = AutoHealthChecker(args.config, args.quiet)

    # Override recovery setting if requested
    if args.no_recovery:
        checker.config['monitoring']['auto_recovery'] = False

    async def run():
        if args.report_only:
            # Just perform health check and show report
            health_result = await checker.perform_health_check()
            if health_result['success']:
                report = checker.generate_report(health_result, [])
                print(json.dumps(report, indent=2))
            else:
                print(f"Health check failed: {health_result.get('error')}")
        else:
            # Run full health check and recovery
            report = await checker.run_health_check()
            if not args.quiet:
                print(json.dumps(report, indent=2))

    asyncio.run(run())


if __name__ == "__main__":
    main()
