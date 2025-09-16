#!/usr/bin/env python3
"""
Enhanced Pipeline Monitoring System
Comprehensive monitoring for evaluation and execution pipelines with automated recovery.
"""

import asyncio
import time
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import argparse
import signal

# Add crypto_bot to path
sys.path.insert(0, str(Path(__file__).parent / "crypto_bot"))

try:
    from crypto_bot.pipeline_monitor import PipelineMonitor
    from crypto_bot.utils.logger import setup_logger, LOG_DIR
    from crypto_bot.config import load_config
    logger = setup_logger(__name__, LOG_DIR / "enhanced_monitoring.log")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class EnhancedMonitor:
    """Enhanced monitoring system with automated recovery capabilities."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "crypto_bot/config.yaml"
        self.config = self._load_config()
        self.pipeline_monitor: Optional[PipelineMonitor] = None
        self.running = False
        self.last_recovery_attempt = 0
        self.recovery_cooldown = 300  # 5 minutes between recovery attempts

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with monitoring settings."""
        try:
            config = load_config(self.config_path)

            # Add default monitoring configuration if not present
            if 'monitoring' not in config:
                config['monitoring'] = {
                    'enabled': True,
                    'check_interval_seconds': 30,
                    'alert_interval_seconds': 300,
                    'auto_recovery_enabled': True,
                    'max_recovery_attempts': 3,
                    'recovery_actions': {
                        'restart_bot': True,
                        'clear_stuck_orders': True,
                        'reset_websocket': True
                    }
                }

            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return minimal config
            return {
                'monitoring': {
                    'enabled': True,
                    'check_interval_seconds': 30,
                    'alert_interval_seconds': 300,
                    'auto_recovery_enabled': True,
                    'max_recovery_attempts': 3
                }
            }

    async def start_monitoring(self) -> None:
        """Start the enhanced monitoring system."""
        logger.info("🚀 Starting Enhanced Pipeline Monitoring System")
        print("🚀 Enhanced Pipeline Monitoring System")
        print("=" * 50)

        try:
            # Initialize pipeline monitor
            self.pipeline_monitor = PipelineMonitor(self.config)
            await self.pipeline_monitor.start_monitoring()

            self.running = True
            logger.info("✅ Monitoring system initialized successfully")

            # Main monitoring loop
            while self.running:
                try:
                    await self._monitoring_cycle()
                    await asyncio.sleep(self.config['monitoring']['check_interval_seconds'])
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {e}")
                    await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            print(f"❌ Failed to start monitoring: {e}")
            sys.exit(1)

    async def _monitoring_cycle(self) -> None:
        """Perform one complete monitoring cycle."""
        try:
            # Get current health status
            health_status = await self.pipeline_monitor.perform_health_check()
            health_summary = self.pipeline_monitor.get_health_summary()

            # Display status
            self._display_status(health_summary)

            # Check for critical issues requiring recovery
            if health_summary['overall_status'] == 'critical':
                await self._attempt_recovery(health_status)

            # Save status report
            self._save_status_report(health_summary)

        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")

    def _display_status(self, health_summary: Dict[str, Any]) -> None:
        """Display current system status."""
        overall_status = health_summary['overall_status']
        components = health_summary['components']

        # Clear screen and show header
        print("\033[2J\033[H", end="")  # Clear screen
        print("🚀 ENHANCED PIPELINE MONITOR")
        print(f"📊 Overall Status: {self._status_emoji(overall_status)} {overall_status.upper()}")
        print(f"🕐 Last Update: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 50)

        # Component status
        for name, status in components.items():
            emoji = self._status_emoji(status['status'])
            print("15")

        # Active alerts
        if health_summary['alerts_active']:
            print("\n🚨 ACTIVE ALERTS:")
            for alert in health_summary['alerts_active']:
                print(f"  • {alert}")

        # Recent metrics
        recent = health_summary.get('recent_metrics', [])
        if recent:
            latest = recent[-1]
            print("📈 RECENT METRICS:")
            print(f"  • Strategy Evaluations: {latest['evaluation_count']}")
            print(f"  • Order Executions: {latest['execution_count']}")
            print(f"  • PnL: ${latest.get('pnl', 0):.1f}")
            print(f"  • Win Rate: {latest.get('win_rate', 0):.1f}%")

        print("-" * 50)

    def _status_emoji(self, status: str) -> str:
        """Get emoji for status."""
        return {
            'healthy': '🟢',
            'warning': '🟡',
            'critical': '🔴'
        }.get(status, '⚪')

    async def _attempt_recovery(self, health_status: Dict[str, Any]) -> None:
        """Attempt automated recovery for critical issues."""
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_recovery_attempt < self.recovery_cooldown:
            logger.info("Recovery on cooldown, skipping")
            return

        logger.warning("🔧 Attempting automated recovery for critical issues")
        self.last_recovery_attempt = current_time

        recovery_actions = []

        # Analyze issues and determine recovery actions
        for component, status in health_status.items():
            if status.status == 'critical':
                if component == 'evaluation_pipeline' and status.metrics.get('process_running') == False:
                    recovery_actions.append('restart_bot')
                elif component == 'execution_pipeline' and status.metrics.get('pending_orders', 0) > 10:
                    recovery_actions.append('clear_stuck_orders')
                elif component == 'websocket_connections' and not status.metrics.get('connectivity_ok'):
                    recovery_actions.append('reset_websocket')

        # Execute recovery actions
        for action in set(recovery_actions):  # Remove duplicates
            try:
                await self._execute_recovery_action(action)
                logger.info(f"✅ Recovery action '{action}' completed")
            except Exception as e:
                logger.error(f"❌ Recovery action '{action}' failed: {e}")

    async def _execute_recovery_action(self, action: str) -> None:
        """Execute a specific recovery action."""
        if action == 'restart_bot':
            logger.info("Restarting trading bot...")
            # Kill existing bot process
            import subprocess
            try:
                subprocess.run(['pkill', '-f', 'crypto_bot.main'], check=False)
                await asyncio.sleep(2)
                # Start new bot process
                subprocess.Popen(['python3', 'start_bot_auto.py'])
            except Exception as e:
                logger.error(f"Failed to restart bot: {e}")

        elif action == 'clear_stuck_orders':
            logger.info("Clearing stuck orders...")
            # Implementation would depend on your order management system
            # This is a placeholder for the actual implementation
            pass

        elif action == 'reset_websocket':
            logger.info("Resetting WebSocket connections...")
            # Implementation would depend on your WebSocket management
            # This is a placeholder for the actual implementation
            pass

    def _save_status_report(self, health_summary: Dict[str, Any]) -> None:
        """Save status report to file."""
        try:
            report_file = Path("crypto_bot/logs/monitoring_report.json")
            report = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': health_summary['overall_status'],
                'components': health_summary['components'],
                'alerts': health_summary['alerts_active']
            }

            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save status report: {e}")

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    async def shutdown(self) -> None:
        """Shutdown the monitoring system."""
        logger.info("Shutting down enhanced monitoring system")
        self.running = False

        if self.pipeline_monitor:
            await self.pipeline_monitor.stop_monitoring()

        print("\n👋 Enhanced monitoring system stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced Pipeline Monitoring System")
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--daemon', '-d', action='store_true', help='Run as daemon')
    args = parser.parse_args()

    monitor = EnhancedMonitor(args.config)

    if args.daemon:
        # Run as daemon
        import daemon
        with daemon.DaemonContext():
            asyncio.run(monitor.start_monitoring())
    else:
        # Run interactively
        try:
            asyncio.run(monitor.start_monitoring())
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        finally:
            asyncio.run(monitor.shutdown())


if __name__ == "__main__":
    main()
