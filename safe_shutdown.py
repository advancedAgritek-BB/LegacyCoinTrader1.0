#!/usr/bin/env python3
"""
Safe Shutdown System for LegacyCoinTrader

A simplified, robust shutdown system that safely terminates all application
processes with proper error handling and cleanup.
"""

import sys
import time
import signal
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

try:
    import psutil
except ImportError:
    print("❌ psutil not installed. Run: pip install psutil")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SafeShutdown:
    """Safe shutdown system for LegacyCoinTrader."""

    def __init__(self) -> None:
        self.project_root = Path(__file__).parent
        self.shutdown_initiated = False

        # Known PID files and patterns
        self.components = {
            "Trading Bot": {
                "pid_file": "bot_pid.txt",
                "pattern": "crypto_bot.main",
                "priority": 1,
                "critical": True
            },
            "Web Frontend": {
                "pid_file": "frontend.pid",
                "pattern": "frontend.app",
                "priority": 2,
                "critical": False
            },
            "Enhanced Scanner": {
                "pid_file": "scanner.pid",
                "pattern": "enhanced_scanner.py",
                "priority": 3,
                "critical": True
            },
            "Strategy Router": {
                "pid_file": "strategy_router.pid",
                "pattern": "strategy_router.py",
                "priority": 4,
                "critical": True
            },
            "WebSocket Monitor": {
                "pid_file": "websocket_monitor.pid",
                "pattern": "websocket_monitor.py",
                "priority": 5,
                "critical": False
            },
            "Monitoring System": {
                "pid_file": "monitoring.pid",
                "pattern": "enhanced_monitoring.py",
                "priority": 6,
                "critical": False
            },
            "Telegram Bot": {
                "pid_file": "telegram.pid",
                "pattern": "telegram_ctl.py",
                "priority": 7,
                "critical": False
            }
        }

    def find_processes(self) -> Dict[str, List[psutil.Process]]:
        """Find all running application processes."""
        logger.info("🔍 Discovering running processes...")
        found_processes: Dict[str, List[psutil.Process]] = {}

        for component_name, config in self.components.items():
            processes = []

            # Check PID file first
            pid_file = self.project_root / config["pid_file"]
            if pid_file.exists():
                try:
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())

                    if psutil.pid_exists(pid):
                        proc = psutil.Process(pid)
                        if self._is_our_process(proc, config["pattern"]):
                            processes.append(proc)
                            logger.info(f"✅ Found {component_name} via PID file "
                                      f"(PID: {pid})")
                        else:
                            logger.warning(f"⚠️ Stale PID file: {pid_file}")
                            pid_file.unlink()
                    else:
                        logger.warning(f"⚠️ Stale PID file: {pid_file}")
                        pid_file.unlink()
                except (ValueError, FileNotFoundError, psutil.NoSuchProcess):
                    if pid_file.exists():
                        pid_file.unlink()

            # Search by pattern if not found via PID file
            if not processes:
                pattern_processes = self._find_by_pattern(config["pattern"])
                processes.extend(pattern_processes)
                if pattern_processes:
                    pids = [p.pid for p in pattern_processes]
                    logger.info(f"✅ Found {component_name} via pattern "
                              f"(PIDs: {pids})")

            if processes:
                found_processes[component_name] = processes

        total_processes = sum(len(procs) for procs in found_processes.values())
        logger.info(f"📊 Found {total_processes} processes across "
                   f"{len(found_processes)} components")
        return found_processes

    def _is_our_process(self, process: psutil.Process, pattern: str) -> bool:
        """Check if process belongs to our application."""
        try:
            cmdline = ' '.join(process.cmdline())
            cwd = process.cwd()
            return (pattern in cmdline and
                    str(self.project_root) in cwd)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def _find_by_pattern(self, pattern: str) -> List[psutil.Process]:
        """Find processes by command line pattern."""
        found = []
        for proc in psutil.process_iter(['pid', 'cmdline', 'cwd']):
            try:
                if (proc.info['cmdline'] and
                        pattern in ' '.join(proc.info['cmdline']) and
                        proc.info['cwd'] and
                        str(self.project_root) in proc.info['cwd']):
                    found.append(psutil.Process(proc.info['pid']))
            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                continue
        return found

    def backup_critical_data(self) -> bool:
        """Backup critical data before shutdown."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = (self.project_root / "backups" /
                         f"shutdown_{timestamp}")
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Files to backup
            critical_files = [
                "logs/trades.csv",
                "logs/positions.log",
                "config.yaml"
            ]

            import shutil
            for file_path in critical_files:
                source = self.project_root / file_path
                if source.exists():
                    dest = backup_dir / file_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dest)

            logger.info(f"💾 Critical data backed up to {backup_dir}")
            return True
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False

    def shutdown_process(self, process: psutil.Process,
                        component_name: str) -> bool:
        """Shutdown a single process safely."""
        try:
            logger.info(f"🛑 Shutting down {component_name} "
                       f"(PID: {process.pid})")

            if not process.is_running():
                logger.info(f"✅ {component_name} already stopped")
                return True

            # Try graceful shutdown first
            process.terminate()
            try:
                process.wait(timeout=10)
                logger.info(f"✅ {component_name} stopped gracefully")
                return True
            except psutil.TimeoutExpired:
                logger.warning(f"⏰ {component_name} didn't stop gracefully, "
                              "force killing...")

            # Force kill
            process.kill()
            try:
                process.wait(timeout=5)
                logger.info(f"💀 {component_name} force killed")
                return True
            except psutil.TimeoutExpired:
                logger.error(f"❌ Failed to kill {component_name}")
                return False

        except psutil.NoSuchProcess:
            logger.info(f"✅ {component_name} already stopped")
            return True
        except Exception as e:
            logger.error(f"❌ Error shutting down {component_name}: {e}")
            return False

    def cleanup_pid_files(self) -> None:
        """Clean up all PID files."""
        logger.info("🧹 Cleaning up PID files...")
        for component_name, config in self.components.items():
            pid_file = self.project_root / config["pid_file"]
            if pid_file.exists():
                try:
                    pid_file.unlink()
                    logger.debug(f"🧹 Removed {config['pid_file']}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to remove {config['pid_file']}: "
                                  f"{e}")

    def execute_shutdown(self, force: bool = False) -> bool:
        """Execute the shutdown process."""
        if self.shutdown_initiated:
            logger.warning("🔄 Shutdown already in progress")
            return False

        self.shutdown_initiated = True
        logger.info("🚀 Starting safe shutdown process...")

        try:
            # Find all processes
            found_processes = self.find_processes()

            if not found_processes:
                logger.info("✅ No processes found - system already shut down")
                return True

            # Backup critical data
            if not force:
                self.backup_critical_data()

            # Sort components by priority and shut down
            sorted_components = sorted(
                found_processes.items(),
                key=lambda x: self.components[x[0]]["priority"]
            )

            all_success = True
            for component_name, processes in sorted_components:
                for process in processes:
                    success = self.shutdown_process(process, component_name)
                    if not success:
                        all_success = False

            # Clean up PID files
            self.cleanup_pid_files()

            # Verify shutdown
            remaining = self.find_processes()
            if remaining:
                logger.warning(f"⚠️ {len(remaining)} components still running")
                for name, procs in remaining.items():
                    logger.warning(f"  • {name}: {len(procs)} processes")
                all_success = False
            else:
                logger.info("✅ All processes shut down successfully")

            return all_success

        except Exception as e:
            logger.error(f"❌ Unexpected error during shutdown: {e}")
            return False


def setup_signal_handlers(shutdown_manager: SafeShutdown) -> None:
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum: int, frame) -> None:
        signal_name = signal.Signals(signum).name
        logger.info(f"📡 Received {signal_name} signal, "
                   "initiating shutdown...")

        force = signum == signal.SIGKILL
        success = shutdown_manager.execute_shutdown(force=force)
        sys.exit(0 if success else 1)

    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)   # Hang up


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Safe Shutdown System for LegacyCoinTrader"
    )
    parser.add_argument("--force", action="store_true",
                       help="Force shutdown without backup")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be shut down")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    shutdown_manager = SafeShutdown()
    setup_signal_handlers(shutdown_manager)

    try:
        if args.dry_run:
            logger.info("🔍 DRY RUN - Discovering processes...")
            found = shutdown_manager.find_processes()

            if found:
                logger.info("📋 Processes that would be shut down:")
                sorted_components = sorted(
                    found.items(),
                    key=lambda x: shutdown_manager.components[x[0]]["priority"]
                )
                for name, processes in sorted_components:
                    priority = shutdown_manager.components[name]["priority"]
                    for proc in processes:
                        logger.info(f"  {priority}. {name} (PID: {proc.pid})")
            else:
                logger.info("✅ No processes found to shutdown")

            return 0

        # Execute shutdown
        success = shutdown_manager.execute_shutdown(force=args.force)
        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("🛑 Shutdown interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
