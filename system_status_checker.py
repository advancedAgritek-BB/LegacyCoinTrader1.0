#!/usr/bin/env python3
"""
System Status Checker for LegacyCoinTrader

This script provides comprehensive status checking for all application components,
helping to determine what processes are running and their health status.
"""

import os
import sys
import psutil
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "crypto_bot"))

@dataclass
class ProcessStatus:
    """Status information for a process."""
    name: str
    pid: Optional[int] = None
    running: bool = False
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    uptime_seconds: float = 0.0
    pid_file_exists: bool = False
    pid_file_valid: bool = False
    health_status: str = "unknown"  # healthy, warning, error, unknown

class SystemStatusChecker:
    """Check the status of all LegacyCoinTrader components."""
    
    def __init__(self):
        self.project_root = project_root
        
        # Known processes and their patterns
        self.process_definitions = {
            "Trading Bot": {
                "pattern": "crypto_bot.main",
                "pid_file": "bot_pid.txt",
                "priority": 1,
                "critical": True
            },
            "Web Frontend": {
                "pattern": "frontend.app",
                "pid_file": "frontend.pid",
                "priority": 2,
                "critical": False
            },
            "Enhanced Scanner": {
                "pattern": "enhanced_scanner.py",
                "pid_file": "scanner.pid",
                "priority": 3,
                "critical": True
            },
            "Strategy Router": {
                "pattern": "strategy_router.py",
                "pid_file": "strategy_router.pid",
                "priority": 4,
                "critical": True
            },
            "WebSocket Monitor": {
                "pattern": "websocket_monitor.py",
                "pid_file": "websocket_monitor.pid",
                "priority": 5,
                "critical": False
            },
            "Monitoring System": {
                "pattern": "enhanced_monitoring.py",
                "pid_file": "monitoring.pid",
                "priority": 6,
                "critical": False
            },
            "Health Check": {
                "pattern": "auto_health_check.py",
                "pid_file": "health_check.pid",
                "priority": 7,
                "critical": False
            },
            "Telegram Bot": {
                "pattern": "telegram_ctl.py",
                "pid_file": "telegram.pid",
                "priority": 8,
                "critical": False
            },
            "Production Monitor": {
                "pattern": "production_monitor.py",
                "pid_file": "production.pid",
                "priority": 9,
                "critical": False
            },
            "API Server": {
                "pattern": "simple_api_server.py",
                "pid_file": "api_server.pid",
                "priority": 10,
                "critical": False
            }
        }

    def check_pid_file(self, pid_file_name: str) -> Tuple[bool, Optional[int], bool]:
        """Check if PID file exists and is valid."""
        pid_file = self.project_root / pid_file_name
        
        if not pid_file.exists():
            return False, None, False
        
        try:
            with open(pid_file, 'r') as f:
                pid_str = f.read().strip()
            
            if not pid_str:
                return True, None, False
            
            pid = int(pid_str)
            
            # Check if process exists
            if psutil.pid_exists(pid):
                return True, pid, True
            else:
                return True, pid, False
                
        except (ValueError, FileNotFoundError):
            return True, None, False

    def find_processes_by_pattern(self, pattern: str) -> List[psutil.Process]:
        """Find all processes matching a pattern."""
        found = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd']):
            try:
                if proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if (pattern in cmdline and 
                        proc.info['cwd'] and 
                        str(self.project_root) in proc.info['cwd']):
                        found.append(psutil.Process(proc.info['pid']))
            except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
                continue
        return found

    def get_process_stats(self, process: psutil.Process) -> Dict:
        """Get detailed stats for a process."""
        try:
            with process.oneshot():
                return {
                    'cpu_percent': process.cpu_percent(),
                    'memory_mb': process.memory_info().rss / 1024 / 1024,
                    'uptime_seconds': time.time() - process.create_time(),
                    'status': process.status(),
                    'num_threads': process.num_threads(),
                    'connections': len(process.connections()) if hasattr(process, 'connections') else 0
                }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'uptime_seconds': 0.0,
                'status': 'unknown',
                'num_threads': 0,
                'connections': 0
            }

    def check_process_health(self, process_name: str, process: Optional[psutil.Process]) -> str:
        """Check the health status of a process."""
        if not process:
            return "error"
        
        try:
            # Basic health checks
            if process.status() == psutil.STATUS_ZOMBIE:
                return "error"
            
            # Check CPU usage (warning if > 90% for extended period)
            cpu_percent = process.cpu_percent()
            if cpu_percent > 90:
                return "warning"
            
            # Check memory usage (warning if > 1GB for most processes)
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > 1024:
                return "warning"
            
            # Process-specific health checks
            if process_name == "Trading Bot":
                # Check if trading bot is responsive (could check log file age, etc.)
                return self._check_trading_bot_health()
            elif process_name == "Web Frontend":
                return self._check_web_frontend_health()
            elif process_name == "Enhanced Scanner":
                return self._check_scanner_health()
            
            return "healthy"
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return "error"

    def _check_trading_bot_health(self) -> str:
        """Check trading bot specific health."""
        try:
            # Check if bot log file is being updated recently
            bot_log = self.project_root / "logs" / "bot.log"
            if bot_log.exists():
                last_modified = bot_log.stat().st_mtime
                if time.time() - last_modified > 300:  # 5 minutes
                    return "warning"
            
            # Check for error patterns in recent logs
            error_log = self.project_root / "logs" / "errors.log"
            if error_log.exists():
                last_modified = error_log.stat().st_mtime
                if time.time() - last_modified < 60:  # Recent errors
                    return "warning"
            
            return "healthy"
        except Exception:
            return "unknown"

    def _check_web_frontend_health(self) -> str:
        """Check web frontend specific health."""
        try:
            import requests
            response = requests.get("http://localhost:8001/api/health", timeout=5)
            if response.status_code == 200:
                return "healthy"
            else:
                return "warning"
        except Exception:
            return "warning"

    def _check_scanner_health(self) -> str:
        """Check scanner specific health."""
        try:
            # Check if asset scores are being updated
            scores_file = self.project_root / "logs" / "asset_scores.json"
            if scores_file.exists():
                last_modified = scores_file.stat().st_mtime
                if time.time() - last_modified > 600:  # 10 minutes
                    return "warning"
            return "healthy"
        except Exception:
            return "unknown"

    def check_all_processes(self) -> Dict[str, ProcessStatus]:
        """Check status of all defined processes."""
        status_results = {}
        
        for process_name, definition in self.process_definitions.items():
            # Check PID file
            pid_file_exists, pid_from_file, pid_file_valid = self.check_pid_file(
                definition["pid_file"]
            )
            
            # Find process by pattern
            found_processes = self.find_processes_by_pattern(definition["pattern"])
            
            if pid_file_valid and pid_from_file:
                # Use PID from file if valid
                try:
                    process = psutil.Process(pid_from_file)
                    if process in found_processes or self._is_our_process(process, definition["pattern"]):
                        active_process = process
                    else:
                        active_process = None
                except psutil.NoSuchProcess:
                    active_process = None
            elif found_processes:
                # Use first found process
                active_process = found_processes[0]
            else:
                active_process = None
            
            # Get process stats
            if active_process:
                stats = self.get_process_stats(active_process)
                health = self.check_process_health(process_name, active_process)
                
                status_results[process_name] = ProcessStatus(
                    name=process_name,
                    pid=active_process.pid,
                    running=True,
                    cpu_percent=stats['cpu_percent'],
                    memory_mb=stats['memory_mb'],
                    uptime_seconds=stats['uptime_seconds'],
                    pid_file_exists=pid_file_exists,
                    pid_file_valid=pid_file_valid,
                    health_status=health
                )
            else:
                status_results[process_name] = ProcessStatus(
                    name=process_name,
                    running=False,
                    pid_file_exists=pid_file_exists,
                    pid_file_valid=pid_file_valid,
                    health_status="error" if definition["critical"] else "warning"
                )
        
        return status_results

    def _is_our_process(self, process: psutil.Process, pattern: str) -> bool:
        """Check if a process belongs to our application."""
        try:
            cmdline = ' '.join(process.cmdline())
            cwd = process.cwd()
            return (pattern in cmdline and 
                    str(self.project_root) in cwd)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def get_system_overview(self) -> Dict:
        """Get overall system status overview."""
        process_statuses = self.check_all_processes()
        
        total_processes = len(process_statuses)
        running_processes = sum(1 for status in process_statuses.values() if status.running)
        healthy_processes = sum(1 for status in process_statuses.values() 
                              if status.health_status == "healthy")
        warning_processes = sum(1 for status in process_statuses.values() 
                              if status.health_status == "warning")
        error_processes = sum(1 for status in process_statuses.values() 
                            if status.health_status == "error")
        
        # Determine overall system health
        critical_processes = [name for name, definition in self.process_definitions.items() 
                            if definition["critical"]]
        critical_running = sum(1 for name in critical_processes 
                             if process_statuses[name].running)
        critical_healthy = sum(1 for name in critical_processes 
                             if process_statuses[name].health_status == "healthy")
        
        if critical_running == len(critical_processes) and critical_healthy == len(critical_processes):
            system_health = "healthy"
        elif critical_running == len(critical_processes):
            system_health = "warning"
        else:
            system_health = "error"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": system_health,
            "total_processes": total_processes,
            "running_processes": running_processes,
            "healthy_processes": healthy_processes,
            "warning_processes": warning_processes,
            "error_processes": error_processes,
            "critical_processes_total": len(critical_processes),
            "critical_processes_running": critical_running,
            "critical_processes_healthy": critical_healthy,
            "processes": {
                name: {
                    "running": status.running,
                    "pid": status.pid,
                    "health": status.health_status,
                    "cpu_percent": status.cpu_percent,
                    "memory_mb": round(status.memory_mb, 1),
                    "uptime_hours": round(status.uptime_seconds / 3600, 1),
                    "pid_file_valid": status.pid_file_valid,
                    "critical": self.process_definitions[name]["critical"]
                }
                for name, status in process_statuses.items()
            }
        }

    def print_status_report(self, detailed: bool = False) -> None:
        """Print a formatted status report."""
        overview = self.get_system_overview()
        
        # System header
        print("🔍 LegacyCoinTrader System Status")
        print("=" * 50)
        print(f"📅 Timestamp: {overview['timestamp']}")
        print(f"🏥 System Health: {self._format_health(overview['system_health'])}")
        print(f"🔢 Processes: {overview['running_processes']}/{overview['total_processes']} running")
        print()
        
        # Process status
        print("📊 Process Status:")
        print("-" * 50)
        
        for name, process_info in overview['processes'].items():
            status_icon = "🟢" if process_info['running'] else "🔴"
            health_icon = self._get_health_icon(process_info['health'])
            critical_icon = "⭐" if process_info['critical'] else "  "
            
            print(f"{status_icon} {health_icon} {critical_icon} {name}")
            
            if process_info['running']:
                print(f"    PID: {process_info['pid']}")
                if detailed:
                    print(f"    CPU: {process_info['cpu_percent']:.1f}%")
                    print(f"    Memory: {process_info['memory_mb']:.1f} MB")
                    print(f"    Uptime: {process_info['uptime_hours']:.1f} hours")
                    print(f"    PID File: {'Valid' if process_info['pid_file_valid'] else 'Invalid/Missing'}")
            else:
                print("    Status: Not Running")
            print()
        
        # Legend
        print("Legend:")
        print("🟢 Running  🔴 Stopped")
        print("🟢 Healthy  🟡 Warning  🔴 Error  ❓ Unknown")
        print("⭐ Critical Process")

    def _format_health(self, health: str) -> str:
        """Format health status with color."""
        health_map = {
            "healthy": "🟢 Healthy",
            "warning": "🟡 Warning", 
            "error": "🔴 Error",
            "unknown": "❓ Unknown"
        }
        return health_map.get(health, f"❓ {health}")

    def _get_health_icon(self, health: str) -> str:
        """Get health icon."""
        health_icons = {
            "healthy": "🟢",
            "warning": "🟡",
            "error": "🔴",
            "unknown": "❓"
        }
        return health_icons.get(health, "❓")

    def save_status_report(self, filename: str = None) -> Path:
        """Save status report to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_status_{timestamp}.json"
        
        report_file = self.project_root / "logs" / filename
        report_file.parent.mkdir(exist_ok=True)
        
        overview = self.get_system_overview()
        
        with open(report_file, 'w') as f:
            json.dump(overview, f, indent=2)
        
        print(f"📄 Status report saved to: {report_file}")
        return report_file

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LegacyCoinTrader System Status Checker")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show detailed process information")
    parser.add_argument("--json", "-j", action="store_true", help="Output status as JSON")
    parser.add_argument("--save", "-s", action="store_true", help="Save status report to file")
    parser.add_argument("--watch", "-w", type=int, metavar="SECONDS", help="Watch mode - refresh every N seconds")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode - only show critical issues")
    
    args = parser.parse_args()
    
    checker = SystemStatusChecker()
    
    try:
        if args.watch:
            print("👁️ Watch mode - Press Ctrl+C to exit")
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                if args.json:
                    overview = checker.get_system_overview()
                    print(json.dumps(overview, indent=2))
                else:
                    checker.print_status_report(detailed=args.detailed)
                
                if args.save:
                    checker.save_status_report()
                
                print(f"\n🔄 Refreshing in {args.watch} seconds... (Ctrl+C to exit)")
                time.sleep(args.watch)
        else:
            if args.json:
                overview = checker.get_system_overview()
                print(json.dumps(overview, indent=2))
            else:
                checker.print_status_report(detailed=args.detailed)
            
            if args.save:
                checker.save_status_report()
        
    except KeyboardInterrupt:
        print("\n👋 Status checking stopped")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
