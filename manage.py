#!/usr/bin/env python3
"""
LegacyCoinTrader Management Script

This script provides a unified interface for managing all aspects of the
LegacyCoinTrader system including startup, shutdown, status checking, and maintenance.
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "crypto_bot"))

class LegacyCoinTraderManager:
    """
    Unified management interface for LegacyCoinTrader.
    """
    
    def __init__(self):
        self.project_root = project_root
        self.python_cmd = self._find_python()
        self.venv_path = self._find_venv()
        
    def _find_python(self) -> str:
        """Find the appropriate Python executable."""
        # Check for virtual environment Python first
        venv_candidates = [
            self.project_root / "venv" / "bin" / "python",
            self.project_root / "modern_trader_env" / "bin" / "python"
        ]
        
        for venv_python in venv_candidates:
            if venv_python.exists():
                return str(venv_python)
        
        # Fall back to system Python
        if subprocess.run(["which", "python3"], capture_output=True).returncode == 0:
            return "python3"
        elif subprocess.run(["which", "python"], capture_output=True).returncode == 0:
            return "python"
        else:
            raise RuntimeError("Python executable not found")
    
    def _find_venv(self) -> Optional[Path]:
        """Find virtual environment path."""
        venv_candidates = [
            self.project_root / "venv",
            self.project_root / "modern_trader_env"
        ]
        
        for venv_path in venv_candidates:
            if venv_path.exists():
                return venv_path
        
        return None
    
    def _run_command(self, command: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Run a command with proper environment setup."""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root / "crypto_bot")
        
        return subprocess.run(
            command,
            cwd=self.project_root,
            env=env,
            **kwargs
        )
    
    def start(self, service: Optional[str] = None, **kwargs) -> bool:
        """Start the system or specific service."""
        print("🚀 Starting LegacyCoinTrader...")
        
        if service:
            return self._start_service(service, **kwargs)
        else:
            return self._start_all_services(**kwargs)
    
    def _start_all_services(self, **kwargs) -> bool:
        """Start all services."""
        script_path = self.project_root / "start_all_services.py"

        try:
            if script_path.exists():
                command = [self.python_cmd, str(script_path)]
            else:
                print("❌ Start script not found. Falling back to basic startup.")
                fallback_script = self.project_root / "start_bot.py"
                if not fallback_script.exists():
                    print(f"❌ Start script not found: {fallback_script}")
                    return False
                command = [self.python_cmd, str(fallback_script), "auto"]

            result = self._run_command(command)
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Failed to start services: {e}")
            return False

    def _start_service(self, service: str, **kwargs) -> bool:
        """Start a specific service."""
        service_scripts = {
            "bot": ["start_bot.py", "auto"],
            "trading": ["start_bot.py", "auto"],
            "frontend": ["frontend/app.py"],
            "web": ["frontend/app.py"],
            "monitor": ["enhanced_monitoring.py"],
            "monitoring": ["enhanced_monitoring.py"],
            "scanner": ["crypto_bot/solana/enhanced_scanner.py"],
            "telegram": ["telegram_ctl.py"],
        }

        script_entry = service_scripts.get(service.lower())
        if not script_entry:
            print(f"❌ Unknown service: {service}")
            print(f"Available services: {', '.join(service_scripts.keys())}")
            return False

        script_path = self.project_root / script_entry[0]
        if not script_path.exists():
            print(f"❌ Service script not found: {script_path}")
            return False

        try:
            print(f"🚀 Starting {service} service...")
            command = [self.python_cmd, str(script_path), *script_entry[1:]]
            result = self._run_command(command)
            return result.returncode == 0
        except Exception as e:
            print(f"❌ Failed to start {service}: {e}")
            return False
    
    def stop(self, force: bool = False, **kwargs) -> bool:
        """Stop the system."""
        print("🛑 Stopping LegacyCoinTrader...")
        
        # Use safe shutdown system
        shutdown_script = self.project_root / "safe_shutdown.py"
        if shutdown_script.exists():
            try:
                cmd = [self.python_cmd, str(shutdown_script)]
                if force:
                    cmd.append("--force")
                
                result = self._run_command(cmd)
                return result.returncode == 0
            except Exception as e:
                print(f"❌ Comprehensive shutdown failed: {e}")
                print("🔄 Falling back to basic shutdown...")
        
        # Fallback to basic shutdown
        return self._basic_stop(force)
    
    def _basic_stop(self, force: bool = False) -> bool:
        """Basic stop functionality."""
        stop_scripts = [
            "stop_integrated.sh",
            "stop_monitoring.sh"
        ]
        
        for script_name in stop_scripts:
            script_path = self.project_root / script_name
            if script_path.exists():
                try:
                    result = subprocess.run([str(script_path)], cwd=self.project_root)
                    if result.returncode == 0:
                        return True
                except Exception as e:
                    print(f"⚠️ {script_name} failed: {e}")
        
        # Last resort: kill processes by pattern
        if force:
            print("💀 Force killing processes...")
            patterns = [
                "crypto_bot.main",
                "frontend.app", 
                "enhanced_monitoring.py",
                "telegram_ctl.py"
            ]
            
            for pattern in patterns:
                try:
                    subprocess.run(["pkill", "-f", pattern], check=False)
                except Exception:
                    pass
        
        return True
    
    def status(self, detailed: bool = False, **kwargs) -> bool:
        """Check system status."""
        print("🔍 Checking LegacyCoinTrader status...")
        
        status_script = self.project_root / "system_status_checker.py"
        if status_script.exists():
            try:
                cmd = [self.python_cmd, str(status_script)]
                if detailed:
                    cmd.append("--detailed")
                
                result = self._run_command(cmd)
                return result.returncode == 0
            except Exception as e:
                print(f"❌ Status check failed: {e}")
        
        # Fallback to basic status check
        return self._basic_status()
    
    def _basic_status(self) -> bool:
        """Basic status check."""
        print("📊 Basic Status Check:")
        print("-" * 30)
        
        # Check for PID files
        pid_files = {
            "bot_pid.txt": "Trading Bot",
            "frontend.pid": "Web Frontend", 
            "monitoring.pid": "Monitoring System",
            "telegram.pid": "Telegram Bot"
        }
        
        running_count = 0
        total_count = len(pid_files)
        
        for pid_file, service_name in pid_files.items():
            pid_path = self.project_root / pid_file
            if pid_path.exists():
                try:
                    with open(pid_path, 'r') as f:
                        pid = int(f.read().strip())
                    
                    # Check if process is running
                    try:
                        os.kill(pid, 0)
                        print(f"🟢 {service_name} (PID: {pid})")
                        running_count += 1
                    except OSError:
                        print(f"🔴 {service_name} (stale PID: {pid})")
                except (ValueError, FileNotFoundError):
                    print(f"🔴 {service_name} (invalid PID file)")
            else:
                print(f"🔴 {service_name} (not running)")
        
        print(f"\n📊 Summary: {running_count}/{total_count} services running")
        return running_count > 0
    
    def restart(self, service: Optional[str] = None, **kwargs) -> bool:
        """Restart the system or specific service."""
        print("🔄 Restarting LegacyCoinTrader...")
        
        # Stop first
        if not self.stop(**kwargs):
            print("⚠️ Stop failed, but continuing with restart...")
        
        # Wait a moment for processes to fully stop
        time.sleep(3)
        
        # Start again
        return self.start(service, **kwargs)
    
    def logs(self, service: Optional[str] = None, follow: bool = False, lines: int = 50) -> bool:
        """Show logs for the system or specific service."""
        log_dir = self.project_root / "logs"
        
        if not log_dir.exists():
            print("❌ Logs directory not found")
            return False
        
        if service:
            log_files = {
                "bot": "bot.log",
                "trading": "bot.log",
                "errors": "errors.log",
                "trades": "trades.csv",
                "positions": "positions.log",
                "monitoring": "monitoring.log",
                "scanner": "scanner.log"
            }
            
            log_file = log_files.get(service.lower())
            if not log_file:
                print(f"❌ Unknown log service: {service}")
                print(f"Available logs: {', '.join(log_files.keys())}")
                return False
            
            log_path = log_dir / log_file
        else:
            # Default to bot log
            log_path = log_dir / "bot.log"
        
        if not log_path.exists():
            print(f"❌ Log file not found: {log_path}")
            return False
        
        try:
            if follow:
                # Use tail -f equivalent
                subprocess.run(["tail", "-f", str(log_path)])
            else:
                # Show last N lines
                subprocess.run(["tail", "-n", str(lines), str(log_path)])
            return True
        except Exception as e:
            print(f"❌ Failed to show logs: {e}")
            return False
    
    def health(self, **kwargs) -> bool:
        """Perform health check."""
        print("🏥 Performing health check...")
        
        # Check if status checker is available
        status_script = self.project_root / "system_status_checker.py"
        if status_script.exists():
            try:
                result = self._run_command([
                    self.python_cmd, 
                    str(status_script),
                    "--json"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    status_data = json.loads(result.stdout)
                    system_health = status_data.get('system_health', 'unknown')
                    
                    if system_health == 'healthy':
                        print("✅ System is healthy")
                        return True
                    elif system_health == 'warning':
                        print("⚠️ System has warnings")
                        return True
                    else:
                        print("❌ System has errors")
                        return False
                
            except Exception as e:
                print(f"❌ Health check failed: {e}")
        
        # Fallback to basic health check
        return self._basic_health()
    
    def _basic_health(self) -> bool:
        """Basic health check."""
        # Check critical files
        critical_files = [
            "crypto_bot/main.py",
            "config.yaml",
            "requirements.txt"
        ]
        
        missing_files = []
        for file_path in critical_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ Missing critical files: {', '.join(missing_files)}")
            return False
        
        # Check if any processes are running
        if self._basic_status():
            print("✅ Basic health check passed")
            return True
        else:
            print("⚠️ No processes running")
            return False
    
    def update(self, **kwargs) -> bool:
        """Update the system (placeholder for future implementation)."""
        print("🔄 System update functionality not yet implemented")
        print("Please update manually using git pull and pip install -r requirements.txt")
        return True
    
    def backup(self, **kwargs) -> bool:
        """Create system backup."""
        print("💾 Creating system backup...")
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.project_root / "backups" / f"backup_{timestamp}"
        
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup critical directories and files
            backup_items = [
                "logs",
                "config",
                "config.yaml",
                "crypto_bot/config.yaml"
            ]
            
            import shutil
            for item in backup_items:
                source = self.project_root / item
                if source.exists():
                    if source.is_file():
                        shutil.copy2(source, backup_dir / item)
                    else:
                        shutil.copytree(source, backup_dir / item, dirs_exist_ok=True)
            
            print(f"✅ Backup created: {backup_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LegacyCoinTrader Management Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                 # Start all services
  %(prog)s start --service bot   # Start only trading bot
  %(prog)s stop                  # Stop all services
  %(prog)s stop --force          # Force stop all services
  %(prog)s status                # Check system status
  %(prog)s status --detailed     # Detailed status check
  %(prog)s restart               # Restart all services
  %(prog)s logs                  # Show recent logs
  %(prog)s logs --follow         # Follow logs in real-time
  %(prog)s health                # Perform health check
  %(prog)s backup                # Create system backup
        """
    )
    
    parser.add_argument("command", 
                       choices=["start", "stop", "restart", "status", "logs", "health", "update", "backup"],
                       help="Command to execute")
    
    parser.add_argument("--service", "-s", 
                       help="Specific service to operate on")
    
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force operation (for stop command)")
    
    parser.add_argument("--detailed", "-d", action="store_true",
                       help="Show detailed information (for status command)")
    
    parser.add_argument("--follow", action="store_true",
                       help="Follow logs in real-time (for logs command)")
    
    parser.add_argument("--lines", "-n", type=int, default=50,
                       help="Number of log lines to show (for logs command)")
    
    args = parser.parse_args()
    
    try:
        manager = LegacyCoinTraderManager()
        
        # Execute command
        command_kwargs = {
            "service": args.service,
            "force": args.force,
            "detailed": args.detailed,
            "follow": args.follow,
            "lines": args.lines
        }
        
        if args.command == "start":
            success = manager.start(**command_kwargs)
        elif args.command == "stop":
            success = manager.stop(**command_kwargs)
        elif args.command == "restart":
            success = manager.restart(**command_kwargs)
        elif args.command == "status":
            success = manager.status(**command_kwargs)
        elif args.command == "logs":
            success = manager.logs(**command_kwargs)
        elif args.command == "health":
            success = manager.health(**command_kwargs)
        elif args.command == "update":
            success = manager.update(**command_kwargs)
        elif args.command == "backup":
            success = manager.backup(**command_kwargs)
        else:
            print(f"❌ Unknown command: {args.command}")
            success = False
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Operation interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
