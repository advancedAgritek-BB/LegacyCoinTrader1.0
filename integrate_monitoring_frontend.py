#!/usr/bin/env python3
"""
Comprehensive Monitoring Frontend Integration
Ensures all monitoring components properly tie into the frontend monitoring page.
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import subprocess
import psutil

# Add crypto_bot to path
sys.path.insert(0, str(Path(__file__).parent / "crypto_bot"))

try:
    from crypto_bot.utils.logger import setup_logger
    from crypto_bot.config import load_config
    logger = setup_logger(__name__)
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class MonitoringFrontendIntegration:
    """Comprehensive monitoring integration for frontend dashboard."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "crypto_bot/config.yaml"
        self.config = self._load_config()
        self.running = False
        
        # Monitoring file paths
        self.log_dir = Path(__file__).parent / "crypto_bot" / "logs"
        self.frontend_status_file = self.log_dir / "frontend_monitoring_status.json"
        self.health_status_file = self.log_dir / "health_status.json"
        self.monitoring_report_file = self.log_dir / "monitoring_report.json"
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration."""
        try:
            config = load_config()
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            "monitoring": {
                "enabled": True,
                "check_interval_seconds": 30,
                "alert_interval_seconds": 300,
                "auto_recovery_enabled": True,
                "frontend_integration_enabled": True
            },
            "frontend_monitoring": {
                "update_interval_seconds": 30,
                "max_metrics_history": 1000,
                "enable_real_time_alerts": True,
                "enable_component_status": True
            }
        }
    
    async def start_integration(self) -> None:
        """Start the monitoring frontend integration."""
        logger.info("🚀 Starting Monitoring Frontend Integration")
        print("🚀 Monitoring Frontend Integration")
        print("=" * 50)
        
        try:
            self.running = True
            logger.info("✅ Monitoring frontend integration started")
            
            # Initial status update
            await self._update_frontend_status()
            
            # Main integration loop
            while self.running:
                try:
                    await self._integration_cycle()
                    await asyncio.sleep(self.config.get('frontend_monitoring', {}).get('update_interval_seconds', 30))
                except Exception as e:
                    logger.error(f"Error in integration cycle: {e}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"Failed to start integration: {e}")
            print(f"❌ Failed to start integration: {e}")
            sys.exit(1)
    
    async def _integration_cycle(self) -> None:
        """Perform one complete integration cycle."""
        try:
            # Update frontend monitoring status
            await self._update_frontend_status()
            
            # Update health status
            await self._update_health_status()
            
            # Generate monitoring report
            await self._generate_monitoring_report()
            
            # Display status
            self._display_status()
            
        except Exception as e:
            logger.error(f"Error in integration cycle: {e}")
    
    async def _update_frontend_status(self) -> None:
        """Update frontend monitoring status file."""
        try:
            # Get comprehensive system status
            status_data = {
                "overall_status": "unknown",
                "components": {},
                "recent_metrics": [],
                "alerts_active": [],
                "last_update": datetime.now().isoformat()
            }
            
            # Check evaluation pipeline
            eval_status = await self._check_evaluation_pipeline()
            status_data["components"]["evaluation_pipeline"] = eval_status
            
            # Check execution pipeline
            exec_status = await self._check_execution_pipeline()
            status_data["components"]["execution_pipeline"] = exec_status
            
            # Check WebSocket connections
            ws_status = await self._check_websocket_connections()
            status_data["components"]["websocket_connections"] = ws_status
            
            # Check system resources
            sys_status = await self._check_system_resources()
            status_data["components"]["system_resources"] = sys_status
            
            # Check position monitoring
            pos_status = await self._check_position_monitoring()
            status_data["components"]["position_monitoring"] = pos_status
            
            # Check strategy router
            strat_status = await self._check_strategy_router()
            status_data["components"]["strategy_router"] = strat_status
            
            # Check enhanced scanning
            scan_status = await self._check_enhanced_scanning()
            status_data["components"]["enhanced_scanning"] = scan_status
            
            # Determine overall status
            statuses = [comp["status"] for comp in status_data["components"].values()]
            if "critical" in statuses:
                status_data["overall_status"] = "critical"
            elif "warning" in statuses:
                status_data["overall_status"] = "warning"
            else:
                status_data["overall_status"] = "healthy"
            
            # Generate alerts
            alerts = []
            for component, status in status_data["components"].items():
                if status["status"] == "critical":
                    alerts.append(f"{component}_critical")
                elif status["status"] == "warning":
                    alerts.append(f"{component}_warning")
            
            status_data["alerts_active"] = alerts
            
            # Save to file
            with open(self.frontend_status_file, 'w') as f:
                json.dump(status_data, f, indent=2, default=str)
            
            logger.info(f"Frontend status updated: {status_data['overall_status']}")
            
        except Exception as e:
            logger.error(f"Error updating frontend status: {e}")
    
    async def _update_health_status(self) -> None:
        """Update health status file."""
        try:
            health_data = {}
            
            # Get component health status
            components = [
                "evaluation_pipeline",
                "execution_pipeline", 
                "websocket_connections",
                "system_resources",
                "position_monitoring",
                "strategy_router",
                "enhanced_scanning"
            ]
            
            for component in components:
                if component == "evaluation_pipeline":
                    health_data[component] = await self._check_evaluation_pipeline()
                elif component == "execution_pipeline":
                    health_data[component] = await self._check_execution_pipeline()
                elif component == "websocket_connections":
                    health_data[component] = await self._check_websocket_connections()
                elif component == "system_resources":
                    health_data[component] = await self._check_system_resources()
                elif component == "position_monitoring":
                    health_data[component] = await self._check_position_monitoring()
                elif component == "strategy_router":
                    health_data[component] = await self._check_strategy_router()
                elif component == "enhanced_scanning":
                    health_data[component] = await self._check_enhanced_scanning()
            
            # Save to file
            with open(self.health_status_file, 'w') as f:
                json.dump(health_data, f, indent=2, default=str)
            
            logger.info("Health status updated")
            
        except Exception as e:
            logger.error(f"Error updating health status: {e}")
    
    async def _generate_monitoring_report(self) -> None:
        """Generate comprehensive monitoring report."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "alerts": [],
                "metrics_summary": {},
                "component_status": {},
                "recommendations": []
            }
            
            # Generate alerts based on current status
            try:
                with open(self.frontend_status_file, 'r') as f:
                    status_data = json.load(f)
                
                for component, status in status_data.get("components", {}).items():
                    if status["status"] == "critical":
                        report["alerts"].append(f"Critical: {component} is not functioning properly")
                    elif status["status"] == "warning":
                        report["alerts"].append(f"Warning: {component} has issues requiring attention")
                
                report["component_status"] = status_data.get("components", {})
                
            except Exception as e:
                report["alerts"].append(f"Error: Unable to read status data - {str(e)}")
            
            # Add system recommendations
            try:
                memory = psutil.virtual_memory()
                if memory.percent > 80:
                    report["recommendations"].append("Consider restarting the system to free up memory")
                
                # Check for bot process
                bot_running = False
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    if proc.info['cmdline'] and any('main.py' in cmd for cmd in proc.info['cmdline']):
                        bot_running = True
                        break
                
                if not bot_running:
                    report["recommendations"].append("Trading bot process is not running - consider restarting")
                    
            except Exception as e:
                report["recommendations"].append(f"Unable to generate system recommendations: {str(e)}")
            
            # Save report
            with open(self.monitoring_report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info("Monitoring report generated")
            
        except Exception as e:
            logger.error(f"Error generating monitoring report: {e}")
    
    async def _check_evaluation_pipeline(self) -> Dict[str, Any]:
        """Check evaluation pipeline status."""
        try:
            # Check if bot process is running
            bot_running = False
            recent_evaluations = 0
            
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    if proc.info['cmdline'] and any('main.py' in cmd for cmd in proc.info['cmdline']):
                        bot_running = True
                        break
            except:
                pass
            
            # Check evaluation log for recent activity
            eval_log = self.log_dir / "evaluation_diagnostic.log"
            if eval_log.exists():
                try:
                    with open(eval_log, 'r') as f:
                        content = f.read()
                        recent_evaluations = content.count('evaluating strategy')
                except:
                    pass
            
            status = "healthy" if bot_running else "critical"
            message = "Trading bot process running normally" if bot_running else "Trading bot process not running"
            
            return {
                "status": status,
                "message": message,
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "process_running": bot_running,
                    "recent_evaluations": recent_evaluations
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking evaluation pipeline: {e}")
            return {
                "status": "unknown",
                "message": f"Error checking evaluation pipeline: {str(e)}",
                "last_check": datetime.now().isoformat(),
                "metrics": {}
            }
    
    async def _check_execution_pipeline(self) -> Dict[str, Any]:
        """Check execution pipeline status."""
        try:
            recent_executions = 0
            recent_errors = 0
            pending_orders = 0
            
            # Check execution log
            exec_log = self.log_dir / "execution.log"
            if exec_log.exists():
                try:
                    with open(exec_log, 'r') as f:
                        content = f.read()
                        recent_executions = content.count('order executed')
                        recent_errors = content.count('execution error')
                except:
                    pass
            
            # Check for pending orders in trade manager
            trade_manager_file = self.log_dir / "trade_manager_state.json"
            if trade_manager_file.exists():
                try:
                    with open(trade_manager_file, 'r') as f:
                        data = json.load(f)
                        pending_orders = len(data.get("pending_orders", []))
                except:
                    pass
            
            return {
                "status": "healthy",
                "message": f"Execution pipeline healthy ({recent_executions} executions)",
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "recent_executions": recent_executions,
                    "recent_errors": recent_errors,
                    "pending_orders": pending_orders
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking execution pipeline: {e}")
            return {
                "status": "unknown",
                "message": f"Error checking execution pipeline: {str(e)}",
                "last_check": datetime.now().isoformat(),
                "metrics": {}
            }
    
    async def _check_websocket_connections(self) -> Dict[str, Any]:
        """Check WebSocket connections status."""
        try:
            ws_active = False
            connections = 0
            
            # Check for WebSocket processes
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    if proc.info['cmdline'] and any('websocket' in cmd.lower() for cmd in proc.info['cmdline']):
                        ws_active = True
                        connections += 1
            except:
                pass
            
            status = "healthy" if ws_active else "warning"
            message = f"WebSocket connections active ({connections} connections)" if ws_active else "WebSocket client not active"
            
            return {
                "status": status,
                "message": message,
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "connectivity_ok": True,
                    "ws_active": ws_active,
                    "connections": connections
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking WebSocket connections: {e}")
            return {
                "status": "unknown",
                "message": f"Error checking WebSocket connections: {str(e)}",
                "last_check": datetime.now().isoformat(),
                "metrics": {}
            }
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resources status."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            status = "healthy" if memory.percent < 80 else "warning"
            message = f"Memory: {memory.percent:.1f}%, CPU: {cpu_percent:.1f}%"
            
            return {
                "status": status,
                "message": message,
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "memory_usage_mb": memory.used / 1024 / 1024,
                    "cpu_usage_percent": cpu_percent,
                    "system_memory_percent": memory.percent
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return {
                "status": "unknown",
                "message": f"Error checking system resources: {str(e)}",
                "last_check": datetime.now().isoformat(),
                "metrics": {}
            }
    
    async def _check_position_monitoring(self) -> Dict[str, Any]:
        """Check position monitoring status."""
        try:
            # Check position log for recent activity
            pos_log = self.log_dir / "positions.log"
            recent_updates = 0
            age_seconds = 0
            
            if pos_log.exists():
                try:
                    age_seconds = time.time() - pos_log.stat().st_mtime
                    with open(pos_log, 'r') as f:
                        content = f.read()
                        recent_updates = content.count('position update')
                except:
                    pass
            
            status = "healthy" if age_seconds < 300 else "warning"  # 5 minutes
            message = f"Position monitoring active (last update: {age_seconds:.1f}s ago)"
            
            return {
                "status": status,
                "message": message,
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "age_seconds": age_seconds,
                    "recent_updates": recent_updates
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking position monitoring: {e}")
            return {
                "status": "unknown",
                "message": f"Error checking position monitoring: {str(e)}",
                "last_check": datetime.now().isoformat(),
                "metrics": {}
            }
    
    async def _check_strategy_router(self) -> Dict[str, Any]:
        """Check strategy router status."""
        try:
            recent_routing = 0
            
            # Check strategy router log
            router_log = self.log_dir / "strategy_rank.log"
            if router_log.exists():
                try:
                    with open(router_log, 'r') as f:
                        content = f.read()
                        recent_routing = content.count('strategy routed')
                except:
                    pass
            
            return {
                "status": "healthy",
                "message": f"Strategy router healthy ({recent_routing} recent routings)",
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "recent_routing": recent_routing
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking strategy router: {e}")
            return {
                "status": "unknown",
                "message": f"Error checking strategy router: {str(e)}",
                "last_check": datetime.now().isoformat(),
                "metrics": {}
            }
    
    async def _check_enhanced_scanning(self) -> Dict[str, Any]:
        """Check enhanced scanning status."""
        try:
            scan_active = False
            tokens_scanned = 0
            
            # Check enhanced scanner log
            scan_log = self.log_dir / "enhanced_scanner.log"
            if scan_log.exists():
                try:
                    with open(scan_log, 'r') as f:
                        content = f.read()
                        scan_active = 'scanning started' in content.lower()
                        tokens_scanned = content.count('tokens found')
                except:
                    pass
            
            status = "healthy" if scan_active else "warning"
            message = "Enhanced scanner active" if scan_active else "Enhanced scanner not active"
            
            return {
                "status": status,
                "message": message,
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "scan_active": scan_active,
                    "tokens_scanned": tokens_scanned
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking enhanced scanning: {e}")
            return {
                "status": "unknown",
                "message": f"Error checking enhanced scanning: {str(e)}",
                "last_check": datetime.now().isoformat(),
                "metrics": {}
            }
    
    def _display_status(self) -> None:
        """Display current integration status."""
        try:
            if self.frontend_status_file.exists():
                with open(self.frontend_status_file, 'r') as f:
                    status_data = json.load(f)
                
                overall_status = status_data.get('overall_status', 'unknown')
                components = status_data.get('components', {})
                
                print(f"\n📊 Monitoring Status: {overall_status.upper()}")
                print("-" * 40)
                
                for component, status in components.items():
                    status_icon = "✅" if status['status'] == 'healthy' else "⚠️" if status['status'] == 'warning' else "❌"
                    print(f"{status_icon} {component}: {status['status']} - {status['message']}")
                
                print(f"\n🔄 Last Update: {status_data.get('last_update', 'Unknown')}")
                
        except Exception as e:
            print(f"Error displaying status: {e}")
    
    async def stop_integration(self) -> None:
        """Stop the monitoring frontend integration."""
        logger.info("Stopping monitoring frontend integration")
        self.running = False
        print("\n👋 Monitoring frontend integration stopped")


async def main():
    """Main entry point."""
    print("🚀 Monitoring Frontend Integration")
    print("=" * 50)
    
    try:
        integration = MonitoringFrontendIntegration()
        await integration.start_integration()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
