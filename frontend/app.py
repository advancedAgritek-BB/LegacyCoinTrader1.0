"""Start the web dashboard and expose REST endpoints for the trading bot.

This module launches the Flask web server, manages the background trading
process and provides REST API routes used by the UI and tests.
"""

import os
import signal
import sys
import warnings
import subprocess
import json
import time
import yaml
import logging
from pathlib import Path
from decimal import Decimal
from datetime import datetime
from typing import Optional, Dict, Any
from crypto_bot import log_reader
from crypto_bot import ml_signal_model as ml
import frontend.utils as utils
from crypto_bot.utils.trade_manager import is_test_position
from frontend.config import get_config
from frontend.auth import get_auth, login_required
from crypto_bot.utils.price_fetcher import (
    get_current_price_for_symbol as _get_current_price_for_symbol,
)

# Suppress urllib3 OpenSSL warning
warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+",
    category=UserWarning,
)

try:
    from dotenv import dotenv_values
except ImportError:
    dotenv_values = None

try:
    from flask import (
        Flask,
        render_template,
        redirect,
        url_for,
        request,
        jsonify,
    )
except (
    Exception
):  # pragma: no cover - provide minimal shim for import-time tests

    class _Dummy:
        def __getattr__(self, _):
            return self

        def __call__(self, *a, **k):
            return None

    def _dummy_flask(*a, **k):
        return _Dummy()

    Flask = _dummy_flask  # type: ignore
    render_template = redirect = url_for = request = jsonify = _Dummy()

# Fix LOG_DIR path to point to the correct crypto_bot/logs directory
LOG_DIR = Path(__file__).resolve().parents[1] / "crypto_bot" / "logs"

logger = logging.getLogger(__name__)

app = Flask(__name__)
# Lightweight healthcheck for container orchestration
@app.route("/health", methods=["GET"])  # Simple 200 OK health endpoint
def health():
    return (json.dumps({"status": "ok"}), 200, {"Content-Type": "application/json"})

# Import secure configuration and authentication, login_required

# Get configuration and authentication instances
config = get_config()
auth = get_auth()

# Disable template caching for development - CRITICAL for template updates
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Set secure session configuration
app.config["SECRET_KEY"] = config.security.session_secret_key
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = config.security.session_timeout
app.config["SESSION_COOKIE_SECURE"] = config.environment == "production"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Rate limiting (simple in-memory implementation)
request_counts = {}
request_windows = {}


def is_rate_limited():
    """Check if current request exceeds rate limit."""
    from flask import request

    client_ip = request.remote_addr
    current_time = time.time()

    # Clean old entries
    if client_ip in request_windows:
        window_time = config.security.rate_limit_window
        if current_time - request_windows[client_ip] > window_time:
            request_counts[client_ip] = 0
            request_windows[client_ip] = current_time

    # Check rate limit
    if client_ip not in request_counts:
        request_counts[client_ip] = 0
        request_windows[client_ip] = current_time

    request_counts[client_ip] += 1

    return request_counts[client_ip] > config.security.rate_limit_requests


@app.before_request
def check_rate_limit():
    """Check rate limit before processing request."""
    if is_rate_limited():
        from flask import jsonify

        return jsonify({"error": "Rate limit exceeded"}), 429


# Secure headers middleware
@app.after_request
def add_secure_headers(response):
    """Add secure headers to all responses."""
    from flask import request

    # Get the origin from the request
    origin = request.headers.get("Origin")

    # Generate secure CSP header
    response.headers["Content-Security-Policy"] = (
        config.security.get_csp_header()
    )

    # Generate secure CORS headers
    if origin:
        cors_headers = config.security.get_cors_headers(origin)
        response.headers.update(cors_headers)

    # Additional security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # HSTS header for production
    if config.environment == "production":
        hsts_value = "max-age=31536000; includeSubDomains"
        response.headers["Strict-Transport-Security"] = hsts_value
    else:
        response.headers["Strict-Transport-Security"] = ""

    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    # Cache control for security
    if config.environment == "development":
        response.headers["Cache-Control"] = (
            "no-cache, no-store, must-revalidate"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    else:
        # 5 minutes for production
        response.headers["Cache-Control"] = "public, max-age=300"

    return response


# Authentication Routes
@app.route("/login", methods=["GET", "POST"])
def login():
    """Handle user login."""
    from flask import request, session, flash, redirect, url_for

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = auth.authenticate(username, password)
        if user:
            session["user"] = user
            session["login_time"] = user["login_time"]
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials", "error")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    """Handle user logout."""
    from flask import session, flash, redirect, url_for

    session.clear()
    flash("Logged out successfully", "info")
    return redirect(url_for("login"))


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    """API endpoint for login."""
    from flask import request, session, jsonify

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    user = auth.authenticate(username, password)
    if user:
        session["user"] = user
        session["login_time"] = user["login_time"]
        return jsonify(
            {
                "message": "Login successful",
                "user": {"username": user["username"], "role": user["role"]},
            }
        )
    else:
        return jsonify({"error": "Invalid credentials"}), 401


@app.route("/api/auth/logout", methods=["POST"])
@login_required
def api_logout():
    """API endpoint for logout."""
    from flask import session, jsonify

    session.clear()
    return jsonify({"message": "Logged out successfully"})


@app.route("/api/auth/status")
def auth_status():
    """Get current authentication status."""
    from flask import session, jsonify

    if "user" in session:
        user = session["user"]
        return jsonify(
            {
                "authenticated": True,
                "user": {"username": user["username"], "role": user["role"]},
            }
        )
    else:
        return jsonify({"authenticated": False})


# Add monitoring API routes first to avoid conflicts with general CORS handler
@app.route("/api/monitoring/health", methods=["GET"])
def api_monitoring_health():
    """Return comprehensive system health status."""
    try:
        import json
        from pathlib import Path

        # Try to read the actual monitoring data first
        frontend_status_file = LOG_DIR / "frontend_monitoring_status.json"
        if frontend_status_file.exists():
            try:
                with open(frontend_status_file, "r") as f:
                    monitoring_data = json.load(f)

                # Add current timestamp and return
                monitoring_data["last_update"] = datetime.now().isoformat()

                return jsonify(
                    {
                        "success": True,
                        "data": monitoring_data,
                        "timestamp": int(time.time() * 1000),
                    }
                )
            except Exception as e:
                print(f"Error reading frontend monitoring status: {e}")

        # Fallback: Get system metrics if monitoring data not available
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Check if bot is running
        bot_running = False
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"]
                if cmdline and len(cmdline) > 0:
                    cmd_str = " ".join(cmdline).lower()
                    if any(
                        pattern in cmd_str
                        for pattern in [
                            "start_bot_auto",
                            "crypto_bot.main",
                            "crypto_bot/main.py",
                        ]
                    ):
                        bot_running = True
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Determine overall system status
        overall_status = "healthy"
        if cpu_percent > 80 or memory.percent > 80:
            overall_status = "warning"
        if cpu_percent > 95 or memory.percent > 95:
            overall_status = "critical"

        # Get component status
        components = {
            "evaluation_pipeline": {
                "status": "healthy" if bot_running else "critical",
                "message": "Trading bot active" if bot_running else "Trading bot not running",
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "process_running": bot_running,
                    "recent_evaluations": 0
                },
            },
            "execution_pipeline": {
                "status": "healthy",
                "message": "Order execution pipeline active",
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "recent_executions": 0,
                    "recent_errors": 0,
                    "pending_orders": 0
                },
            },
            "system_resources": {
                "status": overall_status,
                "message": f"System resources: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%",
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "memory_usage_mb": memory.used / 1024 / 1024,
                    "cpu_usage_percent": cpu_percent,
                    "system_memory_percent": memory.percent
                },
            },
            "monitoring_system": {
                "status": "unknown",
                "message": "Monitoring system status unknown",
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "monitoring_running": False,
                    "health_check_running": False,
                },
            },
            "websocket_connections": {
                "status": "unknown",
                "message": "WebSocket connection status unknown",
                "last_check": datetime.now().isoformat(),
                "metrics": {
                    "connectivity_ok": False,
                    "ws_active": False,
                    "connections": 0
                },
            },
            "strategy_router": {
                "status": "unknown",
                "message": "Strategy router status unknown",
                "last_check": datetime.now().isoformat(),
                "metrics": {"recent_routing": 0},
            },
            "enhanced_scanner": {
                "status": "unknown",
                "message": "Enhanced scanner status unknown",
                "last_check": datetime.now().isoformat(),
                "metrics": {"tokens_scanned": 0, "scanner_active": False},
            },
            "position_monitoring": {
                "status": "unknown",
                "message": "Position monitoring status unknown",
                "last_check": datetime.now().isoformat(),
                "metrics": {"age_seconds": 0, "recent_updates": 0},
            },
        }

        # Try to get recent metrics from monitoring files and log analysis
        try:
            # Count recent log entries to estimate activity
            import glob
            import re

            # Count recent log entries across various log files
            log_files = [
                LOG_DIR / "bot_*.log",
                LOG_DIR / "wallet.log",
                LOG_DIR / "telemetry.log",
            ]

            total_entries = 0
            recent_entries = 0
            current_time = time.time()

            for pattern in log_files:
                for log_file in glob.glob(str(pattern)):
                    try:
                        with open(log_file, "r") as f:
                            lines = f.readlines()
                            total_entries += len(lines)

                            # Count recent entries (last hour)
                            for line in lines:
                                # Extract timestamp from log line
                                timestamp_match = re.search(
                                    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})",
                                    line,
                                )
                                if timestamp_match:
                                    try:
                                        log_time = datetime.strptime(
                                            timestamp_match.group(1),
                                            "%Y-%m-%d %H:%M:%S",
                                        )
                                        log_timestamp = log_time.timestamp()
                                        if (
                                            current_time - log_timestamp < 3600
                                        ):  # Last hour
                                            recent_entries += 1
                                    except (ValueError, AttributeError):
                                        # Skip malformed timestamps
                                        pass
                    except (OSError, IOError):
                        # Skip files that can't be read
                        pass

            # Estimate activity based on log entries
            if recent_entries > 0:
                components["evaluation_pipeline"]["metrics"].update(
                    {
                        "evaluation_count": recent_entries
                        // 10,  # Estimate evaluations
                        "successful_evaluations": recent_entries
                        // 15,  # Estimate successful
                        "failed_evaluations": recent_entries
                        // 50,  # Estimate failed
                    }
                )
                components["execution_pipeline"]["metrics"].update(
                    {
                        "execution_count": recent_entries
                        // 20,  # Estimate executions
                        "pending_orders": 0,  # Default to 0
                        "success_rate": 95.0,  # Default success rate
                    }
                )
        except Exception as e:
            print(f"Error analyzing log files: {e}")

        # Check WebSocket monitoring status
        try:
            ws_monitoring_file = LOG_DIR / "websocket_monitoring.json"
            if ws_monitoring_file.exists():
                with open(ws_monitoring_file, "r") as f:
                    ws_data = json.load(f)

                ws_active = ws_data.get("websocket_active", False)
                ws_status = ws_data.get("current_status", "unknown")
                error_msg = ws_data.get("error_message", "")

                if ws_active:
                    components["websocket_connections"]["status"] = "healthy"
                    components["websocket_connections"][
                        "message"
                    ] = "WebSocket connection active"
                    components["websocket_connections"]["metrics"][
                        "ws_active"
                    ] = True
                    components["websocket_connections"]["metrics"][
                        "connectivity_ok"
                    ] = True
                    components["websocket_connections"]["metrics"][
                        "connections"
                    ] = 1
                elif ws_status == "connection_failed":
                    components["websocket_connections"]["status"] = "warning"
                    components["websocket_connections"][
                        "message"
                    ] = f"WebSocket connection failed: {error_msg}"
                    components["websocket_connections"]["metrics"][
                        "connectivity_ok"
                    ] = False
                else:
                    components["websocket_connections"]["status"] = "warning"
                    components["websocket_connections"][
                        "message"
                    ] = f"WebSocket status: {ws_status}"
        except Exception as e:
            print(f"Error reading WebSocket monitoring file: {e}")

        # Check strategy routing status
        try:
            routing_stats_file = LOG_DIR / "strategy_routing_stats.json"
            if routing_stats_file.exists():
                with open(routing_stats_file, "r") as f:
                    routing_data = json.load(f)

                recent_routing = len(
                    routing_data.get("recent_routing_activity", [])
                )
                last_routing_time = routing_data.get("last_routing_time")

                if (
                    last_routing_time and time.time() - last_routing_time < 300
                ):  # Within last 5 minutes
                    components["strategy_router"]["status"] = "healthy"
                    components["strategy_router"][
                        "message"
                    ] = f"Strategy routing active ({recent_routing} recent activities)"
                    components["strategy_router"]["metrics"][
                        "recent_routing"
                    ] = recent_routing
                elif recent_routing > 0:
                    components["strategy_router"]["status"] = "warning"
                    components["strategy_router"][
                        "message"
                    ] = f"Strategy routing has {recent_routing} activities but not recent"
                    components["strategy_router"]["metrics"][
                        "recent_routing"
                    ] = recent_routing
                else:
                    components["strategy_router"]["status"] = "warning"
                    components["strategy_router"][
                        "message"
                    ] = "No recent strategy routing activity"
                    components["strategy_router"]["metrics"][
                        "recent_routing"
                    ] = 0
        except Exception as e:
            print(f"Error reading strategy routing stats: {e}")

        # Check enhanced scanner status
        try:
            scanner_status_file = LOG_DIR / "enhanced_scanner_status.json"
            if scanner_status_file.exists():
                with open(scanner_status_file, "r") as f:
                    scanner_data = json.load(f)

                tokens_scanned = scanner_data.get("tokens_scanned", 0)
                last_scan_time = scanner_data.get("last_scan_time")

                if (
                    last_scan_time and time.time() - last_scan_time < 300
                ):  # Within last 5 minutes
                    components["enhanced_scanner"]["status"] = "healthy"
                    components["enhanced_scanner"][
                        "message"
                    ] = f"Enhanced scanner active ({tokens_scanned} tokens scanned)"
                    components["enhanced_scanner"]["metrics"][
                        "tokens_scanned"
                    ] = tokens_scanned
                    components["enhanced_scanner"]["metrics"][
                        "scanner_active"
                    ] = True
                elif tokens_scanned > 0:
                    components["enhanced_scanner"]["status"] = "warning"
                    components["enhanced_scanner"][
                        "message"
                    ] = f"Enhanced scanner has scanned {tokens_scanned} tokens but not recently"
                    components["enhanced_scanner"]["metrics"][
                        "tokens_scanned"
                    ] = tokens_scanned
                    components["enhanced_scanner"]["metrics"][
                        "scanner_active"
                    ] = False
                else:
                    components["enhanced_scanner"]["status"] = "warning"
                    components["enhanced_scanner"][
                        "message"
                    ] = "Enhanced scanner not active"
                    components["enhanced_scanner"]["metrics"][
                        "tokens_scanned"
                    ] = 0
                    components["enhanced_scanner"]["metrics"][
                        "scanner_active"
                    ] = False
        except Exception as e:
            print(f"Error reading enhanced scanner status: {e}")

        return jsonify(
            {
                "success": True,
                "overall_status": overall_status,
                "components": components,
                "system_metrics": {
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory.used / (1024 * 1024),
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                },
                "recent_metrics": [
                    {
                        "timestamp": int(time.time() * 1000),
                        "evaluation_count": components["evaluation_pipeline"][
                            "metrics"
                        ]["evaluation_count"],
                        "execution_count": components["execution_pipeline"][
                            "metrics"
                        ]["execution_count"],
                        "memory_mb": memory.used / (1024 * 1024),
                        "cpu_percent": cpu_percent,
                        "errors": components["evaluation_pipeline"]["metrics"][
                            "failed_evaluations"
                        ],
                        "websocket_connections": (
                            1
                            if components["websocket_connections"]["metrics"][
                                "ws_active"
                            ]
                            else 0
                        ),
                        "api_calls": (
                            recent_entries // 5
                            if "recent_entries" in locals()
                            else 0
                        ),
                    }
                ],
                "timestamp": int(time.time() * 1000),
            }
        )

    except Exception as e:
        print(f"Error in api_monitoring_health: {e}")
        return jsonify(
            {
                "success": False,
                "error": str(e),
                "overall_status": "unknown",
                "components": {},
                "timestamp": int(time.time() * 1000),
            }
        )


@app.route("/api/monitoring/metrics", methods=["GET"])
def api_monitoring_metrics():
    """Return performance metrics and statistics."""
    try:
        import json
        from pathlib import Path

        # Try to read from frontend monitoring status first (contains recent_metrics)
        frontend_status_file = LOG_DIR / "frontend_monitoring_status.json"
        metrics_data = {
            "recent_metrics": [],
            "scan_metrics": {
                "tokens_scanned": 0,
                "execution_opportunities": 0,
                "scan_cache_hits": 0,
            },
            "evaluation_metrics": {
                "strategy_evaluations": 0,
                "successful_evaluations": 0,
                "failed_evaluations": 0,
            },
            "alerts_active": [],
        }

        if frontend_status_file.exists():
            try:
                with open(frontend_status_file, "r") as f:
                    status_data = json.load(f)
                    # Extract recent metrics from the monitoring data
                    metrics_data["recent_metrics"] = status_data.get("recent_metrics", [])

                    # Extract component metrics for evaluation counts
                    components = status_data.get("components", {})
                    if "evaluation_pipeline" in components:
                        eval_metrics = components["evaluation_pipeline"].get("metrics", {})
                        metrics_data["evaluation_metrics"]["strategy_evaluations"] = eval_metrics.get("recent_evaluations", 0)

                    if "enhanced_scanner" in components:
                        scanner_metrics = components["enhanced_scanner"].get("metrics", {})
                        metrics_data["scan_metrics"]["tokens_scanned"] = scanner_metrics.get("tokens_scanned", 0)
            except Exception as e:
                print(f"Error reading frontend monitoring status for metrics: {e}")

        # Try to load additional metrics from other files as fallback/supplement
        try:
            metrics_file = LOG_DIR / "monitoring_metrics.json"
            if metrics_file.exists():
                file_data = json.loads(metrics_file.read_text())
                metrics_data.update(file_data)
        except Exception as e:
            print(f"Error reading monitoring metrics file: {e}")

        return jsonify(
            {
                "success": True,
                "data": metrics_data,
                "timestamp": int(time.time() * 1000),
            }
        )

    except Exception as e:
        print(f"Error in api_monitoring_metrics: {e}")
        return jsonify(
            {
                "success": False,
                "error": str(e),
                "data": {},
                "timestamp": int(time.time() * 1000),
            }
        )


@app.route("/api/monitoring/logs", methods=["GET"])
def api_monitoring_logs():
    """Return monitoring logs from various system components."""
    try:
        logs_data = {}

        # Define log files to read
        log_files = {
            "pipeline_monitor": LOG_DIR / "pipeline_monitor.log",
            "health_check": LOG_DIR / "health_check.log",
            "recovery_actions": LOG_DIR
            / "health_check.log",  # Recovery actions logged in health_check.log
            "monitoring_status": LOG_DIR
            / "pipeline_monitor.log",  # Monitoring status in pipeline_monitor.log
        }

        for log_type, log_file in log_files.items():
            if log_file.exists():
                try:
                    # Read last 50 lines of each log file
                    lines = log_file.read_text().splitlines()[-50:]
                    logs_data[log_type] = lines
                except Exception as e:
                    logs_data[log_type] = [f"Error reading log file: {e}"]
            else:
                logs_data[log_type] = ["Log file not found"]

        return jsonify(
            {
                "success": True,
                "data": logs_data,
                "timestamp": int(time.time() * 1000),
            }
        )

    except Exception as e:
        print(f"Error in api_monitoring_logs: {e}")
        return jsonify(
            {
                "success": False,
                "error": str(e),
                "data": {},
                "timestamp": int(time.time() * 1000),
            }
        )


@app.route("/api/monitoring/status", methods=["GET"])
def api_monitoring_status():
    """Return monitoring system status and component health."""
    try:
        status_data = {
            "monitoring_running": False,
            "health_check_running": False,
            "frontend_running": True,  # Frontend is always running if this endpoint is called
            "timestamp": int(time.time() * 1000),
        }

        # Check if monitoring processes are running
        import psutil

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info["cmdline"]
                if cmdline and len(cmdline) > 0:
                    cmd_str = " ".join(cmdline).lower()
                    # Check for various bot startup patterns
                    if any(
                        pattern in cmd_str
                        for pattern in [
                            "pipeline_monitor",
                            "enhanced_monitoring",
                            "start_bot_auto",
                            "crypto_bot.main",
                            "crypto_bot/main.py",
                        ]
                    ):
                        status_data["monitoring_running"] = True
                    if any(
                        pattern in cmd_str
                        for pattern in [
                            "health_check",
                            "auto_health_check",
                            "pipeline_monitor",
                        ]
                    ):
                        status_data["health_check_running"] = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Check for recent monitoring activity
        try:
            health_status_file = LOG_DIR / "health_status.json"
            if health_status_file.exists():
                health_data = json.loads(health_status_file.read_text())
                status_data["last_health_check"] = health_data.get("timestamp")
                status_data["health_status"] = health_data
        except Exception as e:
            print(f"Error reading health status: {e}")

        return jsonify(
            {
                "success": True,
                "data": status_data,
                "timestamp": int(time.time() * 1000),
            }
        )

    except Exception as e:
        print(f"Error in api_monitoring_status: {e}")
        return jsonify(
            {
                "success": False,
                "error": str(e),
                "data": {},
                "timestamp": int(time.time() * 1000),
            }
        )


@app.route('/api/sync-health')
def get_sync_health():
    """Get synchronization health status."""
    try:
        from crypto_bot.sync_service import SyncService
        from pathlib import Path

        # Get the log directory
        log_dir = Path(__file__).parent.parent / "crypto_bot" / "logs"
        sync_service = SyncService(log_dir)

        health = sync_service.get_health_status()
        return jsonify({
            'success': True,
            'data': health
        })
    except Exception as e:
        logger.error(f"Error getting sync health: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/sync-positions', methods=['POST'])
def trigger_sync():
    """Manually trigger position synchronization."""
    try:
        from crypto_bot.sync_service import SyncService, ConflictResolution
        from pathlib import Path
        import asyncio

        # Get the log directory
        log_dir = Path(__file__).parent.parent / "crypto_bot" / "logs"
        sync_service = SyncService(log_dir)

        # For now, we'll return a message that manual sync should be done via bot restart
        return jsonify({
            'success': True,
            'message': 'Synchronization is automatically handled during bot startup. Restart the bot to trigger synchronization.',
            'data': {
                'next_steps': [
                    'Stop the current bot process',
                    'Restart the bot to trigger automatic synchronization',
                    'Check /api/sync-health for synchronization status'
                ]
            }
        })

    except Exception as e:
        logger.error(f"Error triggering sync: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/sync-report/<operation_id>')
def get_sync_report(operation_id):
    """Get detailed report for a specific synchronization operation."""
    try:
        from crypto_bot.sync_service import SyncService
        from pathlib import Path

        # Get the log directory
        log_dir = Path(__file__).parent.parent / "crypto_bot" / "logs"
        sync_service = SyncService(log_dir)

        report = sync_service.get_sync_report(operation_id)

        if 'error' in report:
            return jsonify({
                'success': False,
                'error': report['error']
            })

        return jsonify({
            'success': True,
            'data': report
        })

    except Exception as e:
        logger.error(f"Error getting sync report: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


# CORS preflight is now handled by the secure headers middleware above

# Handle the async trading bot process
bot_proc = None
bot_start_time = None
watch_thread = None

# Global controller instance
CONTROLLER = None


def get_controller():
    """Get or create the TradingBotController instance."""
    global CONTROLLER
    if CONTROLLER is None:
        from crypto_bot.bot_controller import TradingBotController

        CONTROLLER = TradingBotController()
    return CONTROLLER


# Context processor to make bot status available to all templates
@app.context_processor
def inject_bot_status():
    return {
        "running": is_running(),
        "mode": load_execution_mode(),
        "uptime": get_uptime(),
    }


LOG_FILE = LOG_DIR / "bot.log"
STATS_FILE = LOG_DIR / "strategy_stats.json"
SCAN_FILE = LOG_DIR / "asset_scores.json"
MODEL_REPORT = Path("crypto_bot/ml_signal_model/models/model_report.json")
TRADE_FILE = LOG_DIR / "trades.csv"
ERROR_FILE = LOG_DIR / "errors.log"
CONFIG_FILE = Path("crypto_bot/config.yaml")
REGIME_FILE = LOG_DIR / "regime_history.txt"
POSITIONS_FILE = LOG_DIR / "positions.log"

# Define project root for use in various functions
project_root = Path(__file__).parent.parent

# Environment variables will be loaded in the main block


def stop_conflicting_bots() -> None:
    """Stop any other bot processes that might be running to prevent conflicts."""
    try:
        import psutil

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            if proc.info["cmdline"] and "crypto_bot.main" in " ".join(
                proc.info["cmdline"]
            ):
                if proc.info["pid"] != os.getpid():  # Don't kill ourselves
                    print(
                        f"Stopping conflicting bot process (PID {proc.info['pid']})"
                    )
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        proc.kill()
        time.sleep(2)  # Give processes time to terminate
    except ImportError:
        pass


def check_existing_bot() -> bool:
    """Check if there's already a bot process running to prevent conflicts."""
    try:
        import psutil

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            if proc.info["cmdline"]:
                cmdline_str = " ".join(proc.info["cmdline"])
                # Check for various bot startup patterns
                if any(
                    pattern in cmdline_str
                    for pattern in [
                        "crypto_bot.main",
                        "crypto_bot/main.py",
                        "start_bot_noninteractive.py",
                        "start_bot_auto.py",
                    ]
                ):
                    return True
        return False
    except ImportError:
        # psutil not available, use basic check
        return False


def watch_bot() -> None:
    """Monitor the trading bot and restart it if the process exits."""
    global bot_proc, bot_start_time
    while True:
        time.sleep(5)
        if bot_proc is not None and bot_proc.poll() is not None:
            # Check if there's already another bot process running to avoid conflicts
            if not check_existing_bot():
                print("Bot process exited, restarting...")
                venv_python = (
                    Path(__file__).parent.parent / "venv" / "bin" / "python3"
                )
                bot_script = (
                    Path(__file__).parent.parent
                    / "start_bot_noninteractive.py"
                )
                bot_proc = subprocess.Popen(
                    [str(venv_python), str(bot_script)]
                )
                bot_start_time = time.time()
            else:
                print(
                    "Another bot process detected, skipping restart to avoid conflicts"
                )
                bot_proc = None


def is_running() -> bool:
    """Return True if the bot process is running."""
    # Check if we have a tracked subprocess
    if bot_proc and bot_proc.poll() is None:
        return True

    # Also check for existing bot processes
    return check_existing_bot()


def set_execution_mode(mode: str) -> None:
    """Set execution mode in config file."""
    utils.set_execution_mode(mode, CONFIG_FILE)


def load_execution_mode() -> str:
    """Load execution mode from config file."""
    return utils.load_execution_mode(CONFIG_FILE)


def calculate_wallet_balance_from_trade_manager() -> float:
    """Calculate wallet balance from TradeManager (source of truth)."""
    try:
        from crypto_bot.utils.trade_manager import get_trade_manager

        trade_manager = get_trade_manager()

        # Get total realized P&L from TradeManager
        realized_pnl = float(trade_manager.total_realized_pnl)

        # Calculate unrealized P&L from open positions
        unrealized_pnl = 0.0
        positions = trade_manager.get_all_positions()

        for pos in positions:
            if pos.is_open:
                # Get current price from TradeManager's cache
                current_price = float(
                    trade_manager.price_cache.get(
                        pos.symbol, pos.average_price
                    )
                )

                # Calculate unrealized P&L
                pnl, _ = pos.calculate_unrealized_pnl(
                    Decimal(str(current_price))
                )
                unrealized_pnl += float(pnl)

        total_pnl = realized_pnl + unrealized_pnl
        wallet_balance = 10000.0 + total_pnl
        logger.info(
            f"TradeManager-based calculation: realized=${realized_pnl:.2f}, unrealized=${unrealized_pnl:.2f}, total=${total_pnl:.2f}, balance=${wallet_balance:.2f}"
        )
        return wallet_balance

    except Exception as e:
        logger.error(
            f"Error calculating wallet balance from TradeManager: {e}"
        )
        return 10000.0


def calculate_wallet_balance_from_csv() -> float:
    """DEPRECATED: Legacy CSV-based balance calculation - kept for backward compatibility."""
    logger.warning(
        "Using deprecated CSV-based balance calculation. TradeManager should be used instead."
    )
    try:
        df = log_reader._read_trades(TRADE_FILE)
        if df.empty:
            return 10000.0

        # Calculate realized P&L from closed trades
        closed_trades = df[df["status"] == "closed"]
        realized_pnl = (
            closed_trades["pnl"].sum()
            if "pnl" in closed_trades.columns
            else 0.0
        )

        # Calculate unrealized P&L from open positions
        open_trades = df[df["status"] == "open"]
        unrealized_pnl = 0.0

        for _, trade in open_trades.iterrows():
            try:
                # Get current price for the symbol
                symbol = trade["symbol"]
                current_price = get_current_price(symbol)
                entry_price = trade["price"]
                amount = trade["amount"]
                side = trade["side"]

                if current_price and entry_price:
                    if side == "long":
                        pnl = (current_price - entry_price) * amount
                    else:  # short
                        pnl = (entry_price - current_price) * amount
                    unrealized_pnl += pnl
            except Exception as e:
                logger.warning(
                    f"Error calculating unrealized P&L for {symbol}: {e}"
                )
                continue

        total_pnl = realized_pnl + unrealized_pnl
        wallet_balance = 10000.0 + total_pnl
        logger.info(
            f"CSV-based calculation: realized=${realized_pnl:.2f}, unrealized=${unrealized_pnl:.2f}, total=${total_pnl:.2f}, balance=${wallet_balance:.2f}"
        )
        return wallet_balance

    except Exception as e:
        logger.error(f"Error calculating wallet balance from CSV: {e}")
        return 10000.0


def get_current_price(symbol: str) -> float:
    """Get current price for a symbol from various sources."""
    try:
        # Use the existing get_current_price_for_symbol function instead of missing price_manager
        return get_current_price_for_symbol(symbol)
    except Exception:
        # Fallback to basic price fetching
        try:
            import requests

            # Simple price fetching for common symbols
            if symbol == "BTC/USD":
                response = requests.get(
                    "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
                    timeout=5,
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("bitcoin", {}).get("usd", 0)
        except Exception:
            pass
        return 0.0


def get_paper_wallet_balance() -> float:
    """Get paper wallet balance from the single source of truth."""
    try:
        from crypto_bot.utils.balance_manager import get_single_balance

        balance = get_single_balance()
        print(
            f"Frontend got balance from single source of truth: ${balance:.2f}"
        )
        return balance
    except Exception as e:
        print(f"Error getting balance from single source: {e}")
        return 10000.0


def get_available_balance(open_positions: list) -> float:
    """Calculate available balance (wallet balance minus value of open positions)."""
    try:
        # Get current wallet balance
        total_balance = get_paper_wallet_balance()

        # Calculate value of open positions
        positions_value = 0.0
        for position in open_positions:
            if position.get("current_price") and position.get("amount"):
                # Position value = current_price * amount
                positions_value += (
                    position["current_price"] * position["amount"]
                )

        # Available balance = total balance - value of open positions
        available_balance = total_balance - positions_value

        print(
            f"Calculated available balance: ${available_balance:.2f} (total: ${total_balance:.2f}, positions: ${positions_value:.2f})"
        )
        return max(0.0, available_balance)  # Ensure non-negative

    except Exception as e:
        print(f"Error calculating available balance: {e}")
        return get_paper_wallet_balance()  # Fallback to total balance


def set_paper_wallet_balance(balance: float) -> None:
    """Set paper wallet balance in multiple locations for consistency."""
    try:
        # Update paper wallet state file (highest priority)
        paper_wallet_state_file = Path(
            "crypto_bot/logs/paper_wallet_state.yaml"
        )
        if paper_wallet_state_file.exists():
            try:
                with open(paper_wallet_state_file, "r") as f:
                    state = yaml.safe_load(f) or {}
                state["balance"] = balance
                state["initial_balance"] = (
                    balance  # Also update initial balance
                )
                with open(paper_wallet_state_file, "w") as f:
                    yaml.dump(state, f, default_flow_style=False)
                print(
                    f"Frontend updated paper wallet state file: ${balance:.2f}"
                )
            except Exception as e:
                print(
                    f"Frontend failed to update paper wallet state file: {e}"
                )
        else:
            # Create new state file
            state = {
                "balance": balance,
                "initial_balance": balance,
                "realized_pnl": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "positions": {},
            }
            paper_wallet_state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(paper_wallet_state_file, "w") as f:
                yaml.dump(state, f, default_flow_style=False)
            print(
                f"Frontend created new paper wallet state file: ${balance:.2f}"
            )

        # Update paper_wallet.yaml
        paper_wallet_file = LOG_DIR / "paper_wallet.yaml"
        paper_config = {"initial_balance": balance}
        with open(paper_wallet_file, "w") as f:
            yaml.dump(paper_config, f, default_flow_style=False)
        print(f"Frontend updated paper_wallet.yaml: ${balance:.2f}")

        # Update user_config.yaml
        user_config_file = Path("crypto_bot/user_config.yaml")
        if user_config_file.exists():
            with open(user_config_file) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        config["paper_wallet_balance"] = balance
        with open(user_config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Frontend updated user_config.yaml: ${balance:.2f}")

        # Update legacy config if it exists
        legacy_config_path = Path("crypto_bot/paper_wallet_config.yaml")
        if legacy_config_path.exists():
            try:
                with open(legacy_config_path) as f:
                    legacy_config = yaml.safe_load(f) or {}
                legacy_config["initial_balance"] = balance
                with open(legacy_config_path, "w") as f:
                    yaml.dump(legacy_config, f, default_flow_style=False)
                print(
                    f"Frontend updated legacy config {legacy_config_path}: ${balance:.2f}"
                )
            except Exception as e:
                print(
                    f"Frontend failed to update legacy config {legacy_config_path}: {e}"
                )

        print(
            f"Frontend successfully updated paper wallet balance to: ${balance:.2f}"
        )

    except Exception as e:
        print(f"Error setting paper wallet balance: {e}")
        raise


def get_open_positions() -> list:
    """Get open positions from TradeManager (single source of truth)."""
    try:
        # Try to get positions from TradeManager first (highest priority)
        from crypto_bot.utils.trade_manager import get_trade_manager

        trade_manager = get_trade_manager()

        positions = trade_manager.get_all_positions()
        print(f"Found {len(positions)} positions in TradeManager")

        if positions:
            # Convert Position objects to the expected format
            result = []
            for position in positions:
                # Only include positions with non-zero amounts
                if position.total_amount <= 0:
                    continue

                # Get current price for unrealized P&L
                current_price = trade_manager.price_cache.get(position.symbol)
                print(
                    f"Checking price cache for {position.symbol}: {current_price}"
                )

                if not current_price:
                    # Try to fetch current price if not in cache
                    try:
                        print(
                            f"No cached price for {position.symbol}, fetching from exchange..."
                        )
                        # Import and use the same exchange that the price monitor uses
                        from crypto_bot.execution.cex_executor import (
                            get_exchange,
                        )

                        # Use the same config loading logic
                        config_path = (
                            Path(__file__).parent.parent
                            / "crypto_bot"
                            / "config.yaml"
                        )
                        if config_path.exists():
                            with open(config_path, "r") as f:
                                config = yaml.safe_load(f) or {}
                        else:
                            config = {}

                        exchange, _ = get_exchange(config)
                        print(f"Exchange initialized: {exchange}")

                        if hasattr(exchange, "fetch_ticker"):
                            print(f"Fetching ticker for {position.symbol}...")
                            ticker = exchange.fetch_ticker(position.symbol)
                            print(
                                f"Ticker response for {position.symbol}: {ticker}"
                            )

                            if ticker and ticker.get("last"):
                                current_price = Decimal(str(ticker["last"]))
                                # Update the cache with the fetched price
                                trade_manager.update_price(
                                    position.symbol, current_price
                                )
                                print(
                                    f"✅ Successfully fetched and cached current price for {position.symbol}: ${current_price}"
                                )
                            else:
                                print(
                                    f"❌ Invalid ticker response for {position.symbol}: {ticker}"
                                )
                        else:
                            print(
                                f"❌ Exchange does not have fetch_ticker method: {exchange}"
                            )
                    except Exception as e:
                        print(
                            f"❌ Failed to fetch current price for {position.symbol}: {e}"
                        )
                        import traceback

                        traceback.print_exc()

                if current_price:
                    unrealized_pnl, unrealized_pct = (
                        position.calculate_unrealized_pnl(current_price)
                    )
                    current_value = float(position.total_amount) * float(
                        current_price
                    )
                else:
                    # Use entry price as last resort fallback when current price is not available
                    print(
                        f"Using entry price as fallback for {position.symbol} (no current price available)"
                    )
                    current_price = position.average_price
                    unrealized_pnl = Decimal("0")
                    unrealized_pct = Decimal("0")
                    current_value = float(position.total_amount) * float(
                        position.average_price
                    )

                # Calculate additional fields for position cards
                pnl_value = float(unrealized_pnl)
                pnl_pct = float(unrealized_pct)

                # Generate chart data bounds (will be used by JavaScript)
                chart_min = (
                    min(float(current_price), float(position.average_price))
                    * 0.95
                )  # 5% below minimum
                chart_max = (
                    max(float(current_price), float(position.average_price))
                    * 1.05
                )  # 5% above maximum

                # Calculate trend strength and R-squared based on P&L
                trend_strength = (
                    "strong"
                    if abs(pnl_pct) > 2
                    else "moderate" if abs(pnl_pct) > 1 else "weak"
                )
                r_squared = min(99.9, max(60.0, 70.0 + abs(pnl_pct) * 2))

                pos_dict = {
                    "symbol": position.symbol,
                    "side": position.side,
                    "size": float(
                        position.total_amount
                    ),  # Use 'size' for consistency with template
                    "amount": float(
                        position.total_amount
                    ),  # Keep for backward compatibility
                    "entry_price": float(position.average_price),
                    "current_price": float(current_price),
                    "current_value": current_value,
                    "pnl": pnl_pct,  # PnL percentage (template expects this)
                    "pnl_value": pnl_value,  # PnL dollar amount (template expects this)
                    "pnl_percentage": pnl_pct,  # Keep for backward compatibility
                    "chart_min": chart_min,
                    "chart_max": chart_max,
                    "trend_strength": trend_strength,
                    "r_squared": r_squared,
                    "highest_price": (
                        float(position.highest_price)
                        if position.highest_price
                        else None
                    ),
                    "lowest_price": (
                        float(position.lowest_price)
                        if position.lowest_price
                        else None
                    ),
                    "stop_loss_price": (
                        float(position.stop_loss_price)
                        if position.stop_loss_price
                        else None
                    ),
                    "take_profit_price": (
                        float(position.take_profit_price)
                        if position.take_profit_price
                        else None
                    ),
                    "entry_time": (
                        position.entry_time.isoformat()
                        if position.entry_time
                        else None
                    ),
                }
                result.append(pos_dict)

            # Filter out test positions
            filtered_result = []
            for pos in result:
                if not is_test_position(pos["symbol"]):
                    filtered_result.append(pos)
                else:
                    print(
                        f"Filtering out test position from TradeManager result: {pos['symbol']}"
                    )

            print(
                f"Returning {len(filtered_result)} active positions from TradeManager (filtered {len(result) - len(filtered_result)} test positions)"
            )
            return filtered_result

    except Exception as e:
        print(f"Failed to get positions from TradeManager: {e}")

    # Fallback to trade manager state file (second priority)
    try:
        state_file = Path("crypto_bot/logs/trade_manager_state.json")
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)

            positions = state.get("positions", {})
            price_cache = state.get("price_cache", {})

            result = []
            for symbol, pos_data in positions.items():
                # Skip test positions
                if is_test_position(symbol):
                    print(
                        f"Filtering out test position from state file: {symbol}"
                    )
                    continue

                if pos_data.get("total_amount", 0) > 0:  # Only open positions
                    current_price = price_cache.get(
                        symbol, pos_data.get("average_price", 0)
                    )

                    # Calculate PnL
                    amount = pos_data["total_amount"]
                    avg_price = pos_data["average_price"]
                    side = pos_data["side"]

                    if side == "long":
                        pnl = (current_price - avg_price) * amount
                    else:  # short
                        pnl = (avg_price - current_price) * amount

                    pnl_pct = (
                        (pnl / (avg_price * amount)) * 100
                        if avg_price > 0
                        else 0
                    )

                    # Calculate current value
                    current_value = float(amount) * float(current_price)

                    # Calculate additional fields for position cards
                    pnl_value = float(pnl)
                    pnl_pct = float(pnl_pct)

                    # Generate chart data bounds (will be used by JavaScript)
                    chart_min = (
                        min(float(current_price), float(avg_price)) * 0.95
                    )  # 5% below minimum
                    chart_max = (
                        max(float(current_price), float(avg_price)) * 1.05
                    )  # 5% above maximum

                    # Calculate trend strength and R-squared based on P&L
                    trend_strength = (
                        "strong"
                        if abs(pnl_pct) > 2
                        else "moderate" if abs(pnl_pct) > 1 else "weak"
                    )
                    r_squared = min(99.9, max(60.0, 70.0 + abs(pnl_pct) * 2))

                    position_data = {
                        "symbol": symbol,
                        "side": side,
                        "size": float(
                            amount
                        ),  # Use 'size' for consistency with template
                        "amount": float(
                            amount
                        ),  # Keep for backward compatibility
                        "entry_price": float(avg_price),
                        "current_price": float(current_price),
                        "current_value": current_value,
                        "pnl": pnl_pct,  # PnL percentage (template expects this)
                        "pnl_value": pnl_value,  # PnL dollar amount (template expects this)
                        "pnl_percentage": pnl_pct,  # Keep for backward compatibility
                        "chart_min": chart_min,
                        "chart_max": chart_max,
                        "trend_strength": trend_strength,
                        "r_squared": r_squared,
                        "entry_time": pos_data.get("entry_time", ""),
                    }
                    result.append(position_data)

            print(f"Returning {len(result)} positions from state file")
            return result

    except Exception as e:
        print(f"Failed to get positions from state file: {e}")

    # Final fallback to log parsing (lowest priority)
    print("Falling back to log parsing for positions")
    return get_open_positions_from_log()


def get_open_positions_from_log() -> list:
    """Parse open positions from positions.log file (legacy method)."""
    import re
    from datetime import datetime, timedelta

    if not POSITIONS_FILE.exists():
        return []

    positions = []
    # Updated regex pattern to handle more position formats
    pos_patterns = [
        # Pattern 1: Standard format with pnl calculation
        re.compile(
            r"Active (?P<symbol>\S+) (?P<side>\w+) (?P<amount>[0-9.]+) "
            r"entry (?P<entry>[0-9.]+) current (?P<current>[0-9.]+) "
            r"pnl \$?(?P<pnl>[0-9.+-]+).*balance \$?(?P<balance>[0-9.]+)"
        ),
        # Pattern 2: Format without pnl calculation
        re.compile(
            r"Active (?P<symbol>\S+) (?P<side>\w+) (?P<amount>[0-9.]+) "
            r"entry (?P<entry>[0-9.]+) current (?P<current>[0-9.]+)"
        ),
        # Pattern 3: Alternative format
        re.compile(
            r"Active (?P<symbol>\S+) (?P<side>\w+) (?P<amount>[0-9.]+) "
            r"entry (?P<entry>[0-9.]+) current (?P<current>[0-9.]+) "
            r"pnl \$?(?P<pnl>[0-9.+-]+)"
        ),
    ]

    try:
        with open(POSITIONS_FILE) as f:
            lines = f.readlines()

        # Only process recent lines (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_positions = []

        for line in lines:
            # Extract timestamp from the beginning of the line
            timestamp_match = re.match(
                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line
            )
            if timestamp_match:
                try:
                    timestamp_str = timestamp_match.group(1)
                    line_timestamp = datetime.strptime(
                        timestamp_str, "%Y-%m-%d %H:%M:%S,%f"
                    )

                    # Only include positions from the last 24 hours
                    if line_timestamp >= cutoff_time:
                        # Try each pattern
                        position_data = None
                        for pattern in pos_patterns:
                            match = pattern.search(line)
                            if match:
                                # Check if this is a real position (not just a balance update)
                                symbol = match.group("symbol")
                                side = match.group("side")
                                amount = float(match.group("amount"))

                                # Filter out positions with zero amounts or very small amounts
                                if amount > 0.0001:  # Minimum threshold
                                    entry_price = float(match.group("entry"))

                                    # Get LIVE current price instead of cached price
                                    current_price = (
                                        get_current_price_for_symbol(symbol)
                                    )
                                    if current_price <= 0:
                                        # For unknown tokens, use entry price to show 0 PnL
                                        # This is better than using stale cached prices
                                        print(
                                            f"No live price available for {symbol}, using entry price for 0 PnL"
                                        )
                                        current_price = entry_price

                                    # Calculate PnL if not provided
                                    if (
                                        "pnl" in match.groupdict()
                                        and match.group("pnl")
                                    ):
                                        pnl = float(match.group("pnl"))
                                    else:
                                        # Calculate PnL manually
                                        if side == "buy":
                                            pnl = (
                                                current_price - entry_price
                                            ) * amount
                                        else:  # sell/short
                                            pnl = (
                                                entry_price - current_price
                                            ) * amount

                                    # Get balance if available
                                    balance = 0.0
                                    if (
                                        "balance" in match.groupdict()
                                        and match.group("balance")
                                    ):
                                        balance = float(match.group("balance"))

                                    # Calculate current value
                                    current_value = amount * current_price

                                    # Calculate PnL percentage
                                    pnl_pct = (
                                        (pnl / (entry_price * amount)) * 100
                                        if entry_price > 0
                                        else 0
                                    )

                                    # Calculate additional fields for position cards
                                    pnl_value = float(pnl)

                                    # Generate chart data bounds (will be used by JavaScript)
                                    chart_min = (
                                        min(
                                            float(current_price),
                                            float(entry_price),
                                        )
                                        * 0.95
                                    )  # 5% below minimum
                                    chart_max = (
                                        max(
                                            float(current_price),
                                            float(entry_price),
                                        )
                                        * 1.05
                                    )  # 5% above maximum

                                    # Calculate trend strength and R-squared based on P&L
                                    trend_strength = (
                                        "strong"
                                        if abs(pnl_pct) > 2
                                        else (
                                            "moderate"
                                            if abs(pnl_pct) > 1
                                            else "weak"
                                        )
                                    )
                                    r_squared = min(
                                        99.9,
                                        max(60.0, 70.0 + abs(pnl_pct) * 2),
                                    )

                                    position_data = {
                                        "symbol": symbol,
                                        "side": side,
                                        "size": float(
                                            amount
                                        ),  # Use 'size' for consistency with template
                                        "amount": amount,  # Keep for backward compatibility
                                        "entry_price": entry_price,
                                        "current_price": current_price,
                                        "current_value": current_value,
                                        "pnl": pnl_pct,  # PnL percentage (template expects this)
                                        "pnl_value": pnl_value,  # PnL dollar amount (template expects this)
                                        "pnl_percentage": pnl_pct,  # Keep for backward compatibility
                                        "chart_min": chart_min,
                                        "chart_max": chart_max,
                                        "trend_strength": trend_strength,
                                        "r_squared": r_squared,
                                        "balance": balance,
                                        "timestamp": timestamp_str,
                                    }
                                    break

                        if position_data:
                            recent_positions.append(position_data)

                except ValueError as e:
                    print(
                        f"Error parsing timestamp in line: {line.strip()}, error: {e}"
                    )
                    continue

        # Remove duplicates based on symbol and side, keeping the most recent
        seen = set()
        unique_positions = []
        for pos in reversed(
            recent_positions
        ):  # Process in reverse to keep most recent
            key = f"{pos['symbol']}_{pos['side']}"
            if key not in seen:
                seen.add(key)
                unique_positions.append(pos)

        # Filter out test positions
        filtered_positions = []
        for pos in unique_positions:
            if not is_test_position(pos["symbol"]):
                filtered_positions.append(pos)
            else:
                print(
                    f"Filtering out test position from log parsing: {pos['symbol']}"
                )

        # Return positions in chronological order
        return list(reversed(filtered_positions))

    except Exception as e:
        print(f"Error parsing positions from log: {e}")
        return []

    return []


def clear_old_positions() -> None:
    """Clear old position entries from the positions.log file."""
    if not POSITIONS_FILE.exists():
        return

    try:
        import re
        from datetime import datetime, timedelta

        # Read all lines
        with open(POSITIONS_FILE, "r") as f:
            lines = f.readlines()

        # Keep only lines from the last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_lines = []

        for line in lines:
            # Extract timestamp from the beginning of the line
            timestamp_match = re.match(
                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line
            )
            if timestamp_match:
                try:
                    timestamp_str = timestamp_match.group(1)
                    line_timestamp = datetime.strptime(
                        timestamp_str, "%Y-%m-%d %H:%M:%S,%f"
                    )

                    # Keep lines from the last 24 hours
                    if line_timestamp >= cutoff_time:
                        recent_lines.append(line)
                except ValueError:
                    # Keep lines that don't have valid timestamps (they might be important)
                    recent_lines.append(line)
            else:
                # Keep lines without timestamps
                recent_lines.append(line)

        # Write back the filtered lines
        with open(POSITIONS_FILE, "w") as f:
            f.writelines(recent_lines)

    except Exception as e:
        print(f"Error clearing old positions: {e}")


def get_uptime() -> str:
    """Return human readable uptime."""
    return utils.get_uptime(bot_start_time)


def calculate_wallet_pnl() -> Dict[str, float]:
    """Calculate current wallet PnL based on paper wallet state, open positions, and trade history."""
    try:
        # Try to load paper wallet state first
        paper_wallet_state_file = Path(
            "crypto_bot/logs/paper_wallet_state.yaml"
        )
        if paper_wallet_state_file.exists():
            try:
                import numpy as np
                from yaml import Loader, SafeLoader

                # Custom loader to handle numpy scalars
                class NumpyLoader(SafeLoader):
                    pass

                def construct_numpy_scalar(loader, node):
                    """Construct numpy scalar from YAML node."""
                    try:
                        # Get the numpy dtype and binary data
                        if hasattr(node, "value") and isinstance(
                            node.value, list
                        ):
                            dtype_info = node.value[0]
                            binary_data = node.value[1]

                            # Extract binary data
                            if hasattr(binary_data, "value"):
                                import base64
                                import struct

                                # Decode base64 binary data
                                decoded = base64.b64decode(binary_data.value)
                                # Convert to float64 (little endian)
                                value = struct.unpack("<d", decoded)[0]
                                return float(value)
                    except Exception as e:
                        print(f"Error decoding numpy scalar: {e}")
                        return 0.0

                    return 0.0

                # Add constructor for numpy scalars
                NumpyLoader.add_constructor(
                    "tag:yaml.org,2002:python/object/apply:numpy._core.multiarray.scalar",
                    construct_numpy_scalar,
                )

                with open(paper_wallet_state_file, "r") as f:
                    state = yaml.load(f, Loader=NumpyLoader) or {}
                    current_balance = state.get("balance", 0.0)
                    initial_balance = state.get("initial_balance", 10000.0)
                    realized_pnl = state.get("realized_pnl", 0.0)
                    logger.info(
                        f"Loaded paper wallet state: balance=${current_balance:.2f}, realized_pnl=${realized_pnl:.2f}"
                    )
            except Exception as e:
                logger.error(f"Error reading paper wallet state: {e}")
                current_balance = get_paper_wallet_balance()
                initial_balance = current_balance
                realized_pnl = 0.0
        else:
            # Fallback to reading from positions.log
            current_balance = get_paper_wallet_balance()
            initial_balance = current_balance
            realized_pnl = 0.0

        # Always calculate realized P&L from trade history for accuracy
        try:
            from crypto_bot import log_reader

            df = log_reader._read_trades(TRADE_FILE)

            if not df.empty:
                # Track position history for realized P&L calculation
                position_history = {}
                calculated_realized_pnl = 0.0

                for _, row in df.iterrows():
                    symbol = str(row.get("symbol", ""))
                    side = str(row.get("side", ""))
                    amount = float(row.get("amount", 0))
                    price = float(row.get("price", 0))

                    if symbol and amount > 0 and price > 0:
                        # Calculate trade total
                        total = amount * price

                        # Check if this trade closes an existing position
                        if symbol in position_history:
                            existing_pos = position_history[symbol]

                            # Check if this is a closing trade (opposite side)
                            if (
                                side == "sell"
                                and existing_pos["side"] == "buy"
                            ) or (
                                side == "buy"
                                and existing_pos["side"] == "sell"
                            ):

                                # Calculate realized PnL
                                if side == "sell":  # Closing long position
                                    pnl = (
                                        price - existing_pos["price"]
                                    ) * min(amount, existing_pos["amount"])
                                else:  # Closing short position
                                    pnl = (
                                        existing_pos["price"] - price
                                    ) * min(amount, existing_pos["amount"])

                                calculated_realized_pnl += pnl

                                # Update or remove position
                                if amount >= existing_pos["amount"]:
                                    del position_history[symbol]
                                else:
                                    position_history[symbol][
                                        "amount"
                                    ] -= amount
                            else:
                                # Same side trade - average the position
                                if symbol in position_history:
                                    total_cost = (
                                        existing_pos["price"]
                                        * existing_pos["amount"]
                                    ) + total
                                    total_amount = (
                                        existing_pos["amount"] + amount
                                    )
                                    position_history[symbol] = {
                                        "side": side,
                                        "price": total_cost / total_amount,
                                        "amount": total_amount,
                                    }
                                else:
                                    position_history[symbol] = {
                                        "side": side,
                                        "price": price,
                                        "amount": amount,
                                    }
                        else:
                            # New position
                            position_history[symbol] = {
                                "side": side,
                                "price": price,
                                "amount": amount,
                            }

                realized_pnl = calculated_realized_pnl

                # Use dashboard P&L calculation for consistency
                from frontend.utils import compute_performance

                dashboard_perf = compute_performance(df)
                dashboard_total_pnl = dashboard_perf.get("total_pnl", 0.0)

                # Use dashboard P&L calculation for consistency
                realized_pnl = dashboard_total_pnl

        except Exception as e:
            print(f"Error calculating realized P&L from trade history: {e}")

        # Get open positions using the same method as the trades data API
        try:
            from crypto_bot.utils.open_trades import get_open_trades

            open_trades = get_open_trades(TRADE_FILE)
            if not open_trades:
                open_positions = []
            else:
                # Get current prices for PnL calculation (same as trades_data API)
                current_prices = {}
                try:
                    # Get current prices from the current-prices endpoint
                    import requests

                    response = requests.get(
                        "http://localhost:8000/api/current-prices", timeout=5
                    )
                    if response.status_code == 200:
                        current_prices = response.json()
                except Exception:
                    # Fallback: get prices directly
                    for trade in open_trades:
                        price = get_current_price_for_symbol(trade["symbol"])
                        if price > 0:
                            current_prices[trade["symbol"]] = price

                # Convert the open trades format and calculate PnL (same logic as trades_data API)
                open_positions = []
                for trade in open_trades:
                    symbol = trade["symbol"]
                    side = trade["side"]
                    amount = float(trade["amount"])
                    entry_price = float(trade["price"])
                    current_price = current_prices.get(symbol, 0.0)

                    # Calculate unrealized PnL (same logic as trades_data API)
                    unrealized_pnl = 0.0
                    if current_price > 0:
                        if side == "long":
                            unrealized_pnl = (
                                current_price - entry_price
                            ) * amount
                        else:  # short
                            unrealized_pnl = (
                                entry_price - current_price
                            ) * amount

                    open_positions.append(
                        {
                            "symbol": symbol,
                            "side": side,
                            "amount": amount,
                            "entry_price": entry_price,
                            "current_price": current_price,
                            "unrealized_pnl": unrealized_pnl,
                        }
                    )
        except Exception as e:
            print(f"Error getting open positions: {e}")
            open_positions = []

        # Calculate unrealized PnL from open positions and total invested amount
        unrealized_pnl = 0.0
        total_invested_in_active_trades = 0.0
        position_details = []

        for position in open_positions:
            symbol = position["symbol"]
            side = position["side"]
            amount = position["amount"]
            entry_price = position["entry_price"]
            current_price = position["current_price"]
            position_unrealized_pnl = position.get("unrealized_pnl", 0.0)

            # Use the pre-calculated unrealized PnL (same as trades_data API)
            unrealized_pnl += position_unrealized_pnl

            # Calculate total amount invested in this position
            total_invested_in_active_trades += entry_price * amount

            # Calculate PnL percentage
            pnl_percentage = 0.0
            if entry_price > 0 and amount > 0:
                pnl_percentage = (
                    position_unrealized_pnl / (entry_price * amount)
                ) * 100

            position_details.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "amount": amount,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "pnl": position_unrealized_pnl,
                    "pnl_percentage": pnl_percentage,
                }
            )

        # Calculate total PnL (realized + unrealized)
        total_pnl = realized_pnl + unrealized_pnl

        # Current balance = initial balance + realized PnL - amount invested in active trades
        # This gives us the available cash + unrealized PnL from active positions
        total_balance = (
            initial_balance
            + realized_pnl
            - total_invested_in_active_trades
            + unrealized_pnl
        )

        return {
            "initial_balance": initial_balance,
            "current_balance": total_balance,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "total_invested_in_active_trades": total_invested_in_active_trades,
            "pnl_percentage": (
                (total_pnl / initial_balance) * 100
                if initial_balance > 0
                else 0
            ),
            "open_positions": position_details,
            "position_count": len(open_positions),
        }

    except Exception as e:
        print(f"Error calculating wallet PnL: {e}")
        return {
            "initial_balance": 0.0,
            "current_balance": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "pnl_percentage": 0.0,
            "open_positions": [],
            "position_count": 0,
            "error": str(e),
        }


@app.route("/api/test")
def api_test():
    """Simple test endpoint to verify API is working."""
    return jsonify(
        {
            "status": "success",
            "message": "API is working",
            "timestamp": str(datetime.now()),
        }
    )


@app.route("/api/debug-positions")
def api_debug_positions():
    """Debug endpoint to check what data is available."""
    try:
        import json
        from pathlib import Path

        debug_info = {
            "state_file_exists": False,
            "state_file_path": "",
            "positions_count": 0,
            "price_cache_count": 0,
            "legacy_positions_count": 0,
            "error": None,
        }

        # Check state file
        state_file = Path("crypto_bot/logs/trade_manager_state.json")
        debug_info["state_file_path"] = str(state_file)
        debug_info["state_file_exists"] = state_file.exists()

        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)
            debug_info["positions_count"] = len(state.get("positions", {}))
            debug_info["price_cache_count"] = len(state.get("price_cache", {}))

        # Check legacy method
        try:
            legacy_positions = get_open_positions()
            debug_info["legacy_positions_count"] = len(legacy_positions)
        except Exception as e:
            debug_info["error"] = str(e)

        return jsonify(debug_info)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/open-positions")
def api_open_positions():
    """Return open positions data for the dashboard."""
    try:
        logger.info("API: Starting open positions request")

        # Load TradeManager state directly from the resolved LOG_DIR to avoid CWD issues
        import json
        state_file = LOG_DIR / "trade_manager_state.json"
        logger.info(f"API: Checking state file at {state_file.resolve()}")

        if state_file.exists():
            logger.info("API: State file exists, loading data")
            with open(state_file, "r") as f:
                state = json.load(f)

            # Get positions from state file
            positions = state.get("positions", {})
            price_cache = state.get("price_cache", {})
            logger.info(f"API: Found {len(positions)} positions in state file")
            logger.info(f"API: Found {len(price_cache)} prices in cache")

            open_positions = []
            for symbol, pos_data in positions.items():
                logger.info(f"API: Processing position {symbol}: {pos_data}")

                # Skip test positions
                if is_test_position(symbol):
                    logger.warning(
                        f"Filtering out test position in API: {symbol}"
                    )
                    continue

                if pos_data.get("total_amount", 0) > 0:  # Open position
                    # Try to get current price from cache first
                    current_price = price_cache.get(symbol, 0)

                    # If no cached price or price is stale, try to fetch current price on-demand
                    if not current_price or current_price == 0 or current_price == pos_data.get("average_price", 0):
                        try:
                            # Use shared price fetcher alias for robustness  
                            fresh_price = get_current_price_for_symbol(symbol)
                            if fresh_price and fresh_price > 0:
                                current_price = fresh_price
                                logger.debug(
                                    f"Fetched fresh current price for {symbol}: ${current_price}"
                                )
                                # Update the TradeManager cache with fresh price
                                try:
                                    from crypto_bot.utils.trade_manager import get_trade_manager
                                    tm = get_trade_manager()
                                    tm.update_price(symbol, current_price)
                                except Exception as cache_update_error:
                                    logger.warning(f"Failed to update price cache for {symbol}: {cache_update_error}")
                            else:
                                logger.warning(f"No valid fresh price received for {symbol}")
                        except Exception as price_error:
                            logger.warning(
                                f"Failed to fetch current price for {symbol}: {price_error}"
                            )
                            # Try to get from TradeManager cache as fallback
                            try:
                                from crypto_bot.utils.trade_manager import get_trade_manager
                                tm = get_trade_manager()
                                cached_price = tm.price_cache.get(symbol)
                                if cached_price and float(cached_price) > 0:
                                    current_price = float(cached_price)
                                    logger.debug(f"Using TradeManager cached price for {symbol}: ${current_price}")
                                else:
                                    current_price = pos_data.get("average_price", 0)
                                    logger.warning(f"No valid price available for {symbol}, using entry price as fallback")
                            except Exception as tm_error:
                                logger.error(f"Failed to get TradeManager price for {symbol}: {tm_error}")
                                current_price = pos_data.get("average_price", 0)

                    # Calculate PnL
                    amount = pos_data["total_amount"]
                    avg_price = pos_data["average_price"]
                    side = pos_data["side"]

                    if side == "long":
                        pnl = (current_price - avg_price) * amount
                    else:  # short
                        pnl = (avg_price - current_price) * amount

                    pnl_pct = (
                        (pnl / (avg_price * amount)) * 100
                        if avg_price > 0
                        else 0
                    )

                    position_data = {
                        "symbol": symbol,
                        "side": side,
                        "size": float(amount),  # Add size for template consistency
                        "amount": float(amount),
                        "entry_price": float(avg_price),
                        "current_price": float(current_price),
                        "current_value": (
                            float(current_price * amount)
                            if current_price
                            else 0.0
                        ),
                        "pnl": float(pnl_pct),  # Template expects percentage here
                        "pnl_value": float(pnl),  # Template expects dollar amount here
                        "pnl_percentage": float(pnl_pct),  # Keep for backward compatibility
                        "entry_time": pos_data.get("entry_time", ""),
                        "position_value": (
                            float(current_price * amount)
                            if current_price
                            else 0.0
                        ),
                        # Add missing fields for template
                        "r_squared": min(99.9, max(60.0, 70.0 + abs(pnl_pct) * 2)),
                        # Include stop loss/trailing stop price for chart line
                        "stop_price": (
                            float(pos_data.get("stop_loss_price"))
                            if pos_data.get("stop_loss_price") is not None
                            else None
                        ),
                    }
                    open_positions.append(position_data)
                    logger.info(f"API: Added position {symbol} to response")

            # Cross-check against CSV-derived open trades to eliminate stale entries
            try:
                from crypto_bot.utils.open_trades import get_open_trades
                csv_opens = get_open_trades(TRADE_FILE)
                csv_symbols = {o.get("symbol") for o in csv_opens}
                filtered_positions = [p for p in open_positions if p.get("symbol") in csv_symbols] if csv_symbols else open_positions
                logger.info(
                    f"API: Returning {len(filtered_positions)} positions after CSV cross-check (state had {len(open_positions)})"
                )
                return jsonify(filtered_positions)
            except Exception as _csv_check_err:
                logger.warning(f"API: CSV cross-check failed: {_csv_check_err}")
                logger.info(
                    f"API: Returning {len(open_positions)} open positions from state file"
                )
                return jsonify(open_positions)
        else:
            logger.warning(
                "API: State file does not exist, using legacy method"
            )
            # Fallback to legacy method
            open_positions = get_open_positions()
            logger.info(
                f"API: Returning {len(open_positions)} positions from legacy method"
            )
            return jsonify(open_positions)

    except Exception as e:
        logger.error(f"Failed to get open positions for API: {e}")
        # Try legacy method as final fallback
        try:
            open_positions = get_open_positions()
            logger.warning(
                f"API: Using legacy fallback, returning {len(open_positions)} positions"
            )
            return jsonify(open_positions)
        except Exception as fallback_error:
            logger.error(f"API: Legacy fallback also failed: {fallback_error}")
            return jsonify({"error": str(fallback_error)}), 500


def fetch_current_price_for_symbol(symbol):
    """Fetch current price for a symbol using available exchanges."""
    import ccxt

    # Try multiple exchanges in order
    exchanges_to_try = [
        ('kraken', ccxt.kraken()),
        ('binance', ccxt.binance()),
        ('coinbase', ccxt.coinbase()),
        ('bitstamp', ccxt.bitstamp()),
    ]

    for exchange_name, exchange in exchanges_to_try:
        try:
            ticker = exchange.fetch_ticker(symbol)
            price = ticker.get("last", 0)
            if price and price > 0:
                logger.debug(f"Successfully fetched price for {symbol} from {exchange_name}: ${price}")
                return price
        except Exception as e:
            logger.debug(f"{exchange_name} failed for {symbol}: {e}")
            continue

    # If all exchanges fail, try to get price from TradeManager cache as last resort
    try:
        from crypto_bot.utils.trade_manager import get_trade_manager
        tm = get_trade_manager()
        cached_price = tm.price_cache.get(symbol)
        if cached_price:
            logger.debug(f"Using cached price for {symbol}: ${cached_price}")
            return float(cached_price)
    except Exception as e:
        logger.debug(f"Failed to get cached price for {symbol}: {e}")

    # Return 0 if all methods fail
    logger.warning(f"All price fetching methods failed for {symbol}")
    return 0


@app.route("/api/wallet-pnl")
def api_wallet_pnl():
    """Return current wallet PnL calculation from TradeManager."""
    try:
        # Load TradeManager state directly from file to bypass singleton issues
        state_file = LOG_DIR / "trade_manager_state.json"
        if state_file.exists():
            state = safe_json_load(state_file)

            # Calculate P&L from state file data
            trades = state.get("trades", [])
            positions = state.get("positions", {})
            stats = state.get("statistics", {})

            # Get current prices for unrealized P&L calculation
            price_cache = state.get("price_cache", {})

            total_unrealized_pnl = 0.0
            for symbol, pos_data in positions.items():
                if pos_data.get("total_amount", 0) > 0:  # Open position
                    current_price = price_cache.get(
                        symbol, pos_data.get("average_price", 0)
                    )
                    if current_price and pos_data.get("average_price"):
                        amount = pos_data["total_amount"]
                        avg_price = pos_data["average_price"]
                        side = pos_data["side"]

                        if side == "long":
                            pnl = (current_price - avg_price) * amount
                        else:  # short
                            pnl = (avg_price - current_price) * amount

                        total_unrealized_pnl += pnl

            realized_pnl = float(stats.get("total_realized_pnl", 0))
            total_pnl = realized_pnl + total_unrealized_pnl

            pnl_data = {
                "total_pnl": total_pnl,
                "pnl_percentage": (
                    (total_pnl / 10000.0) * 100 if total_pnl != 0 else 0.0
                ),
                "realized_pnl": realized_pnl,
                "unrealized_pnl": total_unrealized_pnl,
                "balance": 10000.0 + total_pnl,  # Initial balance + total P&L
                "initial_balance": 10000.0,
            }

            return jsonify(pnl_data)

        # Fallback if no state file
        return jsonify(
            {
                "total_pnl": 0.0,
                "pnl_percentage": 0.0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "balance": 10000.0,
                "initial_balance": 10000.0,
            }
        )

    except Exception as e:
        logger.warning(f"Failed to get wallet PnL from state file: {e}")
        # Fallback to legacy calculation
        try:
            pnl_data = calculate_wallet_pnl()
            return jsonify(pnl_data)
        except Exception as fallback_error:
            logger.error(
                f"Fallback P&L calculation also failed: {fallback_error}"
            )
            return jsonify({"error": str(fallback_error)})


@app.route("/start", methods=["POST"])
def start():
    global bot_proc, bot_start_time
    mode = request.form.get("mode", "dry_run")
    set_execution_mode(mode)
    if not is_running() and not check_existing_bot():
        # Launch the asyncio-based trading bot using the non-interactive script
        venv_python = Path(__file__).parent.parent / "venv" / "bin" / "python3"
        bot_script = (
            Path(__file__).parent.parent / "start_bot_noninteractive.py"
        )
        bot_proc = subprocess.Popen([str(venv_python), str(bot_script)])
        bot_start_time = time.time()
    return redirect(url_for("index"))


@app.route("/start_bot", methods=["POST"])
def start_bot():
    """Start the trading bot and return JSON status."""
    global bot_proc, bot_start_time
    mode = (
        request.json.get("mode", "dry_run")
        if request.is_json
        else request.form.get("mode", "dry_run")
    )
    print(f"Starting bot with mode: {mode}")
    set_execution_mode(mode)

    # Check if we have a tracked subprocess running
    if utils.is_running(bot_proc):
        print("Bot subprocess is already running")
        return jsonify(
            {
                "status": "already_running",
                "running": True,
                "uptime": get_uptime(),
                "mode": mode,
                "message": "Bot is already running",
            }
        )

    # Check if there's another bot process running (skip in testing)
    if (not app.testing) and check_existing_bot():
        print("Another bot process detected, sending start command")
        try:
            # LOG_DIR already imported above
            control_file = LOG_DIR / "bot_control.json"
            with open(control_file, "w") as f:
                json.dump({"command": "start"}, f)

            # Set start time if not already set (for existing processes)
            global bot_start_time
            if bot_start_time is None:
                bot_start_time = time.time()

            return jsonify(
                {
                    "status": "started",
                    "running": True,
                    "uptime": get_uptime(),
                    "mode": mode,
                    "message": "Started existing bot via control command",
                }
            )
        except Exception as e:
            print(f"Error sending start command: {e}")
            return jsonify(
                {
                    "status": f"error: {e}",
                    "running": False,
                    "uptime": get_uptime(),
                    "mode": mode,
                    "message": f"Failed to send start command: {e}",
                }
            )

    # Start new bot process
    print("Starting new bot process")
    try:
        venv_python = Path(__file__).parent.parent / "venv" / "bin" / "python3"
        bot_script = (
            Path(__file__).parent.parent / "start_bot_noninteractive.py"
        )

        print(f"Using Python: {venv_python}")
        print(f"Using script: {bot_script}")

        if not venv_python.exists():
            print(f"Python executable not found: {venv_python}")
            return jsonify(
                {
                    "status": "error: Python executable not found",
                    "running": False,
                    "uptime": get_uptime(),
                    "mode": mode,
                    "message": "Python executable not found",
                }
            )

        if not bot_script.exists():
            print(f"Bot script not found: {bot_script}")
            return jsonify(
                {
                    "status": "error: Bot script not found",
                    "running": False,
                    "uptime": get_uptime(),
                    "mode": mode,
                    "message": "Bot script not found",
                }
            )

        # Pass environment variables to subprocess
        env = os.environ.copy()
        bot_proc = subprocess.Popen(
            [str(venv_python), str(bot_script)], env=env
        )
        bot_start_time = time.time()

        # Wait a moment to see if the process starts successfully
        time.sleep(1)

        if bot_proc.poll() is None:
            print("Bot process started successfully")
            return jsonify(
                {
                    "status": "started",
                    "running": True,
                    "uptime": get_uptime(),
                    "mode": mode,
                    "message": "Bot process started",
                }
            )
        else:
            print(
                f"Bot process failed to start, return code: {bot_proc.returncode}"
            )
            return jsonify(
                {
                    "status": f"error: Bot process failed to start (return code: {bot_proc.returncode})",
                    "running": False,
                    "uptime": get_uptime(),
                    "mode": mode,
                    "message": "Bot process failed to start",
                }
            )

    except Exception as e:
        print(f"Error starting bot: {e}")
        return jsonify(
            {
                "status": f"error: {e}",
                "running": False,
                "uptime": get_uptime(),
                "mode": mode,
                "message": f"Error starting bot: {e}",
            }
        )


@app.route("/stop")
def stop():
    global bot_proc, bot_start_time
    if is_running():
        bot_proc.terminate()
        bot_proc.wait()
    bot_proc = None
    bot_start_time = None
    return redirect(url_for("index"))


@app.route("/stop_bot", methods=["POST"])
def stop_bot():
    """Stop the trading bot and return JSON status."""
    global bot_proc, bot_start_time
    status = "not_running"

    # Send stop command to running bot if it exists
    if check_existing_bot():
        try:
            # LOG_DIR already imported above
            control_file = LOG_DIR / "bot_control.json"
            with open(control_file, "w") as f:
                json.dump({"command": "stop"}, f)
            status = "stopped"
        except Exception as e:
            status = f"error: {e}"
    elif is_running():
        bot_proc.terminate()
        bot_proc.wait()
        status = "stopped"
        bot_proc = None
        bot_start_time = None

    return jsonify(
        {
            "status": status,
            "running": False,
            "uptime": get_uptime(),
            "mode": load_execution_mode(),
            "message": "Bot stopped" if status == "stopped" else status,
        }
    )


@app.route("/pause_bot", methods=["POST"])
def pause_bot():
    """Pause the trading bot and return JSON status."""
    global bot_proc, bot_start_time
    status = "not_running"
    if is_running():
        # Send SIGSTOP to pause the process
        bot_proc.send_signal(signal.SIGSTOP)
        status = "paused"
    return jsonify(
        {
            "status": status,
            "running": False,
            "uptime": get_uptime(),
            "mode": load_execution_mode(),
        }
    )


@app.route("/resume_bot", methods=["POST"])
def resume_bot():
    """Resume the trading bot and return JSON status."""
    global bot_proc, bot_start_time
    status = "not_running"
    if bot_proc and bot_proc.poll() is None:
        # Send SIGCONT to resume the process
        bot_proc.send_signal(signal.SIGCONT)
        status = "resumed"
    return jsonify(
        {
            "status": status,
            "running": True,
            "uptime": get_uptime(),
            "mode": load_execution_mode(),
        }
    )


@app.route("/bot_logs")
def bot_logs_page():
    """Bot logs page with navigation."""
    mode = load_execution_mode()
    return render_template(
        "bot_logs.html",
        running=is_running(),
        mode=mode,
        uptime=get_uptime(),
        title="Bot Logs",
    )


@app.route("/logs_tail")
def logs_tail():
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text().splitlines()[-200:]
        return "\n".join(lines)
    return ""


@app.route("/stats")
def stats():
    data = {}
    if STATS_FILE.exists():
        with open(STATS_FILE) as f:
            data = json.load(f)
    return render_template("stats.html", stats=data)


@app.route("/scans")
def scans():
    data = {}
    if SCAN_FILE.exists():
        with open(SCAN_FILE) as f:
            data = json.load(f)
    return render_template("scans.html", scans=data)


@app.route("/cli", methods=["GET", "POST"])
def cli():
    """Run CLI commands and display output."""
    output = None
    if request.method == "POST":
        base = request.form.get("base", "bot")
        cmd_args = request.form.get("command", "")
        venv_python = Path(__file__).parent.parent / "venv" / "bin" / "python3"
        if base == "backtest":
            cmd = f"{venv_python} -m crypto_bot.backtest.backtest_runner {cmd_args}"
        elif base == "custom":
            cmd = cmd_args
        else:
            cmd = f"{venv_python} start_bot_noninteractive.py {cmd_args}"
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, check=False
            )
            output = proc.stdout + proc.stderr
        except Exception as exc:  # pragma: no cover - subprocess
            output = str(exc)
    return render_template("cli.html", output=output)


# Global cache for dashboard data (disabled to ensure freshness)
_dashboard_cache = {}
_CACHE_TIMEOUT = 0  # disable caching


@app.route("/")
def index():
    """Root route: redirect to the main dashboard."""
    return redirect(url_for("dashboard"))


def get_cached_dashboard_data():
    """Dashboard caching disabled."""
    return None


def set_cached_dashboard_data(data):
    """Dashboard caching disabled (no-op)."""
    return


def batch_fetch_prices(symbols):
    """Fetch prices for multiple symbols in batch to reduce API calls."""
    import ccxt

    prices = {}

    # Try Kraken first for major pairs
    kraken_symbols = []
    binance_symbols = []

    for symbol in symbols:
        # Normalize for Kraken
        kraken_symbol = symbol.replace("BTC/", "XBT/").replace("/BTC", "/XBT")
        kraken_symbols.append(kraken_symbol)
        binance_symbols.append(symbol)

    try:
        exchange = ccxt.kraken()
        # Fetch tickers in batch if supported
        kraken_tickers = exchange.fetch_tickers(kraken_symbols)
        for symbol, kraken_symbol in zip(symbols, kraken_symbols):
            if kraken_symbol in kraken_tickers:
                prices[symbol] = kraken_tickers[kraken_symbol].get("last", 0)
    except Exception as e:
        logger.debug(f"Kraken batch fetch failed: {e}")

    # Fill missing prices with Binance
    missing_symbols = [s for s in symbols if s not in prices]
    if missing_symbols:
        try:
            exchange = ccxt.binance()
            binance_tickers = exchange.fetch_tickers(missing_symbols)
            for symbol in missing_symbols:
                if symbol in binance_tickers:
                    prices[symbol] = binance_tickers[symbol].get("last", 0)
        except Exception as e:
            logger.debug(f"Binance batch fetch failed: {e}")

    return prices


@app.route("/dashboard")
def dashboard():
    """Dashboard with cache-busting timestamp."""
    # Try to use cached data first
    cached_data = get_cached_dashboard_data()
    if cached_data:
        return render_template("dashboard.html", **cached_data)

    mode = load_execution_mode()

    # Get performance data (cache this operation)
    df = log_reader._read_trades(TRADE_FILE)
    perf = utils.compute_performance(df)

    # Get dynamic allocation data based on actual performance
    allocation = utils.calculate_dynamic_allocation()

    # Fallback to static config if no dynamic data available
    if not allocation and CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg = yaml.safe_load(f) or {}
            allocation = cfg.get("strategy_allocation", {})

    # Final fallback to weights.json if no allocation in config
    if not allocation and (LOG_DIR / "weights.json").exists():
        with open(LOG_DIR / "weights.json") as f:
            weights_data = json.load(f)
            # Convert decimal weights to percentages for consistency
            allocation = {
                strategy: weight * 100
                for strategy, weight in weights_data.items()
            }

    # Get paper wallet balance (always show wallet balance)
    # Calculate correct balance from state file
    try:
        import json
        state_file = LOG_DIR / "trade_manager_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)

            # Calculate total P&L from state file
            positions = state.get("positions", {})
            stats = state.get("statistics", {})
            price_cache = state.get("price_cache", {})

            total_unrealized_pnl = 0.0
            for symbol, pos_data in positions.items():
                if pos_data.get("total_amount", 0) > 0:  # Open position
                    current_price = price_cache.get(
                        symbol, pos_data.get("average_price", 0)
                    )
                    if current_price and pos_data.get("average_price"):
                        amount = pos_data["total_amount"]
                        avg_price = pos_data["average_price"]
                        side = pos_data["side"]

                        if side == "long":
                            pnl = (current_price - avg_price) * amount
                        else:  # short
                            pnl = (avg_price - current_price) * amount

                        total_unrealized_pnl += pnl

            realized_pnl = float(stats.get("total_realized_pnl", 0))
            total_pnl = realized_pnl + total_unrealized_pnl

            # Calculate correct wallet balance
            paper_wallet_balance = 10000.0 + total_pnl
            logger.info(
                f"Dashboard: Calculated wallet balance from state file: ${paper_wallet_balance:.2f}"
            )
        else:
            # If no state file, try TradeManager first, then CSV as fallback
            try:
                paper_wallet_balance = (
                    calculate_wallet_balance_from_trade_manager()
                )
                logger.info(
                    f"Dashboard: Using TradeManager wallet balance: ${paper_wallet_balance:.2f}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to get balance from TradeManager, falling back to CSV: {e}"
                )
                paper_wallet_balance = calculate_wallet_balance_from_csv()
                logger.info(
                    f"Dashboard: Using CSV fallback wallet balance: ${paper_wallet_balance:.2f}"
                )
    except Exception as e:
        logger.warning(
            f"Failed to calculate paper wallet balance from state file: {e}"
        )
        # Try TradeManager first, then CSV as final fallback
        try:
            paper_wallet_balance = (
                calculate_wallet_balance_from_trade_manager()
            )
            logger.info(
                f"Dashboard: Using TradeManager fallback wallet balance: ${paper_wallet_balance:.2f}"
            )
        except Exception as tm_e:
            logger.warning(
                f"Failed to get balance from TradeManager, using CSV as final fallback: {tm_e}"
            )
            paper_wallet_balance = calculate_wallet_balance_from_csv()
            logger.info(
                f"Dashboard: Using CSV final fallback wallet balance: ${paper_wallet_balance:.2f}"
            )

    # Get open positions from TradeManager (single source of truth)
    try:
        # Load TradeManager state directly from LOG_DIR to bypass CWD issues
        import json
        state_file = LOG_DIR / "trade_manager_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)

            # Get positions from state file
            positions = state.get("positions", {})
            price_cache = state.get("price_cache", {})

            open_positions = []
            # Collect all symbols that need fresh prices
            symbols_to_fetch = []
            for symbol, pos_data in positions.items():
                if pos_data.get("total_amount", 0) > 0:  # Open position
                    current_price = price_cache.get(symbol, 0)
                    # If no cached price or price is stale, mark for fetching
                    if (
                        not current_price
                        or current_price == 0
                        or current_price == pos_data.get("average_price", 0)
                    ):
                        symbols_to_fetch.append(symbol)

            # Fetch prices for all symbols that need them, robustly per symbol
            fresh_prices = {}
            if symbols_to_fetch:
                logger.info(
                    f"Fetching prices for {len(symbols_to_fetch)} symbols"
                )
                for sym in symbols_to_fetch:
                    try:
                        price_val = fetch_current_price_for_symbol(sym)
                        if price_val and price_val > 0:
                            fresh_prices[sym] = price_val
                            logger.debug(
                                f"Fetched fresh price for {sym}: ${price_val}"
                            )
                    except Exception as e:
                        logger.debug(f"Failed to fetch price for {sym}: {e}")
                logger.info(f"Fetched {len(fresh_prices)} fresh prices")

            # Process positions with batched prices
            for symbol, pos_data in positions.items():
                if pos_data.get("total_amount", 0) > 0:  # Open position
                    # Try to get current price from cache first
                    current_price = price_cache.get(symbol, 0)

                    # If no cached price or price is stale, use fresh price from batch
                    if (
                        not current_price
                        or current_price == 0
                        or current_price == pos_data.get("average_price", 0)
                    ):
                        fresh_price = fresh_prices.get(symbol)
                        if fresh_price and fresh_price > 0:
                            current_price = fresh_price
                            logger.info(
                                f"Using fresh price for {symbol}: ${current_price}"
                            )
                        else:
                            current_price = pos_data.get("average_price", 0)
                            logger.warning(
                                f"No fresh price available for {symbol}, using entry price"
                            )

                    # Calculate PnL
                    amount = pos_data["total_amount"]
                    avg_price = pos_data["average_price"]
                    side = pos_data["side"]

                    if side == "long":
                        pnl = (current_price - avg_price) * amount
                    else:  # short
                        pnl = (avg_price - current_price) * amount

                    pnl_pct = (
                        (pnl / (avg_price * amount)) * 100
                        if avg_price > 0
                        else 0
                    )

                    # Calculate additional fields for position cards
                    current_value = (
                        float(current_price * amount) if current_price else 0.0
                    )
                    pnl_value = float(pnl)

                    # Generate chart data bounds (will be used by JavaScript)
                    chart_min = (
                        min(current_price, avg_price) * 0.95
                    )  # 5% below minimum
                    chart_max = (
                        max(current_price, avg_price) * 1.05
                    )  # 5% above maximum

                    # Calculate trend strength and R-squared based on P&L
                    trend_strength = (
                        "strong"
                        if abs(pnl_pct) > 2
                        else "moderate" if abs(pnl_pct) > 1 else "weak"
                    )
                    r_squared = min(99.9, max(60.0, 70.0 + abs(pnl_pct) * 2))

                    position_data = {
                        "symbol": symbol,
                        "side": side,
                        "size": float(
                            amount
                        ),  # Use 'size' for consistency with template
                        "entry_price": float(avg_price),
                        "current_price": float(current_price),
                        "pnl": float(pnl_pct),  # PnL percentage
                        "pnl_value": pnl_value,  # PnL dollar amount
                        "current_value": current_value,
                        "entry_time": pos_data.get("entry_time", ""),
                        "position_value": current_value,  # Keep for backward compatibility
                        "chart_min": chart_min,
                        "chart_max": chart_max,
                        "trend_strength": trend_strength,
                        "r_squared": r_squared,
                        # Include stop loss/trailing stop price for chart line
                        "stop_price": (
                            float(pos_data.get("stop_loss_price"))
                            if pos_data.get("stop_loss_price") is not None
                            else None
                        ),
                    }
                    open_positions.append(position_data)

            # Cross-check against CSV-derived open trades to eliminate stale entries
            try:
                from crypto_bot.utils.open_trades import get_open_trades
                csv_opens = get_open_trades(TRADE_FILE)
                csv_symbols = {o.get("symbol") for o in csv_opens}
                if csv_symbols:
                    before = len(open_positions)
                    open_positions = [p for p in open_positions if p.get("symbol") in csv_symbols]
                    logger.info(
                        f"Dashboard: CSV cross-check filtered positions from {before} to {len(open_positions)}"
                    )
            except Exception as _csv_err:
                logger.warning(f"Dashboard: CSV cross-check failed: {_csv_err}")

            logger.info(f"Dashboard: Loaded {len(open_positions)} positions from state file")
        else:
            open_positions = get_open_positions()
    except Exception as e:
        logger.warning(
            f"Failed to get positions from state file for dashboard: {e}, falling back to legacy method"
        )
        open_positions = get_open_positions()

    # Deduplicate positions to prevent duplicate cards
    open_positions = deduplicate_positions(open_positions)
    logger.info(
        f"Dashboard: After deduplication, {len(open_positions)} unique positions"
    )

    # Calculate available balance (wallet balance minus value of open positions)
    available_balance = get_available_balance(open_positions)

    # Calculate open position balance (value of all open positions)
    open_position_balance = 0.0
    try:
        for position in open_positions:
            # Use position_value if available (from API), otherwise calculate it
            if (
                "position_value" in position
                and position["position_value"] is not None
            ):
                open_position_balance += position["position_value"]
            elif position.get("current_price") and position.get("amount"):
                # Position value = current_price * amount (fallback calculation)
                open_position_balance += (
                    position["current_price"] * position["amount"]
                )
    except Exception as e:
        logger.warning(f"Failed to calculate open position balance: {e}")
        open_position_balance = 0.0

    # Calculate total PnL using state file if available
    try:
        # Load TradeManager state directly from file to bypass singleton issues
        import json
        from pathlib import Path

        state_file = Path("crypto_bot/logs/trade_manager_state.json")
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)

            # Calculate P&L from state file data
            positions = state.get("positions", {})
            stats = state.get("statistics", {})
            price_cache = state.get("price_cache", {})

            total_unrealized_pnl = 0.0
            for symbol, pos_data in positions.items():
                if pos_data.get("total_amount", 0) > 0:  # Open position
                    current_price = price_cache.get(
                        symbol, pos_data.get("average_price", 0)
                    )
                    if current_price and pos_data.get("average_price"):
                        amount = pos_data["total_amount"]
                        avg_price = pos_data["average_price"]
                        side = pos_data["side"]

                        if side == "long":
                            pnl = (current_price - avg_price) * amount
                        else:  # short
                            pnl = (avg_price - current_price) * amount

                        total_unrealized_pnl += pnl

            realized_pnl = float(stats.get("total_realized_pnl", 0))

            # Calculate P&L from balance to ensure consistency
            calculated_pnl = paper_wallet_balance - 10000.0

            pnl_data = {
                "total_pnl": calculated_pnl,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": calculated_pnl - realized_pnl,
                "initial_balance": 10000.0,
                "balance": paper_wallet_balance,
            }
            initial_balance = 10000.0
            total_pnl = calculated_pnl  # Use the calculated P&L
            logger.info(f"Dashboard P&L from state file: ${total_pnl:.2f}")
            print(f"DEBUG: P&L calculated as ${total_pnl:.2f}")
        else:
            # Fallback to legacy calculation
            pnl_data = calculate_wallet_pnl()
            total_pnl = float(pnl_data.get("total_pnl", 0.0) or 0.0)
            initial_balance = float(
                pnl_data.get("initial_balance", 10000.0) or 10000.0
            )
    except Exception as e:
        logger.warning(
            f"Failed to get dashboard P&L from state file: {e}, falling back to legacy calculation"
        )
        # Fallback to legacy calculation
        pnl_data = calculate_wallet_pnl()
        total_pnl = float(pnl_data.get("total_pnl", 0.0) or 0.0)
        initial_balance = float(
            pnl_data.get("initial_balance", 10000.0) or 10000.0
        )

    # Get recent regimes
    regimes = utils.get_recent_regimes(REGIME_FILE)

    # Add timestamp for aggressive cache busting to prevent chart cycling
    import time

    cache_bust = int(time.time() * 1000)  # Milliseconds for more uniqueness

    # Prepare template data
    template_data = {
        "running": is_running(),
        "mode": mode,
        "uptime": get_uptime(),
        "last_trade": utils.get_last_trade(TRADE_FILE),
        "regime": utils.get_current_regime(LOG_FILE),
        "last_reason": utils.get_last_decision_reason(LOG_FILE),
        "pnl": total_pnl,
        "performance": perf,
        "allocation": allocation,
        "paper_wallet_balance": paper_wallet_balance,
        "initial_balance": initial_balance,
        "available_balance": available_balance,
        "open_position_balance": open_position_balance,
        "open_positions": open_positions,
        "pnl_data": pnl_data,  # Pass the full PnL data for JavaScript
        "cache_bust": cache_bust,
        "regimes": regimes,
    }

    # Cache the data for future requests
    set_cached_dashboard_data(template_data)

    return render_template("dashboard.html", **template_data)


@app.route("/model")
def model_page():
    report = {}
    if MODEL_REPORT.exists():
        with open(MODEL_REPORT) as f:
            report = json.load(f)
    return render_template("model.html", report=report)


@app.route("/train_model", methods=["POST"])
def train_model_route():
    file = request.files.get("csv")
    if file:
        tmp_path = LOG_DIR / "upload.csv"
        file.save(tmp_path)
        ml.train_from_csv(tmp_path)
        tmp_path.unlink()
    return redirect(url_for("model_page"))


@app.route("/validate_model", methods=["POST"])
def validate_model_route():
    file = request.files.get("csv")
    tmp_path = None
    if file:
        tmp_path = LOG_DIR / "validate.csv"
        file.save(tmp_path)
        metrics = ml.validate_from_csv(tmp_path)
        tmp_path.unlink()
    else:
        default_csv = LOG_DIR / "trades.csv"
        if default_csv.exists():
            metrics = ml.validate_from_csv(default_csv)
        else:
            metrics = ml.validate_from_csv(default_csv)
    if metrics:
        MODEL_REPORT.write_text(json.dumps(metrics))
    return redirect(url_for("model_page"))


@app.route("/api_config")
def api_config_page():
    """API configuration page."""
    # Load current API configuration
    api_config = {}
    user_config_file = Path("crypto_bot/user_config.yaml")
    if user_config_file.exists():
        with open(user_config_file) as f:
            api_config = yaml.safe_load(f) or {}

    return render_template("api_config.html", api_config=api_config)


@app.route("/monitoring")
def monitoring_page():
    """Monitoring dashboard page."""
    return render_template("monitoring.html")


@app.route("/logs")
def logs_page():
    """System logs dashboard page."""
    return render_template("logs.html")


@app.route("/config_settings")
def config_settings_page():
    """General configuration settings page."""
    # Load current configuration
    config_data = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config_data = yaml.safe_load(f) or {}

    return render_template("config_settings.html", config_data=config_data)


@app.route("/api/save_api_config", methods=["POST"])
def save_api_config():
    """Save API configuration."""
    try:
        data = request.get_json()
        user_config_file = Path("crypto_bot/user_config.yaml")

        # Load existing config
        current_config = {}
        if user_config_file.exists():
            with open(user_config_file) as f:
                current_config = yaml.safe_load(f) or {}

        # Update with new values
        current_config.update(data)

        # Save back to file
        with open(user_config_file, "w") as f:
            yaml.dump(current_config, f, default_flow_style=False)

        return jsonify(
            {
                "status": "success",
                "message": "API configuration saved successfully",
            }
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error saving configuration: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/save_config_settings", methods=["POST"])
def save_config_settings():
    """Save general configuration settings."""
    try:
        data = request.get_json()

        # Load existing config
        current_config = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                current_config = yaml.safe_load(f) or {}

        # Update with new values (merge nested structures)
        def deep_merge(d1, d2):
            for key, value in d2.items():
                if (
                    key in d1
                    and isinstance(d1[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_merge(d1[key], value)
                else:
                    d1[key] = value
            return d1

        updated_config = deep_merge(current_config, data)

        # Save back to file
        with open(CONFIG_FILE, "w") as f:
            yaml.dump(updated_config, f, default_flow_style=False)

        return jsonify(
            {
                "status": "success",
                "message": "Configuration saved successfully",
            }
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error saving configuration: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/refresh_config", methods=["POST"])
def refresh_config():
    """Refresh configuration by reloading from files."""
    try:
        # Send reload command to running bot if it exists
        if check_existing_bot():
            # LOG_DIR already imported above
            control_file = LOG_DIR / "bot_control.json"
            with open(control_file, "w") as f:
                json.dump({"command": "reload"}, f)
            return jsonify(
                {"status": "success", "message": "Reload command sent to bot"}
            )
        else:
            # Bot not running, just return success
            return jsonify(
                {
                    "status": "success",
                    "message": "Configuration refreshed successfully",
                }
            )
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error refreshing configuration: {str(e)}",
                }
            ),
            500,
        )


@app.route("/api/refresh-dashboard", methods=["POST"])
def refresh_dashboard():
    """API endpoint to refresh dashboard data."""
    try:
        # Force refresh of TradeManager data
        from crypto_bot.utils.trade_manager import get_trade_manager

        trade_manager = get_trade_manager()

        # Save current state to ensure data is persisted
        trade_manager.save_state()

        logger.info("Dashboard refresh requested - TradeManager state saved")

        return jsonify(
            {
                "success": True,
                "message": "Dashboard data refreshed successfully",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error refreshing dashboard: {e}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            500,
        )


@app.route("/trades")
def trades_page():
    return render_template("trades.html")


@app.route("/trades_data")
def trades_data():
    """Return structured trade data for the trades page."""
    try:
        from crypto_bot import log_reader
        from crypto_bot.utils.open_trades import get_open_trades
        
        # Read all trades from CSV
        df = log_reader._read_trades(TRADE_FILE)
        trades = []
        
        if not df.empty:
            # Convert all trades to structured format
            for _, row in df.iterrows():
                try:
                    trade = {
                        "id": f"{row['symbol']}_{row['timestamp']}",
                        "symbol": row["symbol"],
                        "side": row["side"], 
                        "type": row["side"],  # For compatibility
                        "amount": float(row["amount"]) if row["amount"] else 0.0,
                        "quantity": float(row["amount"]) if row["amount"] else 0.0,  # For compatibility
                        "price": float(row["price"]) if row["price"] else 0.0,
                        "execution_price": float(row["price"]) if row["price"] else 0.0,  # For compatibility
                        "timestamp": row["timestamp"],
                        "date": row["timestamp"],  # For compatibility
                        "status": "completed",  # Historical trades are completed
                        "pnl": 0.0,  # Will calculate for closed positions
                        "pnl_percentage": 0.0,
                        "unrealized_pnl": 0.0,
                        "unrealized_pnl_percentage": 0.0,
                        "current_price": 0.0
                    }
                    trades.append(trade)
                except Exception as e:
                    logger.warning(f"Error processing trade row: {e}")
                    continue
        
        # Calculate realized P&L for completed trades using FIFO method
        def calculate_realized_pnl(trades_list):
            """Calculate realized P&L for completed trades using FIFO method."""
            positions = {}  # symbol -> list of (buy_price, amount, timestamp)
            
            for trade in trades_list:
                symbol = trade["symbol"]
                side = trade["side"]
                amount = trade["amount"]
                price = trade["price"]
                
                if symbol not in positions:
                    positions[symbol] = []
                
                if side == "buy" or side == "long":
                    # Add to position
                    positions[symbol].append((price, amount, trade["timestamp"]))
                elif side == "sell" or side == "short":
                    # Calculate realized P&L by matching with buys (FIFO)
                    remaining_sell = amount
                    realized_pnl = 0.0
                    
                    while remaining_sell > 0 and positions[symbol]:
                        buy_price, buy_amount, _ = positions[symbol][0]
                        
                        if buy_amount <= remaining_sell:
                            # Use entire buy position
                            realized_pnl += (price - buy_price) * buy_amount
                            remaining_sell -= buy_amount
                            positions[symbol].pop(0)
                        else:
                            # Partial sell
                            realized_pnl += (price - buy_price) * remaining_sell
                            positions[symbol][0] = (buy_price, buy_amount - remaining_sell, positions[symbol][0][2])
                            remaining_sell = 0
                    
                    # Update trade with realized P&L
                    if amount > 0:
                        trade["pnl"] = realized_pnl
                        trade["pnl_percentage"] = (realized_pnl / (price * amount)) * 100
            
            return trades_list
        
        # Sort trades chronologically before P&L calculation
        trades.sort(key=lambda x: x.get("timestamp", ""))
        
        # Apply P&L calculation
        trades = calculate_realized_pnl(trades)
        
        # Get open trades and add current prices/PnL
        open_trades = get_open_trades(TRADE_FILE)
        current_prices = {}
        
        # Get current prices for open positions
        try:
            for open_trade in open_trades:
                symbol = open_trade["symbol"]
                if symbol not in current_prices:
                    price = get_current_price_for_symbol(symbol)
                    if price > 0:
                        current_prices[symbol] = price
        except Exception as e:
            logger.warning(f"Error getting current prices: {e}")
        
        # Update trades with current prices and calculate PnL for open positions
        open_symbols = {trade["symbol"] for trade in open_trades}
        
        # Calculate position averages for open positions
        open_position_averages = {}
        for symbol in open_symbols:
            symbol_trades = [t for t in trades if t["symbol"] == symbol]
            buy_total_cost = 0
            buy_total_amount = 0
            
            for trade in symbol_trades:
                if trade["side"] == "buy":
                    buy_total_cost += trade["price"] * trade["amount"]
                    buy_total_amount += trade["amount"]
                elif trade["side"] == "sell":
                    # Reduce the position
                    sell_amount = trade["amount"]
                    if buy_total_amount >= sell_amount:
                        # Calculate average price for the sold portion
                        avg_price = buy_total_cost / buy_total_amount if buy_total_amount > 0 else 0
                        buy_total_cost -= avg_price * sell_amount
                        buy_total_amount -= sell_amount
            
            if buy_total_amount > 0:
                open_position_averages[symbol] = {
                    "average_price": buy_total_cost / buy_total_amount,
                    "total_amount": buy_total_amount
                }
        
        for trade in trades:
            symbol = trade["symbol"]
            
            # Add current price if available
            if symbol in current_prices:
                trade["current_price"] = current_prices[symbol]
            
            # Mark open positions and calculate unrealized PnL
            if symbol in open_symbols and trade["side"] == "buy":
                trade["status"] = "active"
                
                # Calculate unrealized PnL for open positions using position average
                if symbol in open_position_averages:
                    avg_entry_price = open_position_averages[symbol]["average_price"]
                    current_price = trade["current_price"]
                    position_amount = open_position_averages[symbol]["total_amount"]
                    
                    if current_price > 0 and avg_entry_price > 0 and position_amount > 0:
                        # Calculate unrealized P&L for the entire position
                        unrealized_pnl = (current_price - avg_entry_price) * position_amount
                        unrealized_pnl_percentage = (unrealized_pnl / (avg_entry_price * position_amount)) * 100
                        
                        # Only assign unrealized P&L to the most recent buy trade for this symbol
                        symbol_buys = [t for t in trades if t["symbol"] == symbol and t["side"] == "buy"]
                        if symbol_buys and trade == symbol_buys[-1]:
                            trade["unrealized_pnl"] = unrealized_pnl
                            trade["unrealized_pnl_percentage"] = unrealized_pnl_percentage
        
        # Sort trades by timestamp (most recent first)
        trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Return the most recent 200 trades to avoid overwhelming the frontend
        return jsonify(trades[-200:] if len(trades) > 200 else trades)
        
    except Exception as e:
        logger.error(f"Error in trades_data endpoint: {e}")
        return jsonify([])


@app.route("/trades_tail")
def trades_tail():
    trades = ""
    if TRADE_FILE.exists():
        trades = "\n".join(TRADE_FILE.read_text().splitlines()[-100:])
    errors = ""
    if ERROR_FILE.exists():
        errors = "\n".join(ERROR_FILE.read_text().splitlines()[-100:])
    return jsonify({"trades": trades, "errors": errors})


@app.route("/api/current-prices")
def api_current_prices():
    """Return current market prices for symbols with fresh data."""
    try:
        # Check if historical data is requested
        include_history = request.args.get("history", "").lower() == "true"
        symbol_filter = request.args.get("symbol")

        # Return real price history if requested
        if include_history and symbol_filter:
            candle_data = generate_candle_data(symbol_filter, 24)
            if candle_data and len(candle_data) > 0:
                price_history = []
                for candle in candle_data:
                    price_history.append(
                        {
                            "timestamp": candle["timestamp"],
                            "price": candle["close"],
                        }
                    )

                return jsonify(
                    {
                        "symbol": symbol_filter,
                        "include_history": include_history,
                        "history": {
                            "symbol": symbol_filter,
                            "timeframe": "5m",
                            "data": price_history,
                            "mock": False,
                        },
                    }
                )
            else:
                return (
                    jsonify(
                        {
                            "error": "No price history data available",
                            "symbol": symbol_filter,
                            "include_history": include_history,
                        }
                    ),
                    404,
                )

        # Get symbols from active positions in TradeManager
        current_prices = {}
        try:
            from crypto_bot.utils.trade_manager import get_trade_manager
            trade_manager = get_trade_manager()
            positions = trade_manager.get_all_positions()
            symbols = [pos.symbol for pos in positions if pos.is_open and pos.total_amount > 0]
            
            # Force refresh prices for all active positions
            for symbol in symbols:
                try:
                    # Always fetch fresh price, ignore cache
                    fresh_price = get_current_price_for_symbol(symbol)
                    if fresh_price and fresh_price > 0:
                        current_prices[symbol] = fresh_price
                        # Update TradeManager cache with fresh price
                        trade_manager.update_price(symbol, fresh_price)
                        logger.debug(f"Refreshed price for {symbol}: ${fresh_price}")
                    else:
                        # Fallback to cached price if fresh fetch fails
                        cached_price = trade_manager.price_cache.get(symbol)
                        if cached_price:
                            current_prices[symbol] = float(cached_price)
                            logger.warning(f"Using cached price for {symbol}: ${cached_price}")
                except Exception as e:
                    logger.error(f"Error refreshing price for {symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error accessing TradeManager for price refresh: {e}")
            # Fallback to CSV-based approach
            df = log_reader._read_trades(TRADE_FILE)
            if not df.empty:
                symbols = df["symbol"].unique().tolist()
                for symbol in symbols:
                    try:
                        price = get_current_price_for_symbol(symbol)
                        if price > 0:
                            current_prices[symbol] = price
                    except Exception as e:
                        logger.error(f"Error getting price for {symbol}: {e}")
                        continue

        logger.info(f"Returning current prices for {len(current_prices)} symbols")
        return jsonify(current_prices)
    except Exception as e:
        logger.error(f"Error in api_current_prices: {e}")
        return jsonify({"error": str(e)})


@app.route("/api/test-route")
def api_test_route():
    """Simple test route to check if routing is working."""
    import time

    return jsonify(
        {"message": "Test route works!", "timestamp": int(time.time() * 1000)}
    )


# Price history API removed - no chart functionality


def get_latest_candle_timestamp(symbol):
    """Get the timestamp of the most recent 5-minute candle for a symbol."""
    try:
        # Try to get real market data first
        import asyncio
        import time

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Import the enhanced OHLCV fetcher
            from crypto_bot.utils.enhanced_ohlcv_fetcher import (
                EnhancedOHLCVFetcher,
            )
            from crypto_bot.execution.cex_executor import get_exchange
            import yaml
            from pathlib import Path

            # Load user configuration to get the correct exchange
            user_config_path = (
                Path(__file__).resolve().parent.parent
                / "crypto_bot"
                / "user_config.yaml"
            )
            if user_config_path.exists():
                with open(user_config_path, "r") as f:
                    user_config = yaml.safe_load(f)
                exchange_name = user_config.get("exchange", "kraken")
            else:
                # Fallback to kraken if no user config found
                exchange_name = "kraken"

            # Create exchange instance with correct configuration
            config = {
                "exchange": exchange_name,
                "max_concurrent_ohlcv": 3,
                "max_concurrent_dex_ohlcv": 10,
                "min_volume_usd": 0,
            }
            exchange, _ = get_exchange(config)

            # Create fetcher instance
            fetcher = EnhancedOHLCVFetcher(exchange, config)

            # Normalize symbol for specific exchanges (e.g., Kraken uses XBT instead of BTC)
            try:
                ex_id = getattr(exchange, "id", "").lower()
            except Exception:
                ex_id = ""
            normalized_symbol = symbol
            if ex_id == "kraken":
                if isinstance(symbol, str):
                    normalized_symbol = symbol.replace("BTC/", "XBT/").replace(
                        "/BTC", "/XBT"
                    )

            # Fetch just the most recent candle
            cex_data, dex_data = loop.run_until_complete(
                fetcher.fetch_ohlcv_batch([normalized_symbol], "5m", 1)
            )
            # Combine CEX and DEX data for frontend display
            data_map = {**cex_data, **dex_data}

            # Prefer normalized symbol key if present
            symbol_key = (
                normalized_symbol if normalized_symbol in data_map else symbol
            )

            if symbol_key in data_map and data_map[symbol_key]:
                raw_data = data_map[symbol_key]

                if isinstance(raw_data, list) and len(raw_data) > 0:
                    # Get the most recent candle timestamp
                    latest_candle = raw_data[-1]  # Most recent candle
                    if len(latest_candle) >= 1:
                        return int(latest_candle[0])  # Return timestamp

        except Exception as e:
            print(f"Failed to fetch real candle timestamp for {symbol}: {e}")
        finally:
            loop.close()

    except Exception as e:
        print(f"Error getting candle timestamp for {symbol}: {e}")

    # Fallback: return current time rounded to nearest 5-minute interval
    current_time = int(time.time())
    # Round down to nearest 5-minute boundary
    return (current_time // 300) * 300


@app.route("/api/sell-position", methods=["POST"])
def api_sell_position():
    """Sell a position via market order."""
    try:
        data = request.get_json()
        if not data:
            return (
                jsonify({"success": False, "error": "No data provided"}),
                400,
            )

        symbol = data.get("symbol")
        amount = data.get("amount")

        if not symbol or not amount:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Symbol and amount are required",
                    }
                ),
                400,
            )

        print(f"API: Received sell request for {amount} {symbol}")
        try:
            # Debug: Check TradeManager state
            from crypto_bot.utils.trade_manager import get_trade_manager
            trade_manager = get_trade_manager()
            print(f"API: TradeManager instance: {id(trade_manager)}")
            positions = trade_manager.get_all_positions()
            print(f"API: TradeManager has {len(positions)} open positions")
            for pos in positions:
                if pos.symbol == symbol:
                    print(f"API: Found position {symbol}: side={pos.side}, amount={pos.total_amount}")
                    break
            else:
                print(f"API: Position {symbol} not found in TradeManager")

            # Write debug info to file
            with open("/tmp/sell_debug.log", "a") as f:
                f.write(f"Received sell request for {amount} {symbol}\n")
                f.write(f"TradeManager instance: {id(trade_manager)}\n")
                f.write(f"Open positions: {len(positions)}\n")
                for pos in positions:
                    if pos.symbol == symbol:
                        f.write(f"Found position: {pos.symbol} {pos.side} {pos.total_amount}\n")
                        break
            # Check for duplicate sell requests to prevent accumulation
            import json as _json
            from datetime import datetime as _dt, timedelta
            sell_requests_file = LOG_DIR / 'sell_requests.json'
            requests = []
            if sell_requests_file.exists():
                try:
                    with open(sell_requests_file, 'r') as _f:
                        requests = _json.load(_f)
                        if not isinstance(requests, list):
                            requests = []
                except Exception:
                    requests = []

            # Filter out duplicate requests for the same symbol (only keep the most recent)
            filtered_requests = []
            symbol_found = False
            for req in requests:
                if req.get('symbol') == symbol:
                    if not symbol_found:
                        # Keep only the most recent request for this symbol
                        filtered_requests.append({
                            'symbol': symbol,
                            'amount': float(amount),
                            'timestamp': _dt.utcnow().isoformat()
                        })
                        symbol_found = True
                else:
                    filtered_requests.append(req)

            # If this symbol wasn't in the existing requests, add it
            if not symbol_found:
                filtered_requests.append({
                    'symbol': symbol,
                    'amount': float(amount),
                    'timestamp': _dt.utcnow().isoformat()
                })

            sell_requests_file.parent.mkdir(parents=True, exist_ok=True)
            with open(sell_requests_file, 'w') as _f:
                _json.dump(filtered_requests, _f, indent=2)
            logger.info(f"API: Added/updated sell request for {symbol} (prevented duplicates)")
        except Exception as _e:
            logger.warning(f"API: Failed to process sell request for {symbol}: {_e}")

        # IMMEDIATE EXECUTION: Try to sell the position right away
        position_locked = False
        try:
            from decimal import Decimal
            from crypto_bot.utils.trade_manager import (
                get_trade_manager,
                create_trade,
            )

            trade_manager = get_trade_manager()

            # Check if position is already being processed (basic lock mechanism)
            if hasattr(trade_manager, '_sell_locks'):
                if symbol in trade_manager._sell_locks:
                    return jsonify({
                        "success": False,
                        "error": f"Position {symbol} is already being sold. Please wait.",
                    }), 409
                trade_manager._sell_locks.add(symbol)
                position_locked = True

            # Find the position to sell
            positions = trade_manager.get_all_positions()
            position_to_sell = None

            for pos in positions:
                if pos.symbol == symbol and pos.is_open:
                    position_to_sell = pos
                    break

            if position_to_sell:
                print(f"API: Starting immediate execution for {symbol}")
                # IMMEDIATE EXECUTION: Close position right away
                try:
                    # Determine close amount: force full close to avoid precision residuals
                    requested_amount = Decimal(str(amount))
                    total_amt = position_to_sell.total_amount
                    if total_amt > 0 and requested_amount >= (total_amt * Decimal("0.99999999")):
                        close_amount = total_amt
                    else:
                        # For now enforce full close to prevent reappearing cards due to dust
                        close_amount = total_amt

                    # Get current market price for the symbol
                    try:
                        current_price = get_current_price(symbol)
                    except Exception:
                        current_price = float(position_to_sell.average_price)

                    if not current_price or current_price <= 0:
                        current_price = float(position_to_sell.average_price)

                    # Create and record a market sell (or buy if short) to close position
                    side = "sell" if position_to_sell.side == "long" else "buy"
                    print(f"API: Creating {side} trade for {close_amount} {symbol} at ${current_price}")
                    trade = create_trade(
                        symbol=symbol,
                        side=side,
                        amount=Decimal(str(close_amount)),
                        price=Decimal(str(current_price)),
                        strategy="manual_close_immediate",
                        exchange="paper",  # backend default
                        order_id=f"ui_close_{symbol}_{int(datetime.utcnow().timestamp())}",
                        client_order_id=f"ui_close_{symbol}_{int(datetime.utcnow().timestamp())}",
                    )

                    print(f"API: Recording trade in TradeManager")
                    try:
                        trade_manager.record_trade(trade)
                        print(f"API: Trade recorded successfully")
                    except Exception as record_err:
                        print(f"API: Failed to record trade: {record_err}")
                        raise

                    try:
                        trade_manager.save_state()
                        print(f"API: State saved successfully")
                    except Exception as save_err:
                        print(f"API: Failed to save state: {save_err}")
                        raise

                    # Calculate PnL for response
                    entry_price = float(position_to_sell.average_price)
                    pnl = (current_price - entry_price) * float(close_amount) * (1 if position_to_sell.side == "long" else -1)

                    print(f"API: IMMEDIATE EXECUTION: Closed {symbol} position with PnL ${pnl:.2f}")
                    logger.info(f"IMMEDIATE EXECUTION: Closed {symbol} position with PnL ${pnl:.2f}")

                    # Remove position from positions.log to prevent sync service from re-adding it
                    try:
                        from pathlib import Path
                        positions_log_path = LOG_DIR / "positions.log"
                        if positions_log_path.exists():
                            # Read and filter out the closed symbol
                            entries = []
                            with open(positions_log_path, 'r') as f:
                                for line in f:
                                    line = line.strip()
                                    if line and 'Active' in line:
                                        # Parse the line to check if it's the symbol we just closed
                                        parts = line.split()
                                        line_symbol = None
                                        for i, part in enumerate(parts):
                                            if part == 'Active' and i + 1 < len(parts):
                                                line_symbol = parts[i + 1]
                                                break
                                        # Keep the line if it's not the symbol we just closed
                                        if line_symbol != symbol:
                                            entries.append(line)
                                    else:
                                        # Keep non-position lines (comments, balance entries, etc.)
                                        entries.append(line)

                            # Write back to file
                            with open(positions_log_path, 'w') as f:
                                for entry in entries:
                                    f.write(entry + "\n")

                            logger.info(f"API: Removed {symbol} from positions.log after immediate closure")
                    except Exception as log_err:
                        logger.warning(f"API: Failed to remove {symbol} from positions.log: {log_err}")

                    # Return success response
                    return jsonify({
                        "success": True,
                        "message": f"Successfully closed {symbol} position",
                        "pnl": round(pnl, 2),
                        "close_price": current_price,
                        "amount_closed": float(close_amount)
                    })

                except Exception as close_error:
                    print(f"API: IMMEDIATE EXECUTION failed for {symbol}: {close_error}")
                    logger.error(f"IMMEDIATE EXECUTION failed for {symbol}: {close_error}")
                    # Continue to fallback method
                    if position_locked:
                        trade_manager._sell_locks.discard(symbol)

            # Fallback: try to close directly in state file to reconcile UI
            try:
                import json as _json
                from decimal import Decimal as _D
                state_path = LOG_DIR / "trade_manager_state.json"
                if state_path.exists():
                    with open(state_path, "r") as _f:
                        _state = _json.load(_f)
                    pos = _state.get("positions", {}).get(symbol)
                    if pos and float(pos.get("total_amount", 0)) > 0:
                        # Record a synthetic closing trade and zero out amount
                        close_amt = _D(str(pos["total_amount"]))
                        close_price = float(get_current_price(symbol) or pos.get("average_price", 0) or 0)
                        if close_price <= 0:
                            close_price = float(pos.get("average_price", 0) or 0)
                        _state.setdefault("trades", []).append({
                            "id": "manual-ui-close-fallback",
                            "symbol": symbol,
                            "side": "buy" if pos.get("side") == "short" else "sell",
                            "amount": float(close_amt),
                            "price": float(close_price),
                            "timestamp": datetime.utcnow().isoformat(),
                            "strategy": "manual_close_fallback",
                            "exchange": "paper",
                            "fees": 0.0,
                            "status": "filled",
                            "order_id": None,
                            "client_order_id": None,
                            "metadata": {},
                        })
                        _state["positions"][symbol]["total_amount"] = 0.0
                        with open(state_path, "w") as _f:
                            _json.dump(_state, _f, indent=2)
                        logger.warning(
                            f"API: Fallback-closed {symbol} directly in state file to resolve UI inconsistency"
                        )
                        return jsonify({"success": True, "message": f"Closed {symbol}"})
            except Exception as _fallback_err:
                logger.error(f"API: Fallback state close failed for {symbol}: {_fallback_err}")
            return (
                jsonify(
                    {
                        "success": False,
                        "error": f"No open position found for {symbol}",
                    }
                ),
                404,
            )

            # Determine close amount: force full close to avoid precision residuals
            try:
                requested_amount = Decimal(str(amount))
            except Exception:
                requested_amount = position_to_sell.total_amount

            # If requested ~ full size, snap to full; otherwise still close full by default
            total_amt = position_to_sell.total_amount
            if total_amt > 0 and requested_amount >= (total_amt * Decimal("0.99999999")):
                close_amount = total_amt
            else:
                # For now enforce full close to prevent reappearing cards due to dust
                close_amount = total_amt

            # Get current market price for the symbol
            try:
                current_price = get_current_price(symbol)
            except Exception:
                current_price = float(position_to_sell.average_price)

            if not current_price or current_price <= 0:
                current_price = float(position_to_sell.average_price)

            # Create and record a market sell (or buy if short) to close position
            side = "sell" if position_to_sell.side == "long" else "buy"
            trade = create_trade(
                symbol=symbol,
                side=side,
                amount=Decimal(str(close_amount)),
                price=Decimal(str(current_price)),
                strategy="manual_close",
                exchange="paper",  # backend default
            )

            trade_manager.record_trade(trade)

            print(
                f"API: Closed {close_amount} {symbol} at ${current_price} via {side}"
            )

            # Save the updated state
            trade_manager.save_state()

            # Remove position from positions.log to prevent sync service from re-adding it
            try:
                from pathlib import Path
                positions_log_path = LOG_DIR / "positions.log"
                if positions_log_path.exists():
                    # Read and filter out the closed symbol
                    entries = []
                    with open(positions_log_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and 'Active' in line:
                                # Parse the line to check if it's the symbol we just closed
                                parts = line.split()
                                line_symbol = None
                                for i, part in enumerate(parts):
                                    if part == 'Active' and i + 1 < len(parts):
                                        line_symbol = parts[i + 1]
                                        break
                                # Keep the line if it's not the symbol we just closed
                                if line_symbol != symbol:
                                    entries.append(line)
                            else:
                                # Keep non-position lines (comments, balance entries, etc.)
                                entries.append(line)

                    # Write back to file
                    with open(positions_log_path, 'w') as f:
                        for entry in entries:
                            f.write(entry + "\n")

                    logger.info(f"API: Removed {symbol} from positions.log after successful closure")
            except Exception as log_err:
                logger.warning(f"API: Failed to remove {symbol} from positions.log: {log_err}")

            # Verify closure persisted; if not, force-close as a safety net
            try:
                from decimal import Decimal as _D
                pos_after = trade_manager.positions.get(symbol)
                if pos_after and getattr(pos_after, "total_amount", 0) and pos_after.total_amount > 0:
                    logger.warning(
                        f"API: Position {symbol} still open after record_trade; forcing close to prevent UI reappearance"
                    )
                    pos_after.total_amount = _D("0")
                    trade_manager.positions[symbol] = pos_after
                    trade_manager.save_state()
            except Exception as _force_err:
                logger.error(f"API: Force-close verification failed for {symbol}: {_force_err}")

            # Also append to trades.csv to keep CSV-based fallbacks consistent
            try:
                from datetime import datetime as _dt
                import csv as _csv

                TRADE_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(TRADE_FILE, 'a', newline='') as _fh:
                    _writer = _csv.writer(_fh)
                    _writer.writerow([
                        symbol,
                        side,
                        float(close_amount),
                        float(current_price),
                        _dt.utcnow().isoformat(),
                        False,  # is_stop flag
                    ])
            except Exception as csv_err:
                logger.warning(f"Failed to append manual close to trades.csv: {csv_err}")

            return jsonify(
                {
                    "success": True,
                    "message": f"Successfully closed {symbol}",
                }
            )

        except Exception as e:
            print(f"API: Error selling position: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    except Exception as e:
        print(f"API: Error in sell-position endpoint: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# Compatibility alias in case clients post to '/sell-position' without the /api prefix
@app.route("/sell-position", methods=["POST"])
def api_sell_position_alias():
    return api_sell_position()


@app.route("/api/candle-timestamp")
def api_candle_timestamp():
    """Return the timestamp of the most recent 5-minute candle for a symbol."""
    try:
        symbol = request.args.get("symbol")
        if not symbol:
            return jsonify({"error": "Symbol parameter is required"})

        # Get the most recent 5-minute candle timestamp
        timestamp = get_latest_candle_timestamp(symbol)

        return jsonify(
            {"symbol": symbol, "timestamp": timestamp, "timeframe": "5m"}
        )

    except Exception as e:
        print(f"Error in api_candle_timestamp: {e}")
        return jsonify({"error": str(e)})


@app.route("/api/candle-data")
def api_candle_data():
    """Return candle data for a symbol."""
    try:
        symbol = request.args.get("symbol", "BTC/USD")
        limit = int(request.args.get("limit", 50))
        interval = request.args.get("interval", "5m")

        # Normalize symbol for specific exchanges in OHLCV fetch (e.g., Kraken XBT)
        try:
            from crypto_bot.execution.cex_executor import get_exchange

            exchange, _ = get_exchange({"exchange": None})
            ex_id = getattr(exchange, "id", "").lower()
        except Exception:
            ex_id = ""

        normalized_symbol = symbol

        # Apply BTC->XBT conversion when using Kraken OR as fallback for BTC/USD
        if isinstance(symbol, str) and "BTC" in symbol:
            if ex_id == "kraken":
                # Kraken-specific conversion
                normalized_symbol = symbol.replace("BTC/", "XBT/").replace("/BTC", "/XBT")
                logger.debug(f"Kraken BTC symbol conversion: {symbol} -> {normalized_symbol}")
            elif (ex_id in ("", "unknown") or not ex_id) and symbol == "BTC/USD":
                # Fallback for BTC/USD only when exchange detection fails
                normalized_symbol = "XBT/USD"
                logger.debug(f"Fallback BTC/USD conversion (exchange detection failed): {symbol} -> {normalized_symbol}")

        candle_data = generate_candle_data(normalized_symbol, limit)

        # If no data found, try XBT fallback for BTC symbols
        if (
            (not candle_data or len(candle_data) == 0)
            and isinstance(symbol, str)
            and "BTC" in symbol
        ):
            fallback_symbol = symbol.replace("BTC/", "XBT/").replace("/BTC", "/XBT")
            if (
                fallback_symbol != normalized_symbol
            ):  # Only try if different from what we already tried
                logger.info(
                    f"No data for {normalized_symbol}, trying XBT fallback {fallback_symbol}"
                )
                candle_data = generate_candle_data(fallback_symbol, limit)

        if candle_data and len(candle_data) > 0:
            return jsonify(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "candles": candle_data,
                    "success": True,
                }
            )
        else:
            return (
                jsonify(
                    {
                        "error": f"No candle data available for {symbol}",
                        "success": False,
                    }
                ),
                404,
            )

    except Exception as e:
        logger.error(f"Error in api_candle_data: {e}")
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/trend-data")
def api_trend_data():
    """Return 5-minute candle data for the last 100 candles with trend analysis."""
    try:
        symbol = request.args.get("symbol")
        if not symbol:
            return jsonify({"error": "Symbol parameter is required"})

        # Generate 5-minute candle data for the last 100 candles (500 minutes = ~8.3 hours)
        candle_data = generate_candle_data(symbol, 100)

        # Calculate trend line
        trend_data = calculate_trend_line(candle_data)

        return jsonify(
            {
                "symbol": symbol,
                "timeframe": "5m",
                "candle_count": len(candle_data),
                "candles": candle_data,
                "trend": trend_data,
                "generated_at": int(time.time() * 1000),
            }
        )

    except Exception as e:
        print(f"Error in api_trend_data: {e}")
        return jsonify({"error": str(e)})


@app.route("/api/live-signals")
def api_live_signals():
    """Return live trading signals for the dashboard."""
    try:
        # For now, return empty signals - can be extended later
        return jsonify({})
    except Exception as e:
        logger.error(f"Error in api_live_signals: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/positions")
def api_positions():
    """Return positions data (alias for /api/open-positions)."""
    try:
        # Simply redirect to the existing open-positions endpoint
        from flask import redirect, url_for

        return redirect(url_for("api_open_positions"))
    except Exception as e:
        logger.error(f"Error in api_positions: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/balance")
def api_balance():
    """Return account balance information."""
    try:
        # Load TradeManager state to get balance information
        import json
        from pathlib import Path

        state_file = Path("crypto_bot/logs/trade_manager_state.json")
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)

            # Get positions and calculate total balance
            positions = state.get("positions", {})
            total_position_value = 0.0

            for symbol, pos_data in positions.items():
                if pos_data.get("total_amount", 0) > 0:
                    # Try to get current price from price cache
                    price_cache = state.get("price_cache", {})
                    current_price = price_cache.get(
                        symbol, pos_data.get("average_price", 0)
                    )
                    amount = pos_data["total_amount"]
                    position_value = (
                        current_price * amount if current_price else 0.0
                    )
                    total_position_value += position_value

            # Calculate available balance (get from config or use default)
            import yaml
            config_path = Path("config.yaml")
            starting_balance = 10000.0  # Default fallback
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)
                    starting_balance = config.get("risk", {}).get("starting_balance", 10000.0)
                except Exception as e:
                    logger.warning(f"Could not load starting balance from config: {e}")
            
            available_balance = starting_balance - total_position_value

            balance_data = {
                "total_balance": starting_balance,
                "available_balance": max(0, available_balance),
                "position_value": total_position_value,
                "currency": "USD",
                "timestamp": int(time.time() * 1000),
            }

            return jsonify(balance_data)
        else:
            return jsonify(
                {
                    "total_balance": 10000.0,
                    "available_balance": 10000.0,
                    "position_value": 0.0,
                    "currency": "USD",
                    "timestamp": int(time.time() * 1000),
                }
            )

    except Exception as e:
        logger.error(f"Error in api_balance: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/bot-status")
def api_bot_status():
    """Return bot status information."""
    try:
        # Check if bot is running by looking for process or status files
        import psutil

        bot_running = False
        try:
            # Check for bot process
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if (
                        proc.info["name"]
                        and "python" in proc.info["name"].lower()
                    ):
                        cmdline = proc.info["cmdline"]
                        if cmdline and len(cmdline) > 1:
                            if "main.py" in " ".join(
                                cmdline
                            ) or "crypto_bot" in " ".join(cmdline):
                                bot_running = True
                                break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.debug(f"Could not check process status: {e}")

        return jsonify(
            {
                "success": True,
                "data": {
                    "bot_running": bot_running,
                    "execution_running": True,  # Placeholder for execution status
                    "trading_active": True,  # Placeholder for trading status
                    "timestamp": int(time.time() * 1000),
                },
            }
        )
    except Exception as e:
        logger.error(f"Error in api_bot_status: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/bot/logs")
def api_bot_logs():
    """Return bot log data from various log files."""
    try:
        from pathlib import Path

        # Define log directory
        log_dir = Path(__file__).resolve().parents[1] / "crypto_bot" / "logs"

        # Define log files to read
        log_files = {
            "bot_main": ["bot.log", "bot_controller.log", "bot_monitor.log"],
            "bot_execution": ["execution.log", "advanced_orders.log"],
            "bot_trading": ["trading_monitor.log", "trades.csv"],
            "bot_performance": [
                "performance_monitor.log",
                "strategy_stats.json",
            ],
        }

        log_data = {}

        for log_type, files in log_files.items():
            log_data[log_type] = []

            for filename in files:
                log_file = log_dir / filename
                if log_file.exists():
                    try:
                        if filename.endswith(".json"):
                            # Handle JSON files
                            with open(log_file, "r") as f:
                                data = json.load(f)
                                log_data[log_type].extend(
                                    [
                                        f"JSON data from {filename}: {json.dumps(data, indent=2)}"
                                    ]
                                )
                        elif filename.endswith(".csv"):
                            # Handle CSV files - just show recent entries
                            with open(log_file, "r") as f:
                                lines = f.readlines()
                                # Show last 20 lines
                                recent_lines = (
                                    lines[-20:] if len(lines) > 20 else lines
                                )
                                log_data[log_type].extend(
                                    [
                                        f"{filename}: {line.strip()}"
                                        for line in recent_lines
                                    ]
                                )
                        else:
                            # Handle text log files
                            with open(log_file, "r") as f:
                                lines = f.readlines()
                                # Show last 50 lines to avoid too much data
                                recent_lines = (
                                    lines[-50:] if len(lines) > 50 else lines
                                )
                                log_data[log_type].extend(
                                    [line.strip() for line in recent_lines]
                                )
                    except Exception as e:
                        log_data[log_type].append(
                            f"Error reading {filename}: {str(e)}"
                        )

        return jsonify(
            {
                "success": True,
                "data": log_data,
                "timestamp": int(time.time() * 1000),
            }
        )
    except Exception as e:
        logger.error(f"Error in api_bot_logs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/dashboard-metrics")
def api_dashboard_metrics():
    """Return dashboard metrics including performance and allocation."""
    try:
        # Get performance data
        performance_data = get_performance_data()
        allocation_data = get_allocation_data()

        return jsonify(
            {
                "performance": performance_data,
                "allocation": allocation_data,
                "timestamp": int(time.time() * 1000),
            }
        )
    except Exception as e:
        logger.error(f"Error in api_dashboard_metrics: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/price-history")
def api_price_history():
    """Return 5-minute price history for the last 2 hours for trend chart."""
    print(f"DEBUG: api_price_history called with args: {request.args}")
    try:
        # Get symbol from query parameters
        symbol = request.args.get("symbol")
        if not symbol:
            return jsonify({"error": "Symbol parameter is required"})

        # Try to get real price history data
        candle_data = generate_candle_data(symbol, 24)
        if candle_data and len(candle_data) > 0:
            price_history = []
            for candle in candle_data:
                price_history.append(
                    {
                        "timestamp": candle["timestamp"],
                        "price": candle["close"],
                    }
                )

            return jsonify(
                {
                    "symbol": symbol,
                    "timeframe": "5m",
                    "data": price_history,
                    "mock": False,
                }
            )
        else:
            return (
                jsonify(
                    {
                        "error": "No price history data available",
                        "symbol": symbol,
                    }
                ),
                404,
            )

    except Exception as e:
        print(f"DEBUG: Error in api_price_history: {e}")
        return jsonify({"error": str(e)})


def generate_candle_data(symbol, limit):
    """Generate OHLCV candle data for the last N 5-minute intervals using real market data."""
    import time
    import asyncio
    import logging

    logger = logging.getLogger(__name__)

    try:
        # Try to get real market data first
        try:
            # Import the enhanced OHLCV fetcher
            from crypto_bot.utils.enhanced_ohlcv_fetcher import (
                EnhancedOHLCVFetcher,
            )
            from crypto_bot.execution.cex_executor import get_exchange

            # Create exchange instance
            config = {
                "max_concurrent_ohlcv": 3,
                "max_concurrent_dex_ohlcv": 10,
                "min_volume_usd": 0,
            }
            exchange, _ = get_exchange(config)

            # Create fetcher instance
            fetcher = EnhancedOHLCVFetcher(exchange, config)

            # Fetch real 5-minute OHLCV data
            logger.info(
                f"Fetching real 5-minute candle data for {symbol}, limit: {limit}"
            )

            # Use existing event loop if available, otherwise create new one
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, create a new thread to run the async code
                    import concurrent.futures
                    import threading

                    def run_async_fetch():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            result = new_loop.run_until_complete(
                                fetcher.fetch_ohlcv_batch([symbol], "5m", limit)
                            )
                            return result
                        finally:
                            new_loop.close()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_async_fetch)
                        cex_data, dex_data = future.result(timeout=30)
                else:
                    cex_data, dex_data = loop.run_until_complete(
                        fetcher.fetch_ohlcv_batch([symbol], "5m", limit)
                    )
            except RuntimeError:
                # No event loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    cex_data, dex_data = loop.run_until_complete(
                        fetcher.fetch_ohlcv_batch([symbol], "5m", limit)
                    )
                finally:
                    loop.close()
            # Combine CEX and DEX data for frontend display
            data_map = {**cex_data, **dex_data}

            if symbol in data_map and data_map[symbol]:
                raw_data = data_map[symbol]

                # Check if the result is an exception object (not actual data)
                if isinstance(raw_data, Exception):
                    logger.warning(
                        f"Failed to fetch real candle data for {symbol}: {type(raw_data).__name__}: {raw_data}"
                    )
                elif isinstance(raw_data, list) and len(raw_data) > 0:
                    logger.info(
                        f"Successfully fetched {len(raw_data)} real candles for {symbol}"
                    )

                    candles = []
                    for i, row in enumerate(raw_data):
                        if len(row) >= 6:
                            (
                                timestamp,
                                open_price,
                                high_price,
                                low_price,
                                close_price,
                                volume,
                            ) = row[:6]

                            candles.append(
                                {
                                    "timestamp": int(timestamp),
                                    "open": round(float(open_price), 6),
                                    "high": round(float(high_price), 6),
                                    "low": round(float(low_price), 6),
                                    "close": round(float(close_price), 6),
                                    "volume": round(float(volume), 2),
                                    "index": i,
                                }
                            )

                    if candles:
                        # Sort by timestamp (most recent first, as expected by the chart)
                        candles.sort(
                            key=lambda x: x["timestamp"], reverse=True
                        )
                        # Take the most recent 'limit' candles
                        candles = candles[:limit]
                        # Sort back to chronological order for trend calculation
                        candles.sort(key=lambda x: x["timestamp"])
                        logger.info(
                            f"Returning {len(candles)} real candles for {symbol}"
                        )
                        return candles
                else:
                    logger.warning(
                        f"Invalid data format for {symbol}: expected list, got {type(raw_data)}"
                    )

        except Exception as e:
            logger.warning(
                f"Failed to fetch real candle data for {symbol}: {e}"
            )
            # Will fall through to mock data generation
        finally:
            # Only close loop if it was created in this function
            try:
                if 'loop' in locals() and loop and not loop.is_closed():
                    loop.close()
            except (RuntimeError, Exception):
                # Ignore errors when closing event loop
                pass

    except Exception as e:
        logger.warning(f"Error setting up real data fetch for {symbol}: {e}")

    # Return empty list when no real data is available - no mock data fallback
    logger.warning(f"No real market data available for {symbol} - returning empty")
    return []


def calculate_trend_line(candle_data):
    """Calculate trend line using linear regression on closing prices."""
    if len(candle_data) < 2:
        return {"slope": 0, "intercept": 0, "r_squared": 0, "trend_points": []}

    import numpy as np

    # Extract closing prices and indices
    prices = [candle["close"] for candle in candle_data]
    indices = list(range(len(candle_data)))

    # Perform linear regression
    x = np.array(indices)
    y = np.array(prices)

    # Calculate slope and intercept
    slope = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) != 0 else 0
    intercept = np.mean(y) - slope * np.mean(x)

    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Generate trend line points
    trend_points = []
    for i, candle in enumerate(candle_data):
        trend_price = slope * i + intercept
        trend_points.append(
            {
                "timestamp": candle["timestamp"],
                "price": round(trend_price, 6),
                "index": i,
            }
        )

    return {
        "slope": round(slope, 8),
        "intercept": round(intercept, 6),
        "r_squared": round(r_squared, 4),
        "trend_points": trend_points,
        "direction": (
            "upward" if slope > 0 else "downward" if slope < 0 else "flat"
        ),
        "strength": (
            "strong"
            if abs(r_squared) > 0.7
            else "moderate" if abs(r_squared) > 0.4 else "weak"
        ),
    }


# All mock data generation functions removed - application uses only real market data


# Price caching system
_price_cache = {}
_CACHE_TTL = 60  # Cache prices for 60 seconds

# Price source health monitoring
_price_source_health = {}
_HEALTH_CHECK_WINDOW = 10  # Track last 10 attempts per source
_HEALTH_DECAY_FACTOR = 0.9  # Decay factor for old health data

# Manual price overrides
_manual_prices = {}
_MANUAL_PRICES_FILE = LOG_DIR / "manual_prices.json"


def _update_price_source_health(source_name: str, success: bool) -> None:
    """Update health status for a price source."""
    if source_name not in _price_source_health:
        _price_source_health[source_name] = {
            "successes": 0,
            "failures": 0,
            "last_success": None,
            "last_failure": None,
            "consecutive_failures": 0,
        }

    health = _price_source_health[source_name]

    if success:
        health["successes"] += 1
        health["last_success"] = time.time()
        health["consecutive_failures"] = 0
    else:
        health["failures"] += 1
        health["last_failure"] = time.time()
        health["consecutive_failures"] += 1

    # Apply decay to prevent old data from dominating
    health["successes"] = int(health["successes"] * _HEALTH_DECAY_FACTOR)
    health["failures"] = int(health["failures"] * _HEALTH_DECAY_FACTOR)


def _get_price_source_health(source_name: str) -> dict:
    """Get health metrics for a price source."""
    if source_name not in _price_source_health:
        return {"health_score": 0.5, "total_attempts": 0, "success_rate": 0.0}

    health = _price_source_health[source_name]
    total_attempts = health["successes"] + health["failures"]

    if total_attempts == 0:
        return {"health_score": 0.5, "total_attempts": 0, "success_rate": 0.0}

    success_rate = health["successes"] / total_attempts

    # Calculate health score (0-1, higher is better)
    # Penalize consecutive failures and recent failures
    health_score = success_rate

    if health["consecutive_failures"] > 0:
        health_score *= 0.8 ** health["consecutive_failures"]

    # Boost score if recently successful
    if health["last_success"] and health["last_failure"]:
        if health["last_success"] > health["last_failure"]:
            health_score *= 1.2  # Recent success bonus
        else:
            health_score *= 0.8  # Recent failure penalty

    health_score = min(1.0, max(0.0, health_score))

    return {
        "health_score": health_score,
        "total_attempts": total_attempts,
        "success_rate": success_rate,
        "consecutive_failures": health["consecutive_failures"],
    }


def safe_json_load(file_path):
    """Safely load JSON file, handling common corruption issues."""
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Remove trailing non-JSON characters
        content = content.strip()
        if content.endswith('%'):
            content = content[:-1].strip()
        if content.endswith(','):
            content = content[:-1].strip()

        # Find the last valid closing brace
        last_brace = content.rfind('}')
        if last_brace != -1:
            content = content[:last_brace + 1]

        return json.loads(content)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"JSON parsing failed, attempting to repair: {e}")
        try:
            # Try to extract just the JSON portion
            start = content.find('{')
            if start != -1:
                content = content[start:]
                last_brace = content.rfind('}')
                if last_brace != -1:
                    content = content[:last_brace + 1]
                    return json.loads(content)
        except Exception as repair_error:
            logger.error(f"JSON repair also failed: {repair_error}")

        raise e


def get_performance_data():
    """Get performance data for dashboard metrics."""
    try:
        # Try to get performance data from the trade manager state
        import json
        from pathlib import Path

        state_file = Path("crypto_bot/logs/trade_manager_state.json")
        if state_file.exists():
            state = safe_json_load(state_file)

            # Calculate basic performance metrics
            trades = state.get("trades", [])
            total_trades = len(trades)

            if total_trades > 0:
                # Calculate win rate based on pnl/fees
                winning_trades = [
                    t for t in trades if t.get("pnl", 0) > 0 or t.get("fees", 0) > 0
                ]  # Better win detection
                win_rate = (
                    len(winning_trades) / total_trades
                    if total_trades > 0
                    else 0
                )

                return {
                    "total_trades": total_trades,
                    "win_rate": win_rate,
                    "winning_trades": len(winning_trades),
                    "losing_trades": total_trades - len(winning_trades),
                }

        # Fallback to basic metrics
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
        }
    except Exception as e:
        logger.error(f"Error getting performance data: {e}")
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
        }


def get_allocation_data():
    """Get allocation data for dashboard metrics."""
    try:
        # Try to get allocation data from config
        import yaml
        from pathlib import Path

        config_paths = [
            Path("config.yaml"),
            Path("crypto_bot/config.yaml"),
            Path("../config.yaml"),
        ]

        for config_path in config_paths:
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f) or {}

                allocation = config.get("allocation", {})
                if allocation:
                    return allocation
                break

        # Fallback to default allocation
        return {"trend_following": 0.4, "mean_reversion": 0.3, "breakout": 0.3}
    except Exception as e:
        logger.error(f"Error getting allocation data: {e}")
        return {"trend_following": 0.4, "mean_reversion": 0.3, "breakout": 0.3}


def _get_best_price_sources() -> list:
    """Get price sources sorted by health score (best first)."""
    sources = []
    for source_name in _price_source_health.keys():
        health = _get_price_source_health(source_name)
        sources.append((source_name, health["health_score"]))

    # Sort by health score (highest first)
    sources.sort(key=lambda x: x[1], reverse=True)
    return [source[0] for source in sources]


def _load_manual_prices():
    """Load manual price overrides from file."""
    global _manual_prices
    try:
        if _MANUAL_PRICES_FILE.exists():
            with open(_MANUAL_PRICES_FILE, "r") as f:
                data = json.load(f)
                # Filter out expired manual prices (older than 24 hours)
                current_time = time.time()
                _manual_prices = {
                    symbol: price_data
                    for symbol, price_data in data.items()
                    if current_time - price_data.get("timestamp", 0)
                    < 86400  # 24 hours
                }
    except Exception as e:
        print(f"Error loading manual prices: {e}")
        _manual_prices = {}


def _save_manual_prices():
    """Save manual price overrides to file."""
    try:
        with open(_MANUAL_PRICES_FILE, "w") as f:
            json.dump(_manual_prices, f, indent=2)
    except Exception as e:
        print(f"Error saving manual prices: {e}")


def _get_manual_price(symbol: str) -> Optional[float]:
    """Get manual price override for a symbol."""
    if symbol in _manual_prices:
        price_data = _manual_prices[symbol]
        current_time = time.time()

        # Check if manual price is still valid (not expired)
        if (
            current_time - price_data.get("timestamp", 0)
            < price_data.get("validity_hours", 24) * 3600
        ):
            return price_data["price"]
        else:
            # Remove expired manual price
            del _manual_prices[symbol]
            _save_manual_prices()

    return None


def _set_manual_price(symbol: str, price: float, validity_hours: int = 24):
    """Set manual price override for a symbol."""
    _manual_prices[symbol] = {
        "price": price,
        "timestamp": time.time(),
        "validity_hours": validity_hours,
    }
    _save_manual_prices()


# Load manual prices on startup
_load_manual_prices()



# Keep a backward-compatible alias for existing code
def get_current_price_for_symbol(symbol: str) -> float:
    """Get current price for a symbol using available price sources with caching."""
    return _get_current_price_for_symbol(symbol)


def deduplicate_positions(positions):
    """Remove duplicate positions based on symbol with enhanced logic."""
    if not positions:
        return []

    # Group positions by symbol
    symbol_groups = {}
    for position in positions:
        symbol = position.get("symbol", "")
        if not symbol:
            continue

        if symbol not in symbol_groups:
            symbol_groups[symbol] = []
        symbol_groups[symbol].append(position)

    # For each symbol, keep the most recent/accurate position
    unique_positions = []
    for symbol, symbol_positions in symbol_groups.items():
        if len(symbol_positions) == 1:
            # Single position, keep it
            unique_positions.append(symbol_positions[0])
        else:
            # Multiple positions for same symbol, choose the best one
            best_position = select_best_position(symbol_positions)
            if best_position:
                unique_positions.append(best_position)
                print(
                    f"Duplicate position found for {symbol}, kept best match"
                )

    print(
        f"Deduplication: {len(positions)} -> {len(unique_positions)} positions"
    )
    return unique_positions


def select_best_position(positions):
    """Select the best position from multiple candidates for the same symbol."""
    if not positions:
        return None

    # Priority criteria:
    # 1. Has current_price (most recent data)
    # 2. Has non-zero amount
    # 3. Most recent timestamp
    # 4. Has PnL calculation

    best_position = None
    best_score = -1

    for position in positions:
        score = 0

        # Has current price
        if position.get("current_price"):
            score += 10

        # Has non-zero amount
        if position.get("amount", 0) > 0:
            score += 5

        # Has entry price
        if position.get("entry_price"):
            score += 3

        # Has PnL
        if position.get("pnl") is not None:
            score += 2

        # Has timestamp
        if position.get("timestamp"):
            score += 1

        if score > best_score:
            best_score = score
            best_position = position

    return best_position


def get_paper_wallet_balance() -> float:
    """Get paper wallet balance from the single source of truth."""
    try:
        from crypto_bot.utils.balance_manager import get_single_balance

        balance = get_single_balance()
        print(
            f"Frontend got balance from single source of truth: ${balance:.2f}"
        )
        return balance
    except Exception as e:
        print(f"Error getting balance from single source: {e}")
        # Fallback to direct calculation
        try:
            return calculate_wallet_balance_from_trade_manager()
        except Exception as e2:
            print(f"Fallback balance calculation also failed: {e2}")
            return 10000.0  # Default fallback balance


if __name__ == "__main__":
    """Run the Flask app directly with port information for startup scripts."""
    import socket
    import sys

    # Find an available port starting from 8000
    def find_free_port(start_port=8000, max_attempts=10):
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                continue
        return start_port

    # Allow overriding port via environment variable for consistency
    import os as _os
    env_port = _os.environ.get("LCT_PORT") or _os.environ.get("FLASK_PORT_OVERRIDE")
    try:
        port = int(env_port) if env_port else find_free_port()
    except Exception:
        port = find_free_port()
    print(f"FLASK_PORT={port}")  # This is what startup scripts look for
    print(f"Starting Flask app on port {port}...")
    print("Press Ctrl+C to stop the server")

    try:
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\nFlask server stopped by user")
    except Exception as e:
        print(f"Error running Flask app: {e}")
        sys.exit(1)
