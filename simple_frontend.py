#!/usr/bin/env python3
"""Simple frontend server to test logs functionality."""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

LOG_DIR = Path("crypto_bot/logs")

@app.route("/api/monitoring/logs")
def api_monitoring_logs():
    """Return monitoring logs from various system components."""
    try:
        logs_data = {}

        # Define log files to read - updated for microservices architecture
        log_files = {
            "trading_engine": LOG_DIR / "trading_engine.log",
            "portfolio": LOG_DIR / "portfolio.log",
            "market_data": LOG_DIR / "market_data.log",
            "strategy_engine": LOG_DIR / "strategy_engine.log",
            "token_discovery": LOG_DIR / "token_discovery.log",
            "api_gateway": LOG_DIR / "api_gateway.log",
            "pipeline_monitor": LOG_DIR / "pipeline_monitor.log",
            "health_check": LOG_DIR / "health_check.log",
        }

        for log_type, log_file in log_files.items():
            if log_file.exists():
                try:
                    # Read last 50 lines of each log file
                    text = log_file.read_text(encoding='utf-8', errors='ignore')
                    lines = text.splitlines()[-50:]
                    # Filter out empty lines
                    lines = [line.strip() for line in lines if line.strip()]
                    logs_data[log_type] = lines
                    print(f"Read {len(lines)} lines from {log_type}")
                except Exception as e:
                    logs_data[log_type] = [f"Error reading log file: {e}"]
                    print(f"Error reading {log_type}: {e}")
            else:
                logs_data[log_type] = [f"Log file not found: {log_file}"]
                print(f"Log file not found: {log_file}")

        total_entries = sum(len(logs) for logs in logs_data.values())
        print(f"Total log entries: {total_entries}")

        return jsonify({
            "success": True,
            "data": logs_data,
            "timestamp": int(time.time() * 1000),
            "total_entries": total_entries
        })

    except Exception as e:
        print(f"Error in api_monitoring_logs: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "data": {},
            "timestamp": int(time.time() * 1000),
        })

@app.route("/logs")
def logs_page():
    """Serve the logs page."""
    try:
        with open("frontend/templates/logs.html", "r") as f:
            template_content = f.read()
        
        # Simple template processing - replace extends and blocks
        template_content = template_content.replace('{% extends "base.html" %}', '')
        template_content = template_content.replace('{% block content %}', '')
        template_content = template_content.replace('{% endblock %}', '')
        
        # Add basic HTML structure
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>System Logs</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        {get_basic_styles()}
    </style>
</head>
<body>
    {template_content}
</body>
</html>
        """
        
        return html_content
    except Exception as e:
        return f"Error loading logs page: {e}"

def get_basic_styles():
    """Return basic styles for the logs page."""
    return """
        :root {
            --primary: #6366f1;
            --secondary: #8b5cf6;
            --success: #10b981;
            --danger: #ef4444;
            --warning: #f59e0b;
            --info: #06b6d4;
            --bg-primary: #0f0f23;
            --bg-secondary: #161b33;
            --bg-surface: #0c1220;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --border-secondary: rgba(148, 163, 184, 0.2);
            --glass-secondary: rgba(22, 27, 51, 0.8);
            --glass-light: rgba(51, 65, 85, 0.6);
            --radius-lg: 0.75rem;
            --radius-md: 0.5rem;
            --radius-sm: 0.375rem;
            --space-sm: 0.5rem;
            --space-md: 1rem;
            --space-lg: 1.5rem;
            --space-xl: 2rem;
            --space-2xl: 3rem;
            --font-sm: 0.875rem;
            --font-xs: 0.75rem;
            --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            --transition-fast: 0.15s cubic-bezier(0.4, 0, 0.2, 1);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
        }
        
        * { box-sizing: border-box; }
        
        body {
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, var(--bg-primary) 0%, #1a1f3a 30%, var(--bg-secondary) 70%);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        .dashboard-container {
            padding: var(--space-2xl);
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .welcome-section {
            background: var(--glass-secondary);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-secondary);
            border-radius: var(--radius-lg);
            padding: var(--space-xl);
            margin-bottom: var(--space-2xl);
        }
        
        .welcome-title {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: var(--space-sm);
        }
        
        .welcome-subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
            margin-bottom: 0;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: var(--space-lg);
            margin-bottom: var(--space-2xl);
        }
        
        .metric-card {
            background: var(--glass-secondary);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-secondary);
            border-radius: var(--radius-lg);
            padding: var(--space-xl);
            text-align: center;
            transition: all var(--transition-fast);
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 900;
            margin-bottom: var(--space-sm);
        }
        
        .metric-label {
            color: var(--text-muted);
            font-size: var(--font-sm);
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .card {
            background: var(--glass-secondary);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-secondary);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-md);
            transition: all var(--transition-fast);
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }
        
        .card-header {
            padding: var(--space-lg);
            border-bottom: 1px solid var(--border-secondary);
        }
        
        .card-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            gap: var(--space-sm);
        }
        
        .card-body {
            padding: var(--space-lg);
        }
        
        .btn {
            display: inline-flex;
            align-items: center;
            gap: var(--space-sm);
            padding: var(--space-sm) var(--space-lg);
            font-size: var(--font-sm);
            font-weight: 600;
            border-radius: var(--radius-md);
            border: 1px solid transparent;
            cursor: pointer;
            transition: all var(--transition-fast);
            text-decoration: none;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--primary), #818cf8);
            color: white;
        }
        
        .btn-outline-primary {
            background: transparent;
            color: var(--primary);
            border-color: var(--primary);
        }
        
        .btn:hover {
            transform: translateY(-1px);
        }
        
        .logs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: var(--space-lg);
        }
        
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success);
        }
        
        .status-indicator.offline {
            background: var(--danger);
        }
        
        .badge {
            display: inline-flex;
            align-items: center;
            gap: var(--space-sm);
            padding: var(--space-sm);
            font-size: var(--font-sm);
            font-weight: 600;
            border-radius: var(--radius-md);
            border: 1px solid;
        }
        
        .badge.bg-info {
            background: rgba(6, 182, 212, 0.1);
            color: var(--info);
            border-color: var(--info);
        }
    """

if __name__ == "__main__":
    print("Starting simple frontend server on http://localhost:5001")
    print("Visit http://localhost:5001/logs to see the logs page")
    print("Visit http://localhost:5001/api/monitoring/logs to test the API")
    app.run(debug=True, port=5001, host='0.0.0.0')
