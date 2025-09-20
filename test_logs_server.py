#!/usr/bin/env python3
"""Simple test server for logs API."""

import sys
import json
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from flask import Flask, jsonify
    
    app = Flask(__name__)
    
    @app.route("/api/monitoring/logs")
    def test_logs_api():
        """Test logs API endpoint."""
        LOG_DIR = Path("crypto_bot/logs")
        
        # Define log files to read
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
        
        logs_data = {}
        
        for log_type, log_file in log_files.items():
            if log_file.exists():
                try:
                    text = log_file.read_text(encoding='utf-8', errors='ignore')
                    lines = text.splitlines()[-50:]
                    lines = [line.strip() for line in lines if line.strip()]
                    logs_data[log_type] = lines
                except Exception as e:
                    logs_data[log_type] = [f"Error reading log file: {e}"]
            else:
                logs_data[log_type] = [f"Log file not found: {log_file}"]
        
        return jsonify({
            "success": True,
            "data": logs_data,
            "timestamp": int(time.time() * 1000),
            "total_entries": sum(len(logs) for logs in logs_data.values())
        })
    
    @app.route("/logs")
    def logs_page():
        """Serve the logs page."""
        with open("frontend/templates/logs.html", "r") as f:
            content = f.read()
        # Simple template rendering
        content = content.replace("{{ url_for('static', filename='styles.css') }}", "/static/styles.css")
        return content
    
    @app.route("/static/styles.css")
    def serve_css():
        """Serve CSS file."""
        with open("frontend/static/styles.css", "r") as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/css'}
    
    if __name__ == "__main__":
        print("Starting test logs server on http://localhost:5000")
        print("Visit http://localhost:5000/logs to see the logs page")
        print("Visit http://localhost:5000/api/monitoring/logs to test the API")
        app.run(debug=True, port=5000)
        
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install flask: pip install flask")
