#!/usr/bin/env python3
"""
Quick script to open the monitoring dashboard in your browser
"""

import requests
import subprocess
import sys

def find_monitoring_port():
    """Find which port the monitoring page is running on"""
    ports = [8000, 8001, 8002, 8003, 8004, 8005]  # Start with 8000 since that's where it's working
    
    for port in ports:
        try:
            response = requests.get(f"http://localhost:{port}/monitoring", timeout=2)
            if response.status_code == 200:
                return port
        except:
            continue
    
    return None

def main():
    print("🔍 Looking for monitoring dashboard...")
    
    port = find_monitoring_port()
    
    if port:
        url = f"http://localhost:{port}/monitoring"
        print(f"✅ Found monitoring dashboard on port {port}")
        print(f"🚀 Opening: {url}")
        
        # Open in browser
        subprocess.run(['open', url])
        
        print(f"📊 Monitoring Dashboard: {url}")
        print(f"📋 System Logs: http://localhost:{port}/logs")
        print(f"🏠 Main Dashboard: http://localhost:{port}/")
    else:
        print("❌ Monitoring dashboard not found")
        print("💡 Try starting the bot first:")
        print("   python3 start_bot.py auto")

if __name__ == "__main__":
    main()
