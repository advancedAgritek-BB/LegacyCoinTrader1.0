#!/usr/bin/env python3
"""Test script to check if the Flask app is running and serving content correctly."""

import requests
import time
import sys

def test_flask_app():
    """Test if the Flask app is running and serving content."""
    print("Testing Flask app...")
    
    # Try different ports
    ports = [8000, 8001, 8002, 8003, 8004, 5000]
    
    for port in ports:
        try:
            print(f"Trying port {port}...")
            response = requests.get(f'http://localhost:{port}/test', timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"SUCCESS: Flask app is running on port {port}")
                print(f"Response: {data}")
                return True
        except requests.exceptions.RequestException as e:
            print(f"Port {port}: {e}")
            continue
    
    print("ERROR: Flask app is not running on any expected port")
    return False

def test_main_page():
    """Test if the main page loads correctly."""
    print("Testing main page...")
    
    ports = [8000, 8001, 8002, 8003, 8004, 5000]
    
    for port in ports:
        try:
            print(f"Trying main page on port {port}...")
            response = requests.get(f'http://localhost:{port}/', timeout=5)
            if response.status_code == 200:
                print(f"SUCCESS: Main page loads on port {port}")
                # Check if our debug elements are in the HTML
                if 'debugStatus' in response.text and 'debugMessage' in response.text:
                    print("SUCCESS: Debug elements found in HTML")
                    return True
                else:
                    print("WARNING: Debug elements not found in HTML - page might be cached")
                    return False
        except requests.exceptions.RequestException as e:
            print(f"Port {port}: {e}")
            continue
    
    print("ERROR: Main page not accessible on any expected port")
    return False

if __name__ == "__main__":
    print("=== FLASK APP TEST ===")
    
    # Test if Flask app is running
    flask_running = test_flask_app()
    
    # Test if main page loads
    main_page_ok = test_main_page()
    
    if flask_running and main_page_ok:
        print("SUCCESS: Flask app is running and serving updated content")
        sys.exit(0)
    else:
        print("ERROR: Flask app issues detected")
        sys.exit(1)
