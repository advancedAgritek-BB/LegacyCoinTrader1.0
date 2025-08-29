#!/usr/bin/env python3
"""
Simple test script for frontend functionality.
This script tests the basic frontend features without importing complex dependencies.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_frontend_files():
    """Test that frontend files exist and are accessible."""
    print("Testing frontend files...")
    
    # Check if frontend directory exists
    frontend_dir = project_root / "frontend"
    if not frontend_dir.exists():
        print("✗ Frontend directory not found")
        return False
    print("✓ Frontend directory exists")
    
    # Check if key files exist
    key_files = [
        "app.py",
        "templates/index.html",
        "static/app.js",
        "utils.py"
    ]
    
    for file_path in key_files:
        full_path = frontend_dir / file_path
        if full_path.exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} not found")
            return False
    
    return True

def test_config_files():
    """Test configuration files."""
    print("\nTesting configuration files...")
    
    # Check if paper wallet config exists
    paper_wallet_config = project_root / "crypto_bot" / "paper_wallet_config.yaml"
    if paper_wallet_config.exists():
        print("✓ Paper wallet config exists")
        
        # Try to read it
        try:
            import yaml
            with open(paper_wallet_config) as f:
                config = yaml.safe_load(f)
                balance = config.get('initial_balance', 0)
                print(f"✓ Config readable, balance: ${balance:.2f}")
        except Exception as e:
            print(f"✗ Error reading config: {e}")
            return False
    else:
        print("✗ Paper wallet config not found")
        return False
    
    return True

def test_flask_app():
    """Test basic Flask app functionality."""
    print("\nTesting Flask app...")
    
    try:
        # Create a minimal Flask app to test
        from flask import Flask
        
        app = Flask(__name__)
        
        @app.route('/test')
        def test():
            return {'status': 'ok'}
        
        print("✓ Flask app created successfully")
        
        # Test basic routing
        with app.test_client() as client:
            response = client.get('/test')
            if response.status_code == 200:
                print("✓ Basic routing works")
            else:
                print(f"✗ Routing failed: {response.status_code}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Flask test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoint definitions."""
    print("\nTesting API endpoints...")
    
    try:
        # Read the app.py file to check for API endpoints
        app_file = project_root / "frontend" / "app.py"
        
        with open(app_file) as f:
            content = f.read()
        
        # Check for key API endpoints
        required_endpoints = [
            '/api/paper-wallet-balance',
            '/api/live-updates',
            '/api/dashboard-metrics'
        ]
        
        for endpoint in required_endpoints:
            if endpoint in content:
                print(f"✓ {endpoint} endpoint defined")
            else:
                print(f"✗ {endpoint} endpoint not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ API endpoint test failed: {e}")
        return False

def test_html_templates():
    """Test HTML template structure."""
    print("\nTesting HTML templates...")
    
    try:
        # Read the index.html file
        index_file = project_root / "frontend" / "templates" / "index.html"
        
        with open(index_file) as f:
            content = f.read()
        
        # Check for key elements
        required_elements = [
            'walletBalanceSection',
            'executionMode',
            'paper-wallet-balance',
            'toggleWalletBalanceField'
        ]
        
        for element in required_elements:
            if element in content:
                print(f"✓ {element} element found")
            else:
                print(f"✗ {element} element not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ HTML template test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Frontend Simple Test Suite")
    print("=" * 40)
    
    tests = [
        test_frontend_files,
        test_config_files,
        test_flask_app,
        test_api_endpoints,
        test_html_templates
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"\n✗ Test failed: {test.__name__}")
            break
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Frontend structure is correct.")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
