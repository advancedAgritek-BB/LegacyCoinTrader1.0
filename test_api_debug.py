#!/usr/bin/env python3
"""
Debug script to test the API endpoint and see what's happening
"""

import requests
import json
import subprocess
import sys
from pathlib import Path

def test_api_endpoint():
    """Test the API endpoint directly"""
    print("🔍 Testing API endpoint...")
    
    try:
        # Test the Flask API endpoint
        response = requests.get('http://localhost:8000/api/open-positions', timeout=10)
        print(f"📡 API Response Status: {response.status_code}")
        print(f"📡 API Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"📊 API Response Data: {type(data)}")
                print(f"📊 API Response Length: {len(data) if isinstance(data, list) else 'not a list'}")
                print(f"📊 API Response Preview: {str(data)[:200]}...")
            except json.JSONDecodeError as e:
                print(f"❌ JSON Decode Error: {e}")
                print(f"📄 Raw Response: {response.text[:500]}...")
        else:
            print(f"❌ API Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")

def test_helper_script():
    """Test the helper script directly"""
    print("\n🔍 Testing helper script directly...")
    
    try:
        result = subprocess.run(
            [sys.executable, 'get_positions_with_prices.py'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"📡 Script Return Code: {result.returncode}")
        print(f"📡 Script Stdout Length: {len(result.stdout)}")
        print(f"📡 Script Stderr Length: {len(result.stderr)}")
        
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                print(f"📊 Script Response Data: {type(data)}")
                print(f"📊 Script Response Length: {len(data) if isinstance(data, list) else 'not a list'}")
                print(f"📊 Script Response Preview: {str(data)[:200]}...")
            except json.JSONDecodeError as e:
                print(f"❌ Script JSON Decode Error: {e}")
                print(f"📄 Script Raw Output: {result.stdout[:500]}...")
        else:
            print(f"❌ Script Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ Script timed out")
    except Exception as e:
        print(f"❌ Script Exception: {e}")

def test_flask_import():
    """Test if we can import the Flask app functions"""
    print("\n🔍 Testing Flask app imports...")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from frontend.app import get_open_positions, deduplicate_positions
        print("✅ Successfully imported get_open_positions and deduplicate_positions")
        
        # Test the functions directly
        print("🔍 Testing get_open_positions function...")
        positions = get_open_positions()
        print(f"📊 Direct function returned: {type(positions)}")
        print(f"📊 Direct function length: {len(positions) if isinstance(positions, list) else 'not a list'}")
        
        if positions:
            print("🔍 Testing deduplicate_positions function...")
            unique_positions = deduplicate_positions(positions)
            print(f"📊 Deduplicated positions: {len(unique_positions)}")
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
    except Exception as e:
        print(f"❌ Function Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting API Debug Test")
    print("=" * 50)
    
    test_api_endpoint()
    test_helper_script()
    test_flask_import()
    
    print("\n" + "=" * 50)
    print("🏁 Debug Test Complete")
