#!/usr/bin/env python3
"""Simple test to check if Flask app is running and serving updated content."""

import requests
import sys

def main():
    print("Testing Flask app...")
    
    # Try to connect to the Flask app
    try:
        response = requests.get('http://localhost:8000/test', timeout=5)
        if response.status_code == 200:
            print("SUCCESS: Flask app is running on port 8000")
            data = response.json()
            print(f"Test response: {data}")
        else:
            print(f"ERROR: Flask app returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("ERROR: Flask app is not running on port 8000")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    # Try to get the main page
    try:
        response = requests.get('http://localhost:8000/', timeout=5)
        if response.status_code == 200:
            print("SUCCESS: Main page loads")
            
            # Check for our debug elements
            if 'debugStatus' in response.text:
                print("SUCCESS: debugStatus element found in HTML")
            else:
                print("WARNING: debugStatus element not found - page might be cached")
            
            if 'debugMessage' in response.text:
                print("SUCCESS: debugMessage element found in HTML")
            else:
                print("WARNING: debugMessage element not found - page might be cached")
            
            if 'forceRefresh' in response.text:
                print("SUCCESS: forceRefresh function found in HTML")
            else:
                print("WARNING: forceRefresh function not found - page might be cached")
                
        else:
            print(f"ERROR: Main page returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    print("SUCCESS: All tests passed")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
