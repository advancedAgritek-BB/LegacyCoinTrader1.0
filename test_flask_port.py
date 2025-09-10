#!/usr/bin/env python3
"""
Test script to verify Flask port detection works correctly.
"""

import subprocess
import tempfile
import time
import os
import sys

def test_flask_port_detection():
    """Test that Flask app outputs FLASK_PORT correctly."""
    print("Testing Flask port detection...")

    # Create temp file to capture output
    temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    temp_file.close()

    try:
        # Start Flask app
        print("Starting Flask app...")
        proc = subprocess.Popen([sys.executable, "-m", "frontend.app"],
                               stdout=open(temp_file.name, 'w'),
                               stderr=subprocess.STDOUT)

        # Wait for Flask to start
        time.sleep(3)

        # Read output
        with open(temp_file.name, 'r') as f:
            output = f.read()

        # Extract FLASK_PORT
        flask_port = 8000  # default fallback
        for line in output.split('\n'):
            if line.startswith('FLASK_PORT='):
                flask_port = int(line.strip().split('=')[1])
                break

        # Kill the process
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except:
            try:
                proc.kill()
            except:
                pass

        print(f"SUCCESS: Flask app started on port {flask_port}")
        print(f"Browser should open to: http://localhost:{flask_port}")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False
    finally:
        # Clean up
        try:
            os.unlink(temp_file.name)
        except:
            pass

if __name__ == "__main__":
    # Activate virtual environment if it exists
    venv_path = "venv/bin/activate"
    if os.path.exists(venv_path):
        os.system(f"source {venv_path}")

    success = test_flask_port_detection()
    sys.exit(0 if success else 1)
