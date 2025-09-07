#!/usr/bin/env python3
"""
Comprehensive test runner for Position Monitor functionality.

This script runs all position monitor tests to ensure complete functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import subprocess
import os

def run_tests():
    """Run all position monitor tests."""
    print("🚀 Running comprehensive Position Monitor tests...\n")

    test_files = [
        "tests/test_position_monitor.py",
        "tests/test_position_monitor_integration.py",
        "tests/test_position_monitor_performance_simple.py",
        "tests/test_position_monitor_error_handling.py"
    ]

    all_passed = True
    total_tests = 0
    total_passed = 0

    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"📋 Running {test_file}...")
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
                ], capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    print(f"✅ {test_file} PASSED")
                    # Count passed tests from output
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'passed' in line and 'failed' not in line:
                            try:
                                count = int(line.split()[0])
                                total_passed += count
                                total_tests += count
                            except (ValueError, IndexError):
                                pass
                else:
                    print(f"❌ {test_file} FAILED")
                    print("STDOUT:", result.stdout)
                    print("STDERR:", result.stderr)
                    all_passed = False

                    # Count failed tests
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'failed' in line:
                            try:
                                parts = line.split()
                                failed_count = int(parts[0])
                                passed_count = int(parts[2]) if len(parts) > 2 else 0
                                total_tests += failed_count + passed_count
                                total_passed += passed_count
                            except (ValueError, IndexError):
                                pass

            except subprocess.TimeoutExpired:
                print(f"⏰ {test_file} TIMED OUT")
                all_passed = False
            except Exception as e:
                print(f"💥 {test_file} ERROR: {e}")
                all_passed = False
        else:
            print(f"⚠️  {test_file} not found, skipping...")
            all_passed = False

    print(f"\n📊 Test Results Summary:")
    print(f"   Total tests: {total_tests}")
    print(f"   Passed: {total_passed}")
    print(f"   Failed: {total_tests - total_passed}")

    if all_passed:
        print("\n🎉 All Position Monitor tests PASSED!")
        print("✅ Position monitoring system is fully functional and tested.")
        return True
    else:
        print("\n❌ Some tests FAILED or were SKIPPED.")
        print("⚠️  Please review the failures above.")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
