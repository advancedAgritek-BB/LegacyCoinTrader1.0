#!/usr/bin/env python3
"""
Trade Pipeline Test Runner

This script runs all tests related to the trade pipeline and ensures comprehensive coverage
of logging, monitoring, and debugging functionality for both live and paper trading modes.
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def run_tests():
    """Run all trade pipeline tests and report results."""
    
    print("🚀 Starting Trade Pipeline Test Suite")
    print("=" * 50)
    
    # Test files to run
    test_files = [
        "tests/test_trade_pipeline_comprehensive.py",
        "tests/test_trade_pipeline_monitor.py",
        "tests/test_enhanced_logger.py"
    ]
    
    # Additional test files that might exist
    additional_tests = [
        "tests/test_paper_wallet_integration.py",
        "tests/test_cex_executor.py",
        "tests/test_solana_trading.py",
        "tests/test_risk_manager.py"
    ]
    
    # Check which additional test files exist
    existing_additional_tests = []
    for test_file in additional_tests:
        if Path(test_file).exists():
            existing_additional_tests.append(test_file)
    
    all_test_files = test_files + existing_additional_tests
    
    print(f"📋 Found {len(all_test_files)} test files to run:")
    for test_file in all_test_files:
        print(f"   - {test_file}")
    
    print("\n" + "=" * 50)
    
    # Run tests
    results = {}
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_file in all_test_files:
        if not Path(test_file).exists():
            print(f"⚠️  Test file not found: {test_file}")
            continue
            
        print(f"\n🧪 Running tests in: {test_file}")
        print("-" * 40)
        
        try:
            # Run pytest on the specific test file
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            # Parse results
            output_lines = result.stdout.split('\n')
            test_summary = None
            
            for line in output_lines:
                if 'collected' in line and 'items' in line:
                    # Extract test count
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'collected':
                            if i + 1 < len(parts):
                                total_tests += int(parts[i + 1])
                            break
                elif 'passed' in line and 'failed' in line:
                    test_summary = line
                    break
            
            if result.returncode == 0:
                print(f"✅ All tests passed in {test_file}")
                if test_summary:
                    print(f"   Summary: {test_summary}")
                passed_tests += total_tests
            else:
                print(f"❌ Some tests failed in {test_file}")
                print(f"   Return code: {result.returncode}")
                if test_summary:
                    print(f"   Summary: {test_summary}")
                
                # Show error output
                if result.stderr:
                    print("   Errors:")
                    for line in result.stderr.split('\n')[-10:]:  # Last 10 lines
                        if line.strip():
                            print(f"     {line}")
                
                failed_tests += 1
            
            results[test_file] = {
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Tests in {test_file} timed out after 5 minutes")
            results[test_file] = {'timeout': True}
            failed_tests += 1
        except Exception as e:
            print(f"💥 Error running tests in {test_file}: {e}")
            results[test_file] = {'error': str(e)}
            failed_tests += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    print(f"Total test files: {len(all_test_files)}")
    print(f"Files with tests: {len([f for f in all_test_files if Path(f).exists()])}")
    print(f"Files passed: {len([f for f, r in results.items() if r.get('return_code') == 0])}")
    print(f"Files failed: {failed_tests}")
    
    if failed_tests == 0:
        print("\n🎉 All test files passed successfully!")
        return True
    else:
        print(f"\n⚠️  {failed_tests} test file(s) had issues")
        
        # Show details of failures
        print("\n📋 FAILURE DETAILS:")
        for test_file, result in results.items():
            if result.get('return_code', 0) != 0:
                print(f"\n   {test_file}:")
                if 'timeout' in result:
                    print("     - Timed out")
                elif 'error' in result:
                    print(f"     - Error: {result['error']}")
                else:
                    print(f"     - Return code: {result['return_code']}")
        
        return False

def run_coverage_tests():
    """Run tests with coverage reporting."""
    
    print("\n📊 Running Coverage Tests")
    print("=" * 50)
    
    try:
        # Run pytest with coverage
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_trade_pipeline_comprehensive.py",
            "tests/test_trade_pipeline_monitor.py",
            "--cov=crypto_bot.utils.enhanced_logger",
            "--cov=crypto_bot.utils.trade_pipeline_monitor",
            "--cov-report=term-missing",
            "--cov-report=html"
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print("✅ Coverage tests completed successfully")
            print("\n📈 Coverage Report:")
            print(result.stdout)
        else:
            print("❌ Coverage tests failed")
            print("Error output:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Coverage tests timed out after 10 minutes")
    except Exception as e:
        print(f"💥 Error running coverage tests: {e}")

def check_test_dependencies():
    """Check if required test dependencies are available."""
    
    print("🔍 Checking Test Dependencies")
    print("=" * 50)
    
    required_packages = [
        'pytest',
        'pytest-asyncio',
        'pytest-cov'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All required packages are available")
        return True

def main():
    """Main function to run all tests."""
    
    print("🧪 Trade Pipeline Test Suite")
    print("=" * 50)
    print("This script will run comprehensive tests for:")
    print("  - Enhanced logging system")
    print("  - Trade pipeline monitoring")
    print("  - Debugging tools")
    print("  - Paper vs live trading modes")
    print("  - Error handling and recovery")
    print("=" * 50)
    
    # Check dependencies
    if not check_test_dependencies():
        print("\n❌ Cannot run tests due to missing dependencies")
        sys.exit(1)
    
    # Run basic tests
    basic_tests_passed = run_tests()
    
    # Run coverage tests if basic tests passed
    if basic_tests_passed:
        run_coverage_tests()
    
    # Final summary
    print("\n" + "=" * 50)
    print("🏁 TEST RUN COMPLETED")
    print("=" * 50)
    
    if basic_tests_passed:
        print("🎉 All tests passed successfully!")
        print("The trade pipeline is ready for production use.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        print("The trade pipeline may have issues that need to be resolved.")
        sys.exit(1)

if __name__ == "__main__":
    main()
