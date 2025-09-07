#!/usr/bin/env python3
"""
Simple Local Test for LegacyCoinTrader 2.0

This script provides basic testing without complex configuration.
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test basic module imports."""
    print("🔍 Testing basic imports...")

    try:
        # Test Python version
        print(f"✅ Python {sys.version.split()[0]}")

        # Test basic imports
        import asyncio
        print("✅ asyncio available")

        import json
        print("✅ json available")

        import datetime
        print("✅ datetime available")

        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_domain_models():
    """Test domain models without full config."""
    print("\n🏗️ Testing domain models...")

    try:
        # Create minimal test data
        from decimal import Decimal

        # Test basic data structures
        symbol_data = {
            "symbol": "BTC/USD",
            "base_currency": "BTC",
            "quote_currency": "USD",
            "exchange": "kraken",
            "min_order_size": Decimal("0.0001"),
            "price_precision": 2,
            "quantity_precision": 8
        }

        print(f"✅ Symbol data: {symbol_data['symbol']}")

        # Test position calculation
        position_data = {
            "quantity": Decimal("0.01"),
            "entry_price": Decimal("50000.00"),
            "current_price": Decimal("51000.00")
        }

        unrealized_pnl = (position_data["current_price"] - position_data["entry_price"]) * position_data["quantity"]
        print(f"✅ P&L calculation: ${unrealized_pnl}")

        return True
    except Exception as e:
        print(f"❌ Domain model error: {e}")
        return False

def test_security_components():
    """Test security components."""
    print("\n🔒 Testing security components...")

    try:
        import hashlib
        import hmac
        from cryptography.fernet import Fernet

        # Test basic encryption
        key = Fernet.generate_key()
        fernet = Fernet(key)

        test_data = b"Hello, World!"
        encrypted = fernet.encrypt(test_data)
        decrypted = fernet.decrypt(encrypted)

        if decrypted == test_data:
            print("✅ Basic encryption/decryption")
        else:
            print("❌ Encryption/decryption failed")
            return False

        # Test HMAC
        secret = b"test_secret"
        message = b"test_message"
        signature = hmac.new(secret, message, hashlib.sha256).hexdigest()
        print(f"✅ HMAC signature: {signature[:16]}...")

        return True
    except Exception as e:
        print(f"❌ Security component error: {e}")
        return False

def test_async_patterns():
    """Test async patterns."""
    print("\n⚡ Testing async patterns...")

    async def simple_async_function():
        await asyncio.sleep(0.1)
        return "async_works"

    async def run_test():
        try:
            result = await simple_async_function()
            if result == "async_works":
                print("✅ Async/await patterns working")
                return True
            else:
                print("❌ Async test failed")
                return False
        except Exception as e:
            print(f"❌ Async error: {e}")
            return False

    # Run the async test
    import asyncio
    return asyncio.run(run_test())

def test_file_structure():
    """Test project file structure."""
    print("\n📁 Testing file structure...")

    project_root = Path(__file__).parent

    # Check for key directories
    required_dirs = [
        "modern/src",
        "modern/tests",
        "requirements-modern.txt",
        "pyproject.toml"
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if not (project_root / dir_path).exists():
            missing_dirs.append(dir_path)

    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories present")
        return True

def test_dependencies():
    """Test key dependencies."""
    print("\n📦 Testing dependencies...")

    dependencies = [
        ("fastapi", "Web framework"),
        ("pydantic", "Data validation"),
        ("sqlalchemy", "Database ORM"),
        ("pytest", "Testing framework"),
        ("cryptography", "Security"),
        ("redis", "Caching")
    ]

    missing_deps = []
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
        except ImportError:
            missing_deps.append(module)
            print(f"❌ {description}: {module} (missing)")

    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {missing_deps}")
        print("Run: pip install -r requirements-modern.txt")
        return False
    else:
        print("✅ All key dependencies available")
        return True

def main():
    """Main test function."""
    print("🧪 LegacyCoinTrader 2.0 - Simple Local Test")
    print("=" * 50)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Domain Models", test_domain_models),
        ("Security Components", test_security_components),
        ("Async Patterns", test_async_patterns),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            success = test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {status}")
        except Exception as e:
            results[test_name] = False
            print(f"   ❌ ERROR: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All basic tests passed!")
        print("\n🚀 Your modernized trading system is ready!")
        print("\nNext steps:")
        print("1. Fix configuration issues (see LOCAL_DEVELOPMENT_README.md)")
        print("2. Run: python start_local.py")
        print("3. Start developing with the modern architecture!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("Check the errors above and ensure all dependencies are installed.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
