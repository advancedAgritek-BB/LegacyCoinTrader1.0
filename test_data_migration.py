#!/usr/bin/env python3
"""
Test script for data migration functionality.

This script demonstrates how to migrate legacy data to the microservice architecture.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data_migration import DataMigrationService, MigrationConfig


async def test_csv_migration():
    """Test CSV trade migration."""
    print("🧪 Testing CSV trade migration...")

    config = MigrationConfig(dry_run=True)
    csv_path = Path("sample_trades.csv")

    async with DataMigrationService(config) as migrator:
        try:
            result = await migrator.migrate_csv_trades(csv_path)
            print(f"✅ CSV Migration Result: {result}")
            return True
        except Exception as e:
            print(f"❌ CSV Migration Failed: {e}")
            return False


async def test_json_migration():
    """Test JSON state migration."""
    print("🧪 Testing JSON state migration...")

    config = MigrationConfig(dry_run=True)
    json_path = Path("sample_trade_manager_state.json")

    async with DataMigrationService(config) as migrator:
        try:
            result = await migrator.migrate_json_state(json_path)
            print(f"✅ JSON Migration Result: {result}")
            return True
        except Exception as e:
            print(f"❌ JSON Migration Failed: {e}")
            return False


async def test_directory_scan():
    """Test directory scanning for migration."""
    print("🧪 Testing directory scan migration...")

    config = MigrationConfig(dry_run=True)
    current_dir = Path(".")

    async with DataMigrationService(config) as migrator:
        try:
            result = await migrator.scan_and_migrate_directory(current_dir)
            print(f"✅ Directory Scan Result: {result}")
            return True
        except Exception as e:
            print(f"❌ Directory Scan Failed: {e}")
            return False


def test_frontend_integration():
    """Test that frontend can import gateway utilities."""
    print("🧪 Testing frontend gateway integration...")

    try:
        from frontend.gateway import get_gateway_json, post_gateway_json, ApiGatewayError
        print("✅ Frontend gateway imports successful")
        return True
    except Exception as e:
        print(f"❌ Frontend gateway import failed: {e}")
        return False


async def main():
    """Run all migration tests."""
    print("🚀 Starting Data Migration Tests\n")

    tests = [
        ("CSV Migration", test_csv_migration),
        ("JSON Migration", test_json_migration),
        ("Directory Scan", test_directory_scan),
        ("Frontend Integration", test_frontend_integration),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)

        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()

        results.append((test_name, result))

    print(f"\n{'='*50}")
    print("MIGRATION TEST SUMMARY")
    print('='*50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print("25")
        if result:
            passed += 1

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All migration tests passed!")
        print("\n📝 Next Steps:")
        print("1. Start the microservices with: docker-compose up")
        print("2. Run live migration: python data_migration.py --source-file your_data.csv")
        print("3. Update frontend environment to use API_GATEWAY_URL=http://localhost:8000")
        print("4. Test the updated frontend endpoints")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
