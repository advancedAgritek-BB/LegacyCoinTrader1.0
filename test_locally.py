#!/usr/bin/env python3
"""
Local Testing Script for LegacyCoinTrader 2.0

This script provides comprehensive local testing capabilities for the modernized
trading system, including unit tests, integration tests, and health checks.
"""

import asyncio
import sys
import os
from pathlib import Path
import subprocess
import time
from typing import Dict, List, Optional

# Add modern source path
project_root = Path(__file__).parent
modern_src = project_root / "modern" / "src"
if str(modern_src) not in sys.path:
    sys.path.insert(0, str(modern_src))

from core.config import init_config, get_settings, AppConfig, Environment
from core.container import init_container, get_container


class LocalTester:
    """Comprehensive local testing suite."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.modern_dir = self.project_root / "modern"
        self.test_results = {}

    async def run_all_tests(self) -> Dict[str, bool]:
        """Run comprehensive test suite."""
        print("🚀 Starting comprehensive local testing...\n")

        tests = [
            ("Configuration Tests", self.test_configuration),
            ("Domain Models Tests", self.test_domain_models),
            ("Security Tests", self.test_security),
            ("Database Tests", self.test_database),
            ("Integration Tests", self.test_integration),
            ("Performance Tests", self.test_performance),
            ("Health Checks", self.test_health_checks)
        ]

        results = {}
        for test_name, test_func in tests:
            print(f"📋 Running {test_name}...")
            try:
                success = await test_func()
                results[test_name] = success
                status = "✅ PASSED" if success else "❌ FAILED"
                print(f"   {status}\n")
            except Exception as e:
                results[test_name] = False
                print(f"   ❌ ERROR: {e}\n")

        return results

    async def test_configuration(self) -> bool:
        """Test configuration system."""
        try:
            # Set minimal environment for testing
            import os
            os.environ.setdefault("ENVIRONMENT", "development")
            os.environ.setdefault("APP_NAME", "LegacyCoinTrader")
            os.environ.setdefault("VERSION", "2.0.0")
            os.environ.setdefault("EXCHANGE_NAME", "kraken")  # ExchangeConfig uses EXCHANGE_ prefix
            os.environ.setdefault("EXCHANGE_API_KEY", "test_key")
            os.environ.setdefault("EXCHANGE_API_SECRET", "test_secret")
            os.environ.setdefault("SECURITY_JWT_SECRET_KEY", "test_jwt_secret")
            os.environ.setdefault("TRADING_EXECUTION_MODE", "dry_run")
            os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

            # Test config initialization
            config = init_config()
            assert isinstance(config, AppConfig)
            assert config.app_name == "LegacyCoinTrader"
            assert config.version == "2.0.0"

            # Test settings access
            settings = get_settings()
            assert settings is config

            print("   ✓ Configuration initialization")
            print("   ✓ Environment settings")
            print("   ✓ Settings access")

            return True
        except Exception as e:
            print(f"   ✗ Configuration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def test_domain_models(self) -> bool:
        """Test domain models."""
        try:
            from domain.models import TradingSymbol, Order, Position, Trade, OrderSide, OrderType

            # Test TradingSymbol
            symbol = TradingSymbol(
                symbol="BTC/USD",
                base_currency="BTC",
                quote_currency="USD",
                exchange="kraken",
                min_order_size=0.0001,
                price_precision=2,
                quantity_precision=8
            )
            assert symbol.symbol == "BTC/USD"
            print("   ✓ TradingSymbol model")

            # Test Order
            order = Order(
                id="test_order_123",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                quantity=0.01,
                price=50000.00
            )
            assert order.id == "test_order_123"
            assert order.side == OrderSide.BUY
            print("   ✓ Order model")

            # Test Position
            position = Position(
                id="test_pos_123",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                quantity=0.01,
                entry_price=50000.00,
                current_price=51000.00
            )
            assert position.unrealized_pnl == 100.00  # (51000 - 50000) * 0.01
            print("   ✓ Position model with P&L calculation")

            # Test Trade
            trade = Trade(
                id="test_trade_123",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                quantity=0.01,
                price=50000.00,
                value=500.00,
                pnl=100.00,
                pnl_percentage=2.0,
                commission=0.50,
                order_id="test_order_123"
            )
            assert trade.is_profitable is True
            print("   ✓ Trade model")

            return True
        except Exception as e:
            print(f"   ✗ Domain models test failed: {e}")
            return False

    async def test_security(self) -> bool:
        """Test security components."""
        try:
            from core.security import EncryptionService, PasswordService, APIKeyService

            # Test encryption
            encryption = EncryptionService()
            original = "test_secret_data"
            encrypted = encryption.encrypt(original)
            decrypted = encryption.decrypt(encrypted)
            assert decrypted == original
            print("   ✓ Encryption service")

            # Test password hashing
            password_service = PasswordService()
            password = "test_password_123"
            hashed = password_service.hash_password(password)
            is_valid = password_service.verify_password(password, hashed)
            assert is_valid is True
            print("   ✓ Password hashing")

            # Test API key generation
            api_key_service = APIKeyService("test_secret")
            keys = api_key_service.generate_api_key("test")
            assert "key_id" in keys
            assert "secret" in keys
            print("   ✓ API key generation")

            return True
        except Exception as e:
            print(f"   ✗ Security test failed: {e}")
            return False

    async def test_database(self) -> bool:
        """Test database components."""
        try:
            from infrastructure.database import DatabaseConnection, DatabaseRepository

            # Test database connection creation
            config = get_settings()
            db_conn = DatabaseConnection(config.database)

            # Test repository creation
            repo = DatabaseRepository(
                connection=db_conn,
                table_name="test_table",
                model_class=object
            )

            assert repo.connection == db_conn
            assert repo.table_name == "test_table"
            print("   ✓ Database connection and repository")

            return True
        except Exception as e:
            print(f"   ✗ Database test failed: {e}")
            return False

    async def test_integration(self) -> bool:
        """Test integration components."""
        try:
            # Test container initialization
            config = get_settings()
            container = init_container(config)

            # Test basic service access
            db_conn = container.database_connection()
            assert db_conn is not None
            print("   ✓ Dependency injection container")

            return True
        except Exception as e:
            print(f"   ✗ Integration test failed: {e}")
            return False

    async def test_performance(self) -> bool:
        """Test performance components."""
        try:
            from infrastructure.cache import InMemoryCache

            # Test in-memory cache
            cache = InMemoryCache()

            # Test basic operations
            await cache.set("test_key", "test_value", ttl=60)
            value = await cache.get("test_key")
            assert value == "test_value"

            exists = await cache.exists("test_key")
            assert exists is True

            await cache.delete("test_key")
            deleted_value = await cache.get("test_key")
            assert deleted_value is None

            print("   ✓ In-memory cache performance")

            return True
        except Exception as e:
            print(f"   ✗ Performance test failed: {e}")
            return False

    async def test_health_checks(self) -> bool:
        """Test health check functionality."""
        try:
            # Test configuration health
            config = get_settings()
            assert config.app_name == "LegacyCoinTrader"
            print("   ✓ Configuration health")

            # Test import health
            import domain.models
            import core.config
            import infrastructure.database
            print("   ✓ Module imports")

            return True
        except Exception as e:
            print(f"   ✗ Health check failed: {e}")
            return False

    def run_pytest_suite(self) -> bool:
        """Run the pytest test suite."""
        try:
            print("🧪 Running pytest test suite...")

            # Change to project root
            os.chdir(self.project_root)

            # Run pytest with coverage
            result = subprocess.run([
                "python", "-m", "pytest",
                "modern/tests/",
                "--tb=short",
                "--disable-warnings",
                "-v",
                "--cov=modern",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-fail-under=80"
            ], capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print("✅ Pytest suite passed!")
                print(result.stdout.split('\n')[-10:])  # Show last 10 lines (coverage summary)
                return True
            else:
                print("❌ Pytest suite failed!")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False

        except subprocess.TimeoutExpired:
            print("❌ Pytest suite timed out")
            return False
        except Exception as e:
            print(f"❌ Pytest execution error: {e}")
            return False

    def run_code_quality_checks(self) -> bool:
        """Run code quality checks."""
        try:
            print("🔍 Running code quality checks...")

            checks = [
                ("Black formatting", ["python", "-m", "black", "--check", "--diff", "modern/"]),
                ("isort import sorting", ["python", "-m", "isort", "--check-only", "--diff", "modern/"]),
                ("flake8 linting", ["python", "-m", "flake8", "modern/"]),
            ]

            all_passed = True
            for check_name, cmd in checks:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"   ✅ {check_name}: PASSED")
                else:
                    print(f"   ❌ {check_name}: FAILED")
                    print(f"   Output: {result.stdout}")
                    if result.stderr:
                        print(f"   Error: {result.stderr}")
                    all_passed = False

            return all_passed

        except Exception as e:
            print(f"❌ Code quality check error: {e}")
            return False


async def main():
    """Main testing function."""
    print("🧪 LegacyCoinTrader 2.0 Local Testing Suite")
    print("=" * 50)

    # Set minimal environment for testing
    import os
    os.environ.setdefault("ENVIRONMENT", "development")
    os.environ.setdefault("APP_NAME", "LegacyCoinTrader")
    os.environ.setdefault("VERSION", "2.0.0")
    os.environ.setdefault("EXCHANGE_NAME", "kraken")  # ExchangeConfig uses EXCHANGE_ prefix
    os.environ.setdefault("EXCHANGE_API_KEY", "test_key")
    os.environ.setdefault("EXCHANGE_API_SECRET", "test_secret")
    os.environ.setdefault("SECURITY_JWT_SECRET_KEY", "test_jwt_secret")
    os.environ.setdefault("TRADING_EXECUTION_MODE", "dry_run")
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

    # Initialize configuration
    try:
        config = init_config()
        print(f"✅ Configuration loaded: {config.app_name} v{config.version}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run comprehensive tests
    tester = LocalTester()

    # Run our custom tests
    print("\n📋 Running custom integration tests...")
    test_results = await tester.run_all_tests()

    # Run pytest suite
    print("\n🧪 Running pytest test suite...")
    pytest_success = tester.run_pytest_suite()

    # Run code quality checks
    print("\n🔍 Running code quality checks...")
    quality_success = tester.run_code_quality_checks()

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    print("Custom Tests:")
    for test_name, success in test_results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")

    print(f"\nPytest Suite: {'✅ PASSED' if pytest_success else '❌ FAILED'}")
    print(f"Code Quality: {'✅ PASSED' if quality_success else '❌ FAILED'}")

    # Overall result
    overall_success = (
        all(test_results.values()) and
        pytest_success and
        quality_success
    )

    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")

    if overall_success:
        print("\n🎉 Congratulations! Your modernized trading system is ready for development!")
        print("\nNext steps:")
        print("1. Run: python modern/src/trading_bot.py")
        print("2. Check: http://localhost:8000/docs (FastAPI docs)")
        print("3. Monitor: http://localhost:8000/health")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        print("You can run individual tests with:")
        print("  python -m pytest modern/tests/unit/test_config.py -v")
        print("  python -m pytest modern/tests/unit/test_security.py -v")
        print("  etc.")


if __name__ == "__main__":
    # Set environment variable for testing
    os.environ.setdefault("ENVIRONMENT", "testing")

    # Run async main
    asyncio.run(main())
