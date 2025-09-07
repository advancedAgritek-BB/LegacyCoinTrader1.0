#!/usr/bin/env python3
"""
Local Development Startup Script for LegacyCoinTrader 2.0

This script provides easy local development setup and startup.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_environment():
    """Check if environment is properly set up."""
    print("🔍 Checking environment setup...")

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 9):
        print(f"❌ Python {python_version.major}.{python_version.minor} detected. Need Python 3.9+")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.minor}")

    # Check virtual environment
    if not hasattr(sys, 'real_prefix') and sys.base_prefix == sys.prefix:
        print("⚠️  Warning: Not running in virtual environment")
        print("   Consider using: python3 -m venv modern_trader_env && source modern_trader_env/bin/activate")
    else:
        print("✅ Running in virtual environment")

    return True

def setup_environment():
    """Setup local development environment."""
    print("\n📋 Setting up local environment...")

    # Copy environment file
    env_example = Path("env_local_example")
    env_local = Path(".env.local")

    if env_example.exists() and not env_local.exists():
        env_local.write_text(env_example.read_text())
        print("✅ Created .env.local from template")
        print("   ⚠️  Please edit .env.local with your actual API keys for full testing")
    elif env_local.exists():
        print("✅ .env.local already exists")
    else:
        print("❌ env_local_example not found")

    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print("✅ Created logs directory")

    # Create database directory
    db_dir = Path("data")
    db_dir.mkdir(exist_ok=True)
    print("✅ Created data directory")

def install_dependencies():
    """Install/update dependencies."""
    print("\n📦 Installing dependencies...")

    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "-r", "requirements-modern.txt", "--quiet"
        ], check=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_tests():
    """Run the test suite."""
    print("\n🧪 Running test suite...")

    try:
        result = subprocess.run([
            sys.executable, "test_locally.py"
        ], timeout=300)

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out")
        return False
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def start_application():
    """Start the application."""
    print("\n🚀 Starting LegacyCoinTrader 2.0...")

    try:
        # Set environment variable
        os.environ["ENV_FILE"] = ".env.local"

        # Start the application
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "modern.src.interfaces.api:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")

def show_menu():
    """Show the main menu."""
    print("\n" + "="*50)
    print("🏠 LegacyCoinTrader 2.0 Local Development")
    print("="*50)
    print("1. Setup environment")
    print("2. Install dependencies")
    print("3. Run tests")
    print("4. Start application")
    print("5. Run full setup (1-3)")
    print("6. Show status")
    print("0. Exit")
    print("="*50)

def show_status():
    """Show current status."""
    print("\n📊 System Status")

    # Check files
    files_to_check = [
        (".env.local", "Environment configuration"),
        ("requirements-modern.txt", "Dependencies file"),
        ("modern/src/", "Modern source code"),
        ("logs/", "Logs directory"),
        ("data/", "Data directory")
    ]

    for file_path, description in files_to_check:
        exists = Path(file_path).exists()
        status = "✅" if exists else "❌"
        print(f"{status} {description}: {file_path}")

    # Check Python modules
    modules_to_check = [
        ("fastapi", "Web framework"),
        ("pydantic", "Data validation"),
        ("pytest", "Testing framework"),
        ("sqlalchemy", "Database ORM")
    ]

    print("\n📦 Python Modules:")
    for module, description in modules_to_check:
        try:
            __import__(module)
            print(f"✅ {description}: {module}")
        except ImportError:
            print(f"❌ {description}: {module} (not installed)")

def main():
    """Main function."""
    print("🏠 LegacyCoinTrader 2.0 Local Development Setup")
    print("=" * 55)

    if not check_environment():
        print("❌ Environment check failed. Please fix issues above.")
        return

    while True:
        show_menu()
        try:
            choice = input("Choose an option (0-6): ").strip()

            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                setup_environment()
            elif choice == "2":
                install_dependencies()
            elif choice == "3":
                success = run_tests()
                if success:
                    print("✅ All tests passed!")
                else:
                    print("❌ Some tests failed")
            elif choice == "4":
                start_application()
            elif choice == "5":
                print("🔄 Running full setup...")
                setup_environment()
                if install_dependencies():
                    if run_tests():
                        print("✅ Full setup completed successfully!")
                    else:
                        print("⚠️  Setup completed but tests failed")
                else:
                    print("❌ Setup failed during dependency installation")
            elif choice == "6":
                show_status()
            else:
                print("❌ Invalid choice. Please select 0-6.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
