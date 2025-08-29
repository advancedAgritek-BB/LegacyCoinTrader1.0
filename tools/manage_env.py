#!/usr/bin/env python3
"""
Environment File Manager for LegacyCoinTrader

This script helps manage .env files across the application to ensure consistency
and prevent overwriting of real API keys with templates.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

# Define possible .env file locations
ENV_LOCATIONS = [
    Path(".env"),  # Root directory
    Path("crypto_bot/.env"),  # crypto_bot subdirectory
]

# Template placeholder values to detect template files
TEMPLATE_PLACEHOLDERS = [
    "your_kraken_api_key_here",
    "your_telegram_token_here", 
    "your_helius_key_here",
    "your_kraken_api_secret_here",
    "your_coinbase_api_key_here",
    "your_coinbase_api_secret_here",
    "your_wallet_address_here",
    "your_solana_private_key_here",
    "your_supabase_url_here",
    "your_supabase_key_here",
    "your_lunarcrush_api_key_here"
]

def is_template_file(env_path: Path) -> bool:
    """Check if a .env file contains template placeholder values."""
    if not env_path.exists():
        return False
    
    try:
        content = env_path.read_text()
        return any(placeholder in content for placeholder in TEMPLATE_PLACEHOLDERS)
    except Exception:
        return False

def find_env_files() -> List[Tuple[Path, bool, bool]]:
    """Find all .env files and determine if they're templates or contain real keys."""
    results = []
    
    for env_path in ENV_LOCATIONS:
        exists = env_path.exists()
        is_template = is_template_file(env_path) if exists else False
        results.append((env_path, exists, is_template))
    
    return results

def get_real_env_file() -> Optional[Path]:
    """Get the first .env file that contains real API keys."""
    for env_path, exists, is_template in find_env_files():
        if exists and not is_template:
            return env_path
    return None

def create_env_template(target_path: Path) -> None:
    """Create a new .env template file."""
    template_content = """# .env File for LegacyCoinTrader
# Updated with actual API keys and configuration

# Exchange Configuration
EXCHANGE=kraken
API_KEY=your_kraken_api_key_here
API_SECRET=your_kraken_api_secret_here
KRAKEN_API_KEY=your_kraken_api_key_here
KRAKEN_API_SECRET=your_kraken_api_secret_here

# Alternative Exchange (Coinbase)
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_API_SECRET=your_coinbase_api_secret_here
# COINBASE_PASSPHRASE=your_coinbase_passphrase_here

# Telegram Configuration
TELEGRAM_TOKEN=your_telegram_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TELE_CHAT_ADMINS=your_admin_chat_id_here

# Solana Configuration
HELIUS_KEY=your_helius_key_here
WALLET_ADDRESS=your_wallet_address_here
SOLANA_PRIVATE_KEY=your_solana_private_key_here

# Supabase Configuration
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_key_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here

# LunarCrush Sentiment Analysis (Optional)
LUNARCRUSH_API_KEY=your_lunarcrush_api_key_here

# Trading Mode
MODE=cex
EXECUTION_MODE=dry_run

# CloudTrader Configuration
CT_MODELS_BUCKET=models
CT_REGIME_PREFIX=
CT_SYMBOL=XRPUSD
"""
    
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(template_content)

def consolidate_env_files() -> None:
    """Consolidate .env files to ensure consistency across the application."""
    print("🔍 Checking .env file locations...")
    
    env_files = find_env_files()
    real_env = get_real_env_file()
    
    print("\n📁 Current .env file status:")
    for env_path, exists, is_template in env_files:
        status = "✅ Real keys" if exists and not is_template else "📝 Template" if exists else "❌ Missing"
        print(f"  {env_path}: {status}")
    
    if not real_env:
        print("\n❌ No .env file with real API keys found!")
        print("Creating new template in root directory...")
        create_env_template(Path(".env"))
        print("✅ Created .env template. Please edit with your real API keys.")
        return
    
    print(f"\n✅ Found .env file with real keys: {real_env}")
    
    # Check if we need to create a copy in the root directory for the startup script
    root_env = Path(".env")
    if not root_env.exists():
        print("📋 Creating copy in root directory for startup script compatibility...")
        shutil.copy2(real_env, root_env)
        print("✅ Created .env in root directory")
    
    # Check if we need to create a copy in crypto_bot/ for the main application
    crypto_bot_env = Path("crypto_bot/.env")
    if not crypto_bot_env.exists():
        print("📋 Creating copy in crypto_bot/ directory for main application...")
        shutil.copy2(real_env, crypto_bot_env)
        print("✅ Created .env in crypto_bot/ directory")
    
    print("\n✅ Environment files are now consistent across the application!")

def validate_env_file(env_path: Path) -> bool:
    """Validate that a .env file contains all required keys."""
    if not env_path.exists():
        print(f"❌ {env_path} does not exist")
        return False
    
    try:
        content = env_path.read_text()
        lines = content.split('\n')
        
        required_keys = [
            "API_KEY", "API_SECRET", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"
        ]
        
        missing_keys = []
        for key in required_keys:
            if not any(line.startswith(f"{key}=") for line in lines):
                missing_keys.append(key)
        
        if missing_keys:
            print(f"❌ {env_path} is missing required keys: {', '.join(missing_keys)}")
            return False
        
        # Check for template values
        if is_template_file(env_path):
            print(f"❌ {env_path} contains template placeholder values")
            return False
        
        print(f"✅ {env_path} is valid and contains real API keys")
        return True
        
    except Exception as e:
        print(f"❌ Error reading {env_path}: {e}")
        return False

def main():
    """Main function to manage environment files."""
    print("🔧 LegacyCoinTrader Environment File Manager")
    print("=" * 50)
    
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "consolidate":
            consolidate_env_files()
        elif command == "validate":
            print("🔍 Validating .env files...")
            for env_path, exists, _ in find_env_files():
                if exists:
                    validate_env_file(env_path)
        elif command == "status":
            env_files = find_env_files()
            print("📁 .env file status:")
            for env_path, exists, is_template in env_files:
                status = "✅ Real keys" if exists and not is_template else "📝 Template" if exists else "❌ Missing"
                print(f"  {env_path}: {status}")
        elif command == "help":
            print("Available commands:")
            print("  consolidate  - Consolidate .env files across locations")
            print("  validate     - Validate existing .env files")
            print("  status       - Show status of all .env files")
            print("  help         - Show this help message")
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'help' to see available commands")
    else:
        # Default action: consolidate
        consolidate_env_files()

if __name__ == "__main__":
    main()
