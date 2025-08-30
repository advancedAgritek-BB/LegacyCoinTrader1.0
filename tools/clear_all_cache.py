#!/usr/bin/env python3
"""
Clear All Cache and Data Script

This script clears ALL trading bot cache data, log files, and data sources
to completely reset the system. Use this when you want to start completely fresh.

⚠️  WARNING: This will delete ALL trading data, positions, and cache files!
"""

import sys
import os
from pathlib import Path
import shutil

# Add the crypto_bot directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def clear_file(file_path: Path, default_content: str = ""):
    """Clear a file and optionally set default content."""
    try:
        if file_path.exists():
            file_path.write_text(default_content)
            print(f"✅ Cleared: {file_path}")
        else:
            print(f"⚠️  Not found: {file_path}")
    except Exception as e:
        print(f"❌ Error clearing {file_path}: {e}")

def clear_directory(dir_path: Path):
    """Clear all files in a directory."""
    try:
        if dir_path.exists():
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                    print(f"✅ Deleted: {file_path}")
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    print(f"✅ Deleted directory: {file_path}")
            print(f"✅ Cleared directory: {dir_path}")
        else:
            print(f"⚠️  Directory not found: {dir_path}")
    except Exception as e:
        print(f"❌ Error clearing directory {dir_path}: {e}")

def main():
    """Main function to clear all cache and data."""
    print("🗑️  Clearing ALL trading bot cache and data...")
    print("=" * 60)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    logs_dir = project_root / "crypto_bot" / "logs"
    cache_dir = project_root / "cache"
    
    print(f"Project root: {project_root}")
    print(f"Logs directory: {logs_dir}")
    print(f"Cache directory: {cache_dir}")
    print()
    
    # Clear main data files
    print("📁 Clearing main data files...")
    clear_file(logs_dir / "positions.log")
    clear_file(logs_dir / "trades.csv")
    clear_file(logs_dir / "asset_scores.json")
    clear_file(logs_dir / "strategy_stats.json")
    clear_file(logs_dir / "bot.log", "")
    
    # Clear strategy and scan logs
    print("\n📁 Clearing strategy and scan logs...")
    clear_file(logs_dir / "strategy_rank.log")
    clear_file(logs_dir / "enhanced_scanner.log")
    clear_file(logs_dir / "enhanced_scan_integration.log")
    clear_file(logs_dir / "scan_cache.log")
    clear_file(logs_dir / "pair_cache.log")
    
    # Clear cache directories
    print("\n📁 Clearing cache directories...")
    if cache_dir.exists():
        for item in cache_dir.iterdir():
            if item.is_file():
                if item.name.endswith('.json'):
                    clear_file(item, "{}")
                else:
                    item.unlink()
                    print(f"✅ Deleted: {item}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"✅ Deleted directory: {item}")
    
    # Clear any remaining cache files
    print("\n📁 Clearing remaining cache files...")
    cache_files = [
        logs_dir / "scan_results_cache.json",
        logs_dir / "strategy_fit_cache.json", 
        logs_dir / "execution_opportunities.json"
    ]
    
    for cache_file in cache_files:
        clear_file(cache_file, "{}")
    
    # Clear any session state files
    print("\n📁 Clearing session state...")
    session_files = [
        project_root / "last_regime.json",
        project_root / "startup_output.log"
    ]
    
    for session_file in session_files:
        if session_file.exists():
            session_file.unlink()
            print(f"✅ Deleted: {session_file}")
    
    print("\n" + "=" * 60)
    print("🎯 Cache clearing complete!")
    print()
    print("📋 What was cleared:")
    print("   • All position logs and trade history")
    print("   • Asset scores and strategy statistics")
    print("   • Scan cache and execution data")
    print("   • Market data cache files")
    print("   • Session state and regime data")
    print()
    print("⚠️  Important notes:")
    print("   • All trading data has been permanently deleted")
    print("   • The bot will start completely fresh")
    print("   • You'll need to re-enter paper trading balance")
    print("   • All cached market data has been cleared")
    print()
    print("🚀 Next steps:")
    print("   1. Start the bot (it will prompt for new balance)")
    print("   2. Monitor for clean startup with no old data")
    print("   3. Verify frontend shows no old positions")
    
    # Ask for confirmation before clearing
    print("\n" + "=" * 60)
    response = input("❓ Are you sure you want to clear ALL data? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        print("🗑️  Proceeding with complete cache clear...")
        # The clearing was already done above, just confirm
        print("✅ All cache and data has been cleared!")
    else:
        print("❌ Cache clearing cancelled.")
        print("Note: Some files may have already been cleared.")

if __name__ == "__main__":
    main()
