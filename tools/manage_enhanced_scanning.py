#!/usr/bin/env python3
"""
Enhanced Scanning Management Tool

Command-line interface for managing the enhanced scanning system with
integrated caching and continuous strategy review.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from crypto_bot.enhanced_scan_integration import (
    start_enhanced_scan_integration,
    stop_enhanced_scan_integration,
    get_enhanced_scan_integration,
    get_integration_stats
)
from crypto_bot.utils.logger import setup_logger, LOG_DIR
from crypto_bot.utils.telegram import TelegramNotifier

logger = setup_logger(__name__, LOG_DIR / "enhanced_scanning.log")


class EnhancedScanCLI:
    """Command-line interface for enhanced scanning management."""
    
    def __init__(self):
        self.config = self._load_config()
        self.notifier = self._create_notifier()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load bot configuration."""
        try:
            config_path = project_root / "crypto_bot" / "config.yaml"
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info("Loaded bot configuration")
                return config
            else:
                logger.warning("Bot config not found, using defaults")
                return {"telegram": {"enabled": False}}
        except Exception as exc:
            logger.error(f"Failed to load config: {exc}")
            return {"telegram": {"enabled": False}}
    
    def _create_notifier(self) -> TelegramNotifier:
        """Create Telegram notifier if configured."""
        try:
            if self.config.get("telegram", {}).get("enabled", False):
                return TelegramNotifier(self.config)
            else:
                return None
        except Exception as exc:
            logger.warning(f"Failed to create Telegram notifier: {exc}")
            return None
    
    async def start_scanning(self):
        """Start the enhanced scanning system."""
        try:
            logger.info("Starting enhanced scanning system...")
            await start_enhanced_scan_integration(self.config, self.notifier)
            logger.info("Enhanced scanning system started successfully")
            
            if self.notifier:
                self.notifier.notify("🚀 Enhanced scanning system started")
                
        except Exception as exc:
            logger.error(f"Failed to start enhanced scanning: {exc}")
            sys.exit(1)
    
    async def stop_scanning(self):
        """Stop the enhanced scanning system."""
        try:
            logger.info("Stopping enhanced scanning system...")
            await stop_enhanced_scan_integration()
            logger.info("Enhanced scanning system stopped successfully")
            
            if self.notifier:
                self.notifier.notify("🛑 Enhanced scanning system stopped")
                
        except Exception as exc:
            logger.error(f"Failed to stop enhanced scanning: {exc}")
            sys.exit(1)
    
    async def show_status(self):
        """Show the current status of the enhanced scanning system."""
        try:
            stats = get_integration_stats()
            
            if "error" in stats:
                print("❌ Enhanced scanning system not running")
                return
            
            print("\n🔍 Enhanced Scanning System Status")
            print("=" * 50)
            
            # Basic status
            status = "🟢 Running" if stats.get("running", False) else "🔴 Stopped"
            print(f"Status: {status}")
            
            # Cache statistics
            cache_stats = stats.get("cache_stats", {})
            print(f"\n📊 Cache Statistics:")
            print(f"  Scan Results: {cache_stats.get('scan_results', 0)}")
            print(f"  Strategy Fits: {cache_stats.get('strategy_fits', 0)}")
            print(f"  Execution Opportunities: {cache_stats.get('execution_opportunities', 0)}")
            print(f"  Review Queue Size: {cache_stats.get('review_queue_size', 0)}")
            print(f"  Cache Size Limit: {cache_stats.get('cache_size_limit', 0)}")
            
            # Scanner statistics
            scanner_stats = stats.get("scanner_stats", {})
            print(f"\n🔍 Scanner Statistics:")
            print(f"  Total Scans: {scanner_stats.get('total_scans', 0)}")
            print(f"  Tokens Discovered: {scanner_stats.get('tokens_discovered', 0)}")
            print(f"  Tokens Cached: {scanner_stats.get('tokens_cached', 0)}")
            print(f"  Last Scan: {scanner_stats.get('last_scan_time', 0)}")
            
            # Performance statistics
            perf_stats = stats.get("performance_stats", {})
            print(f"\n⚡ Performance Statistics:")
            print(f"  Cache Hits: {perf_stats.get('cache_hits', 0)}")
            print(f"  Cache Misses: {perf_stats.get('cache_misses', 0)}")
            print(f"  Strategy Analyses: {perf_stats.get('strategy_analyses', 0)}")
            print(f"  Execution Opportunities: {perf_stats.get('execution_opportunities', 0)}")
            print(f"  Integration Errors: {perf_stats.get('integration_errors', 0)}")
            
            # Calculate cache hit rate
            total_access = perf_stats.get("cache_hits", 0) + perf_stats.get("cache_misses", 0)
            if total_access > 0:
                hit_rate = (perf_stats.get("cache_hits", 0) / total_access) * 100
                print(f"  Cache Hit Rate: {hit_rate:.1f}%")
            
            print("=" * 50)
            
        except Exception as exc:
            logger.error(f"Failed to show status: {exc}")
            print(f"❌ Error getting status: {exc}")
    
    async def show_opportunities(self, limit: int = 10):
        """Show top execution opportunities."""
        try:
            integration = get_enhanced_scan_integration(self.config, self.notifier)
            opportunities = integration.get_top_opportunities(limit)
            
            if not opportunities:
                print("📭 No execution opportunities available")
                return
            
            print(f"\n🎯 Top {len(opportunities)} Execution Opportunities")
            print("=" * 80)
            
            for i, opp in enumerate(opportunities, 1):
                print(f"\n{i}. {opp.symbol}")
                print(f"   Strategy: {opp.strategy}")
                print(f"   Direction: {opp.direction.upper()}")
                print(f"   Confidence: {opp.confidence:.1%}")
                print(f"   Entry Price: ${opp.entry_price:.6f}")
                print(f"   Stop Loss: ${opp.stop_loss:.6f}")
                print(f"   Take Profit: ${opp.take_profit:.6f}")
                print(f"   Risk/Reward: {opp.risk_reward_ratio:.2f}")
                print(f"   Position Size: {opp.position_size:.6f}")
                print(f"   Status: {opp.status}")
                print(f"   Timestamp: {opp.timestamp}")
            
            print("=" * 80)
            
        except Exception as exc:
            logger.error(f"Failed to show opportunities: {exc}")
            print(f"❌ Error getting opportunities: {exc}")
    
    async def force_scan(self):
        """Force an immediate scan cycle."""
        try:
            integration = get_enhanced_scan_integration(self.config, self.notifier)
            await integration.force_scan()
            print("🔍 Forced scan cycle completed")
            
        except Exception as exc:
            logger.error(f"Failed to force scan: {exc}")
            print(f"❌ Error forcing scan: {exc}")
    
    async def clear_cache(self):
        """Clear all scan caches."""
        try:
            integration = get_enhanced_scan_integration(self.config, self.notifier)
            await integration.clear_cache()
            print("🗑️ All scan caches cleared")
            
        except Exception as exc:
            logger.error(f"Failed to clear cache: {exc}")
            print(f"❌ Error clearing cache: {exc}")
    
    async def show_cache_details(self):
        """Show detailed cache information."""
        try:
            integration = get_enhanced_scan_integration(self.config, self.notifier)
            cache_stats = integration.cache_manager.get_cache_stats()
            
            print("\n📋 Detailed Cache Information")
            print("=" * 50)
            
            for key, value in cache_stats.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value}")
                else:
                    print(f"{key}: {value}")
            
            print("=" * 50)
            
        except Exception as exc:
            logger.error(f"Failed to show cache details: {exc}")
            print(f"❌ Error getting cache details: {exc}")
    
    async def export_stats(self, filename: str):
        """Export statistics to a JSON file."""
        try:
            stats = get_integration_stats()
            
            if "error" in stats:
                print("❌ No statistics available for export")
                return
            
            # Add timestamp
            import time
            stats["export_timestamp"] = time.time()
            stats["export_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Write to file
            with open(filename, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            print(f"📊 Statistics exported to {filename}")
            
        except Exception as exc:
            logger.error(f"Failed to export stats: {exc}")
            print(f"❌ Error exporting stats: {exc}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Scanning Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_enhanced_scanning.py start          # Start the system
  python manage_enhanced_scanning.py stop           # Stop the system
  python manage_enhanced_scanning.py status         # Show current status
  python manage_enhanced_scanning.py opportunities  # Show top opportunities
  python manage_enhanced_scanning.py force-scan     # Force immediate scan
  python manage_enhanced_scanning.py clear-cache    # Clear all caches
  python manage_enhanced_scanning.py export-stats   # Export statistics
        """
    )
    
    parser.add_argument(
        "command",
        choices=[
            "start", "stop", "status", "opportunities", "force-scan",
            "clear-cache", "cache-details", "export-stats"
        ],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10,
        help="Number of opportunities to show (default: 10)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="enhanced_scan_stats.json",
        help="Output filename for export (default: enhanced_scan_stats.json)"
    )
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = EnhancedScanCLI()
    
    # Execute command
    try:
        if args.command == "start":
            asyncio.run(cli.start_scanning())
        elif args.command == "stop":
            asyncio.run(cli.stop_scanning())
        elif args.command == "status":
            asyncio.run(cli.show_status())
        elif args.command == "opportunities":
            asyncio.run(cli.show_opportunities(args.limit))
        elif args.command == "force-scan":
            asyncio.run(cli.force_scan())
        elif args.command == "clear-cache":
            asyncio.run(cli.clear_cache())
        elif args.command == "cache-details":
            asyncio.run(cli.show_cache_details())
        elif args.command == "export-stats":
            asyncio.run(cli.export_stats(args.output))
        else:
            print(f"❌ Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(0)
    except Exception as exc:
        logger.error(f"CLI error: {exc}")
        print(f"❌ Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
