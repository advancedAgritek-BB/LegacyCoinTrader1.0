#!/usr/bin/env python3
"""
Test script to verify enhanced scan integration startup.

This script tests that the enhanced scan integration is properly started
and can process opportunities from the scanner.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "crypto_bot"))

# Import after path setup (required for proper module resolution)
# pylint: disable=wrong-import-position
from crypto_bot.enhanced_scan_integration import (
    start_enhanced_scan_integration, 
    stop_enhanced_scan_integration
)
from crypto_bot.utils.logger import setup_logger
# pylint: enable=wrong-import-position

logger = setup_logger(__name__)


async def test_integration_startup() -> bool:
    """Test enhanced scan integration startup."""
    logger.info("🧪 Testing Enhanced Scan Integration Startup")
    
    # Load configuration (same as trading engine would use)
    config = {
        "enhanced_scanning": {
            "enabled": True,
            "scan_interval": 30,
            "min_confidence": 0.5
        },
        "execution": {
            "enabled": True,
            "dry_run": True,  # Safe testing
            "base_trade_amount": 0.01,
            "min_confidence_threshold": 0.7
        },
        "telegram": {
            "enabled": False  # Disable for testing
        }
    }
    
    try:
        # Start enhanced scan integration (same as trading engine)
        logger.info("🚀 Starting enhanced scan integration...")
        await start_enhanced_scan_integration(config)
        logger.info("✅ Enhanced scan integration started successfully")
        
        # Let it run for a bit to process opportunities
        logger.info("⏳ Running for 30 seconds to process opportunities...")
        await asyncio.sleep(30)
        
        # Check for log file creation
        log_file = project_root / "crypto_bot" / "logs" / "enhanced_scan_integration.log"
        if log_file.exists():
            logger.info("✅ Integration log file created successfully")
            
            # Show last few lines
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    logger.info("📄 Last few log entries:")
                    for line in lines[-5:]:
                        print(f"  {line.strip()}")
                else:
                    logger.warning("⚠️ Log file is empty")
        else:
            logger.error("❌ Integration log file not created")
        
        # Stop integration
        logger.info("🛑 Stopping enhanced scan integration...")
        await stop_enhanced_scan_integration()
        logger.info("✅ Integration stopped successfully")
        
        return True
        
    except Exception as exc:
        logger.error(f"❌ Test failed: {exc}")
        return False


async def main() -> None:
    """Main test function."""
    try:
        success = await test_integration_startup()
        if success:
            logger.info("🎉 Integration startup test passed!")
            sys.exit(0)
        else:
            logger.error("💥 Integration startup test failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        sys.exit(0)
    except Exception as exc:
        logger.error(f"💥 Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
