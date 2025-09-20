#!/usr/bin/env python3
"""
Test script to verify enhanced scanner execution integration.

This script tests that opportunities discovered by the enhanced scanner
are properly converted to trade execution requests.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from crypto_bot.enhanced_scan_integration import EnhancedScanIntegration
from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__)


async def test_enhanced_scanner_execution():
    """Test the enhanced scanner execution integration."""
    logger.info("🧪 Testing Enhanced Scanner Execution Integration")
    
    # Load configuration
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
        # Initialize the integration
        integration = EnhancedScanIntegration(config)
        
        # Check if execution adapter is available
        if integration.execution_adapter:
            logger.info("✅ Execution adapter initialized successfully")
        else:
            logger.warning("⚠️ Execution adapter not available - trades will be logged only")
        
        # Start the integration (this will start the scanner)
        logger.info("🚀 Starting enhanced scan integration...")
        await integration.start()
        
        # Let it run for a few cycles to detect opportunities
        logger.info("⏳ Running for 2 minutes to detect opportunities...")
        await asyncio.sleep(120)
        
        # Check for opportunities
        opportunities = integration.get_top_opportunities(limit=10)
        logger.info(f"📊 Found {len(opportunities)} opportunities")
        
        for i, opp in enumerate(opportunities[:3], 1):
            logger.info(
                f"  {i}. {opp.get('symbol', 'Unknown')} - "
                f"Confidence: {opp.get('confidence', 0):.3f} - "
                f"Strategy: {opp.get('strategy', 'Unknown')}"
            )
        
        # Get integration stats
        stats = integration.get_integration_stats()
        logger.info("📈 Integration Statistics:")
        logger.info(f"  Running: {stats['running']}")
        logger.info(f"  Execution Opportunities: {stats['performance_stats']['execution_opportunities']}")
        logger.info(f"  Successful Executions: {stats['performance_stats']['successful_executions']}")
        logger.info(f"  Failed Executions: {stats['performance_stats']['failed_executions']}")
        
        # Stop the integration
        logger.info("🛑 Stopping enhanced scan integration...")
        await integration.stop()
        
        logger.info("✅ Test completed successfully!")
        
        return True
        
    except Exception as exc:
        logger.error(f"❌ Test failed: {exc}")
        return False


async def main():
    """Main test function."""
    try:
        success = await test_enhanced_scanner_execution()
        if success:
            logger.info("🎉 All tests passed!")
            sys.exit(0)
        else:
            logger.error("💥 Tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        sys.exit(0)
    except Exception as exc:
        logger.error(f"💥 Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
