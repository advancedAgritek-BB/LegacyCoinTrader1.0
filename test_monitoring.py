#!/usr/bin/env python3
"""
Test script for monitoring system functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add crypto_bot to path
sys.path.insert(0, str(Path(__file__).parent / "crypto_bot"))

async def test_monitoring():
    """Test the monitoring system."""
    try:
        from crypto_bot.pipeline_monitor import PipelineMonitor
        from crypto_bot.main import load_config

        print("Testing monitoring system...")

        # Load config
        config = load_config()
        print("✅ Config loaded")

        # Create monitoring instance
        monitor = PipelineMonitor(config)
        print("✅ Monitoring instance created")

        # Perform health check
        print("Performing health check...")
        health_status = await monitor.perform_health_check()
        print(f"✅ Health check completed, found {len(health_status)} components")

        # Get health summary
        summary = monitor.get_health_summary()
        print(f"✅ Health summary: {summary['overall_status']}")

        # Check if frontend status file was created
        frontend_file = Path("crypto_bot/logs/frontend_monitoring_status.json")
        if frontend_file.exists():
            print(f"✅ Frontend status file created: {frontend_file}")
            with open(frontend_file, 'r') as f:
                import json
                data = json.load(f)
                print(f"   Status: {data.get('overall_status', 'unknown')}")
                print(f"   Components: {len(data.get('components', {}))}")
        else:
            print("❌ Frontend status file not created")

        print("Test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_monitoring())
