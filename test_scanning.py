#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_scanning():
    """Test the scanning and token analysis system."""
    try:
        from crypto_bot.solana.scanner import get_solana_new_tokens
        from crypto_bot.utils.logger import setup_logger, LOG_DIR
        
        logger = setup_logger("test_scanning", LOG_DIR / "test_scanning.log")
        
        # Load config
        import yaml
        with open("crypto_bot/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        scanner_config = config.get("solana_scanner", {})
        
        print("🔍 Testing Solana scanner...")
        logger.info("Starting scanner test")
        
        # Test scanner
        tokens = await get_solana_new_tokens(scanner_config)
        
        print(f"✅ Scanner test successful! Found {len(tokens)} tokens")
        logger.info(f"Scanner test completed: {len(tokens)} tokens found")
        
        if tokens:
            print("📋 Sample tokens:")
            for i, token in enumerate(tokens[:5]):
                print(f"   {i+1}. {token}")
        
        return True
        
    except Exception as e:
        print(f"❌ Scanner test failed: {e}")
        logger.error(f"Scanner test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_scanning())
    sys.exit(0 if success else 1)
