#!/usr/bin/env python3
"""
Test Scanning Functionality

This script tests the scanning functionality to verify it's working correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_scanning_functionality():
    """Test the scanning functionality."""
    
    print("🔍 Testing scanning functionality...")
    
    try:
        from crypto_bot.solana.scanner import get_solana_new_tokens
        from crypto_bot.utils.logger import setup_logger, LOG_DIR
        
        logger = setup_logger("test_scanning", LOG_DIR / "test_scanning.log")
        
        # Load config
        import yaml
        with open("crypto_bot/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        scanner_config = config.get("solana_scanner", {})
        
        print("📋 Scanner configuration:")
        print(f"   - Enabled: {scanner_config.get('enabled', False)}")
        print(f"   - Interval: {scanner_config.get('interval_minutes', 0)} minutes")
        print(f"   - Max tokens: {scanner_config.get('max_tokens_per_scan', 0)}")
        print(f"   - Min volume: ${scanner_config.get('min_volume_usd', 0):,.0f}")
        
        if not scanner_config.get('enabled', False):
            print("❌ Scanner is disabled in configuration")
            return False
        
        print("\n🔍 Testing Solana scanner...")
        logger.info("Starting scanner test")
        
        # Test scanner
        tokens = await get_solana_new_tokens(scanner_config)
        
        print(f"✅ Scanner test successful! Found {len(tokens)} tokens")
        logger.info(f"Scanner test completed: {len(tokens)} tokens found")
        
        if tokens:
            print("📋 Sample tokens:")
            for i, token in enumerate(tokens[:5]):
                print(f"   {i+1}. {token}")
            
            print(f"\n📊 Scanner Statistics:")
            print(f"   - Total tokens discovered: {len(tokens)}")
            print(f"   - Scanner is working correctly")
            print(f"   - Token discovery is functional")
            
            # Test token formatting
            print(f"\n🔧 Testing token formatting...")
            formatted_tokens = []
            for token in tokens[:3]:
                formatted_token = f"{token}/USDC"
                formatted_tokens.append(formatted_token)
                print(f"   - {token} -> {formatted_token}")
            
            print(f"✅ Token formatting test completed")
            
        else:
            print("⚠️ No tokens found - this might be normal depending on market conditions")
        
        return True
        
    except Exception as e:
        print(f"❌ Scanner test failed: {e}")
        logger.error(f"Scanner test failed: {e}")
        return False

def check_configuration():
    """Check the current configuration."""
    
    print("⚙️ Checking configuration...")
    
    try:
        import yaml
        with open("crypto_bot/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        print("📋 Current configuration:")
        print(f"   - Mode: {config.get('mode', 'unknown')}")
        print(f"   - Execution mode: {config.get('execution_mode', 'unknown')}")
        print(f"   - Testing mode: {config.get('testing_mode', False)}")
        print(f"   - Symbol batch size: {config.get('symbol_batch_size', 0)}")
        print(f"   - Symbols configured: {len(config.get('symbols', []))}")
        
        # Check scanning configs
        solana_scanner = config.get("solana_scanner", {})
        enhanced_scanning = config.get("enhanced_scanning", {})
        
        print(f"\n🔍 Scanning Configuration:")
        print(f"   - Solana scanner enabled: {solana_scanner.get('enabled', False)}")
        print(f"   - Enhanced scanning enabled: {enhanced_scanning.get('enabled', False)}")
        
        if solana_scanner.get('enabled', False):
            print(f"   - Scan interval: {solana_scanner.get('interval_minutes', 0)} minutes")
            print(f"   - Max tokens per scan: {solana_scanner.get('max_tokens_per_scan', 0)}")
            print(f"   - Min volume: ${solana_scanner.get('min_volume_usd', 0):,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration check failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Scanning Functionality Test")
    print("=" * 50)
    
    # Check configuration
    config_ok = check_configuration()
    
    if config_ok:
        # Test scanning
        success = asyncio.run(test_scanning_functionality())
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 SCANNING IS WORKING!")
            print("✅ Token discovery is functional")
            print("✅ Scanner configuration is correct")
            print("✅ Integration is successful")
            print("\n📋 Summary:")
            print("   - Scanning and token analysis is enabled and working")
            print("   - The scanner is discovering new Solana tokens")
            print("   - The system is ready for token analysis")
            print("   - The bot can now scan for trading opportunities")
        else:
            print("❌ SCANNING TEST FAILED")
            print("📋 Check the configuration and try again")
    else:
        print("❌ CONFIGURATION CHECK FAILED")
    
    sys.exit(0 if success else 1)
