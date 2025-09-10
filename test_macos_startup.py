#!/usr/bin/env python3
"""
Test script to verify macOS startup process without GPU dependencies.
"""

import platform
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_platform_detection():
    """Test platform detection."""
    logger.info(f"Platform: {platform.system()}")
    logger.info(f"Platform version: {platform.release()}")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Processor: {platform.processor()}")
    
    if platform.system() == "Darwin":
        logger.info("✅ macOS detected")
        return True
    else:
        logger.info(f"⚠️  Non-macOS platform: {platform.system()}")
        return False

def test_gpu_imports():
    """Test GPU library imports."""
    logger.info("Testing GPU library imports...")
    
    # Test CuPy import
    try:
        import cupy as cp
        logger.info("✅ CuPy imported successfully")
        return True
    except ImportError as e:
        logger.info(f"⚠️  CuPy not available: {e}")
        return False

def test_enhanced_backtester():
    """Test enhanced backtester import."""
    logger.info("Testing enhanced backtester import...")
    
    try:
        from crypto_bot.backtest.enhanced_backtester import GPU_AVAILABLE, EnhancedBacktestConfig
        logger.info(f"✅ Enhanced backtester imported successfully")
        logger.info(f"GPU_AVAILABLE: {GPU_AVAILABLE}")
        
        # Test config creation
        config = EnhancedBacktestConfig(use_gpu=False)
        logger.info(f"✅ Config created: use_gpu={config.use_gpu}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Enhanced backtester import failed: {e}")
        return False

def test_gpu_accelerator():
    """Test GPU accelerator import."""
    logger.info("Testing GPU accelerator import...")
    
    try:
        from crypto_bot.backtest.gpu_accelerator import GPUAccelerator
        logger.info("✅ GPU accelerator imported successfully")
        
        # Test accelerator creation
        config = {'use_gpu': False}
        accelerator = GPUAccelerator(config)
        logger.info(f"✅ GPU accelerator created: gpu_available={accelerator.gpu_available}")
        
        return True
    except Exception as e:
        logger.error(f"❌ GPU accelerator import failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting macOS compatibility tests...")
    
    tests = [
        ("Platform Detection", test_platform_detection),
        ("GPU Imports", test_gpu_imports),
        ("Enhanced Backtester", test_enhanced_backtester),
        ("GPU Accelerator", test_gpu_accelerator),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("="*50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 All tests passed! macOS startup should work correctly.")
    else:
        logger.info("\n⚠️  Some tests failed. Check the logs above for details.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
