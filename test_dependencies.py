#!/usr/bin/env python3
"""
Test script to verify macOS dependency installation
"""

import sys
import importlib

def test_dependencies():
    """Test that all critical dependencies can be imported"""
    
    critical_packages = [
        'pandas',
        'numpy', 
        'sklearn',  # scikit-learn is imported as sklearn
        'lightgbm',
        'ccxt',
        'solana',
        'joblib',
        'scipy'
    ]
    
    optional_packages = [
        'memory_profiler',
        'psutil',
        'multiprocessing_logging'
    ]
    
    print("🔍 Testing critical dependencies...")
    failed_critical = []
    
    for pkg in critical_packages:
        try:
            importlib.import_module(pkg)
            print(f"✅ {pkg}")
        except ImportError as e:
            print(f"❌ {pkg}: {e}")
            failed_critical.append(pkg)
    
    print("\n🔍 Testing optional dependencies...")
    failed_optional = []
    
    for pkg in optional_packages:
        try:
            importlib.import_module(pkg)
            print(f"✅ {pkg}")
        except ImportError as e:
            print(f"⚠️  {pkg}: {e}")
            failed_optional.append(pkg)
    
    # Test problematic packages that should NOT be available
    problematic_packages = ['mkl', 'ccxtpro', 'openblas']
    print("\n🔍 Testing problematic packages (should fail gracefully)...")
    
    for pkg in problematic_packages:
        try:
            importlib.import_module(pkg)
            print(f"⚠️  {pkg}: Available (may cause issues)")
        except ImportError:
            print(f"✅ {pkg}: Not available (expected)")
    
    # Test numpy and scipy BLAS optimization
    print("\n🔍 Testing numerical library optimization...")
    try:
        import numpy as np
        import scipy as sp
        print(f"✅ numpy version: {np.__version__}")
        print(f"✅ scipy version: {sp.__version__}")
        
        # Test basic operations
        a = np.random.rand(100, 100)
        b = np.random.rand(100, 100)
        c = np.dot(a, b)
        print("✅ Basic matrix operations working")
        
    except Exception as e:
        print(f"❌ Numerical library test failed: {e}")
        failed_critical.append('numerical_optimization')
    
    print("\n📊 Summary:")
    if failed_critical:
        print(f"❌ Critical packages missing: {failed_critical}")
        return False
    else:
        print("✅ All critical packages available")
    
    if failed_optional:
        print(f"⚠️  Optional packages missing: {failed_optional}")
    else:
        print("✅ All optional packages available")
    
    return True

if __name__ == "__main__":
    success = test_dependencies()
    sys.exit(0 if success else 1)
