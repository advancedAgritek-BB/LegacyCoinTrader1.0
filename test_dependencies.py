"""
Test for dependency imports.
"""

import pytest
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

    # Test critical packages - these must be available
    for pkg in critical_packages:
        try:
            importlib.import_module(pkg)
        except ImportError as e:
            pytest.fail(f"Critical package '{pkg}' could not be imported: {e}")

    # Test numpy and scipy functionality
    try:
        import numpy as np
        import scipy as sp

        # Test basic operations
        a = np.random.rand(100, 100)
        b = np.random.rand(100, 100)
        c = np.dot(a, b)

        assert c.shape == (100, 100), "Matrix multiplication should produce correct shape"
        assert np.all(np.isfinite(c)), "Matrix multiplication should produce finite values"

    except Exception as e:
        pytest.fail(f"Numerical library test failed: {e}")


