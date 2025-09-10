#!/bin/bash

# LegacyCoinTrader macOS Installation Script
# Handles dependency issues and provides fallbacks

set -e

echo "🚀 Installing LegacyCoinTrader on macOS..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "❌ Please activate a virtual environment first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    exit 1
fi

echo "✅ Virtual environment detected: $VIRTUAL_ENV"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install core requirements first (without problematic packages)
echo "📦 Installing core dependencies..."
pip install -r requirements.txt

# Install macOS-specific optimizations (with error handling)
echo "🍎 Installing macOS-specific optimizations..."
if pip install -r requirements_macos.txt; then
    echo "✅ macOS optimizations installed successfully"
else
    echo "⚠️  Some macOS optimizations failed - continuing with core functionality"
fi

# Handle optional packages that might fail
echo "🔧 Installing optional packages..."

# Try to install ccxtpro (optional - requires license)
if pip install ccxtpro 2>/dev/null; then
    echo "✅ ccxtpro installed successfully (license required for full functionality)"
else
    echo "⚠️  ccxtpro not installed (requires paid license - using ccxt fallback)"
fi

# Try to install pyobjc packages via conda if available
if command -v conda &> /dev/null; then
    echo "🐍 Conda detected - installing Apple framework bindings..."
    conda install -c conda-forge pyobjc-framework-accelerate pyobjc-framework-metal -y || echo "⚠️  Apple framework bindings not available"
else
    echo "ℹ️  Conda not found - Apple framework bindings skipped (optional)"
fi

# Install OpenMP for LightGBM if not present
echo "🔧 Installing OpenMP for LightGBM..."
if command -v brew &> /dev/null; then
    echo "🍺 Installing libomp via Homebrew..."
    brew install libomp || echo "⚠️  libomp installation failed - LightGBM may not work optimally"
else
    echo "⚠️  Homebrew not found - LightGBM may not work optimally without libomp"
fi

# Ensure compatible NumPy version (don't upgrade beyond what numba supports)
echo "🔧 Ensuring compatible NumPy version..."
pip install "numpy<2.3,>=1.24" --force-reinstall

# Verify critical packages
echo "🔍 Verifying installation..."
python -c "
import sys
packages = ['pandas', 'numpy', 'sklearn', 'lightgbm', 'ccxt', 'solana']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError as e:
        missing.append(pkg)
        print(f'❌ {pkg}: {e}')
if missing:
    print(f'\\n❌ Missing critical packages: {missing}')
    sys.exit(1)
else:
    print('\\n🎉 All critical packages installed successfully!')
"

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Configure your API keys in config/config.yaml"
echo "   2. Run: python crypto_bot/main.py"
echo ""
echo "📚 For more information, see README.md"
