# 🍎 macOS Compatibility Guide

## Overview

LegacyCoinTrader has been updated to provide full compatibility with macOS systems. Since CUDA (NVIDIA's GPU computing platform) doesn't run on macOS, the system automatically detects the platform and provides appropriate fallbacks.

## 🔧 What Changed

### 1. Platform Detection
- The startup script now detects macOS and skips GPU dependency installation
- GPU libraries (CuPy, CUDA) are only imported on Windows systems
- Enhanced backtester gracefully falls back to CPU-only mode on macOS

### 2. Dependency Management
- **Windows**: Installs `requirements_gpu.txt` for full GPU acceleration
- **macOS**: Installs `requirements_macos.txt` for CPU-optimized performance
- **Linux**: Installs `requirements_gpu.txt` (GPU support varies by distribution)

### 3. Graceful Fallbacks
- GPU-accelerated backtesting automatically falls back to CPU
- All functionality remains available, just without GPU acceleration
- Performance is still optimized using macOS-specific libraries

## 🚀 Getting Started on macOS

### 1. Run the Startup Script
```bash
./startup.sh
```

The script will:
- Detect macOS automatically
- Skip GPU dependencies installation
- Install macOS-optimized performance libraries
- Set up the environment for CPU-based trading

### 2. Verify Installation
Run the compatibility test:
```bash
python3 test_macos_startup.py
```

This will verify that:
- Platform detection works correctly
- GPU imports are handled gracefully
- Enhanced backtester works in CPU mode
- GPU accelerator provides proper fallbacks

## 📊 Performance Expectations

### With GPU (Windows)
- Parameter optimization: ~10-50x faster
- Large backtesting runs: ~5-20x faster
- Memory usage: Higher (GPU memory)

### Without GPU (macOS)
- Parameter optimization: Standard speed
- Large backtesting runs: Standard speed
- Memory usage: Lower (CPU memory only)
- **Still fully functional for all trading operations**

## 🛠️ macOS-Specific Optimizations

The system automatically installs these macOS-optimized libraries:

- **Intel Math Kernel Library (MKL)**: Optimized linear algebra for Intel Macs
- **OpenBLAS**: High-performance BLAS implementation
- **Accelerate Framework**: Apple's performance optimization framework
- **Metal Framework**: Apple's compute framework (if available)

## 🔍 Troubleshooting

### Issue: "CuPy not available" warnings
**Solution**: This is expected and normal on macOS. The system automatically falls back to CPU mode.

### Issue: Performance seems slow
**Solution**: 
1. Ensure you're using the latest macOS version
2. Check that MKL and OpenBLAS are installed
3. Consider reducing backtesting complexity for large datasets

### Issue: Import errors
**Solution**: Run the compatibility test to identify specific issues:
```bash
python3 test_macos_startup.py
```

## 📈 Trading Performance

**Important**: GPU acceleration only affects backtesting and parameter optimization speed. It does NOT affect:

- Live trading performance
- Signal generation accuracy
- Risk management
- Portfolio management
- Real-time market data processing

Your trading bot will perform identically on macOS vs Windows - only the backtesting speed differs.

## 🔄 Future Updates

- The system will automatically detect when GPU acceleration becomes available
- Future versions may include Metal-based acceleration for Apple Silicon Macs
- All updates maintain backward compatibility

## 📞 Support

If you encounter issues:
1. Run the compatibility test first
2. Check the logs for specific error messages
3. Ensure you're using the latest version
4. GPU acceleration is only available on Windows - this is expected behavior

---

**Note**: This system is designed to work optimally on all platforms. GPU acceleration is a performance enhancement, not a requirement for functionality.
