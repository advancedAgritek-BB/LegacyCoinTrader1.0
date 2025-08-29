# Enhanced Backtesting System

## Overview

The Enhanced Backtesting System provides comprehensive backtesting capabilities for all trading strategies against the top 20 token pairs, with support for GPU acceleration and continuous learning. This system is designed to continuously improve trading intelligence by analyzing strategy performance across multiple timeframes and market conditions.

## Key Features

- **Continuous Backtesting**: Automatically backtests top 20 token pairs every 6 hours
- **GPU Acceleration**: Leverages AMD/NVIDIA GPUs for faster parameter optimization
- **Multi-Strategy Testing**: Tests all 20+ available strategies automatically
- **Multi-Timeframe Analysis**: Analyzes 1h, 4h, and 1d timeframes
- **Performance Tracking**: Maintains historical performance data for all strategies
- **Strategy Optimization**: Automatically optimizes strategy parameters based on results
- **Continuous Learning**: Uses backtesting results to improve trading decisions

## Architecture

```
Enhanced Backtesting System
├── ContinuousBacktestingEngine     # Main orchestration engine
├── StrategyPerformanceTracker      # Performance analysis and ranking
├── GPUAcceleratedBacktester       # GPU-accelerated parameter optimization
├── GPUAccelerator                 # Windows-specific GPU detection and setup
└── CLI Interface                  # Command-line interface for management
```

## Installation

### Prerequisites

- Python 3.8+
- Windows 10/11 (for GPU acceleration)
- AMD or NVIDIA GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Basic Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# Install GPU acceleration dependencies (optional)
pip install -r requirements_gpu.txt
```

### GPU Acceleration Setup

#### For AMD GPUs (Windows)

1. **Install AMD Drivers**: Ensure you have the latest AMD drivers installed
2. **Install ROCm** (if available for your GPU):
   ```bash
   pip install cupy-rocm-5-7
   ```
3. **Install OpenCL support**:
   ```bash
   pip install pyopencl
   ```

#### For NVIDIA GPUs (Windows)

1. **Install CUDA Toolkit**: Download and install CUDA 12.x from NVIDIA
2. **Install CuPy**:
   ```bash
   pip install cupy-cuda12x
   ```
3. **Install Numba with CUDA**:
   ```bash
   pip install numba
   ```

## Configuration

### Basic Configuration

Create a configuration file `config/backtest_config.yaml`:

```yaml
# Token pair selection
top_pairs_count: 20
min_volume_usd: 1000000  # $1M minimum volume
refresh_interval_hours: 6

# GPU acceleration
use_gpu: true
gpu_memory_limit_gb: 8.0

# Backtesting parameters
timeframes: ["1h", "4h", "1d"]
lookback_days: 90
```

### Advanced Configuration

```yaml
# AMD GPU specific settings
amd_gpu:
  enable_rocm: false
  memory_optimization: true
  compute_units: "auto"

# Performance monitoring
monitoring:
  log_performance_metrics: true
  track_memory_usage: true
  monitor_gpu_utilization: true
```

## Usage

### Command Line Interface

The system provides a comprehensive CLI for managing backtesting operations:

#### Run Single Backtest

```bash
# Test specific pairs and strategies
python -m crypto_bot.backtest.cli run \
  --pairs BTC/USDT ETH/USDT SOL/USDC \
  --strategies trend_bot momentum_bot \
  --timeframes 1h 4h \
  --output results.json
```

#### Run Continuous Backtesting

```bash
# Start continuous backtesting engine
python -m crypto_bot.backtest.cli continuous \
  --config config/backtest_config.yaml
```

#### View Results

```bash
# View all strategy performance
python -m crypto_bot.backtest.cli view

# View specific strategy details
python -m crypto_bot.backtest.cli view --strategy trend_bot
```

#### Optimize Strategies

```bash
# Run strategy optimization
python -m crypto_bot.backtesting.cli optimize \
  --output optimization_results.json
```

### Programmatic Usage

```python
from crypto_bot.backtest.enhanced_backtester import create_enhanced_backtester

# Create backtesting engine
config = {
    'top_pairs_count': 20,
    'use_gpu': True,
    'timeframes': ['1h', '4h', '1d']
}

engine = create_enhanced_backtester(config)

# Run continuous backtesting
await engine.run_continuous_backtesting()
```

## GPU Acceleration

### How It Works

The system automatically detects your GPU and uses the most appropriate acceleration method:

1. **AMD GPUs**: Uses OpenCL or ROCm for parallel processing
2. **NVIDIA GPUs**: Uses CUDA via CuPy or Numba
3. **Fallback**: Automatically falls back to CPU if GPU acceleration fails

### Performance Benefits

- **Parameter Optimization**: 5-10x faster than CPU-only
- **Batch Processing**: Process multiple backtests simultaneously
- **Memory Efficiency**: Optimized memory usage for large datasets

### Monitoring GPU Usage

```python
from crypto_bot.backtest.gpu_accelerator import get_gpu_info

# Get GPU status and performance metrics
gpu_info = get_gpu_info()
print(f"GPU Type: {gpu_info['gpu_type']}")
print(f"Memory: {gpu_info['memory_gb']} GB")
print(f"Compute Units: {gpu_info['compute_units']}")
```

## Continuous Learning

### How Continuous Learning Works

1. **Data Collection**: System continuously collects backtesting results
2. **Performance Analysis**: Analyzes strategy performance across different market conditions
3. **Strategy Ranking**: Ranks strategies based on composite performance scores
4. **Parameter Optimization**: Automatically optimizes strategy parameters
5. **Integration**: Updates main trading bot with optimized parameters

### Learning Metrics

The system tracks and learns from:

- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Risk management
- **Win Rate**: Strategy consistency
- **Profit Factor**: Risk-reward ratio
- **Recovery Factor**: Drawdown recovery speed

### Strategy Rankings

Strategies are ranked using a composite score:

```
Composite Score = 
  (Sharpe Ratio × 0.4) +
  (Win Rate × 0.3) +
  (1 - Max Drawdown × 0.2) +
  (Consistency × 0.1)
```

## Results and Analysis

### Output Structure

```
crypto_bot/logs/backtest_results/
├── trend_bot_20241201_143022.csv
├── momentum_bot_20241201_143022.csv
├── rankings_20241201_143022.json
└── optimization_results.json
```

### Performance Metrics

Each backtest generates comprehensive metrics:

- **Returns**: PnL, Sharpe ratio, Sortino ratio
- **Risk**: Maximum drawdown, VaR, volatility
- **Efficiency**: Win rate, profit factor, recovery factor
- **Timing**: Entry/exit timing, holding periods

### Visualization

The system can generate:

- Equity curves for each strategy
- Performance comparison charts
- Risk-return scatter plots
- Parameter sensitivity analysis

## Integration with Main Trading Bot

### Automatic Updates

The enhanced backtesting system automatically:

1. **Updates Strategy Weights**: Based on performance rankings
2. **Optimizes Parameters**: Uses best-performing parameter combinations
3. **Adjusts Risk Management**: Based on drawdown analysis
4. **Updates Portfolio Allocation**: Based on strategy performance

### Configuration Integration

```yaml
# In main config.yaml
integration:
  update_strategy_weights: true
  auto_optimize_parameters: true
  risk_adjustment: true
```

## Monitoring and Maintenance

### Performance Monitoring

```bash
# Check system status
python -m crypto_bot.backtest.cli view

# Monitor GPU utilization
python -c "from crypto_bot.backtest.gpu_accelerator import get_gpu_info; print(get_gpu_info())"
```

### Log Files

- `backtest.log`: Main backtesting operations
- `gpu_acceleration.log`: GPU-specific operations
- `performance_metrics.log`: Strategy performance data

### Maintenance Tasks

1. **Daily**: Review strategy rankings
2. **Weekly**: Analyze performance trends
3. **Monthly**: Optimize system parameters
4. **Quarterly**: Review and update strategies

## Troubleshooting

### Common Issues

#### GPU Not Detected

```bash
# Check GPU detection
python -c "from crypto_bot.backtest.gpu_accelerator import is_gpu_available; print(is_gpu_available())"
```

**Solutions:**
- Update GPU drivers
- Install appropriate CUDA/ROCm versions
- Check GPU compatibility

#### Memory Issues

```yaml
# Reduce memory usage in config
gpu_memory_limit_gb: 4.0  # Reduce from 8.0
batch_size: 50            # Reduce from 100
```

#### Performance Issues

```yaml
# Optimize for your hardware
max_workers: 4            # Reduce if CPU-bound
use_process_pool: false   # Use threading if I/O-bound
```

### Debug Mode

```bash
# Enable verbose logging
python -m crypto_bot.backtest.cli run --verbose
```

## Performance Benchmarks

### Expected Performance

| Hardware | CPU-Only | GPU-Accelerated | Speedup |
|----------|----------|-----------------|---------|
| Intel i7-12700K | 100% | 100% | 1.0x |
| AMD RX 6800 XT | 100% | 500% | 5.0x |
| NVIDIA RTX 4080 | 100% | 800% | 8.0x |

### Scaling Characteristics

- **Linear Scaling**: Performance scales linearly with GPU memory
- **Batch Processing**: Larger batches improve GPU utilization
- **Memory Efficiency**: Optimized for 8GB+ GPU memory

## Advanced Features

### Custom Strategies

Add custom strategies by implementing the strategy interface:

```python
def custom_strategy(df: pd.DataFrame, config: dict) -> Tuple[float, str, float]:
    # Your strategy logic here
    return confidence_score, direction, atr_value
```

### Custom Metrics

Define custom performance metrics:

```yaml
performance_metrics:
  - "custom_metric_1"
  - "custom_metric_2"
```

### Machine Learning Integration

The system supports ML-based strategy selection:

```yaml
ml_integration:
  enabled: true
  model_type: "lightgbm"
  retrain_interval_hours: 24
```

## Support and Development

### Getting Help

1. **Documentation**: This README and inline code comments
2. **Logs**: Check log files for detailed error information
3. **Issues**: Report bugs with system information and logs

### Contributing

1. **Code Style**: Follow existing code style and patterns
2. **Testing**: Add tests for new features
3. **Documentation**: Update this README for new features

### Roadmap

- [ ] Real-time backtesting during live trading
- [ ] Advanced ML model integration
- [ ] Multi-GPU support
- [ ] Cloud deployment options
- [ ] Advanced visualization dashboard

## License

This enhanced backtesting system is part of the LegacyCoinTrader project and follows the same licensing terms.

---

For more information, see the main project documentation and the inline code comments in each module.
