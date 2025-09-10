# Symbol Processing Modes

## Overview

The LegacyCoinTrader **now defaults to comprehensive mode** for maximum market coverage and trading opportunities.

## 1. Comprehensive Mode (NEW DEFAULT)
- **Configuration**: `process_all_symbols: true` (now default)
- **Behavior**: Processes ALL available USD symbols on the exchange
- **Current Performance**: Up to 100 symbols per cycle (configurable batch_size)
- **Pros**:
  - ✅ Complete market coverage (497+ USD symbols on Kraken)
  - ✅ Catches all potential opportunities
  - ✅ No dependency on scanner quality
  - ✅ True comprehensive analysis
  - ✅ Optimized with chunked processing and memory management
- **Cons**:
  - ⚠️ Higher resource usage (optimized with chunking)
  - ⚠️ Longer processing cycles (~60-120s vs ~30s)
  - ⚠️ Higher API rate limit usage (optimized with concurrency controls)

## 2. Filtered Mode (Legacy)
- **Configuration**: `process_all_symbols: false`
- **Behavior**: Uses intelligent evaluation pipeline to select high-quality opportunities
- **Current Performance**: ~12 symbols per cycle
- **Pros**:
  - Faster processing
  - Lower API usage
  - Higher quality signals (pre-filtered)
  - Better performance on limited hardware
- **Cons**:
  - Limited market coverage (2.4% of available symbols)
  - May miss emerging opportunities
  - Depends on scanner accuracy

## Current Configuration (Optimized for Comprehensive Mode)

```yaml
# Default configuration for comprehensive symbol processing
process_all_symbols: true  # Now the default

# Optimized performance settings
symbol_batch_size: 100     # Process 100 symbols per cycle
cycle_delay_seconds: 120   # 2 minutes between cycles for processing time
max_concurrent_ohlcv: 10   # Increased concurrency for faster fetching
max_concurrent_requests: 10 # Increased concurrent API requests

# Memory and performance optimizations
comprehensive_mode_optimization:
  enable_memory_optimization: true
  batch_chunk_size: 25      # Process in chunks of 25 for memory management
  enable_progress_tracking: true
  adaptive_rate_limiting: true
  memory_cleanup_interval: 50 # Clean memory every 50 symbols
```

## Configuration

Add to your `crypto_bot/config.yaml`:

```yaml
# For filtered mode (default)
process_all_symbols: false

# For comprehensive mode
process_all_symbols: true
```

## Performance Comparison

| Metric | Filtered Mode | Comprehensive Mode |
|--------|---------------|-------------------|
| Symbols/Cycle | ~12 | Up to batch_size (default 50) |
| API Calls | Low | High |
| Memory Usage | Low | High |
| Processing Time | Fast (~30s) | Slower (~60-120s+) |
| Market Coverage | 2.4% | 100% |
| Signal Quality | High (pre-filtered) | Variable (all symbols) |

## Recommendations

### Use Filtered Mode If:
- You have limited hardware resources
- You want faster processing cycles
- You trust the scanner's symbol selection
- You're focused on quality over quantity

### Use Comprehensive Mode If:
- You have ample hardware resources
- You want maximum market coverage
- You want to catch all potential opportunities
- You're willing to accept slower processing for completeness

## Hybrid Approach

You can also combine both approaches by:
1. Using comprehensive mode to discover new opportunities
2. Switching back to filtered mode for regular operation
3. Manually maintaining a high-quality symbol list in config

## Current Status

The bot currently uses **Filtered Mode** by default, processing only the top opportunities identified by the evaluation pipeline. This provides a good balance of performance and quality but limits market coverage significantly.
