# Performance Optimization Guide

This project runs multiple services simultaneously, so improving runtime efficiency keeps the trading bot responsive. Use the checklist below when profiling or shipping changes that may affect latency.

## Quick Wins
- **Disable debug tooling** in production (`LOG_LEVEL=INFO`) to avoid excessive logging overhead.
- **Reuse exchange clients** instead of recreating HTTP/WebSocket sessions on every request.
- **Batch price lookups** by symbol to reduce round-trip calls to external APIs.

## Monitoring Signals
- Track CPU load and memory consumption with `system_status.sh` or the monitoring dashboards.
- Watch the order submission queue in `production_monitor.py`; a growing queue signals latency issues.
- Record before/after metrics (latency, fills per minute) when applying optimizations.

## Deep Dives
- Profile Python hotspots with `cProfile` on `start_bot_debug.py`.
- Audit indicator calculations under `ta/` for redundant pandas operations.
- Verify network timeouts and retries in `config/` to avoid cascading delays.
