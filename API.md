# API Documentation

This document provides comprehensive API documentation for all the performance enhancement classes and utilities implemented in the trading bot.

## Table of Contents

1. [MemoryManager](#memorymanager)
2. [AdaptiveRateLimiter](#adaptiveratelimiter)
3. [AdaptiveCacheManager](#adaptivecachemanager)
4. [PerformanceMonitor](#performancemonitor)
5. [DatabaseManager](#databasemanager)
6. [Configuration](#configuration)

## MemoryManager

The `MemoryManager` class provides dynamic memory management for the trading bot.

### Class Definition

```python
class MemoryManager:
    def __init__(self, memory_threshold: float = 0.8)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_threshold` | `float` | `0.8` | Percentage of total RAM to trigger optimization (0.8 = 80%) |

### Methods

#### `check_memory_pressure() -> bool`

Check if system memory usage exceeds the configured threshold.

**Returns:**
- `bool`: `True` if memory pressure is detected, `False` otherwise

**Example:**
```python
memory_manager = MemoryManager(memory_threshold=0.8)
if memory_manager.check_memory_pressure():
    print("Memory pressure detected!")
```

#### `optimize_cache_sizes(df_cache: dict, regime_cache: dict) -> None`

Dynamically reduce cache sizes when memory pressure is detected.

**Parameters:**
- `df_cache` (`dict`): OHLCV data cache to optimize
- `regime_cache` (`dict`): Regime analysis cache to optimize

**Example:**
```python
memory_manager.optimize_cache_sizes(df_cache, regime_cache)
```

#### `force_garbage_collection() -> None`

Force garbage collection to free memory.

**Example:**
```python
memory_manager.force_garbage_collection()
```

#### `get_memory_stats() -> dict`

Get current memory usage statistics.

**Returns:**
- `dict`: Dictionary containing memory statistics:
  - `total_mb`: Total memory in MB
  - `available_mb`: Available memory in MB
  - `used_mb`: Used memory in MB
  - `percent`: Memory usage percentage
  - `optimization_count`: Number of optimizations performed

**Example:**
```python
stats = memory_manager.get_memory_stats()
print(f"Memory usage: {stats['percent']}%")
```

## AdaptiveRateLimiter

The `AdaptiveRateLimiter` class provides intelligent rate limiting for API calls.

### Class Definition

```python
class AdaptiveRateLimiter:
    def __init__(
        self,
        max_requests_per_minute: int = 10,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        error_backoff_multiplier: float = 2.0,
        success_recovery_factor: float = 0.8,
        window_size: int = 1000
    )
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_requests_per_minute` | `int` | `10` | Maximum requests allowed per minute |
| `base_delay` | `float` | `1.0` | Base delay in seconds between requests |
| `max_delay` | `float` | `60.0` | Maximum delay in seconds |
| `error_backoff_multiplier` | `float` | `2.0` | Multiplier for delay on errors |
| `success_recovery_factor` | `float` | `0.8` | Factor to reduce delay on success |
| `window_size` | `int` | `1000` | Size of the request history window |

### Methods

#### `wait_if_needed() -> None`

Wait if necessary to respect rate limits and apply adaptive delays.

**Example:**
```python
rate_limiter = AdaptiveRateLimiter()
await rate_limiter.wait_if_needed()
```

#### `record_error() -> None`

Record an error and increase the backoff delay.

**Example:**
```python
try:
    # Make API call
    response = await api_call()
except Exception as e:
    rate_limiter.record_error()
```

#### `record_success() -> None`

Record a successful request and gradually reduce the delay.

**Example:**
```python
response = await api_call()
rate_limiter.record_success()
```

#### `get_current_delay() -> float`

Get the current delay value.

**Returns:**
- `float`: Current delay in seconds

**Example:**
```python
current_delay = rate_limiter.get_current_delay()
print(f"Current delay: {current_delay}s")
```

#### `get_stats() -> dict`

Get current statistics.

**Returns:**
- `dict`: Dictionary containing rate limiter statistics:
  - `total_requests`: Total number of requests
  - `total_errors`: Total number of errors
  - `error_rate`: Error rate as a percentage
  - `consecutive_errors`: Number of consecutive errors
  - `current_delay`: Current delay in seconds
  - `requests_last_minute`: Requests in the last minute
  - `max_requests_per_minute`: Maximum requests per minute

**Example:**
```python
stats = rate_limiter.get_stats()
print(f"Error rate: {stats['error_rate']}%")
```

## AdaptiveCacheManager

The `AdaptiveCacheManager` class provides intelligent cache management with adaptive sizing.

### Class Definition

```python
class AdaptiveCacheManager:
    def __init__(
        self,
        initial_size: int = 1000,
        max_size: int = 10000,
        min_size: int = 100,
        hit_rate_window: int = 100,
        eviction_policy: str = "lru",
        enable_compression: bool = False
    )
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_size` | `int` | `1000` | Initial cache size for each cache type |
| `max_size` | `int` | `10000` | Maximum cache size allowed |
| `min_size` | `int` | `100` | Minimum cache size allowed |
| `hit_rate_window` | `int` | `100` | Number of accesses to track for hit rate calculation |
| `eviction_policy` | `str` | `"lru"` | Cache eviction policy ('lru', 'lfu', 'adaptive') |
| `enable_compression` | `bool` | `False` | Whether to enable data compression |

### Methods

#### `get_cache_size(cache_type: str) -> int`

Get adaptive cache size based on hit rates.

**Parameters:**
- `cache_type` (`str`): Type of cache (e.g., 'ohlcv', 'orderbook', 'regime')

**Returns:**
- `int`: Recommended cache size

**Example:**
```python
cache_manager = AdaptiveCacheManager()
size = cache_manager.get_cache_size("ohlcv")
print(f"Recommended cache size: {size}")
```

#### `get(cache_type: str, key: str) -> Optional[Any]`

Retrieve data from cache with hit tracking.

**Parameters:**
- `cache_type` (`str`): Type of cache
- `key` (`str`): Cache key

**Returns:**
- `Optional[Any]`: Cached data or `None` if not found

**Example:**
```python
data = cache_manager.get("ohlcv", "BTC/USD")
if data is not None:
    print("Cache hit!")
```

#### `set(cache_type: str, key: str, data: Any, ttl: Optional[int] = None) -> None`

Store data in cache with adaptive sizing.

**Parameters:**
- `cache_type` (`str`): Type of cache
- `key` (`str`): Cache key
- `data` (`Any`): Data to cache
- `ttl` (`Optional[int]`): Time to live in seconds (optional)

**Example:**
```python
cache_manager.set("ohlcv", "BTC/USD", ohlcv_data, ttl=3600)
```

#### `invalidate(cache_type: str, key: str) -> bool`

Remove a specific entry from cache.

**Parameters:**
- `cache_type` (`str`): Type of cache
- `key` (`str`): Cache key

**Returns:**
- `bool`: `True` if entry was found and removed

**Example:**
```python
if cache_manager.invalidate("ohlcv", "BTC/USD"):
    print("Entry removed from cache")
```

#### `clear(cache_type: Optional[str] = None) -> None`

Clear cache entries.

**Parameters:**
- `cache_type` (`Optional[str]`): Specific cache type to clear, or `None` for all

**Example:**
```python
# Clear specific cache
cache_manager.clear("ohlcv")

# Clear all caches
cache_manager.clear()
```

#### `get_stats(cache_type: Optional[str] = None) -> dict`

Get cache statistics.

**Parameters:**
- `cache_type` (`Optional[str]`): Specific cache type, or `None` for all

**Returns:**
- `dict`: Dictionary containing cache statistics

**Example:**
```python
# Get specific cache stats
stats = cache_manager.get_stats("ohlcv")
print(f"Hit rate: {stats['hit_rate']}%")

# Get aggregate stats
all_stats = cache_manager.get_stats()
print(f"Total entries: {all_stats['total_entries']}")
```

## PerformanceMonitor

The `PerformanceMonitor` class provides comprehensive performance monitoring and alerting.

### Class Definition

```python
class PerformanceMonitor:
    def __init__(
        self,
        memory_threshold: float = 85.0,
        error_rate_threshold: float = 10.0,
        response_time_threshold: float = 5.0,
        cache_hit_rate_threshold: float = 50.0,
        metrics_window_size: int = 1000,
        alert_enabled: bool = True
    )
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_threshold` | `float` | `85.0` | Memory usage threshold for alerts (%) |
| `error_rate_threshold` | `float` | `10.0` | Error rate threshold for alerts (%) |
| `response_time_threshold` | `float` | `5.0` | Response time threshold for alerts (seconds) |
| `cache_hit_rate_threshold` | `float` | `50.0` | Cache hit rate threshold for alerts (%) |
| `metrics_window_size` | `int` | `1000` | Size of rolling metrics window |
| `alert_enabled` | `bool` | `True` | Whether to enable performance alerts |

### Methods

#### `record_metric(metric_type: str, value: float, **kwargs) -> None`

Record a performance metric.

**Parameters:**
- `metric_type` (`str`): Type of metric ('memory', 'response_time', 'cache_hit', 'error', 'throughput')
- `value` (`float`): Metric value
- `**kwargs`: Additional context (e.g., cache_type, api_endpoint)

**Example:**
```python
monitor = PerformanceMonitor()

# Record memory usage
monitor.record_metric("memory", 75.5)

# Record API response time
monitor.record_metric("response_time", 2.5)

# Record cache hit rate
monitor.record_metric("cache_hit", 85.0, cache_type="ohlcv")

# Record error rate
monitor.record_metric("error", 5.0)
```

#### `get_performance_report() -> dict`

Generate comprehensive performance summary.

**Returns:**
- `dict`: Dictionary containing performance metrics and analysis

**Example:**
```python
report = monitor.get_performance_report()
print(f"Overall health: {report['summary']['overall_health']}")
```

#### `get_alert_conditions() -> List[Dict[str, Any]]`

Identify performance issues that require attention.

**Returns:**
- `List[Dict[str, Any]]`: List of alert conditions

**Example:**
```python
alerts = monitor.get_alert_conditions()
for alert in alerts:
    print(f"Alert: {alert['message']}")
```

#### `export_metrics(format: str = "json", path: Optional[Union[str, Path]] = None) -> None`

Export metrics to file.

**Parameters:**
- `format` (`str`): Export format ('json', 'csv')
- `path` (`Optional[Union[str, Path]]`): Output file path

**Example:**
```python
# Export to JSON
monitor.export_metrics("json", "performance_report.json")

# Export to CSV
monitor.export_metrics("csv", "performance_report.csv")
```

## DatabaseManager

The `DatabaseManager` class provides optimized database operations with connection pooling.

### Class Definition

```python
class DatabaseManager:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "crypto_bot",
        user: str = "postgres",
        password: str = "",
        min_size: int = 5,
        max_size: int = 20,
        command_timeout: int = 30,
        ssl_mode: str = "prefer",
        application_name: str = "crypto_bot"
    )
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | `str` | `"localhost"` | Database host |
| `port` | `int` | `5432` | Database port |
| `database` | `str` | `"crypto_bot"` | Database name |
| `user` | `str` | `"postgres"` | Database user |
| `password` | `str` | `""` | Database password |
| `min_size` | `int` | `5` | Minimum connection pool size |
| `max_size` | `int` | `20` | Maximum connection pool size |
| `command_timeout` | `int` | `30` | Command timeout in seconds |
| `ssl_mode` | `str` | `"prefer"` | SSL mode for connections |
| `application_name` | `str` | `"crypto_bot"` | Application name for connection identification |

### Methods

#### `initialize() -> None`

Initialize the connection pool.

**Example:**
```python
db_manager = DatabaseManager()
await db_manager.initialize()
```

#### `close() -> None`

Close all database connections.

**Example:**
```python
await db_manager.close()
```

#### `get_connection()`

Get a database connection from the pool.

**Returns:**
- Async context manager yielding database connection

**Example:**
```python
async with db_manager.get_connection() as conn:
    result = await conn.fetch("SELECT * FROM performance_metrics")
```

#### `execute(query: str, *args, **kwargs) -> str`

Execute a query and return the result.

**Parameters:**
- `query` (`str`): SQL query to execute
- `*args`: Query parameters
- `**kwargs`: Additional connection options

**Returns:**
- `str`: Query result

**Example:**
```python
result = await db_manager.execute(
    "INSERT INTO performance_metrics (metric_type, metric_value) VALUES ($1, $2)",
    "memory", 75.5
)
```

#### `fetch(query: str, *args, **kwargs) -> List[asyncpg.Record]`

Fetch rows from a query.

**Parameters:**
- `query` (`str`): SQL query to execute
- `*args`: Query parameters
- `**kwargs`: Additional connection options

**Returns:**
- `List[asyncpg.Record]`: List of records

**Example:**
```python
records = await db_manager.fetch("SELECT * FROM performance_metrics WHERE metric_type = $1", "memory")
```

#### `fetchrow(query: str, *args, **kwargs) -> Optional[asyncpg.Record]`

Fetch a single row from a query.

**Parameters:**
- `query` (`str`): SQL query to execute
- `*args`: Query parameters
- `**kwargs`: Additional connection options

**Returns:**
- `Optional[asyncpg.Record]`: Single record or `None`

**Example:**
```python
record = await db_manager.fetchrow("SELECT * FROM performance_metrics WHERE id = $1", 1)
```

#### `execute_many(query: str, args_list: List[tuple]) -> None`

Execute a query with multiple parameter sets.

**Parameters:**
- `query` (`str`): SQL query to execute
- `args_list` (`List[tuple]`): List of parameter tuples

**Example:**
```python
args_list = [
    ("memory", 75.5),
    ("response_time", 2.5),
    ("error_rate", 5.0)
]
await db_manager.execute_many(
    "INSERT INTO performance_metrics (metric_type, metric_value) VALUES ($1, $2)",
    args_list
)
```

#### `prepare(query: str, name: Optional[str] = None) -> str`

Prepare a query for repeated execution.

**Parameters:**
- `query` (`str`): SQL query to prepare
- `name` (`Optional[str]`): Optional name for the prepared statement

**Returns:**
- `str`: Prepared statement name

**Example:**
```python
stmt_name = await db_manager.prepare(
    "INSERT INTO performance_metrics (metric_type, metric_value) VALUES ($1, $2)"
)
```

#### `execute_prepared(name: str, *args, **kwargs) -> str`

Execute a prepared statement.

**Parameters:**
- `name` (`str`): Prepared statement name
- `*args`: Query parameters
- `**kwargs`: Additional connection options

**Returns:**
- `str`: Query result

**Example:**
```python
await db_manager.execute_prepared(stmt_name, "memory", 75.5)
```

#### `check_health() -> bool`

Check database connection health.

**Returns:**
- `bool`: `True` if healthy, `False` otherwise

**Example:**
```python
is_healthy = await db_manager.check_health()
if not is_healthy:
    print("Database health check failed")
```

#### `get_stats() -> dict`

Get database statistics.

**Returns:**
- `dict`: Dictionary containing database statistics

**Example:**
```python
stats = db_manager.get_stats()
print(f"Total queries: {stats['total_queries']}")
print(f"Error rate: {stats['error_rate']}")
```

## Configuration

### Global Functions

#### `get_rate_limiter() -> AdaptiveRateLimiter`

Get or create the global rate limiter instance.

**Returns:**
- `AdaptiveRateLimiter`: Rate limiter instance

**Example:**
```python
from crypto_bot.utils.market_loader import get_rate_limiter

rate_limiter = get_rate_limiter()
```

#### `configure_rate_limiter(**kwargs) -> None`

Configure the global rate limiter with new settings.

**Parameters:**
- `**kwargs`: Configuration parameters

**Example:**
```python
from crypto_bot.utils.market_loader import configure_rate_limiter

configure_rate_limiter(
    max_requests_per_minute=20,
    base_delay=0.5,
    max_delay=5.0
)
```

#### `get_cache_manager() -> AdaptiveCacheManager`

Get or create the global cache manager instance.

**Returns:**
- `AdaptiveCacheManager`: Cache manager instance

**Example:**
```python
from crypto_bot.utils.scan_cache_manager import get_cache_manager

cache_manager = get_cache_manager()
```

#### `configure_cache_manager(**kwargs) -> None`

Configure the global cache manager with new settings.

**Parameters:**
- `**kwargs`: Configuration parameters

**Example:**
```python
from crypto_bot.utils.scan_cache_manager import configure_cache_manager

configure_cache_manager(
    initial_size=2000,
    max_size=15000,
    min_size=200
)
```

#### `get_database_manager() -> DatabaseManager`

Get or create the global database manager instance.

**Returns:**
- `DatabaseManager`: Database manager instance

**Example:**
```python
from crypto_bot.utils.database import get_database_manager

db_manager = get_database_manager()
```

#### `initialize_database(config: Dict[str, Any]) -> DatabaseManager`

Initialize the global database manager with configuration.

**Parameters:**
- `config` (`Dict[str, Any]`): Database configuration dictionary

**Returns:**
- `DatabaseManager`: Initialized database manager instance

**Example:**
```python
from crypto_bot.utils.database import initialize_database

config = {
    "host": "localhost",
    "port": 5432,
    "database": "crypto_bot",
    "user": "postgres",
    "password": "password"
}

db_manager = await initialize_database(config)
```

#### `close_database() -> None`

Close the global database manager.

**Example:**
```python
from crypto_bot.utils.database import close_database

await close_database()
```

## Integration Examples

### Complete Performance Monitoring Setup

```python
import asyncio
from crypto_bot.main import MemoryManager
from crypto_bot.utils.market_loader import AdaptiveRateLimiter
from crypto_bot.utils.scan_cache_manager import AdaptiveCacheManager
from crypto_bot.utils.telemetry import PerformanceMonitor
from crypto_bot.utils.database import DatabaseManager

async def setup_performance_monitoring():
    # Initialize all components
    memory_manager = MemoryManager(memory_threshold=0.8)
    rate_limiter = AdaptiveRateLimiter(max_requests_per_minute=10)
    cache_manager = AdaptiveCacheManager(initial_size=1000)
    performance_monitor = PerformanceMonitor()
    
    # Initialize database (optional)
    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "crypto_bot",
        "user": "postgres",
        "password": "password"
    }
    db_manager = DatabaseManager(**db_config)
    await db_manager.initialize()
    
    return {
        "memory_manager": memory_manager,
        "rate_limiter": rate_limiter,
        "cache_manager": cache_manager,
        "performance_monitor": performance_monitor,
        "db_manager": db_manager
    }

async def monitor_performance(components):
    """Monitor performance and take action when needed."""
    
    # Check memory pressure
    if components["memory_manager"].check_memory_pressure():
        components["memory_manager"].optimize_cache_sizes(df_cache, regime_cache)
        components["memory_manager"].force_garbage_collection()
        components["performance_monitor"].record_metric("memory", 85.0)
    
    # Record API performance
    components["performance_monitor"].record_metric("response_time", 2.5)
    components["performance_monitor"].record_metric("error", 5.0)
    
    # Check for alerts
    alerts = components["performance_monitor"].get_alert_conditions()
    for alert in alerts:
        print(f"Performance Alert: {alert['message']}")
    
    # Export metrics periodically
    components["performance_monitor"].export_metrics("json", "performance_report.json")

# Usage
async def main():
    components = await setup_performance_monitoring()
    
    # Monitor performance every minute
    while True:
        await monitor_performance(components)
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
```

### Rate Limiting Integration

```python
from crypto_bot.utils.market_loader import AdaptiveRateLimiter

async def make_api_call(rate_limiter, endpoint):
    """Make an API call with rate limiting."""
    
    # Wait if needed
    await rate_limiter.wait_if_needed()
    
    try:
        # Make the API call
        response = await api_client.get(endpoint)
        rate_limiter.record_success()
        return response
    except Exception as e:
        rate_limiter.record_error()
        raise e

# Usage
rate_limiter = AdaptiveRateLimiter()
response = await make_api_call(rate_limiter, "/api/v1/ticker")
```

### Cache Management Integration

```python
from crypto_bot.utils.scan_cache_manager import AdaptiveCacheManager

def get_cached_data(cache_manager, cache_type, key, fetch_func):
    """Get data from cache or fetch if not available."""
    
    # Try to get from cache
    data = cache_manager.get(cache_type, key)
    if data is not None:
        return data
    
    # Fetch data if not in cache
    data = fetch_func(key)
    
    # Store in cache
    cache_manager.set(cache_type, key, data)
    
    return data

# Usage
cache_manager = AdaptiveCacheManager()
ohlcv_data = get_cached_data(
    cache_manager,
    "ohlcv",
    "BTC/USD",
    lambda key: fetch_ohlcv_data(key)
)
```

---

For more information about performance optimization, see [PERFORMANCE.md](PERFORMANCE.md).
