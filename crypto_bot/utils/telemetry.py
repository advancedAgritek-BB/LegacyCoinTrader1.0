from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import asyncio
import time
import psutil
import json
import csv

from prometheus_client import Counter, Histogram, Gauge, REGISTRY

from .logger import LOG_DIR, setup_logger
from .metrics_logger import log_metrics_to_csv

LOG_FILE = LOG_DIR / "telemetry.csv"
logger = setup_logger(__name__, LOG_DIR / "telemetry.log")

def clear_prometheus_registry():
    """Clear the Prometheus registry to prevent duplicate metric errors."""
    try:
        # Get all collectors
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            REGISTRY.unregister(collector)
        logger.info("Cleared Prometheus registry")
    except Exception as e:
        logger.warning(f"Failed to clear Prometheus registry: {e}")

# Clear registry on module import to prevent duplicate errors
clear_prometheus_registry()

# Prometheus counters keyed by telemetry name
PROM_COUNTERS: Dict[str, Counter] = {
    "analysis.skipped_no_df": Counter(
        "analysis_skipped_no_df",
        "Number of symbols skipped due to missing OHLCV data",
    ),
    "scan.ws_errors": Counter(
        "scan_ws_errors",
        "Number of WebSocket errors encountered while scanning",
    ),
}

# Prometheus histograms for performance metrics
PROM_HISTOGRAMS: Dict[str, Histogram] = {
    "api_response_time": Histogram(
        "api_response_time_seconds",
        "API response time in seconds",
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    ),
    "memory_usage": Histogram(
        "memory_usage_percent",
        "Memory usage percentage",
        buckets=[50, 60, 70, 80, 85, 90, 95]
    ),
    "cache_hit_rate": Histogram(
        "cache_hit_rate",
        "Cache hit rate",
        buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    ),
}

# Prometheus gauges for current values
PROM_GAUGES: Dict[str, Gauge] = {
    "current_memory_usage": Gauge("current_memory_usage_percent", "Current memory usage percentage"),
    "current_error_rate": Gauge("current_error_rate", "Current error rate"),
    "current_throughput": Gauge("current_throughput_rps", "Current requests per second"),
    "active_connections": Gauge("active_connections", "Number of active connections"),
}


class PerformanceMonitor:
    """Comprehensive performance monitoring for the trading bot."""
    
    def __init__(
        self,
        memory_threshold: float = 85.0,
        error_rate_threshold: float = 10.0,
        response_time_threshold: float = 5.0,
        cache_hit_rate_threshold: float = 50.0,
        metrics_window_size: int = 1000,
        alert_enabled: bool = True
    ):
        """
        Initialize the performance monitor.
        
        Args:
            memory_threshold: Memory usage threshold for alerts (%)
            error_rate_threshold: Error rate threshold for alerts (%)
            response_time_threshold: Response time threshold for alerts (seconds)
            cache_hit_rate_threshold: Cache hit rate threshold for alerts (%)
            metrics_window_size: Size of rolling metrics window
            alert_enabled: Whether to enable performance alerts
        """
        self.memory_threshold = memory_threshold
        self.error_rate_threshold = error_rate_threshold
        self.response_time_threshold = response_time_threshold
        self.cache_hit_rate_threshold = cache_hit_rate_threshold
        self.metrics_window_size = metrics_window_size
        self.alert_enabled = alert_enabled
        
        # Rolling windows for metrics
        self.memory_usage = deque(maxlen=metrics_window_size)
        self.api_response_times = deque(maxlen=metrics_window_size)
        self.cache_hit_rates = defaultdict(lambda: deque(maxlen=metrics_window_size))
        self.error_rates = deque(maxlen=metrics_window_size)
        self.throughput_metrics = deque(maxlen=metrics_window_size)
        
        # Counters
        self.total_requests = 0
        self.total_errors = 0
        self.total_cache_hits = 0
        self.total_cache_misses = 0
        
        # Timestamps for rate calculations
        self.last_throughput_calc = time.time()
        self.request_times = deque(maxlen=metrics_window_size)
        
        # Alert tracking
        self.alerts_triggered = defaultdict(int)
        self.last_alert_time = defaultdict(float)
        self.alert_cooldown = 300  # 5 minutes between alerts
        
        self.logger = setup_logger("performance_monitor", LOG_DIR / "performance_monitor.log")
        
    def record_metric(self, metric_type: str, value: float, **kwargs) -> None:
        """
        Record a performance metric.
        
        Args:
            metric_type: Type of metric ('memory', 'response_time', 'cache_hit', 'error', 'throughput')
            value: Metric value
            **kwargs: Additional context (e.g., cache_type, api_endpoint)
        """
        timestamp = time.time()
        
        if metric_type == "memory":
            self.memory_usage.append((timestamp, value))
            PROM_GAUGES["current_memory_usage"].set(value)
            PROM_HISTOGRAMS["memory_usage"].observe(value)
            
        elif metric_type == "response_time":
            self.api_response_times.append((timestamp, value))
            PROM_HISTOGRAMS["api_response_time"].observe(value)
            
        elif metric_type == "cache_hit":
            cache_type = kwargs.get("cache_type", "default")
            self.cache_hit_rates[cache_type].append((timestamp, value))
            PROM_HISTOGRAMS["cache_hit_rate"].observe(value)
            
        elif metric_type == "error":
            self.error_rates.append((timestamp, value))
            PROM_GAUGES["current_error_rate"].set(value)
            
        elif metric_type == "throughput":
            self.throughput_metrics.append((timestamp, value))
            PROM_GAUGES["current_throughput"].set(value)
            
        elif metric_type == "request":
            self.total_requests += 1
            self.request_times.append(timestamp)
            
        elif metric_type == "cache_access":
            if value > 0:  # Hit
                self.total_cache_hits += 1
            else:  # Miss
                self.total_cache_misses += 1
                
        # Update Prometheus counters
        if metric_type in PROM_COUNTERS:
            PROM_COUNTERS[metric_type].inc()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance summary.
        
        Returns:
            Dictionary with performance metrics and analysis
        """
        now = time.time()
        
        # Calculate memory metrics
        memory_stats = self._calculate_memory_stats()
        
        # Calculate API response time metrics
        response_stats = self._calculate_response_stats()
        
        # Calculate cache performance
        cache_stats = self._calculate_cache_stats()
        
        # Calculate error rates
        error_stats = self._calculate_error_stats()
        
        # Calculate throughput
        throughput_stats = self._calculate_throughput_stats()
        
        # Check for performance issues
        alerts = self.get_alert_conditions()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "memory": memory_stats,
            "api_response_times": response_stats,
            "cache_performance": cache_stats,
            "error_rates": error_stats,
            "throughput": throughput_stats,
            "alerts": alerts,
            "summary": {
                "overall_health": "good" if not alerts else "degraded",
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "uptime_seconds": now - self.last_throughput_calc if self.last_throughput_calc else 0
            }
        }
    
    def get_alert_conditions(self) -> List[Dict[str, Any]]:
        """
        Identify performance issues that require attention.
        
        Returns:
            List of alert conditions
        """
        alerts = []
        now = time.time()
        
        # Memory usage alert
        if self.memory_usage:
            latest_memory = self.memory_usage[-1][1]
            if latest_memory > self.memory_threshold:
                if self._should_trigger_alert("memory", now):
                    alerts.append({
                        "type": "high_memory_usage",
                        "severity": "warning",
                        "value": latest_memory,
                        "threshold": self.memory_threshold,
                        "message": f"Memory usage {latest_memory:.1f}% exceeds threshold {self.memory_threshold}%"
                    })
        
        # Error rate alert
        if self.error_rates:
            latest_error_rate = self.error_rates[-1][1]
            if latest_error_rate > self.error_rate_threshold:
                if self._should_trigger_alert("error_rate", now):
                    alerts.append({
                        "type": "high_error_rate",
                        "severity": "error",
                        "value": latest_error_rate,
                        "threshold": self.error_rate_threshold,
                        "message": f"Error rate {latest_error_rate:.1f}% exceeds threshold {self.error_rate_threshold}%"
                    })
        
        # Response time alert
        if self.api_response_times:
            latest_response_time = self.api_response_times[-1][1]
            if latest_response_time > self.response_time_threshold:
                if self._should_trigger_alert("response_time", now):
                    alerts.append({
                        "type": "slow_response_time",
                        "severity": "warning",
                        "value": latest_response_time,
                        "threshold": self.response_time_threshold,
                        "message": f"Response time {latest_response_time:.2f}s exceeds threshold {self.response_time_threshold}s"
                    })
        
        # Cache hit rate alert
        for cache_type, hit_rates in self.cache_hit_rates.items():
            if hit_rates:
                latest_hit_rate = hit_rates[-1][1]
                if latest_hit_rate < self.cache_hit_rate_threshold:
                    if self._should_trigger_alert(f"cache_hit_rate_{cache_type}", now):
                        alerts.append({
                            "type": "low_cache_hit_rate",
                            "severity": "info",
                            "cache_type": cache_type,
                            "value": latest_hit_rate,
                            "threshold": self.cache_hit_rate_threshold,
                            "message": f"Cache hit rate for {cache_type}: {latest_hit_rate:.1f}% below threshold {self.cache_hit_rate_threshold}%"
                        })
        
        return alerts
    
    def export_metrics(self, format: str = "json", path: Optional[Union[str, Path]] = None) -> None:
        """
        Export metrics to file.
        
        Args:
            format: Export format ('json', 'csv')
            path: Output file path
        """
        if path is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = LOG_DIR / f"performance_metrics_{timestamp}.{format}"
        
        path = Path(path)
        report = self.get_performance_report()
        
        if format.lower() == "json":
            with open(path, 'w') as f:
                json.dump(report, f, indent=2)
        elif format.lower() == "csv":
            self._export_csv(report, path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Performance metrics exported to {path}")
    
    def _calculate_memory_stats(self) -> Dict[str, Any]:
        """Calculate memory usage statistics."""
        if not self.memory_usage:
            return {"current": 0, "average": 0, "max": 0, "min": 0}
        
        values = [entry[1] for entry in self.memory_usage]
        return {
            "current": values[-1],
            "average": sum(values) / len(values),
            "max": max(values),
            "min": min(values),
            "samples": len(values)
        }
    
    def _calculate_response_stats(self) -> Dict[str, Any]:
        """Calculate API response time statistics."""
        if not self.api_response_times:
            return {"current": 0, "average": 0, "max": 0, "min": 0, "p95": 0}
        
        values = [entry[1] for entry in self.api_response_times]
        sorted_values = sorted(values)
        p95_index = int(len(sorted_values) * 0.95)
        
        return {
            "current": values[-1],
            "average": sum(values) / len(values),
            "max": max(values),
            "min": min(values),
            "p95": sorted_values[p95_index] if p95_index < len(sorted_values) else 0,
            "samples": len(values)
        }
    
    def _calculate_cache_stats(self) -> Dict[str, Any]:
        """Calculate cache performance statistics."""
        stats = {}
        
        for cache_type, hit_rates in self.cache_hit_rates.items():
            if hit_rates:
                values = [entry[1] for entry in hit_rates]
                stats[cache_type] = {
                    "current": values[-1],
                    "average": sum(values) / len(values),
                    "max": max(values),
                    "min": min(values),
                    "samples": len(values)
                }
        
        # Overall cache stats
        total_accesses = self.total_cache_hits + self.total_cache_misses
        if total_accesses > 0:
            stats["overall"] = {
                "hit_rate": (self.total_cache_hits / total_accesses) * 100,
                "total_hits": self.total_cache_hits,
                "total_misses": self.total_cache_misses,
                "total_accesses": total_accesses
            }
        
        return stats
    
    def _calculate_error_stats(self) -> Dict[str, Any]:
        """Calculate error rate statistics."""
        if not self.error_rates:
            return {"current": 0, "average": 0, "max": 0, "min": 0}
        
        values = [entry[1] for entry in self.error_rates]
        return {
            "current": values[-1],
            "average": sum(values) / len(values),
            "max": max(values),
            "min": min(values),
            "samples": len(values)
        }
    
    def _calculate_throughput_stats(self) -> Dict[str, Any]:
        """Calculate throughput statistics."""
        if not self.throughput_metrics:
            return {"current": 0, "average": 0, "max": 0, "min": 0}
        
        values = [entry[1] for entry in self.throughput_metrics]
        return {
            "current": values[-1],
            "average": sum(values) / len(values),
            "max": max(values),
            "min": min(values),
            "samples": len(values)
        }
    
    def _should_trigger_alert(self, alert_type: str, current_time: float) -> bool:
        """Check if an alert should be triggered based on cooldown."""
        if not self.alert_enabled:
            return False
        
        last_time = self.last_alert_time.get(alert_type, 0)
        if current_time - last_time > self.alert_cooldown:
            self.last_alert_time[alert_type] = current_time
            self.alerts_triggered[alert_type] += 1
            return True
        
        return False
    
    def _export_csv(self, report: Dict[str, Any], path: Path) -> None:
        """Export performance report to CSV format."""
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(["Metric", "Value", "Unit"])
            
            # Write memory stats
            memory = report.get("memory", {})
            writer.writerow(["Memory_Current", memory.get("current", 0), "%"])
            writer.writerow(["Memory_Average", memory.get("average", 0), "%"])
            writer.writerow(["Memory_Max", memory.get("max", 0), "%"])
            
            # Write response time stats
            response = report.get("api_response_times", {})
            writer.writerow(["ResponseTime_Current", response.get("current", 0), "seconds"])
            writer.writerow(["ResponseTime_Average", response.get("average", 0), "seconds"])
            writer.writerow(["ResponseTime_P95", response.get("p95", 0), "seconds"])
            
            # Write cache stats
            cache = report.get("cache_performance", {}).get("overall", {})
            writer.writerow(["CacheHitRate", cache.get("hit_rate", 0), "%"])
            writer.writerow(["CacheTotalHits", cache.get("total_hits", 0), "count"])
            writer.writerow(["CacheTotalMisses", cache.get("total_misses", 0), "count"])


class Telemetry:
    """Collect simple counter metrics during runtime."""

    def __init__(self) -> None:
        self._counters: Dict[str, int] = defaultdict(int)

    def inc(self, name: str, value: int = 1) -> None:
        """Increment ``name`` by ``value``."""
        self._counters[name] += value
        counter = PROM_COUNTERS.get(name)
        if counter is not None:
            counter.inc(value)

    def snapshot(self) -> Dict[str, int]:
        """Return a copy of all counters."""
        return dict(self._counters)

    def reset(self) -> None:
        """Reset all counters."""
        self._counters.clear()

    def export_csv(self, path: Union[str, Path, None] = None) -> None:
        """Append current counters to a CSV file."""
        file = Path(path or LOG_FILE)
        metrics = {**self.snapshot(), "timestamp": datetime.utcnow().isoformat()}
        log_metrics_to_csv(metrics, str(file))


# Global instances
telemetry = Telemetry()
performance_monitor = PerformanceMonitor()


def write_cycle_metrics(metrics: Dict[str, Any], cfg: Dict) -> None:
    """Write cycle metrics and export telemetry counters.

    Parameters
    ----------
    metrics:
        Mapping of metric names to values.
    cfg:
        Bot configuration dictionary providing metric settings.
    """
    if cfg.get("metrics_enabled") and cfg.get("metrics_backend") == "csv":
        log_metrics_to_csv(
            metrics,
            cfg.get("metrics_output_file", str(LOG_DIR / "metrics.csv")),
        )
        telemetry.export_csv(
            cfg.get("metrics_output_file", str(LOG_DIR / "telemetry.csv"))
        )
    
    # Record performance metrics
    if cfg.get("performance", {}).get("enable_performance_monitoring", True):
        # Record memory usage
        try:
            memory_percent = psutil.virtual_memory().percent
            performance_monitor.record_metric("memory", memory_percent)
        except Exception as e:
            logger.warning(f"Failed to record memory metric: {e}")
        
        # Record cycle timing metrics
        if "ticker_fetch_time" in metrics:
            performance_monitor.record_metric("response_time", metrics["ticker_fetch_time"])
        
        # Export performance metrics periodically
        export_interval = cfg.get("performance", {}).get("metrics_export_interval", 300)
        if int(time.time()) % export_interval == 0:
            performance_monitor.export_metrics("json")


def dump() -> str:
    """Log current counters and return formatted string."""
    snap = telemetry.snapshot()
    msg = ", ".join(f"{k}: {v}" for k, v in sorted(snap.items()))
    logger.info("Telemetry snapshot - %s", msg)
    return msg


