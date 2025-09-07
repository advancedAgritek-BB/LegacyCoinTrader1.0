"""
Caching Infrastructure

High-performance caching layer with Redis and in-memory fallback,
support for TTL, serialization, and comprehensive monitoring.
"""

import asyncio
import json
import pickle
import time
from typing import Any, Optional, Dict, Union, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta

import redis.asyncio as redis
from redis.exceptions import RedisError, ConnectionError

from ..core.config import RedisConfig, get_settings
from ..utils.logger import get_logger
from ..utils.metrics import get_metrics_collector


logger = get_logger(__name__)


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    avg_response_time: float = 0.0
    last_access: Optional[datetime] = None

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_operations(self) -> int:
        """Get total cache operations."""
        return self.hits + self.misses + self.sets + self.deletes


class CacheInterface(ABC):
    """Abstract base class for cache implementations."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get TTL for cache key."""
        pass

    @abstractmethod
    async def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform cache health check."""
        pass


class RedisCache(CacheInterface):
    """
    Redis-based cache implementation with connection pooling and error handling.

    Features:
    - Connection pooling and automatic reconnection
    - JSON serialization for complex objects
    - TTL support with automatic expiration
    - Comprehensive error handling and logging
    - Performance monitoring and health checks
    """

    def __init__(self, config: RedisConfig, logger=None, metrics=None):
        """
        Initialize Redis cache.

        Args:
            config: Redis configuration.
            logger: Logger instance.
            metrics: Metrics collector instance.
        """
        self.config = config
        self.logger = logger or get_logger(__name__)
        self.metrics = metrics or get_metrics_collector()
        self._client: Optional[redis.Redis] = None
        self._stats = CacheStats()
        self._connected = False

    async def _ensure_connection(self) -> None:
        """Ensure Redis connection is established."""
        if self._client is None:
            try:
                self._client = redis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    password=self.config.password.get_secret_value() if self.config.password else None,
                    ssl=self.config.ssl,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.socket_connect_timeout,
                    socket_keepalive=self.config.socket_keepalive,
                    socket_keepalive_options=self.config.socket_keepalive_options,
                    retry_on_timeout=True,
                    max_connections=20,  # Connection pool size
                    decode_responses=False  # Keep bytes for serialization
                )

                # Test connection
                await self._client.ping()
                self._connected = True
                self.logger.info("Redis cache connection established")

            except Exception as e:
                self.logger.error(f"Failed to connect to Redis: {e}")
                self._client = None
                self._connected = False
                raise

    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value for storage.

        Args:
            value: Value to serialize.

        Returns:
            bytes: Serialized value.
        """
        try:
            # Try JSON serialization first for better readability
            json_str = json.dumps(value, default=str, separators=(',', ':'))
            return json_str.encode('utf-8')
        except (TypeError, ValueError):
            # Fallback to pickle for complex objects
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize value from storage.

        Args:
            data: Serialized data.

        Returns:
            Any: Deserialized value.
        """
        try:
            # Try JSON deserialization first
            json_str = data.decode('utf-8')
            return json.loads(json_str)
        except (UnicodeDecodeError, json.JSONDecodeError):
            # Fallback to pickle
            return pickle.loads(data)

    async def _execute_with_timing(self, operation: str, func) -> Any:
        """Execute cache operation with timing and error handling."""
        start_time = time.time()

        try:
            result = await func()
            response_time = time.time() - start_time

            # Update statistics
            if operation == "get" and result is not None:
                self._stats.hits += 1
            elif operation == "get":
                self._stats.misses += 1
            elif operation == "set":
                self._stats.sets += 1
            elif operation == "delete":
                self._stats.deletes += 1

            # Update average response time
            total_ops = self._stats.total_operations
            if total_ops > 0:
                self._stats.avg_response_time = (
                    (self._stats.avg_response_time * (total_ops - 1)) + response_time
                ) / total_ops

            self._stats.last_access = datetime.utcnow()

            # Record metrics
            self.metrics.histogram(
                f"cache.{operation}.duration",
                response_time,
                tags={"cache_type": "redis"}
            )

            return result

        except Exception as e:
            self._stats.errors += 1
            self.logger.error(f"Redis cache {operation} error: {e}")

            # Record error metrics
            self.metrics.increment(
                f"cache.{operation}.errors",
                tags={"cache_type": "redis", "error_type": type(e).__name__}
            )

            raise

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        await self._ensure_connection()

        def _get():
            if not self._client:
                raise ConnectionError("Redis client not available")
            return self._client.get(key)

        try:
            data = await self._execute_with_timing("get", _get)
            if data is None:
                return None
            return self._deserialize(data)
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache with optional TTL."""
        await self._ensure_connection()

        def _set():
            if not self._client:
                raise ConnectionError("Redis client not available")
            serialized = self._serialize(value)
            return self._client.set(key, serialized, ex=ttl)

        try:
            result = await self._execute_with_timing("set", _set)
            return bool(result)
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        await self._ensure_connection()

        def _delete():
            if not self._client:
                raise ConnectionError("Redis client not available")
            return self._client.delete(key)

        try:
            result = await self._execute_with_timing("delete", _delete)
            return result > 0
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        await self._ensure_connection()

        def _exists():
            if not self._client:
                raise ConnectionError("Redis client not available")
            return self._client.exists(key)

        try:
            result = await self._exists_timing("_exists", _exists)
            return result > 0
        except Exception:
            return False

    async def clear(self) -> bool:
        """Clear all cache entries in current database."""
        await self._ensure_connection()

        def _clear():
            if not self._client:
                raise ConnectionError("Redis client not available")
            return self._client.flushdb()

        try:
            result = await self._execute_with_timing("clear", _clear)
            return bool(result)
        except Exception:
            return False

    async def get_ttl(self, key: str) -> Optional[int]:
        """Get TTL for Redis cache key."""
        await self._ensure_connection()

        def _get_ttl():
            if not self._client:
                raise ConnectionError("Redis client not available")
            return self._client.ttl(key)

        try:
            ttl = await self._get_ttl_timing("_get_ttl", _get_ttl)
            return ttl if ttl > 0 else None
        except Exception:
            return None

    async def get_stats(self) -> CacheStats:
        """Get Redis cache statistics."""
        return self._stats

    async def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check."""
        health_info = {
            "cache_type": "redis",
            "connected": self._connected,
            "host": self.config.host,
            "port": self.config.port,
            "database": self.config.db,
            "timestamp": datetime.utcnow().isoformat()
        }

        if self._connected and self._client:
            try:
                # Test basic operations
                ping_result = await self._client.ping()
                info_result = await self._client.info()

                health_info.update({
                    "ping_success": bool(ping_result),
                    "info_available": bool(info_result),
                    "status": "healthy"
                })

                # Add some Redis stats
                if info_result:
                    health_info.update({
                        "used_memory": info_result.get("used_memory_human", "unknown"),
                        "connected_clients": info_result.get("connected_clients", 0),
                        "uptime_seconds": info_result.get("uptime_in_seconds", 0)
                    })

            except Exception as e:
                health_info.update({
                    "status": "unhealthy",
                    "error": str(e)
                })
        else:
            health_info["status"] = "disconnected"

        # Add cache statistics
        stats = self.get_stats()
        health_info.update({
            "stats": {
                "hits": stats.hits,
                "misses": stats.misses,
                "hit_rate": stats.hit_rate,
                "total_operations": stats.total_operations,
                "errors": stats.errors,
                "avg_response_time": stats.avg_response_time
            }
        })

        return health_info


class InMemoryCache(CacheInterface):
    """
    In-memory cache implementation with TTL support.

    Features:
    - Fast in-memory storage with TTL
    - Automatic cleanup of expired entries
    - Thread-safe operations
    - Memory usage monitoring
    """

    def __init__(self, logger=None, metrics=None, max_size: int = 10000):
        """
        Initialize in-memory cache.

        Args:
            logger: Logger instance.
            metrics: Metrics collector instance.
            max_size: Maximum number of cache entries.
        """
        self.logger = logger or get_logger(__name__)
        self.metrics = metrics or get_metrics_collector()
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._stats = CacheStats()

    def _cleanup_expired(self) -> None:
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = []

        for key, data in self._cache.items():
            if data.get("expires_at") and current_time > data["expires_at"]:
                expired_keys.append(key)

        for key in expired_keys:
            del self._cache[key]

    def _evict_lru(self) -> None:
        """Evict least recently used entries when cache is full."""
        if len(self._cache) >= self.max_size:
            # Simple LRU: remove oldest entries
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].get("last_access", 0)
            )
            to_remove = len(sorted_entries) - self.max_size + 100  # Remove 100 extra

            for key, _ in sorted_entries[:to_remove]:
                del self._cache[key]

    async def get(self, key: str) -> Optional[Any]:
        """Get value from in-memory cache."""
        start_time = time.time()

        try:
            self._cleanup_expired()

            if key not in self._cache:
                self._stats.misses += 1
                return None

            data = self._cache[key]
            expires_at = data.get("expires_at")

            if expires_at and time.time() > expires_at:
                del self._cache[key]
                self._stats.misses += 1
                return None

            # Update last access time
            data["last_access"] = time.time()
            self._stats.hits += 1

            response_time = time.time() - start_time
            self._stats.avg_response_time = (
                (self._stats.avg_response_time * (self._stats.total_operations - 1)) + response_time
            ) / self._stats.total_operations if self._stats.total_operations > 0 else response_time

            self._stats.last_access = datetime.utcnow()

            self.metrics.histogram(
                "cache.get.duration",
                response_time,
                tags={"cache_type": "memory"}
            )

            return data["value"]

        except Exception as e:
            self._stats.errors += 1
            self.logger.error(f"In-memory cache get error: {e}")
            self.metrics.increment(
                "cache.get.errors",
                tags={"cache_type": "memory", "error_type": type(e).__name__}
            )
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in in-memory cache with optional TTL."""
        start_time = time.time()

        try:
            self._evict_lru()

            expires_at = time.time() + ttl if ttl else None

            self._cache[key] = {
                "value": value,
                "expires_at": expires_at,
                "created_at": time.time(),
                "last_access": time.time()
            }

            self._stats.sets += 1

            response_time = time.time() - start_time
            self._stats.avg_response_time = (
                (self._stats.avg_response_time * (self._stats.total_operations - 1)) + response_time
            ) / self._stats.total_operations if self._stats.total_operations > 0 else response_time

            self.metrics.histogram(
                "cache.set.duration",
                response_time,
                tags={"cache_type": "memory"}
            )

            return True

        except Exception as e:
            self._stats.errors += 1
            self.logger.error(f"In-memory cache set error: {e}")
            self.metrics.increment(
                "cache.set.errors",
                tags={"cache_type": "memory", "error_type": type(e).__name__}
            )
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from in-memory cache."""
        start_time = time.time()

        try:
            if key in self._cache:
                del self._cache[key]
                self._stats.deletes += 1

                response_time = time.time() - start_time
                self._stats.avg_response_time = (
                    (self._stats.avg_response_time * (self._stats.total_operations - 1)) + response_time
                ) / self._stats.total_operations if self._stats.total_operations > 0 else response_time

                self.metrics.histogram(
                    "cache.delete.duration",
                    response_time,
                    tags={"cache_type": "memory"}
                )

                return True

            return False

        except Exception as e:
            self._stats.errors += 1
            self.logger.error(f"In-memory cache delete error: {e}")
            self.metrics.increment(
                "cache.delete.errors",
                tags={"cache_type": "memory", "error_type": type(e).__name__}
            )
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in in-memory cache."""
        try:
            self._cleanup_expired()
            return key in self._cache
        except Exception as e:
            self.logger.error(f"In-memory cache exists error: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all in-memory cache entries."""
        try:
            self._cache.clear()
            self.logger.info("In-memory cache cleared")
            return True
        except Exception as e:
            self.logger.error(f"In-memory cache clear error: {e}")
            return False

    async def get_ttl(self, key: str) -> Optional[int]:
        """Get TTL for in-memory cache key."""
        try:
            if key not in self._cache:
                return None

            data = self._cache[key]
            expires_at = data.get("expires_at")

            if not expires_at:
                return None

            remaining = int(expires_at - time.time())
            return remaining if remaining > 0 else None

        except Exception as e:
            self.logger.error(f"In-memory cache get_ttl error: {e}")
            return None

    async def get_stats(self) -> CacheStats:
        """Get in-memory cache statistics."""
        return self._stats

    async def health_check(self) -> Dict[str, Any]:
        """Perform in-memory cache health check."""
        current_time = time.time()

        # Count valid entries
        valid_entries = 0
        expired_entries = 0

        for data in self._cache.values():
            expires_at = data.get("expires_at")
            if not expires_at or current_time <= expires_at:
                valid_entries += 1
            else:
                expired_entries += 1

        health_info = {
            "cache_type": "memory",
            "status": "healthy",
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "max_size": self.max_size,
            "utilization_percent": (len(self._cache) / self.max_size) * 100,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Add cache statistics
        stats = self.get_stats()
        health_info.update({
            "stats": {
                "hits": stats.hits,
                "misses": stats.misses,
                "hit_rate": stats.hit_rate,
                "total_operations": stats.total_operations,
                "errors": stats.errors,
                "avg_response_time": stats.avg_response_time
            }
        })

        return health_info


class CacheManager:
    """
    Cache manager with Redis primary and in-memory fallback.

    Provides unified interface for caching with automatic failover
    and performance optimization.
    """

    def __init__(self, redis_cache: RedisCache, memory_cache: InMemoryCache):
        """
        Initialize cache manager.

        Args:
            redis_cache: Redis cache instance.
            memory_cache: In-memory cache instance.
        """
        self.redis_cache = redis_cache
        self.memory_cache = memory_cache
        self.logger = get_logger(__name__)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with fallback."""
        # Try Redis first
        try:
            value = await self.redis_cache.get(key)
            if value is not None:
                return value
        except Exception as e:
            self.logger.warning(f"Redis cache failed, falling back to memory: {e}")

        # Fallback to memory cache
        return await self.memory_cache.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with replication."""
        redis_success = False
        memory_success = False

        # Try Redis first
        try:
            redis_success = await self.redis_cache.set(key, value, ttl)
        except Exception as e:
            self.logger.warning(f"Redis cache set failed: {e}")

        # Always try memory cache
        try:
            memory_success = await self.memory_cache.set(key, value, ttl)
        except Exception as e:
            self.logger.error(f"Memory cache set failed: {e}")

        return redis_success or memory_success

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        redis_success = False
        memory_success = False

        # Try Redis first
        try:
            redis_success = await self.redis_cache.delete(key)
        except Exception as e:
            self.logger.warning(f"Redis cache delete failed: {e}")

        # Always try memory cache
        try:
            memory_success = await self.memory_cache.delete(key)
        except Exception as e:
            self.logger.error(f"Memory cache delete failed: {e}")

        return redis_success or memory_success

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        # Try Redis first
        try:
            if await self.redis_cache.exists(key):
                return True
        except Exception as e:
            self.logger.warning(f"Redis cache exists check failed: {e}")

        # Fallback to memory cache
        try:
            return await self.memory_cache.exists(key)
        except Exception as e:
            self.logger.error(f"Memory cache exists check failed: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cache entries."""
        redis_success = False
        memory_success = False

        # Try Redis first
        try:
            redis_success = await self.redis_cache.clear()
        except Exception as e:
            self.logger.warning(f"Redis cache clear failed: {e}")

        # Always try memory cache
        try:
            memory_success = await self.memory_cache.clear()
        except Exception as e:
            self.logger.error(f"Memory cache clear failed: {e}")

        return redis_success and memory_success

    async def get_ttl(self, key: str) -> Optional[int]:
        """Get TTL for cache key."""
        # Try Redis first
        try:
            ttl = await self.redis_cache.get_ttl(key)
            if ttl is not None:
                return ttl
        except Exception as e:
            self.logger.warning(f"Redis cache get_ttl failed: {e}")

        # Fallback to memory cache
        try:
            return await self.memory_cache.get_ttl(key)
        except Exception as e:
            self.logger.error(f"Memory cache get_ttl failed: {e}")
            return None

    async def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all cache layers."""
        return {
            "redis": await self.redis_cache.get_stats(),
            "memory": await self.memory_cache.get_stats()
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        redis_health = await self.redis_cache.health_check()
        memory_health = await self.memory_cache.health_check()

        overall_status = "healthy"
        if redis_health.get("status") == "unhealthy" and memory_health.get("status") == "unhealthy":
            overall_status = "unhealthy"
        elif redis_health.get("status") == "unhealthy":
            overall_status = "degraded"

        return {
            "overall_status": overall_status,
            "redis": redis_health,
            "memory": memory_health,
            "timestamp": datetime.utcnow().isoformat()
        }


# Export all cache components
__all__ = [
    "CacheStats",
    "CacheInterface",
    "RedisCache",
    "InMemoryCache",
    "CacheManager",
]
