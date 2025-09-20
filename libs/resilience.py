"""Comprehensive resilience layer combining circuit breakers, retry policies, and monitoring."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenException,
    call_with_circuit_breaker,
    circuit_breaker_registry,
)
from .retry_policy import (
    RetryConfig,
    RetryExhaustedException,
    RetryPolicy,
    execute_with_retry,
    retry_policy_registry,
)

logger = logging.getLogger(__name__)


@dataclass
class ResilienceConfig:
    """Configuration for comprehensive resilience."""
    service_name: str
    circuit_breaker: Optional[CircuitBreakerConfig] = None
    retry_policy: Optional[RetryConfig] = None
    monitoring_enabled: bool = True
    fallback_enabled: bool = True
    fallback_function: Optional[Callable] = None


@dataclass
class ResilienceMetrics:
    """Comprehensive metrics for resilience operations."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    circuit_breaker_trips: int = 0
    retry_exhaustions: int = 0
    fallback_executions: int = 0
    average_response_time: float = 0.0
    total_response_time: float = 0.0


class ResilienceManager:
    """Manages resilience for service calls with circuit breakers, retries, and fallbacks."""

    def __init__(self, config: ResilienceConfig):
        self.config = config
        self.metrics = ResilienceMetrics()
        self._circuit_breaker: Optional[CircuitBreaker] = None
        self._retry_policy: Optional[RetryPolicy] = None
        self._logger = logging.getLogger(f"{__name__}.{config.service_name}")

    async def initialize(self):
        """Initialize circuit breaker and retry policy."""
        if self.config.circuit_breaker:
            self._circuit_breaker = await circuit_breaker_registry.get_or_create(
                f"{self.config.service_name}_cb",
                self.config.circuit_breaker
            )

        if self.config.retry_policy:
            self._retry_policy = await retry_policy_registry.get_or_create(
                f"{self.config.service_name}_retry",
                self.config.retry_policy
            )

    async def call(
        self,
        func: Callable,
        *args,
        use_circuit_breaker: bool = True,
        use_retry: bool = True,
        use_fallback: bool = True,
        **kwargs
    ) -> Any:
        """Execute a function with full resilience protection."""
        start_time = time.time()

        try:
            self.metrics.total_calls += 1

            # Try circuit breaker first (if enabled)
            if use_circuit_breaker and self._circuit_breaker:
                try:
                    result = await self._circuit_breaker.call(func, *args, **kwargs)
                    self._record_success(time.time() - start_time)
                    return result
                except CircuitBreakerOpenException:
                    self.metrics.circuit_breaker_trips += 1
                    self._logger.warning(
                        f"Circuit breaker open for {self.config.service_name}, attempting direct call"
                    )

            # Try with retry policy (if enabled)
            if use_retry and self._retry_policy:
                try:
                    result = await self._retry_policy.execute(func, *args, **kwargs)
                    self._record_success(time.time() - start_time)
                    return result
                except RetryExhaustedException as e:
                    self.metrics.retry_exhaustions += 1
                    self._logger.warning(
                        f"Retry exhausted for {self.config.service_name}: {e.last_exception}"
                    )
                    # Continue to fallback if available

            # Try direct call (no resilience)
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._record_success(time.time() - start_time)
            return result

        except Exception as e:
            self._record_failure(time.time() - start_time)

            # Try fallback if enabled and available
            if use_fallback and self.config.fallback_enabled and self.config.fallback_function:
                try:
                    self.metrics.fallback_executions += 1
                    self._logger.info(
                        f"Executing fallback for {self.config.service_name} after error: {e}"
                    )
                    fallback_result = await self.config.fallback_function(*args, **kwargs)
                    return fallback_result
                except Exception as fallback_error:
                    self._logger.error(
                        f"Fallback also failed for {self.config.service_name}: {fallback_error}"
                    )

            # Re-raise original exception if no fallback or fallback failed
            raise e

    def _record_success(self, response_time: float):
        """Record a successful call."""
        self.metrics.successful_calls += 1
        self.metrics.total_response_time += response_time
        self.metrics.average_response_time = (
            self.metrics.total_response_time / self.metrics.successful_calls
        )

    def _record_failure(self, response_time: float):
        """Record a failed call."""
        self.metrics.failed_calls += 1
        self.metrics.total_response_time += response_time
        if self.metrics.successful_calls > 0:
            self.metrics.average_response_time = (
                self.metrics.total_response_time / (self.metrics.successful_calls + self.metrics.failed_calls)
            )

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics including circuit breaker and retry metrics."""
        base_metrics = {
            "service_name": self.config.service_name,
            "total_calls": self.metrics.total_calls,
            "successful_calls": self.metrics.successful_calls,
            "failed_calls": self.metrics.failed_calls,
            "success_rate": (
                self.metrics.successful_calls / self.metrics.total_calls
                if self.metrics.total_calls > 0 else 0
            ),
            "circuit_breaker_trips": self.metrics.circuit_breaker_trips,
            "retry_exhaustions": self.metrics.retry_exhaustions,
            "fallback_executions": self.metrics.fallback_executions,
            "average_response_time": self.metrics.average_response_time,
        }

        # Add circuit breaker metrics if available
        if self._circuit_breaker:
            cb_metrics = self._circuit_breaker.get_metrics()
            base_metrics["circuit_breaker"] = cb_metrics

        # Add retry policy metrics if available
        if self._retry_policy:
            retry_metrics = self._retry_policy.get_metrics()
            base_metrics["retry_policy"] = retry_metrics

        return base_metrics


class ResilienceManagerRegistry:
    """Registry for managing multiple resilience managers."""

    def __init__(self):
        self._managers: Dict[str, ResilienceManager] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        service_name: str,
        config: Optional[ResilienceConfig] = None
    ) -> ResilienceManager:
        """Get or create a resilience manager."""
        async with self._lock:
            if service_name not in self._managers:
                if config is None:
                    config = ResilienceConfig(service_name=service_name)
                manager = ResilienceManager(config)
                await manager.initialize()
                self._managers[service_name] = manager
            return self._managers[service_name]

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all resilience managers."""
        return {
            name: manager.get_comprehensive_metrics()
            for name, manager in self._managers.items()
        }


# Global registry instance
resilience_registry = ResilienceManagerRegistry()


async def call_with_resilience(
    service_name: str,
    func: Callable,
    *args,
    resilience_config: Optional[ResilienceConfig] = None,
    **kwargs
) -> Any:
    """Call a function with comprehensive resilience protection."""
    manager = await resilience_registry.get_or_create(service_name, resilience_config)
    return await manager.call(func, *args, **kwargs)


# Convenience functions for common service calls
async def call_market_data_service(
    func: Callable,
    *args,
    **kwargs
) -> Any:
    """Call market data service with optimized resilience settings."""
    config = ResilienceConfig(
        service_name="market-data",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            timeout=10.0
        ),
        retry_policy=RetryConfig(
            max_attempts=2,
            initial_delay=0.5,
            max_delay=5.0
        )
    )
    return await call_with_resilience("market-data", func, *args, resilience_config=config, **kwargs)


async def call_strategy_engine_service(
    func: Callable,
    *args,
    **kwargs
) -> Any:
    """Call strategy engine service with optimized resilience settings."""
    config = ResilienceConfig(
        service_name="strategy-engine",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            timeout=30.0
        ),
        retry_policy=RetryConfig(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=10.0
        )
    )
    return await call_with_resilience("strategy-engine", func, *args, resilience_config=config, **kwargs)


async def call_execution_service(
    func: Callable,
    *args,
    **kwargs
) -> Any:
    """Call execution service with optimized resilience settings."""
    config = ResilienceConfig(
        service_name="execution",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=2,  # More sensitive for trading
            recovery_timeout=120.0,  # Longer recovery for critical service
            timeout=15.0
        ),
        retry_policy=RetryConfig(
            max_attempts=1,  # No retries for execution to avoid duplicate orders
            initial_delay=0.1,
            max_delay=0.5
        )
    )
    return await call_with_resilience("execution", func, *args, resilience_config=config, **kwargs)


async def call_token_discovery_service(
    func: Callable,
    *args,
    **kwargs
) -> Any:
    """Call token discovery service with tuned resilience settings."""
    config = ResilienceConfig(
        service_name="token-discovery",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=4,
            recovery_timeout=45.0,
            timeout=20.0,
        ),
        retry_policy=RetryConfig(
            max_attempts=3,
            initial_delay=0.75,
            max_delay=6.0,
        ),
    )
    return await call_with_resilience("token-discovery", func, *args, resilience_config=config, **kwargs)
