"""Circuit breaker implementation for resilient microservice communication."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 60.0      # Seconds before trying recovery
    expected_exception: tuple = (Exception,)  # Exceptions to count as failures
    success_threshold: int = 3          # Successes needed to close circuit
    timeout: float = 30.0               # Request timeout
    name: str = "default"               # Circuit breaker name


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: int = 0


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker implementation with configurable behavior."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(f"{__name__}.{config.name}")

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with circuit breaker protection."""
        async with self._lock:
            # Check if circuit should transition from OPEN to HALF_OPEN
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    await self._transition_to_half_open()
                else:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker '{self.config.name}' is OPEN"
                    )

            # Check if circuit is HALF_OPEN (limited requests)
            if self.state == CircuitState.HALF_OPEN:
                # In HALF_OPEN, we allow requests but monitor closely
                pass

        try:
            # Execute the function with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                # For sync functions, run in executor with manual timeout
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, func, *args, **kwargs),
                    timeout=self.config.timeout
                )

            await self._record_success()
            return result

        except self.config.expected_exception as e:
            await self._record_failure()
            raise e
        except asyncio.TimeoutError as e:
            await self._record_failure()
            raise e

    async def _record_success(self):
        """Record a successful request."""
        async with self._lock:
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = time.time()

            # Transition from HALF_OPEN to CLOSED if success threshold met
            if (self.state == CircuitState.HALF_OPEN and
                self.metrics.consecutive_successes >= self.config.success_threshold):
                await self._transition_to_closed()

    async def _record_failure(self):
        """Record a failed request."""
        async with self._lock:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = time.time()

            # Transition to OPEN if failure threshold exceeded
            if (self.state == CircuitState.CLOSED and
                self.metrics.consecutive_failures >= self.config.failure_threshold):
                await self._transition_to_open()

            # Transition back to OPEN if HALF_OPEN fails
            elif self.state == CircuitState.HALF_OPEN:
                await self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset from OPEN to HALF_OPEN."""
        if self.metrics.last_failure_time is None:
            return True

        elapsed = time.time() - self.metrics.last_failure_time
        return elapsed >= self.config.recovery_timeout

    async def _transition_to_open(self):
        """Transition circuit to OPEN state."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.metrics.state_changes += 1
        self._logger.warning(
            f"Circuit breaker '{self.config.name}' transitioned from {old_state.value} to {self.state.value} "
            f"(failures: {self.metrics.consecutive_failures}/{self.config.failure_threshold})"
        )

    async def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.metrics.consecutive_successes = 0
        self.metrics.state_changes += 1
        self._logger.info(
            f"Circuit breaker '{self.config.name}' transitioned from {old_state.value} to {self.state.value} "
            "(attempting recovery)"
        )

    async def _transition_to_closed(self):
        """Transition circuit to CLOSED state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.metrics.consecutive_failures = 0
        self.metrics.state_changes += 1
        self._logger.info(
            f"Circuit breaker '{self.config.name}' transitioned from {old_state.value} to {self.state.value} "
            "(recovery successful)"
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "consecutive_failures": self.metrics.consecutive_failures,
            "consecutive_successes": self.metrics.consecutive_successes,
            "failure_rate": (
                self.metrics.failed_requests / self.metrics.total_requests
                if self.metrics.total_requests > 0 else 0
            ),
            "last_failure_time": self.metrics.last_failure_time,
            "last_success_time": self.metrics.last_success_time,
            "state_changes": self.metrics.state_changes,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            }
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        async with self._lock:
            if name not in self._breakers:
                if config is None:
                    config = CircuitBreakerConfig(name=name)
                self._breakers[name] = CircuitBreaker(config)
            return self._breakers[name]

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        return {
            name: breaker.get_metrics()
            for name, breaker in self._breakers.items()
        }

    async def reset_all(self):
        """Reset all circuit breakers to CLOSED state."""
        async with self._lock:
            for breaker in self._breakers.values():
                if breaker.state != CircuitState.CLOSED:
                    await breaker._transition_to_closed()
                    breaker.metrics.consecutive_failures = 0
                    breaker.metrics.consecutive_successes = 0


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


async def call_with_circuit_breaker(
    name: str,
    func: Callable,
    *args,
    config: Optional[CircuitBreakerConfig] = None,
    **kwargs
) -> Any:
    """Call a function with circuit breaker protection."""
    breaker = await circuit_breaker_registry.get_or_create(name, config)
    return await breaker.call(func, *args, **kwargs)
