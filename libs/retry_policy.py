"""Retry policy implementation with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0     # seconds
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    retryable_exceptions: tuple = (Exception,)
    timeout: Optional[float] = None


@dataclass
class RetryMetrics:
    """Metrics for retry operations."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_retry_time: float = 0.0
    average_retry_delay: float = 0.0


class RetryExhaustedException(Exception):
    """Exception raised when all retry attempts are exhausted."""
    def __init__(self, last_exception: Exception, attempts: int):
        self.last_exception = last_exception
        self.attempts = attempts
        super().__init__(f"Retry exhausted after {attempts} attempts: {last_exception}")


class RetryPolicy:
    """Retry policy with exponential backoff and jitter."""

    def __init__(self, config: RetryConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.metrics = RetryMetrics()
        self._logger = logging.getLogger(f"{__name__}.{name}")

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        # Exponential backoff: initial_delay * (backoff_multiplier ^ attempt)
        delay = self.config.initial_delay * (self.config.backoff_multiplier ** attempt)

        # Cap at max_delay
        delay = min(delay, self.config.max_delay)

        # Add jitter if enabled
        if self.config.jitter:
            jitter_range = delay * self.config.jitter_factor
            jitter = random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay + jitter)  # Ensure minimum delay

        return delay

    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute a function with retry policy."""
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                self.metrics.total_attempts += 1

                if self.config.timeout:
                    # Execute with timeout
                    if asyncio.iscoroutinefunction(func):
                        result = await asyncio.wait_for(
                            func(*args, **kwargs),
                            timeout=self.config.timeout
                        )
                    else:
                        # Run sync function in executor
                        loop = asyncio.get_event_loop()
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, func, *args, **kwargs),
                            timeout=self.config.timeout
                        )
                else:
                    # Execute without timeout
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                self.metrics.successful_attempts += 1
                return result

            except self.config.retryable_exceptions as e:
                last_exception = e
                self.metrics.failed_attempts += 1

                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    self.metrics.total_retry_time += delay

                    self._logger.warning(
                        f"Attempt {attempt + 1}/{self.config.max_attempts} failed for '{self.name}': {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    await asyncio.sleep(delay)
                else:
                    self._logger.error(
                        f"All {self.config.max_attempts} attempts failed for '{self.name}': {e}"
                    )

        # All attempts exhausted
        raise RetryExhaustedException(last_exception, self.config.max_attempts)

    def get_metrics(self) -> dict[str, Any]:
        """Get retry metrics."""
        total_attempts = self.metrics.total_attempts
        return {
            "name": self.name,
            "total_attempts": total_attempts,
            "successful_attempts": self.metrics.successful_attempts,
            "failed_attempts": self.metrics.failed_attempts,
            "success_rate": (
                self.metrics.successful_attempts / total_attempts
                if total_attempts > 0 else 0
            ),
            "total_retry_time": self.metrics.total_retry_time,
            "average_retry_delay": (
                self.metrics.total_retry_time / (total_attempts - self.metrics.successful_attempts)
                if total_attempts > self.metrics.successful_attempts else 0
            ),
            "config": {
                "max_attempts": self.config.max_attempts,
                "initial_delay": self.config.initial_delay,
                "max_delay": self.config.max_delay,
                "backoff_multiplier": self.config.backoff_multiplier,
                "jitter": self.config.jitter,
                "timeout": self.config.timeout,
            }
        }


class RetryPolicyRegistry:
    """Registry for managing multiple retry policies."""

    def __init__(self):
        self._policies: dict[str, RetryPolicy] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        config: Optional[RetryConfig] = None
    ) -> RetryPolicy:
        """Get or create a retry policy."""
        async with self._lock:
            if name not in self._policies:
                if config is None:
                    config = RetryConfig()
                self._policies[name] = RetryPolicy(config, name)
            return self._policies[name]

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all retry policies."""
        return {
            name: policy.get_metrics()
            for name, policy in self._policies.items()
        }


# Global registry instance
retry_policy_registry = RetryPolicyRegistry()


async def execute_with_retry(
    name: str,
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Any:
    """Execute a function with retry policy."""
    policy = await retry_policy_registry.get_or_create(name, config)
    return await policy.execute(func, *args, **kwargs)
