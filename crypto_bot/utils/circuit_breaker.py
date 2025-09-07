"""
Circuit Breaker Pattern Implementation

This module provides a robust circuit breaker pattern for handling external API calls
and preventing cascading failures in the trading pipeline.
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Dict, List
from enum import Enum
from collections import defaultdict
import functools

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Failing, reject all calls
    HALF_OPEN = "HALF_OPEN"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    expected_exception: type = Exception
    success_threshold: int = 1  # Number of successful calls to close circuit
    monitor_interval: float = 10.0  # seconds
    enable_metrics: bool = True


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: List[Dict[str, Any]] = field(default_factory=list)
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    current_failure_count: int = 0


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Features:
    - Automatic failure detection and circuit opening
    - Configurable recovery timeouts
    - Half-open state for testing recovery
    - Comprehensive metrics collection
    - Thread-safe operation
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = None
        self._event_loop = None
        self._last_state_change = time.time()
        
        # Track recent failures for more sophisticated failure detection
        self._recent_failures = []
        self._recent_successes = []
        
        logger.info(f"Circuit breaker '{name}' initialized with config: {self.config}")

    def _get_lock(self):
        """Get or create a lock for the current event loop."""
        current_loop = asyncio.get_running_loop()
        if self._lock is None or self._event_loop != current_loop:
            self._lock = asyncio.Lock()
            self._event_loop = current_loop
            logger.debug(f"Created new lock for circuit breaker '{self.name}' in event loop {id(current_loop)}")
        return self._lock

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        async with self._get_lock():
            # Increment total calls
            self.metrics.total_calls += 1
            
            # Check circuit state
            if self.state == CircuitState.OPEN:
                if time.time() - self._last_state_change < self.config.recovery_timeout:
                    self._record_rejected_call()
                    raise Exception(f"Circuit breaker '{self.name}' is OPEN")
                else:
                    # Try to transition to half-open
                    await self._transition_to_half_open()
            
            # Execute function
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = await asyncio.to_thread(func, *args, **kwargs)
                
                await self._record_success()
                return result
                
            except self.config.expected_exception as e:
                await self._record_failure()
                raise e
            except Exception as e:
                # Unexpected exception - still record as failure
                await self._record_failure()
                raise e
    
    async def _record_success(self):
        """Record a successful call."""
        self.metrics.successful_calls += 1
        self.metrics.last_success_time = time.time()
        
        # Update recent successes
        self._recent_successes.append(time.time())
        # Keep only recent successes (last 10 minutes)
        cutoff = time.time() - 600
        self._recent_successes = [t for t in self._recent_successes if t > cutoff]
        
        if self.state == CircuitState.HALF_OPEN:
            if len(self._recent_successes) >= self.config.success_threshold:
                await self._transition_to_closed()
        
        logger.debug(f"Circuit breaker '{self.name}' recorded success")
    
    async def _record_failure(self):
        """Record a failed call."""
        self.metrics.failed_calls += 1
        self.metrics.last_failure_time = time.time()
        self.metrics.current_failure_count += 1
        
        # Update recent failures
        self._recent_failures.append(time.time())
        # Keep only recent failures (last 10 minutes)
        cutoff = time.time() - 600
        self._recent_failures = [t for t in self._recent_failures if t > cutoff]
        
        # Check if we should open the circuit
        if self.state == CircuitState.CLOSED:
            if self.metrics.current_failure_count >= self.config.failure_threshold:
                await self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in HALF_OPEN state should reopen the circuit
            await self._transition_to_open()
        
        logger.warning(f"Circuit breaker '{self.name}' recorded failure (count: {self.metrics.current_failure_count})")
    
    def _record_rejected_call(self):
        """Record a rejected call (circuit open)."""
        self.metrics.rejected_calls += 1
        logger.debug(f"Circuit breaker '{self.name}' rejected call (circuit open)")
    
    async def _transition_to_open(self):
        """Transition circuit to OPEN state."""
        if self.state != CircuitState.OPEN:
            old_state = self.state
            self.state = CircuitState.OPEN
            self._last_state_change = time.time()
            
            self.metrics.state_changes.append({
                'timestamp': time.time(),
                'from_state': old_state.value,
                'to_state': self.state.value,
                'reason': 'failure_threshold_exceeded'
            })
            
            logger.warning(f"Circuit breaker '{self.name}' opened (failures: {self.metrics.current_failure_count})")
    
    async def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state."""
        if self.state == CircuitState.OPEN:
            self.state = CircuitState.HALF_OPEN
            self._last_state_change = time.time()
            self.metrics.current_failure_count = 0
            
            self.metrics.state_changes.append({
                'timestamp': time.time(),
                'from_state': 'OPEN',
                'to_state': self.state.value,
                'reason': 'recovery_timeout_expired'
            })
            
            logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")
    
    async def _transition_to_closed(self):
        """Transition circuit to CLOSED state."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self._last_state_change = time.time()
            self.metrics.current_failure_count = 0
            
            self.metrics.state_changes.append({
                'timestamp': time.time(),
                'from_state': 'HALF_OPEN',
                'to_state': self.state.value,
                'reason': 'success_threshold_met'
            })
            
            logger.info(f"Circuit breaker '{self.name}' closed (recovered)")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics."""
        return {
            'name': self.name,
            'state': self.state.value,
            'total_calls': self.metrics.total_calls,
            'successful_calls': self.metrics.successful_calls,
            'failed_calls': self.metrics.failed_calls,
            'rejected_calls': self.metrics.rejected_calls,
            'current_failure_count': self.metrics.current_failure_count,
            'last_failure_time': self.metrics.last_failure_time,
            'last_success_time': self.metrics.last_success_time,
            'state_changes': self.metrics.state_changes[-10:],  # Last 10 changes
            'recent_failures': len(self._recent_failures),
            'recent_successes': len(self._recent_successes),
            'uptime': time.time() - self._last_state_change
        }
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.metrics.current_failure_count = 0
        self._last_state_change = time.time()
        logger.info(f"Circuit breaker '{self.name}' manually reset")


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers.
    
    Provides centralized management and monitoring of circuit breakers
    across different services and APIs.
    """
    
    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = None
        self._event_loop = None

    def _get_lock(self):
        """Get or create a lock for the current event loop."""
        current_loop = asyncio.get_running_loop()
        if self._lock is None or self._event_loop != current_loop:
            self._lock = asyncio.Lock()
            self._event_loop = current_loop
            logger.debug(f"Created new lock for circuit breaker manager in event loop {id(current_loop)}")
        return self._lock

    async def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker by name."""
        async with self._get_lock():
            if name not in self._circuit_breakers:
                self._circuit_breakers[name] = CircuitBreaker(name, config)
            return self._circuit_breakers[name]
    
    async def call_with_circuit_breaker(
        self, 
        name: str, 
        func: Callable, 
        *args, 
        config: Optional[CircuitBreakerConfig] = None,
        **kwargs
    ) -> Any:
        """Execute function with circuit breaker protection."""
        circuit_breaker = await self.get_circuit_breaker(name, config)
        return await circuit_breaker.call(func, *args, **kwargs)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        return {name: cb.get_metrics() for name, cb in self._circuit_breakers.items()}
    
    async def reset_all(self):
        """Reset all circuit breakers."""
        async with self._get_lock():
            for cb in self._circuit_breakers.values():
                cb.reset()
    
    async def reset_circuit_breaker(self, name: str):
        """Reset a specific circuit breaker."""
        if name in self._circuit_breakers:
            self._circuit_breakers[name].reset()


# Global circuit breaker manager instance
_circuit_breaker_manager = CircuitBreakerManager()


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager instance."""
    return _circuit_breaker_manager


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator for applying circuit breaker pattern to functions.
    
    Args:
        name: Circuit breaker name
        config: Circuit breaker configuration
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            manager = get_circuit_breaker_manager()
            return await manager.call_with_circuit_breaker(name, func, *args, config=config, **kwargs)
        return wrapper
    return decorator


# Predefined circuit breaker configurations for common use cases
EXCHANGE_API_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=120.0,
    expected_exception=Exception,
    success_threshold=2
)

WEBHOOK_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=300.0,
    expected_exception=Exception,
    success_threshold=1
)

DATABASE_CONFIG = CircuitBreakerConfig(
    failure_threshold=2,
    recovery_timeout=60.0,
    expected_exception=Exception,
    success_threshold=3
)
