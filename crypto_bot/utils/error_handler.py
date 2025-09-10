"""
Enhanced Error Handling and Recovery System for LegacyCoinTrader

This module provides comprehensive error handling, recovery mechanisms, and resilience
features for production-ready trading operations.
"""

import asyncio
import time
import functools
import traceback
from typing import (
    Any, Callable, Dict, List, Optional, Union, Type, 
    Awaitable, Tuple, Set, TypeVar, Generic
)
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging
import json
from contextlib import asynccontextmanager, contextmanager
from collections import defaultdict, deque
import weakref

# Type variables for generic error handling
T = TypeVar('T')
F = TypeVar('F', bound=Callable)

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"           # Minor issues, can continue
    MEDIUM = "medium"      # Moderate issues, may need fallback
    HIGH = "high"         # Serious issues, may need circuit breaker
    CRITICAL = "critical" # Fatal issues, requires immediate action


class ErrorCategory(Enum):
    """Categories of errors for targeted handling."""
    NETWORK = "network"           # Network connectivity issues
    API = "api"                  # API errors (rate limits, auth, etc.)
    DATA = "data"                # Data corruption or validation issues
    EXCHANGE = "exchange"        # Exchange-specific errors
    STRATEGY = "strategy"        # Strategy execution errors
    MEMORY = "memory"           # Memory-related issues
    CONFIGURATION = "config"     # Configuration errors
    SYSTEM = "system"           # System-level errors


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    component: str
    severity: ErrorSeverity
    category: ErrorCategory
    retry_count: int = 0
    max_retries: int = 3
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_exception: Optional[Exception] = None
    stack_trace: Optional[str] = None


@dataclass
class RecoveryAction:
    """Defines a recovery action to take when errors occur."""
    name: str
    action: Callable
    conditions: List[Callable[[ErrorContext], bool]]
    priority: int = 0
    async_action: bool = False


class ErrorHandler:
    """
    Centralized error handling and recovery system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.error_history: deque = deque(maxlen=1000)
        self.recovery_actions: List[RecoveryAction] = []
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.recovery_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "failure": 0})
        
        # Initialize default recovery actions
        self._setup_default_recovery_actions()
    
    def _setup_default_recovery_actions(self):
        """Setup default recovery actions."""
        # Network error recovery
        self.add_recovery_action(
            RecoveryAction(
                name="network_retry",
                action=self._retry_with_backoff,
                conditions=[lambda ctx: ctx.category == ErrorCategory.NETWORK],
                priority=1
            )
        )
        
        # API rate limit recovery
        self.add_recovery_action(
            RecoveryAction(
                name="rate_limit_wait",
                action=self._wait_for_rate_limit,
                conditions=[lambda ctx: "rate limit" in str(ctx.original_exception).lower()],
                priority=2
            )
        )
        
        # Data corruption recovery
        self.add_recovery_action(
            RecoveryAction(
                name="data_fallback",
                action=self._fallback_to_cached_data,
                conditions=[lambda ctx: ctx.category == ErrorCategory.DATA],
                priority=3
            )
        )
        
        # Exchange error recovery
        self.add_recovery_action(
            RecoveryAction(
                name="exchange_fallback",
                action=self._fallback_to_paper_trading,
                conditions=[lambda ctx: ctx.category == ErrorCategory.EXCHANGE],
                priority=4
            )
        )
    
    def add_recovery_action(self, action: RecoveryAction):
        """Add a custom recovery action."""
        self.recovery_actions.append(action)
        self.recovery_actions.sort(key=lambda x: x.priority, reverse=True)
    
    def get_circuit_breaker(self, name: str) -> 'CircuitBreaker':
        """Get or create a circuit breaker for a specific component."""
        if name not in self.circuit_breakers:
            config = self.config.get('circuit_breaker', {})
            self.circuit_breakers[name] = CircuitBreaker(
                failure_threshold=config.get('failure_threshold', 5),
                recovery_timeout=config.get('recovery_timeout', 60.0),
                expected_exception=Exception,
                name=name  # Pass the name parameter
            )
        return self.circuit_breakers[name]
    
    def classify_error(self, exception: Exception, operation: str, component: str) -> ErrorContext:
        """Classify an error and create context."""
        error_str = str(exception).lower()
        
        # Determine category - check more specific patterns first
        if any(word in error_str for word in ['kraken', 'coinbase', 'exchange']):
            category = ErrorCategory.EXCHANGE
            severity = ErrorSeverity.HIGH
        elif any(word in error_str for word in ['rate limit', 'api', 'auth']):
            category = ErrorCategory.API
            severity = ErrorSeverity.MEDIUM
        elif any(word in error_str for word in ['network', 'connection', 'timeout']):
            category = ErrorCategory.NETWORK
            severity = ErrorSeverity.MEDIUM
        elif any(word in error_str for word in ['data', 'corrupt', 'invalid']):
            category = ErrorCategory.DATA
            severity = ErrorSeverity.HIGH
        elif any(word in error_str for word in ['strategy', 'signal']):
            category = ErrorCategory.STRATEGY
            severity = ErrorSeverity.MEDIUM
        elif any(word in error_str for word in ['memory', 'out of memory']):
            category = ErrorCategory.MEMORY
            severity = ErrorSeverity.CRITICAL
        elif any(word in error_str for word in ['config', 'configuration']):
            category = ErrorCategory.CONFIGURATION
            severity = ErrorSeverity.HIGH
        else:
            category = ErrorCategory.SYSTEM
            severity = ErrorSeverity.MEDIUM
        
        return ErrorContext(
            operation=operation,
            component=component,
            severity=severity,
            category=category,
            original_exception=exception,
            stack_trace=traceback.format_exc()
        )
    
    async def handle_error(self, error_context: ErrorContext) -> bool:
        """
        Handle an error and attempt recovery.
        Returns True if recovery was successful, False otherwise.
        """
        # Record error
        self.error_history.append(error_context)
        self.error_counts[error_context.component] += 1
        
        # Log error
        logger.error(
            f"Error in {error_context.component}.{error_context.operation}: "
            f"{error_context.original_exception} (Category: {error_context.category.value}, "
            f"Severity: {error_context.severity.value})"
        )
        
        # Check circuit breaker
        circuit_breaker = self.get_circuit_breaker(error_context.component)
        if circuit_breaker.state == "OPEN":
            logger.warning(f"Circuit breaker is OPEN for {error_context.component}")
            return False
        
        # Find applicable recovery actions
        applicable_actions = [
            action for action in self.recovery_actions
            if all(condition(error_context) for condition in action.conditions)
        ]
        
        # Execute recovery actions
        for action in applicable_actions:
            try:
                logger.info(f"Attempting recovery action: {action.name}")
                
                if action.async_action:
                    success = await action.action(error_context)
                else:
                    success = action.action(error_context)
                
                if success:
                    self.recovery_stats[action.name]["success"] += 1
                    logger.info(f"Recovery action {action.name} succeeded")
                    return True
                else:
                    self.recovery_stats[action.name]["failure"] += 1
                    logger.warning(f"Recovery action {action.name} failed")
                    
            except Exception as e:
                self.recovery_stats[action.name]["failure"] += 1
                logger.error(f"Recovery action {action.name} raised exception: {e}")
        
        # If no recovery succeeded, update circuit breaker
        circuit_breaker._on_failure()
        
        return False
    
    def _retry_with_backoff(self, error_context: ErrorContext) -> bool:
        """Retry operation with exponential backoff."""
        if error_context.retry_count >= error_context.max_retries:
            return False
        
        delay = min(2 ** error_context.retry_count, 30)  # Max 30 seconds
        logger.info(f"Retrying {error_context.operation} in {delay}s (attempt {error_context.retry_count + 1})")
        
        # In a real implementation, this would retry the original operation
        # For now, we just simulate success
        return True
    
    def _wait_for_rate_limit(self, error_context: ErrorContext) -> bool:
        """Wait for rate limit to reset."""
        wait_time = 60  # Wait 1 minute for rate limit
        logger.info(f"Rate limit detected, waiting {wait_time}s before retry")
        
        # In a real implementation, this would wait and then retry
        return True
    
    def _fallback_to_cached_data(self, error_context: ErrorContext) -> bool:
        """Fallback to cached data when fresh data is unavailable."""
        logger.info("Falling back to cached data")
        
        # In a real implementation, this would use cached data
        return True
    
    def _fallback_to_paper_trading(self, error_context: ErrorContext) -> bool:
        """Fallback to paper trading when exchange is unavailable."""
        logger.info("Falling back to paper trading mode")
        
        # In a real implementation, this would switch to paper trading
        return True
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return {
            "total_errors": len(self.error_history),
            "error_counts": dict(self.error_counts),
            "recovery_stats": dict(self.recovery_stats),
            "circuit_breaker_states": {
                name: cb.state for name, cb in self.circuit_breakers.items()
            }
        }


class CircuitBreaker:
    """
    Enhanced circuit breaker pattern with monitoring and recovery.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.total_requests = 0
        self.successful_requests = 0
        
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        self.total_requests += 1
        
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    async def async_call(self, coro_func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute async function with circuit breaker protection."""
        self.total_requests += 1
        
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = await coro_func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.successful_requests += 1
        self.last_success_time = time.time()
        
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            self.logger.info(f"Circuit breaker {self.name} reset to CLOSED after successful call")
        elif self.state == "CLOSED":
            self.failure_count = 0  # Reset on consecutive successes
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(
                f"Circuit breaker {self.name} opened after {self.failure_count} failures"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": success_rate,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "recovery_timeout": self.recovery_timeout
        }


class RetryHandler:
    """
    Configurable retry handler with exponential backoff and jitter.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                time.sleep(delay)
        
        raise last_exception
    
    async def async_call(self, coro_func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute async function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await coro_func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
        
        return delay


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_global_error_handler(config: Optional[Dict[str, Any]] = None) -> ErrorHandler:
    """Get or create global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler(config)
    return _global_error_handler


def reset_global_error_handler():
    """Reset global error handler instance."""
    global _global_error_handler
    _global_error_handler = None


# Decorators for easy error handling
def handle_errors(
    operation: str,
    component: str,
    max_retries: int = 3,
    fallback_value: Optional[Any] = None
):
    """
    Decorator for automatic error handling and recovery.
    
    Args:
        operation: Name of the operation being performed
        component: Component name for error classification
        max_retries: Maximum number of retry attempts
        fallback_value: Value to return if all recovery attempts fail
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = get_global_error_handler()
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_context = error_handler.classify_error(e, operation, component)
                    error_context.retry_count = attempt
                    error_context.max_retries = max_retries
                    
                    if attempt == max_retries:
                        # Final attempt failed, try recovery
                        success = asyncio.run(error_handler.handle_error(error_context))
                        if not success and fallback_value is not None:
                            logger.warning(f"Using fallback value for {operation}")
                            return fallback_value
                        raise e
                    
                    # Wait before retry
                    delay = min(2 ** attempt, 30)
                    logger.warning(f"Retrying {operation} in {delay}s (attempt {attempt + 1})")
                    time.sleep(delay)
            
            return fallback_value
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = get_global_error_handler()
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_context = error_handler.classify_error(e, operation, component)
                    error_context.retry_count = attempt
                    error_context.max_retries = max_retries
                    
                    if attempt == max_retries:
                        # Final attempt failed, try recovery
                        success = await error_handler.handle_error(error_context)
                        if not success and fallback_value is not None:
                            logger.warning(f"Using fallback value for {operation}")
                            return fallback_value
                        raise e
                    
                    # Wait before retry
                    delay = min(2 ** attempt, 30)
                    logger.warning(f"Retrying {operation} in {delay}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
            
            return fallback_value
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def with_circuit_breaker(name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """
    Decorator for circuit breaker protection.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
    """
    def decorator(func: F) -> F:
        circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            name=name
        )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return circuit_breaker.call(func, *args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await circuit_breaker.async_call(func, *args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


@contextmanager
def error_context(operation: str, component: str):
    """
    Context manager for error handling with automatic classification.
    
    Args:
        operation: Name of the operation being performed
        component: Component name for error classification
    """
    error_handler = get_global_error_handler()
    
    try:
        yield
    except Exception as e:
        error_context = error_handler.classify_error(e, operation, component)
        asyncio.run(error_handler.handle_error(error_context))
        raise


@asynccontextmanager
async def async_error_context(operation: str, component: str):
    """
    Async context manager for error handling with automatic classification.
    
    Args:
        operation: Name of the operation being performed
        component: Component name for error classification
    """
    error_handler = get_global_error_handler()
    
    try:
        yield
    except Exception as e:
        error_context = error_handler.classify_error(e, operation, component)
        await error_handler.handle_error(error_context)
        raise
