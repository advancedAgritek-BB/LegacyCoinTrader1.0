"""
Enhanced Error Handling and Retry Logic

This module provides sophisticated error handling and retry mechanisms
for the trading pipeline, including exponential backoff, jitter, and
context-aware retry strategies.
"""

import asyncio
import time
import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Dict, List, Type, Union
from enum import Enum
import functools
import traceback

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Different retry strategies for different types of failures."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    CONSTANT_DELAY = "constant_delay"
    IMMEDIATE_RETRY = "immediate_retry"


class ErrorSeverity(Enum):
    """Error severity levels for different types of failures."""
    LOW = "low"           # Temporary issues, safe to retry
    MEDIUM = "medium"      # Moderate issues, retry with caution
    HIGH = "high"          # Serious issues, limited retries
    CRITICAL = "critical" # Critical issues, no retries


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter_factor: float = 0.1  # 10% jitter
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retry_on_exceptions: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    retry_on_conditions: List[Callable[[Any], bool]] = field(default_factory=list)
    backoff_multiplier: float = 1.0
    timeout: Optional[float] = None
    enable_logging: bool = True


@dataclass
class RetryMetrics:
    """Metrics for retry operations."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    retry_count: int = 0
    total_time: float = 0.0
    last_attempt_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    error_history: List[Dict[str, Any]] = field(default_factory=list)


class RetryableError(Exception):
    """Base exception for retryable errors."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None, 
                 retry_count: int = 0, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        super().__init__(message)
        self.original_exception = original_exception
        self.retry_count = retry_count
        self.severity = severity


class NonRetryableError(Exception):
    """Exception for non-retryable errors."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None,
                 severity: ErrorSeverity = ErrorSeverity.CRITICAL):
        super().__init__(message)
        self.original_exception = original_exception
        self.severity = severity


class RetryHandler:
    """
    Enhanced retry handler with sophisticated error handling.
    
    Features:
    - Multiple retry strategies (exponential backoff, linear, constant)
    - Jitter to prevent thundering herd
    - Context-aware retry decisions
    - Comprehensive metrics collection
    - Exception classification and handling
    """
    
    def __init__(self, name: str, config: Optional[RetryConfig] = None):
        self.name = name
        self.config = config or RetryConfig()
        self.metrics = RetryMetrics()
        self._lock = None
        self._event_loop = None

        logger.info(f"Retry handler '{name}' initialized with config: {self.config}")

    def _get_lock(self):
        """Get or create a lock for the current event loop."""
        current_loop = asyncio.get_running_loop()
        if self._lock is None or self._event_loop != current_loop:
            self._lock = asyncio.Lock()
            self._event_loop = current_loop
            logger.debug(f"Created new lock for retry handler '{self.name}' in event loop {id(current_loop)}")
        return self._lock

    async def execute_with_retry(
        self, 
        func: Callable, 
        *args, 
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with enhanced retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            context: Additional context for retry decisions
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RetryableError: If all retries are exhausted
            NonRetryableError: If error is non-retryable
        """
        context = context or {}
        start_time = time.time()
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            self.metrics.total_attempts += 1
            self.metrics.last_attempt_time = time.time()
            
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(*args, **kwargs), 
                        timeout=self.config.timeout
                    ) if self.config.timeout else await func(*args, **kwargs)
                else:
                    if self.config.timeout:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(func, *args, **kwargs),
                            timeout=self.config.timeout
                        )
                    else:
                        result = await asyncio.to_thread(func, *args, **kwargs)
                
                # Success
                self.metrics.successful_attempts += 1
                self.metrics.last_success_time = time.time()
                self.metrics.total_time = time.time() - start_time
                
                if self.config.enable_logging:
                    logger.info(f"Retry handler '{self.name}' succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                self.metrics.failed_attempts += 1
                self.metrics.last_failure_time = time.time()
                
                # Check if this is a retryable error
                if not self._should_retry(e, attempt, context):
                    self.metrics.total_time = time.time() - start_time
                    raise NonRetryableError(
                        f"Non-retryable error in '{self.name}': {str(e)}",
                        original_exception=e,
                        severity=self._classify_error_severity(e)
                    )
                
                # Check if we should stop retrying
                if attempt >= self.config.max_retries:
                    self.metrics.total_time = time.time() - start_time
                    raise RetryableError(
                        f"All retries exhausted for '{self.name}' after {attempt + 1} attempts",
                        original_exception=e,
                        retry_count=attempt,
                        severity=self._classify_error_severity(e)
                    )
                
                # Log retry attempt
                if self.config.enable_logging:
                    logger.warning(
                        f"Retry handler '{self.name}' attempt {attempt + 1} failed: {str(e)}"
                    )
                
                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                if self.config.enable_logging:
                    logger.info(f"Retry handler '{self.name}' waiting {delay:.2f}s before retry")
                
                await asyncio.sleep(delay)
                self.metrics.retry_count += 1
    
    def _should_retry(self, exception: Exception, attempt: int, context: Dict[str, Any]) -> bool:
        """Determine if the error should be retried."""
        # Check if exception type is in retry list
        if not any(isinstance(exception, exc_type) for exc_type in self.config.retry_on_exceptions):
            return False
        
        # Check custom retry conditions
        for condition in self.config.retry_on_conditions:
            if not condition(exception):
                return False
        
        # Check context-specific conditions
        if context.get("force_no_retry", False):
            return False
        
        # Check error severity
        severity = self._classify_error_severity(exception)
        if severity == ErrorSeverity.CRITICAL:
            return False
        
        return True
    
    def _classify_error_severity(self, exception: Exception) -> ErrorSeverity:
        """Classify error severity based on exception type and message."""
        error_str = str(exception).lower()
        
        # Authentication errors are usually not retryable
        if any(keyword in error_str for keyword in ["auth", "unauthorized", "forbidden", "invalid key", "api key"]):
            return ErrorSeverity.CRITICAL
        
        # Network-related errors are usually retryable
        if any(keyword in error_str for keyword in ["timeout", "connection", "network", "rate limit"]):
            return ErrorSeverity.LOW
        
        # Validation errors are usually not retryable
        if any(keyword in error_str for keyword in ["validation", "invalid", "bad request"]):
            return ErrorSeverity.HIGH
        
        # Server errors are usually retryable
        if any(keyword in error_str for keyword in ["server", "internal", "500", "502", "503", "504"]):
            return ErrorSeverity.MEDIUM
        
        # Default to medium severity
        return ErrorSeverity.MEDIUM
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on retry strategy."""
        if self.config.strategy == RetryStrategy.IMMEDIATE_RETRY:
            return 0.0
        
        elif self.config.strategy == RetryStrategy.CONSTANT_DELAY:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * (attempt + 1) * self.config.backoff_multiplier
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.exponential_base ** attempt) * self.config.backoff_multiplier
        
        else:
            delay = self.config.base_delay
        
        # Apply jitter
        if self.config.jitter_factor > 0:
            jitter = random.uniform(-self.config.jitter_factor, self.config.jitter_factor)
            delay *= (1 + jitter)
        
        # Cap at max delay
        return min(delay, self.config.max_delay)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current retry metrics."""
        return {
            'name': self.name,
            'total_attempts': self.metrics.total_attempts,
            'successful_attempts': self.metrics.successful_attempts,
            'failed_attempts': self.metrics.failed_attempts,
            'retry_count': self.metrics.retry_count,
            'success_rate': (self.metrics.successful_attempts / self.metrics.total_attempts) if self.metrics.total_attempts > 0 else 0.0,
            'total_time': self.metrics.total_time,
            'last_attempt_time': self.metrics.last_attempt_time,
            'last_success_time': self.metrics.last_success_time,
            'last_failure_time': self.metrics.last_failure_time,
            'error_history': self.metrics.error_history[-10:]  # Last 10 errors
        }
    
    def reset_metrics(self):
        """Reset retry metrics."""
        self.metrics = RetryMetrics()
        logger.info(f"Retry handler '{self.name}' metrics reset")


class RetryManager:
    """
    Manager for multiple retry handlers.
    
    Provides centralized management and monitoring of retry handlers
    across different operations in the trading pipeline.
    """
    
    def __init__(self):
        self._retry_handlers: Dict[str, RetryHandler] = {}
        self._lock = None
        self._event_loop = None

    def _get_lock(self):
        """Get or create a lock for the current event loop."""
        current_loop = asyncio.get_running_loop()
        if self._lock is None or self._event_loop != current_loop:
            self._lock = asyncio.Lock()
            self._event_loop = current_loop
            logger.debug(f"Created new lock for retry manager in event loop {id(current_loop)}")
        return self._lock

    async def get_retry_handler(self, name: str, config: Optional[RetryConfig] = None) -> RetryHandler:
        """Get or create a retry handler by name."""
        async with self._get_lock():
            if name not in self._retry_handlers:
                self._retry_handlers[name] = RetryHandler(name, config)
            return self._retry_handlers[name]
    
    async def execute_with_retry(
        self, 
        name: str, 
        func: Callable, 
        *args, 
        config: Optional[RetryConfig] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Execute function with retry protection."""
        retry_handler = await self.get_retry_handler(name, config)
        return await retry_handler.execute_with_retry(func, *args, context=context, **kwargs)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all retry handlers."""
        return {name: handler.get_metrics() for name, handler in self._retry_handlers.items()}
    
    async def reset_all_metrics(self):
        """Reset metrics for all retry handlers."""
        async with self._get_lock():
            for handler in self._retry_handlers.values():
                handler.reset_metrics()
    
    async def reset_handler_metrics(self, name: str):
        """Reset metrics for a specific retry handler."""
        if name in self._retry_handlers:
            self._retry_handlers[name].reset_metrics()


# Global retry manager instance
_retry_manager = RetryManager()


def get_retry_manager() -> RetryManager:
    """Get the global retry manager instance."""
    return _retry_manager


def retry_handler(name: str, config: Optional[RetryConfig] = None):
    """
    Decorator for applying retry logic to functions.
    
    Args:
        name: Retry handler name
        config: Retry configuration
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            manager = get_retry_manager()
            return await manager.execute_with_retry(name, func, *args, config=config, **kwargs)
        return wrapper
    return decorator


# Predefined retry configurations for common use cases
EXCHANGE_API_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    retry_on_exceptions=[Exception],
    timeout=30.0
)

OHLCV_FETCH_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    base_delay=2.0,
    max_delay=60.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    retry_on_exceptions=[Exception],
    timeout=60.0
)

ORDER_PLACEMENT_RETRY_CONFIG = RetryConfig(
    max_retries=2,
    base_delay=0.5,
    max_delay=10.0,
    strategy=RetryStrategy.LINEAR_BACKOFF,
    retry_on_exceptions=[Exception],
    timeout=15.0
)

WEBHOOK_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=5.0,
    max_delay=300.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    retry_on_exceptions=[Exception],
    timeout=30.0
)
