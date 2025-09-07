"""
Enhanced Error Handling and Recovery System

This module provides comprehensive error handling, recovery mechanisms, and
system resilience features to ensure the trading system remains stable
even under adverse conditions.
"""

import asyncio
import threading
import logging
import traceback
import json
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import functools
import time

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryAction(Enum):
    """Recovery actions that can be taken."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    RESTART = "restart"
    ALERT = "alert"
    IGNORE = "ignore"

@dataclass
class ErrorContext:
    """Context information for an error."""
    component: str
    operation: str
    timestamp: datetime
    error_type: str
    error_message: str
    stack_trace: str
    severity: ErrorSeverity
    recovery_action: RecoveryAction
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryStrategy:
    """Strategy for handling specific error types."""
    error_patterns: List[str]
    severity: ErrorSeverity
    recovery_action: RecoveryAction
    retry_delay: float = 1.0
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    fallback_function: Optional[Callable] = None
    alert_threshold: int = 1

@dataclass
class SystemHealth:
    """System health metrics."""
    total_errors: int = 0
    errors_by_severity: Dict[ErrorSeverity, int] = field(default_factory=dict)
    errors_by_component: Dict[str, int] = field(default_factory=dict)
    recovery_success_rate: float = 0.0
    last_error_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    component_status: Dict[str, str] = field(default_factory=dict)

class EnhancedErrorHandler:
    """
    Enhanced error handling and recovery system.
    
    This system provides:
    - Automatic error classification and severity assessment
    - Configurable recovery strategies
    - Circuit breaker integration
    - Health monitoring and alerting
    - Graceful degradation and fallback mechanisms
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.lock = threading.RLock()
        
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.max_error_history = self.config.get('max_error_history', 1000)
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self._setup_default_strategies()
        
        # System health
        self.system_health = SystemHealth()
        self.start_time = datetime.now()
        
        # Callbacks
        self.error_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # Circuit breaker reference
        self.circuit_breaker = None
        
        # Recovery state
        self.recovery_in_progress = False
        self.recovery_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("Enhanced Error Handler initialized")
    
    def _setup_default_strategies(self):
        """Setup default recovery strategies."""
        default_strategies = [
            # API errors
            RecoveryStrategy(
                error_patterns=["api", "rate_limit"],
                severity=ErrorSeverity.MEDIUM,
                recovery_action=RecoveryAction.RETRY,
                retry_delay=2.0,
                max_retries=5,
                backoff_multiplier=1.5
            ),
            
            # Network errors
            RecoveryStrategy(
                error_patterns=["connection", "network", "socket"],
                severity=ErrorSeverity.HIGH,
                recovery_action=RecoveryAction.RETRY,
                retry_delay=5.0,
                max_retries=3,
                backoff_multiplier=2.0
            ),
            
            # Timeout errors (separate from API)
            RecoveryStrategy(
                error_patterns=["timeout"],
                severity=ErrorSeverity.MEDIUM,
                recovery_action=RecoveryAction.RETRY,
                retry_delay=3.0,
                max_retries=3,
                backoff_multiplier=1.5
            ),
            
            # Data errors
            RecoveryStrategy(
                error_patterns=["data", "parsing", "validation"],
                severity=ErrorSeverity.MEDIUM,
                recovery_action=RecoveryAction.FALLBACK,
                max_retries=1
            ),
            
            # Critical system errors
            RecoveryStrategy(
                error_patterns=["system", "memory", "critical"],
                severity=ErrorSeverity.CRITICAL,
                recovery_action=RecoveryAction.CIRCUIT_BREAK,
                max_retries=0
            ),
            
            # Trading errors
            RecoveryStrategy(
                error_patterns=["trade", "order", "position"],
                severity=ErrorSeverity.HIGH,
                recovery_action=RecoveryAction.ALERT,
                max_retries=1
            )
        ]
        
        for strategy in default_strategies:
            for pattern in strategy.error_patterns:
                self.recovery_strategies[pattern.lower()] = strategy
    
    def register_circuit_breaker(self, circuit_breaker):
        """Register circuit breaker for integration."""
        self.circuit_breaker = circuit_breaker
    
    def add_recovery_strategy(self, name: str, strategy: RecoveryStrategy):
        """Add a custom recovery strategy."""
        self.recovery_strategies[name.lower()] = strategy
    
    def handle_error(self, error: Exception, component: str, operation: str, 
                    context: Dict[str, Any] = None) -> ErrorContext:
        """Handle an error and determine recovery action."""
        with self.lock:
            # Create error context
            error_context = self._create_error_context(error, component, operation, context)
            
            # Determine recovery strategy
            strategy = self._determine_recovery_strategy(error_context)
            error_context.recovery_action = strategy.recovery_action
            error_context.max_retries = strategy.max_retries
            
            # Update system health
            self._update_system_health(error_context)
            
            # Add to history
            self.error_history.append(error_context)
            if len(self.error_history) > self.max_error_history:
                self.error_history.pop(0)
            
            # Execute recovery action (safely handle async context)
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, create task
                asyncio.create_task(self._execute_recovery(error_context, strategy))
            except RuntimeError:
                # No running event loop, schedule for later execution
                # For now, just log the recovery action
                logger.info(f"Recovery action scheduled: {strategy.recovery_action.value} for {component}.{operation}")
            
            # Notify callbacks
            self._notify_error_callbacks(error_context)
            
            logger.error(f"Error in {component}.{operation}: {error}")
            return error_context
    
    def _create_error_context(self, error: Exception, component: str, operation: str,
                             context: Dict[str, Any] = None) -> ErrorContext:
        """Create error context from exception."""
        return ErrorContext(
            component=component,
            operation=operation,
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            severity=ErrorSeverity.MEDIUM,  # Will be updated by strategy
            recovery_action=RecoveryAction.IGNORE,  # Will be updated by strategy
            metadata=context or {}
        )
    
    def _determine_recovery_strategy(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Determine the appropriate recovery strategy for an error."""
        error_text = f"{error_context.error_type} {error_context.error_message}".lower()
        
        # Find matching strategy
        for pattern, strategy in self.recovery_strategies.items():
            if pattern in error_text:
                error_context.severity = strategy.severity
                return strategy
        
        # Default strategy for unknown errors
        return RecoveryStrategy(
            error_patterns=["unknown"],
            severity=ErrorSeverity.MEDIUM,
            recovery_action=RecoveryAction.ALERT,
            max_retries=1
        )
    
    def _update_system_health(self, error_context: ErrorContext):
        """Update system health metrics."""
        self.system_health.total_errors += 1
        self.system_health.last_error_time = error_context.timestamp
        
        # Update severity counts
        severity = error_context.severity
        self.system_health.errors_by_severity[severity] = \
            self.system_health.errors_by_severity.get(severity, 0) + 1
        
        # Update component counts
        component = error_context.component
        self.system_health.errors_by_component[component] = \
            self.system_health.errors_by_component.get(component, 0) + 1
        
        # Update uptime
        self.system_health.uptime_seconds = \
            (datetime.now() - self.start_time).total_seconds()
    
    async def _execute_recovery(self, error_context: ErrorContext, strategy: RecoveryStrategy):
        """Execute the recovery action."""
        try:
            if strategy.recovery_action == RecoveryAction.RETRY:
                await self._retry_operation(error_context, strategy)
            elif strategy.recovery_action == RecoveryAction.FALLBACK:
                await self._execute_fallback(error_context, strategy)
            elif strategy.recovery_action == RecoveryAction.CIRCUIT_BREAK:
                await self._trigger_circuit_break(error_context)
            elif strategy.recovery_action == RecoveryAction.ALERT:
                await self._send_alert(error_context)
            elif strategy.recovery_action == RecoveryAction.RESTART:
                await self._restart_component(error_context)
            
            # Notify recovery callbacks
            self._notify_recovery_callbacks(error_context, True)
            
        except Exception as recovery_error:
            logger.error(f"Recovery failed for {error_context.component}.{error_context.operation}: {recovery_error}")
            self._notify_recovery_callbacks(error_context, False)
    
    async def _retry_operation(self, error_context: ErrorContext, strategy: RecoveryStrategy):
        """Retry the failed operation with exponential backoff."""
        if error_context.retry_count >= strategy.max_retries:
            logger.warning(f"Max retries exceeded for {error_context.component}.{error_context.operation}")
            return
        
        delay = strategy.retry_delay * (strategy.backoff_multiplier ** error_context.retry_count)
        await asyncio.sleep(delay)
        
        error_context.retry_count += 1
        logger.info(f"Retrying {error_context.component}.{error_context.operation} (attempt {error_context.retry_count})")
        
        # Here you would re-execute the original operation
        # This is a placeholder - actual implementation would depend on the specific operation
    
    async def _execute_fallback(self, error_context: ErrorContext, strategy: RecoveryStrategy):
        """Execute fallback function if available."""
        if strategy.fallback_function:
            try:
                await strategy.fallback_function(error_context)
                logger.info(f"Fallback executed for {error_context.component}.{error_context.operation}")
            except Exception as fallback_error:
                logger.error(f"Fallback failed: {fallback_error}")
        else:
            logger.warning(f"No fallback function available for {error_context.component}.{error_context.operation}")
    
    async def _trigger_circuit_break(self, error_context: ErrorContext):
        """Trigger circuit breaker."""
        if self.circuit_breaker:
            try:
                # Record the error in circuit breaker
                self.circuit_breaker.record_api_error()
                logger.warning(f"Circuit breaker triggered for {error_context.component}.{error_context.operation}")
            except Exception as cb_error:
                logger.error(f"Failed to trigger circuit breaker: {cb_error}")
        else:
            logger.warning("Circuit breaker not available")
    
    async def _send_alert(self, error_context: ErrorContext):
        """Send alert for the error."""
        alert_message = {
            'type': 'error_alert',
            'component': error_context.component,
            'operation': error_context.operation,
            'error_type': error_context.error_type,
            'error_message': error_context.error_message,
            'severity': error_context.severity.value,
            'timestamp': error_context.timestamp.isoformat()
        }
        
        for callback in self.alert_callbacks:
            try:
                await callback(alert_message)
            except Exception as alert_error:
                logger.error(f"Alert callback failed: {alert_error}")
        
        logger.warning(f"Alert sent for {error_context.component}.{error_context.operation}")
    
    async def _restart_component(self, error_context: ErrorContext):
        """Restart the component that encountered the error."""
        logger.warning(f"Restarting component {error_context.component}")
        # This would integrate with component management system
        # For now, just log the action
    
    def _notify_error_callbacks(self, error_context: ErrorContext):
        """Notify error callbacks."""
        for callback in self.error_callbacks:
            try:
                callback(error_context)
            except Exception as callback_error:
                logger.error(f"Error callback failed: {callback_error}")
    
    def _notify_recovery_callbacks(self, error_context: ErrorContext, success: bool):
        """Notify recovery callbacks."""
        for callback in self.recovery_callbacks:
            try:
                callback(error_context, success)
            except Exception as callback_error:
                logger.error(f"Recovery callback failed: {callback_error}")
    
    def add_error_callback(self, callback: Callable):
        """Add error callback."""
        self.error_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable):
        """Add recovery callback."""
        self.recovery_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health."""
        with self.lock:
            return self.system_health
    
    def get_error_history(self, component: str = None, severity: ErrorSeverity = None,
                         since: datetime = None) -> List[ErrorContext]:
        """Get filtered error history."""
        with self.lock:
            filtered = self.error_history
            
            if component:
                filtered = [e for e in filtered if e.component == component]
            
            if severity:
                filtered = [e for e in filtered if e.severity == severity]
            
            if since:
                filtered = [e for e in filtered if e.timestamp >= since]
            
            return filtered.copy()
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self.lock:
            # Convert enum values to strings for JSON serialization
            errors_by_severity = {}
            for severity, count in self.system_health.errors_by_severity.items():
                errors_by_severity[severity.value] = count
            
            return {
                'total_errors': self.system_health.total_errors,
                'errors_by_severity': errors_by_severity,
                'errors_by_component': dict(self.system_health.errors_by_component),
                'recovery_success_rate': self.system_health.recovery_success_rate,
                'uptime_seconds': self.system_health.uptime_seconds,
                'last_error_time': self.system_health.last_error_time.isoformat() if self.system_health.last_error_time else None
            }
    
    def clear_error_history(self):
        """Clear error history."""
        with self.lock:
            self.error_history.clear()
            logger.info("Error history cleared")
    
    def reset_system_health(self):
        """Reset system health metrics."""
        with self.lock:
            self.system_health = SystemHealth()
            self.start_time = datetime.now()
            logger.info("System health reset")

def error_handler(component: str, operation: str = None, 
                 recovery_strategy: RecoveryStrategy = None):
    """
    Decorator for automatic error handling.
    
    Usage:
    @error_handler("api_client", "fetch_data")
    async def fetch_data():
        # Your function here
        pass
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Get error handler instance (you'd need to make this available globally)
            error_handler_instance = getattr(func, '_error_handler', None)
            
            if error_handler_instance:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    op_name = operation or func.__name__
                    error_context = error_handler_instance.handle_error(
                        e, component, op_name, {'args': args, 'kwargs': kwargs}
                    )
                    raise  # Re-raise the exception after handling
            else:
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def retry_on_error(max_retries: int = 3, delay: float = 1.0, 
                   backoff_multiplier: float = 2.0,
                   exceptions: tuple = (Exception,)):
    """
    Decorator for automatic retry on specific exceptions.
    
    Usage:
    @retry_on_error(max_retries=5, delay=2.0)
    async def unreliable_function():
        # Your function here
        pass
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (backoff_multiplier ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed: {e}")
                        raise last_exception
            
            raise last_exception
        
        return wrapper
    return decorator

def circuit_breaker_protected(circuit_breaker):
    """
    Decorator for circuit breaker protection.
    
    Usage:
    @circuit_breaker_protected(my_circuit_breaker)
    async def protected_function():
        # Your function here
        pass
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if circuit_breaker and not circuit_breaker.is_trading_allowed():
                raise Exception("Circuit breaker is open")
            
            try:
                result = await func(*args, **kwargs)
                # Record success
                if circuit_breaker:
                    circuit_breaker.record_api_request()
                return result
            except Exception as e:
                # Record failure
                if circuit_breaker:
                    circuit_breaker.record_api_error()
                raise
        
        return wrapper
    return decorator
