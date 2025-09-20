"""Dual-write manager for gradual microservice migration."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import time

logger = logging.getLogger(__name__)


@dataclass
class DualWriteConfig:
    """Configuration for dual-write operations."""
    service_name: str
    enabled: bool = True
    primary_system: str = "monolith"  # "monolith" or "microservice"
    secondary_system: str = "microservice"  # "monolith" or "microservice"
    compare_results: bool = True
    fail_on_mismatch: bool = False
    timeout_primary: float = 30.0
    timeout_secondary: float = 30.0
    log_differences: bool = True
    sampling_rate: float = 1.0  # 1.0 = 100% of requests, 0.1 = 10%


@dataclass
class DualWriteMetrics:
    """Metrics for dual-write operations."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    mismatches: int = 0
    primary_failures: int = 0
    secondary_failures: int = 0
    average_response_time_primary: float = 0.0
    average_response_time_secondary: float = 0.0
    last_operation_time: Optional[float] = None


@dataclass
class OperationResult:
    """Result of a dual-write operation."""
    success: bool
    data: Any
    error: Optional[str] = None
    response_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


class DualWriteManager:
    """Manages dual-write operations during microservice migration."""

    def __init__(self, config: DualWriteConfig):
        self.config = config
        self.metrics = DualWriteMetrics()
        self._primary_callable: Optional[Callable] = None
        self._secondary_callable: Optional[Callable] = None
        self._result_comparator: Optional[Callable] = None
        self._logger = logging.getLogger(f"{__name__}.{config.service_name}")

    def set_primary_callable(self, callable: Callable):
        """Set the primary system callable."""
        self._primary_callable = callable

    def set_secondary_callable(self, callable: Callable):
        """Set the secondary system callable."""
        self._secondary_callable = callable

    def set_result_comparator(self, comparator: Callable):
        """Set the result comparison function."""
        self._result_comparator = comparator

    async def execute_dual_write(
        self,
        operation_name: str,
        *args,
        **kwargs
    ) -> OperationResult:
        """Execute operation on both primary and secondary systems."""
        if not self.config.enabled:
            # If dual-write is disabled, just call primary
            return await self._execute_single_operation(
                self._primary_callable, *args, **kwargs
            )

        # Check sampling rate
        if self.config.sampling_rate < 1.0:
            import random
            if random.random() > self.config.sampling_rate:
                # Skip this operation based on sampling rate
                return await self._execute_single_operation(
                    self._primary_callable, *args, **kwargs
                )

        self.metrics.total_operations += 1
        self.metrics.last_operation_time = time.time()

        # Execute operations concurrently
        primary_task = asyncio.create_task(
            self._execute_operation_with_timeout(
                self._primary_callable, self.config.timeout_primary, *args, **kwargs
            )
        )
        secondary_task = asyncio.create_task(
            self._execute_operation_with_timeout(
                self._secondary_callable, self.config.timeout_secondary, *args, **kwargs
            )
        )

        results = await asyncio.gather(primary_task, secondary_task, return_exceptions=True)

        primary_result = results[0]
        secondary_result = results[1]

        # Process results
        success, final_result = await self._process_dual_write_results(
            operation_name, primary_result, secondary_result
        )

        if success:
            self.metrics.successful_operations += 1
        else:
            self.metrics.failed_operations += 1

        return final_result

    async def _execute_operation_with_timeout(
        self,
        callable: Callable,
        timeout: float,
        *args,
        **kwargs
    ) -> OperationResult:
        """Execute operation with timeout."""
        try:
            start_time = time.time()
            if asyncio.iscoroutinefunction(callable):
                result = await asyncio.wait_for(
                    callable(*args, **kwargs), timeout=timeout
                )
            else:
                # Run sync function in thread pool
                import concurrent.futures
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(executor, callable, *args, **kwargs),
                        timeout=timeout
                    )
            response_time = time.time() - start_time

            return OperationResult(
                success=True,
                data=result,
                response_time=response_time
            )

        except Exception as e:
            return OperationResult(
                success=False,
                data=None,
                error=str(e),
                response_time=time.time() - start_time
            )

    async def _execute_single_operation(
        self,
        callable: Callable,
        *args,
        **kwargs
    ) -> OperationResult:
        """Execute single operation (when dual-write is disabled)."""
        result = await self._execute_operation_with_timeout(
            callable, self.config.timeout_primary, *args, **kwargs
        )
        return result

    async def _process_dual_write_results(
        self,
        operation_name: str,
        primary_result: OperationResult,
        secondary_result: OperationResult
    ) -> tuple[bool, OperationResult]:
        """Process results from dual-write operations."""

        # Update metrics
        if not primary_result.success:
            self.metrics.primary_failures += 1
        if not secondary_result.success:
            self.metrics.secondary_failures += 1

        if primary_result.success:
            self.metrics.average_response_time_primary = (
                (self.metrics.average_response_time_primary * (self.metrics.successful_operations - 1) +
                 primary_result.response_time) / self.metrics.successful_operations
            )

        if secondary_result.success:
            self.metrics.average_response_time_secondary = (
                (self.metrics.average_response_time_secondary * (self.metrics.successful_operations - 1) +
                 secondary_result.response_time) / self.metrics.successful_operations
            )

        # Determine which result to return based on configuration
        if self.config.primary_system == "monolith":
            result_to_return = primary_result if primary_result.success else secondary_result
        else:
            result_to_return = secondary_result if secondary_result.success else primary_result

        # Compare results if both succeeded and comparison is enabled
        if (primary_result.success and secondary_result.success and
            self.config.compare_results and self._result_comparator):

            try:
                comparison_result = await self._compare_results(
                    operation_name, primary_result, secondary_result
                )

                if not comparison_result:
                    self.metrics.mismatches += 1

                    if self.config.fail_on_mismatch:
                        self._logger.error(
                            f"Result mismatch in {operation_name}, failing operation"
                        )
                        return False, OperationResult(
                            success=False,
                            data=None,
                            error="Result mismatch between systems"
                        )

            except Exception as e:
                self._logger.warning(f"Error comparing results for {operation_name}: {e}")

        # Log operation summary
        self._log_operation_summary(operation_name, primary_result, secondary_result)

        return result_to_return.success, result_to_return

    async def _compare_results(
        self,
        operation_name: str,
        primary_result: OperationResult,
        secondary_result: OperationResult
    ) -> bool:
        """Compare results from primary and secondary systems."""
        if self._result_comparator:
            return await self._result_comparator(
                primary_result.data, secondary_result.data
            )
        else:
            # Default comparison - exact equality
            return primary_result.data == secondary_result.data

    def _log_operation_summary(
        self,
        operation_name: str,
        primary_result: OperationResult,
        secondary_result: OperationResult
    ):
        """Log summary of dual-write operation."""
        if not self.config.log_differences:
            return

        status = "SUCCESS" if primary_result.success and secondary_result.success else "PARTIAL"
        if not primary_result.success or not secondary_result.success:
            status = "FAILURE"

        self._logger.info(
            f"Dual-write {operation_name}: {status} "
            f"(Primary: {primary_result.response_time:.3f}s, "
            f"Secondary: {secondary_result.response_time:.3f}s)"
        )

        if self.config.log_differences and primary_result.success and secondary_result.success:
            if primary_result.data != secondary_result.data:
                self._logger.warning(
                    f"Result difference in {operation_name}: "
                    f"Primary != Secondary"
                )

    def get_metrics(self) -> Dict[str, Any]:
        """Get dual-write metrics."""
        return {
            "service_name": self.config.service_name,
            "enabled": self.config.enabled,
            "total_operations": self.metrics.total_operations,
            "successful_operations": self.metrics.successful_operations,
            "failed_operations": self.metrics.failed_operations,
            "mismatches": self.metrics.mismatches,
            "primary_failures": self.metrics.primary_failures,
            "secondary_failures": self.metrics.secondary_failures,
            "success_rate": (
                self.metrics.successful_operations / self.metrics.total_operations
                if self.metrics.total_operations > 0 else 0
            ),
            "mismatch_rate": (
                self.metrics.mismatches / self.metrics.total_operations
                if self.metrics.total_operations > 0 else 0
            ),
            "average_response_time_primary": self.metrics.average_response_time_primary,
            "average_response_time_secondary": self.metrics.average_response_time_secondary,
            "last_operation_time": self.metrics.last_operation_time,
        }


class DualWriteRegistry:
    """Registry for managing multiple dual-write managers."""

    def __init__(self):
        self._managers: Dict[str, DualWriteManager] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        service_name: str,
        config: Optional[DualWriteConfig] = None
    ) -> DualWriteManager:
        """Get or create a dual-write manager."""
        async with self._lock:
            if service_name not in self._managers:
                if config is None:
                    config = DualWriteConfig(service_name=service_name)
                self._managers[service_name] = DualWriteManager(config)
            return self._managers[service_name]

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all dual-write managers."""
        return {
            name: manager.get_metrics()
            for name, manager in self._managers.items()
        }

    async def disable_dual_write(self, service_name: str):
        """Disable dual-write for a service."""
        async with self._lock:
            if service_name in self._managers:
                self._managers[service_name].config.enabled = False

    async def enable_dual_write(self, service_name: str):
        """Enable dual-write for a service."""
        async with self._lock:
            if service_name in self._managers:
                self._managers[service_name].config.enabled = True


# Global registry instance
dual_write_registry = DualWriteRegistry()


async def execute_with_dual_write(
    service_name: str,
    operation_name: str,
    primary_callable: Callable,
    secondary_callable: Callable,
    *args,
    dual_write_config: Optional[DualWriteConfig] = None,
    result_comparator: Optional[Callable] = None,
    **kwargs
) -> Any:
    """Execute operation with dual-write support."""
    manager = await dual_write_registry.get_or_create(service_name, dual_write_config)

    # Set callables
    manager.set_primary_callable(primary_callable)
    manager.set_secondary_callable(secondary_callable)

    if result_comparator:
        manager.set_result_comparator(result_comparator)

    result = await manager.execute_dual_write(operation_name, *args, **kwargs)
    return result.data if result.success else None
