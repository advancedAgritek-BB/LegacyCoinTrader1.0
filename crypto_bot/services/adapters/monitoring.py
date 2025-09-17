"""In-process adapter for monitoring utilities."""

from __future__ import annotations

from crypto_bot.monitoring import record_sol_scanner_metrics
from crypto_bot.services.interfaces import (
    MonitoringService,
    RecordScannerMetricsRequest,
)


class MonitoringAdapter(MonitoringService):
    """Adapter for :mod:`crypto_bot.monitoring`."""

    def record_scanner_metrics(self, request: RecordScannerMetricsRequest) -> None:
        record_sol_scanner_metrics(
            request.tokens,
            request.latency,
            request.config,
        )
