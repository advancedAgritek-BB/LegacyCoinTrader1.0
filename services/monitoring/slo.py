"""Runtime helpers for tenant SLO tracking and synthetic guard-rail checks."""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Dict, Iterable, List, Optional, Tuple


@dataclass
class SLOSample:
    """Single request observation used for rolling SLO calculations."""

    timestamp: float
    duration: float
    status_code: int
    method: str
    route: str


@dataclass
class TenantSLOSnapshot:
    """Aggregated SLO view for a tenant/service-role pairing."""

    tenant: str
    service_role: str
    window_seconds: int
    request_count: int
    error_count: int
    average_latency_ms: float
    p95_latency_ms: float
    error_rate: float
    throughput_rps: float
    latency_target_ms: float
    latency_slo_met: bool
    error_rate_target: float
    error_rate_slo_met: bool
    throughput_target_rps: float
    throughput_slo_met: bool
    last_updated: float

    def to_dict(self) -> Dict[str, float | int | str | bool]:
        """Return a serialisable representation of the snapshot."""

        return {
            "tenant": self.tenant,
            "service_role": self.service_role,
            "window_seconds": self.window_seconds,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "average_latency_ms": self.average_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "error_rate": self.error_rate,
            "throughput_rps": self.throughput_rps,
            "latency_target_ms": self.latency_target_ms,
            "latency_slo_met": self.latency_slo_met,
            "error_rate_target": self.error_rate_target,
            "error_rate_slo_met": self.error_rate_slo_met,
            "throughput_target_rps": self.throughput_target_rps,
            "throughput_slo_met": self.throughput_slo_met,
            "last_updated": self.last_updated,
        }


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    k = (len(ordered) - 1) * percentile
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ordered[int(k)]
    lower = ordered[f] * (c - k)
    upper = ordered[c] * (k - f)
    return lower + upper


class TenantSLOAggregator:
    """Maintain rolling SLO metrics per tenant and service role."""

    def __init__(
        self,
        *,
        window_seconds: int,
        latency_target_ms: float,
        error_rate_target: float,
        throughput_target_rps: float,
    ) -> None:
        self.window_seconds = window_seconds
        self.latency_target_ms = latency_target_ms
        self.error_rate_target = error_rate_target
        self.throughput_target_rps = throughput_target_rps
        self._samples: Dict[Tuple[str, str], Deque[SLOSample]] = defaultdict(deque)
        self._lock = Lock()

    def _trim(self, samples: Deque[SLOSample], now: float) -> None:
        cutoff = now - self.window_seconds
        while samples and samples[0].timestamp < cutoff:
            samples.popleft()

    def _build_snapshot(
        self,
        key: Tuple[str, str],
        samples: Iterable[SLOSample],
        now: float,
    ) -> TenantSLOSnapshot:
        tenant, service_role = key
        sample_list = list(samples)
        request_count = len(sample_list)
        error_count = sum(1 for sample in sample_list if sample.status_code >= 500)
        latencies = [sample.duration for sample in sample_list]
        average_latency = sum(latencies) / request_count if request_count else 0.0
        p95_latency = _percentile(latencies, 0.95) if request_count else 0.0
        error_rate = error_count / request_count if request_count else 0.0
        throughput = request_count / self.window_seconds if self.window_seconds else 0.0
        last_updated = sample_list[-1].timestamp if sample_list else now
        return TenantSLOSnapshot(
            tenant=tenant,
            service_role=service_role,
            window_seconds=self.window_seconds,
            request_count=request_count,
            error_count=error_count,
            average_latency_ms=average_latency * 1000.0,
            p95_latency_ms=p95_latency * 1000.0,
            error_rate=error_rate,
            throughput_rps=throughput,
            latency_target_ms=self.latency_target_ms,
            latency_slo_met=(p95_latency * 1000.0) <= self.latency_target_ms
            if request_count
            else True,
            error_rate_target=self.error_rate_target,
            error_rate_slo_met=error_rate <= self.error_rate_target if request_count else True,
            throughput_target_rps=self.throughput_target_rps,
            throughput_slo_met=throughput >= self.throughput_target_rps if request_count else True,
            last_updated=last_updated,
        )

    def add_sample(
        self,
        *,
        tenant: str,
        service_role: str,
        method: str,
        route: str,
        status_code: int,
        duration: float,
        timestamp: Optional[float] = None,
    ) -> TenantSLOSnapshot:
        """Record a new request observation and return the latest snapshot."""

        now = timestamp or time.time()
        sample = SLOSample(
            timestamp=now,
            duration=duration,
            status_code=status_code,
            method=method,
            route=route,
        )
        key = (tenant, service_role)
        with self._lock:
            bucket = self._samples[key]
            bucket.append(sample)
            self._trim(bucket, now)
            return self._build_snapshot(key, bucket, now)

    def get_snapshot(
        self, tenant: str, service_role: Optional[str] = None
    ) -> Optional[TenantSLOSnapshot]:
        """Return the latest snapshot for a tenant (optionally scoping by role)."""

        now = time.time()
        with self._lock:
            if service_role is not None:
                key = (tenant, service_role)
                bucket = self._samples.get(key)
                if not bucket:
                    return None
                self._trim(bucket, now)
                if not bucket:
                    return None
                return self._build_snapshot(key, bucket, now)

            merged: List[SLOSample] = []
            for (existing_tenant, existing_role), bucket in self._samples.items():
                if existing_tenant != tenant:
                    continue
                self._trim(bucket, now)
                merged.extend(bucket)
            if not merged:
                return None
            merged.sort(key=lambda sample: sample.timestamp)
            return self._build_snapshot((tenant, "aggregate"), merged, now)

    def get_all_snapshots(self) -> List[TenantSLOSnapshot]:
        """Return snapshots for every tracked tenant/service role."""

        now = time.time()
        snapshots: List[TenantSLOSnapshot] = []
        with self._lock:
            for key, bucket in self._samples.items():
                self._trim(bucket, now)
                if not bucket:
                    continue
                snapshots.append(self._build_snapshot(key, bucket, now))
        snapshots.sort(key=lambda snapshot: (snapshot.tenant, snapshot.service_role))
        return snapshots


@dataclass
class SyntheticCheckStatus:
    """Outcome of a synthetic guard-rail execution."""

    name: str
    tenant: str
    service_role: str
    status: bool
    latency_seconds: float
    checked_at: float
    recovery_time_seconds: Optional[float]
    data_lag_seconds: Optional[float]
    rto_target_seconds: float
    rpo_target_seconds: float

    @property
    def is_rto_compliant(self) -> bool:
        if self.recovery_time_seconds is None:
            return True
        if self.rto_target_seconds <= 0:
            return True
        return self.recovery_time_seconds <= self.rto_target_seconds

    @property
    def is_rpo_compliant(self) -> bool:
        if self.data_lag_seconds is None:
            return True
        if self.rpo_target_seconds <= 0:
            return True
        return self.data_lag_seconds <= self.rpo_target_seconds

    @property
    def overall_ok(self) -> bool:
        return self.status and self.is_rto_compliant and self.is_rpo_compliant

    def to_dict(self) -> Dict[str, float | int | str | bool | None]:
        return {
            "name": self.name,
            "tenant": self.tenant,
            "service_role": self.service_role,
            "status": self.status,
            "latency_seconds": self.latency_seconds,
            "checked_at": self.checked_at,
            "recovery_time_seconds": self.recovery_time_seconds,
            "data_lag_seconds": self.data_lag_seconds,
            "rto_target_seconds": self.rto_target_seconds,
            "rpo_target_seconds": self.rpo_target_seconds,
            "rto_compliant": self.is_rto_compliant,
            "rpo_compliant": self.is_rpo_compliant,
            "overall_compliant": self.overall_ok,
        }


class SyntheticMonitor:
    """Maintain synthetic check health for RTO/RPO monitoring."""

    def __init__(self, *, rto_target_seconds: float, rpo_target_seconds: float) -> None:
        self.rto_target_seconds = rto_target_seconds
        self.rpo_target_seconds = rpo_target_seconds
        self._lock = Lock()
        self._checks: Dict[Tuple[str, str, str], SyntheticCheckStatus] = {}

    def record_check(
        self,
        *,
        name: str,
        tenant: str,
        service_role: str,
        status: bool,
        latency_seconds: float,
        recovery_time_seconds: Optional[float] = None,
        data_lag_seconds: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> SyntheticCheckStatus:
        """Store the latest check result and return the resulting status."""

        checked_at = timestamp or time.time()
        record = SyntheticCheckStatus(
            name=name,
            tenant=tenant,
            service_role=service_role,
            status=status,
            latency_seconds=latency_seconds,
            checked_at=checked_at,
            recovery_time_seconds=recovery_time_seconds,
            data_lag_seconds=data_lag_seconds,
            rto_target_seconds=self.rto_target_seconds,
            rpo_target_seconds=self.rpo_target_seconds,
        )
        with self._lock:
            self._checks[(tenant, service_role, name)] = record
        return record

    def get_check(
        self, name: str, tenant: str, service_role: str
    ) -> Optional[SyntheticCheckStatus]:
        """Return the latest status for the requested synthetic check."""

        with self._lock:
            return self._checks.get((tenant, service_role, name))

    def get_all(self) -> List[SyntheticCheckStatus]:
        """Return all known synthetic check statuses."""

        with self._lock:
            return sorted(
                self._checks.values(),
                key=lambda record: (record.tenant, record.service_role, record.name),
            )


__all__ = [
    "TenantSLOAggregator",
    "TenantSLOSnapshot",
    "SyntheticMonitor",
    "SyntheticCheckStatus",
]
