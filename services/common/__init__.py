"""Shared primitives for LegacyCoinTrader microservices."""

from .contracts import (
    EventEnvelope,
    HealthCheckResult,
    ServiceMetadata,
    ServiceRegistration,
)
from .discovery import ServiceDiscoveryClient, ServiceDiscoveryConfig, ServiceDiscoveryError
from .messaging import RedisEventBus, RedisSubscriber

__all__ = [
    "EventEnvelope",
    "HealthCheckResult",
    "RedisEventBus",
    "RedisSubscriber",
    "ServiceDiscoveryClient",
    "ServiceDiscoveryConfig",
    "ServiceDiscoveryError",
    "ServiceMetadata",
    "ServiceRegistration",
]
