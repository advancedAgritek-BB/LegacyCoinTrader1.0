"""Core dataclasses used across the execution service."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional


@dataclass(slots=True)
class SecretRef:
    """Reference to a secret stored in Vault, Kubernetes, or the environment."""

    source: str = "env"
    name: str = ""
    key: Optional[str] = None

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        default_source: str = "env",
    ) -> Optional["SecretRef"]:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            src = str(value.get("source", default_source or "env")).lower()
            name = value.get("name") or value.get("path") or value.get("value")
            if not name:
                return None
            return cls(source=src, name=str(name), key=value.get("key"))
        if isinstance(value, str):
            if default_source == "literal":
                return cls(source="literal", name=value)
            return cls(source=default_source or "env", name=value)
        return cls(source="literal", name=str(value))


@dataclass(slots=True)
class ExchangeCredentials:
    """Resolved exchange credentials."""

    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    ws_token: Optional[str] = None
    api_token: Optional[str] = None


@dataclass(slots=True)
class ExchangeSession:
    """Represents an active exchange session managed by the service."""

    exchange: Any
    ws_client: Optional[Any]
    config_hash: str


@dataclass(slots=True)
class OrderRequest:
    """Order submission payload accepted by the execution service."""

    symbol: str
    side: str
    amount: float
    client_order_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    dry_run: bool = True
    use_websocket: bool = False
    score: float = 0.0
    config: Mapping[str, Any] = field(default_factory=dict)
    notifier: Optional[Any] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrderAck:
    """Acknowledgement message published after accepting an order."""

    client_order_id: str
    accepted: bool
    reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrderFill:
    """Fill event published when the exchange returns a result."""

    client_order_id: str
    success: bool
    order: Optional[Mapping[str, Any]] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Mapping[str, Any] = field(default_factory=dict)
