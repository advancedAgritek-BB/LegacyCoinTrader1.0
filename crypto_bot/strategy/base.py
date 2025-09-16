"""Common strategy interfaces and helpers.

This module centralises the callable contract used across the various
strategy implementations.  Historically each strategy only exposed a module
level :func:`generate_signal`.  The trading pipeline and selectors rely on
that callable, but different modules gradually added small variations (extra
hooks, wrapper classes, etc.).

To make this contract explicit we now provide :class:`StrategyBase` and a
runtime protocol that downstream consumers can rely on.  Strategies can
subclass :class:`StrategyBase` to take advantage of the shared hooks, while
existing function based implementations can be adapted using
:class:`FunctionStrategy` or :class:`ModuleStrategyAdapter`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any, Callable, Optional, Protocol, Tuple, runtime_checkable

SignalResult = Tuple[float, str]
"""The normalised signal output ``(score, direction)``."""

StrategyCallable = Callable[..., SignalResult]
"""Callable signature every strategy implementation must provide."""


class StrategyBase(ABC):
    """Base class for stateful strategy implementations.

    Sub-classes only need to expose a :pyattr:`generate_signal` property that
    returns a callable following :data:`StrategyCallable`.  The base class
    exposes ``preprocess``/``postprocess`` hooks so strategies can override
    shared behaviour without re-implementing the entire call pipeline.
    """

    name: Optional[str] = None
    """Human readable name for the strategy."""

    regime_filter: Any = None
    """Optional regime filter attached to the strategy."""

    def __init__(self, name: Optional[str] = None, regime_filter: Any = None) -> None:
        if name is not None:
            self.name = name
        elif self.name is None:
            # Fallback to class name if nothing explicit was provided.
            self.name = self.__class__.__name__

        if regime_filter is not None:
            self.regime_filter = regime_filter

    def preprocess(self, market_data: Any, *args: Any, **kwargs: Any) -> Any:
        """Hook executed before :func:`generate_signal`.

        The default implementation simply forwards the received ``market_data``
        as strategies historically expected to operate on pandas DataFrames.
        Implementations can override this hook to perform lightweight
        transformations (e.g. column validation, symbol filtering).
        """

        return market_data

    def postprocess(
        self, result: SignalResult, *args: Any, **kwargs: Any
    ) -> SignalResult:
        """Hook executed after :func:`generate_signal`.

        Returning the result unchanged keeps behaviour identical to the
        historical function based implementations.
        """

        return result

    def signal(self, market_data: Any, *args: Any, **kwargs: Any) -> SignalResult:
        """Evaluate the strategy for ``market_data``.

        ``signal`` mirrors the legacy helper that several pipelines expect.
        The method delegates to :meth:`preprocess`, the callable returned by
        :pyattr:`generate_signal` and finally :meth:`postprocess`.
        """

        processed = self.preprocess(market_data, *args, **kwargs)
        signal_fn = self.generate_signal
        result = signal_fn(processed, *args, **kwargs)
        return self.postprocess(result, *args, **kwargs)

    def __call__(self, market_data: Any, *args: Any, **kwargs: Any) -> SignalResult:
        """Alias to :meth:`signal` so strategies remain directly callable."""

        return self.signal(market_data, *args, **kwargs)

    @property  # type: ignore[misc]
    @abstractmethod
    def generate_signal(self) -> StrategyCallable:
        """Return the underlying signal generator callable."""


@runtime_checkable
class StrategyProtocol(Protocol):
    """Runtime protocol used by loaders to type-check strategies."""

    name: Optional[str]
    regime_filter: Any

    @property
    def generate_signal(self) -> StrategyCallable:  # pragma: no cover - protocol
        ...

    def signal(self, market_data: Any, *args: Any, **kwargs: Any) -> SignalResult:
        ...

    def __call__(self, market_data: Any, *args: Any, **kwargs: Any) -> SignalResult:
        ...


class FunctionStrategy(StrategyBase):
    """Strategy wrapper around a plain callable."""

    def __init__(
        self,
        name: str,
        fn: StrategyCallable,
        regime_filter: Any = None,
    ) -> None:
        super().__init__(name=name, regime_filter=regime_filter)
        self._fn = fn

    @property
    def generate_signal(self) -> StrategyCallable:
        return self._fn


class ModuleStrategyAdapter(FunctionStrategy):
    """Adapter exposing a module level strategy through :class:`StrategyBase`."""

    def __init__(self, name: str, module: ModuleType) -> None:
        try:
            fn = getattr(module, "generate_signal")
        except AttributeError as exc:  # pragma: no cover - defensive
            raise TypeError(
                f"Module {module.__name__!r} does not expose generate_signal"
            ) from exc

        super().__init__(
            name=name,
            fn=fn,
            regime_filter=getattr(module, "regime_filter", None),
        )
        self._module = module

    def __getattr__(self, item: str) -> Any:
        # Delegate attribute access to the underlying module so that existing
        # code accessing helpers (e.g. cached stats, constants) keeps working.
        return getattr(self._module, item)

    def __dir__(self) -> list[str]:  # pragma: no cover - convenience
        return sorted(set(super().__dir__()) | set(dir(self._module)))


def ensure_strategy(name: str, candidate: Any) -> StrategyProtocol:
    """Return a :class:`StrategyProtocol` adapter for ``candidate``.

    ``candidate`` can be a module, a ``Strategy`` instance or any object that
    already implements the protocol.  The helper keeps behaviour backwards
    compatible by falling back to :class:`ModuleStrategyAdapter` when the
    legacy module exposes a ``generate_signal`` function.
    """

    if isinstance(candidate, StrategyProtocol):
        return candidate

    strategy_cls = getattr(candidate, "Strategy", None)
    if callable(strategy_cls):
        try:
            strategy = strategy_cls()
        except Exception:  # pragma: no cover - instantiation failures
            strategy = None
        if isinstance(strategy, StrategyProtocol):
            return strategy

    if isinstance(candidate, ModuleType):
        return ModuleStrategyAdapter(name, candidate)

    fn = getattr(candidate, "generate_signal", None)
    if callable(fn):
        return FunctionStrategy(
            name=name,
            fn=fn,
            regime_filter=getattr(candidate, "regime_filter", None),
        )

    raise TypeError(f"Object {candidate!r} does not implement the strategy interface")
