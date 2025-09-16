"""Core strategy abstractions.

This module provides a lightweight object oriented interface for strategy
implementations.  Historically each strategy in :mod:`crypto_bot.strategy`
exposed a module level :func:`generate_signal` function.  While convenient, the
lack of a shared protocol made it difficult to attach metadata, provide common
pre/post hooks or reason about strategies in a generic way.  The classes below
establish a minimal contract that the rest of the codebase can rely on while
remaining backwards compatible with the existing function based strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple, runtime_checkable

import pandas as pd

Signal = Tuple[float, str]
HookState = Any


@runtime_checkable
class StrategyProtocol(Protocol):
    """Protocol implemented by all strategies.

    The protocol intentionally mirrors :class:`BaseStrategy` so that both
    concrete subclasses and thin wrapper objects created from legacy callables
    can be treated uniformly throughout the codebase.  ``__call__`` is included
    to preserve compatibility with places that store strategies as generic
    callables.
    """

    __name__: str

    def generate_signal(  # pragma: no cover - runtime structural typing
        self,
        df: pd.DataFrame,
        config: Optional[Mapping[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Signal:
        """Return a trading signal for ``df``."""

    def __call__(  # pragma: no cover - runtime structural typing
        self,
        df: pd.DataFrame,
        config: Optional[Mapping[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Signal:
        """Alias for :meth:`generate_signal` used by existing pipelines."""


class BaseStrategy(ABC):
    """Base class for strategy implementations.

    Subclasses only need to implement :meth:`generate_signal`.  ``before`` and
    ``after`` hooks are provided for future extensibility (for example logging
    or telemetry) without forcing every strategy to implement them today.
    """

    name: str

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or getattr(self, "name", self.__class__.__name__)
        # Provide ``__name__`` so strategies behave like callables in logs.
        self.__name__ = self.name

    def before_generate(
        self,
        df: pd.DataFrame,
        config: Optional[Mapping[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> HookState:
        """Hook executed before :meth:`generate_signal`.

        The default implementation returns ``None`` and performs no action.  A
        subclass can return arbitrary context which will be passed to
        :meth:`after_generate`.
        """

        return None

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        config: Optional[Mapping[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Signal:
        """Return a trading signal."""

    def after_generate(
        self,
        result: Signal,
        hook_state: HookState,
        df: pd.DataFrame,
        config: Optional[Mapping[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Signal:
        """Hook executed after :meth:`generate_signal`.

        By default the original ``result`` is returned unchanged.
        """

        return result

    def supports_regime(self, regime: str) -> bool:
        """Return ``True`` when the strategy can operate under ``regime``.

        Many strategies expose a ``regime_filter`` helper with a ``matches``
        method.  The default implementation introspects for that helper and
        delegates when available.
        """

        reg_filter = getattr(self, "regime_filter", None)
        matches = getattr(reg_filter, "matches", None)
        if callable(matches):
            try:  # pragma: no cover - defensive programming
                return bool(matches(regime))
            except Exception:
                return False
        return True

    def __call__(
        self,
        df: pd.DataFrame,
        config: Optional[Mapping[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Signal:
        hook_state = self.before_generate(df, config=config, *args, **kwargs)
        result = self.generate_signal(df, config=config, *args, **kwargs)
        return self.after_generate(result, hook_state, df, config=config, *args, **kwargs)


class FunctionStrategy(BaseStrategy):
    """Adapter turning an arbitrary callable into a :class:`BaseStrategy`."""

    def __init__(
        self,
        func: Callable[..., Signal],
        *,
        name: Optional[str] = None,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(name or getattr(func, "__name__", None))
        self._func = func
        self.source = func
        if extras:
            for key, value in extras.items():
                setattr(self, key, value)

    def generate_signal(
        self,
        df: pd.DataFrame,
        config: Optional[Mapping[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Signal:
        if config is not None:
            try:
                return self._func(df, config, *args, **kwargs)
            except TypeError:
                extra_kwargs = dict(kwargs)
                extra_kwargs.setdefault("config", config)
                return self._func(df, *args, **extra_kwargs)
        return self._func(df, *args, **kwargs)


def as_strategy(
    candidate: Any,
    *,
    name: Optional[str] = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> StrategyProtocol:
    """Coerce ``candidate`` into a :class:`StrategyProtocol` instance.

    Parameters
    ----------
    candidate:
        Module, class, instance or callable implementing ``generate_signal``.
    name:
        Optional explicit strategy name.  When omitted the function or class
        name is used.
    extras:
        Optional attributes copied onto the resulting strategy instance.
    """

    extras_dict: Dict[str, Any] = {
        key: value for key, value in (extras or {}).items() if value is not None
    }

    strategy: StrategyProtocol
    if isinstance(candidate, BaseStrategy):
        strategy = candidate
        if name:
            strategy.name = name
            strategy.__name__ = name
    elif isinstance(candidate, type) and issubclass(candidate, BaseStrategy):
        strategy = candidate()  # type: ignore[call-arg]
        if name:
            strategy.name = name
            strategy.__name__ = name
    elif hasattr(candidate, "generate_signal") and callable(candidate.generate_signal):
        inferred = name or getattr(candidate, "name", None) or getattr(
            candidate, "NAME", None
        )
        strategy = FunctionStrategy(
            candidate.generate_signal,
            name=inferred,
            extras=extras_dict,
        )
        strategy.source = candidate
        extras_dict = {}
    elif callable(candidate):
        strategy = FunctionStrategy(candidate, name=name, extras=extras_dict)
        extras_dict = {}
    else:
        raise TypeError(f"Object {candidate!r} cannot be adapted to a strategy")

    for key, value in extras_dict.items():
        setattr(strategy, key, value)

    module_name = getattr(candidate, "__module__", None)
    if module_name is None:
        module_name = getattr(candidate, "__name__", None)
    if module_name:
        setattr(strategy, "__module__", module_name)

    if not getattr(strategy, "name", None):
        setattr(strategy, "name", getattr(strategy, "__name__", strategy.__class__.__name__))
    strategy.__name__ = getattr(strategy, "name", strategy.__class__.__name__)
    return strategy


__all__ = [
    "BaseStrategy",
    "FunctionStrategy",
    "StrategyProtocol",
    "as_strategy",
]
