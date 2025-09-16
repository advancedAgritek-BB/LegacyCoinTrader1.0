"""Common strategy interfaces and helpers."""
from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from types import ModuleType
from typing import Any, Callable, Optional, Protocol, Tuple, runtime_checkable

import pandas as pd

StrategySignal = Tuple[float, str]
PreHook = Callable[..., pd.DataFrame]
PostHook = Callable[..., StrategySignal]


@runtime_checkable
class StrategyProtocol(Protocol):
    """Runtime protocol implemented by all strategy objects."""

    name: str

    def before_generate(
        self, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        """Hook executed prior to generating a signal."""

    def after_generate(
        self, signal: StrategySignal, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> StrategySignal:
        """Hook executed after generating a signal."""

    def generate_signal(
        self, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> StrategySignal:
        """Return the trading signal for ``df``."""

    def __call__(
        self, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> StrategySignal:
        """Allow strategy objects to be used as callables."""


class Strategy(StrategyProtocol, ABC):
    """Base class for all trading strategies."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    def before_generate(
        self, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        """Hook executed prior to :meth:`_generate_signal`.

        Sub-classes can override this to normalise the dataframe or augment
        keyword arguments. The default implementation is a no-op.
        """

        return df

    def after_generate(
        self, signal: StrategySignal, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> StrategySignal:
        """Hook executed after :meth:`_generate_signal`.

        Sub-classes can override this to clamp scores, apply cooldowns or log
        telemetry information. The default implementation is a no-op.
        """

        return signal

    def __call__(
        self, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> StrategySignal:
        return self.generate_signal(df, *args, **kwargs)

    def generate_signal(
        self, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> StrategySignal:
        """Return the strategy signal, applying hooks if present."""

        prepared = self.before_generate(df, *args, **kwargs)
        signal = self._generate_signal(prepared, *args, **kwargs)
        return self.after_generate(signal, prepared, *args, **kwargs)

    @abstractmethod
    def _generate_signal(
        self, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> StrategySignal:
        """Concrete strategy implementation."""


class CallableStrategy(Strategy):
    """Adapter for legacy function-based strategy implementations."""

    def __init__(
        self,
        name: str,
        func: Callable[..., StrategySignal],
        *,
        module: ModuleType | None = None,
        before: PreHook | None = None,
        after: PostHook | None = None,
    ) -> None:
        super().__init__(name=name)
        self._func = func
        self._module = module
        self._before_hook = before
        self._after_hook = after

    def before_generate(
        self, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        if self._before_hook is not None:
            return self._before_hook(df, *args, **kwargs)
        return super().before_generate(df, *args, **kwargs)

    def after_generate(
        self, signal: StrategySignal, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> StrategySignal:
        if self._after_hook is not None:
            return self._after_hook(signal, df, *args, **kwargs)
        return super().after_generate(signal, df, *args, **kwargs)

    def _generate_signal(
        self, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> StrategySignal:
        return self._func(df, *args, **kwargs)

    @property
    def module(self) -> ModuleType | None:
        """Return the underlying module, if one was provided."""

        return self._module

    def __getattr__(self, item: str) -> Any:
        if self._module is not None and hasattr(self._module, item):
            return getattr(self._module, item)
        raise AttributeError(
            f"{self.__class__.__name__} object has no attribute {item!r}"
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"CallableStrategy(name={self.name!r}, func={self._func!r})"


def coerce_to_strategy(obj: Any, *, name: str | None = None) -> StrategyProtocol:
    """Return a :class:`StrategyProtocol` implementation for ``obj``.

    Existing :class:`StrategyProtocol` instances are returned unchanged. Modules
    or objects exposing a ``generate_signal`` callable are wrapped in a
    :class:`CallableStrategy`. ``name`` overrides the inferred strategy name.
    """

    if isinstance(obj, StrategyProtocol):
        return obj

    candidate = getattr(obj, "strategy", None)
    if isinstance(candidate, StrategyProtocol):
        return candidate

    if inspect.isclass(candidate) and issubclass(candidate, Strategy):  # pragma: no cover - defensive
        return candidate()  # type: ignore[call-arg]

    if inspect.isclass(obj) and issubclass(obj, Strategy):  # pragma: no cover - defensive
        return obj()  # type: ignore[call-arg]

    generate_signal = getattr(obj, "generate_signal", None)
    if callable(generate_signal):
        strategy_name = (
            name
            or getattr(obj, "name", None)
            or getattr(obj, "__name__", None)
            or obj.__class__.__name__
        )
        module: ModuleType | None = obj if isinstance(obj, ModuleType) else None
        return CallableStrategy(strategy_name, generate_signal, module=module)

    raise TypeError(f"Object {obj!r} cannot be coerced to a strategy")


__all__ = [
    "Strategy",
    "StrategySignal",
    "StrategyProtocol",
    "CallableStrategy",
    "coerce_to_strategy",
]
