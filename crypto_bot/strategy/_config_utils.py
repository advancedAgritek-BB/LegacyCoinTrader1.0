"""Helpers for building structured strategy configurations."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Iterable, Mapping, MutableMapping, Type, TypeVar


T = TypeVar("T")


def _coerce_mapping(data: object) -> Mapping[str, Any]:
    """Return ``data`` if it behaves like a mapping, otherwise empty mapping."""

    if isinstance(data, Mapping):
        return data
    return {}


def extract_params(
    data: object,
    field_names: Iterable[str],
    section_names: Iterable[str],
) -> dict[str, Any]:
    """Return merged configuration parameters for a strategy.

    ``data`` may be either the entire bot configuration or the strategy specific
    section. Only keys present in ``field_names`` are copied. ``section_names``
    defines additional nested keys that should be merged on top of the base
    mapping when present.
    """

    base: Mapping[str, Any] = _coerce_mapping(data)
    params: dict[str, Any] = {}
    for key in field_names:
        if key in base:
            params[key] = base[key]
    for section in section_names:
        nested = base.get(section)
        nested_map = _coerce_mapping(nested)
        if nested_map:
            for key in field_names:
                if key in nested_map:
                    params[key] = nested_map[key]
    return params


def apply_defaults(cls: Type[T], params: Mapping[str, Any]) -> T:
    """Instantiate ``cls`` using values from ``params`` with defaults applied."""

    kwargs: MutableMapping[str, Any] = {}
    for field in fields(cls):
        if field.name in params:
            kwargs[field.name] = params[field.name]
    return cls(**kwargs)  # type: ignore[arg-type]
