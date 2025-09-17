"""Utilities for calculating chart scaling and canvas coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class ChartCoordinates:
    """Represents the calculated chart bounds and projected canvas coordinates."""

    min_price: float
    max_price: float
    price_range: float
    entry_y: float
    current_y: float
    stop_loss_y: Optional[float]


def _to_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return None


def _project_to_canvas(value: Optional[float], *, lower: float, span: float, height: int) -> Optional[float]:
    numeric = _to_float(value)
    if numeric is None:
        return None
    normalised = (numeric - lower) / span if span else 0.0
    normalised = max(0.0, min(1.0, normalised))
    return float(height - (normalised * height))


def compute_chart_coordinates(
    entry_price: float,
    current_price: float,
    *,
    stop_loss_price: Optional[float] = None,
    price_points: Sequence[float] | None = None,
    trend_points: Sequence[float] | None = None,
    canvas_height: int = 120,
    include_current_price: bool = True,
    padding_ratio: float = 0.05,
) -> ChartCoordinates:
    """Calculate chart bounds and canvas coordinates.

    The implementation mirrors the JavaScript visualisation logic used by the
    dashboard so we can unit test critical rendering fixes. It ensures that the
    price range never collapses to zero and that additional context such as the
    current price or stop loss is considered when computing chart bounds.
    """

    padding_ratio = max(0.0, padding_ratio)
    candidates: list[float] = []

    for series in (price_points or ()):
        maybe_number = _to_float(series)
        if maybe_number is not None:
            candidates.append(maybe_number)

    for series in (trend_points or ()):
        maybe_number = _to_float(series)
        if maybe_number is not None:
            candidates.append(maybe_number)

    for baseline in (entry_price, stop_loss_price):
        maybe_number = _to_float(baseline)
        if maybe_number is not None:
            candidates.append(maybe_number)

    if include_current_price:
        maybe_number = _to_float(current_price)
        if maybe_number is not None:
            candidates.append(maybe_number)

    if not candidates:
        candidates.append(float(entry_price or 0.0))

    raw_min = min(candidates)
    raw_max = max(candidates)
    raw_span = raw_max - raw_min

    if raw_span <= 0:
        pad = max(abs(raw_max) * max(padding_ratio, 0.001), 0.01)
    else:
        pad = raw_span * padding_ratio

    min_price = max(0.0, raw_min - pad)
    max_price = raw_max + pad
    price_range = max_price - min_price

    if price_range <= 0:
        baseline = max(max_price, 1.0)
        price_range = max(baseline * 0.001, 0.01)
        max_price = min_price + price_range

    entry_y = _project_to_canvas(entry_price, lower=min_price, span=price_range, height=canvas_height)
    current_y = _project_to_canvas(current_price, lower=min_price, span=price_range, height=canvas_height)
    stop_loss_y = _project_to_canvas(stop_loss_price, lower=min_price, span=price_range, height=canvas_height)

    return ChartCoordinates(
        min_price=min_price,
        max_price=max_price,
        price_range=price_range,
        entry_y=entry_y if entry_y is not None else float(canvas_height),
        current_y=current_y if current_y is not None else float(canvas_height),
        stop_loss_y=stop_loss_y,
    )


def calculate_bounds(
    entry_price: float,
    current_price: float,
    *,
    stop_loss_price: Optional[float] = None,
    padding_ratio: float = 0.05,
) -> tuple[float, float]:
    """Convenience helper returning only the chart bounds."""

    coords = compute_chart_coordinates(
        entry_price,
        current_price,
        stop_loss_price=stop_loss_price,
        include_current_price=True,
        padding_ratio=padding_ratio,
    )
    return coords.min_price, coords.max_price


__all__ = ["ChartCoordinates", "compute_chart_coordinates", "calculate_bounds"]
