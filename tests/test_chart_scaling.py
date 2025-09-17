"""Unit tests derived from the historical JavaScript chart validation scripts."""

from frontend.chart_scaling import compute_chart_coordinates


def test_long_position_coordinates_are_ordered():
    prices = [49500, 49800, 50200, 50500, 50800, 51000]
    trend = [49600, 49900, 50200, 50500, 50800, 51100]

    coords = compute_chart_coordinates(
        entry_price=50000,
        current_price=51000,
        stop_loss_price=49000,
        price_points=prices,
        trend_points=trend,
    )

    assert 0.0 <= coords.entry_y <= 120.0
    assert 0.0 <= coords.current_y <= 120.0
    assert coords.current_y < coords.entry_y
    assert coords.stop_loss_y is not None and 0.0 <= coords.stop_loss_y <= 120.0


def test_short_position_coordinates_are_ordered():
    prices = [50500, 50200, 49900, 49600, 49300, 49000]
    trend = [50400, 50100, 49800, 49500, 49200, 48900]

    coords = compute_chart_coordinates(
        entry_price=50000,
        current_price=49000,
        stop_loss_price=51000,
        price_points=prices,
        trend_points=trend,
    )

    assert 0.0 <= coords.entry_y <= 120.0
    assert 0.0 <= coords.current_y <= 120.0
    assert coords.current_y > coords.entry_y


def test_flat_price_range_never_collapses():
    flat_series = [50000] * 6

    coords = compute_chart_coordinates(
        entry_price=50000,
        current_price=50000,
        stop_loss_price=50000,
        price_points=flat_series,
        trend_points=flat_series,
    )

    assert coords.price_range > 0
    assert 0.0 <= coords.entry_y <= 120.0
    assert 0.0 <= coords.current_y <= 120.0
    assert coords.stop_loss_y is not None


def test_current_price_is_included_in_scaling():
    prices = [49500, 49800, 50200, 50500, 50800]
    trend = [49600, 49900, 50200, 50500, 50800]

    coords_without_current = compute_chart_coordinates(
        entry_price=50000,
        current_price=51000,
        stop_loss_price=49000,
        price_points=prices,
        trend_points=trend,
        include_current_price=False,
    )

    coords_with_current = compute_chart_coordinates(
        entry_price=50000,
        current_price=51000,
        stop_loss_price=49000,
        price_points=prices,
        trend_points=trend,
        include_current_price=True,
    )

    assert coords_with_current.price_range >= coords_without_current.price_range
    assert coords_with_current.current_y != coords_without_current.current_y
