from decimal import Decimal

from services.portfolio.models import PositionModel
from services.portfolio.service import PortfolioService


def _build_service(config: dict) -> PortfolioService:
    service = PortfolioService.__new__(PortfolioService)
    service._bot_config = config
    return service


def test_set_protective_levels_long_sets_initial_stops() -> None:
    service = _build_service(
        {
            "risk": {"stop_loss_pct": 0.02, "take_profit_pct": 0.05},
            "exit_strategy": {"trailing_stop_pct": 0.01},
        }
    )

    position = PositionModel(symbol="BTC/USD")
    position.side = "long"
    position.average_price = Decimal("100")
    position.highest_price = Decimal("100")
    position.lowest_price = Decimal("100")
    position.is_open = True

    service._set_protective_levels(position)

    assert position.stop_loss_price == Decimal("99")
    assert position.take_profit_price == Decimal("105")
    assert position.trailing_stop_pct == Decimal("0.01")


def test_set_protective_levels_short_sets_initial_stops() -> None:
    service = _build_service(
        {
            "risk": {"stop_loss_pct": 0.02, "take_profit_pct": 0.05},
            "exit_strategy": {"trailing_stop_pct": 0.01},
        }
    )

    position = PositionModel(symbol="ETH/USD")
    position.side = "short"
    position.average_price = Decimal("50")
    position.highest_price = Decimal("50")
    position.lowest_price = Decimal("50")
    position.is_open = True

    service._set_protective_levels(position)

    assert position.stop_loss_price == Decimal("50.5")
    assert position.take_profit_price == Decimal("47.5")
    assert position.trailing_stop_pct == Decimal("0.01")


def test_trailing_stop_updates_when_price_makes_new_highs() -> None:
    service = _build_service(
        {
            "risk": {"stop_loss_pct": 0.02, "take_profit_pct": 0.05},
            "exit_strategy": {"trailing_stop_pct": 0.01},
        }
    )

    position = PositionModel(symbol="SOL/USD")
    position.side = "long"
    position.average_price = Decimal("100")
    position.highest_price = Decimal("100")
    position.lowest_price = Decimal("100")
    position.is_open = True

    service._set_protective_levels(position)

    position.highest_price = Decimal("110")
    service._update_trailing_stop(position)

    assert position.stop_loss_price == Decimal("108.9")
