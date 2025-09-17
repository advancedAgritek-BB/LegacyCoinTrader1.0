import pytest

from tests.regression.utils import run_regression_cycle


@pytest.mark.regression
def test_simple_cycle_regression() -> None:
    observed, expected = run_regression_cycle("simple_cycle")

    assert observed.signal == expected.signal
    assert observed.risk == expected.risk
    assert observed.execution == expected.execution
