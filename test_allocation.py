"""
Test for strategy allocation calculation.
"""

import pytest
from frontend.utils import calculate_dynamic_allocation, get_allocation_comparison

def test_allocation():
    """Test the dynamic allocation calculation."""

    # Test dynamic allocation
    dynamic_allocation = calculate_dynamic_allocation()
    assert dynamic_allocation is not None, "Dynamic allocation should not be None"
    assert isinstance(dynamic_allocation, dict), "Dynamic allocation should be a dictionary"

    # Test allocation comparison
    comparison = get_allocation_comparison()
    assert comparison is not None, "Allocation comparison should not be None"

    # Verify the results make sense
    if dynamic_allocation:
        total_percentage = sum(dynamic_allocation.values())
        assert abs(total_percentage - 100.0) < 1.0, f"Allocation percentages should sum to ~100%, got {total_percentage:.2f}%"

        # Verify all percentages are positive
        for strategy, percentage in dynamic_allocation.items():
            assert percentage >= 0, f"Strategy {strategy} has negative percentage: {percentage}"
            assert percentage <= 100, f"Strategy {strategy} has percentage > 100%: {percentage}"