"""
Simple test to verify circuit breaker integration is working.
"""

import pytest
import asyncio
from crypto_bot.utils.circuit_breaker import get_circuit_breaker_manager


@pytest.mark.asyncio
async def test_basic_circuit_breaker_integration():
    """Test basic circuit breaker functionality."""
    manager = get_circuit_breaker_manager()
    
    # Test successful call
    def success_func():
        return "success"
    
    result = await manager.call_with_circuit_breaker("test_success", success_func)
    assert result == "success"
    
    # Test failed call
    def fail_func():
        raise Exception("test failure")
    
    with pytest.raises(Exception, match="test failure"):
        await manager.call_with_circuit_breaker("test_fail", fail_func)
    
    # Test circuit breaker metrics
    metrics = manager.get_all_metrics()
    assert "test_success" in metrics
    assert "test_fail" in metrics
    
    assert metrics["test_success"]["state"] == "CLOSED"
    assert metrics["test_success"]["successful_calls"] == 1
    
    assert metrics["test_fail"]["state"] == "CLOSED"
    assert metrics["test_fail"]["failed_calls"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
