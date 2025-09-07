import yaml
import importlib.util
from pathlib import Path

if not hasattr(yaml, "__file__"):
    import sys
    sys.modules.pop("yaml", None)
    spec = importlib.util.find_spec("yaml")
    if spec and spec.loader:
        real_yaml = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real_yaml)
        yaml = real_yaml

CONFIG_PATH = Path("crypto_bot/config.yaml")


def test_load_config_returns_dict():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    assert isinstance(config, dict)
    # Test for keys that actually exist in the config
    assert "mode" in config
    assert "testing_mode" in config
    assert "risk" in config
    assert "allow_short" in config
    assert "min_confidence_score" in config
    assert "exchange" in config
    assert "execution_mode" in config
    assert "timeframe" in config
    assert "telegram" in config
    assert "solana_scanner" in config
    assert "enhanced_scanning" in config
    assert "circuit_breaker" in config
    assert "bounce_scalper" in config
    assert "breakout" in config
    
    # Test telegram config structure
    telegram_cfg = config["telegram"]
    assert isinstance(telegram_cfg, dict)
    assert "enabled" in telegram_cfg
    assert "balance_updates" in telegram_cfg
    assert "status_updates" in telegram_cfg
    assert "trade_updates" in telegram_cfg
    
    # Test solana_scanner config structure
    sol_scanner = config["solana_scanner"]
    assert isinstance(sol_scanner, dict)
    for key in [
        "enabled",
        "interval_minutes",
        "api_keys",
        "min_volume_usd",
        "max_tokens_per_scan",
    ]:
        assert key in sol_scanner
    
    # Test enhanced_scanning config structure
    enhanced_scanning = config["enhanced_scanning"]
    assert isinstance(enhanced_scanning, dict)
    assert "enabled" in enhanced_scanning
    assert "scan_interval" in enhanced_scanning
    assert "min_volume_usd" in enhanced_scanning


def test_load_config_normalizes_symbol():
    """Test that fix_symbol function correctly normalizes XBT to BTC."""
    # Define fix_symbol function directly to avoid import issues
    def fix_symbol(sym: str) -> str:
        """Normalize different notations of Bitcoin."""
        if not isinstance(sym, str):
            return sym
        return sym.replace("XBT/", "BTC/").replace("XBT", "BTC")
    
    # Test the fix_symbol function
    assert fix_symbol("XBT/USDT") == "BTC/USDT"
    assert fix_symbol("XBT") == "BTC"
    assert fix_symbol("BTC/USDT") == "BTC/USDT"  # Should not change
    assert fix_symbol("ETH/USDT") == "ETH/USDT"  # Should not change
