"""User configuration management module."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

USER_CONFIG_PATH = Path("crypto_bot/user_config.yaml")


def load_user_config(config_path: str = None) -> Dict[str, Any]:
    """Load user configuration from YAML file."""
    if config_path is None:
        config_path = USER_CONFIG_PATH
    
    if not Path(config_path).exists():
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def save_user_config(config: Dict[str, Any], config_path: str = None) -> bool:
    """Save user configuration to YAML file."""
    if config_path is None:
        config_path = USER_CONFIG_PATH
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception:
        return False


def get_user_setting(key: str, default: Any = None) -> Any:
    """Get a specific user setting."""
    config = load_user_config()
    return config.get(key, default)


def set_user_setting(key: str, value: Any) -> bool:
    """Set a specific user setting."""
    config = load_user_config()
    config[key] = value
    return save_user_config(config)


__all__ = ['load_user_config', 'save_user_config', 'get_user_setting', 'set_user_setting']
