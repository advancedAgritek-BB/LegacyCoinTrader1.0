"""Configuration management module."""

import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import ValidationError

from ..main import load_config as _load_config

# Import schema classes from the correct location
try:
    from schema.scanner import ScannerConfig, SolanaScannerConfig, PythConfig
except ImportError:
    # Fallback for when schema module is not available
    ScannerConfig = SolanaScannerConfig = PythConfig = None

CONFIG_PATH = Path("crypto_bot/config.yaml")


def load_config(config_path: str = None) -> dict:
    """Load YAML configuration for the bot."""
    if config_path:
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = _load_config()
    return data


def save_config(config: Dict[str, Any], config_path: str = None) -> None:
    """Save configuration to YAML file."""
    if config_path is None:
        config_path = CONFIG_PATH
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration using Pydantic models."""
    try:
        if hasattr(ScannerConfig, "model_validate"):
            ScannerConfig.model_validate(config)
        else:  # pragma: no cover - for Pydantic < 2
            ScannerConfig.parse_obj(config)
        
        # Validate solana_scanner if present
        raw_scanner = config.get("solana_scanner", {}) or {}
        if raw_scanner:
            if hasattr(SolanaScannerConfig, "model_validate"):
                SolanaScannerConfig.model_validate(raw_scanner)
            else:  # pragma: no cover - for Pydantic < 2
                SolanaScannerConfig.parse_obj(raw_scanner)
        
        # Validate pyth config if present
        raw_pyth = config.get("pyth", {}) or {}
        if raw_pyth:
            if hasattr(PythConfig, "model_validate"):
                PythConfig.model_validate(raw_pyth)
            else:  # pragma: no cover - for Pydantic < 2
                PythConfig.parse_obj(raw_pyth)
        
        return True
    except ValidationError:
        return False


__all__ = ['load_config', 'save_config', 'validate_config']
