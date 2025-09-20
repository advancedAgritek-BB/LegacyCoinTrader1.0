"""
JSON utilities for safe loading and handling of JSON data.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def safe_json_load(
    file_path: Union[str, Path], 
    default: Optional[Any] = None
) -> Any:
    """
    Safely load JSON from a file, returning default value on error.
    
    Args:
        file_path: Path to the JSON file
        default: Default value to return on error (defaults to empty dict)
        
    Returns:
        Loaded JSON data or default value
    """
    if default is None:
        default = {}
        
    try:
        path = Path(file_path)
        if not path.exists():
            logger.debug(f"JSON file not found: {file_path}")
            return default
            
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in file {file_path}: {e}")
        return default
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return default


def safe_json_save(
    data: Any, 
    file_path: Union[str, Path], 
    indent: int = 2,
    create_dirs: bool = True
) -> bool:
    """
    Safely save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        indent: JSON indentation level
        create_dirs: Whether to create parent directories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
            
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            
        return True
        
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


def safe_json_loads(json_string: str, default: Optional[Any] = None) -> Any:
    """
    Safely parse a JSON string, returning default value on error.
    
    Args:
        json_string: JSON string to parse
        default: Default value to return on error
        
    Returns:
        Parsed JSON data or default value
    """
    if default is None:
        default = {}
        
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON string: {e}")
        return default
    except Exception as e:
        logger.error(f"Error parsing JSON string: {e}")
        return default


def safe_json_dumps(data: Any, indent: int = 2, default_str: str = "{}") -> str:
    """
    Safely serialize data to JSON string, returning default on error.
    
    Args:
        data: Data to serialize
        indent: JSON indentation level
        default_str: Default string to return on error
        
    Returns:
        JSON string or default string
    """
    try:
        return json.dumps(data, indent=indent, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error serializing to JSON: {e}")
        return default_str
