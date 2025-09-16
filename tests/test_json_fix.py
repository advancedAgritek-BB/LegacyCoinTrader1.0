#!/usr/bin/env python3
"""Test script to verify JSON parsing fix."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json

# Import the safe_json_load function from the frontend app
def safe_json_load(file_path):
    """Safely load JSON file, handling common corruption issues."""
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Remove trailing non-JSON characters
        content = content.strip()
        if content.endswith('%'):
            content = content[:-1].strip()
        if content.endswith(','):
            content = content[:-1].strip()

        # Find the last valid closing brace
        last_brace = content.rfind('}')
        if last_brace != -1:
            content = content[:last_brace + 1]

        return json.loads(content)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing failed, attempting to repair: {e}")
        try:
            # Try to extract just the JSON portion
            start = content.find('{')
            if start != -1:
                content = content[start:]
                last_brace = content.rfind('}')
                if last_brace != -1:
                    content = content[:last_brace + 1]
                    return json.loads(content)
        except Exception as repair_error:
            print(f"JSON repair also failed: {repair_error}")

        raise e


def test_json_parsing():
    """Test that JSON parsing works with corrupted files."""
    print("Testing JSON parsing fix...")

    # Test with the actual trade_manager_state.json file
    state_file = Path("crypto_bot/logs/trade_manager_state.json")

    if state_file.exists():
        print(f"Testing with actual file: {state_file}")
        try:
            state = safe_json_load(state_file)
            print("✅ JSON parsing successful!")
            print(f"Found {len(state.get('trades', []))} trades")
            print(f"Found {len(state.get('positions', {}))} positions")
            return True
        except Exception as e:
            print(f"❌ JSON parsing failed: {e}")
            return False
    else:
        print(f"File not found: {state_file}")
        return False


if __name__ == "__main__":
    test_json_parsing()
