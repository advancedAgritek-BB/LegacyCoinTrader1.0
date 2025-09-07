#!/usr/bin/env python3
"""
Helper script to get positions with current prices.
This can be called by Flask when the direct import approach fails.
"""

import sys
import os
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Get positions with current prices and output as JSON."""
    try:
        # Load environment variables directly
        env_file = project_root / '.env'
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()

        # Import and call the function
        from frontend.app import get_open_positions, deduplicate_positions

        # Get positions from TradeManager
        positions = get_open_positions()

        # Deduplicate positions
        unique_positions = deduplicate_positions(positions)

        # Convert to JSON and output
        print(json.dumps(unique_positions))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
