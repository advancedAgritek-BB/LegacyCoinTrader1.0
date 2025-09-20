#!/usr/bin/env python3
"""Print a concise status snapshot for critical services."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict


STATUS_FILE = Path(__file__).resolve().parents[1] / "crypto_bot" / "logs" / "system_status.json"


def _load() -> Dict[str, Any]:
    if not STATUS_FILE.exists():
        return {}
    try:
        with STATUS_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _fmt_age(ts: Any) -> str:
    try:
        value = float(ts)
    except (TypeError, ValueError):
        return "unknown"
    delta = max(0.0, time.time() - value)
    if delta < 1:
        return "just now"
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    return f"{int(delta // 3600)}h ago"


def main() -> int:
    snapshot = _load()
    if not snapshot:
        print("No status data available yet.")
        print("Run trading services or wait for the next status refresh.")
        return 1

    now = time.time()
    print(f"System status at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}\n")
    for section, data in sorted(snapshot.items()):
        updated = _fmt_age(data.get("updated_at"))
        state = data.get("state", "unknown")
        print(f"[{section}] state={state} updated={updated}")
        for key in (
            "symbol",
            "side",
            "amount",
            "dry_run",
            "order_id",
            "reason",
            "cache_entries",
            "cache_hit_rate",
            "execution_opportunities",
            "successful_executions",
            "failed_executions",
            "last_execution_result",
        ):
            if key in data and data[key] not in (None, ""):
                print(f"  {key}: {data[key]}")
        print()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI helper
    sys.exit(main())

