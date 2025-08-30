#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

LOG_DIR = Path(__file__).resolve().parents[1] / "crypto_bot" / "logs"
BOT_LOG = LOG_DIR / "bot.log"
TELEMETRY_PRIMARY = LOG_DIR / "telemetry.csv"
TELEMETRY_FALLBACK = LOG_DIR / "metrics.csv"
BACKLOG_JSON = LOG_DIR / "issue_backlog.json"
FIX_PROMPTS_MD = LOG_DIR / "fix_prompts.md"


ISSUE_PATTERNS = [
    (re.compile(r"ERROR|CRITICAL", re.I), "error"),
    (re.compile(r"WebSocket ping failed|ws_errors", re.I), "websocket"),
    (re.compile(r"fetch_tickers failed|api_errors", re.I), "exchange_api"),
    (re.compile(r"No symbols met volume/spread requirements", re.I), "no_symbols"),
    (re.compile(r"All signals filtered out - nothing actionable", re.I), "no_actionable"),
    (re.compile(r"Trade blocked .*risk", re.I), "risk_block"),
]


def tail_file(path: Path, offset: int) -> Tuple[List[str], int]:
    if not path.exists():
        return [], offset
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        fh.seek(offset)
        lines = fh.readlines()
        return lines, fh.tell()


def parse_telemetry(path: Path) -> Dict[str, int]:
    metrics: Dict[str, int] = {}
    if not path.exists():
        return metrics
    try:
        last = None
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                last = line.strip()
        if not last:
            return metrics
        # naive CSV parser for last line
        fields = [f.strip() for f in last.split(",")]
        # Expect header in first line; recover keys by reading header
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            header = fh.readline().strip()
        keys = [k.strip() for k in header.split(",")]
        if len(keys) == len(fields):
            for k, v in zip(keys, fields):
                if k == "timestamp":
                    continue
                try:
                    metrics[k] = int(float(v))
                except Exception:
                    pass
    except Exception:
        pass
    return metrics


def classify_issue(line: str) -> Tuple[str, str]:
    for pat, kind in ISSUE_PATTERNS:
        if pat.search(line):
            return kind, line.strip()
    return "info", line.strip()


def build_fix_prompt(kind: str, context: str, telemetry: Dict[str, int]) -> str:
    ts = datetime.utcnow().isoformat()
    if kind == "no_symbols":
        counts = {
            "considered": telemetry.get("scan.symbols_considered", 0),
            "selected": telemetry.get("scan.selected", 0),
            "skipped": telemetry.get("scan.symbols_skipped", 0),
            "skip_low_volume_uncached": telemetry.get("scan.skip_low_volume_uncached", 0),
            "skip_spread": telemetry.get("scan.skip_spread", 0),
            "skip_volume_percentile": telemetry.get("scan.skip_volume_percentile", 0),
            "skip_min_score": telemetry.get("scan.skip_min_score", 0),
        }
        return (
            f"[{ts}] Improve symbol throughput\n"
            f"Context: {context}\n\n"
            f"Observed: {counts}\n\n"
            "Proposed fixes:\n"
            "- Lower min_volume_usd or volume_percentile dynamically during droughts.\n"
            "- Relax max_spread_pct slightly when spread skip dominates.\n"
            "- Reduce min_symbol_score when skip_min_score is high.\n"
            "- Add fallback to include top-N by volume regardless of change_pct.\n"
        )
    if kind == "no_actionable":
        return (
            f"[{ts}] Increase actionable signals\n"
            f"Context: {context}\n\n"
            "Proposed fixes:\n"
            "- Log per-strategy scores for top candidates to tune thresholds.\n"
            "- Enable ensemble mode with ensemble_min_conf slightly lower.\n"
            "- Add alternative strategies for current regime via router.\n"
        )
    if kind == "risk_block":
        return (
            f"[{ts}] Risk manager blocked trades\n"
            f"Context: {context}\n\n"
            "Proposed fixes:\n"
            "- Surface detailed allow_trade reasons and map to config toggles.\n"
            "- Adjust allocation caps or enable partial allocations.\n"
        )
    if kind == "websocket":
        return (
            f"[{ts}] WebSocket instability\n"
            f"Context: {context}\n\n"
            "Proposed fixes:\n"
            "- Backoff and auto-disable WS after repeated failures (already present).\n"
            "- Ensure HTTP fallback batch sizes are tuned (kraken_batch_size).\n"
        )
    if kind == "exchange_api":
        return (
            f"[{ts}] Exchange API failures\n"
            f"Context: {context}\n\n"
            "Proposed fixes:\n"
            "- Increase ticker_retry_attempts and add jitter to retries.\n"
            "- Reduce max_concurrent_ohlcv under high volatility factor.\n"
        )
    return f"[{ts}] General issue\nContext: {context}\n\nProposed: Investigate logs and telemetry for root cause.\n"


def prioritize(issues: Dict[str, int]) -> List[Tuple[str, int]]:
    # Simple priority: error > websocket/api > no_symbols/no_actionable > others
    weights = {
        "error": 5,
        "websocket": 4,
        "exchange_api": 4,
        "no_symbols": 3,
        "no_actionable": 3,
        "risk_block": 2,
        "info": 1,
    }
    scored = [(k, v * weights.get(k, 1)) for k, v in issues.items()]
    return sorted(scored, key=lambda kv: kv[1], reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor logs and telemetry to maintain an issue backlog")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    backlog: Dict[str, Dict] = {}
    if BACKLOG_JSON.exists():
        try:
            backlog = json.loads(BACKLOG_JSON.read_text())
        except Exception:
            backlog = {}

    offset = 0
    if BOT_LOG.exists():
        offset = BOT_LOG.stat().st_size

    while True:
        # Pick whichever telemetry file exists
        telem_path = TELEMETRY_PRIMARY if TELEMETRY_PRIMARY.exists() else TELEMETRY_FALLBACK
        telemetry = parse_telemetry(telem_path)
        lines, offset = tail_file(BOT_LOG, offset)

        by_kind: Dict[str, List[str]] = defaultdict(list)
        for ln in lines:
            kind, msg = classify_issue(ln)
            if kind != "info":
                by_kind[kind].append(msg)

        counts = {k: len(v) for k, v in by_kind.items()}
        if counts:
            ranked = prioritize(counts)
            for kind, score in ranked:
                ctx = by_kind[kind][-1]
                prompt = build_fix_prompt(kind, ctx, telemetry)
                backlog[kind] = {
                    "score": score,
                    "last_seen": datetime.utcnow().isoformat(),
                    "context": ctx,
                    "prompt": prompt,
                }
            # Write outputs
            BACKLOG_JSON.write_text(json.dumps(backlog, indent=2))
            with FIX_PROMPTS_MD.open("w", encoding="utf-8") as fh:
                fh.write("# Auto-generated Fix Prompts\n\n")
                for kind, entry in sorted(backlog.items(), key=lambda kv: kv[1]["score"], reverse=True):
                    fh.write(f"## {kind} (priority {entry['score']})\n\n")
                    fh.write(entry["prompt"].rstrip() + "\n\n")

        if args.once:
            break
        time.sleep(max(args.interval, 1.0))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

