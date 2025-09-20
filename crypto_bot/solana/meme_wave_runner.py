from __future__ import annotations

import asyncio
from typing import Mapping, Optional

from .watcher import PoolWatcher
from .safety import is_safe
from .score import score_event
from .risk import RiskTracker
from .executor import snipe


async def _run(cfg: Mapping[str, object]) -> None:
    """Background task that watches pools and triggers snipes."""
    pool_cfg = cfg.get("pool", {})
    watcher = PoolWatcher(
        pool_cfg.get("url", ""),
        pool_cfg.get("interval", 5),
        pool_cfg.get("websocket_url"),
        pool_cfg.get("raydium_program_id"),
    )
    tracker = RiskTracker(cfg.get("risk_file", "crypto_bot/logs/sniper_risk.json"))
    safety_cfg = cfg.get("safety", {})
    scoring_cfg = cfg.get("scoring", {})
    risk_cfg = cfg.get("risk", {})
    exec_cfg = cfg.get("execution", {})

    # Extract paper trading parameters
    dry_run = exec_cfg.get("dry_run", True)
    paper_wallet = exec_cfg.get("paper_wallet")

    async for event in watcher.watch():
        if not is_safe(event, safety_cfg):
            continue
        score = score_event(event, scoring_cfg)
        if not tracker.allow_snipe(event.token_mint, risk_cfg):
            continue
        tracker.add_snipe(event.token_mint, event.liquidity)

        # Update exec_cfg with paper trading parameters
        updated_exec_cfg = exec_cfg.copy()
        updated_exec_cfg["dry_run"] = dry_run
        updated_exec_cfg["paper_wallet"] = paper_wallet
        if "wallet_context" in exec_cfg:
            updated_exec_cfg["wallet_context"] = exec_cfg["wallet_context"]
        elif "wallet_override" in exec_cfg:
            updated_exec_cfg["wallet_override"] = exec_cfg["wallet_override"]

        await snipe(event, score, updated_exec_cfg)


def start_runner(cfg: Mapping[str, object]) -> Optional[asyncio.Task]:
    """Return a task running the meme-wave sniping loop when enabled."""
    if not cfg.get("enabled"):
        return None
    return asyncio.create_task(_run(cfg))
