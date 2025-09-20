"""Execution helpers for Solana sniping."""

from __future__ import annotations

import asyncio
from typing import Mapping, Dict, Any, Optional

from .watcher import NewPoolEvent


async def snipe(event: NewPoolEvent, score: float, cfg: Mapping[str, object]) -> Dict:
    """Execute a snipe trade for ``event`` using :func:`crypto_bot.solana_trading.sniper_trade`."""

    from crypto_bot.solana_trading import sniper_trade

    wallet = str(cfg.get("wallet_address", ""))
    base_token = str(cfg.get("base_token", "USDC"))
    amount = float(cfg.get("amount", 0))
    dry_run = bool(cfg.get("dry_run", True))
    paper_wallet = cfg.get("paper_wallet")
    wallet_override: Optional[Dict[str, Any]] = None

    context = cfg.get("wallet_context")
    if context and hasattr(context, "execution_override"):
        wallet_override = context.execution_override()
    else:
        override = cfg.get("wallet_override")
        if isinstance(override, dict):
            wallet_override = dict(override)

    return await sniper_trade(
        wallet,
        base_token,
        event.token_mint,
        amount,
        dry_run=dry_run,
        slippage_bps=int(cfg.get("slippage_bps", 50)),
        notifier=cfg.get("notifier"),
        paper_wallet=paper_wallet,
        wallet_override=wallet_override,
    )
