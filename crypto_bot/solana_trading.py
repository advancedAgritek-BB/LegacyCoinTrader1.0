from __future__ import annotations

import asyncio
import os
import time
from typing import Dict, Optional

import aiohttp
from solana.rpc.async_api import AsyncClient

from crypto_bot.execution.solana_executor import (
    execute_swap,
    JUPITER_QUOTE_URL,
)
from crypto_bot.fund_manager import auto_convert_funds
from crypto_bot.utils.logger import LOG_DIR, setup_logger


logger = setup_logger(__name__, LOG_DIR / "solana_trading.log")


async def _fetch_price(token_in: str, token_out: str) -> float:
    """Return current price for ``token_in``/``token_out`` using Jupiter."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            JUPITER_QUOTE_URL,
            params={
                "inputMint": token_in,
                "outputMint": token_out,
                "amount": 1_000_000,
                "slippageBps": 50,
            },
            timeout=10,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    route = (data.get("data") or [{}])[0]
    try:
        return float(route["outAmount"]) / float(route["inAmount"])
    except Exception:
        return 0.0


async def monitor_profit(tx_sig: str, threshold: float = 0.2) -> float:
    """Return profit when price change exceeds ``threshold``.

    Parameters
    ----------
    tx_sig:
        Signature of the entry swap transaction.
    threshold:
        Percentage gain required to trigger profit taking.
    """

    rpc_url = os.getenv(
        "SOLANA_RPC_URL",
        f"https://mainnet.helius-rpc.com/?api-key={os.getenv('HELIUS_KEY', '')}",
    )
    client = AsyncClient(rpc_url)
    try:
        entry_price = None
        out_amount = 0.0
        in_mint = out_mint = ""
        # wait for confirmed tx with token balances
        for _ in range(30):
            try:
                resp = await client.get_confirmed_transaction(tx_sig, encoding="jsonParsed")
            except Exception:
                resp = None
            tx = resp.get("result") if resp else None
            if tx:
                meta = tx.get("meta", {})
                pre = meta.get("preTokenBalances") or []
                post = meta.get("postTokenBalances") or []
                if len(pre) >= 2 and len(post) >= 2:
                    in_mint = pre[0].get("mint", "")
                    out_mint = post[1].get("mint", "")
                    try:
                        in_amt = float(pre[0]["uiTokenAmount"].get("uiAmount", pre[0]["uiTokenAmount"].get("uiAmountString", 0)))
                        out_amount = float(post[1]["uiTokenAmount"].get("uiAmount", post[1]["uiTokenAmount"].get("uiAmountString", 0)))
                    except Exception:
                        in_amt = 0.0
                        out_amount = 0.0
                    if in_amt and out_amount:
                        entry_price = in_amt / out_amount
                        break
            await asyncio.sleep(2)
        if entry_price is None:
            return 0.0

        start = time.time()
        while time.time() - start < 300:
            price = await _fetch_price(out_mint, in_mint)
            if price:
                change = (price - entry_price) / entry_price
                if change >= threshold:
                    return out_amount * change
            await asyncio.sleep(5)
    finally:
        await client.close()
    return 0.0


async def sniper_trade(
    wallet: str,
    base_token: str,
    target_token: str,
    amount: float,
    *,
    dry_run: bool = True,
    slippage_bps: int = 50,
    notifier: Optional[object] = None,
    profit_threshold: float = 0.2,
    paper_wallet=None,
    wallet_override: Optional[Dict[str, object]] = None,
) -> Dict:
    """Buy ``target_token`` then convert profits when threshold reached."""

    trade = await execute_swap(
        base_token,
        target_token,
        amount,
        notifier=notifier,
        slippage_bps=slippage_bps,
        dry_run=dry_run,
        wallet_override=wallet_override,
    )
    tx_sig = trade.get("tx_hash")
    if not tx_sig or tx_sig == "DRYRUN":
        # Update paper wallet for dry run sniper trades
        if dry_run and paper_wallet:
            try:
                # Simulate buying target token with base token
                # For paper trading, we'll treat this as a buy order
                trade_id = paper_wallet.open(f"{target_token}/{base_token}", "buy", amount, 1.0)  # Use 1.0 as placeholder price
                logger.info(f"Paper sniper trade opened: buy {amount} {target_token} for {amount} {base_token}, balance: ${paper_wallet.balance:.2f}")
            except Exception as e:
                logger.error(f"Failed to open paper sniper trade: {e}")
        return trade

    profit = await monitor_profit(tx_sig, profit_threshold)
    if profit > 0:
        await auto_convert_funds(
            wallet,
            target_token,
            base_token,
            profit,
            dry_run=dry_run,
            slippage_bps=slippage_bps,
            notifier=notifier,
            wallet_override=wallet_override,
        )
        
        # Update paper wallet for profit conversion in dry run mode
        if dry_run and paper_wallet:
            try:
                # Close the position and realize profit
                pnl = paper_wallet.close(f"{target_token}/{base_token}", amount, 1.0 + profit_threshold)
                logger.info(f"Paper sniper trade closed with profit: {pnl:.2f}, balance: ${paper_wallet.balance:.2f}")
            except Exception as e:
                logger.error(f"Failed to close paper sniper trade: {e}")
    
    return trade
