from __future__ import annotations

import os
import base64
from typing import Any, Mapping, Optional

try:
    import aiohttp
except Exception:  # pragma: no cover - optional during tests
    aiohttp = None
try:
    from solana.rpc.async_api import AsyncClient
    from solana.transaction import Transaction
    from solders.transaction import VersionedTransaction
    from solders.signature import Signature
    from solders.keypair import Keypair
except Exception:  # pragma: no cover - make solana optional
    AsyncClient = Transaction = VersionedTransaction = Signature = Keypair = None

from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.solana.wallet_context import PrivateKeyLike, decode_private_key_bytes


logger = setup_logger(__name__, LOG_DIR / "raydium_client.log")

QUOTE_URL = "https://transaction-v1.raydium.io/compute/swap-base-in"
TX_URL = "https://transaction-v1.raydium.io/transaction/swap-base-in"


def get_wallet(private_key_override: Optional[PrivateKeyLike] = None) -> Keypair:
    """Return wallet keypair from configuration or override."""

    key_source: Optional[PrivateKeyLike] = private_key_override or os.getenv("SOLANA_PRIVATE_KEY")
    if not key_source:
        raise ValueError("Solana private key not configured")

    secret = decode_private_key_bytes(key_source)
    if len(secret) == 64:
        return Keypair.from_bytes(secret)
    if len(secret) == 32:
        return Keypair.from_seed(secret)
    raise ValueError("Unsupported Solana private key length; expected 32 or 64 bytes")


async def get_swap_quote(
    input_mint: str,
    output_mint: str,
    amount: int,
    slippage_bps: int = 50,
    tx_version: str = "V0",
) -> Mapping[str, Any]:
    """Return Raydium swap quote json."""
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": amount,
        "slippageBps": slippage_bps,
        "txVersion": tx_version,
    }
    if aiohttp is None:
        return {}
    async with aiohttp.ClientSession() as session:
        async with session.get(QUOTE_URL, params=params, timeout=10) as resp:
            resp.raise_for_status()
            data = await resp.json()
    logger.info("Fetched quote %s", data.get("id"))
    return data


async def execute_swap(
    wallet_address: str,
    input_account: str,
    output_account: str,
    swap_response: Mapping[str, Any],
    *,
    wrap_sol: bool = True,
    unwrap_sol: bool = False,
    compute_unit_price: int = 1000,
    tx_version: str = "V0",
    risk_manager: Optional[object] = None,
    wallet_override: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Execute a Raydium swap and return the RPC result."""
    wallet_override = wallet_override or {}
    resolved_wallet_address = str(wallet_override.get("public_key", wallet_address))

    payload = dict(swap_response)
    payload.update(
        {
            "walletAddress": resolved_wallet_address,
            "inputAccount": input_account,
            "outputAccount": output_account,
            "wrapAndUnwrapSol": {"wrapSol": wrap_sol, "unwrapSol": unwrap_sol},
            "computeUnitPriceMicroLamports": compute_unit_price,
            "txVersion": tx_version,
        }
    )
    if aiohttp is None:
        return {"error": "aiohttp not available"}
    async with aiohttp.ClientSession() as session:
        async with session.post(TX_URL, json=payload, timeout=10) as resp:
            resp.raise_for_status()
            tx_data = await resp.json()

    tx_b64 = tx_data.get("swapTransaction") or tx_data.get("transaction")
    if not tx_b64:
        logger.error("Transaction missing from response")
        return tx_data
    raw = base64.b64decode(tx_b64)
    kp = get_wallet(wallet_override.get("private_key"))
    if tx_version == "V0":
        vt = VersionedTransaction.from_bytes(raw)
        vt = VersionedTransaction(vt.message, [kp])
        send_bytes = bytes(vt)
    else:
        tx = Transaction.deserialize(raw)
        tx.sign(kp)
        send_bytes = tx.serialize()

    if AsyncClient is None:
        return {"tx_hash": "", "response": tx_data}
    rpc_url = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    async with AsyncClient(rpc_url) as client:
        res = await client.send_raw_transaction(send_bytes)
        sig = getattr(res, "value", None) or res.get("result")
        if isinstance(sig, str):
            sig_obj = Signature.from_string(sig)
        else:
            sig_obj = sig
        if sig_obj is not None:
            await client.confirm_transaction(sig_obj)
            tx_hash = str(sig_obj)
        else:
            tx_hash = ""
    result = {"tx_hash": tx_hash, "response": tx_data}
    logger.info("Swap executed tx=%s", tx_hash)
    return result


async def sniper_trade(
    input_mint: str,
    output_mint: str,
    amount: int,
    notifier: Optional[Any] = None,
    config: Optional[Mapping[str, Any]] = None,
    wallet_override: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Execute a simple snipe trade and convert profits to BTC."""
    from crypto_bot.risk.risk_manager import RiskManager, RiskConfig
    from crypto_bot.fund_manager import auto_convert_funds

    cfg = config or {}
    quote = await get_swap_quote(
        input_mint,
        output_mint,
        amount,
        tx_version=cfg.get("tx_version", "V0"),
    )

    risk_cfg = RiskConfig(max_drawdown=1.0, stop_loss_pct=0.01, take_profit_pct=0.01)
    rm = RiskManager(risk_cfg)
    size = rm.position_size(1.0, float(amount))

    swap_res = await execute_swap(
        cfg.get("wallet_address", ""),
        input_mint,
        output_mint,
        quote.get("data", quote),
        tx_version=cfg.get("tx_version", "V0"),
        risk_manager=rm,
        wallet_override=wallet_override,
    )

    await auto_convert_funds(
        cfg.get("wallet_address", ""),
        output_mint,
        "BTC",
        size,
        dry_run=True,
        notifier=notifier,
        wallet_override=dict(wallet_override) if wallet_override else None,
    )
    return swap_res
