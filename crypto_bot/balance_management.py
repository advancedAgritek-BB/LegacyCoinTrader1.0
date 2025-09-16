"""Utility functions for managing balances and paper wallet state."""
from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Optional

import pandas as pd

from crypto_bot.utils.logger import LOG_DIR
from crypto_bot.utils.position_logger import log_balance
from crypto_bot.utils.telegram import TelegramNotifier

logger = logging.getLogger("bot")


async def fetch_balance(exchange, paper_wallet, config):
    """Return the latest wallet balance without logging."""
    if config["execution_mode"] != "dry_run":
        if asyncio.iscoroutinefunction(getattr(exchange, "fetch_balance", None)):
            bal = await exchange.fetch_balance()
        else:
            bal = await asyncio.to_thread(exchange.fetch_balance_with_retry)
        return bal["USDT"]["free"] if isinstance(bal["USDT"], dict) else bal["USDT"]
    return paper_wallet.balance if paper_wallet else 0.0


async def fetch_and_log_balance(exchange, paper_wallet, config):
    """Return the latest wallet balance and log it."""
    latest_balance = await fetch_balance(exchange, paper_wallet, config)
    log_balance(float(latest_balance))
    return latest_balance


def notify_balance_change(
    notifier: Optional[TelegramNotifier],
    previous: Optional[float],
    new_balance: float,
    enabled: bool,
    is_paper_trading: bool = False,
) -> float:
    """Send a notification if the balance changed."""
    if notifier and enabled and previous is not None and new_balance != previous:
        prefix = "📄 Paper" if is_paper_trading else "💰 Live"
        notifier.notify(f"{prefix} Balance changed: ${new_balance:.2f}")
    return new_balance


def sync_paper_wallet_balance(ctx):
    """Ensure paper wallet balance is synchronized with context."""
    if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
        if abs(ctx.balance - ctx.paper_wallet.balance) > 0.01:
            logger.warning(
                "Balance mismatch detected: ctx.balance=$%s, paper_wallet.balance=$%s",
                f"{ctx.balance:.2f}",
                f"{ctx.paper_wallet.balance:.2f}",
            )
            ctx.balance = ctx.paper_wallet.balance
            logger.info("Balance synchronized: $%.2f", ctx.balance)
            if hasattr(ctx, "risk_manager"):
                ctx.risk_manager.update_equity(ctx.balance)
                logger.info("Risk manager equity updated to: $%.2f", ctx.balance)
        return ctx.paper_wallet.balance
    return ctx.balance


def update_position_pnl(ctx):
    """Update PnL for all positions based on current market prices using TradeManager."""
    if not ctx.trade_manager:
        _update_position_pnl_legacy(ctx)
        return

    for sym, pos in ctx.positions.items():
        try:
            current_price = None
            if hasattr(ctx, "df_cache") and sym in ctx.df_cache:
                df = ctx.df_cache[sym]
                if not df.empty:
                    current_price = float(df.iloc[-1]["close"])
            if current_price is not None:
                ctx.trade_manager.update_price(sym, Decimal(str(current_price)))
                position = ctx.trade_manager.get_position(sym)
                if position:
                    unrealized_pnl, unrealized_pct = position.calculate_unrealized_pnl(
                        Decimal(str(current_price))
                    )
                    pos["pnl"] = float(unrealized_pnl)
                    pos["highest_price"] = (
                        float(position.highest_price)
                        if position.highest_price
                        else current_price
                    )
                    pos["lowest_price"] = (
                        float(position.lowest_price)
                        if position.lowest_price
                        else current_price
                    )
                    pos["trailing_stop"] = (
                        float(position.stop_loss_price)
                        if position.stop_loss_price
                        else 0.0
                    )
                    logger.debug(
                        "Updated PnL for %s: $%.2f (%.2f%%) at price $%.6f",
                        sym,
                        unrealized_pnl,
                        unrealized_pct,
                        current_price,
                    )
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning("Failed to update PnL for %s: %s", sym, exc)


def _update_position_pnl_legacy(ctx):
    """Legacy PnL calculation method for backward compatibility."""
    if not ctx.config.get("execution_mode") == "dry_run" or not ctx.paper_wallet:
        return

    for sym, pos in ctx.positions.items():
        try:
            if hasattr(ctx, "df_cache") and sym in ctx.df_cache:
                df = ctx.df_cache[sym]
                if not df.empty:
                    current_price = float(df.iloc[-1]["close"])
                    if ctx.paper_wallet and sym in ctx.paper_wallet.positions:
                        unrealized_pnl = ctx.paper_wallet.unrealized(sym, current_price)
                        pos["pnl"] = unrealized_pnl
                        if pos["side"] == "buy":
                            pos["highest_price"] = max(
                                pos.get("highest_price", current_price), current_price
                            )
                        else:
                            pos["lowest_price"] = min(
                                pos.get("lowest_price", current_price), current_price
                            )
                        logger.debug(
                            "Updated PnL for %s: $%.2f at price $%.6f",
                            sym,
                            unrealized_pnl,
                            current_price,
                        )
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning("Failed to update PnL for %s: %s", sym, exc)


def validate_paper_wallet_consistency(ctx) -> bool:
    """Validate that paper wallet state is consistent and log any issues."""
    if not ctx.config.get("execution_mode") == "dry_run" or not ctx.paper_wallet:
        return True

    try:
        balance_diff = abs(ctx.balance - ctx.paper_wallet.balance)
        if balance_diff > 0.01:
            logger.warning(
                "Balance inconsistency detected: ctx.balance=$%.2f, paper_wallet.balance=$%.2f, diff=$%.2f",
                ctx.balance,
                ctx.paper_wallet.balance,
                balance_diff,
            )
            return False

        if ctx.paper_wallet.balance < 0 and len(ctx.paper_wallet.positions) == 0:
            logger.error(
                "Negative paper wallet balance detected with no open positions: $%.2f",
                ctx.paper_wallet.balance,
            )
            return False
        elif ctx.paper_wallet.balance < 0:
            logger.info(
                "Negative cash balance ($%.2f) is normal with %d open positions",
                ctx.paper_wallet.balance,
                len(ctx.paper_wallet.positions),
            )

        if hasattr(ctx.paper_wallet, "validate_wallet_state"):
            if not ctx.paper_wallet.validate_wallet_state():
                logger.warning("Paper wallet state validation failed")
                return False

        if hasattr(ctx, "validate_position_consistency") and ctx.use_trade_manager_as_source:
            if not ctx.validate_position_consistency():
                logger.warning("Position system consistency check failed")
                return False
        else:
            ctx_positions = len(ctx.positions)
            wallet_positions = len(ctx.paper_wallet.positions)
            if ctx_positions != wallet_positions:
                logger.warning(
                    "Position count mismatch: ctx.positions=%d, paper_wallet.positions=%d",
                    ctx_positions,
                    wallet_positions,
                )
                logger.warning(
                    "This indicates a desynchronization between trading context and paper wallet",
                )
                try:
                    from crypto_bot.utils.wallet_sync_utility import auto_fix_wallet_sync

                    logger.info("Attempting automatic wallet synchronization...")
                    sync_success, sync_message = auto_fix_wallet_sync(ctx)
                    if sync_success:
                        logger.info("✅ Auto-sync successful: %s", sync_message)
                        ctx_positions_after = len(ctx.positions)
                        wallet_positions_after = len(ctx.paper_wallet.positions)
                        if ctx_positions_after == wallet_positions_after:
                            logger.info("✅ Position count mismatch resolved")
                            return True
                        logger.warning(
                            "⚠️ Auto-sync completed but count mismatch persists: ctx=%d, wallet=%d",
                            ctx_positions_after,
                            wallet_positions_after,
                        )
                        return False
                    logger.error("❌ Auto-sync failed: %s", sync_message)
                    logger.warning(
                        "Consider enabling TradeManager as single source of truth to prevent this issue",
                    )
                    return False
                except Exception as exc:  # pragma: no cover - sync failures
                    logger.error("❌ Error during auto-sync attempt: %s", exc)
                    logger.warning(
                        "Consider enabling TradeManager as single source of truth to prevent this issue",
                    )
                    return False

        logger.debug("Paper wallet consistency check passed")
        return True
    except Exception as exc:  # pragma: no cover - safety net
        logger.error("Error during paper wallet consistency check: %s", exc)
        return False


def ensure_paper_wallet_sync(ctx):
    """Ensure paper wallet is fully synchronized after all operations."""
    if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
        ctx.balance = ctx.paper_wallet.balance
        logger.debug("Paper wallet balance synchronized: $%.2f", ctx.balance)
        if hasattr(ctx, "sync_positions_from_trade_manager") and ctx.use_trade_manager_as_source:
            ctx.sync_positions_from_trade_manager()
            logger.debug("Positions synchronized from TradeManager")
        if not validate_paper_wallet_consistency(ctx):
            logger.warning("Paper wallet consistency issues detected - attempting recovery")
            ctx.balance = ctx.paper_wallet.balance
        status = get_paper_wallet_status(ctx)
        if status:
            logger.info(
                "Paper wallet status: Balance=$%s, PnL=$%s, Win Rate=%s",
                status["balance"],
                status["realized_pnl"],
                status["win_rate"],
            )
        return ctx.paper_wallet.balance
    return ctx.balance


async def sync_paper_wallet_with_positions_log(ctx):
    """Synchronize TradeManager, PaperWallet, and positions.log using SyncService."""
    if not ctx.config.get("execution_mode") == "dry_run" or not ctx.paper_wallet:
        return
    try:
        from crypto_bot.sync_service import ConflictResolution, SyncService

        sync_service = SyncService(LOG_DIR)
        positions_log = LOG_DIR / "positions.log"
        results = await sync_service.full_synchronization(
            trade_manager=ctx.trade_manager,
            paper_wallet=ctx.paper_wallet,
            positions_log_path=positions_log,
            conflict_resolution=ConflictResolution.TRADE_MANAGER_WINS,
        )
        log_to_tm = results.get("log_to_tm")
        tm_to_pw = results.get("tm_to_pw")
        if log_to_tm and log_to_tm.result.name == "SUCCESS":
            if log_to_tm.target_positions:
                logger.info("✅ Recovered %d positions from positions.log", len(log_to_tm.target_positions))
            if log_to_tm.conflicts:
                logger.warning("⚠️ Resolved %d conflicts during recovery", len(log_to_tm.conflicts))
        if tm_to_pw and tm_to_pw.result.name == "SUCCESS":
            logger.info(
                "✅ PaperWallet synchronized with TradeManager in %.1fms",
                tm_to_pw.duration_ms,
            )
        health = sync_service.get_health_status()
        logger.info(
            "🔄 Sync health: %s | Success rate: %s/%s",
            health["overall_health"],
            health["metrics"]["successful_syncs"],
            health["metrics"]["total_syncs"],
        )
    except Exception as exc:  # pragma: no cover - sync best effort
        logger.error("❌ Enterprise synchronization failed: %s", exc)
        try:
            positions_log = LOG_DIR / "positions.log"
            if positions_log.exists():
                with open(positions_log, "r", encoding="utf-8") as file:
                    lines = file.readlines()
                active_positions = sum(1 for line in lines if "Active" in line)
                logger.info("Found %d active positions in positions.log", active_positions)
        except Exception as fallback_error:  # pragma: no cover - logging best effort
            logger.error("Fallback logging also failed: %s", fallback_error)


def get_paper_wallet_status(ctx):
    """Get comprehensive status of paper wallet for monitoring."""
    if not ctx.config.get("execution_mode") == "dry_run" or not ctx.paper_wallet:
        return None

    summary = ctx.paper_wallet.get_position_summary()
    total_unrealized_pnl = 0.0
    for sym, pos_ctx in ctx.positions.items():
        if sym in ctx.paper_wallet.positions and hasattr(ctx, "df_cache") and sym in ctx.df_cache:
            df = ctx.df_cache[sym]
            if isinstance(df, pd.DataFrame) and not df.empty:
                current_price = float(df.iloc[-1]["close"])
                unrealized = ctx.paper_wallet.unrealized(sym, current_price)
                total_unrealized_pnl += unrealized

    status = {
        "balance": summary["balance"],
        "initial_balance": summary["initial_balance"],
        "realized_pnl": summary["realized_pnl"],
        "unrealized_pnl": total_unrealized_pnl,
        "total_pnl": summary["realized_pnl"] + total_unrealized_pnl,
        "total_trades": summary["total_trades"],
        "winning_trades": summary["winning_trades"],
        "win_rate": summary["win_rate"],
        "open_positions": summary["open_positions"],
        "positions": {},
    }

    for pid, pos in summary["positions"].items():
        unrealized_pnl = 0.0
        current_price = None
        df = ctx.df_cache.get(pid) if hasattr(ctx, "df_cache") else None
        if isinstance(df, pd.DataFrame) and not df.empty:
            current_price = float(df.iloc[-1]["close"])
            unrealized_pnl = ctx.paper_wallet.unrealized(pid, current_price)
        status["positions"][pid] = {
            "symbol": pos.get("symbol", "Unknown"),
            "side": pos["side"],
            "size": pos["size"],
            "entry_price": pos["entry_price"],
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
            "reserved": pos.get("reserved", 0.0),
        }

    return status
