"""
Wallet Synchronization Utility

Provides comprehensive synchronization between trading context and paper wallet
to resolve position count mismatches and ensure data consistency.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .logger import LOG_DIR, setup_logger

logger = setup_logger(__name__, LOG_DIR / "wallet_sync_utility.log")


class WalletSyncUtility:
    """Utility for synchronizing wallet and trading context positions."""

    def __init__(self, trade_manager=None, position_sync_manager=None):
        self.trade_manager = trade_manager
        self.position_sync_manager = position_sync_manager

    def detect_sync_issues(self, ctx) -> Dict[str, Any]:
        """Detect synchronization issues between context and paper wallet."""
        issues = {
            "has_issues": False,
            "ctx_positions_count": 0,
            "wallet_positions_count": 0,
            "mismatched_symbols": [],
            "missing_in_context": [],
            "missing_in_wallet": [],
            "quantity_mismatches": [],
            "recommendations": []
        }

        try:
            # Get position counts
            ctx_positions = ctx.positions if hasattr(ctx, 'positions') else {}
            wallet_positions = ctx.paper_wallet.positions if ctx.paper_wallet else {}

            issues["ctx_positions_count"] = len(ctx_positions)
            issues["wallet_positions_count"] = len(wallet_positions)

            # Check for count mismatch
            if len(ctx_positions) != len(wallet_positions):
                issues["has_issues"] = True
                issues["recommendations"].append("Position count mismatch detected")

            # Check for symbol mismatches
            ctx_symbols = set(ctx_positions.keys())
            wallet_symbols = set(wallet_positions.keys())

            missing_in_context = wallet_symbols - ctx_symbols
            missing_in_wallet = ctx_symbols - wallet_symbols

            if missing_in_context:
                issues["has_issues"] = True
                issues["missing_in_context"] = list(missing_in_context)
                issues["recommendations"].append(f"Symbols missing in context: {missing_in_context}")

            if missing_in_wallet:
                issues["has_issues"] = True
                issues["missing_in_wallet"] = list(missing_in_wallet)
                issues["recommendations"].append(f"Symbols missing in wallet: {missing_in_wallet}")

            # Check quantity mismatches for common symbols
            common_symbols = ctx_symbols & wallet_symbols
            for symbol in common_symbols:
                ctx_pos = ctx_positions.get(symbol, {})
                wallet_pos = wallet_positions.get(symbol, {})

                ctx_qty = ctx_pos.get('quantity', 0) if isinstance(ctx_pos, dict) else getattr(ctx_pos, 'quantity', 0)
                wallet_qty = wallet_pos.get('quantity', 0) if isinstance(wallet_pos, dict) else getattr(wallet_pos, 'quantity', 0)

                if abs(ctx_qty - wallet_qty) > 0.0001:  # Allow for small floating point differences
                    issues["has_issues"] = True
                    issues["quantity_mismatches"].append({
                        "symbol": symbol,
                        "ctx_quantity": ctx_qty,
                        "wallet_quantity": wallet_qty
                    })

            if issues["quantity_mismatches"]:
                issues["recommendations"].append(f"Quantity mismatches found for {len(issues['quantity_mismatches'])} symbols")

        except Exception as e:
            logger.error(f"Error detecting sync issues: {e}")
            issues["has_issues"] = True
            issues["recommendations"].append(f"Error during sync detection: {e}")

        return issues

    def auto_sync_positions(self, ctx) -> Tuple[bool, str]:
        """Automatically synchronize positions between context and wallet."""
        try:
            # Detect issues first
            issues = self.detect_sync_issues(ctx)

            if not issues["has_issues"]:
                logger.info("No synchronization issues detected")
                return True, "No issues found - positions are synchronized"

            logger.info(f"Detected sync issues: {issues}")

            # Use TradeManager as source of truth if available
            if self.trade_manager and hasattr(self.trade_manager, 'get_all_positions'):
                logger.info("Using TradeManager as source of truth for sync")
                return self._sync_from_trade_manager(ctx)

            # Fallback to paper wallet as source of truth
            elif ctx.paper_wallet:
                logger.info("Using paper wallet as source of truth for sync")
                return self._sync_from_paper_wallet(ctx)

            else:
                return False, "No reliable source of truth available for synchronization"

        except Exception as e:
            logger.error(f"Error during auto-sync: {e}")
            return False, f"Auto-sync failed: {e}"

    def _sync_from_trade_manager(self, ctx) -> Tuple[bool, str]:
        """Sync positions using TradeManager as source of truth."""
        try:
            tm_positions = self.trade_manager.get_all_positions()

            # Update context positions
            ctx.positions = {}
            if ctx.paper_wallet:
                ctx.paper_wallet.positions = {}

            for pos in tm_positions:
                # Convert to context format
                ctx_pos = {
                    'symbol': pos.symbol,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'pnl': getattr(pos, 'pnl', 0),
                    'timestamp': getattr(pos, 'timestamp', time.time())
                }
                ctx.positions[pos.symbol] = ctx_pos

                # Update paper wallet if available
                if ctx.paper_wallet:
                    ctx.paper_wallet.positions[pos.symbol] = ctx_pos

            logger.info(f"Successfully synced {len(tm_positions)} positions from TradeManager")
            return True, f"Synchronized {len(tm_positions)} positions from TradeManager"

        except Exception as e:
            logger.error(f"Error syncing from TradeManager: {e}")
            return False, f"TradeManager sync failed: {e}"

    def _sync_from_paper_wallet(self, ctx) -> Tuple[bool, str]:
        """Sync positions using paper wallet as source of truth."""
        try:
            if not ctx.paper_wallet:
                return False, "No paper wallet available"

            wallet_positions = ctx.paper_wallet.positions

            # Update context positions
            ctx.positions = dict(wallet_positions)

            logger.info(f"Successfully synced {len(wallet_positions)} positions from paper wallet")
            return True, f"Synchronized {len(wallet_positions)} positions from paper wallet"

        except Exception as e:
            logger.error(f"Error syncing from paper wallet: {e}")
            return False, f"Paper wallet sync failed: {e}"

    def force_balance_reset(self, ctx) -> Tuple[bool, str]:
        """Force a complete balance reset and synchronization."""
        try:
            logger.warning("Performing force balance reset...")

            # Clear all positions
            ctx.positions = {}
            if ctx.paper_wallet:
                ctx.paper_wallet.positions = {}

            # Reset any cached balances
            if hasattr(ctx, 'balances'):
                ctx.balances = {}

            logger.info("Force balance reset completed")
            return True, "Force balance reset completed successfully"

        except Exception as e:
            logger.error(f"Error during force balance reset: {e}")
            return False, f"Force balance reset failed: {e}"

    def validate_sync_status(self, ctx) -> Tuple[bool, str]:
        """Validate that synchronization was successful."""
        try:
            issues = self.detect_sync_issues(ctx)

            if not issues["has_issues"]:
                return True, "Synchronization validation passed"

            return False, f"Synchronization validation failed: {issues['recommendations']}"

        except Exception as e:
            logger.error(f"Error during sync validation: {e}")
            return False, f"Sync validation error: {e}"


def auto_fix_wallet_sync(ctx) -> Tuple[bool, str]:
    """Convenience function to automatically fix wallet synchronization issues."""
    utility = WalletSyncUtility()

    # First try auto-sync
    success, message = utility.auto_sync_positions(ctx)
    if success:
        # Validate the sync
        valid, validation_msg = utility.validate_sync_status(ctx)
        if valid:
            return True, f"{message} - Validation passed"
        else:
            return False, f"{message} - {validation_msg}"

    # If auto-sync fails, try force reset as last resort
    logger.warning("Auto-sync failed, attempting force balance reset...")
    reset_success, reset_message = utility.force_balance_reset(ctx)

    if reset_success:
        return True, f"Force reset successful: {reset_message}"
    else:
        return False, f"All sync methods failed: {message}, {reset_message}"
