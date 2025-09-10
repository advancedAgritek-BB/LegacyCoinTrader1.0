"""
Synchronization Manager for Wallet Balance Components

Ensures consistent state between TradeManager, Paper Wallet, and Balance Manager.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class SynchronizationManager:
    """Manages synchronization between wallet components."""

    def __init__(self):
        self.last_sync_time = None
        self.sync_interval = 60  # seconds

    def sync_all_components(self, ctx=None) -> bool:
        """
        Synchronize all wallet components.
        Returns True if sync was successful.
        """
        try:
            if ctx is None:
                logger.warning("No context provided for synchronization")
                return False

            # Get TradeManager instance
            from crypto_bot.utils.trade_manager import get_trade_manager
            tm = get_trade_manager()

            if ctx.config.get("execution_mode") == "dry_run" and ctx.paper_wallet:
                # Calculate portfolio value from TradeManager positions
                tm_positions = tm.get_all_positions()
                portfolio_value = ctx.paper_wallet.initial_balance

                for pos in tm_positions:
                    current_price = tm.price_cache.get(pos.symbol, pos.average_price)
                    if pos.side == 'long':
                        portfolio_value += (current_price - pos.average_price) * pos.total_amount
                    else:  # short
                        portfolio_value += (pos.average_price - current_price) * pos.total_amount
                    portfolio_value -= pos.fees_paid

                # Update paper wallet balance
                old_balance = ctx.paper_wallet.balance
                ctx.paper_wallet.balance = portfolio_value
                ctx.balance = portfolio_value

                # Update balance manager
                try:
                    from crypto_bot.utils.balance_manager import set_single_balance
                    set_single_balance(portfolio_value)
                except Exception as e:
                    logger.warning(f"Failed to update balance manager: {e}")

                if abs(old_balance - portfolio_value) > 0.01:
                    logger.info(f"🔄 Balance synchronized: ${old_balance:.2f} → ${portfolio_value:.2f}")

                self.last_sync_time = datetime.utcnow()
                return True

        except Exception as e:
            logger.error(f"Error during component synchronization: {e}")
            return False

# Global instance
_sync_manager = SynchronizationManager()

def get_sync_manager() -> SynchronizationManager:
    """Get the global synchronization manager instance."""
    return _sync_manager
