"""
Single Source of Truth for Balance Management

This module ensures all balance references use a single, consistent source
and prevents negative balance values throughout the system.
"""

from pathlib import Path
from typing import Optional
import yaml
import json
import logging

logger = logging.getLogger(__name__)

# Single source of truth: paper_wallet_state.yaml
BALANCE_STATE_FILE = Path("crypto_bot/logs/paper_wallet_state.yaml")


class BalanceManager:
    """Manages the single source of truth for wallet balance."""

    @staticmethod
    def get_balance() -> float:
        """Get the current balance from the single source of truth."""
        try:
            if not BALANCE_STATE_FILE.exists():
                logger.warning(f"Balance state file not found: {BALANCE_STATE_FILE}")
                return 10000.0

            with open(BALANCE_STATE_FILE, 'r') as f:
                state = yaml.safe_load(f) or {}

            balance = state.get('balance', 10000.0)

            # Ensure balance is never negative
            if balance < 0:
                logger.warning(f"Found negative balance ${balance:.2f} in state file, correcting to $0.00")
                BalanceManager.set_balance(0.0)
                return 0.0

            return max(0.0, balance)

        except Exception as e:
            logger.error(f"Error reading balance from state file: {e}")
            return 10000.0

    @staticmethod
    def set_balance(balance: float) -> None:
        """Set the balance in the single source of truth."""
        try:
            # Ensure balance is never negative
            safe_balance = max(0.0, balance)

            # Load existing state
            state = {}
            if BALANCE_STATE_FILE.exists():
                with open(BALANCE_STATE_FILE, 'r') as f:
                    state = yaml.safe_load(f) or {}

            # Update balance
            state['balance'] = safe_balance

            # Ensure directory exists
            BALANCE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Save updated state
            with open(BALANCE_STATE_FILE, 'w') as f:
                yaml.dump(state, f, default_flow_style=False)

            logger.info(f"Balance updated to ${safe_balance:.2f} in single source of truth")

        except Exception as e:
            logger.error(f"Error setting balance in state file: {e}")

    @staticmethod
    def synchronize_all_sources() -> None:
        """Synchronize all balance sources to use the single source of truth."""
        try:
            current_balance = BalanceManager.get_balance()

            # Update paper_wallet.yaml
            paper_wallet_file = Path("crypto_bot/logs/paper_wallet.yaml")
            if paper_wallet_file.exists():
                try:
                    with open(paper_wallet_file, 'r') as f:
                        config = yaml.safe_load(f) or {}

                    config['initial_balance'] = current_balance

                    with open(paper_wallet_file, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)

                    logger.info(f"Synchronized paper_wallet.yaml to ${current_balance:.2f}")
                except Exception as e:
                    logger.error(f"Error synchronizing paper_wallet.yaml: {e}")

            # Update user_config.yaml if it exists
            user_config_file = Path("crypto_bot/user_config.yaml")
            if user_config_file.exists():
                try:
                    with open(user_config_file, 'r') as f:
                        config = yaml.safe_load(f) or {}

                    config['paper_wallet_balance'] = current_balance

                    with open(user_config_file, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)

                    logger.info(f"Synchronized user_config.yaml to ${current_balance:.2f}")
                except Exception as e:
                    logger.error(f"Error synchronizing user_config.yaml: {e}")

            logger.info("All balance sources synchronized to single source of truth")

        except Exception as e:
            logger.error(f"Error synchronizing balance sources: {e}")

    @staticmethod
    def validate_balance_integrity() -> bool:
        """Validate that all balance sources are consistent."""
        try:
            current_balance = BalanceManager.get_balance()

            # Check paper_wallet.yaml
            paper_wallet_file = Path("crypto_bot/logs/paper_wallet.yaml")
            if paper_wallet_file.exists():
                with open(paper_wallet_file, 'r') as f:
                    config = yaml.safe_load(f) or {}
                    pw_balance = config.get('initial_balance', 10000.0)
                    if abs(pw_balance - current_balance) > 0.01:
                        logger.warning(f"Balance mismatch: paper_wallet.yaml=${pw_balance:.2f}, state=${current_balance:.2f}")
                        return False

            # Check user_config.yaml
            user_config_file = Path("crypto_bot/user_config.yaml")
            if user_config_file.exists():
                with open(user_config_file, 'r') as f:
                    config = yaml.safe_load(f) or {}
                    uc_balance = config.get('paper_wallet_balance', 10000.0)
                    if abs(uc_balance - current_balance) > 0.01:
                        logger.warning(f"Balance mismatch: user_config.yaml=${uc_balance:.2f}, state=${current_balance:.2f}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Error validating balance integrity: {e}")
            return False


# Global functions for easy access
def get_single_balance() -> float:
    """Get balance from the single source of truth."""
    return BalanceManager.get_balance()


def set_single_balance(balance: float) -> None:
    """Set balance in the single source of truth."""
    BalanceManager.set_balance(balance)


def synchronize_balance_sources() -> None:
    """Synchronize all balance sources."""
    BalanceManager.synchronize_all_sources()


def validate_balance_sources() -> bool:
    """Validate balance source consistency."""
    return BalanceManager.validate_balance_integrity()
