#!/usr/bin/env python3
"""
SET STOP LOSSES ON ALL OPEN POSITIONS

This script automatically sets appropriate stop losses on all open positions
based on the bot's configuration and risk management rules.
"""

import asyncio
import logging
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
import sys
import json

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from crypto_bot.utils.trade_manager import TradeManager
from crypto_bot.utils.logger import setup_logger

logger = setup_logger(__name__, Path("crypto_bot/logs/stop_loss_setup.log"))

class StopLossSetter:
    """Set stop losses on all open positions."""

    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path("crypto_bot/config.yaml")
        self.config = self.load_config()
        self.trade_manager = TradeManager()

    def load_config(self):
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Extract relevant stop loss settings
            self.stop_loss_pct = Decimal(str(config.get('exits', {}).get('default_sl_pct', 0.008)))
            self.take_profit_pct = Decimal(str(config.get('exits', {}).get('default_tp_pct', 0.045)))
            self.trailing_stop_pct = Decimal(str(config.get('exits', {}).get('trailing_stop_pct', 0.008)))

            logger.info(f"Loaded stop loss settings:")
            logger.info(f"  Stop Loss: {self.stop_loss_pct * 100}%")
            logger.info(f"  Take Profit: {self.take_profit_pct * 100}%")
            logger.info(f"  Trailing Stop: {self.trailing_stop_pct * 100}%")

            return config

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Use default values
            self.stop_loss_pct = Decimal('0.008')  # 0.8%
            self.take_profit_pct = Decimal('0.045')  # 4.5%
            self.trailing_stop_pct = Decimal('0.008')  # 0.8%
            return {}

    def calculate_stop_loss_price(self, entry_price: Decimal, side: str) -> Decimal:
        """Calculate stop loss price for a position."""
        if side.lower() == 'long':
            return entry_price * (1 - self.stop_loss_pct)
        else:  # short
            return entry_price * (1 + self.stop_loss_pct)

    def calculate_take_profit_price(self, entry_price: Decimal, side: str) -> Decimal:
        """Calculate take profit price for a position."""
        if side.lower() == 'long':
            return entry_price * (1 + self.take_profit_pct)
        else:  # short
            return entry_price * (1 - self.take_profit_pct)

    def set_stop_losses_on_positions(self):
        """Set stop losses on all open positions."""
        logger.info("=" * 60)
        logger.info("SETTING STOP LOSSES ON OPEN POSITIONS")
        logger.info("=" * 60)

        # Get all open positions
        positions = self.trade_manager.get_all_positions()
        if not positions:
            logger.warning("No open positions found!")
            return 0

        logger.info(f"Found {len(positions)} open positions")

        updated_count = 0

        for position in positions:
            if not position.is_open:
                logger.debug(f"Skipping closed position: {position.symbol}")
                continue

            logger.info(f"\nProcessing position: {position.symbol}")
            logger.info(f"  Side: {position.side}")
            logger.info(f"  Size: {position.total_amount}")
            logger.info(f"  Entry Price: ${position.average_price}")

            # Check if stop loss is already set
            if position.stop_loss_price is not None:
                logger.info(f"  ✅ Stop Loss already set: ${position.stop_loss_price}")
            else:
                # Calculate and set stop loss
                stop_loss_price = self.calculate_stop_loss_price(position.average_price, position.side)
                position.stop_loss_price = stop_loss_price
                logger.info(f"  🆕 Set Stop Loss: ${stop_loss_price}")

            # Check if take profit is already set
            if position.take_profit_price is not None:
                logger.info(f"  ✅ Take Profit already set: ${position.take_profit_price}")
            else:
                # Calculate and set take profit
                take_profit_price = self.calculate_take_profit_price(position.average_price, position.side)
                position.take_profit_price = take_profit_price
                logger.info(f"  🆕 Set Take Profit: ${take_profit_price}")

            # Set trailing stop percentage if not set
            if position.trailing_stop_pct is None:
                position.trailing_stop_pct = self.trailing_stop_pct
                logger.info(f"  🆕 Set Trailing Stop %: {self.trailing_stop_pct * 100}%")

            # Update position in trade manager
            self.trade_manager.positions[position.symbol] = position
            updated_count += 1

            logger.info(f"  ✅ Position {position.symbol} updated successfully")

        # Save the updated state to disk
        logger.info("Saving updated positions to disk...")
        self.trade_manager.save_state()

        logger.info("\n" + "=" * 60)
        logger.info(f"SUCCESS: Updated {updated_count} positions with stop losses")
        logger.info("=" * 60)

        return updated_count

    def verify_stop_losses(self):
        """Verify that all positions now have stop losses set."""
        logger.info("\n" + "=" * 60)
        logger.info("VERIFYING STOP LOSS SETUP")
        logger.info("=" * 60)

        positions = self.trade_manager.get_all_positions()
        if not positions:
            logger.warning("No positions to verify!")
            return False

        all_have_stops = True
        verified_count = 0

        for position in positions:
            if not position.is_open:
                continue

            logger.info(f"\nVerifying {position.symbol}:")
            logger.info(f"  Side: {position.side}")
            logger.info(f"  Entry: ${position.average_price}")

            if position.stop_loss_price is not None:
                logger.info(f"  ✅ Stop Loss: ${position.stop_loss_price}")
                verified_count += 1
            else:
                logger.error(f"  ❌ Stop Loss: NOT SET")
                all_have_stops = False

            if position.take_profit_price is not None:
                logger.info(f"  ✅ Take Profit: ${position.take_profit_price}")
            else:
                logger.warning(f"  ⚠️  Take Profit: NOT SET")

            if position.trailing_stop_pct is not None:
                logger.info(f"  ✅ Trailing Stop: {position.trailing_stop_pct * 100}%")
            else:
                logger.warning(f"  ⚠️  Trailing Stop: NOT SET")

        logger.info("\n" + "=" * 60)
        if all_have_stops:
            logger.info(f"✅ VERIFICATION PASSED: All {verified_count} positions have stop losses set")
        else:
            logger.error("❌ VERIFICATION FAILED: Some positions missing stop losses")
        logger.info("=" * 60)

        return all_have_stops

    def generate_report(self):
        """Generate a comprehensive report of stop loss setup."""
        positions = self.trade_manager.get_all_positions()

        from datetime import datetime
        report = {
            "timestamp": str(datetime.utcnow()),
            "total_positions": len(positions),
            "open_positions": len([p for p in positions if p.is_open]),
            "positions_with_stops": len([p for p in positions if p.is_open and p.stop_loss_price is not None]),
            "positions_with_take_profit": len([p for p in positions if p.is_open and p.take_profit_price is not None]),
            "positions_with_trailing": len([p for p in positions if p.is_open and p.trailing_stop_pct is not None]),
            "stop_loss_settings": {
                "stop_loss_pct": float(self.stop_loss_pct),
                "take_profit_pct": float(self.take_profit_pct),
                "trailing_stop_pct": float(self.trailing_stop_pct)
            },
            "positions": []
        }

        for position in positions:
            if position.is_open:
                pos_data = {
                    "symbol": position.symbol,
                    "side": position.side,
                    "size": float(position.total_amount),
                    "entry_price": float(position.average_price),
                    "stop_loss_price": float(position.stop_loss_price) if position.stop_loss_price else None,
                    "take_profit_price": float(position.take_profit_price) if position.take_profit_price else None,
                    "trailing_stop_pct": float(position.trailing_stop_pct) if position.trailing_stop_pct else None,
                    "highest_price": float(position.highest_price) if position.highest_price else None,
                    "lowest_price": float(position.lowest_price) if position.lowest_price else None
                }
                report["positions"].append(pos_data)

        # Save report
        report_path = Path("stop_loss_setup_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to: {report_path}")
        return report

def main():
    """Main function to set stop losses on all positions."""
    print("🚀 STOP LOSS SETUP UTILITY")
    print("=" * 50)

    setter = StopLossSetter()

    try:
        # Set stop losses
        updated = setter.set_stop_losses_on_positions()

        if updated > 0:
            # Verify setup
            verification_passed = setter.verify_stop_losses()

            # Generate report
            report = setter.generate_report()

            print("\n" + "=" * 50)
            print("📊 SUMMARY:")
            print(f"  • Updated positions: {updated}")
            print(f"  • Verification: {'✅ PASSED' if verification_passed else '❌ FAILED'}")
            print(f"  • Report saved: stop_loss_setup_report.json")
            print("=" * 50)

            if verification_passed:
                print("🎉 SUCCESS: All open positions now have stop losses set!")
                print("   The bot will now properly monitor and execute stop losses.")
            else:
                print("⚠️  WARNING: Some positions may still be missing stop losses.")
                print("   Please check the logs for details.")
        else:
            print("⚠️  No positions were updated. Check the logs for details.")

    except Exception as e:
        logger.error(f"Error setting stop losses: {e}")
        print(f"❌ ERROR: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
