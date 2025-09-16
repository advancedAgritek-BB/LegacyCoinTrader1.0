from __future__ import annotations

"""Simple console monitor for runtime status."""

import asyncio
from pathlib import Path
from typing import Optional, Any, Union, List, Dict
import sys
import os

from rich.console import Console
from rich.table import Table

from crypto_bot.utils.logger import LOG_DIR
from crypto_bot.utils.logger import setup_logger

from .utils.open_trades import get_open_trades
from . import log_reader

# Constants
TRADE_FILE = LOG_DIR / "trades.csv"

# Local logger for this module
logger = setup_logger(__name__, str(LOG_DIR / "bot.log"))


async def monitor_loop(
    exchange: object,
    paper_wallet: Optional[object] = None,
    log_file: Union[str, Path] = LOG_DIR / "bot.log",
    trade_file: Union[str, Path] = TRADE_FILE,
) -> None:
    """Periodically output balance, last log line and open trade stats.

    This coroutine runs until cancelled and is intentionally lightweight so
    tests can easily patch it. The monitor fetches the current balance from
    ``exchange`` or ``paper_wallet`` and prints the last line of ``log_file``.
    Open trade PnL lines are generated from ``trade_file`` and printed below
    the status line when positions exist.
    """
    log_path = Path(log_file)
    last_line = ""
    prev_lines = 0
    prev_output = ""
    offset = 0

    try:
        with log_path.open("r", encoding="utf-8") as fh:
            while True:
                await asyncio.sleep(5)
                balance = None
                try:
                    if paper_wallet is not None:
                        balance = getattr(paper_wallet, "balance", None)
                    elif hasattr(exchange, "fetch_balance"):
                        # Try calling fetch_balance and check if it returns a
                        # coroutine
                        bal_result = exchange.fetch_balance()
                        if asyncio.iscoroutine(bal_result):
                            bal = await bal_result
                        else:
                            bal = bal_result
                        balance = (
                            bal.get("USDT", {}).get("free", 0)
                            if isinstance(bal.get("USDT"), dict)
                            else bal.get("USDT", 0)
                        )
                except Exception:
                    pass

                fh.seek(offset)
                for line in fh:
                    if "Loading config" not in line:
                        last_line = line.rstrip("\n")
                offset = fh.tell()

                message = f"[Monitor] balance={balance} log='{last_line}'"
                lines = await trade_stats_lines(exchange, Path(trade_file))

                output = message
                if lines:
                    output += "\n" + "\n".join(lines)

                simple_console = bool(os.environ.get("FORCE_SIMPLE_CONSOLE"))
                if sys.stdout.isatty() and not simple_console:
                    # Clear previously printed lines
                    if prev_lines:
                        print("\033[2K", end="")
                        for _ in range(prev_lines - 1):
                            print("\033[F\033[2K", end="")
                    print(output, end="\r", flush=True)
                    prev_lines = output.count("\n") + 1
                    prev_output = output
                else:
                    if output != prev_output:
                        print(output)
                        prev_output = output
    except asyncio.CancelledError:
        # Propagate cancellation after the file handle is closed by the
        # context manager.
        raise


def display_trades(
    exchange: Optional[Any] = None, wallet: Optional[Any] = None, trade_file: Path = TRADE_FILE
) -> str:
    """Display trades from TradeManager and print them as a table.

    Returns the rendered table as text so tests can verify the output.
    """
    console = Console(record=True)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Symbol", style="bold cyan")
    table.add_column("Side", justify="center")
    table.add_column("Amount", justify="right")
    table.add_column("Entry Price", justify="right")
    table.add_column("Unrealized P&L", justify="right")

    # Try to get trades from TradeManager first
    try:
        from crypto_bot.utils.trade_manager import get_trade_manager
        trade_manager = get_trade_manager()

        # Get all positions from TradeManager
        positions = trade_manager.get_all_positions()
        
        # Filter to only show open positions with non-zero amounts
        open_positions = [
            pos for pos in positions 
            if pos.is_open and pos.total_amount > 0
        ]
        
        if open_positions:
            print(f"\n📊 Open Positions ({len(open_positions)} active):")
            for pos in open_positions:
                # Get current price for P&L calculation
                current_price = trade_manager.price_cache.get(
                    pos.symbol, pos.average_price
                )
                
                # Calculate unrealized P&L
                unrealized_pnl, unrealized_pct = (
                    pos.calculate_unrealized_pnl(current_price)
                )
                
                # Format P&L with color
                pnl_color = "green" if unrealized_pnl >= 0 else "red"
                pnl_sign = "+" if unrealized_pnl >= 0 else ""
                
                pnl_display = (
                    f"[{pnl_color}]{pnl_sign}${unrealized_pnl:.2f} "
                    f"({pnl_sign}{unrealized_pct:.2f}%)[/{pnl_color}]"
                )
                
                table.add_row(
                    pos.symbol,
                    pos.side.upper(),
                    f"{pos.total_amount:.6f}",
                    f"${pos.average_price:.4f}",
                    pnl_display
                )
            
            
            console.print(table)
            return console.export_text()
        else:
            print("\n📊 No open positions found.")
            return "No open positions found."

    except Exception as e:
        logger.warning(f"Failed to get trades from TradeManager: {e}, falling back to CSV")

    # Fallback to CSV-based display (deprecated)
    try:
        df = log_reader._read_trades(trade_file)

        for _, row in df.iterrows():
            status = row.get("status", "unknown")
            table.add_row(
                str(row.get("symbol", "")),
                str(row.get("side", "")),
                str(row.get("amount", "")),
                str(row.get("price", "")),
                status
            )
    except Exception as e:
        logger.error(f"Failed to read trades from CSV: {e}")
        table.add_row("ERROR", "reading", "trades", "data", "failed")

    console.print(table)
    return console.export_text()


async def trade_stats_lines(exchange: Any, trade_file: Path = TRADE_FILE) -> List[str]:
    """Return a list of lines summarizing PnL for each open trade using TradeManager."""
    # Try to get positions from TradeManager first
    try:
        from crypto_bot.utils.trade_manager import get_trade_manager
        trade_manager = get_trade_manager()

        positions = trade_manager.get_all_positions()
        if not positions:
            return []

        lines = []

        for pos in positions:
            if not pos.is_open:
                continue

            # Get current price from TradeManager's cache or exchange
            current_price = float(trade_manager.price_cache.get(pos.symbol, pos.average_price))

            # If price is not in cache, try to fetch from exchange
            if current_price <= 0:
                try:
                    ticker_result = exchange.fetch_ticker(pos.symbol)
                    if asyncio.iscoroutine(ticker_result):
                        ticker = await ticker_result
                    else:
                        ticker = ticker_result
                    current_price = float(ticker.get("last") or ticker.get("close") or 0.0)

                    # Update TradeManager's price cache
                    from decimal import Decimal
                    trade_manager.update_price(pos.symbol, Decimal(str(current_price)))
                except Exception:
                    # Fallback to shared price fetching utility
                    try:
                        from crypto_bot.utils.price_fetcher import get_current_price_for_symbol
                        current_price = get_current_price_for_symbol(pos.symbol)
                        if current_price > 0:
                            trade_manager.update_price(pos.symbol, Decimal(str(current_price)))
                    except Exception:
                        current_price = 0.0

            # Calculate P&L using TradeManager's method
            if current_price > 0:
                from decimal import Decimal
                pnl, _ = pos.calculate_unrealized_pnl(Decimal(str(current_price)))
                pnl_value = float(pnl)
            else:
                pnl_value = 0.0

            lines.append(f"{pos.symbol} -- {pos.average_price:.2f} -- {pnl_value:+.2f}")

        return lines

    except Exception as e:
        logger.warning(f"Failed to get trade stats from TradeManager: {e}")
        # If TradeManager state exists, do not use CSV fallback to avoid stale/phantom trades
        try:
            state_file = LOG_DIR / "trade_manager_state.json"
            if state_file.exists():
                return []
        except Exception:
            pass

    # Fallback to CSV-based calculation (deprecated)
    try:
        open_trades = get_open_trades(trade_file)
        if not open_trades:
            return []

        symbols = {t["symbol"] for t in open_trades}
        prices: Dict[str, float] = {}

        for sym in symbols:
            try:
                ticker_result = exchange.fetch_ticker(sym)
                if asyncio.iscoroutine(ticker_result):
                    ticker = await ticker_result
                else:
                    ticker = ticker_result
                current_price = float(ticker.get("last") or ticker.get("close") or 0.0)
                prices[sym] = current_price
            except Exception:
                try:
                    from frontend.app import get_current_price_for_symbol
                    current_price = get_current_price_for_symbol(sym)
                    prices[sym] = current_price
                except Exception:
                    prices[sym] = 0.0

        lines = []
        for trade in open_trades:
            sym = trade.get("symbol")
            entry = float(trade.get("price", 0))
            amount = float(trade.get("amount", 0))
            side = trade.get("side", "long")
            current = prices.get(sym, 0.0)

            if side == "short":
                pnl_value = (entry - current) * amount
            else:
                pnl_value = (current - entry) * amount

            lines.append(f"{sym} -- {entry:.2f} -- {pnl_value:+.2f}")
        return lines
    except Exception as e:
        logger.error(f"Failed to get trade stats from CSV: {e}")
        return []


async def trade_stats_line(exchange: Any, trade_file: Path = TRADE_FILE) -> str:
    """Return a single line summarizing PnL for each open trade."""
    lines = await trade_stats_lines(exchange, trade_file)
    return " | ".join(lines)
