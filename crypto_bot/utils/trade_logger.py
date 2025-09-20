import pandas as pd
from typing import Dict
from datetime import datetime
from dotenv import dotenv_values
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
except Exception:  # pragma: no cover - make Google Sheets optional
    gspread = None
    ServiceAccountCredentials = None
from crypto_bot.utils.logger import LOG_DIR, setup_logger
from crypto_bot.utils.single_source_trade_manager import get_single_source_trade_manager, create_trade
import fcntl
import os
import shutil


logger = setup_logger(__name__, LOG_DIR / "execution.log")


def log_trade(order: Dict, is_stop: bool = False) -> None:
    """
    Log trade to Google Sheets only.

    This function now only handles secondary logging (Google Sheets) since:
    - Primary trade storage is handled by SingleSourceTradeManager
    - CSV audit logging is handled by SingleSourceTradeManager
    - All components read from the single source of truth

    If ``is_stop`` is ``True`` the order is recorded as a stop placement.
    """
    try:
        # Validate order object
        if not order or not isinstance(order, dict):
            logger.debug(f"Invalid order object (skipping): {order}")
            return

        # Sanitize and validate required fields
        symbol = order.get('symbol')
        side = order.get('side')
        amount = order.get('amount', 0)

        # Handle missing or invalid symbol
        if not symbol or symbol is None or not isinstance(symbol, str):
            logger.debug(f"Order missing or invalid symbol (skipping): {order}")
            return

        # Handle missing or invalid side
        if not side or side not in ['buy', 'sell']:
            logger.debug(f"Order missing or invalid side (skipping): {order}")
            return

        # Handle zero or negative amounts
        if amount is None or amount <= 0:
            logger.debug(f"Order with zero/negative amount (skipping): {order}")
            return

        # Create record for Google Sheets logging
        ts = order.get("timestamp") or datetime.utcnow().isoformat()
        record = {
            "symbol": symbol,
            "side": side,
            "amount": float(amount),
            "price": order.get("price") or order.get("average") or 0.0,
            "timestamp": ts,
            "is_stop": is_stop,
        }
        if is_stop:
            record["stop_price"] = order.get("stop") or order.get("stop_price") or 0.0

        # Validate record data
        if not record["symbol"] or not record["side"]:
            logger.error(f"Invalid record data: {record}")
            return

        # Log to Google Sheets (only secondary logging now)
        try:
            if gspread and ServiceAccountCredentials:
                creds_path = dotenv_values('crypto_bot/.env').get('GOOGLE_CRED_JSON')
                if creds_path:
                    scope = ['https://spreadsheets.google.com/feeds',
                             'https://www.googleapis.com/auth/drive']
                    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
                    client = gspread.authorize(creds)
                    sheet = client.open('trade_logs').sheet1
                    sheet.append_row([record[k] for k in ["symbol", "side", "amount", "price", "timestamp"]])
                    logger.debug(f"Trade logged to Google Sheets: {record['symbol']} {record['side']} {record['amount']} @ {record['price']}")
        except Exception as e:
            logger.debug(f"Google Sheets logging failed: {e}")

        msg = "Stop order placed: %s" if is_stop else "Trade logged to secondary systems: %s"
        logger.info(msg, record)

    except Exception as e:
        logger.error(f"Error in log_trade: {e}")
        logger.error(f"Order that failed to log: {order}")


def recover_trades_from_backup() -> bool:
    """Attempt to recover trades from backup file."""
    log_file = LOG_DIR / "trades.csv"
    backup_file = LOG_DIR / "trades_backup.csv"
    
    if backup_file.exists() and backup_file.stat().st_size > 0:
        try:
            shutil.copy2(backup_file, log_file)
            logger.info(f"Recovered trades from backup: {backup_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to recover from backup: {e}")
            return False
    return False


def validate_trades_file() -> bool:
    """Validate the trades.csv file and attempt to fix issues."""
    log_file = LOG_DIR / "trades.csv"
    
    if not log_file.exists():
        logger.warning("trades.csv file does not exist")
        return False
    
    try:
        # Try to read the file to check for corruption
        df = pd.read_csv(log_file)
        logger.info(f"trades.csv is valid, contains {len(df)} trades")
        return True
    except Exception as e:
        logger.error(f"trades.csv appears to be corrupted: {e}")
        # Try to recover from backup
        if recover_trades_from_backup():
            return True
        return False


def periodic_backup_and_validate():
    """Periodic backup and validation of trades file."""
    log_file = LOG_DIR / "trades.csv"
    backup_file = LOG_DIR / "trades_backup.csv"
    
    if not log_file.exists():
        logger.warning("trades.csv does not exist for backup")
        return False
    
    try:
        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_backup = LOG_DIR / f"trades_backup_{timestamp}.csv"
        
        # Create backup
        shutil.copy2(log_file, timestamped_backup)
        shutil.copy2(log_file, backup_file)  # Also update the main backup
        
        # Validate the file
        df = pd.read_csv(log_file)
        logger.info(f"Periodic backup created: {timestamped_backup} ({len(df)} trades)")
        
        # Keep only the last 5 timestamped backups
        backup_files = sorted(LOG_DIR.glob("trades_backup_*.csv"))
        if len(backup_files) > 5:
            for old_backup in backup_files[:-5]:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup}")
        
        return True
        
    except Exception as e:
        logger.error(f"Periodic backup failed: {e}")
        return False


def get_trade_summary() -> Dict:
    """Get a summary of trading activity."""
    log_file = LOG_DIR / "trades.csv"
    
    if not log_file.exists():
        return {"error": "trades.csv not found"}
    
    try:
        df = pd.read_csv(log_file)
        
        if df.empty:
            return {"message": "No trades found"}
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        summary = {
            "total_trades": len(df),
            "date_range": {
                "start": df['timestamp'].min().isoformat() if not df['timestamp'].isna().all() else None,
                "end": df['timestamp'].max().isoformat() if not df['timestamp'].isna().all() else None
            },
            "symbols": df['symbol'].unique().tolist(),
            "total_volume": float(df['amount'].sum()),
            "buy_trades": len(df[df['side'] == 'buy']),
            "sell_trades": len(df[df['side'] == 'sell']),
            "file_size_bytes": log_file.stat().st_size,
            "last_modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
        }
        
        return summary
        
    except Exception as e:
        return {"error": f"Failed to read trades.csv: {e}"}
