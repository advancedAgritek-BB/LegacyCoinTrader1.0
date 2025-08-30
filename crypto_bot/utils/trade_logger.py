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
import fcntl
import os


logger = setup_logger(__name__, LOG_DIR / "execution.log")


def log_trade(order: Dict, is_stop: bool = False) -> None:
    """Append executed order details to a CSV and optionally Google Sheets.

    If ``is_stop`` is ``True`` the order is recorded as a stop placement rather
    than an executed trade.
    """
    try:
        # Validate order object
        if not order or not isinstance(order, dict):
            logger.error(f"Invalid order object: {order}")
            return
        
        # Check for required fields
        required_fields = ['symbol', 'side', 'amount']
        missing_fields = [field for field in required_fields if not order.get(field)]
        if missing_fields:
            logger.error(f"Order missing required fields {missing_fields}: {order}")
            return
        
        # Skip orders with zero amounts
        if order.get('amount', 0) <= 0:
            logger.warning(f"Skipping order with zero amount: {order}")
            return
        
        order = dict(order)
        ts = order.get("timestamp") or datetime.utcnow().isoformat()
        record = {
            "symbol": order.get("symbol", ""),
            "side": order.get("side", ""),
            "amount": order.get("amount", 0.0),
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

        df = pd.DataFrame([record])
        log_file = LOG_DIR / "trades.csv"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Use file locking to prevent concurrent access issues
        try:
            with open(log_file, 'a') as f:
                # Acquire an exclusive lock on the file
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                try:
                    # Append rows without a header so repeated logs don't duplicate columns
                    df.to_csv(f, mode="a", header=False, index=False)
                    f.flush()  # Ensure data is written to disk
                    os.fsync(f.fileno())  # Force sync to disk
                    
                    logger.info(f"Trade written to CSV: {record['symbol']} {record['side']} {record['amount']} @ {record['price']}")
                    
                finally:
                    # Release the lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    
        except Exception as file_error:
            logger.error(f"File write error: {file_error}")
            # Fallback: try direct string writing
            try:
                record_line = f"{record['symbol']},{record['side']},{record['amount']},{record['price']},{record['timestamp']},{record['is_stop']}\n"
                with open(log_file, 'a') as f:
                    f.write(record_line)
                    f.flush()
                logger.info(f"Trade written via fallback method: {record['symbol']} {record['side']} {record['amount']} @ {record['price']}")
            except Exception as fallback_error:
                logger.error(f"Fallback write also failed: {fallback_error}")
                return
        
        # Verify the write was successful
        if log_file.exists():
            file_size = log_file.stat().st_size
            logger.debug(f"Trade logged successfully to {log_file} (size: {file_size} bytes)")
        else:
            logger.error(f"Failed to create trades.csv file")
            return
        
        msg = "Stop order placed: %s" if is_stop else "Logged trade: %s"
        logger.info(msg, record)
        
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
        except Exception as e:
            logger.debug(f"Google Sheets logging failed: {e}")
            
    except Exception as e:
        logger.error(f"Error in log_trade: {e}")
        logger.error(f"Order that failed to log: {order}")
        # Try to log at least basic info to prevent complete loss
        try:
            basic_record = f"{order.get('symbol', 'UNKNOWN')},{order.get('side', 'UNKNOWN')},{order.get('amount', 0)},{order.get('price', 0)},{datetime.utcnow().isoformat()},False"
            log_file = LOG_DIR / "trades.csv"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'a') as f:
                f.write(basic_record + '\n')
            logger.info(f"Basic trade info logged as fallback: {basic_record}")
        except Exception as fallback_error:
            logger.error(f"Fallback logging also failed: {fallback_error}")
