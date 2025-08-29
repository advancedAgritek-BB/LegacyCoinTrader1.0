# Remove all imports that cause circular dependencies
# These will be imported on-demand when needed

# from .telegram_bot_ui import TelegramBotUI
# from .telegram_ctl import BotController
# from .log_reader import trade_summary
# from .console_monitor import monitor_loop, display_trades
# from .console_control import control_loop
# from .grid_state import update_bar, record_fill, in_cooldown, active_leg_count
# from .grid_center_model import predict_centre
# from .tax_logger import record_entry, record_exit, export_csv
# from .portfolio_rotator import PortfolioRotator
# from .volatility_filter import calc_atr, too_flat, too_hot, fetch_funding_rate
# from .fund_manager import auto_convert_funds
# from .strategy import (
#     grid_bot,
#     trend_bot,
#     breakout_bot,
#     sniper_bot,
#     mean_bot,
#     dca_bot,
#     bounce_scalper,
#     solana_scalping,
# )
# from .strategy_router import strategy_for, route, RouterConfig
# from .phase_runner import PhaseRunner, BotContext
# from .paper_wallet import PaperWallet

# Only import basic utilities that don't have circular dependencies
from .utils.logger import LOG_DIR, setup_logger
