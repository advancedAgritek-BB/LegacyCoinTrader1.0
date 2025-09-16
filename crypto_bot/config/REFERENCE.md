# Configuration Reference

The settings below are loaded by ``crypto_bot.config.settings.BotSettings``.
Environment variables use the ``BOT_`` prefix and ``__`` for nested fields (e.g. ``BOT_RISK__MAX_POSITIONS``).

## allow_short
- Type: `bool`
- Default: `True`

## bounce_scalper
### enabled
- Type: `bool`
- Default: `True`
- Description: Whether the strategy is active

### max_concurrent_signals
- Type: `int`
- Default: `20`
- Description: Bounce scalper maximum simultaneous signals

### min_score
- Type: `float`
- Default: `0.01`
- Description: Minimum bounce score

### timeframe
- Type: `Optional[str]`
- Default: `None`
- Description: Primary timeframe used by the strategy

### stop_loss_pct
- Type: `Optional[float]`
- Default: `None`
- Description: Stop loss expressed as a percentage of entry

### take_profit_pct
- Type: `Optional[float]`
- Default: `None`
- Description: Take profit expressed as a percentage of entry

### trailing_stop_enabled
- Type: `Optional[bool]`
- Default: `None`
- Description: Enable trailing stop for the strategy

### trailing_stop_pct
- Type: `Optional[float]`
- Default: `None`
- Description: Trailing stop distance in percent

### ema_fast
- Type: `Optional[int]`
- Default: `None`
- Description: Fast EMA period used by some strategies

### ema_slow
- Type: `Optional[int]`
- Default: `None`
- Description: Slow EMA period used by some strategies

### confirmation_timeframes
- Type: `Optional[List[str]]`
- Default: `None`
- Description: Additional timeframes used for confirmation

### momentum_threshold
- Type: `Optional[float]`
- Default: `None`
- Description: Minimum momentum score required

### volume_multiplier
- Type: `Optional[float]`
- Default: `None`
- Description: Multiplier applied to base volume requirements

### gas_threshold_gwei
- Type: `Optional[float]`
- Default: `None`
- Description: Maximum gas threshold for on-chain trades

### queue_timeout_ms
- Type: `Optional[int]`
- Default: `None`
- Description: Timeout for high-frequency trade queue

### atr_threshold
- Type: `Optional[float]`
- Default: `None`
- Description: ATR threshold used to filter volatile regimes

### max_spread_bp
- Type: `Optional[float]`
- Default: `None`
- Description: Maximum allowed spread in basis points

## ultra_scalp_bot
### enabled
- Type: `bool`
- Default: `True`
- Description: Whether the strategy is active

### max_concurrent_signals
- Type: `int`
- Default: `25`

### min_score
- Type: `float`
- Default: `0.08`

### timeframe
- Type: `str`
- Default: `1m`
- Description: Execution timeframe

### stop_loss_pct
- Type: `float`
- Default: `0.005`

### take_profit_pct
- Type: `float`
- Default: `0.02`

### trailing_stop_enabled
- Type: `bool`
- Default: `True`

### trailing_stop_pct
- Type: `float`
- Default: `0.008`

### ema_fast
- Type: `Optional[int]`
- Default: `None`
- Description: Fast EMA period used by some strategies

### ema_slow
- Type: `Optional[int]`
- Default: `None`
- Description: Slow EMA period used by some strategies

### confirmation_timeframes
- Type: `Optional[List[str]]`
- Default: `None`
- Description: Additional timeframes used for confirmation

### momentum_threshold
- Type: `Optional[float]`
- Default: `None`
- Description: Minimum momentum score required

### volume_multiplier
- Type: `Optional[float]`
- Default: `None`
- Description: Multiplier applied to base volume requirements

### gas_threshold_gwei
- Type: `Optional[float]`
- Default: `None`
- Description: Maximum gas threshold for on-chain trades

### queue_timeout_ms
- Type: `Optional[int]`
- Default: `None`
- Description: Timeout for high-frequency trade queue

### atr_threshold
- Type: `Optional[float]`
- Default: `None`
- Description: ATR threshold used to filter volatile regimes

### max_spread_bp
- Type: `Optional[float]`
- Default: `None`
- Description: Maximum allowed spread in basis points

## micro_scalp_bot
### enabled
- Type: `bool`
- Default: `True`
- Description: Whether the strategy is active

### max_concurrent_signals
- Type: `int`
- Default: `20`

### min_score
- Type: `float`
- Default: `0.06`

### timeframe
- Type: `str`
- Default: `1m`

### stop_loss_pct
- Type: `float`
- Default: `0.006`

### take_profit_pct
- Type: `float`
- Default: `0.024`

### trailing_stop_enabled
- Type: `Optional[bool]`
- Default: `None`
- Description: Enable trailing stop for the strategy

### trailing_stop_pct
- Type: `Optional[float]`
- Default: `None`
- Description: Trailing stop distance in percent

### ema_fast
- Type: `int`
- Default: `3`

### ema_slow
- Type: `int`
- Default: `8`

### confirmation_timeframes
- Type: `List[str]`
- Default: `["5m", "15m"]`
- Description: Additional timeframes for confirmation

### momentum_threshold
- Type: `Optional[float]`
- Default: `None`
- Description: Minimum momentum score required

### volume_multiplier
- Type: `Optional[float]`
- Default: `None`
- Description: Multiplier applied to base volume requirements

### gas_threshold_gwei
- Type: `Optional[float]`
- Default: `None`
- Description: Maximum gas threshold for on-chain trades

### queue_timeout_ms
- Type: `Optional[int]`
- Default: `None`
- Description: Timeout for high-frequency trade queue

### atr_threshold
- Type: `Optional[float]`
- Default: `None`
- Description: ATR threshold used to filter volatile regimes

### max_spread_bp
- Type: `Optional[float]`
- Default: `None`
- Description: Maximum allowed spread in basis points

## momentum_exploiter
### enabled
- Type: `bool`
- Default: `True`
- Description: Whether the strategy is active

### max_concurrent_signals
- Type: `int`
- Default: `15`

### min_score
- Type: `float`
- Default: `0.07`

### timeframe
- Type: `str`
- Default: `1m`

### stop_loss_pct
- Type: `float`
- Default: `0.008`

### take_profit_pct
- Type: `float`
- Default: `0.032`

### trailing_stop_enabled
- Type: `Optional[bool]`
- Default: `None`
- Description: Enable trailing stop for the strategy

### trailing_stop_pct
- Type: `Optional[float]`
- Default: `None`
- Description: Trailing stop distance in percent

### ema_fast
- Type: `Optional[int]`
- Default: `None`
- Description: Fast EMA period used by some strategies

### ema_slow
- Type: `Optional[int]`
- Default: `None`
- Description: Slow EMA period used by some strategies

### confirmation_timeframes
- Type: `Optional[List[str]]`
- Default: `None`
- Description: Additional timeframes used for confirmation

### momentum_threshold
- Type: `float`
- Default: `0.015`

### volume_multiplier
- Type: `float`
- Default: `1.5`

### gas_threshold_gwei
- Type: `Optional[float]`
- Default: `None`
- Description: Maximum gas threshold for on-chain trades

### queue_timeout_ms
- Type: `Optional[int]`
- Default: `None`
- Description: Timeout for high-frequency trade queue

### atr_threshold
- Type: `Optional[float]`
- Default: `None`
- Description: ATR threshold used to filter volatile regimes

### max_spread_bp
- Type: `Optional[float]`
- Default: `None`
- Description: Maximum allowed spread in basis points

## dex_scalper
### enabled
- Type: `bool`
- Default: `False`

### max_concurrent_signals
- Type: `int`
- Default: `25`

### min_score
- Type: `float`
- Default: `0.03`

### timeframe
- Type: `str`
- Default: `1m`

### stop_loss_pct
- Type: `float`
- Default: `0.004`

### take_profit_pct
- Type: `float`
- Default: `0.01`

### trailing_stop_enabled
- Type: `Optional[bool]`
- Default: `None`
- Description: Enable trailing stop for the strategy

### trailing_stop_pct
- Type: `Optional[float]`
- Default: `None`
- Description: Trailing stop distance in percent

### ema_fast
- Type: `int`
- Default: `3`

### ema_slow
- Type: `int`
- Default: `8`

### confirmation_timeframes
- Type: `Optional[List[str]]`
- Default: `None`
- Description: Additional timeframes used for confirmation

### momentum_threshold
- Type: `Optional[float]`
- Default: `None`
- Description: Minimum momentum score required

### volume_multiplier
- Type: `Optional[float]`
- Default: `None`
- Description: Multiplier applied to base volume requirements

### gas_threshold_gwei
- Type: `float`
- Default: `15.0`

### queue_timeout_ms
- Type: `Optional[int]`
- Default: `None`
- Description: Timeout for high-frequency trade queue

### atr_threshold
- Type: `Optional[float]`
- Default: `None`
- Description: ATR threshold used to filter volatile regimes

### max_spread_bp
- Type: `Optional[float]`
- Default: `None`
- Description: Maximum allowed spread in basis points

## hft_engine
### enabled
- Type: `bool`
- Default: `False`

### max_concurrent_signals
- Type: `int`
- Default: `50`

### min_score
- Type: `float`
- Default: `0.02`

### timeframe
- Type: `str`
- Default: `1m`

### stop_loss_pct
- Type: `float`
- Default: `0.002`

### take_profit_pct
- Type: `float`
- Default: `0.006`

### trailing_stop_enabled
- Type: `Optional[bool]`
- Default: `None`
- Description: Enable trailing stop for the strategy

### trailing_stop_pct
- Type: `Optional[float]`
- Default: `None`
- Description: Trailing stop distance in percent

### ema_fast
- Type: `Optional[int]`
- Default: `None`
- Description: Fast EMA period used by some strategies

### ema_slow
- Type: `Optional[int]`
- Default: `None`
- Description: Slow EMA period used by some strategies

### confirmation_timeframes
- Type: `Optional[List[str]]`
- Default: `None`
- Description: Additional timeframes used for confirmation

### momentum_threshold
- Type: `Optional[float]`
- Default: `None`
- Description: Minimum momentum score required

### volume_multiplier
- Type: `Optional[float]`
- Default: `None`
- Description: Multiplier applied to base volume requirements

### gas_threshold_gwei
- Type: `Optional[float]`
- Default: `None`
- Description: Maximum gas threshold for on-chain trades

### queue_timeout_ms
- Type: `int`
- Default: `2000`

### atr_threshold
- Type: `Optional[float]`
- Default: `None`
- Description: ATR threshold used to filter volatile regimes

### max_spread_bp
- Type: `Optional[float]`
- Default: `None`
- Description: Maximum allowed spread in basis points

## volatility_harvester
### enabled
- Type: `bool`
- Default: `False`

### max_concurrent_signals
- Type: `int`
- Default: `15`

### min_score
- Type: `float`
- Default: `0.05`

### timeframe
- Type: `str`
- Default: `1m`

### stop_loss_pct
- Type: `float`
- Default: `0.006`

### take_profit_pct
- Type: `float`
- Default: `0.014`

### trailing_stop_enabled
- Type: `Optional[bool]`
- Default: `None`
- Description: Enable trailing stop for the strategy

### trailing_stop_pct
- Type: `Optional[float]`
- Default: `None`
- Description: Trailing stop distance in percent

### ema_fast
- Type: `Optional[int]`
- Default: `None`
- Description: Fast EMA period used by some strategies

### ema_slow
- Type: `Optional[int]`
- Default: `None`
- Description: Slow EMA period used by some strategies

### confirmation_timeframes
- Type: `Optional[List[str]]`
- Default: `None`
- Description: Additional timeframes used for confirmation

### momentum_threshold
- Type: `Optional[float]`
- Default: `None`
- Description: Minimum momentum score required

### volume_multiplier
- Type: `Optional[float]`
- Default: `None`
- Description: Multiplier applied to base volume requirements

### gas_threshold_gwei
- Type: `Optional[float]`
- Default: `None`
- Description: Maximum gas threshold for on-chain trades

### queue_timeout_ms
- Type: `Optional[int]`
- Default: `None`
- Description: Timeout for high-frequency trade queue

### atr_threshold
- Type: `float`
- Default: `0.001`

### max_spread_bp
- Type: `Optional[float]`
- Default: `None`
- Description: Maximum allowed spread in basis points

## maker_spread
### enabled
- Type: `bool`
- Default: `False`

### max_concurrent_signals
- Type: `int`
- Default: `10`

### min_score
- Type: `float`
- Default: `0.08`

### timeframe
- Type: `str`
- Default: `1m`

### stop_loss_pct
- Type: `float`
- Default: `0.008`

### take_profit_pct
- Type: `float`
- Default: `0.016`

### trailing_stop_enabled
- Type: `Optional[bool]`
- Default: `None`
- Description: Enable trailing stop for the strategy

### trailing_stop_pct
- Type: `Optional[float]`
- Default: `None`
- Description: Trailing stop distance in percent

### ema_fast
- Type: `Optional[int]`
- Default: `None`
- Description: Fast EMA period used by some strategies

### ema_slow
- Type: `Optional[int]`
- Default: `None`
- Description: Slow EMA period used by some strategies

### confirmation_timeframes
- Type: `Optional[List[str]]`
- Default: `None`
- Description: Additional timeframes used for confirmation

### momentum_threshold
- Type: `Optional[float]`
- Default: `None`
- Description: Minimum momentum score required

### volume_multiplier
- Type: `Optional[float]`
- Default: `None`
- Description: Multiplier applied to base volume requirements

### gas_threshold_gwei
- Type: `Optional[float]`
- Default: `None`
- Description: Maximum gas threshold for on-chain trades

### queue_timeout_ms
- Type: `Optional[int]`
- Default: `None`
- Description: Timeout for high-frequency trade queue

### atr_threshold
- Type: `Optional[float]`
- Default: `None`
- Description: ATR threshold used to filter volatile regimes

### max_spread_bp
- Type: `float`
- Default: `8.0`

## breakout
### enabled
- Type: `bool`
- Default: `True`
- Description: Whether the strategy is active

### max_concurrent_signals
- Type: `int`
- Default: `0`
- Description: Unused placeholder

### min_score
- Type: `float`
- Default: `0.01`

### timeframe
- Type: `Optional[str]`
- Default: `None`
- Description: Primary timeframe used by the strategy

### stop_loss_pct
- Type: `Optional[float]`
- Default: `None`
- Description: Stop loss expressed as a percentage of entry

### take_profit_pct
- Type: `Optional[float]`
- Default: `None`
- Description: Take profit expressed as a percentage of entry

### trailing_stop_enabled
- Type: `Optional[bool]`
- Default: `None`
- Description: Enable trailing stop for the strategy

### trailing_stop_pct
- Type: `Optional[float]`
- Default: `None`
- Description: Trailing stop distance in percent

### ema_fast
- Type: `Optional[int]`
- Default: `None`
- Description: Fast EMA period used by some strategies

### ema_slow
- Type: `Optional[int]`
- Default: `None`
- Description: Slow EMA period used by some strategies

### confirmation_timeframes
- Type: `Optional[List[str]]`
- Default: `None`
- Description: Additional timeframes used for confirmation

### momentum_threshold
- Type: `Optional[float]`
- Default: `None`
- Description: Minimum momentum score required

### volume_multiplier
- Type: `Optional[float]`
- Default: `None`
- Description: Multiplier applied to base volume requirements

### gas_threshold_gwei
- Type: `Optional[float]`
- Default: `None`
- Description: Maximum gas threshold for on-chain trades

### queue_timeout_ms
- Type: `Optional[int]`
- Default: `None`
- Description: Timeout for high-frequency trade queue

### atr_threshold
- Type: `Optional[float]`
- Default: `None`
- Description: ATR threshold used to filter volatile regimes

### max_spread_bp
- Type: `Optional[float]`
- Default: `None`
- Description: Maximum allowed spread in basis points

## circuit_breaker
### enabled
- Type: `bool`
- Default: `True`
- Description: Enable global circuit breaker

### expected_exception
- Type: `str`
- Default: `Exception`
- Description: Exception type that triggers the circuit breaker

### failure_threshold
- Type: `int`
- Default: `10`
- Description: Number of failures before opening the circuit

### half_open_max_calls
- Type: `int`
- Default: `3`
- Description: Calls allowed while the breaker is half-open

### recovery_timeout
- Type: `int`
- Default: `300`
- Description: Cooldown before retrying after failure

## comprehensive_mode_optimization
### adaptive_rate_limiting
- Type: `bool`
- Default: `True`

### batch_chunk_size
- Type: `int`
- Default: `30`

### enable_memory_optimization
- Type: `bool`
- Default: `True`

### enable_progress_tracking
- Type: `bool`
- Default: `True`

### memory_cleanup_interval
- Type: `int`
- Default: `20`

## cycle_delay_seconds
- Type: `int`
- Default: `18`

## indicator_lookback
- Type: `int`
- Default: `15`

## profit_focus_mode
- Type: `bool`
- Default: `True`

## min_profit_potential
- Type: `float`
- Default: `0.015`

## max_trade_frequency
- Type: `int`
- Default: `8`

## data_validation
### allow_empty_volume
- Type: `bool`
- Default: `True`

### max_data_age_minutes
- Type: `int`
- Default: `60`

### min_data_points
- Type: `int`
- Default: `50`

### require_volume
- Type: `bool`
- Default: `False`

### validate_ohlcv
- Type: `bool`
- Default: `True`

## enhanced_backtesting
### enabled
- Type: `bool`
- Default: `False`

## enhanced_scanning
### data_sources
#### orderbook
- Type: `List[str]`
- Default: `["jupiter"]`

#### price
- Type: `List[str]`
- Default: `["pyth", "jupiter"]`

#### volume
- Type: `List[str]`
- Default: `["birdeye"]`

### discovery_sources
- Type: `List[str]`
- Default: `["basic_scanner", "dex_aggregators"]`

### enable_pyth_prices
- Type: `bool`
- Default: `True`

### enable_sentiment
- Type: `bool`
- Default: `False`

### enabled
- Type: `bool`
- Default: `True`

### max_spread_pct
- Type: `float`
- Default: `1.2`

### max_tokens_per_scan
- Type: `int`
- Default: `20`

### min_confidence
- Type: `float`
- Default: `0.45`

### min_liquidity_score
- Type: `float`
- Default: `0.6`

### min_score_threshold
- Type: `float`
- Default: `0.35`

### min_strategy_fit
- Type: `float`
- Default: `0.65`

### min_volume_usd
- Type: `float`
- Default: `5000`

### scan_interval
- Type: `int`
- Default: `15`
- Description: Scan interval in minutes

### min_price_movement
- Type: `float`
- Default: `0.008`

### max_price_staleness
- Type: `int`
- Default: `30`
- Description: Maximum age of price data

### volatility_filter
- Type: `bool`
- Default: `True`

### min_daily_volume_multiplier
- Type: `float`
- Default: `2.0`

## error_handling
### continue_on_error
- Type: `bool`
- Default: `True`

### exponential_backoff
- Type: `bool`
- Default: `True`

### fallback_data_sources
- Type: `bool`
- Default: `True`

### graceful_degradation
- Type: `bool`
- Default: `True`

### log_errors
- Type: `bool`
- Default: `True`

### max_backoff
- Type: `float`
- Default: `30.0`

### max_retries
- Type: `int`
- Default: `2`

### retry_delay
- Type: `float`
- Default: `2.0`

### scan_error_cooldown
- Type: `int`
- Default: `300`

### scan_error_threshold
- Type: `int`
- Default: `5`

## exchange
- Type: `str`
- Default: `kraken`

## execution_mode
- Type: `str`
- Default: `paper`

## exit_strategy
### min_gain_to_trail
- Type: `float`
- Default: `0.012`

### momentum_aware_exits
- Type: `bool`
- Default: `True`

### momentum_tp_scaling
- Type: `bool`
- Default: `True`

### momentum_trail_adjustment
- Type: `bool`
- Default: `True`

### scale_out_enabled
- Type: `bool`
- Default: `True`

### scale_out_levels
- Type: `List[ScaleOutLevel]`
- Default: `[ScaleOutLevel(pct=0.015, portion=0.25), ScaleOutLevel(pct=0.03, portion=0.25), ScaleOutLevel(pct=0.06, portion=0.5)]`

### time_based_exits
- Type: `bool`
- Default: `True`

### max_hold_time_minutes
- Type: `int`
- Default: `45`

### breakeven_activation
- Type: `float`
- Default: `0.01`

### real_time_monitoring
#### check_interval_seconds
- Type: `int`
- Default: `3`

#### enabled
- Type: `bool`
- Default: `True`

### take_profit_pct
- Type: `float`
- Default: `0.08`

### trailing_stop_pct
- Type: `float`
- Default: `0.02`

## exits
### default_sl_pct
- Type: `float`
- Default: `0.01`

### default_tp_pct
- Type: `float`
- Default: `0.08`

### trailing_stop_pct
- Type: `float`
- Default: `0.015`

### profit_lock_enabled
- Type: `bool`
- Default: `True`

### profit_lock_levels
- Type: `List[ProfitLockLevel]`
- Default: `[ProfitLockLevel(threshold=0.02, trail_pct=0.008), ProfitLockLevel(threshold=0.05, trail_pct=0.012)]`

## kraken
### enabled
- Type: `bool`
- Default: `True`

### max_retries
- Type: `int`
- Default: `3`

### rate_limit
- Type: `bool`
- Default: `True`

### retry_on_failure
- Type: `bool`
- Default: `True`

### sandbox
- Type: `bool`
- Default: `False`

### timeout
- Type: `int`
- Default: `30`

## max_concurrent_ohlcv
- Type: `int`
- Default: `20`

## min_confidence_score
- Type: `float`
- Default: `0.001`

## mode
- Type: `str`
- Default: `auto`

## pipeline_stability
### circuit_breaker_enabled
- Type: `bool`
- Default: `True`

### circuit_breaker_threshold
- Type: `int`
- Default: `3`

### continue_on_error
- Type: `bool`
- Default: `True`

### disable_complex_filters
- Type: `bool`
- Default: `False`

### enable_rate_limiting
- Type: `bool`
- Default: `True`

### fallback_to_simple_mode
- Type: `bool`
- Default: `True`

### max_concurrent_requests
- Type: `int`
- Default: `15`

### max_failures_per_cycle
- Type: `int`
- Default: `5`

### rate_limit_delay
- Type: `float`
- Default: `0.8`

### request_timeout
- Type: `int`
- Default: `25`

### retry_attempts
- Type: `int`
- Default: `3`

## position_sync_enabled
- Type: `bool`
- Default: `True`

## process_all_symbols
- Type: `bool`
- Default: `True`

## rate_limiting
### burst_limit
- Type: `int`
- Default: `8`

### burst_window
- Type: `float`
- Default: `5.0`

### enabled
- Type: `bool`
- Default: `True`

### max_retries
- Type: `int`
- Default: `3`

### requests_per_minute
- Type: `int`
- Default: `55`

### retry_delay
- Type: `float`
- Default: `1.2`

## regime_timeframes
- Type: `List[str]`
- Default: `["1m", "15m", "1h", "4h", "1d", "1w"]`

## risk
### enable_partial_exits
- Type: `bool`
- Default: `True`

### enable_trailing_stop
- Type: `bool`
- Default: `True`

### max_drawdown
- Type: `float`
- Default: `0.12`

### max_positions
- Type: `int`
- Default: `35`

### max_risk_per_trade
- Type: `float`
- Default: `0.015`

### max_total_risk
- Type: `float`
- Default: `0.25`

### min_position_size_usd
- Type: `float`
- Default: `25`

### partial_exit_pct
- Type: `float`
- Default: `0.33`

### position_size_pct
- Type: `float`
- Default: `0.08`

### stop_loss_pct
- Type: `float`
- Default: `0.008`

### take_profit_pct
- Type: `float`
- Default: `0.025`

### trailing_stop_pct
- Type: `float`
- Default: `0.01`

### scalp_stop_loss_pct
- Type: `float`
- Default: `0.006`

### scalp_take_profit_pct
- Type: `float`
- Default: `0.024`

### scalp_max_positions
- Type: `int`
- Default: `45`

### dynamic_risk_scaling
- Type: `bool`
- Default: `True`

### win_rate_threshold
- Type: `float`
- Default: `0.55`

### risk_scaling_factor
- Type: `float`
- Default: `1.2`

## scan_cache
### auto_cleanup
- Type: `bool`
- Default: `True`

### max_age_hours
- Type: `int`
- Default: `8`

### max_cache_size
- Type: `int`
- Default: `750`

### min_score_threshold
- Type: `float`
- Default: `0.3`

### persist_to_disk
- Type: `bool`
- Default: `True`

### review_interval_minutes
- Type: `int`
- Default: `15`

## scan_monitoring
### alert_on_scan_failures
- Type: `bool`
- Default: `True`

### enable_scan_metrics
- Type: `bool`
- Default: `True`

### log_scan_results
- Type: `bool`
- Default: `True`

### max_scan_logs
- Type: `int`
- Default: `1000`

### scan_metrics_interval
- Type: `int`
- Default: `60`

### track_scan_performance
- Type: `bool`
- Default: `True`

## skip_symbol_filters
- Type: `bool`
- Default: `True`

## solana_scanner
### enabled
- Type: `bool`
- Default: `True`

### interval_minutes
- Type: `int`
- Default: `45`

### api_keys
#### bitquery
- Type: `str`
- Default: `YOUR_KEY`
- Description: Bitquery API key

#### moralis
- Type: `str`
- Default: `YOUR_KEY`
- Description: Moralis API key

### min_volume_usd
- Type: `float`
- Default: `3000`

### max_tokens_per_scan
- Type: `int`
- Default: `75`

### gecko_search
- Type: `bool`
- Default: `True`

## strategy_evaluation_mode
- Type: `str`
- Default: `best`

## strategy_router
### regimes
Mapping of market regimes to strategies

#### breakout
- Type: `List[str]`
- Default: `["breakout_bot", "sniper_bot", "ultra_scalp_bot"]`

#### mean-reverting
- Type: `List[str]`
- Default: `["mean_bot", "range_arb_bot", "stat_arb_bot"]`

#### sideways
- Type: `List[str]`
- Default: `["market_making_bot", "grid_bot", "maker_spread"]`

#### trending
- Type: `List[str]`
- Default: `["momentum_bot", "lstm_bot", "trend_bot"]`

#### volatile
- Type: `List[str]`
- Default: `["arbitrage_engine", "volatility_harvester", "flash_crash_bot"]`

#### scalp
- Type: `List[str]`
- Default: `["ultra_scalp_bot", "micro_scalp_bot", "bounce_scalper", "momentum_exploiter", "dex_scalper", "hft_engine", "volatility_harvester", "maker_spread"]`

#### scalp_timeframe
- Type: `str`
- Default: `1m`

## symbol_batch_size
- Type: `int`
- Default: `75`

## symbol_validation
### allowed_quotes
- Type: `List[str]`
- Default: `["USD", "EUR", "USDC", "USDT"]`

### enabled
- Type: `bool`
- Default: `True`

### max_price_deviation
- Type: `float`
- Default: `0.1`

### max_symbol_length
- Type: `int`
- Default: `25`

### min_liquidity_score
- Type: `float`
- Default: `0.5`

### min_volume_usd
- Type: `float`
- Default: `3000`

### require_kraken_support
- Type: `bool`
- Default: `True`

### skip_unknown_addresses
- Type: `bool`
- Default: `True`

### strict_mode
- Type: `bool`
- Default: `False`

### timeout_seconds
- Type: `int`
- Default: `10`

### validate_liquidity
- Type: `bool`
- Default: `True`

### validate_price
- Type: `bool`
- Default: `True`

### validate_volume
- Type: `bool`
- Default: `True`

## symbols
- Type: `List[str]`
- Default: `["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", "LINK/USD", "UNI/USD", "AAVE/USD", "AVAX/USD", "MATIC/USD", "ATOM/USD", "NEAR/USD"]`

## telegram
### balance_updates
- Type: `bool`
- Default: `False`

### enabled
- Type: `bool`
- Default: `True`

### status_updates
- Type: `bool`
- Default: `False`

### trade_updates
- Type: `bool`
- Default: `True`

## testing_mode
- Type: `bool`
- Default: `True`

## timeframe
- Type: `str`
- Default: `15m`

## use_enhanced_ohlcv_fetcher
- Type: `bool`
- Default: `True`

## use_trade_manager_as_source
- Type: `bool`
- Default: `True`

## use_websocket
- Type: `bool`
- Default: `True`

## websocket
### connection_timeout
- Type: `int`
- Default: `35`

### enabled
- Type: `bool`
- Default: `True`

### health_check
#### enabled
- Type: `bool`
- Default: `True`

#### failure_threshold
- Type: `int`
- Default: `2`

#### interval
- Type: `int`
- Default: `25`

### kraken
#### enhanced_error_handling
- Type: `bool`
- Default: `True`

#### health_monitoring
- Type: `bool`
- Default: `True`

#### message_validation
- Type: `bool`
- Default: `True`

### max_connections
- Type: `int`
- Default: `12`

### max_message_size
- Type: `int`
- Default: `1048576`

### monitoring
#### alert_on_failures
- Type: `bool`
- Default: `True`

#### log_level
- Type: `str`
- Default: `INFO`

#### metrics_interval
- Type: `int`
- Default: `45`

### ping_interval
- Type: `int`
- Default: `15`

### ping_timeout
- Type: `int`
- Default: `8`

### reconnect
#### backoff_factor
- Type: `float`
- Default: `1.8`

#### base_delay
- Type: `float`
- Default: `0.8`

#### max_attempts
- Type: `int`
- Default: `12`

#### max_delay
- Type: `float`
- Default: `45.0`

### solana
#### enhanced_monitor
- Type: `bool`
- Default: `True`

#### exponential_backoff
- Type: `bool`
- Default: `True`

#### health_checks
- Type: `bool`
- Default: `True`

#### message_validation
- Type: `bool`
- Default: `True`

## performance_optimization
### memory
#### enable_adaptive_gc
- Type: `bool`
- Default: `True`

#### gc_threshold
- Type: `float`
- Default: `0.75`

#### max_memory_usage_pct
- Type: `float`
- Default: `85.0`

#### cache_size_limit_mb
- Type: `int`
- Default: `512`

### concurrency
#### enable_adaptive_limits
- Type: `bool`
- Default: `True`

#### min_concurrent_requests
- Type: `int`
- Default: `2`

#### max_concurrent_requests
- Type: `int`
- Default: `15`

#### response_time_threshold
- Type: `float`
- Default: `1.2`

#### success_rate_threshold
- Type: `float`
- Default: `0.88`

### caching
#### enable_intelligent_cache
- Type: `bool`
- Default: `True`

#### cache_hit_rate_threshold
- Type: `float`
- Default: `0.8`

#### adaptive_ttl
- Type: `bool`
- Default: `True`

#### min_cache_ttl
- Type: `int`
- Default: `45`

#### max_cache_ttl
- Type: `int`
- Default: `2400`

### database
#### enable_connection_pooling
- Type: `bool`
- Default: `True`

#### pool_size
- Type: `int`
- Default: `10`

#### max_overflow
- Type: `int`
- Default: `20`

#### pool_timeout
- Type: `int`
- Default: `30`

### monitoring
#### enable_performance_alerts
- Type: `bool`
- Default: `True`

#### cpu_threshold
- Type: `float`
- Default: `80.0`

#### memory_threshold
- Type: `float`
- Default: `85.0`

#### response_time_threshold
- Type: `float`
- Default: `1.8`

#### error_rate_threshold
- Type: `float`
- Default: `0.03`

## profitability_optimization
### target_daily_return
- Type: `float`
- Default: `0.025`

### target_win_rate
- Type: `float`
- Default: `0.6`

### target_avg_win_loss_ratio
- Type: `float`
- Default: `2.5`

### target_max_drawdown
- Type: `float`
- Default: `0.08`

### enable_strategy_adaptation
- Type: `bool`
- Default: `True`

### performance_review_interval
- Type: `int`
- Default: `300`

### strategy_switch_threshold
- Type: `float`
- Default: `0.15`

### sharpe_ratio_target
- Type: `float`
- Default: `2.0`

### sortino_ratio_target
- Type: `float`
- Default: `2.5`

### enable_profit_compounding
- Type: `bool`
- Default: `True`

### compounding_frequency
- Type: `int`
- Default: `3600`

### profit_reinvestment_pct
- Type: `float`
- Default: `0.5`

### enable_high_frequency_mode
- Type: `bool`
- Default: `False`

### enable_scalp_acceleration
- Type: `bool`
- Default: `True`

### acceleration_trigger_win_rate
- Type: `float`
- Default: `0.65`

### acceleration_factor
- Type: `float`
- Default: `1.5`
