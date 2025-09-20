"""Runtime configuration models for the trading bot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import os
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    from services.configuration import ManagedConfigService, SecretNotFoundError
except ImportError:
    # Fallback for when services.configuration is not available
    class SecretNotFoundError(Exception):
        pass
    
    class ManagedConfigService:
        @staticmethod
        def resolve_secrets(config_data):
            return config_data


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
CONFIG_PATH_ENV = "BOT_CONFIG_FILE"
MANAGED_SECRETS_PATH = Path(__file__).resolve().parents[2] / "config" / "managed_secrets.yaml"


class ConfigError(RuntimeError):
    """Raised when configuration files cannot be parsed or validated."""


class StrategySettings(BaseModel):
    """Base configuration shared by strategies that emit signals."""

    enabled: bool = Field(True, description="Whether the strategy is active")
    max_concurrent_signals: int = Field(
        0,
        ge=0,
        description="Maximum concurrent signals allowed for the strategy",
    )
    min_score: float = Field(
        0.0,
        ge=0.0,
        description="Minimum signal quality score accepted",
    )
    timeframe: Optional[str] = Field(
        None, description="Primary timeframe used by the strategy"
    )
    stop_loss_pct: Optional[float] = Field(
        None,
        ge=0.0,
        description="Stop loss expressed as a percentage of entry",
    )
    take_profit_pct: Optional[float] = Field(
        None,
        ge=0.0,
        description="Take profit expressed as a percentage of entry",
    )
    trailing_stop_enabled: Optional[bool] = Field(
        None, description="Enable trailing stop for the strategy"
    )
    trailing_stop_pct: Optional[float] = Field(
        None,
        ge=0.0,
        description="Trailing stop distance in percent",
    )
    ema_fast: Optional[int] = Field(
        None, ge=1, description="Fast EMA period used by some strategies"
    )
    ema_slow: Optional[int] = Field(
        None, ge=1, description="Slow EMA period used by some strategies"
    )
    confirmation_timeframes: Optional[List[str]] = Field(
        None,
        description="Additional timeframes used for confirmation",
    )
    momentum_threshold: Optional[float] = Field(
        None, ge=0.0, description="Minimum momentum score required"
    )
    volume_multiplier: Optional[float] = Field(
        None,
        ge=0.0,
        description="Multiplier applied to base volume requirements",
    )
    gas_threshold_gwei: Optional[float] = Field(
        None, ge=0.0, description="Maximum gas threshold for on-chain trades"
    )
    queue_timeout_ms: Optional[int] = Field(
        None, ge=0, description="Timeout for high-frequency trade queue"
    )
    atr_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        description="ATR threshold used to filter volatile regimes",
    )
    max_spread_bp: Optional[float] = Field(
        None,
        ge=0.0,
        description="Maximum allowed spread in basis points",
    )

    model_config = ConfigDict(extra="forbid")


class BounceScalperSettings(StrategySettings):
    max_concurrent_signals: int = Field(
        20, ge=0, description="Bounce scalper maximum simultaneous signals"
    )
    min_score: float = Field(0.01, ge=0.0, description="Minimum bounce score")


class UltraScalpBotSettings(StrategySettings):
    max_concurrent_signals: int = Field(25, ge=0)
    min_score: float = Field(0.08, ge=0.0)
    timeframe: str = Field("1m", description="Execution timeframe")
    stop_loss_pct: float = Field(0.005, ge=0.0)
    take_profit_pct: float = Field(0.020, ge=0.0)
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = Field(0.008, ge=0.0)


class MicroScalpBotSettings(StrategySettings):
    max_concurrent_signals: int = Field(20, ge=0)
    min_score: float = Field(0.06, ge=0.0)
    timeframe: str = "1m"
    ema_fast: int = Field(3, ge=1)
    ema_slow: int = Field(8, ge=1)
    stop_loss_pct: float = Field(0.006, ge=0.0)
    take_profit_pct: float = Field(0.024, ge=0.0)
    confirmation_timeframes: List[str] = Field(
        default_factory=lambda: ["5m", "15m"],
        description="Additional timeframes for confirmation",
    )


class MomentumExploiterSettings(StrategySettings):
    max_concurrent_signals: int = Field(15, ge=0)
    min_score: float = Field(0.07, ge=0.0)
    timeframe: str = "1m"
    stop_loss_pct: float = Field(0.008, ge=0.0)
    take_profit_pct: float = Field(0.032, ge=0.0)
    momentum_threshold: float = Field(0.015, ge=0.0)
    volume_multiplier: float = Field(1.5, ge=0.0)


class DexScalperSettings(StrategySettings):
    enabled: bool = False
    max_concurrent_signals: int = Field(25, ge=0)
    min_score: float = Field(0.03, ge=0.0)
    timeframe: str = "1m"
    ema_fast: int = Field(3, ge=1)
    ema_slow: int = Field(8, ge=1)
    stop_loss_pct: float = Field(0.004, ge=0.0)
    take_profit_pct: float = Field(0.010, ge=0.0)
    gas_threshold_gwei: float = Field(15.0, ge=0.0)


class HftEngineSettings(StrategySettings):
    enabled: bool = False
    max_concurrent_signals: int = Field(50, ge=0)
    min_score: float = Field(0.02, ge=0.0)
    timeframe: str = "1m"
    stop_loss_pct: float = Field(0.002, ge=0.0)
    take_profit_pct: float = Field(0.006, ge=0.0)
    queue_timeout_ms: int = Field(2000, ge=0)


class VolatilityHarvesterSettings(StrategySettings):
    enabled: bool = False
    max_concurrent_signals: int = Field(15, ge=0)
    min_score: float = Field(0.05, ge=0.0)
    timeframe: str = "1m"
    stop_loss_pct: float = Field(0.006, ge=0.0)
    take_profit_pct: float = Field(0.014, ge=0.0)
    atr_threshold: float = Field(0.001, ge=0.0)


class MakerSpreadSettings(StrategySettings):
    enabled: bool = False
    max_concurrent_signals: int = Field(10, ge=0)
    min_score: float = Field(0.08, ge=0.0)
    timeframe: str = "1m"
    stop_loss_pct: float = Field(0.008, ge=0.0)
    take_profit_pct: float = Field(0.016, ge=0.0)
    max_spread_bp: float = Field(8.0, ge=0.0)


class BreakoutSettings(StrategySettings):
    max_concurrent_signals: int = Field(0, ge=0, description="Unused placeholder")
    min_score: float = Field(0.01, ge=0.0)


class CircuitBreakerSettings(BaseModel):
    enabled: bool = Field(True, description="Enable global circuit breaker")
    expected_exception: str = Field(
        "Exception",
        description="Exception type that triggers the circuit breaker",
    )
    failure_threshold: int = Field(
        10, ge=1, description="Number of failures before opening the circuit"
    )
    half_open_max_calls: int = Field(
        3, ge=1, description="Calls allowed while the breaker is half-open"
    )
    recovery_timeout: int = Field(
        300, ge=1, description="Cooldown before retrying after failure",
    )

    model_config = ConfigDict(extra="forbid")


class ComprehensiveModeOptimizationSettings(BaseModel):
    adaptive_rate_limiting: bool = True
    batch_chunk_size: int = Field(30, ge=1)
    enable_memory_optimization: bool = True
    enable_progress_tracking: bool = True
    memory_cleanup_interval: int = Field(20, ge=1)

    model_config = ConfigDict(extra="forbid")


class DataValidationSettings(BaseModel):
    allow_empty_volume: bool = True
    max_data_age_minutes: int = Field(60, ge=1)
    min_data_points: int = Field(50, ge=1)
    require_volume: bool = False
    validate_ohlcv: bool = True

    model_config = ConfigDict(extra="forbid")


class EnhancedBacktestingSettings(BaseModel):
    enabled: bool = False

    model_config = ConfigDict(extra="forbid")


class EnhancedScanningDataSources(BaseModel):
    orderbook: List[str] = Field(default_factory=lambda: ["jupiter"])
    price: List[str] = Field(default_factory=lambda: ["pyth", "jupiter"])
    volume: List[str] = Field(default_factory=lambda: ["birdeye"])

    model_config = ConfigDict(extra="forbid")


class EnhancedScanningSettings(BaseModel):
    data_sources: EnhancedScanningDataSources = Field(
        default_factory=EnhancedScanningDataSources
    )
    discovery_sources: List[str] = Field(
        default_factory=lambda: ["basic_scanner", "dex_aggregators"]
    )
    enable_pyth_prices: bool = True
    enable_sentiment: bool = False
    enabled: bool = True
    max_spread_pct: float = Field(1.2, ge=0.0)
    max_tokens_per_scan: int = Field(20, ge=1)
    min_confidence: float = Field(0.45, ge=0.0)
    min_liquidity_score: float = Field(0.60, ge=0.0)
    min_score_threshold: float = Field(0.35, ge=0.0)
    min_strategy_fit: float = Field(0.65, ge=0.0)
    min_volume_usd: float = Field(5000, ge=0.0)
    scan_interval: int = Field(15, ge=1, description="Scan interval in minutes")
    min_price_movement: float = Field(0.008, ge=0.0)
    max_price_staleness: int = Field(30, ge=1, description="Maximum age of price data")
    volatility_filter: bool = True
    min_daily_volume_multiplier: float = Field(2.0, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class ErrorHandlingSettings(BaseModel):
    continue_on_error: bool = True
    exponential_backoff: bool = True
    fallback_data_sources: bool = True
    graceful_degradation: bool = True
    log_errors: bool = True
    max_backoff: float = Field(30.0, ge=0.0)
    max_retries: int = Field(2, ge=0)
    retry_delay: float = Field(2.0, ge=0.0)
    scan_error_cooldown: int = Field(300, ge=0)
    scan_error_threshold: int = Field(5, ge=0)

    model_config = ConfigDict(extra="forbid")


class RealTimeMonitoringSettings(BaseModel):
    check_interval_seconds: int = Field(3, ge=1)
    enabled: bool = True

    model_config = ConfigDict(extra="forbid")


class ScaleOutLevel(BaseModel):
    pct: float = Field(..., ge=0.0)
    portion: float = Field(..., ge=0.0)

    model_config = ConfigDict(extra="forbid")


class ExitStrategySettings(BaseModel):
    min_gain_to_trail: float = Field(0.012, ge=0.0)
    momentum_aware_exits: bool = True
    momentum_tp_scaling: bool = True
    momentum_trail_adjustment: bool = True
    scale_out_enabled: bool = True
    scale_out_levels: List[ScaleOutLevel] = Field(
        default_factory=lambda: [
            ScaleOutLevel(pct=0.015, portion=0.25),
            ScaleOutLevel(pct=0.030, portion=0.25),
            ScaleOutLevel(pct=0.060, portion=0.50),
        ]
    )
    time_based_exits: bool = True
    max_hold_time_minutes: int = Field(45, ge=1)
    breakeven_activation: float = Field(0.010, ge=0.0)
    real_time_monitoring: RealTimeMonitoringSettings = Field(
        default_factory=RealTimeMonitoringSettings
    )
    take_profit_pct: float = Field(0.08, ge=0.0)
    trailing_stop_pct: float = Field(0.020, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class ProfitLockLevel(BaseModel):
    threshold: float = Field(..., ge=0.0)
    trail_pct: float = Field(..., ge=0.0)

    model_config = ConfigDict(extra="forbid")


class ExitsSettings(BaseModel):
    default_sl_pct: float = Field(0.010, ge=0.0)
    default_tp_pct: float = Field(0.080, ge=0.0)
    trailing_stop_pct: float = Field(0.015, ge=0.0)
    profit_lock_enabled: bool = True
    profit_lock_levels: List[ProfitLockLevel] = Field(
        default_factory=lambda: [
            ProfitLockLevel(threshold=0.020, trail_pct=0.008),
            ProfitLockLevel(threshold=0.050, trail_pct=0.012),
        ]
    )

    model_config = ConfigDict(extra="forbid")


class KrakenSettings(BaseModel):
    enabled: bool = True
    max_retries: int = Field(3, ge=0)
    rate_limit: bool = True
    retry_on_failure: bool = True
    sandbox: bool = False
    timeout: int = Field(30, ge=0)

    model_config = ConfigDict(extra="forbid")


class PipelineStabilitySettings(BaseModel):
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = Field(3, ge=0)
    continue_on_error: bool = True
    disable_complex_filters: bool = False
    enable_rate_limiting: bool = True
    fallback_to_simple_mode: bool = True
    max_concurrent_requests: int = Field(15, ge=0)
    max_failures_per_cycle: int = Field(5, ge=0)
    rate_limit_delay: float = Field(0.8, ge=0.0)
    request_timeout: int = Field(25, ge=0)
    retry_attempts: int = Field(3, ge=0)

    model_config = ConfigDict(extra="forbid")


class RateLimitingSettings(BaseModel):
    burst_limit: int = Field(8, ge=0)
    burst_window: float = Field(5.0, ge=0.0)
    enabled: bool = True
    max_retries: int = Field(3, ge=0)
    requests_per_minute: int = Field(55, ge=0)
    retry_delay: float = Field(1.2, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class RiskSettings(BaseModel):
    enable_partial_exits: bool = True
    enable_trailing_stop: bool = True
    max_drawdown: float = Field(0.12, ge=0.0)
    max_positions: int = Field(35, ge=0)
    max_risk_per_trade: float = Field(0.015, ge=0.0)
    max_total_risk: float = Field(0.25, ge=0.0)
    min_position_size_usd: float = Field(25, ge=0.0)
    partial_exit_pct: float = Field(0.33, ge=0.0)
    position_size_pct: float = Field(0.08, ge=0.0)
    stop_loss_pct: float = Field(0.008, ge=0.0)
    take_profit_pct: float = Field(0.025, ge=0.0)
    trailing_stop_pct: float = Field(0.010, ge=0.0)
    scalp_stop_loss_pct: float = Field(0.006, ge=0.0)
    scalp_take_profit_pct: float = Field(0.024, ge=0.0)
    scalp_max_positions: int = Field(45, ge=0)
    dynamic_risk_scaling: bool = True
    win_rate_threshold: float = Field(0.55, ge=0.0)
    risk_scaling_factor: float = Field(1.2, ge=0.0)
    starting_balance: float = Field(10000.0, ge=0.0)
    min_confidence_threshold: float = Field(0.3, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class ScanCacheSettings(BaseModel):
    auto_cleanup: bool = True
    max_age_hours: int = Field(8, ge=0)
    max_cache_size: int = Field(750, ge=0)
    min_score_threshold: float = Field(0.3, ge=0.0)
    persist_to_disk: bool = True
    review_interval_minutes: int = Field(15, ge=0)

    model_config = ConfigDict(extra="forbid")


class ScanMonitoringSettings(BaseModel):
    alert_on_scan_failures: bool = True
    enable_scan_metrics: bool = True
    log_scan_results: bool = True
    max_scan_logs: int = Field(1000, ge=0)
    scan_metrics_interval: int = Field(60, ge=0)
    track_scan_performance: bool = True

    model_config = ConfigDict(extra="forbid")


class SolanaScannerApiKeys(BaseModel):
    bitquery: str = Field("YOUR_KEY", description="Bitquery API key")
    moralis: str = Field("YOUR_KEY", description="Moralis API key")

    model_config = ConfigDict(extra="forbid")


class SolanaScannerSettings(BaseModel):
    enabled: bool = True
    interval_minutes: int = Field(45, ge=1)
    api_keys: SolanaScannerApiKeys = Field(default_factory=SolanaScannerApiKeys)
    min_volume_usd: float = Field(3000, ge=0.0)
    max_tokens_per_scan: int = Field(75, ge=0)
    gecko_search: bool = True
    min_score_threshold: float = Field(0.1, ge=0.0)
    min_liquidity_score: float = Field(0.2, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class StrategyRouterRegimes(BaseModel):
    breakout: List[str] = Field(
        default_factory=lambda: ["breakout_bot", "sniper_bot", "ultra_scalp_bot"],
        alias="breakout",
    )
    mean_reverting: List[str] = Field(
        default_factory=lambda: ["mean_bot", "range_arb_bot", "stat_arb_bot"],
        alias="mean-reverting",
    )
    sideways: List[str] = Field(
        default_factory=lambda: ["market_making_bot", "grid_bot", "maker_spread"],
        alias="sideways",
    )
    trending: List[str] = Field(
        default_factory=lambda: ["momentum_bot", "lstm_bot", "trend_bot"],
        alias="trending",
    )
    volatile: List[str] = Field(
        default_factory=lambda: ["arbitrage_engine", "volatility_harvester", "flash_crash_bot"],
        alias="volatile",
    )
    scalp: List[str] = Field(
        default_factory=lambda: [
            "ultra_scalp_bot",
            "micro_scalp_bot",
            "bounce_scalper",
            "momentum_exploiter",
            "dex_scalper",
            "hft_engine",
            "volatility_harvester",
            "maker_spread",
        ],
        alias="scalp",
    )
    scalp_timeframe: str = Field("1m", alias="scalp_timeframe")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class StrategyRouterSettings(BaseModel):
    regimes: StrategyRouterRegimes = Field(
        default_factory=StrategyRouterRegimes,
        description="Mapping of market regimes to strategies",
    )

    model_config = ConfigDict(extra="forbid")


class SymbolValidationSettings(BaseModel):
    allowed_quotes: List[str] = Field(
        default_factory=lambda: ["USD"]
    )
    enabled: bool = True
    max_price_deviation: float = Field(0.1, ge=0.0)
    max_symbol_length: int = Field(25, ge=1)
    min_liquidity_score: float = Field(0.5, ge=0.0)
    min_volume_usd: float = Field(3000, ge=0.0)
    require_kraken_support: bool = True
    skip_unknown_addresses: bool = True
    strict_mode: bool = False
    timeout_seconds: int = Field(10, ge=0)
    validate_liquidity: bool = True
    validate_price: bool = True
    validate_volume: bool = True

    model_config = ConfigDict(extra="forbid")


class TelegramSettings(BaseModel):
    balance_updates: bool = False
    enabled: bool = True
    status_updates: bool = False
    trade_updates: bool = True
    token: str = ""
    chat_id: str = ""
    fail_silently: bool = True

    model_config = ConfigDict(extra="forbid")


class WebsocketHealthCheckSettings(BaseModel):
    enabled: bool = True
    failure_threshold: int = Field(2, ge=0)
    interval: int = Field(25, ge=0)

    model_config = ConfigDict(extra="forbid")


class WebsocketKrakenSettings(BaseModel):
    enhanced_error_handling: bool = True
    health_monitoring: bool = True
    message_validation: bool = True

    model_config = ConfigDict(extra="forbid")


class WebsocketMonitoringSettings(BaseModel):
    alert_on_failures: bool = True
    log_level: str = "INFO"
    metrics_interval: int = Field(45, ge=0)

    model_config = ConfigDict(extra="forbid")


class WebsocketReconnectSettings(BaseModel):
    backoff_factor: float = Field(1.8, ge=0.0)
    base_delay: float = Field(0.8, ge=0.0)
    max_attempts: int = Field(12, ge=0)
    max_delay: float = Field(45.0, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class WebsocketSolanaSettings(BaseModel):
    enhanced_monitor: bool = True
    exponential_backoff: bool = True
    health_checks: bool = True
    message_validation: bool = True

    model_config = ConfigDict(extra="forbid")


class WebsocketSettings(BaseModel):
    connection_timeout: int = Field(35, ge=0)
    enabled: bool = True
    health_check: WebsocketHealthCheckSettings = Field(
        default_factory=WebsocketHealthCheckSettings
    )
    kraken: WebsocketKrakenSettings = Field(default_factory=WebsocketKrakenSettings)
    max_connections: int = Field(12, ge=0)
    max_message_size: int = Field(1048576, ge=0)
    monitoring: WebsocketMonitoringSettings = Field(
        default_factory=WebsocketMonitoringSettings
    )
    ping_interval: int = Field(15, ge=0)
    ping_timeout: int = Field(8, ge=0)
    reconnect: WebsocketReconnectSettings = Field(
        default_factory=WebsocketReconnectSettings
    )
    solana: WebsocketSolanaSettings = Field(default_factory=WebsocketSolanaSettings)

    model_config = ConfigDict(extra="forbid")


class PerformanceMemorySettings(BaseModel):
    enable_adaptive_gc: bool = True
    gc_threshold: float = Field(0.75, ge=0.0)
    max_memory_usage_pct: float = Field(85.0, ge=0.0)
    cache_size_limit_mb: int = Field(512, ge=0)

    model_config = ConfigDict(extra="forbid")


class PerformanceConcurrencySettings(BaseModel):
    enable_adaptive_limits: bool = True
    min_concurrent_requests: int = Field(2, ge=0)
    max_concurrent_requests: int = Field(15, ge=0)
    response_time_threshold: float = Field(1.2, ge=0.0)
    success_rate_threshold: float = Field(0.88, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class PerformanceCachingSettings(BaseModel):
    enable_intelligent_cache: bool = True
    cache_hit_rate_threshold: float = Field(0.80, ge=0.0)
    adaptive_ttl: bool = True
    min_cache_ttl: int = Field(45, ge=0)
    max_cache_ttl: int = Field(2400, ge=0)

    model_config = ConfigDict(extra="forbid")


class PerformanceDatabaseSettings(BaseModel):
    enable_connection_pooling: bool = True
    pool_size: int = Field(10, ge=0)
    max_overflow: int = Field(20, ge=0)
    pool_timeout: int = Field(30, ge=0)

    model_config = ConfigDict(extra="forbid")


class PerformanceMonitoringSettings(BaseModel):
    enable_performance_alerts: bool = True
    cpu_threshold: float = Field(80.0, ge=0.0)
    memory_threshold: float = Field(85.0, ge=0.0)
    response_time_threshold: float = Field(1.8, ge=0.0)
    error_rate_threshold: float = Field(0.03, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class PerformanceOptimizationSettings(BaseModel):
    memory: PerformanceMemorySettings = Field(default_factory=PerformanceMemorySettings)
    concurrency: PerformanceConcurrencySettings = Field(
        default_factory=PerformanceConcurrencySettings
    )
    caching: PerformanceCachingSettings = Field(
        default_factory=PerformanceCachingSettings
    )
    database: PerformanceDatabaseSettings = Field(
        default_factory=PerformanceDatabaseSettings
    )
    monitoring: PerformanceMonitoringSettings = Field(
        default_factory=PerformanceMonitoringSettings
    )

    model_config = ConfigDict(extra="forbid")


class ProfitabilityOptimizationSettings(BaseModel):
    target_daily_return: float = Field(0.025, ge=0.0)
    target_win_rate: float = Field(0.60, ge=0.0)
    target_avg_win_loss_ratio: float = Field(2.5, ge=0.0)
    target_max_drawdown: float = Field(0.08, ge=0.0)
    enable_strategy_adaptation: bool = True
    performance_review_interval: int = Field(300, ge=0)
    strategy_switch_threshold: float = Field(0.15, ge=0.0)
    sharpe_ratio_target: float = Field(2.0, ge=0.0)
    sortino_ratio_target: float = Field(2.5, ge=0.0)
    enable_profit_compounding: bool = True
    compounding_frequency: int = Field(3600, ge=0)
    profit_reinvestment_pct: float = Field(0.50, ge=0.0)
    enable_high_frequency_mode: bool = False
    enable_scalp_acceleration: bool = True
    acceleration_trigger_win_rate: float = Field(0.65, ge=0.0)
    acceleration_factor: float = Field(1.5, ge=0.0)

    model_config = ConfigDict(extra="forbid")


class BotSettings(BaseSettings):
    """Complete runtime configuration for the trading bot."""

    allow_short: bool = True
    bounce_scalper: BounceScalperSettings = Field(
        default_factory=BounceScalperSettings
    )
    ultra_scalp_bot: UltraScalpBotSettings = Field(
        default_factory=UltraScalpBotSettings
    )
    micro_scalp_bot: MicroScalpBotSettings = Field(
        default_factory=MicroScalpBotSettings
    )
    momentum_exploiter: MomentumExploiterSettings = Field(
        default_factory=MomentumExploiterSettings
    )
    dex_scalper: DexScalperSettings = Field(default_factory=DexScalperSettings)
    hft_engine: HftEngineSettings = Field(default_factory=HftEngineSettings)
    volatility_harvester: VolatilityHarvesterSettings = Field(
        default_factory=VolatilityHarvesterSettings
    )
    maker_spread: MakerSpreadSettings = Field(default_factory=MakerSpreadSettings)
    breakout: BreakoutSettings = Field(default_factory=BreakoutSettings)
    circuit_breaker: CircuitBreakerSettings = Field(
        default_factory=CircuitBreakerSettings
    )
    comprehensive_mode_optimization: ComprehensiveModeOptimizationSettings = Field(
        default_factory=ComprehensiveModeOptimizationSettings
    )
    cycle_delay_seconds: int = Field(18, ge=0)
    indicator_lookback: int = Field(15, ge=0)
    profit_focus_mode: bool = True
    min_profit_potential: float = Field(0.015, ge=0.0)
    max_trade_frequency: int = Field(8, ge=0)
    data_validation: DataValidationSettings = Field(
        default_factory=DataValidationSettings
    )
    enhanced_backtesting: EnhancedBacktestingSettings = Field(
        default_factory=EnhancedBacktestingSettings
    )
    enhanced_scanning: EnhancedScanningSettings = Field(
        default_factory=EnhancedScanningSettings
    )
    error_handling: ErrorHandlingSettings = Field(
        default_factory=ErrorHandlingSettings
    )
    exchange: str = "kraken"
    execution_mode: str = "paper"
    exit_strategy: ExitStrategySettings = Field(default_factory=ExitStrategySettings)
    exits: ExitsSettings = Field(default_factory=ExitsSettings)
    kraken: KrakenSettings = Field(default_factory=KrakenSettings)
    max_concurrent_ohlcv: int = Field(20, ge=0)
    min_confidence_score: float = Field(0.001, ge=0.0)
    mode: str = "auto"
    pipeline_stability: PipelineStabilitySettings = Field(
        default_factory=PipelineStabilitySettings
    )
    position_sync_enabled: bool = True
    process_all_symbols: bool = True
    rate_limiting: RateLimitingSettings = Field(default_factory=RateLimitingSettings)
    regime_timeframes: List[str] = Field(
        default_factory=lambda: ["1m", "15m", "1h", "4h", "1d", "1w"]
    )
    risk: RiskSettings = Field(default_factory=RiskSettings)
    scan_cache: ScanCacheSettings = Field(default_factory=ScanCacheSettings)
    scan_monitoring: ScanMonitoringSettings = Field(
        default_factory=ScanMonitoringSettings
    )
    skip_symbol_filters: bool = True
    solana_scanner: SolanaScannerSettings = Field(
        default_factory=SolanaScannerSettings
    )
    strategy_evaluation_mode: str = "best"
    strategy_router: StrategyRouterSettings = Field(
        default_factory=StrategyRouterSettings
    )
    symbol_batch_size: int = Field(75, ge=0)
    symbol_validation: SymbolValidationSettings = Field(
        default_factory=SymbolValidationSettings
    )
    symbols: List[str] = Field(
        default_factory=lambda: [
            "BTC/USD",
            "ETH/USD",
            "SOL/USD",
            "ADA/USD",
            "DOT/USD",
            "LINK/USD",
            "UNI/USD",
            "AAVE/USD",
            "AVAX/USD",
            "MATIC/USD",
            "ATOM/USD",
            "NEAR/USD",
        ]
    )
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    testing_mode: bool = True
    timeframe: str = "15m"
    use_enhanced_ohlcv_fetcher: bool = True
    use_trade_manager_as_source: bool = True
    use_websocket: bool = True
    websocket: WebsocketSettings = Field(default_factory=WebsocketSettings)
    performance_optimization: PerformanceOptimizationSettings = Field(
        default_factory=PerformanceOptimizationSettings
    )
    profitability_optimization: ProfitabilityOptimizationSettings = Field(
        default_factory=ProfitabilityOptimizationSettings
    )
    default_regime: str = "trending"
    token_discovery_feed: Dict[str, Any] = Field(default_factory=dict)
    optimization: Dict[str, Any] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        extra="forbid",
        env_prefix="BOT_",
        env_nested_delimiter="__",
        env_file=(".env",),
        env_file_encoding="utf-8",
    )


def resolve_config_path(
    config_path: Union[str, Path, None] = None,
    *,
    env: Union[Mapping[str, str], None] = None,
) -> Path:
    """Return the configuration path to use for overrides."""

    env_mapping = env or os.environ
    if config_path is not None:
        return Path(config_path)
    env_value = env_mapping.get(CONFIG_PATH_ENV)
    if env_value:
        return Path(env_value)
    return DEFAULT_CONFIG_PATH


def _load_yaml_file(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - best effort
        raise ConfigError(f"Invalid YAML configuration in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {path} must contain a mapping")
    return data


def load_settings(
    config_path: Union[str, Path, None] = None,
    *,
    env: Union[Mapping[str, str], None] = None,
) -> BotSettings:
    """Load bot settings by combining defaults, YAML overrides and environment."""

    path = resolve_config_path(config_path, env=env)
    yaml_values = _load_yaml_file(path)
    managed_service = ManagedConfigService(
        manifest_path=MANAGED_SECRETS_PATH,
        env=env,
    )
    try:
        merged_values = managed_service.merge(yaml_values)
    except SecretNotFoundError as exc:
        raise ConfigError(str(exc)) from exc

    missing_env = managed_service.missing_environment()
    if missing_env:
        missing = ", ".join(sorted(missing_env))
        raise ConfigError(
            "Missing required managed secrets: "
            f"{missing}"
        )

    try:
        return BotSettings(**merged_values)
    except ValidationError as exc:  # pragma: no cover - exercised via unit tests
        raise ConfigError(
            f"Invalid configuration supplied by {path}: {exc}") from exc


def load_config(
    config_path: Union[str, Path, None] = None,
    *,
    env: Union[Mapping[str, str], None] = None,
    by_alias: bool = True,
) -> Dict[str, object]:
    """Return the configuration as a plain dictionary."""

    settings = load_settings(config_path=config_path, env=env)
    return settings.model_dump(mode="python", by_alias=by_alias)
