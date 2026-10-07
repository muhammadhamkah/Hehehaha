"""Central configuration.

All tunables live here as nested dataclasses. Values can be overridden by:
  1. a JSON file passed via ``--config`` (nested keys mirror the dataclasses), then
  2. selected environment variables (secrets and safety switches).

Safety: the bot defaults to ``dry_run=True``. Live order placement requires ALL of
  * mode == "live"
  * dry_run == False
  * env LIVE_TRADING_CONFIRM == "I_UNDERSTAND_THE_RISK"
  * API key + secret present
See :meth:`BotConfig.live_trading_enabled`.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

LIVE_CONFIRM_PHRASE = "I_UNDERSTAND_THE_RISK"
HORIZONS_S: tuple[int, ...] = (1, 3, 5, 10, 30, 60)


@dataclass
class ExchangeConfig:
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    rest_base: str = "https://fapi.binance.com"
    ws_base: str = "wss://fstream.binance.com"
    testnet_rest_base: str = "https://demo-fapi.binance.com"
    testnet_ws_base: str = "wss://fstream.binancefuture.com"
    recv_window_ms: int = 5000
    request_timeout_s: float = 10.0
    # Back off when the used request weight (per minute) approaches this.
    max_weight_per_min: int = 2000
    ws_reconnect_max_backoff_s: float = 30.0
    # Binance drops connections at 24h; reconnect proactively before that.
    ws_max_connection_age_s: float = 23 * 3600
    # Binance allows up to 1024 streams per futures connection; stay well below.
    ws_max_streams_per_connection: int = 200
    # Reconnect if a subscribed connection delivers no data for this long.
    ws_silence_timeout_s: float = 10.0

    @property
    def rest_url(self) -> str:
        return self.testnet_rest_base if self.testnet else self.rest_base

    @property
    def ws_url(self) -> str:
        return self.testnet_ws_base if self.testnet else self.ws_base


@dataclass
class ScannerConfig:
    quote_asset: str = "USDT"
    min_quote_volume_24h: float = 50_000_000.0
    max_spread_bps: float = 5.0
    top_n: int = 20
    # Incumbents stay selected until they fall below rank top_n * hysteresis.
    hysteresis: float = 1.5
    rescan_interval_s: float = 5.0
    volatility_window_s: float = 60.0
    activity_window_s: float = 60.0
    min_age_for_ranking_s: float = 10.0
    exclude_symbols: tuple[str, ...] = ()
    # If set, always analyse exactly these symbols (ranking still computed for logging).
    # Used for fixed-universe recording and for replaying archive data without tickers.
    static_symbols: tuple[str, ...] = ()
    # Ranking weights (applied to cross-sectional percentile ranks).
    w_volume: float = 1.0
    w_spread: float = 1.5
    w_volatility: float = 1.0
    w_activity: float = 1.0
    w_volume_accel: float = 1.0
    w_imbalance: float = 0.5


@dataclass
class MarketDataConfig:
    # "partial" -> <sym>@depth20@100ms snapshots (robust, no sync needed)
    # "diff"    -> <sym>@depth@100ms diffs + REST snapshot (full local book)
    # "bbo"     -> top of book only from bookTicker (L1 data, e.g. public archives)
    depth_mode: str = "partial"
    depth_levels: int = 20
    diff_snapshot_limit: int = 1000
    book_history_len: int = 600          # ~60s at 100ms
    # bbo (tick-level L1) mode conflates book history into buckets of this size so the
    # history covers the same ~60s as depth20@100ms; partial/diff modes record every update.
    bbo_history_interval_ms: int = 100
    trade_history_s: float = 120.0
    stale_after_ms: int = 2000


@dataclass
class FeatureConfig:
    imbalance_levels: tuple[int, ...] = (1, 5, 10)
    imbalance_decay: float = 0.7          # per-level weight decay for weighted imbalance
    depth_bands_bps: tuple[float, ...] = (5.0, 10.0, 25.0)
    flow_windows_s: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0)
    momentum_windows_s: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0)
    persistence_window: int = 20          # book snapshots
    depletion_lookback_s: float = 2.0
    vol_window_s: float = 30.0
    large_trade_quantile_mult: float = 3.0  # trade > mult * median size counts as large


@dataclass
class StrategyConfig:
    predictor: str = "rule"               # "rule" (V1), "linear" (V1 + trained linear) or "v2"
    v2_model_dir: str = "models/v2/lightgbm"
    v2_threshold: float | None = None     # None -> threshold selected during model research
    model_path: str = "models/linear_model.json"
    eval_interval_ms: int = 250
    # Rule-based score weights
    w_book_imbalance: float = 1.0
    w_ofi: float = 1.0
    w_flow_short: float = 1.5
    w_flow_long: float = 1.0
    w_microprice: float = 0.75
    w_momentum: float = 0.75
    w_persistence: float = 0.5
    w_depletion: float = 0.5
    logistic_k: float = 4.0
    # Drift per unit score expressed as a fraction of 1s volatility per second
    drift_per_score: float = 0.35
    max_hold_s: float = 60.0
    min_hold_s: float = 1.0
    # Signal recording (lower than entry threshold to build an unbiased dataset)
    record_score_threshold: float = 0.15
    baseline_sample_prob: float = 0.01
    record_min_interval_ms: int = 1000


@dataclass
class CostConfig:
    maker_fee: float = 0.0002            # 0.02%
    taker_fee: float = 0.0005            # 0.05%
    use_exchange_commission: bool = True # live: query /fapi/v1/commissionRate
    maker_entry_expected: bool = True    # cost model assumes maker entry if execution is maker-first
    maker_adverse_selection_bps: float = 0.5
    extra_slippage_bps: float = 0.5      # buffer on top of book-walk estimate
    latency_slippage_bps: float = 0.3


@dataclass
class EntryConfig:
    min_net_profit_usdt: float = 0.10
    min_confidence: float = 0.62          # directional probability
    min_p_target_before_stop: float = 0.55
    max_spread_bps: float = 3.0
    min_depth_usdt_within_10bps: float = 20_000.0
    max_expected_slippage_bps: float = 2.0
    min_target_bps: float = 6.0
    max_target_bps: float = 60.0
    # Candidate targets = required_move * multiplier; the one maximizing expected net wins.
    target_multipliers: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 3.0)
    stop_bps: float = 8.0
    stop_vol_mult: float = 0.0            # >0: stop = max(stop_bps, mult * sigma_horizon)
    require_flow_confirmation: bool = True
    # Directional thresholds for research sweeps; -1.0 disables them (default behaviour).
    min_book_imbalance: float = -1.0      # direction * imb_weighted must be >= this
    min_flow_imbalance: float = -1.0      # direction * flow_imb_3s must be >= this
    max_book_imbalance_abs: float = 0.97  # reject near-one-sided books (likely abnormal)
    min_liquidity_change: float = -0.5    # reject if top-10 depth fell >50% vs its 10s average
    min_trades_per_s: float = 1.0
    max_realized_vol_bps_1s: float = 15.0 # abnormal volatility guard


@dataclass
class SizingConfig:
    account_equity_usdt: float = 100.0
    position_notional_usdt: float = 150.0
    leverage: int = 5
    max_risk_per_trade_usdt: float = 1.0
    max_simultaneous_positions: int = 1
    margin_type: str = "ISOLATED"


@dataclass
class ExecutionConfig:
    entry_mode: str = "maker_first"       # "maker_first" | "taker"
    maker_ttl_ms: int = 1500
    # On TTL expiry: "skip" or "taker_if_edge" (re-check net edge with taker costs)
    ttl_fallback: str = "taker_if_edge"
    taker_max_slippage_bps: float = 3.0   # IOC price cap; never a blind market order
    exit_max_slippage_bps: float = 10.0
    emergency_market_exit: bool = True
    min_partial_fill_ratio: float = 0.3   # below this, a partial entry is unwound
    # Paper simulator
    sim_latency_ms: int = 50
    sim_queue_factor: float = 1.0         # fraction of visible level qty ahead of us
    sim_extra_taker_slippage_bps: float = 0.2


@dataclass
class ExitConfig:
    # "v1": break-even / trailing / flow-reversal / imbalance / exhaustion / time exits
    # "barrier": exactly the economics V2 is trained on -- TP at target, SL at stop,
    #            time stop at strategy.max_hold_s (+ emergencies)
    mode: str = "v1"
    # Break-even arms at max(trigger, net-break-even + buffer) so it never stops out instantly.
    break_even_trigger_bps: float = 10.0
    break_even_buffer_bps: float = 2.0
    trail_activate_bps: float = 14.0
    trail_distance_bps: float = 5.0
    take_profit_on_target_if_no_momentum: bool = True
    reversal_score: float = 0.35          # opposite-direction score triggering exit
    imbalance_loss_evals: int = 8         # consecutive evals with book against us
    exhaustion_velocity_ratio: float = 0.3
    time_stop_mult: float = 2.0           # x expected hold, capped by strategy.max_hold_s
    emergency_spread_bps: float = 15.0


@dataclass
class RiskConfig:
    daily_loss_limit_usdt: float = 3.0
    max_consecutive_losses: int = 5
    max_trades_per_hour: int = 30
    symbol_cooldown_after_loss_s: float = 300.0
    max_position_notional_usdt: float = 300.0
    max_leverage: int = 10
    max_spread_bps: float = 4.0
    max_slippage_bps: float = 3.0
    stale_data_ms: int = 2000
    max_api_errors: int = 5
    api_error_window_s: float = 60.0
    disconnect_halt_s: float = 10.0
    kill_switch_file: str = "KILL_SWITCH"
    # RESEARCH-ONLY: ignore the consecutive-loss halt so offline replays can observe the
    # full distribution of a losing strategy. Refused by the bot outside offline replay.
    research_mode: bool = False


@dataclass
class RecorderConfig:
    db_path: str = "data/microstructure.sqlite"
    record_raw_book: bool = False
    raw_book_interval_ms: int = 1000
    record_scanner: bool = True
    # Barrier labels should match a TRADEABLE outcome: target ~= required move for
    # min_net_profit at the configured notional, i.e. (0.10 + ~0.13 costs) / 150 ~= 16 bps.
    label_target_bps: float = 16.0
    label_stop_bps: float = 8.0
    flush_interval_s: float = 1.0
    trades_jsonl: str = "data/trades.jsonl"
    # Raw WebSocket event capture for exact replay (~3-6 GB/day gzipped at 30 symbols).
    record_events: bool = True
    events_dir: str = "data/events"


@dataclass
class BotConfig:
    mode: str = "paper"                   # "record" | "paper" | "live"
    dry_run: bool = True
    log_level: str = "INFO"
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)

    def live_trading_enabled(self) -> bool:
        return (
            self.mode == "live"
            and not self.dry_run
            and os.environ.get("LIVE_TRADING_CONFIRM") == LIVE_CONFIRM_PHRASE
            and bool(self.exchange.api_key)
            and bool(self.exchange.api_secret)
        )

    def trading_enabled(self) -> bool:
        """True when the bot should generate (simulated or live) trades."""
        return self.mode in ("paper", "live")

    def validate(self) -> list[str]:
        """Return a list of configuration problems (empty == OK)."""
        problems: list[str] = []
        if self.mode not in ("record", "paper", "live"):
            problems.append(f"unknown mode {self.mode!r}")
        if self.sizing.leverage > self.risk.max_leverage:
            problems.append("sizing.leverage exceeds risk.max_leverage")
        if self.sizing.position_notional_usdt > self.risk.max_position_notional_usdt:
            problems.append("sizing.position_notional_usdt exceeds risk.max_position_notional_usdt")
        margin = self.sizing.position_notional_usdt / max(self.sizing.leverage, 1)
        if margin > self.sizing.account_equity_usdt * 0.8:
            problems.append("position margin would use >80% of account equity")
        if self.sizing.max_simultaneous_positions < 1:
            problems.append("sizing.max_simultaneous_positions must be >= 1")
        if self.execution.ttl_fallback not in ("skip", "taker_if_edge"):
            problems.append("execution.ttl_fallback must be 'skip' or 'taker_if_edge'")
        if self.execution.entry_mode not in ("maker_first", "taker"):
            problems.append("execution.entry_mode must be 'maker_first' or 'taker'")
        if self.market_data.depth_mode not in ("partial", "diff", "bbo"):
            problems.append("market_data.depth_mode must be 'partial', 'diff' or 'bbo'")
        if self.exit.mode not in ("v1", "barrier"):
            problems.append("exit.mode must be 'v1' or 'barrier'")
        if self.strategy.predictor == "v2" and self.mode == "live":
            problems.append("V2 predictor is not approved for live trading (no credible out-of-sample edge yet)")
        if self.entry.min_net_profit_usdt <= 0:
            problems.append("entry.min_net_profit_usdt must be positive")
        if self.mode == "live" and not self.dry_run and not self.live_trading_enabled():
            problems.append(
                "live mode requested with dry_run=False but live trading is not fully "
                f"enabled (need API keys and LIVE_TRADING_CONFIRM={LIVE_CONFIRM_PHRASE})"
            )
        return problems

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["exchange"]["api_key"] = "***" if self.exchange.api_key else ""
        d["exchange"]["api_secret"] = "***" if self.exchange.api_secret else ""
        return d


def _merge(dc: Any, overrides: dict[str, Any]) -> None:
    names = {f.name: f for f in fields(dc)}
    for key, value in overrides.items():
        if key.startswith("_"):
            continue  # comment keys in JSON configs
        if key not in names:
            raise KeyError(f"unknown config key {key!r} for {type(dc).__name__}")
        current = getattr(dc, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(dc, key, tuple(value))
        else:
            setattr(dc, key, value)


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def load_config(path: str | None = None, overrides: dict[str, Any] | None = None) -> BotConfig:
    cfg = BotConfig()
    if path:
        with open(path, encoding="utf-8") as fh:
            _merge(cfg, json.load(fh))
    if overrides:
        _merge(cfg, overrides)

    cfg.exchange.api_key = os.environ.get("BINANCE_API_KEY", cfg.exchange.api_key)
    cfg.exchange.api_secret = os.environ.get("BINANCE_API_SECRET", cfg.exchange.api_secret)
    if (mode := os.environ.get("BOT_MODE")) is not None:
        cfg.mode = mode
    if (dry := _env_bool("DRY_RUN")) is not None:
        cfg.dry_run = dry
    if (testnet := _env_bool("BINANCE_TESTNET")) is not None:
        cfg.exchange.testnet = testnet
    return cfg
