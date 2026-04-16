"""
Backtest Engine

Main backtest executor supporting both vectorized and event-driven modes.
"""

import warnings
from datetime import time
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .broker import FillModel
from .config import BacktestConfig
from .core_event_driven import run_event_driven
from .core_vectorized import run_vectorized
from .costs import DefaultTradeCost
from .exit_rules import PercentageStopLoss, PercentageTakeProfit
from .fees import BaseFeeModel, SimpleFeeModel, get_fee_model
from .hooks import BacktestHook
from .latency import LatencyHook
from .position_sizing import AllInSizer, FixedSizer, FractionSizer
from .results import BacktestResult, Trade

# Import utility functions
from .utils import (
    TRADING_DAYS_STOCK,
    annualize_return,
    calculate_trade_metrics,
    compute_buy_and_hold,
    ensure_datetime_index,
    validate_ohlcv,
)
from .utils import (
    max_drawdown as calc_max_drawdown,
)
from .utils import (
    sharpe_ratio as calc_sharpe,
)
from .utils import (
    sortino_ratio as calc_sortino,
)


class ExecutionMode(Enum):
    """Execution mode."""
    VECTORIZED = "vectorized"        # Vectorized mode (fast)
    EVENT_DRIVEN = "event_driven"    # Event-driven mode (precise)


class PositionSizing(Enum):
    """Position sizing method."""
    FIXED_FRACTION = "fixed_fraction"  # Fixed fraction of equity
    FIXED_SIZE = "fixed_size"          # Fixed quantity
    ALL_IN = "all_in"                  # Full position
    RISK_BASED = "risk_based"          # Risk-based sizing


class BacktestEngine:
    """
    Backtest engine supporting vectorized and event-driven execution modes.

    Parameters:
        fee_model: Fee model instance.
        initial_capital: Starting capital.
        mode: Execution mode.
        position_sizing: Position sizing method.
        position_size: Position size (fraction or fixed quantity).
        max_position_size: Maximum single position as fraction of equity.
        stop_loss: Stop loss percentage.
        take_profit: Take profit percentage.
        allow_short: Whether short selling is allowed.
        fill_model: Fill model (event-driven mode).
        risk_manager: External risk manager (optional).
        agent_expert: AI Agent expert (optional).
        agent_trigger_interval: Agent trigger interval (every N bars).
        agent_enabled_strategies: Agent-controlled strategy enable states.
        hooks: List of backtest hooks (optional).
        include_benchmark: Whether to compute buy-and-hold benchmark.

    Example:
        >>> engine = BacktestEngine(
        ...     fee_model=ChinaAShareFeeModel(),
        ...     initial_capital=100000,
        ...     stop_loss=0.05
        ... )
        >>> engine.set_strategy(my_strategy)
        >>> results = engine.run(data)
        >>> print(results.summary())
    """

    def __init__(
        self,
        fee_model: Optional[BaseFeeModel] = None,
        initial_capital: float = 100000,
        mode: Union[str, ExecutionMode] = ExecutionMode.VECTORIZED,
        position_sizing: Union[str, PositionSizing] = PositionSizing.FIXED_FRACTION,
        position_size: float = 0.95,
        max_position_size: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        allow_short: bool = True,
        fill_model: FillModel = FillModel.NEXT_BAR_OPEN,
        risk_manager=None,
        agent_expert=None,
        agent_trigger_interval: int = 1,
        agent_enabled_strategies: Optional[Dict[str, bool]] = None,
        hooks: Optional[List[BacktestHook]] = None,
        include_benchmark: bool = True,
        fee_allocation: str = "proportional",
        data_validation: str = "warn",
        session_end_time: Optional[time] = None,
        immediate_fill_price: str = "close",
        capital_allocator=None,
        asset_fee_models: Optional[Dict] = None,
        asset_fill_models: Optional[Dict] = None,
        margin_config=None,
        asset_margin_configs: Optional[Dict] = None,
        periodic_cost_model=None,
        portfolio_exit_rules: Optional[List] = None,
        lot_size: int = 1,
        asset_lot_sizes: Optional[Dict] = None,
        max_position_pct: float = 1.0,
        asset_max_position_pcts: Optional[Dict] = None,
        signal_transform=None,
        turnover_config=None,
        risk_rules=None,
        trailing_stop_rule=None,
        initial_universe: Optional[List[str]] = None,
        impact_model=None,
        impact_adv_lookback: int = 20,
        impact_vol_lookback: int = 20,
    ):
        # Fee model
        if isinstance(fee_model, str):
            self.fee_model = get_fee_model(fee_model)
        else:
            self.fee_model = fee_model or SimpleFeeModel()

        # Capital
        self.initial_capital = initial_capital

        # Execution mode
        if isinstance(mode, str):
            mode = ExecutionMode(mode.lower())
        self.mode = mode

        # Position sizing
        if isinstance(position_sizing, str):
            position_sizing = PositionSizing(position_sizing.lower())
        self.position_sizing = position_sizing
        self.position_size = position_size
        self.max_position_size = max_position_size

        # Risk management
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.allow_short = allow_short

        # Fill model
        self.fill_model = fill_model

        # Risk manager (optional)
        self.risk_manager = risk_manager
        if risk_manager:
            risk_manager.initialize(initial_capital)

        # AI Agent (optional)
        self.agent_expert = agent_expert
        self.agent_trigger_interval = agent_trigger_interval
        self.agent_enabled_strategies = agent_enabled_strategies or {}
        self._agent_bar_count = 0

        # Hooks (optional)
        self.hooks: List[BacktestHook] = hooks or []

        # Benchmark
        self.include_benchmark = include_benchmark

        # Fee allocation for partial close trades
        self.fee_allocation = fee_allocation

        # Data validation level ('strict', 'warn', 'none')
        self.data_validation = data_validation

        # Session end time for DAY order expiration
        self.session_end_time = session_end_time

        # Fill price for same-bar IOC/FOK second pass
        self.immediate_fill_price = immediate_fill_price

        # Multi-asset (v0.7)
        self.capital_allocator = capital_allocator
        self.asset_fee_models: Dict = asset_fee_models or {}
        self.asset_fill_models: Dict = asset_fill_models or {}

        # Margin (v0.8)
        self.margin_config = margin_config
        self.asset_margin_configs: Dict = asset_margin_configs or {}
        self.periodic_cost_model = periodic_cost_model
        self.portfolio_exit_rules: List = portfolio_exit_rules or []

        # Lot sizes (v0.8)
        if not isinstance(lot_size, int) or lot_size < 1:
            raise ValueError(f"lot_size must be an int >= 1, got {lot_size!r}")
        self.lot_size = lot_size
        self.signal_transform = signal_transform
        self.turnover_config = turnover_config
        self.risk_rules = risk_rules
        self.trailing_stop_rule = trailing_stop_rule
        self.initial_universe = initial_universe

        # Market impact (v1.9.4)
        self.impact_model = impact_model
        self.impact_adv_lookback = impact_adv_lookback
        self.impact_vol_lookback = impact_vol_lookback
        self.asset_lot_sizes: Dict = asset_lot_sizes or {}
        for sym, ls in self.asset_lot_sizes.items():
            if not isinstance(ls, int) or ls < 1:
                raise ValueError(
                    f"lot_size for '{sym}' must be an int >= 1, got {ls!r}")

        # Per-asset position limits (v0.8)
        if not 0 < max_position_pct <= 1.0:
            raise ValueError(
                f"max_position_pct must be in (0, 1.0], got {max_position_pct}")
        self.max_position_pct = max_position_pct
        self.asset_max_position_pcts: Dict = asset_max_position_pcts or {}
        for sym, pct in self.asset_max_position_pcts.items():
            if not 0 < pct <= 1.0:
                raise ValueError(
                    f"max_position_pct for '{sym}' must be in (0, 1.0], "
                    f"got {pct}")

        # Custom benchmark config defaults
        self._config_benchmark: Optional[pd.Series] = None
        self._config_benchmark_type: str = "auto"
        self._config_benchmark_name: str = "Buy & Hold"

        # Feature modules
        self._stop_loss_rule = (
            PercentageStopLoss(stop_loss) if stop_loss else None
        )
        self._take_profit_rule = (
            PercentageTakeProfit(take_profit) if take_profit else None
        )
        self._trade_cost = DefaultTradeCost()
        self._position_sizer = self._create_position_sizer()

        # Internal state
        self._strategy = None
        self._signals = None
        self._data = None
        self._target_weights = None

    def _create_position_sizer(self):
        """Create the appropriate position sizer based on config."""
        if self.position_sizing == PositionSizing.FIXED_SIZE:
            return FixedSizer(self.position_size)
        elif self.position_sizing == PositionSizing.ALL_IN:
            return AllInSizer(self.position_size)
        elif self.position_sizing == PositionSizing.FIXED_FRACTION:
            return FractionSizer(self.position_size)
        else:
            # RISK_BASED falls back to FractionSizer
            warnings.warn(
                "RISK_BASED position sizing is not yet implemented, "
                "falling back to FIXED_FRACTION"
            )
            return FractionSizer(self.position_size)

    # ========== Setup Methods ==========

    def set_strategy(self, strategy) -> 'BacktestEngine':
        """
        Set the trading strategy.

        Parameters:
            strategy: Strategy object with a ``generate_signals`` method.

        Returns:
            self: For method chaining.
        """
        self._strategy = strategy
        self._signals = None
        self._target_weights = None
        return self

    def set_signals(
        self, signals: Union[pd.Series, Dict[str, pd.Series]],
    ) -> 'BacktestEngine':
        """
        Set pre-computed trading signals directly.

        Parameters:
            signals: Signal series (single-asset) or dict of signal
                series keyed by symbol (multi-asset).

        Returns:
            self: For method chaining.
        """
        self._signals = signals
        self._strategy = None
        self._target_weights = None
        return self

    def set_target_weights(
        self,
        weights_schedule: Dict[str, Dict[str, float]],
    ) -> 'BacktestEngine':
        """Set target portfolio weights for rebalancing.

        Parameters:
            weights_schedule: Mapping of date string to per-symbol
                weight dict.  Example::

                    {
                        "2024-01-01": {"AAPL": 0.3, "TSLA": 0.7},
                        "2024-02-01": {"AAPL": 0.5, "TSLA": 0.5},
                    }

                Between rebalance dates, positions are held (NaN signal).
                Weight=0 on a rebalance date closes the position.

        Returns:
            self: For method chaining.
        """
        self._target_weights = weights_schedule
        self._signals = None
        self._strategy = None
        return self

    def set_fee_model(self, fee_model: Union[BaseFeeModel, str]) -> 'BacktestEngine':
        """
        Set the fee model.

        Parameters:
            fee_model: Fee model instance or market name string.

        Returns:
            self: For method chaining.
        """
        if isinstance(fee_model, str):
            self.fee_model = get_fee_model(fee_model)
        else:
            self.fee_model = fee_model
        return self

    # ========== Run Methods ==========

    def run(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        symbol: str = "default",
        *,
        benchmark: Optional[pd.Series] = None,
        benchmark_type: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        secondary_data: Optional[Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]] = None,
        secondary_bar_align: str = "close",
    ) -> BacktestResult:
        """
        Run the backtest.

        Parameters:
            data: OHLCV data (single-asset ``pd.DataFrame``) or dict of
                DataFrames keyed by symbol (multi-asset).
            start: Start date (optional, single-asset only).
            end: End date (optional, single-asset only).
            symbol: Instrument symbol (single-asset only).
            benchmark: Custom benchmark series (prices or returns).
            benchmark_type: Type of benchmark data.
            weights: Per-symbol weights for vectorized multi-asset.
            secondary_data: Secondary timeframe data (event-driven only).
                Mapping of timeframe name to DataFrame (global) or dict
                of per-symbol DataFrames.
            secondary_bar_align: Alignment mode for secondary bars.
                ``"close"`` (default) or ``"open"``.

        Returns:
            BacktestResult: Backtest results.
        """
        is_multi = isinstance(data, dict)

        # --- Multi-asset path ---
        if is_multi:
            return self._run_multi(
                data, benchmark=benchmark,
                benchmark_type=benchmark_type, weights=weights,
                secondary_data=secondary_data,
                secondary_bar_align=secondary_bar_align,
            )

        # --- Single-asset path ---
        # Validate and prepare data
        data = self._prepare_data(data, start, end)
        self._data = data

        # Reset per-run state
        self._agent_bar_count = 0

        # Generate signals
        signals = self._get_signals(data)

        # Build config bundle (with run-time benchmark overrides)
        config = self._build_config(
            benchmark=benchmark,
            benchmark_type=benchmark_type,
            symbols=[symbol],
        )

        # Guard: multiple LatencyHook instances wrapping the same inner_hook
        latency_hooks = [h for h in self.hooks if isinstance(h, LatencyHook)]
        if len(latency_hooks) > 1:
            inner_ids: List[int] = []
            for lh in latency_hooks:
                iid = id(lh.inner_hook)
                if iid in inner_ids:
                    raise ValueError(
                        "Multiple LatencyHook instances wrap the same "
                        "inner_hook. Use a single LatencyHook per agent, or "
                        "SymbolRoutingLatencyHook for per-symbol latency."
                    )
                inner_ids.append(iid)

        # Validate secondary data
        if secondary_data is not None:
            for tf_name, tf_data in secondary_data.items():
                if isinstance(tf_data, pd.DataFrame):
                    validate_ohlcv(
                        tf_data,
                        required=['open', 'high', 'low', 'close'],
                        validation_level=self.data_validation,
                    )
                else:
                    for sym_name, sdf in tf_data.items():
                        validate_ohlcv(
                            sdf,
                            required=['open', 'high', 'low', 'close'],
                            validation_level=self.data_validation,
                        )

        # Dispatch to execution core
        if self.mode == ExecutionMode.VECTORIZED:
            raw = run_vectorized(data, signals, config, symbol)
        else:
            # Wrap single-asset as dict for the unified core
            raw = run_event_driven(
                data_dict={symbol: data},
                signals_dict={symbol: signals},
                config=config,
                symbols=[symbol],
                strategy=self._strategy,
                secondary_data=secondary_data,
                secondary_bar_align=secondary_bar_align,
            )

        return self._build_result(raw, data, config)

    def _run_multi(
        self,
        data_dict: Dict[str, pd.DataFrame],
        *,
        benchmark: Optional[pd.Series] = None,
        benchmark_type: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        secondary_data: Optional[Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]] = None,
        secondary_bar_align: str = "close",
    ) -> BacktestResult:
        """Run a multi-asset backtest."""
        from .capital_allocator import EqualWeightAllocator

        symbols = sorted(data_dict.keys())

        # Validate each asset's data
        for sym, df in data_dict.items():
            validate_ohlcv(
                df,
                required=['open', 'high', 'low', 'close'],
                validation_level=self.data_validation,
            )
            data_dict[sym] = ensure_datetime_index(df).sort_index().copy()

        # Validate secondary data
        if secondary_data is not None:
            for tf_name, tf_data in secondary_data.items():
                if isinstance(tf_data, pd.DataFrame):
                    validate_ohlcv(
                        tf_data,
                        required=['open', 'high', 'low', 'close'],
                        validation_level=self.data_validation,
                    )
                else:
                    for sym_name, sdf in tf_data.items():
                        validate_ohlcv(
                            sdf,
                            required=['open', 'high', 'low', 'close'],
                            validation_level=self.data_validation,
                        )

        # Generate signals
        signals_dict = self._get_signals_multi(data_dict)

        # Build config
        config = self._build_config(
            benchmark=benchmark,
            benchmark_type=benchmark_type,
            symbols=symbols,
        )
        if self._target_weights is not None:
            config.is_weight_mode = True

        # Auto-set allocator for multi-asset if not provided
        if config.capital_allocator is None:
            if config.margin_config is not None:
                from .capital_allocator import MarginAllocator
                warnings.warn(
                    "No capital_allocator set for multi-asset margin mode. "
                    "Using MarginAllocator (buying_power based). "
                    "Set capital_allocator explicitly to suppress."
                )
                config.capital_allocator = MarginAllocator()
            else:
                warnings.warn(
                    "No capital_allocator set for multi-asset mode. "
                    "Using EqualWeightAllocator (equal budget per signal). "
                    "Set capital_allocator explicitly to suppress."
                )
                config.capital_allocator = EqualWeightAllocator()

        # Dispatch
        if self.mode == ExecutionMode.VECTORIZED:
            raw = self._run_vectorized_multi(
                data_dict, signals_dict, config, weights)
        else:
            raw = run_event_driven(
                data_dict=data_dict,
                signals_dict=signals_dict,
                config=config,
                symbols=symbols,
                strategy=self._strategy,
                secondary_data=secondary_data,
                secondary_bar_align=secondary_bar_align,
            )

        # Build result (use first asset's data for benchmark alignment)
        first_df = data_dict[symbols[0]]
        result = self._build_result(raw, first_df, config)

        # Attach multi-asset fields
        if 'per_asset_pnl' in raw:
            result.per_asset_pnl = raw['per_asset_pnl']
        result.symbols = symbols

        # Group trades by symbol
        if result.trades:
            per_asset_trades = {}
            for t in result.trades:
                per_asset_trades.setdefault(t.symbol, []).append(t)
            result.per_asset_trades = per_asset_trades

        return result

    def _get_signals_multi(
        self,
        data_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.Series]:
        """Get signals for multi-asset mode."""
        # Target weights mode: convert schedule to signal series
        if self._target_weights is not None:
            return self._weights_to_signals(
                self._target_weights, data_dict)

        if isinstance(self._signals, dict):
            signals_dict = {}
            for sym, df in data_dict.items():
                if sym in self._signals:
                    sig = self._signals[sym].reindex(df.index)
                else:
                    sig = pd.Series(np.nan, index=df.index, dtype=float)
                signals_dict[sym] = sig.replace(
                    [np.inf, -np.inf], np.nan)
            return signals_dict
        elif self._strategy is not None:
            result = self._strategy.generate_signals(data_dict)
            if isinstance(result, dict):
                return result
            raise TypeError(
                "Strategy.generate_signals() must return "
                "Dict[str, pd.Series] for multi-asset mode"
            )
        else:
            raise ValueError(
                "Must set either a strategy or signals (via set_signals "
                "or set_strategy) before running a multi-asset backtest"
            )

    @staticmethod
    def _weights_to_signals(
        weights_schedule: Dict[str, Dict[str, float]],
        data_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.Series]:
        """Convert target weight schedule to per-symbol signal Series.

        Between rebalance dates: NaN (hold). On rebalance dates: weight value.
        """
        all_syms = set()
        for w_dict in weights_schedule.values():
            all_syms.update(w_dict.keys())
        all_syms.update(data_dict.keys())

        signals = {}
        for sym in data_dict:
            sig = pd.Series(np.nan, index=data_dict[sym].index, dtype=float)
            for date_str, w_dict in weights_schedule.items():
                ts = pd.Timestamp(date_str)
                if ts in sig.index:
                    sig.loc[ts] = w_dict.get(sym, 0.0)
            signals[sym] = sig
        return signals

    def _run_vectorized_multi(
        self,
        data_dict: Dict[str, pd.DataFrame],
        signals_dict: Dict[str, pd.Series],
        config: BacktestConfig,
        weights: Optional[Dict[str, float]] = None,
    ) -> dict:
        """Run vectorized multi-asset: per-asset runs + merge."""
        import dataclasses

        symbols = sorted(data_dict.keys())
        if weights is None:
            weights = {s: 1.0 / len(symbols) for s in symbols}

        # Validate weights
        for s, w in weights.items():
            if w <= 0:
                raise ValueError(
                    f"Weight for '{s}' must be > 0, got {w}")
        total_w = sum(weights.values())
        if total_w > 1.0 + 1e-9:
            raise ValueError(
                f"Sum of weights ({total_w:.4f}) exceeds 1.0")
        if total_w < 1.0 - 1e-9:
            warnings.warn(
                f"Sum of weights ({total_w:.4f}) < 1.0. "
                f"{1 - total_w:.1%} of capital held as cash."
            )

        per_asset = {}
        for sym in symbols:
            w = weights.get(sym, 0)
            asset_capital = config.initial_capital * w
            asset_config = dataclasses.replace(
                config, initial_capital=asset_capital)
            per_asset[sym] = run_vectorized(
                data_dict[sym], signals_dict[sym],
                asset_config, sym)

        return self._merge_vectorized_results(
            per_asset, config.initial_capital)

    @staticmethod
    def _merge_vectorized_results(
        per_asset: Dict[str, dict],
        initial_capital: float,
    ) -> dict:
        """Merge per-asset vectorized results into a portfolio result."""
        # Align and sum equity curves
        equity_curves = {}
        all_trades = []
        all_orders = []
        for sym, raw in per_asset.items():
            eq = raw['equity_curve']
            equity_curves[sym] = eq
            all_trades.extend(raw.get('trades', []))
            odf = raw.get('orders_df')
            if odf is not None and len(odf) > 0:
                all_orders.append(odf)

        eq_df = pd.DataFrame(equity_curves)
        # Forward-fill and sum
        eq_df = eq_df.ffill()
        portfolio_equity = eq_df.sum(axis=1)

        orders_df = (pd.concat(all_orders, ignore_index=True)
                     if all_orders else pd.DataFrame())

        from .utils import calculate_returns
        daily_returns = (calculate_returns(portfolio_equity)
                         if len(portfolio_equity) > 0 else None)

        # Per-asset PnL from independent equity curves
        per_asset_pnl = {}
        for sym, eq in equity_curves.items():
            per_asset_pnl[sym] = eq.diff().fillna(0.0)
            per_asset_pnl[sym].name = sym

        return {
            'equity_curve': portfolio_equity,
            'trades': all_trades,
            'positions_df': pd.DataFrame(),
            'orders_df': orders_df,
            'daily_returns': daily_returns,
            'final_capital': (float(portfolio_equity.iloc[-1])
                              if len(portfolio_equity) > 0 else 0.0),
            'per_asset_pnl': per_asset_pnl,
        }

    def _prepare_data(
        self,
        data: pd.DataFrame,
        start: Optional[str],
        end: Optional[str]
    ) -> pd.DataFrame:
        """Validate and prepare data."""
        validate_ohlcv(
            data,
            required=['open', 'high', 'low', 'close'],
            validation_level=self.data_validation,
        )
        data = ensure_datetime_index(data)
        data = data.sort_index()

        if start:
            data = data[data.index >= pd.Timestamp(start)]
        if end:
            data = data[data.index <= pd.Timestamp(end)]

        if len(data) == 0:
            raise ValueError("No data after date filtering")

        return data.copy()

    def _get_signals(self, data: pd.DataFrame) -> pd.Series:
        """Get trading signals. NaN = hold, 0 = flat, nonzero = trade."""
        if self._signals is not None:
            signals = self._signals.reindex(data.index)
            # NaN from reindex means "no signal" = hold (preserve NaN)
        elif self._strategy is not None:
            signals = self._strategy.generate_signals(data)
        else:
            raise ValueError("Must set either a strategy or signals")

        signals = signals.replace([np.inf, -np.inf], np.nan)
        return signals

    # ========== Config and Result Building ==========

    def _build_config(
        self,
        benchmark: Optional[pd.Series] = None,
        benchmark_type: Optional[str] = None,
        symbols: Optional[List[str]] = None,
    ) -> BacktestConfig:
        """Build a BacktestConfig from the engine's attributes.

        Parameters:
            benchmark: Run-time benchmark override (takes precedence over
                the engine-level config).
            benchmark_type: Run-time benchmark_type override.
            symbols: List of symbols for this run.
        """
        return BacktestConfig(
            initial_capital=self.initial_capital,
            fee_model=self.fee_model,
            allow_short=self.allow_short,
            fee_allocation=self.fee_allocation,
            fill_model=self.fill_model,
            stop_loss_rule=self._stop_loss_rule,
            take_profit_rule=self._take_profit_rule,
            trade_cost=self._trade_cost,
            position_sizer=self._position_sizer,
            risk_manager=self.risk_manager,
            hooks=self.hooks,
            include_benchmark=self.include_benchmark,
            data_validation=self.data_validation,
            max_position_size=self.max_position_size,
            session_end_time=self.session_end_time,
            immediate_fill_price=self.immediate_fill_price,
            mode=self.mode.value,
            has_signals=self._signals is not None,
            has_strategy=self._strategy is not None,
            benchmark=benchmark if benchmark is not None else self._config_benchmark,
            benchmark_type=benchmark_type if benchmark_type is not None else self._config_benchmark_type,
            benchmark_name=self._config_benchmark_name,
            symbols=symbols or [],
            capital_allocator=self.capital_allocator,
            asset_fee_models=self.asset_fee_models,
            asset_fill_models=self.asset_fill_models,
            margin_config=self.margin_config,
            asset_margin_configs=self.asset_margin_configs,
            periodic_cost_model=self.periodic_cost_model,
            portfolio_exit_rules=self.portfolio_exit_rules,
            lot_size=self.lot_size,
            asset_lot_sizes=self.asset_lot_sizes,
            max_position_pct=self.max_position_pct,
            asset_max_position_pcts=self.asset_max_position_pcts,
            signal_transform=self.signal_transform,
            turnover_config=self.turnover_config,
            risk_rules=self.risk_rules,
            trailing_stop_rule=self.trailing_stop_rule,
            initial_universe=self.initial_universe,
            impact_model=self.impact_model,
            impact_adv_lookback=self.impact_adv_lookback,
            impact_vol_lookback=self.impact_vol_lookback,
        )

    def _build_result(
        self,
        raw: dict,
        data: pd.DataFrame,
        config: Optional[BacktestConfig] = None,
    ) -> BacktestResult:
        """Build a BacktestResult from raw core output."""
        equity_curve = raw['equity_curve']
        trades = raw['trades']
        positions_df = raw['positions_df']
        net_returns = raw.get('net_returns')
        daily_returns = raw.get('daily_returns')
        orders_df = raw.get('orders_df', pd.DataFrame())
        final_capital = raw.get('final_capital', 0.0)

        # Determine which returns series to use for metrics
        returns_for_metrics = net_returns if net_returns is not None else daily_returns

        # Compute metrics
        if returns_for_metrics is not None and len(returns_for_metrics) > 0:
            metrics = self._calculate_metrics(returns_for_metrics, equity_curve, trades)
        else:
            metrics = {}

        # Strategy name
        strategy_name = (
            getattr(self._strategy, 'name', 'Custom')
            if self._strategy else "Custom"
        )

        # Benchmark
        benchmark_equity = None
        benchmark_metrics = None
        benchmark_name = "Buy & Hold"
        if self.include_benchmark:
            benchmark_equity, benchmark_metrics, benchmark_name = (
                self._compute_benchmark(data, config)
            )

        # Use net_returns or daily_returns for the result
        result_returns = net_returns if net_returns is not None else daily_returns

        result_kwargs = dict(
            equity_curve=equity_curve,
            trades=trades,
            positions=positions_df,
            metrics=metrics,
            initial_capital=self.initial_capital,
            strategy_name=strategy_name,
            parameters=(
                getattr(self._strategy, 'params', {})
                if self._strategy else {}
            ),
            daily_returns=result_returns,
            benchmark_equity=benchmark_equity,
            benchmark_metrics=benchmark_metrics,
            benchmark_name=benchmark_name,
        )

        if orders_df is not None and len(orders_df) > 0:
            result_kwargs['orders'] = orders_df

        if final_capital is not None:
            result_kwargs['final_capital'] = final_capital

        result = BacktestResult(**result_kwargs)

        # Attach turnover history if present
        if 'turnover_history' in raw:
            result.turnover_history = raw['turnover_history']

        # Attach MetaContext audit trail (v1.2)
        if 'meta_audit' in raw and raw['meta_audit']:
            result.metadata['meta_audit'] = raw['meta_audit']

        return result

    # ========== Performance Calculation ==========

    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity: pd.Series,
        trades: List[Trade]
    ) -> Dict[str, float]:
        """Calculate performance metrics.

        Delegates to shared utility functions so that the engine and
        PerformanceAnalyzer use the same calculations.
        """
        metrics: Dict[str, float] = {}

        if len(returns) == 0:
            return metrics

        # --- Return metrics ---
        if len(equity) > 0 and equity.iloc[0] != 0:
            total_return = equity.iloc[-1] / equity.iloc[0] - 1
        else:
            total_return = 0.0
        metrics['total_return'] = total_return

        n_days = len(returns)
        metrics['annualized_return'] = (
            annualize_return(total_return, n_days, TRADING_DAYS_STOCK) if n_days > 0 else 0.0
        )

        # --- Risk metrics ---
        metrics['sharpe_ratio'] = calc_sharpe(returns, trading_days=TRADING_DAYS_STOCK)
        metrics['sortino_ratio'] = calc_sortino(returns, trading_days=TRADING_DAYS_STOCK)
        metrics['max_drawdown'] = calc_max_drawdown(equity)
        metrics['calmar_ratio'] = (
            metrics['annualized_return'] / metrics['max_drawdown']
            if metrics['max_drawdown'] > 0 else 0.0
        )

        # --- Trade metrics (delegated to utils) ---
        metrics.update(calculate_trade_metrics(trades))

        # --- Simple inline metrics ---
        metrics['volatility'] = float(returns.std() * np.sqrt(TRADING_DAYS_STOCK))
        metrics['mean_daily_return'] = float(returns.mean())
        metrics['win_days'] = int((returns > 1e-8).sum())
        metrics['lose_days'] = int((returns < -1e-8).sum())
        metrics['flat_days'] = int(len(returns) - metrics['win_days'] - metrics['lose_days'])

        return metrics

    def _compute_benchmark(
        self,
        data: pd.DataFrame,
        config: Optional[BacktestConfig] = None,
    ) -> tuple:
        """Compute benchmark equity and metrics.

        If a custom benchmark series is available (via *config*), it is
        used after type detection and alignment.  Otherwise the default
        buy-and-hold benchmark is computed via :func:`compute_buy_and_hold`.

        Returns:
            (benchmark_equity, benchmark_metrics, benchmark_name) tuple.
        """
        custom = config.benchmark if config is not None else None
        btype = config.benchmark_type if config is not None else "auto"
        bname = config.benchmark_name if config is not None else "Buy & Hold"

        if custom is not None:
            # --- Determine benchmark type ---
            if btype == "auto":
                # Heuristic: all positive and minimum > 1.0 → prices
                if (custom > 0).all() and custom.min() > 1.0:
                    detected = "prices"
                else:
                    detected = "returns"
                warnings.warn(
                    f"benchmark_type='auto': detected as '{detected}'. "
                    "Consider specifying benchmark_type explicitly.",
                    stacklevel=2,
                )
                btype = detected

            # --- Convert to equity curve ---
            if btype == "prices":
                if custom.iloc[0] != 0:
                    benchmark_equity = custom / custom.iloc[0] * self.initial_capital
                else:
                    benchmark_equity = pd.Series(
                        self.initial_capital, index=custom.index
                    )
            else:
                # returns
                benchmark_equity = (1 + custom).cumprod() * self.initial_capital

            # --- Align to data index ---
            benchmark_equity = benchmark_equity.reindex(data.index).ffill()

            # Warn if >5% missing after alignment
            n_missing = int(benchmark_equity.isna().sum())
            if n_missing > 0:
                pct_missing = n_missing / len(data.index)
                if pct_missing > 0.05:
                    warnings.warn(
                        f"Custom benchmark has {pct_missing:.1%} missing values "
                        f"after alignment ({n_missing}/{len(data.index)} bars). "
                        "Results may be unreliable.",
                        stacklevel=2,
                    )
                # Fill any remaining leading NaN with the initial capital
                benchmark_equity = benchmark_equity.bfill()

        else:
            # Default: buy-and-hold
            benchmark_equity = compute_buy_and_hold(data, self.initial_capital)
            bname = "Buy & Hold"

        bh_returns = benchmark_equity.pct_change().fillna(0)
        benchmark_metrics = self._calculate_metrics(
            bh_returns, benchmark_equity, trades=[]
        )
        return benchmark_equity, benchmark_metrics, bname

    def __repr__(self):
        return (f"BacktestEngine(mode={self.mode.value}, "
                f"capital={self.initial_capital:,.0f}, "
                f"fee_model={self.fee_model.name})")


# ========== Convenience Functions ==========

def backtest(
    data: pd.DataFrame,
    strategy=None,
    signals: pd.Series = None,
    initial_capital: float = 100000,
    fee_model: Union[BaseFeeModel, str] = None,
    mode: str = "vectorized",
    stop_loss: float = None,
    benchmark: Optional[pd.Series] = None,
    benchmark_type: Optional[str] = None,
    benchmark_name: Optional[str] = None,
    **kwargs
) -> BacktestResult:
    """
    Convenience backtest function.

    Parameters:
        data: OHLCV data.
        strategy: Strategy object.
        signals: Signal series (mutually exclusive with strategy).
        initial_capital: Starting capital.
        fee_model: Fee model.
        mode: Execution mode.
        stop_loss: Stop loss percentage.
        benchmark: Custom benchmark series (prices or returns).
        benchmark_type: Benchmark type — ``"prices"``, ``"returns"``,
            or ``"auto"``.
        benchmark_name: Display name for the benchmark in results.
        **kwargs: Additional engine parameters.

    Returns:
        BacktestResult: Backtest results.

    Example:
        >>> result = backtest(data, strategy=MAStrategy())
        >>> result = backtest(data, signals=my_signals, fee_model='china')
    """
    engine = BacktestEngine(
        initial_capital=initial_capital,
        mode=mode,
        stop_loss=stop_loss,
        **kwargs
    )

    if fee_model:
        engine.set_fee_model(fee_model)

    if benchmark_name is not None:
        engine._config_benchmark_name = benchmark_name

    if strategy:
        engine.set_strategy(strategy)
    elif signals is not None:
        engine.set_signals(signals)
    else:
        raise ValueError("Must provide either strategy or signals")

    return engine.run(data, benchmark=benchmark, benchmark_type=benchmark_type)
