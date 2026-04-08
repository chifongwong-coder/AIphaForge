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
        return self

    def set_signals(self, signals: pd.Series) -> 'BacktestEngine':
        """
        Set pre-computed trading signals directly.

        Parameters:
            signals: Signal series (index=time, values={1, -1, 0}).

        Returns:
            self: For method chaining.
        """
        self._signals = signals
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
        data: pd.DataFrame,
        start: Optional[str] = None,
        end: Optional[str] = None,
        symbol: str = "default",
        *,
        benchmark: Optional[pd.Series] = None,
        benchmark_type: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run the backtest.

        Parameters:
            data: OHLCV data with open, high, low, close, volume columns.
            start: Start date (optional).
            end: End date (optional).
            symbol: Instrument symbol.
            benchmark: Custom benchmark series (prices or returns).
                Overrides the value set in BacktestConfig when provided.
            benchmark_type: Type of benchmark data — ``"prices"``,
                ``"returns"``, or ``"auto"``.  Overrides BacktestConfig
                when provided.

        Returns:
            BacktestResult: Backtest results.
        """
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
        )

        # Guard: multiple LatencyHook instances wrapping the same inner_hook
        latency_hooks = [h for h in self.hooks if isinstance(h, LatencyHook)]
        if len(latency_hooks) > 1:
            inner_ids: list[int] = []
            for lh in latency_hooks:
                iid = id(lh.inner_hook)
                if iid in inner_ids:
                    raise ValueError(
                        "Multiple LatencyHook instances wrap the same "
                        "inner_hook. Use a single LatencyHook per agent, or "
                        "SymbolRoutingLatencyHook for per-symbol latency."
                    )
                inner_ids.append(iid)

        # Dispatch to execution core
        if self.mode == ExecutionMode.VECTORIZED:
            raw = run_vectorized(data, signals, config, symbol)
        else:
            raw = run_event_driven(
                data, signals, config, symbol,
                strategy=self._strategy, full_data=self._data,
            )

        return self._build_result(raw, data, config)

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
        """Get trading signals."""
        if self._signals is not None:
            signals = self._signals.reindex(data.index).fillna(0)
        elif self._strategy is not None:
            signals = self._strategy.generate_signals(data)
        else:
            raise ValueError("Must set either a strategy or signals")

        signals = signals.replace([np.inf, -np.inf], 0).fillna(0)
        return signals

    # ========== Config and Result Building ==========

    def _build_config(
        self,
        benchmark: Optional[pd.Series] = None,
        benchmark_type: Optional[str] = None,
    ) -> BacktestConfig:
        """Build a BacktestConfig from the engine's attributes.

        Parameters:
            benchmark: Run-time benchmark override (takes precedence over
                the engine-level config).
            benchmark_type: Run-time benchmark_type override.
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

        return BacktestResult(**result_kwargs)

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
