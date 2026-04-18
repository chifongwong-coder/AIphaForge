"""
Backtest Hook Framework
=======================

Provides an extensible hook mechanism for the backtest engine.

Hooks are called after each bar is processed in event-driven mode.
They can be used for async triggering, real-time monitoring, logging, etc.

When no hooks are registered, the engine behaves identically to the original code.
"""
import inspect
import warnings
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Union

import pandas as pd

from .portfolio import Portfolio
from .portfolio_optimizer import BasePortfolioOptimizer


@dataclass
class SecondaryTimeframe:
    """Data for one secondary timeframe, per-asset.

    Attributes:
        bar_data: Per-symbol current bar (``pd.Series`` or ``None`` if no
            completed bar exists yet).
        data: Per-symbol historical data up to the current secondary bar
            (empty ``pd.DataFrame`` when no completed bar exists).
    """
    bar_data: Dict[str, Any]
    data: Dict[str, Any]


@dataclass
class HookContext:
    """
    Context data passed to hook functions.

    In single-asset mode, ``bar_data``, ``data``, ``symbol``, and ``broker``
    are populated (identical to v0.6).  In multi-asset mode, the
    ``active_symbols``, ``all_bar_data``, ``all_data``, and ``all_brokers``
    fields are populated instead.

    Attributes:
        bar_index: Index of the current bar (unified timeline index in multi).
        timestamp: Timestamp of the current bar.
        portfolio: Current portfolio state.
        bar_data: OHLCV for the current bar (single-asset only).
        data: Historical data up to current bar (single-asset only).
        symbol: Instrument symbol (single-asset only).
        broker: Broker instance (single-asset only).
        active_symbols: Symbols with a bar on this timestamp (multi-asset).
        all_bar_data: Per-symbol bar data for active symbols (multi-asset).
        all_data: Per-symbol historical data sliced to current timestamp (multi-asset).
        all_brokers: Per-symbol Broker instances (multi-asset).
    """
    bar_index: int
    timestamp: datetime
    portfolio: Portfolio
    # Single-asset fields (populated when single, else default):
    bar_data: Optional[pd.Series] = None
    data: Optional[pd.DataFrame] = None
    symbol: str = ""
    broker: Any = None
    # Multi-asset fields (populated when multi, else None):
    active_symbols: Optional[List[str]] = None
    all_bar_data: Optional[Dict[str, pd.Series]] = None
    all_data: Optional[Dict[str, pd.DataFrame]] = None
    all_brokers: Optional[Dict[str, Any]] = None
    # MetaContext for agent control (v1.2, event-driven only)
    meta: Optional[Any] = None
    # Secondary timeframe data (v1.3, event-driven only)
    secondary: Optional[Dict[str, SecondaryTimeframe]] = None


@dataclass
class LifecycleContext:
    """Context passed to ``on_backtest_start`` / ``on_backtest_end``.

    The lifecycle phase precedes any per-bar execution, so this context
    intentionally omits ``portfolio``, ``broker``, and ``bar_data`` —
    those are not yet meaningful. Wrapper hooks that need to forward to
    a single-symbol inner hook can use ``primary_symbol`` and
    ``primary_data`` (the alphabetically first symbol) for convenience.

    Attributes:
        phase: ``"start"`` or ``"end"``.
        timestamp: First bar's timestamp for ``"start"``; last for ``"end"``.
        symbols: All symbols in this run (sorted alphabetically).
        config: BacktestConfig instance (typed as ``Any`` to avoid
            circular imports).
        data_dict: Per-symbol full OHLCV DataFrames.
        primary_symbol: ``symbols[0]`` — convenience for single-asset
            wrapper composition.
        primary_data: ``data_dict[primary_symbol]``.
    """
    phase: Literal["start", "end"]
    timestamp: pd.Timestamp
    symbols: List[str]
    config: Any
    data_dict: Dict[str, pd.DataFrame]
    primary_symbol: str
    primary_data: pd.DataFrame


# Track which (subclass, method) pairs have already emitted a
# DeprecationWarning so we don't spam users running many backtests.
_DEPRECATED_LIFECYCLE_NOTIFIED: Set[Tuple[type, str]] = set()


def _emit_lifecycle_deprecation(cls: type, method_name: str) -> None:
    key = (cls, method_name)
    if key in _DEPRECATED_LIFECYCLE_NOTIFIED:
        return
    _DEPRECATED_LIFECYCLE_NOTIFIED.add(key)
    warnings.warn(
        f"{cls.__name__}.{method_name} uses the legacy signature. "
        f"Migrate to the LifecycleContext-based signature: "
        f"def {method_name}(self, ctx: LifecycleContext) -> None. "
        f"The legacy signature will be removed in v2.0.",
        DeprecationWarning,
        stacklevel=3,
    )


def _count_positional(method: Any) -> int:
    """Count positional (non-self) parameters of a bound method."""
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return -1
    return sum(
        1 for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )


def call_hook_lifecycle_start(hook: Any, ctx: LifecycleContext) -> None:
    """Dispatch ``on_backtest_start`` honoring legacy ``(data, symbol)`` signatures."""
    n_pos = _count_positional(hook.on_backtest_start)
    if n_pos == 1:
        hook.on_backtest_start(ctx)
    elif n_pos >= 2:
        _emit_lifecycle_deprecation(type(hook), "on_backtest_start")
        hook.on_backtest_start(ctx.primary_data, ctx.primary_symbol, config=ctx.config)
    else:
        # Best-effort fallback (no positional args at all).
        hook.on_backtest_start(ctx)


def call_hook_lifecycle_end(hook: Any, ctx: LifecycleContext) -> None:
    """Dispatch ``on_backtest_end`` honoring the legacy zero-arg signature."""
    n_pos = _count_positional(hook.on_backtest_end)
    if n_pos == 0:
        _emit_lifecycle_deprecation(type(hook), "on_backtest_end")
        hook.on_backtest_end()
    else:
        hook.on_backtest_end(ctx)


class BacktestHook(ABC):
    """
    Base class for backtest hooks.

    All methods have default no-op implementations. Override only the
    callbacks you need: ``on_pre_signal`` for pre-signal agent logic,
    ``on_bar`` for post-signal observation, or the lifecycle callbacks
    ``on_backtest_start`` / ``on_backtest_end``.

    Lifecycle callbacks fire **once per backtest** (not per symbol) and
    receive a :class:`LifecycleContext`. Subclasses written against the
    pre-v1.9.6 signature ``(data, symbol, *, config=None)`` continue to
    work via a runtime-detected adapter that emits a one-time
    ``DeprecationWarning`` per subclass.
    """

    def on_backtest_start(self, ctx: LifecycleContext) -> None:
        """Called once when the backtest starts (optional override).

        Parameters:
            ctx: :class:`LifecycleContext` describing the run.
        """
        pass

    def on_pre_signal(self, context: HookContext) -> None:
        """
        Called before signal processing on each bar.

        Override this to submit orders directly via ``context.broker``
        before the engine processes its own signals.

        Parameters:
            context: Context data for the current bar.
        """
        pass

    def on_bar(self, context: HookContext) -> Optional[Dict[str, Any]]:
        """
        Called after each bar is processed.

        Parameters:
            context: Context data for the current bar.

        Returns:
            None or a dict containing action instructions.
        """
        return None

    def on_backtest_end(self, ctx: LifecycleContext) -> None:
        """Called once when the backtest ends (optional override)."""
        pass


_VALID_FREQUENCIES = {"daily", "weekly", "monthly", "quarterly"}


class ScheduleHook(BacktestHook):
    """Execute a callback on a periodic schedule.

    Triggers on the FIRST bar of each period. For "last bar of period"
    semantics, use a custom Hook with look-ahead logic.

    IMPORTANT: On trigger bars, if the callback sets target_weights
    via MetaContext, it will override strategy signals for that bar.

    Parameters:
        frequency: Rebalance frequency:
            - "daily": every bar
            - "weekly": first bar of each ISO week
            - "monthly": first bar of each month
            - "quarterly": first bar of each quarter
            - int: every N bars
        callback: Function(ctx: HookContext) -> None.
        start_delay: Bars to skip before first trigger. Default 0.
    """

    def __init__(
        self,
        frequency: Union[str, int],
        callback: Callable[[HookContext], None],
        start_delay: int = 0,
    ) -> None:
        if isinstance(frequency, str) and frequency not in _VALID_FREQUENCIES:
            raise ValueError(
                f"Unknown frequency: {frequency!r}. "
                f"Must be one of {_VALID_FREQUENCIES} or an integer.")
        if isinstance(frequency, int) and frequency < 1:
            raise ValueError(
                f"Integer frequency must be >= 1, got {frequency}")
        self.frequency = frequency
        self.callback = callback
        self.start_delay = start_delay
        self._last_trigger: Optional[Any] = None

    def on_backtest_start(self, ctx: LifecycleContext) -> None:
        """Reset state for clean runs (important for reuse across
        multiple engine.run calls and monte_carlo_test deep-copy)."""
        self._last_trigger = None
        config = ctx.config
        if config is not None and getattr(config, 'mode', None) == 'vectorized':
            warnings.warn(
                "ScheduleHook has no effect in vectorized mode. "
                "Use mode='event_driven'.")

    def on_pre_signal(self, context: HookContext) -> None:
        if self._should_trigger(context):
            self.callback(context)

    def _should_trigger(self, ctx: HookContext) -> bool:
        if ctx.bar_index < self.start_delay:
            return False

        ts = ctx.timestamp
        if isinstance(self.frequency, int):
            return (ctx.bar_index - self.start_delay) % self.frequency == 0

        if self.frequency == "daily":
            return True
        elif self.frequency == "weekly":
            ts_yw = ts.isocalendar()[:2]  # (year, week)
            if self._last_trigger is None:
                self._last_trigger = ts
                return True
            last_yw = self._last_trigger.isocalendar()[:2]
            if ts_yw != last_yw:
                self._last_trigger = ts
                return True
        elif self.frequency == "monthly":
            if (self._last_trigger is None
                    or (ts.year, ts.month) != (self._last_trigger.year,
                                                self._last_trigger.month)):
                self._last_trigger = ts
                return True
        elif self.frequency == "quarterly":
            ts_q = (ts.year, (ts.month - 1) // 3)
            if self._last_trigger is None:
                self._last_trigger = ts
                return True
            last_q = (self._last_trigger.year,
                      (self._last_trigger.month - 1) // 3)
            if ts_q != last_q:
                self._last_trigger = ts
                return True
        return False


def schedule_rebalance(
    weights: Dict[str, float],
    frequency: Union[str, int] = "monthly",
    start_delay: int = 0,
) -> ScheduleHook:
    """Create a ScheduleHook that rebalances to target weights.

    Parameters:
        weights: Target weights per symbol.
        frequency: See ScheduleHook.
        start_delay: See ScheduleHook.

    Returns:
        ScheduleHook configured for rebalancing.

    Note: On trigger bars, strategy signals are overridden by the
    target weights. Non-trigger bars use normal strategy signals.
    Requires event-driven mode with MetaController.
    """
    weights = dict(weights)  # defensive copy

    def _rebalance(ctx: HookContext) -> None:
        if ctx.meta:
            ctx.meta.set_target_weights(weights)

    return ScheduleHook(frequency, _rebalance, start_delay)


class DriftRebalanceHook(BacktestHook):
    """Rebalance when portfolio drifts from target weights.

    Checks each bar whether max(|actual_weight - target_weight|)
    exceeds the threshold. If so, rebalances to target.

    Parameters:
        target_weights: Target allocation. Either:
            - Dict[str, float]: static weights
            - Callable(HookContext) -> Dict[str, float]: dynamic weights
        threshold: Max single-asset drift before triggering (e.g., 0.05 = 5%).
        min_interval: Minimum bars between rebalances (cooldown).
    """

    def __init__(
        self,
        target_weights: Union[Dict[str, float], Callable],
        threshold: float = 0.05,
        min_interval: int = 1,
    ) -> None:
        if callable(target_weights):
            self._get_weights = target_weights
        else:
            weights = dict(target_weights)
            self._static_weights = weights
            self._get_weights = self._return_static_weights
        self.threshold = threshold
        self.min_interval = min_interval
        self._last_rebalance_bar: int = -999

    def _return_static_weights(self, ctx: HookContext) -> Dict[str, float]:
        """Return stored static weights (deep-copy safe, no lambda)."""
        return self._static_weights  # type: ignore[return-value]

    def on_backtest_start(self, ctx: LifecycleContext) -> None:
        self._last_rebalance_bar = -999
        config = ctx.config
        if config is not None and getattr(config, 'mode', None) == 'vectorized':
            warnings.warn(
                "DriftRebalanceHook has no effect in vectorized mode. "
                "Use mode='event_driven'.")

    def on_pre_signal(self, context: HookContext) -> None:
        if context.bar_index - self._last_rebalance_bar < self.min_interval:
            return
        if context.meta is None:
            return

        target = self._get_weights(context)
        current = context.portfolio.get_weights()

        max_drift = 0.0
        for sym, target_w in target.items():
            actual_w = current.get(sym, 0.0)
            max_drift = max(max_drift, abs(actual_w - target_w))

        # Assets in portfolio but not in target should be 0
        for sym in current:
            if sym not in target:
                max_drift = max(max_drift, abs(current[sym]))

        if max_drift >= self.threshold:
            # Include weight=0 for unwanted assets so engine closes them
            complete_target = dict(target)
            for sym in current:
                if sym not in target:
                    complete_target[sym] = 0.0
            context.meta.set_target_weights(complete_target)
            self._last_rebalance_bar = context.bar_index


class BandRebalanceHook(BacktestHook):
    """Rebalance only assets that drift beyond a per-asset band.

    Assets within their band are not touched. Only drifted assets
    are rebalanced to their target weight. Runs on a schedule.

    Parameters:
        target_weights: Static dict or callable(HookContext) -> dict.
        band: Allowed drift per asset (e.g., 0.03 = +/-3%).
        frequency: Check frequency (same as ScheduleHook).
    """

    def __init__(
        self,
        target_weights: Union[Dict[str, float], Callable],
        band: float = 0.03,
        frequency: Union[str, int] = "monthly",
    ) -> None:
        if callable(target_weights):
            self._get_weights = target_weights
        else:
            weights = dict(target_weights)
            self._static_weights = weights
            self._get_weights = self._return_static_weights
        self.band = band
        self._schedule = ScheduleHook(frequency, self._check_bands)

    def _return_static_weights(self, ctx: HookContext) -> Dict[str, float]:
        """Return stored static weights (deep-copy safe, no lambda)."""
        return self._static_weights  # type: ignore[return-value]

    def on_backtest_start(self, ctx: LifecycleContext) -> None:
        self._schedule.on_backtest_start(ctx)

    def on_pre_signal(self, context: HookContext) -> None:
        self._schedule.on_pre_signal(context)

    def _check_bands(self, ctx: HookContext) -> None:
        if ctx.meta is None:
            return
        target = self._get_weights(ctx)
        current = ctx.portfolio.get_weights()

        # Only include out-of-band assets in target_weights.
        # The engine rebalances only these; in-band assets are untouched.
        adjusted: Dict[str, float] = {}
        for sym, target_w in target.items():
            actual_w = current.get(sym, 0.0)
            if abs(actual_w - target_w) > self.band:
                adjusted[sym] = target_w

        if not adjusted:
            return

        ctx.meta.set_target_weights(adjusted)


class CostAwareRebalanceHook(BacktestHook):
    """Rebalance only if expected benefit exceeds transaction costs.

    Only triggers when the total turnover (sum of absolute weight
    changes) is large enough to justify the transaction costs.

    The threshold is ``fee_rate × cost_multiplier``. With defaults
    (fee_rate=0.002, cost_multiplier=5.0), turnover must exceed 1%
    to trigger. Increase cost_multiplier to rebalance less often.

    Parameters:
        target_weights: Static dict or callable(HookContext) -> dict.
        frequency: Check frequency.
        fee_rate: Estimated round-trip fee rate (default 0.002 = 0.2%).
        cost_multiplier: Turnover must exceed fee_rate × cost_multiplier
            to trigger (default 5.0). Higher = more conservative.
    """

    def __init__(
        self,
        target_weights: Union[Dict[str, float], Callable],
        frequency: Union[str, int] = "monthly",
        fee_rate: float = 0.002,
        cost_multiplier: float = 5.0,
    ) -> None:
        if callable(target_weights):
            self._get_weights = target_weights
        else:
            weights = dict(target_weights)
            self._static_weights = weights
            self._get_weights = self._return_static_weights
        self.fee_rate = fee_rate
        self.cost_multiplier = cost_multiplier
        self._schedule = ScheduleHook(frequency, self._evaluate)

    def _return_static_weights(self, ctx: HookContext) -> Dict[str, float]:
        """Return stored static weights (deep-copy safe, no lambda)."""
        return self._static_weights  # type: ignore[return-value]

    def on_backtest_start(self, ctx: LifecycleContext) -> None:
        self._schedule.on_backtest_start(ctx)

    def on_pre_signal(self, context: HookContext) -> None:
        self._schedule.on_pre_signal(context)

    def _evaluate(self, ctx: HookContext) -> None:
        if ctx.meta is None:
            return
        target = self._get_weights(ctx)
        current = ctx.portfolio.get_weights()

        # Total turnover (sum of absolute weight changes)
        turnover = sum(
            abs(current.get(sym, 0.0) - target_w)
            for sym, target_w in target.items()
        )
        for sym in current:
            if sym not in target:
                turnover += abs(current[sym])

        # Only rebalance if turnover justifies the trading cost.
        # Threshold = fee_rate × cost_multiplier. With default
        # fee_rate=0.002 and cost_multiplier=5.0, turnover must
        # exceed 1.0% to trigger.
        threshold = self.fee_rate * self.cost_multiplier
        if turnover > threshold:
            ctx.meta.set_target_weights(target)


class OptimizedRebalanceHook(BacktestHook):
    """Periodically rebalance using a portfolio optimizer.

    Computes target weights from historical return data using the
    provided optimizer, then sets them via MetaContext. Uses the
    active universe (v1.9.2) when available.

    Parameters:
        optimizer: A BasePortfolioOptimizer instance that computes
            target weights from returns.
        frequency: Rebalance frequency (same as ScheduleHook):
            "daily", "weekly", "monthly", "quarterly", or int.
        lookback: Number of bars of price history to pass to the
            optimizer for return calculation.
        start_delay: Bars to skip before first rebalance.
    """

    def __init__(
        self,
        optimizer: BasePortfolioOptimizer,
        frequency: Union[str, int] = "monthly",
        lookback: int = 60,
        start_delay: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.lookback = lookback
        self._schedule = ScheduleHook(frequency, self._rebalance, start_delay)

    def on_backtest_start(self, ctx: LifecycleContext) -> None:
        self._schedule.on_backtest_start(ctx)

    def on_pre_signal(self, context: HookContext) -> None:
        self._schedule.on_pre_signal(context)

    def _rebalance(self, ctx: HookContext) -> None:
        if ctx.meta is None:
            return

        # Use active universe if populated, else all available data
        if ctx.meta.active_universe:
            target_symbols = ctx.meta.active_universe
        elif ctx.all_data is not None:
            target_symbols = list(ctx.all_data.keys())
        elif ctx.data is not None:
            target_symbols = [ctx.symbol]
        else:
            return

        # Build returns DataFrame from price history
        data_source = ctx.all_data or {ctx.symbol: ctx.data}
        returns_dict: Dict[str, pd.Series] = {}
        for sym in target_symbols:
            df = data_source.get(sym)
            if df is not None and len(df) > self.lookback:
                close = df["close"].iloc[-self.lookback:]
                returns_dict[sym] = close.pct_change().dropna()

        if not returns_dict:
            return

        returns = pd.DataFrame(returns_dict).dropna()
        if len(returns) < 2:
            return

        weights = self.optimizer.compute_weights(returns)
        if weights is not None:  # on_failure="none" returns None
            ctx.meta.set_target_weights(weights)
