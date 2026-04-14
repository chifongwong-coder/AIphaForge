"""
Backtest Hook Framework
=======================

Provides an extensible hook mechanism for the backtest engine.

Hooks are called after each bar is processed in event-driven mode.
They can be used for async triggering, real-time monitoring, logging, etc.

When no hooks are registered, the engine behaves identically to the original code.
"""
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from .portfolio import Portfolio


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


class BacktestHook(ABC):
    """
    Base class for backtest hooks.

    All methods have default no-op implementations. Override only the
    callbacks you need: ``on_pre_signal`` for pre-signal agent logic,
    ``on_bar`` for post-signal observation, or the lifecycle callbacks
    ``on_backtest_start`` / ``on_backtest_end``.
    """

    def on_backtest_start(
        self,
        data: pd.DataFrame,
        symbol: str,
        *,
        config: Any = None,
    ) -> None:
        """
        Called once when the backtest starts (optional override).

        Parameters:
            data: Full backtest dataset.
            symbol: Instrument symbol.
            config: Backtest configuration (keyword-only, optional).
                    Passed by the engine so hooks can inspect settings.
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

    def on_backtest_end(self) -> None:
        """Called once when the backtest ends (optional override)."""
        pass


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
        self.frequency = frequency
        self.callback = callback
        self.start_delay = start_delay
        self._last_trigger: Optional[Any] = None

    def on_backtest_start(
        self,
        data: pd.DataFrame,
        symbol: str,
        *,
        config: Any = None,
    ) -> None:
        """Reset state for clean runs (important for reuse across
        multiple engine.run calls and monte_carlo_test deep-copy)."""
        self._last_trigger = None
        if config and hasattr(config, 'mode') and config.mode == 'vectorized':
            import warnings
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
    def _rebalance(ctx: HookContext) -> None:
        if ctx.meta:
            ctx.meta.set_target_weights(weights)

    return ScheduleHook(frequency, _rebalance, start_delay)
