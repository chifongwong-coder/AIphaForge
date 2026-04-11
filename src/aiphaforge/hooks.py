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
from typing import Any, Dict, List, Optional

import pandas as pd

from .portfolio import Portfolio


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
