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
from typing import Any, Dict, Optional

import pandas as pd

from .portfolio import Portfolio


@dataclass
class HookContext:
    """
    Context data passed to hook functions.

    Attributes:
        bar_index: Index of the current bar in the full dataset.
        timestamp: Timestamp of the current bar.
        bar_data: OHLCV data for the current bar (pd.Series).
        data: All historical data up to and including the current bar.
        portfolio: Current portfolio state.
        symbol: Instrument symbol.
        broker: Broker instance (typed as Any to avoid coupling).
    """
    bar_index: int
    timestamp: datetime
    bar_data: pd.Series
    data: pd.DataFrame
    portfolio: Portfolio
    symbol: str
    broker: Any = None


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
