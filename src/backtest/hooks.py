"""
Backtest Hook Framework
=======================

Provides an extensible hook mechanism for the backtest engine.

Hooks are called after each bar is processed in event-driven mode.
They can be used for async triggering, real-time monitoring, logging, etc.

When no hooks are registered, the engine behaves identically to the original code.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List

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
    """
    bar_index: int
    timestamp: datetime
    bar_data: pd.Series
    data: pd.DataFrame
    portfolio: Portfolio
    symbol: str


class BacktestHook(ABC):
    """
    Base class for backtest hooks.

    Subclasses must implement the ``on_bar`` method.
    ``on_backtest_start`` and ``on_backtest_end`` are optional lifecycle callbacks.
    """

    def on_backtest_start(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> None:
        """
        Called once when the backtest starts (optional override).

        Parameters:
            data: Full backtest dataset.
            symbol: Instrument symbol.
        """
        pass

    @abstractmethod
    def on_bar(self, context: HookContext) -> Optional[Dict[str, Any]]:
        """
        Called after each bar is processed.

        Parameters:
            context: Context data for the current bar.

        Returns:
            None or a dict containing action instructions.
        """
        ...

    def on_backtest_end(self) -> None:
        """Called once when the backtest ends (optional override)."""
        pass
