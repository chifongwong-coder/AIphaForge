"""
Corporate Action Handling
=========================

Hook-based dividend and stock split processing. No core engine changes
needed — hooks have full portfolio access.

Usage::

    actions = pd.DataFrame([
        {"date": "2024-06-15", "symbol": "AAPL", "type": "dividend", "value": 0.25},
        {"date": "2024-08-10", "symbol": "AAPL", "type": "split", "value": 4.0},
    ])
    engine = BacktestEngine(hooks=[CorporateActionHook(actions)])
"""

import pandas as pd

from .hooks import BacktestHook, HookContext


class CorporateActionHook(BacktestHook):
    """Process dividends and stock splits during backtesting.

    Parameters:
        actions: DataFrame with columns ``[date, symbol, type, value]``.
            ``type`` is ``"dividend"`` or ``"split"``.
            ``value`` is dividend per share or split ratio (e.g., 4.0 for 4:1).
    """

    def __init__(self, actions: pd.DataFrame):
        required = {"date", "symbol", "type", "value"}
        if not required.issubset(actions.columns):
            raise ValueError(
                f"actions must have columns {required}, "
                f"got {set(actions.columns)}")
        self._actions = actions.copy()
        self._actions["date"] = pd.to_datetime(self._actions["date"])

    def on_pre_signal(self, ctx: HookContext) -> None:
        today = self._actions[self._actions["date"] == ctx.timestamp]
        if today.empty:
            return

        for _, action in today.iterrows():
            sym = action["symbol"]

            # Get position (single or multi-asset)
            pos = ctx.portfolio.positions.get(sym)
            if pos is None or pos.is_flat:
                continue

            if action["type"] == "dividend":
                self._process_dividend(ctx, sym, pos, action["value"])
            elif action["type"] == "split":
                self._process_split(ctx, sym, pos, action["value"])

    @staticmethod
    def _process_dividend(ctx, sym, pos, per_share):
        """Credit cash dividend."""
        dividend = per_share * abs(pos.size)
        ctx.portfolio.cash += dividend

    @staticmethod
    def _process_split(ctx, sym, pos, ratio):
        """Adjust position for stock split.

        Also updates Portfolio._pending_entries to keep trade records
        correct on eventual position close.
        """
        pos.size *= ratio
        pos.avg_entry_price /= ratio

        # Sync pending entry bookkeeping
        entry = ctx.portfolio._pending_entries.get(sym)
        if entry:
            entry["entry_price"] /= ratio
            entry["size"] *= ratio
