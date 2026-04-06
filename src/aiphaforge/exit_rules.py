"""
Exit Rules

Pluggable exit rule framework for stop-loss, take-profit, and custom exits.
Each rule implements both vectorized and event-driven interfaces.
"""

import numpy as np
import pandas as pd

from .broker import Broker
from .portfolio import Portfolio


class BaseExitRule:
    """Abstract base for exit rules (stop-loss, take-profit, etc.).

    Default methods are no-ops. Subclasses override only the modes they
    support. Unused modes are silently skipped.
    """

    def apply_vectorized(
        self,
        returns: pd.Series,
        positions: pd.Series,
        data: pd.DataFrame,
    ) -> pd.Series:
        """Vectorized mode. Default: no-op (returns unchanged)."""
        return returns

    def check_event_driven(
        self,
        portfolio: Portfolio,
        broker: Broker,
        symbol: str,
        bar: pd.Series,
        timestamp: pd.Timestamp,
    ) -> None:
        """Event-driven mode. Default: no-op."""
        pass


class PercentageStopLoss(BaseExitRule):
    """Stop-loss exit rule based on a percentage threshold.

    Parameters:
        threshold: Maximum allowed loss as a positive fraction (e.g. 0.05 = 5%).
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def apply_vectorized(
        self,
        returns: pd.Series,
        positions: pd.Series,
        data: pd.DataFrame,
    ) -> pd.Series:
        """Per-position vectorized stop loss.

        When the cumulative PnL from the entry price breaches the stop-loss
        threshold:
        - On the trigger bar: the return is approximately ``-threshold``
          (the actual loss to reach the stop level).
        - After the trigger bar: positions are flat (returns are 0) until
          the next entry signal appears.
        """
        close = data['close']

        # Identify entry points: bars where position changes
        pos_changes = positions.diff().fillna(positions.iloc[0])
        entry_mask = pos_changes != 0
        entry_prices = close[entry_mask].reindex(close.index).ffill()

        # PnL percentage from entry, signed by position direction
        direction = positions.shift(1).apply(np.sign)
        pos_pnl_pct = (close / entry_prices - 1) * direction

        # Find first stop trigger for each position segment
        stop_raw = pos_pnl_pct < -self.threshold

        # Build a mask of bars that are "stopped out": from trigger bar
        # until the next new entry signal
        stopped_out = pd.Series(False, index=positions.index)
        is_stopped = False
        for i in range(len(positions)):
            idx = positions.index[i]
            # A new entry signal resets the stopped state
            if entry_mask.iloc[i] and i > 0:
                is_stopped = False
            if is_stopped:
                stopped_out.iloc[i] = True
            elif stop_raw.iloc[i]:
                stopped_out.iloc[i] = True
                is_stopped = True

        returns_with_stop = returns.copy()

        # On the trigger bar (first bar of each stopped segment):
        # return should be approximately -threshold * |position|
        # (the loss to reach the stop level)
        trigger_bars = stopped_out & ~stopped_out.shift(1, fill_value=False)
        prev_close = close.shift(1)
        for idx in trigger_bars[trigger_bars].index:
            prev = prev_close.loc[idx]
            entry_p = entry_prices.loc[idx]
            dir_val = direction.loc[idx]
            if pd.notna(prev) and pd.notna(entry_p) and prev != 0 and dir_val != 0:
                # Stop price is the entry price moved by -threshold
                stop_price = entry_p * (1 - self.threshold * dir_val)
                # Return on this bar is the move from prev close to stop price
                returns_with_stop.loc[idx] = (stop_price - prev) / prev * dir_val * dir_val
            else:
                returns_with_stop.loc[idx] = 0

        # After trigger bar: position is flat, returns are 0
        after_trigger = stopped_out & ~trigger_bars
        returns_with_stop[after_trigger] = 0

        return returns_with_stop

    def check_event_driven(
        self,
        portfolio: Portfolio,
        broker: Broker,
        symbol: str,
        bar: pd.Series,
        timestamp: pd.Timestamp,
    ) -> None:
        """Check and execute stop loss in event-driven mode."""
        position = portfolio.get_position(symbol)
        if position is None or position.is_flat:
            return

        if position.unrealized_pnl_pct <= -self.threshold:
            order = broker.create_market_order(
                symbol,
                "sell" if position.is_long else "buy",
                abs(position.size),
                "stop_loss",
                timestamp,
            )
            broker.submit_order(order, timestamp)


class PercentageTakeProfit(BaseExitRule):
    """Take-profit exit rule based on a percentage threshold.

    Parameters:
        threshold: Target gain as a positive fraction (e.g. 0.10 = 10%).
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    # apply_vectorized: default no-op (take-profit not implemented in vectorized)

    def check_event_driven(
        self,
        portfolio: Portfolio,
        broker: Broker,
        symbol: str,
        bar: pd.Series,
        timestamp: pd.Timestamp,
    ) -> None:
        """Check and execute take profit in event-driven mode."""
        position = portfolio.get_position(symbol)
        if position is None or position.is_flat:
            return

        if position.unrealized_pnl_pct >= self.threshold:
            order = broker.create_market_order(
                symbol,
                "sell" if position.is_long else "buy",
                abs(position.size),
                "take_profit",
                timestamp,
            )
            broker.submit_order(order, timestamp)
