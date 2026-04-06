"""
Trade Costs

Pluggable trade cost framework for vectorized backtesting.
"""

import pandas as pd

from .fees import BaseFeeModel


class BaseTradeCost:
    """Abstract base for vectorized trade cost calculation.

    Default: no-op (returns unchanged).
    """

    def apply_vectorized(
        self,
        returns: pd.Series,
        positions: pd.Series,
        data: pd.DataFrame,
        fee_model: BaseFeeModel,
        initial_capital: float,
    ) -> pd.Series:
        """Apply trade costs to strategy returns.

        Parameters:
            returns: Strategy returns before costs.
            positions: Position series.
            data: OHLCV data.
            fee_model: Fee model for cost estimation.
            initial_capital: Starting capital.

        Returns:
            Net returns after costs.
        """
        return returns


class DefaultTradeCost(BaseTradeCost):
    """Default trade cost model for vectorized backtesting.

    Uses position changes to detect trades and applies commission +
    slippage costs proportional to the notional value of each trade.
    """

    def apply_vectorized(
        self,
        returns: pd.Series,
        positions: pd.Series,
        data: pd.DataFrame,
        fee_model: BaseFeeModel,
        initial_capital: float,
    ) -> pd.Series:
        """Apply trade costs based on position changes and fee model.

        Uses ``positions.diff().abs()`` as the trade size (the actual
        change in position) and ``fee_model.estimate_commission_rate()``
        for the commission rate.  Each position change incurs a single-
        side cost proportional to its notional value.
        """
        # Trade size = absolute change in position
        trade_size = positions.diff().abs().fillna(0)

        # Get rates from fee model
        commission_rate = fee_model.estimate_commission_rate()
        slippage_rate = (
            fee_model.slippage_pct
            if hasattr(fee_model, 'slippage_pct')
            else 0.001
        )

        # Notional value of each trade (trade_size * price)
        trade_notional = trade_size * data['close']

        # Single-side cost for each position change
        trade_cost = trade_notional * (commission_rate + slippage_rate) / initial_capital

        return returns - trade_cost
