"""
Trade Costs

Pluggable trade cost framework for vectorized backtesting.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .fees import BaseFeeModel

# v2.8: public surface lock.
__all__ = [
    "BaseTradeCost",
    "DefaultTradeCost",
]


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
        *,
        representative_notional: Optional[float] = None,
        representative_size: Optional[float] = None,
    ) -> pd.Series:
        """Apply trade costs to strategy returns.

        Parameters:
            returns: Strategy returns before costs.
            positions: Position series.
            data: OHLCV data.
            fee_model: Fee model for cost estimation.
            initial_capital: Starting capital.
            representative_notional: v2.8.1 — per-trade dollar notional
                for vectorized cost estimation. When provided (> 0,
                finite), drives the commission-rate query directly.
            representative_size: v2.8.1 — per-trade unit count. Used
                when ``representative_notional`` is None; the notional
                is then computed as ``representative_size *
                data["close"].median()``. Typically populated for
                ``FixedSizer`` users.

        Returns:
            Net returns after costs.
        """
        return returns


# Module-level flag so the v2.8.1 degenerate-input warning fires at
# most once per process. The warning identifies which sizer hit the
# branch so users can pass representative_notional explicitly.
_DEGENERATE_WARNED = False


class DefaultTradeCost(BaseTradeCost):
    """Default trade cost model for vectorized backtesting.

    Uses position changes to detect trades and applies commission +
    slippage costs proportional to the notional value of each trade.

    v2.8.1: queries ``fee_model.estimate_commission_rate`` with a
    representative (price, size) pair derived from the engine config,
    rather than the bare-default (100, 100) that v2.8.0 used. The
    bare-default produced ~1% cost rates for US stock fee models with
    min-commission floors — see v2.8.1 plan H1.
    """

    def apply_vectorized(
        self,
        returns: pd.Series,
        positions: pd.Series,
        data: pd.DataFrame,
        fee_model: BaseFeeModel,
        initial_capital: float,
        *,
        representative_notional: Optional[float] = None,
        representative_size: Optional[float] = None,
    ) -> pd.Series:
        # Trade size = absolute change in position
        trade_size = positions.diff().abs().fillna(0)

        # Slippage stays a model-level scalar (independent of trade size).
        slippage_rate = (
            fee_model.slippage_pct
            if hasattr(fee_model, "slippage_pct")
            else 0.001
        )

        # Resolve a representative (price, size) for the commission-rate
        # query. pd.Series.median defaults to skipna=True; partial-NaN
        # is acceptable, all-NaN trips the degenerate-input branch.
        close = data["close"] if "close" in data.columns else None
        median_close = (
            float(close.median()) if close is not None and len(close) else float("nan")
        )

        # Dispatch order, per v2.8.1 plan Commit A step 3:
        #   1. User-provided representative_notional (Q3 user-wins).
        #   2. representative_size from sizer (B1 / FixedSizer path).
        #   3. Degenerate: zero cost + one-time warning.
        rep_notional: Optional[float] = None
        rep_size: Optional[float] = None
        if (
            representative_notional is not None
            and np.isfinite(representative_notional)
            and representative_notional > 0
            and np.isfinite(median_close)
            and median_close > 0
        ):
            rep_notional = float(representative_notional)
            rep_size = rep_notional / median_close
        elif (
            representative_size is not None
            and np.isfinite(representative_size)
            and representative_size > 0
            and np.isfinite(median_close)
            and median_close > 0
        ):
            rep_size = float(representative_size)
            rep_notional = rep_size * median_close

        if rep_notional is None:
            # Degenerate input. Do NOT fall back to no-args
            # estimate_commission_rate() — that re-introduces the
            # v2.8.0 over-billing bug. Return strategy returns unchanged
            # and warn once.
            global _DEGENERATE_WARNED
            if not _DEGENERATE_WARNED:
                _DEGENERATE_WARNED = True
                warnings.warn(
                    "DefaultTradeCost.apply_vectorized: no representative "
                    "trade notional could be derived "
                    "(representative_notional/representative_size both "
                    "unset or non-positive, or data['close'] is empty/"
                    "all-NaN). Treating trade cost as zero for this run. "
                    "Pass representative_notional=... to BacktestEngine "
                    "for an explicit estimate.",
                    UserWarning,
                    stacklevel=2,
                )
            return returns

        commission_rate = fee_model.estimate_commission_rate(
            price=median_close,
            size=rep_size,
            side="average",
        )

        # Notional value of each trade (trade_size * price)
        trade_notional = trade_size * data["close"]

        # Single-side cost for each position change
        trade_cost = (
            trade_notional * (commission_rate + slippage_rate) / initial_capital
        )

        return returns - trade_cost
