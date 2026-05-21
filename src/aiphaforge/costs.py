"""
Trade Costs

Pluggable trade cost framework for vectorized backtesting.
"""

import warnings
from typing import Literal, Optional

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

# v2.8.1 post-review (Commit L / Option D): fired once per process
# when the user passes BOTH representative_notional AND
# representative_size — notional takes precedence, size is dropped.
# A warning is preferable to silent overwrite (round-2 architect
# HIGH-1 tail). Honoring both via implied-price reconstruction was
# considered (Option A2) and rejected as over-engineering for an
# edge case with no user reports.
_BOTH_KWARGS_WARNED = False


class DefaultTradeCost(BaseTradeCost):
    """Default trade cost model for vectorized backtesting.

    Uses position changes to detect trades and applies commission +
    slippage costs proportional to the notional value of each trade.

    v2.8.1: queries ``fee_model.estimate_commission_rate`` with a
    representative (price, size) pair derived from the engine config,
    rather than the bare-default (100, 100) that v2.8.0 used. The
    bare-default produced ~1% cost rates for US stock fee models with
    min-commission floors — see v2.8.1 plan H1.

    v2.8.2: opt-in ``cost_normalization``:

    - ``"initial_capital"`` (default, preserves v2.8.1 semantics):
      ``trade_cost = gross_cost / initial_capital``. Path-independent;
      cost-as-return is anchored to starting capital.
    - ``"current_equity"``: ``trade_cost = gross_cost / running_equity``
      where ``running_equity = initial_capital * (1 + returns).cumprod()``.
      Reports cost as a fraction of CURRENT equity, which is more
      faithful for drawn-down portfolios (a $50 fee on $30k equity is
      a 0.17% return contribution, not the 0.05% the "initial_capital"
      mode would report).

      **First-order approximation**: ``running_equity`` is computed
      from GROSS (pre-cost) returns, not true realized equity. The
      systematic bias UNDER-states cost; magnitude grows with
      turnover × horizon (small for low-turnover / short backtests,
      material for active strategies over long horizons). The
      iteratively-correct version is a v2.9 follow-up.

      **NaN handling**: ``returns.fillna(0)`` is applied before the
      ``cumprod`` so a leading NaN (typical from ``pct_change``) does
      not propagate and zero out the entire equity curve. Mid-run
      NaN (e.g. from a data gap propagating through the strategy
      returns) is also treated as a 0% bar — ``running_equity``
      stays at the pre-gap level. This is acceptable for the
      diagnostic-grade semantics of this mode; for backtests with
      genuine mid-run data gaps, prefer the default
      ``"initial_capital"`` mode.

      **Clip floor**: divides by ``max(running_equity, 0.01 *
      initial_capital)``. A backtest below 1% of starting capital is
      already blown up; the floor caps per-bar cost-return at a
      finite interpretable value.

    Parameters:
        cost_normalization: ``"initial_capital"`` (default) or
            ``"current_equity"``. See class docstring for semantics.
    """

    def __init__(
        self,
        cost_normalization: Literal[
            "initial_capital", "current_equity"
        ] = "initial_capital",
    ):
        if cost_normalization not in ("initial_capital", "current_equity"):
            raise ValueError(
                f"cost_normalization must be 'initial_capital' or "
                f"'current_equity'; got {cost_normalization!r}"
            )
        self.cost_normalization = cost_normalization

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
        #
        # Precedence (v2.8.1 Commit L / Option D): if the user passes
        # BOTH representative_notional AND representative_size, notional
        # wins (size is derived from notional/median_close); the size
        # input is dropped with a one-time warning. The fee model's
        # estimate_commission_rate takes only (price, size), so honoring
        # both literally would require a synthetic implied-price
        # reconstruction (price=notional/size) — explicit non-goal here.
        rep_notional: Optional[float] = None
        rep_size: Optional[float] = None
        if (
            representative_notional is not None
            and representative_size is not None
            and np.isfinite(representative_notional)
            and np.isfinite(representative_size)
            and representative_notional > 0
            and representative_size > 0
        ):
            global _BOTH_KWARGS_WARNED
            if not _BOTH_KWARGS_WARNED:
                _BOTH_KWARGS_WARNED = True
                warnings.warn(
                    "DefaultTradeCost.apply_vectorized: both "
                    "representative_notional and representative_size "
                    "were provided. The fee model's commission-rate "
                    "query takes (price, size) only; notional takes "
                    "precedence and size is dropped. Pass just one to "
                    "silence this warning.",
                    UserWarning,
                    stacklevel=2,
                )
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
        gross_cost = trade_notional * (commission_rate + slippage_rate)

        # v2.8.2 M5: cost_normalization dispatch.
        if self.cost_normalization == "current_equity":
            # Running equity from cumulative PRE-cost (gross) returns.
            # First-order approximation: true realized equity is
            # smaller in cost-positive runs; under-states cost. See
            # class docstring + v2.8.2 plan.
            cum_returns = (1 + returns.fillna(0)).cumprod()
            running_equity = initial_capital * cum_returns
            # Clip to 1% of initial capital — a blown-up backtest's
            # cost-return is capped at gross_cost / (0.01 *
            # initial_capital) instead of diverging to inf.
            trade_cost = gross_cost / running_equity.clip(
                lower=0.01 * initial_capital
            )
        else:  # "initial_capital" (default, v2.8.1 semantics)
            trade_cost = gross_cost / initial_capital

        return returns - trade_cost
