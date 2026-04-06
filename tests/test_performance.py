"""
Tests for performance analysis utilities and PerformanceAnalyzer.
"""

import numpy as np
import pandas as pd
import pytest

from aiphaforge.performance import PerformanceAnalyzer
from aiphaforge.results import BacktestResult, Trade
from aiphaforge.utils import (
    TRADING_DAYS_STOCK,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

# ---------------------------------------------------------------------------
# Utility-level tests
# ---------------------------------------------------------------------------

class TestUtilityMetrics:
    """Tests for standalone utility functions (sharpe, max drawdown, sortino)."""

    def test_sharpe_ratio_known_value(self):
        """Sharpe ratio for a constant positive return series should be large."""
        rng = np.random.default_rng(123)
        daily = rng.normal(loc=0.001, scale=0.01, size=252)
        returns_varied = pd.Series(daily)
        sr_varied = sharpe_ratio(returns_varied, trading_days=TRADING_DAYS_STOCK)

        # mean ~ 0.001, std ~ 0.01 -> daily Sharpe ~ 0.1
        # annualized ~ 0.1 * sqrt(252) ~ 1.59
        assert 0.5 < sr_varied < 3.0

    def test_max_drawdown_known_sequence(self):
        """Equity [100, 110, 90, 95, 80, 100] -> MDD = (110-80)/110 = 27.27%."""
        equity = pd.Series([100, 110, 90, 95, 80, 100], dtype=float)
        mdd = max_drawdown(equity)
        expected = (110 - 80) / 110
        assert mdd == pytest.approx(expected, abs=0.001)

    def test_sortino_full_vs_negative_only(self):
        """'full' downside_method should yield a larger sortino than 'negative_only'.

        The 'full' method divides by a smaller downside deviation (uses N=all
        observations in the denominator mean), so the ratio should be larger.
        """
        rng = np.random.default_rng(42)
        daily = rng.normal(loc=0.0005, scale=0.01, size=252)
        returns = pd.Series(daily)

        sortino_full = sortino_ratio(
            returns, trading_days=TRADING_DAYS_STOCK, downside_method="full"
        )
        sortino_neg = sortino_ratio(
            returns, trading_days=TRADING_DAYS_STOCK, downside_method="negative_only"
        )

        # Both should be positive for a slightly positive-mean series
        assert sortino_full > 0
        assert sortino_neg > 0
        # "full" divides by sqrt(mean(min(excess,0)^2)) over ALL N observations,
        # which gives a smaller denominator than only negative observations.
        assert sortino_full > sortino_neg


# ---------------------------------------------------------------------------
# PerformanceAnalyzer tests
# ---------------------------------------------------------------------------

def _make_simple_result(equity_values: list) -> BacktestResult:
    """Build a minimal BacktestResult from a list of equity values."""
    index = pd.bdate_range("2024-01-01", periods=len(equity_values), freq="B")
    equity = pd.Series(equity_values, index=index, dtype=float)
    returns = equity.pct_change().dropna()
    return BacktestResult(
        equity_curve=equity,
        trades=[],
        positions=pd.DataFrame(),
        metrics={},
        initial_capital=equity_values[0],
        daily_returns=returns,
    )


class TestOmegaRatio:

    def test_omega_ratio_with_threshold(self):
        """Regression: non-zero threshold should use excess returns over threshold."""
        rng = np.random.default_rng(99)
        daily = rng.normal(loc=0.0005, scale=0.01, size=100)
        equity = 100_000 * np.exp(np.cumsum(daily))
        equity = np.insert(equity, 0, 100_000)

        result = _make_simple_result(equity.tolist())
        analyzer = PerformanceAnalyzer(result)

        omega_zero = analyzer.omega_ratio(threshold=0.0)
        omega_pos = analyzer.omega_ratio(threshold=0.001)

        assert omega_pos <= omega_zero
        assert omega_zero > 0
        assert omega_pos > 0


# ---------------------------------------------------------------------------
# Trade.net_pnl_pct tests
# ---------------------------------------------------------------------------

class TestNetPnlPct:
    """Tests for Trade.net_pnl_pct property."""

    def test_net_pnl_pct_long_trade(self):
        """net_pnl_pct should equal pnl / (entry_price * size)."""
        trade = Trade(
            trade_id="T1",
            symbol="X",
            direction=1,
            entry_time=pd.Timestamp("2024-01-01"),
            exit_time=pd.Timestamp("2024-01-05"),
            entry_price=100.0,
            exit_price=110.0,
            size=50,
            pnl=480.0,  # e.g. 500 gross - 20 fees
            pnl_pct=0.10,
            commission=15.0,
            slippage_cost=5.0,
        )

        expected = 480.0 / (100.0 * 50)  # 0.096
        assert trade.net_pnl_pct == pytest.approx(expected, abs=1e-9)

    def test_net_pnl_pct_zero_guard(self):
        """net_pnl_pct with entry_price=0 should return 0.0, not crash."""
        trade = Trade(
            trade_id="T2",
            symbol="Y",
            direction=1,
            entry_time=pd.Timestamp("2024-01-01"),
            exit_time=pd.Timestamp("2024-01-02"),
            entry_price=0.0,
            exit_price=1.0,
            size=10,
            pnl=10.0,
            pnl_pct=0.0,
        )

        assert trade.net_pnl_pct == 0.0
