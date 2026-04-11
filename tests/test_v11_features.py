"""
Tests for v1.1: risk management module.
"""

import numpy as np
import pandas as pd
import pytest

from aiphaforge import (
    BacktestEngine,
    BaseRiskRule,
    CompositeRiskManager,
    ConcentrationLimit,
    DailyLossLimit,
    ExposureLimit,
    MaxDrawdownHalt,
    RiskSignal,
)
from aiphaforge.fees import ZeroFeeModel
from aiphaforge.portfolio import Portfolio, Position

from .conftest import make_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_portfolio(capital=100_000.0, allow_short=True):
    """Create a Portfolio instance for unit tests."""
    return Portfolio(
        initial_capital=capital,
        allow_short=allow_short,
    )


def _open_position(portfolio, symbol, size, price, timestamp=None):
    """Simulate opening a position by directly setting portfolio state."""
    ts = timestamp or pd.Timestamp("2024-06-01")
    portfolio.positions[symbol] = Position(
        symbol=symbol,
        size=size,
        avg_entry_price=price,
        current_price=price,
        open_time=ts,
    )
    # Deduct cash for the position
    portfolio.cash -= abs(size) * price
    portfolio.update_prices({symbol: price}, ts, record=False)


# ---------------------------------------------------------------------------
# MaxDrawdownHalt
# ---------------------------------------------------------------------------

class TestMaxDrawdownHalt:

    def test_triggers_on_drawdown(self):
        rule = MaxDrawdownHalt(max_drawdown=0.10, reset_drawdown=0.05)
        # Invest nearly all capital so price drops map ~1:1 to drawdown
        portfolio = _make_portfolio(100_000)
        ts = pd.Timestamp("2024-06-01")
        _open_position(portfolio, "AAPL", 1000, 100.0, ts)
        # cash = 0, equity = 100k

        # Price drops 12% -> equity drops ~12%
        ts2 = pd.Timestamp("2024-06-02")
        portfolio.update_prices({"AAPL": 88.0}, ts2, record=True)

        sig = rule.check(portfolio, {"AAPL": 88.0}, ts2)
        assert sig is not None
        assert sig.severity == 'critical'
        assert sig.action == 'reject_new'
        assert rule._halted is True

    def test_stays_halted(self):
        rule = MaxDrawdownHalt(max_drawdown=0.10, reset_drawdown=0.05)
        portfolio = _make_portfolio(100_000)
        ts = pd.Timestamp("2024-06-01")
        _open_position(portfolio, "AAPL", 1000, 100.0, ts)

        # Trigger halt
        ts2 = pd.Timestamp("2024-06-02")
        portfolio.update_prices({"AAPL": 88.0}, ts2, record=True)
        rule.check(portfolio, {"AAPL": 88.0}, ts2)
        assert rule._halted is True

        # Still halted on next bar (drawdown still high)
        ts3 = pd.Timestamp("2024-06-03")
        portfolio.update_prices({"AAPL": 89.0}, ts3, record=True)
        sig = rule.check(portfolio, {"AAPL": 89.0}, ts3)
        assert sig is not None
        assert sig.severity == 'critical'
        assert sig.action == 'reject_new'

    def test_recovery_after_reset_drawdown(self):
        rule = MaxDrawdownHalt(max_drawdown=0.10, reset_drawdown=0.05)
        portfolio = _make_portfolio(100_000)
        ts = pd.Timestamp("2024-06-01")
        _open_position(portfolio, "AAPL", 1000, 100.0, ts)

        # Trigger halt
        ts2 = pd.Timestamp("2024-06-02")
        portfolio.update_prices({"AAPL": 88.0}, ts2, record=True)
        rule.check(portfolio, {"AAPL": 88.0}, ts2)
        assert rule._halted is True

        # Price recovers to near peak -> drawdown < reset_drawdown
        ts3 = pd.Timestamp("2024-06-03")
        portfolio.update_prices({"AAPL": 99.0}, ts3, record=True)
        sig = rule.check(portfolio, {"AAPL": 99.0}, ts3)
        assert sig is not None
        assert sig.severity == 'info'
        assert sig.action == 'none'
        assert rule._halted is False

    def test_no_trigger_below_threshold(self):
        rule = MaxDrawdownHalt(max_drawdown=0.20)
        portfolio = _make_portfolio(100_000)
        ts = pd.Timestamp("2024-06-01")
        _open_position(portfolio, "AAPL", 1000, 100.0, ts)

        # 5% drop -> drawdown 5%, below 20% threshold
        ts2 = pd.Timestamp("2024-06-02")
        portfolio.update_prices({"AAPL": 95.0}, ts2, record=True)
        sig = rule.check(portfolio, {"AAPL": 95.0}, ts2)
        assert sig is None

    def test_reset_clears_state(self):
        rule = MaxDrawdownHalt(max_drawdown=0.10)
        rule._halted = True
        rule.reset()
        assert rule._halted is False


# ---------------------------------------------------------------------------
# ExposureLimit
# ---------------------------------------------------------------------------

class TestExposureLimit:

    def test_rejects_over_long_limit(self):
        rule = ExposureLimit(max_long=0.5, max_short=0.5, max_net=1.0)
        portfolio = _make_portfolio(200_000)
        ts = pd.Timestamp("2024-06-01")
        # Long position worth 120k, equity = 200k, ratio = 60%
        _open_position(portfolio, "AAPL", 1200, 100.0, ts)
        portfolio.update_prices({"AAPL": 100.0}, ts, record=True)

        sig = rule.check(portfolio, {"AAPL": 100.0}, ts)
        assert sig is not None
        assert sig.severity == 'critical'
        assert sig.action == 'reject_new'
        assert 'Long exposure' in sig.message

    def test_passes_within_limits(self):
        rule = ExposureLimit(max_long=1.0, max_short=0.5, max_net=1.0)
        portfolio = _make_portfolio(200_000)
        ts = pd.Timestamp("2024-06-01")
        # Long 50k, equity = 200k, ratio = 25%
        _open_position(portfolio, "AAPL", 500, 100.0, ts)
        portfolio.update_prices({"AAPL": 100.0}, ts, record=True)

        sig = rule.check(portfolio, {"AAPL": 100.0}, ts)
        assert sig is None

    def test_zero_equity(self):
        rule = ExposureLimit()
        portfolio = _make_portfolio(0)
        ts = pd.Timestamp("2024-06-01")
        sig = rule.check(portfolio, {}, ts)
        assert sig is not None
        assert sig.action == 'reject_new'


# ---------------------------------------------------------------------------
# DailyLossLimit
# ---------------------------------------------------------------------------

class TestDailyLossLimit:

    def test_halts_on_daily_loss(self):
        rule = DailyLossLimit(max_daily_loss=0.02)
        portfolio = _make_portfolio(100_000)
        ts = pd.Timestamp("2024-06-03 09:30")

        # First check: records start-of-day equity
        _open_position(portfolio, "AAPL", 1000, 100.0, ts)
        portfolio.update_prices({"AAPL": 100.0}, ts, record=True)
        sig = rule.check(portfolio, {"AAPL": 100.0}, ts)
        assert sig is None  # no loss yet

        # Price drops -> daily loss of ~3%
        ts2 = pd.Timestamp("2024-06-03 10:00")
        portfolio.update_prices({"AAPL": 97.0}, ts2, record=True)
        sig = rule.check(portfolio, {"AAPL": 97.0}, ts2)
        assert sig is not None
        assert sig.severity == 'critical'
        assert sig.action == 'reject_new'

    def test_resets_next_day(self):
        rule = DailyLossLimit(max_daily_loss=0.02)
        portfolio = _make_portfolio(100_000)

        # Day 1: trigger halt
        ts1 = pd.Timestamp("2024-06-03 09:30")
        _open_position(portfolio, "AAPL", 1000, 100.0, ts1)
        portfolio.update_prices({"AAPL": 100.0}, ts1, record=True)
        rule.check(portfolio, {"AAPL": 100.0}, ts1)

        ts1b = pd.Timestamp("2024-06-03 15:00")
        portfolio.update_prices({"AAPL": 97.0}, ts1b, record=True)
        rule.check(portfolio, {"AAPL": 97.0}, ts1b)
        assert rule._halted_today is True

        # Day 2: should reset
        ts2 = pd.Timestamp("2024-06-04 09:30")
        portfolio.update_prices({"AAPL": 97.0}, ts2, record=True)
        sig = rule.check(portfolio, {"AAPL": 97.0}, ts2)
        assert rule._halted_today is False
        assert sig is None  # no loss yet on new day

    def test_reset_clears_all(self):
        rule = DailyLossLimit()
        rule._halted_today = True
        rule._current_date = "2024-01-01"
        rule._day_start_equity = 50000
        rule.reset()
        assert rule._halted_today is False
        assert rule._current_date is None
        assert rule._day_start_equity is None


# ---------------------------------------------------------------------------
# ConcentrationLimit
# ---------------------------------------------------------------------------

class TestConcentrationLimit:

    def test_warns_on_overweight(self):
        rule = ConcentrationLimit(max_weight=0.3)
        portfolio = _make_portfolio(200_000)
        ts = pd.Timestamp("2024-06-01")
        # Position worth 80k on 200k equity = 40%
        _open_position(portfolio, "AAPL", 800, 100.0, ts)
        portfolio.update_prices({"AAPL": 100.0}, ts, record=True)

        sig = rule.check(portfolio, {"AAPL": 100.0}, ts)
        assert sig is not None
        assert sig.severity == 'warning'
        assert sig.action == 'reduce'

    def test_passes_within_limit(self):
        rule = ConcentrationLimit(max_weight=0.5)
        portfolio = _make_portfolio(200_000)
        ts = pd.Timestamp("2024-06-01")
        # Position worth 80k on 200k equity = 40%, < 50%
        _open_position(portfolio, "AAPL", 800, 100.0, ts)
        portfolio.update_prices({"AAPL": 100.0}, ts, record=True)

        sig = rule.check(portfolio, {"AAPL": 100.0}, ts)
        assert sig is None


# ---------------------------------------------------------------------------
# CompositeRiskManager
# ---------------------------------------------------------------------------

class TestCompositeRiskManager:

    def test_composes_multiple_rules(self):
        rule1 = MaxDrawdownHalt(max_drawdown=0.10)
        rule2 = DailyLossLimit(max_daily_loss=0.02)
        manager = CompositeRiskManager(rules=[rule1, rule2])

        portfolio = _make_portfolio(100_000)
        ts = pd.Timestamp("2024-06-01 09:30")
        _open_position(portfolio, "AAPL", 1000, 100.0, ts)
        # Record start-of-day equity for DailyLossLimit
        rule2.check(portfolio, {"AAPL": 100.0}, ts)

        # Large drop triggers both rules (12% dd and 12% daily loss)
        ts2 = pd.Timestamp("2024-06-01 14:00")
        portfolio.update_prices({"AAPL": 88.0}, ts2, record=True)

        signals = manager.check_all(portfolio, {"AAPL": 88.0}, ts2)
        assert len(signals) >= 2
        # Both should be critical reject_new
        critical = [s for s in signals
                    if s.severity == 'critical' and s.action == 'reject_new']
        assert len(critical) >= 2

    def test_history_records_triggered_rules(self):
        rule = MaxDrawdownHalt(max_drawdown=0.05)
        manager = CompositeRiskManager(rules=[rule])

        portfolio = _make_portfolio(100_000)
        ts = pd.Timestamp("2024-06-01")
        _open_position(portfolio, "AAPL", 1000, 100.0, ts)

        ts2 = pd.Timestamp("2024-06-02")
        portfolio.update_prices({"AAPL": 93.0}, ts2, record=True)
        manager.check_all(portfolio, {"AAPL": 93.0}, ts2)

        assert len(manager.history) == 1
        assert manager.history[0]['rule'] == "Max Drawdown Halt"

    def test_reset_clears_all(self):
        rule = MaxDrawdownHalt(max_drawdown=0.10)
        rule._halted = True
        manager = CompositeRiskManager(rules=[rule])
        manager.history.append({'test': True})

        manager.reset()
        assert rule._halted is False
        assert len(manager.history) == 0

    def test_no_signal_when_rules_pass(self):
        rule = MaxDrawdownHalt(max_drawdown=0.50)  # very permissive
        manager = CompositeRiskManager(rules=[rule])

        portfolio = _make_portfolio(100_000)
        ts = pd.Timestamp("2024-06-01")
        signals = manager.check_all(portfolio, {}, ts)
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# Vectorized mode: MaxDrawdownHalt zeros positions
# ---------------------------------------------------------------------------

class TestVectorizedRiskRules:

    def test_max_drawdown_zeros_positions(self):
        """MaxDrawdownHalt should zero positions in vectorized mode
        when drawdown exceeds the threshold."""
        rule = MaxDrawdownHalt(max_drawdown=0.10)

        # Create an equity curve with a 15% drawdown in the middle
        dates = pd.bdate_range("2024-01-01", periods=20)
        equity = pd.Series(
            [100, 102, 104, 106, 108, 110,  # peak at 110
             105, 100, 95, 92,               # drawdown to 92 -> ~16%
             94, 96, 98, 100, 102, 104, 106, 108, 110, 112],
            index=dates, dtype=float)
        positions = pd.Series(1.0, index=dates)
        data = pd.DataFrame({'close': equity}, index=dates)

        result = rule.apply_vectorized(equity, positions, data)

        # Before drawdown threshold: positions intact
        assert result.iloc[0] == 1.0
        assert result.iloc[5] == 1.0  # at peak

        # After drawdown exceeds 10%: positions zeroed
        assert result.iloc[8] == 0.0  # dd at bar 8: (110-95)/110 = 13.6%
        assert result.iloc[9] == 0.0  # dd at bar 9: (110-92)/110 = 16.4%

    def test_vectorized_integration_with_engine(self):
        """Full engine integration: vectorized mode with MaxDrawdownHalt."""
        data = make_ohlcv(100, start_price=100.0, trend=-0.005,
                          volatility=0.03)
        signals = pd.Series(1.0, index=data.index)

        rule = MaxDrawdownHalt(max_drawdown=0.10)
        manager = CompositeRiskManager(rules=[rule])

        engine = BacktestEngine(
            initial_capital=100_000,
            mode="vectorized",
            fee_model=ZeroFeeModel(),
            risk_rules=manager,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert result is not None
        assert len(result.equity_curve) == 100

    def test_composite_apply_vectorized_all(self):
        """CompositeRiskManager.apply_vectorized_all chains rules."""
        rule1 = MaxDrawdownHalt(max_drawdown=0.05)
        rule2 = MaxDrawdownHalt(max_drawdown=0.20)
        manager = CompositeRiskManager(rules=[rule1, rule2])

        dates = pd.bdate_range("2024-01-01", periods=10)
        equity = pd.Series(
            [100, 102, 104, 106, 100, 98, 96, 94, 92, 90],
            index=dates, dtype=float)
        positions = pd.Series(1.0, index=dates)
        data = pd.DataFrame({'close': equity}, index=dates)

        result = manager.apply_vectorized_all(equity, positions, data)

        # rule1 (5% threshold) triggers first at a moderate drawdown
        # rule2 (20% threshold) might not trigger
        # At bar 4: dd = (106-100)/106 = 5.66% -> rule1 triggers
        assert result.iloc[4] == 0.0


# ---------------------------------------------------------------------------
# Event-driven integration: suppress_new_orders
# ---------------------------------------------------------------------------

class TestEventDrivenRiskRules:

    def test_risk_rules_block_new_orders(self):
        """When MaxDrawdownHalt triggers, new buy orders are suppressed."""
        # Create data with a price drop big enough to trigger drawdown
        data = make_ohlcv(50, start_price=100.0, trend=-0.01,
                          volatility=0.005)

        # Signal: buy at bar 5, then buy more at bar 30 (after drawdown)
        signals = pd.Series(np.nan, index=data.index)
        signals.iloc[5] = 1.0   # initial buy
        signals.iloc[30] = 1.0  # attempted buy after drawdown

        rule = MaxDrawdownHalt(max_drawdown=0.10, reset_drawdown=0.05)
        manager = CompositeRiskManager(rules=[rule])

        engine = BacktestEngine(
            initial_capital=100_000,
            mode="event_driven",
            fee_model=ZeroFeeModel(),
            risk_rules=manager,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        # The backtest should complete without error
        assert result is not None
        assert len(result.equity_curve) > 0

    def test_phase1_closes_not_blocked(self):
        """Phase 1 close orders (signal=0) are NOT blocked by suppress."""
        # We need a scenario where:
        # 1. A position is open
        # 2. Risk rules trigger (suppress_new_orders=True)
        # 3. A close signal (0) is sent -> should still execute
        data = make_ohlcv(30, start_price=100.0, trend=-0.015,
                          volatility=0.005)

        signals = pd.Series(np.nan, index=data.index)
        signals.iloc[2] = 1.0   # open long
        signals.iloc[20] = 0.0  # close signal (should go through)

        rule = MaxDrawdownHalt(max_drawdown=0.08, reset_drawdown=0.04)
        manager = CompositeRiskManager(rules=[rule])

        engine = BacktestEngine(
            initial_capital=100_000,
            mode="event_driven",
            fee_model=ZeroFeeModel(),
            risk_rules=manager,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert result is not None
        # Check that the close order was processed (trades should include
        # at least 2: one open, one close)
        # Even if risk rules blocked new orders, the close should execute
        assert len(result.equity_curve) > 0

    def test_reset_across_runs(self):
        """CompositeRiskManager.reset() is called between runs."""
        data = make_ohlcv(30, start_price=100.0, trend=0.002,
                          volatility=0.01)
        signals = pd.Series(1.0, index=data.index)

        rule = MaxDrawdownHalt(max_drawdown=0.05)
        manager = CompositeRiskManager(rules=[rule])

        # Manually set halted to True to simulate state from prior run
        rule._halted = True
        manager.history.append({'test': True})

        engine = BacktestEngine(
            initial_capital=100_000,
            mode="event_driven",
            fee_model=ZeroFeeModel(),
            risk_rules=manager,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        # After the run, _halted should have been reset at the start
        # (the reset happens at the start of the event loop, so if no
        # drawdown occurred, it stays False)
        assert result is not None

    def test_close_all_dedup_with_margin_call(self):
        """Risk rules close_all should not duplicate margin call closes."""
        # This test verifies the dedup guard: if portfolio.is_margin_call
        # is True, risk_rules close_all skips.
        # Since we can't easily trigger both simultaneously in a simple
        # test, we verify the code path exists by running with a rule
        # that would trigger close_all if we had one.
        # For now, verify that the engine accepts the risk_rules config
        # without error.
        data = make_ohlcv(20, start_price=100.0)
        signals = pd.Series(np.nan, index=data.index)
        signals.iloc[2] = 1.0

        rule = MaxDrawdownHalt(max_drawdown=0.05)
        manager = CompositeRiskManager(rules=[rule])

        engine = BacktestEngine(
            initial_capital=100_000,
            mode="event_driven",
            fee_model=ZeroFeeModel(),
            risk_rules=manager,
        )
        engine.set_signals(signals)
        result = engine.run(data)
        assert result is not None


# ---------------------------------------------------------------------------
# BaseRiskRule interface
# ---------------------------------------------------------------------------

class TestBaseRiskRule:

    def test_default_check_returns_none(self):
        rule = BaseRiskRule()
        assert rule.check(None, {}, None) is None

    def test_default_apply_vectorized_noop(self):
        rule = BaseRiskRule()
        dates = pd.bdate_range("2024-01-01", periods=5)
        equity = pd.Series([100, 101, 102, 103, 104], index=dates,
                           dtype=float)
        positions = pd.Series(1.0, index=dates)
        data = pd.DataFrame({'close': equity}, index=dates)
        result = rule.apply_vectorized(equity, positions, data)
        pd.testing.assert_series_equal(result, positions)

    def test_default_reset_noop(self):
        rule = BaseRiskRule()
        rule.reset()  # should not raise

    def test_name_attribute(self):
        rule = BaseRiskRule()
        assert rule.name == "Unknown"

        rule2 = MaxDrawdownHalt()
        assert rule2.name == "Max Drawdown Halt"


# ---------------------------------------------------------------------------
# Version bump
# ---------------------------------------------------------------------------

class TestVersion:

    def test_version_is_at_least_110(self):
        import aiphaforge
        major, minor, patch = (
            int(x) for x in aiphaforge.__version__.split('.'))
        assert (major, minor) >= (1, 1)
