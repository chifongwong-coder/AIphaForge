"""
Tests for BacktestEngine — vectorized and event-driven modes.
"""


import numpy as np
import pandas as pd
import pytest

from aiphaforge.engine import BacktestEngine, ExecutionMode
from aiphaforge.fees import ZeroFeeModel
from aiphaforge.hooks import BacktestHook, HookContext

from .conftest import make_ohlcv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rising_data(n: int = 50) -> pd.DataFrame:
    """Generate steadily rising OHLCV data (zero volatility noise)."""
    return make_ohlcv(n, start_price=100, trend=0.005, volatility=0.001)


def _all_long_signals(data: pd.DataFrame) -> pd.Series:
    """Return a signal series that goes long on bar 1 and flat at end."""
    signals = pd.Series(0, index=data.index, dtype=float)
    signals.iloc[1] = 1    # buy
    signals.iloc[-2] = -1  # sell near end
    return signals


# ---------------------------------------------------------------------------
# Vectorized & Event-driven basic backtest
# ---------------------------------------------------------------------------

class TestBasicBacktest:
    """Positive-return smoke tests for both execution modes."""

    def test_vectorized_backtest_positive_return(self):
        """Rising prices + long signal should produce positive total return."""
        data = _rising_data()
        signals = _all_long_signals(data)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode=ExecutionMode.VECTORIZED,
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert result.total_return > 0

    def test_event_driven_backtest_positive_return(self):
        """Rising prices + long signal should produce positive total return."""
        data = _rising_data()
        signals = _all_long_signals(data)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode=ExecutionMode.EVENT_DRIVEN,
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert result.total_return > 0


# ---------------------------------------------------------------------------
# Cross-mode consistency
# ---------------------------------------------------------------------------

class TestCrossModeConsistency:

    def test_cross_mode_direction_agreement(self):
        """Both modes should agree on the sign of total return for identical signals."""
        data = _rising_data(80)
        signals = pd.Series(0, index=data.index, dtype=float)
        signals.iloc[5] = 1
        signals.iloc[40] = -1

        results = {}
        for mode in [ExecutionMode.VECTORIZED, ExecutionMode.EVENT_DRIVEN]:
            engine = BacktestEngine(
                fee_model=ZeroFeeModel(),
                mode=mode,
                allow_short=False,
                include_benchmark=False,
            )
            engine.set_signals(signals)
            results[mode] = engine.run(data)

        vec_return = results[ExecutionMode.VECTORIZED].total_return
        evt_return = results[ExecutionMode.EVENT_DRIVEN].total_return

        # Both should be positive on rising data with a long trade
        assert vec_return > 0
        assert evt_return > 0


# ---------------------------------------------------------------------------
# Stop-loss / Take-profit (event-driven)
# ---------------------------------------------------------------------------

class TestStopLossTakeProfit:

    def test_stop_loss_triggers_event_driven(self):
        """Position should be closed when unrealized loss exceeds stop_loss.

        Verify by comparing final equity: with stop-loss, the loss should be
        capped, so equity is higher than a run without stop-loss on the same
        falling data.
        """
        data = make_ohlcv(30, start_price=100, trend=-0.02, volatility=0.001)
        signals = pd.Series(0, index=data.index, dtype=float)
        signals.iloc[1] = 1  # buy into a falling market

        # Run WITH stop-loss
        engine_sl = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode=ExecutionMode.EVENT_DRIVEN,
            stop_loss=0.05,
            include_benchmark=False,
        )
        engine_sl.set_signals(signals)
        result_sl = engine_sl.run(data)

        # Run WITHOUT stop-loss (position stays open, suffers full loss)
        engine_no_sl = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode=ExecutionMode.EVENT_DRIVEN,
            include_benchmark=False,
        )
        engine_no_sl.set_signals(signals)
        result_no_sl = engine_no_sl.run(data)

        # Stop-loss run should have a completed trade (position exited)
        assert len(result_sl.trades) >= 1
        # Stop-loss should cap losses: final equity should be higher
        assert result_sl.final_capital > result_no_sl.final_capital

    def test_take_profit_triggers_event_driven(self):
        """Position should be closed when unrealized gain exceeds take_profit.

        With a very low take_profit threshold on rising data, the position
        should be closed early, producing a completed trade.
        """
        data = make_ohlcv(30, start_price=100, trend=0.02, volatility=0.001)
        signals = pd.Series(0, index=data.index, dtype=float)
        signals.iloc[1] = 1  # buy into a rising market

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode=ExecutionMode.EVENT_DRIVEN,
            take_profit=0.05,  # 5% take profit
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        # Should have at least one completed trade from take-profit exit
        assert len(result.trades) >= 1
        # The trade should be profitable
        assert result.trades[0].pnl > 0


# ---------------------------------------------------------------------------
# Short selling control
# ---------------------------------------------------------------------------

class TestAllowShort:

    def test_allow_short_false_blocks_short(self):
        """When allow_short=False, sell signals should not open short positions."""
        data = _rising_data(40)
        signals = pd.Series(0, index=data.index, dtype=float)
        signals.iloc[5] = -1  # attempt short

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode=ExecutionMode.EVENT_DRIVEN,
            allow_short=False,
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        # No short trades should exist
        short_trades = [t for t in result.trades if t.direction == -1]
        assert len(short_trades) == 0


# ---------------------------------------------------------------------------
# Constructor / input validation
# ---------------------------------------------------------------------------

class TestConstructorAndValidation:

    def test_fee_model_string_in_constructor(self):
        """Regression: BacktestEngine(fee_model='china') should resolve correctly."""
        engine = BacktestEngine(fee_model="china")
        assert engine.fee_model.name == "china_a_share"

    def test_no_strategy_no_signals_raises(self):
        """Engine.run() without strategy or signals should raise ValueError."""
        data = _rising_data(10)
        engine = BacktestEngine(include_benchmark=False)
        with pytest.raises(ValueError, match="Must set either"):
            engine.run(data)

    def test_empty_data_after_filter_raises(self):
        """Filtering to an empty range should raise ValueError."""
        data = _rising_data(10)
        signals = pd.Series(0, index=data.index, dtype=float)
        engine = BacktestEngine(include_benchmark=False)
        engine.set_signals(signals)

        with pytest.raises(ValueError, match="No data after date filtering"):
            engine.run(data, start="2099-01-01")

    def test_nan_inf_signals_sanitized(self):
        """NaN and Inf values in signals should be replaced with 0."""
        data = _rising_data(20)
        signals = pd.Series(0.0, index=data.index)
        signals.iloc[3] = np.nan
        signals.iloc[5] = np.inf
        signals.iloc[7] = -np.inf

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode=ExecutionMode.VECTORIZED,
            include_benchmark=False,
        )
        engine.set_signals(signals)

        # Should run without error; NaN/Inf treated as 0 (no trade)
        result = engine.run(data)
        assert result is not None
        assert len(result.equity_curve) == len(data)


# ---------------------------------------------------------------------------
# Hook integration (v0.3)
# ---------------------------------------------------------------------------

class TestHookIntegration:
    """Tests for hook lifecycle, broker access, and signal conflict warnings."""

    def test_hook_pre_signal_receives_broker(self):
        """on_pre_signal should receive a HookContext with a non-None broker."""
        broker_seen = []

        class BrokerCheckHook(BacktestHook):
            def on_pre_signal(self, context: HookContext) -> None:
                broker_seen.append(context.broker is not None)

        data = _rising_data(20)
        signals = _all_long_signals(data)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode=ExecutionMode.EVENT_DRIVEN,
            include_benchmark=False,
            hooks=[BrokerCheckHook()],
        )
        engine.set_signals(signals)
        engine.run(data)

        # Hook should have been called for every bar
        assert len(broker_seen) == len(data)
        assert all(broker_seen)

    def test_hook_signal_conflict_warning(self):
        """Hook submitting orders + non-zero signal should emit a UserWarning."""

        class OrderSubmittingHook(BacktestHook):
            def on_pre_signal(self, context: HookContext) -> None:
                # Submit a small market order via the broker
                order = context.broker.create_market_order(
                    context.symbol, "buy", 1, "hook_test", context.timestamp,
                )
                context.broker.submit_order(order, context.timestamp)

        data = _rising_data(20)
        signals = pd.Series(0, index=data.index, dtype=float)
        signals.iloc[5] = 1  # non-zero signal on same bar the hook acts

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode=ExecutionMode.EVENT_DRIVEN,
            include_benchmark=False,
            hooks=[OrderSubmittingHook()],
        )
        engine.set_signals(signals)

        with pytest.warns(UserWarning, match="hook submitted orders"):
            engine.run(data)

    def test_fee_allocation_passthrough(self):
        """Engine with fee_allocation='first_close' should run without error."""
        data = _rising_data(30)
        signals = _all_long_signals(data)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode=ExecutionMode.EVENT_DRIVEN,
            include_benchmark=False,
            fee_allocation="first_close",
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert result is not None
        assert len(result.equity_curve) == len(data)
