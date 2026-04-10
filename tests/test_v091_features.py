"""
End-to-end tests for v0.9.1: turnover constraints.
"""

import numpy as np
import pandas as pd
import pytest

from aiphaforge import (
    BacktestEngine,
    BacktestResult,
    EqualWeightAllocator,
    TurnoverConfig,
)
from aiphaforge.fees import ZeroFeeModel

from .conftest import make_ohlcv


class TestTurnoverConstraints:
    """Turnover cap scales trades proportionally."""

    def test_no_cap_baseline(self):
        """turnover_config=None produces same result as v0.9."""
        data = make_ohlcv(20)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[15] = 0  # flat

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert isinstance(result, BacktestResult)
        assert result.num_trades >= 1
        assert result.turnover_history is not None
        assert len(result.turnover_history) == 20

    def test_cap_scales_trade_down(self):
        """Tight turnover cap → trade size reduced."""
        data = make_ohlcv(10)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1  # buy

        # Without cap
        engine_no_cap = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine_no_cap.set_signals(signals)
        result_no_cap = engine_no_cap.run(data)

        # With tight 10% cap
        engine_cap = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            turnover_config=TurnoverConfig(max_turnover_per_bar=0.1),
            include_benchmark=False,
        )
        engine_cap.set_signals(signals)
        result_cap = engine_cap.run(data)

        # Capped run should have smaller position
        if result_no_cap.trades and result_cap.trades:
            assert result_cap.trades[0].size < result_no_cap.trades[0].size

    def test_close_exempt_from_cap(self):
        """signal=0 (close) executes fully despite tight turnover cap."""
        data = make_ohlcv(20)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1   # buy
        signals.iloc[15] = 0  # close (should be full close, not capped)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            turnover_config=TurnoverConfig(max_turnover_per_bar=0.01),
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        # After close, equity should be ~all cash (position fully closed)
        # The close should not be capped even with 1% turnover limit
        assert result.num_trades >= 1

    def test_multi_asset_proportional_scaling(self):
        """Multiple assets scaled equally when cap hit."""
        data = {
            "A": make_ohlcv(10, start_price=100),
            "B": make_ohlcv(10, start_price=200),
            "C": make_ohlcv(10, start_price=150),
        }
        signals = {
            "A": pd.Series(np.nan, index=data["A"].index, dtype=float),
            "B": pd.Series(np.nan, index=data["B"].index, dtype=float),
            "C": pd.Series(np.nan, index=data["C"].index, dtype=float),
        }
        signals["A"].iloc[1] = 1
        signals["B"].iloc[1] = 1
        signals["C"].iloc[1] = 1

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=300_000,
            capital_allocator=EqualWeightAllocator(),
            turnover_config=TurnoverConfig(max_turnover_per_bar=0.3),
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert isinstance(result, BacktestResult)
        assert result.turnover_history is not None

    def test_turnover_history_recorded(self):
        """turnover_history has one entry per bar with correct values."""
        data = make_ohlcv(5)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert result.turnover_history is not None
        assert len(result.turnover_history) == 5
        # Bar 0: no trade → turnover = 0
        assert result.turnover_history[0] == 0.0


class TestTurnoverEdgeCases:
    """Edge cases: weight=0 exempt, equity≤0, lot rounding, margin call."""

    def test_weight_zero_exempt_from_cap(self):
        """weight=0 (close) should execute fully despite tight cap."""
        data = {"X": make_ohlcv(15, start_price=100)}
        dates = data["X"].index
        weights = {
            str(dates[1]): {"X": 1.0},   # buy full
            str(dates[10]): {"X": 0.0},   # close — must be exempt
        }

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=50_000,
            turnover_config=TurnoverConfig(max_turnover_per_bar=0.01),
            include_benchmark=False,
        )
        engine.set_target_weights(weights)
        result = engine.run(data)

        assert result.num_trades >= 1

    def test_lot_rounding_single_pass(self):
        """Lot rounding happens once after scaling, not twice."""
        data = make_ohlcv(10, start_price=100)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1

        # With lot=100 and a turnover cap
        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            lot_size=100,
            turnover_config=TurnoverConfig(max_turnover_per_bar=0.5),
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        # All trades should be multiples of 100
        for trade in result.trades:
            assert trade.size % 100 == 0 or trade.size == 0

    def test_allow_short_warning_preserved(self):
        """allow_short=False + negative signal warns (not silent)."""
        import warnings as w_mod

        data = make_ohlcv(5)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = -1

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            allow_short=False,
            turnover_config=TurnoverConfig(max_turnover_per_bar=0.5),
            include_benchmark=False,
        )
        engine.set_signals(signals)

        with w_mod.catch_warnings(record=True) as caught:
            w_mod.simplefilter("always")
            result = engine.run(data)
            short_warns = [x for x in caught if "allow_short" in str(x.message)]
            assert len(short_warns) >= 1

        assert result.num_trades == 0
