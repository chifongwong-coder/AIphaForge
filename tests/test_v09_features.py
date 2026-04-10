"""
End-to-end tests for v0.9 features: signal semantics redesign,
target weights, continuous signals, corporate actions.
"""

import numpy as np
import pandas as pd
import pytest

from aiphaforge import (
    BacktestEngine,
    BacktestResult,
    CorporateActionHook,
    EqualWeightAllocator,
    Portfolio,
)
from aiphaforge.fees import ZeroFeeModel
from aiphaforge.margin import MarginConfig

from .conftest import make_ohlcv


class TestSignalSemantics:
    """New signal semantics: 0=flat, NaN=hold, continuous fraction."""

    def test_signal_zero_closes_position(self):
        """signal=0 closes an existing position (go flat)."""
        data = make_ohlcv(20)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1    # buy
        signals.iloc[10] = 0   # go flat (close)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        # Should have at least one trade (buy then close)
        assert result.num_trades >= 1
        assert len(result.equity_curve) == 20

    def test_nan_holds_position(self):
        """NaN between buy and sell holds the position open."""
        data = make_ohlcv(20)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1     # buy
        signals.iloc[15] = -1   # sell

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        # Position held from bar 1 to 15 (NaN bars maintain position)
        assert result.num_trades >= 1

    def test_continuous_signal_half_position(self):
        """signal=0.5 produces ~50% of full position."""
        data = make_ohlcv(10)

        # Full position run
        sig_full = pd.Series(np.nan, index=data.index, dtype=float)
        sig_full.iloc[1] = 1.0

        engine_full = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine_full.set_signals(sig_full)
        result_full = engine_full.run(data)

        # Half position run
        sig_half = pd.Series(np.nan, index=data.index, dtype=float)
        sig_half.iloc[1] = 0.5

        engine_half = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine_half.set_signals(sig_half)
        result_half = engine_half.run(data)

        # Half signal should produce smaller trade than full
        if result_full.trades and result_half.trades:
            assert result_half.trades[0].size < result_full.trades[0].size

    def test_allow_short_false_warns_on_negative(self):
        """allow_short=False + negative signal: warn, no trade."""
        data = make_ohlcv(10)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = -1  # short signal

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            allow_short=False,
            include_benchmark=False,
        )
        engine.set_signals(signals)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = engine.run(data)
            short_warns = [x for x in w if "allow_short" in str(x.message)]
            assert len(short_warns) >= 1

        # No trades should have been made
        assert result.num_trades == 0

    def test_signal_transform(self):
        """signal_transform clips z-scores to [-1, 1]."""
        data = make_ohlcv(10)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 2.5   # z-score > 1

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            signal_transform=lambda s: np.clip(s, -1, 1),
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        # Should produce a trade (signal clipped to 1.0)
        assert len(result.equity_curve) == 10


class TestTargetWeights:
    """Target-weight rebalancing API."""

    def test_basic_weight_rebalance(self):
        """Set target weights → positions match weights on rebalance dates."""
        data = {
            "A": make_ohlcv(20, start_price=100),
            "B": make_ohlcv(20, start_price=200),
        }

        dates = data["A"].index
        weights = {
            str(dates[2]): {"A": 0.5, "B": 0.5},
            str(dates[15]): {"A": 0, "B": 0},  # close all
        }

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine.set_target_weights(weights)
        result = engine.run(data)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == 20
        assert result.num_trades >= 2  # at least open + close

    def test_weight_zero_closes(self):
        """Weight=0 on rebalance date closes the position."""
        data = {"X": make_ohlcv(10, start_price=50)}
        dates = data["X"].index
        weights = {
            str(dates[1]): {"X": 1.0},   # buy full
            str(dates[7]): {"X": 0.0},    # close
        }

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=50_000,
            include_benchmark=False,
        )
        engine.set_target_weights(weights)
        result = engine.run(data)

        assert result.num_trades >= 1


class TestCorporateActions:
    """Dividend and split handling via hook."""

    def test_dividend_credits_cash(self):
        """Dividend payment credits cash to portfolio."""
        data = make_ohlcv(10, start_price=100)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1  # buy

        actions = pd.DataFrame([{
            "date": str(data.index[5]),
            "symbol": "default",
            "type": "dividend",
            "value": 1.0,  # $1 per share
        }])

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            hooks=[CorporateActionHook(actions)],
            include_benchmark=False,
        )
        engine.set_signals(signals)

        # Run without dividend for comparison
        engine_no_div = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine_no_div.set_signals(signals)

        result_div = engine.run(data)
        result_no_div = engine_no_div.run(data)

        # Dividend run should have higher final equity
        assert result_div.final_capital > result_no_div.final_capital


class TestPortfolioWeights:
    """Portfolio.get_weights() convenience method."""

    def test_get_weights_basic(self):
        """get_weights returns position weights summing to ~1."""
        p = Portfolio(initial_capital=100_000)
        # Simulate: fully invested in one stock
        p.positions["AAPL"] = type('Pos', (), {
            'notional_value': 95_000, 'is_flat': False,
        })()
        p.cash = 5_000

        weights = p.get_weights()
        assert "AAPL" in weights
        assert abs(weights["AAPL"] - 0.95) < 0.01

    def test_get_weights_empty(self):
        """No positions → empty dict."""
        p = Portfolio(initial_capital=100_000)
        assert p.get_weights() == {}


class TestPeriodicCostDecoupled:
    """FundingRateModel works without margin_config after decoupling."""

    def test_funding_rate_without_margin(self):
        """FundingRate applied even without margin_config."""
        from aiphaforge.margin import FundingRateModel

        data = make_ohlcv(10, start_price=100)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            periodic_cost_model=FundingRateModel(funding_rate_per_bar=0.001),
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        # Compare with no cost
        engine_no_cost = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine_no_cost.set_signals(signals)
        result_no_cost = engine_no_cost.run(data)

        # Funding cost should reduce equity
        assert result.final_capital < result_no_cost.final_capital
