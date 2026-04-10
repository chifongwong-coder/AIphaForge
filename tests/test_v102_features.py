"""
End-to-end tests for v1.0.2: strategy templates.
"""

import numpy as np
import pandas as pd
import pytest

from aiphaforge import BacktestResult
from aiphaforge.fees import ZeroFeeModel
from aiphaforge.strategies import (
    BaseStrategy,
    BollingerBreakout,
    MACDStrategy,
    MACrossover,
    RSIMeanReversion,
)

from .conftest import make_ohlcv


class TestMACrossover:

    def test_one_line_backtest(self):
        """MACrossover.backtest() runs end-to-end."""
        data = make_ohlcv(100)
        result = MACrossover(short=5, long=20).backtest(
            data, fee_model=ZeroFeeModel(), include_benchmark=False)
        assert isinstance(result, BacktestResult)
        assert result.num_trades >= 1

    def test_generate_signals(self):
        """generate_signals returns Series with 1/-1/NaN."""
        data = make_ohlcv(50)
        signals = MACrossover(short=5, long=15).generate_signals(data)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(data)
        values = signals.dropna().unique()
        assert set(values).issubset({1.0, -1.0})

    def test_multi_asset(self):
        """generate_signals with dict data → dict signals."""
        data = {
            "A": make_ohlcv(50, start_price=100),
            "B": make_ohlcv(50, start_price=200),
        }
        signals = MACrossover(short=5, long=15).generate_signals(data)
        assert isinstance(signals, dict)
        assert "A" in signals
        assert "B" in signals

    def test_ema_variant(self):
        """ma_type='ema' uses EMA instead of SMA."""
        data = make_ohlcv(50)
        sma_result = MACrossover(short=5, long=20, ma_type='sma').backtest(
            data, fee_model=ZeroFeeModel(), include_benchmark=False)
        ema_result = MACrossover(short=5, long=20, ma_type='ema').backtest(
            data, fee_model=ZeroFeeModel(), include_benchmark=False)
        # Both should produce results (may differ)
        assert isinstance(sma_result, BacktestResult)
        assert isinstance(ema_result, BacktestResult)


class TestRSIMeanReversion:

    def test_backtest(self):
        data = make_ohlcv(100)
        result = RSIMeanReversion(period=14, oversold=30, overbought=70).backtest(
            data, fee_model=ZeroFeeModel(), include_benchmark=False)
        assert isinstance(result, BacktestResult)


class TestBollingerBreakout:

    def test_backtest(self):
        data = make_ohlcv(100)
        result = BollingerBreakout(window=20, num_std=2).backtest(
            data, fee_model=ZeroFeeModel(), include_benchmark=False)
        assert isinstance(result, BacktestResult)


class TestMACDStrategy:

    def test_backtest(self):
        data = make_ohlcv(100)
        result = MACDStrategy(fast=12, slow=26, signal=9).backtest(
            data, fee_model=ZeroFeeModel(), include_benchmark=False)
        assert isinstance(result, BacktestResult)


class TestStrategyWithOptimize:
    """Strategy templates work with optimize()."""

    def test_optimize_with_strategy_factory(self):
        from aiphaforge import optimize

        data = make_ohlcv(100)

        def make_strategy(params):
            return MACrossover(short=params['short'], long=params['long'])

        results = optimize(
            data,
            strategy_factory=make_strategy,
            param_grid={'short': [5, 10], 'long': [20, 30]},
            metric='sharpe_ratio',
            fee_model=ZeroFeeModel(),
            include_benchmark=False,
        )
        assert len(results) == 4  # 2 × 2
        assert 'sharpe_ratio' in results.columns


class TestRepr:

    def test_repr(self):
        s = MACrossover(short=5, long=20)
        r = repr(s)
        assert "MACrossover" in r
        assert "short=5" in r
