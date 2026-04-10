"""
End-to-end tests for v1.0.1: technical indicator library.
"""

import numpy as np
import pandas as pd
import pytest

from aiphaforge.indicators import (
    ATR,
    BBANDS,
    EMA,
    MACD,
    OBV,
    ROC,
    RSI,
    SMA,
    STOCH,
    VWAP,
)

from .conftest import make_ohlcv


@pytest.fixture
def prices():
    """Simple price series for indicator testing."""
    return pd.Series(
        [10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 14, 13, 12, 11, 10],
        dtype=float,
    )


@pytest.fixture
def ohlcv():
    return make_ohlcv(50)


class TestTrend:

    def test_sma_basic(self, prices):
        result = SMA(prices, 3)
        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)
        # First 2 values should be NaN (window=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # Third value: mean(10, 11, 12) = 11.0
        assert abs(result.iloc[2] - 11.0) < 0.001

    def test_ema_basic(self, prices):
        result = EMA(prices, 3)
        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)
        # EMA starts from first value
        assert not pd.isna(result.iloc[0])

    def test_macd_returns_dict(self, prices):
        result = MACD(prices, fast=3, slow=5, signal=3)
        assert isinstance(result, dict)
        assert 'macd' in result
        assert 'signal' in result
        assert 'histogram' in result
        assert len(result['macd']) == len(prices)
        # histogram = macd - signal
        diff = result['macd'] - result['signal'] - result['histogram']
        assert diff.abs().max() < 1e-10


class TestMomentum:

    def test_rsi_range(self, ohlcv):
        result = RSI(ohlcv['close'], 14)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_roc_basic(self, prices):
        result = ROC(prices, 1)
        # Second value: (11/10 - 1) * 100 = 10%
        assert abs(result.iloc[1] - 10.0) < 0.001

    def test_stoch_returns_dict(self, ohlcv):
        result = STOCH(ohlcv['high'], ohlcv['low'], ohlcv['close'])
        assert 'k' in result
        assert 'd' in result
        valid_k = result['k'].dropna()
        assert valid_k.min() >= 0
        assert valid_k.max() <= 100


class TestVolatility:

    def test_bbands_returns_dict(self, ohlcv):
        result = BBANDS(ohlcv['close'], 20, 2)
        assert 'upper' in result
        assert 'middle' in result
        assert 'lower' in result
        valid = result['upper'].dropna()
        # Upper should be above middle, lower below
        idx = result['middle'].dropna().index
        assert (result['upper'][idx] >= result['middle'][idx] - 1e-10).all()
        assert (result['lower'][idx] <= result['middle'][idx] + 1e-10).all()

    def test_atr_positive(self, ohlcv):
        result = ATR(ohlcv['high'], ohlcv['low'], ohlcv['close'], 14)
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()


class TestVolume:

    def test_vwap_basic(self, ohlcv):
        result = VWAP(ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'])
        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv)
        # VWAP should be between low and high on average
        assert result.iloc[-1] > 0

    def test_obv_basic(self, ohlcv):
        result = OBV(ohlcv['close'], ohlcv['volume'])
        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv)


class TestEdgeCases:

    def test_window_larger_than_data(self):
        """Window > data length → all NaN."""
        short = pd.Series([1.0, 2.0, 3.0])
        result = SMA(short, 10)
        assert result.isna().all()

    def test_single_row(self):
        """Single-row input doesn't crash."""
        one = pd.Series([100.0])
        assert len(SMA(one, 1)) == 1
        assert len(RSI(one, 14)) == 1

    def test_indicators_with_engine(self):
        """Indicators can generate signals for the engine."""
        from aiphaforge import BacktestEngine
        from aiphaforge.fees import ZeroFeeModel

        data = make_ohlcv(100)
        sma_short = SMA(data['close'], 10)
        sma_long = SMA(data['close'], 30)

        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals[sma_short > sma_long] = 1
        signals[sma_short < sma_long] = -1

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode='event_driven',
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)
        assert result.num_trades >= 1
