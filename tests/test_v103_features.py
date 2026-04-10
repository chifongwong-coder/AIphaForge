"""
Tests for v1.0.3: extended indicators, strategies, and runtime params.
"""

import numpy as np
import pandas as pd
import pytest

from aiphaforge import BacktestResult
from aiphaforge.fees import ZeroFeeModel
from aiphaforge.indicators import (
    AD, ADX, CMF, CCI, DEMA, DONCHIAN, ICHIMOKU, KELTNER, MFI,
    PSAR, STOCHRSI, SUPERTREND, TEMA, WILLR, WMA,
)
from aiphaforge.strategies import (
    ADXTrendFollowing, BaseStrategy, DonchianBreakout, IchimokuStrategy,
    MACrossover, MeanReversionBollinger, MomentumRank, MultiIndicatorStrategy,
    PairsTrading, SupertrendStrategy, VWAPReversion,
)

from .conftest import make_ohlcv


@pytest.fixture
def ohlcv():
    return make_ohlcv(100)


# ---------------------------------------------------------------------------
# New indicators
# ---------------------------------------------------------------------------

class TestNewIndicators:

    def test_wma(self, ohlcv):
        r = WMA(ohlcv['close'], 10)
        assert len(r) == 100
        assert r.dropna().iloc[0] > 0

    def test_dema(self, ohlcv):
        r = DEMA(ohlcv['close'], 10)
        assert not r.isna().all()

    def test_tema(self, ohlcv):
        r = TEMA(ohlcv['close'], 10)
        assert not r.isna().all()

    def test_adx_range(self, ohlcv):
        r = ADX(ohlcv['high'], ohlcv['low'], ohlcv['close'], 14)
        valid = r.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0

    def test_psar_returns_dict(self, ohlcv):
        r = PSAR(ohlcv['high'], ohlcv['low'])
        assert 'psar' in r
        assert 'direction' in r
        assert set(r['direction'].dropna().unique()).issubset({1.0, -1.0})

    def test_supertrend(self, ohlcv):
        r = SUPERTREND(ohlcv['high'], ohlcv['low'], ohlcv['close'])
        assert 'supertrend' in r
        assert 'direction' in r

    def test_ichimoku(self, ohlcv):
        r = ICHIMOKU(ohlcv['high'], ohlcv['low'])
        assert 'tenkan_sen' in r
        assert 'kijun_sen' in r
        assert 'senkou_a' in r
        assert 'senkou_b' in r

    def test_keltner(self, ohlcv):
        r = KELTNER(ohlcv['high'], ohlcv['low'], ohlcv['close'])
        assert 'upper' in r
        assert 'middle' in r
        assert 'lower' in r

    def test_donchian(self, ohlcv):
        r = DONCHIAN(ohlcv['high'], ohlcv['low'])
        assert 'upper' in r
        assert 'lower' in r

    def test_cci(self, ohlcv):
        r = CCI(ohlcv['high'], ohlcv['low'], ohlcv['close'])
        assert len(r.dropna()) > 0

    def test_willr(self, ohlcv):
        r = WILLR(ohlcv['high'], ohlcv['low'], ohlcv['close'])
        valid = r.dropna()
        assert valid.max() <= 0
        assert valid.min() >= -100

    def test_mfi(self, ohlcv):
        r = MFI(ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'])
        valid = r.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_stochrsi(self, ohlcv):
        r = STOCHRSI(ohlcv['close'])
        assert 'k' in r
        assert 'd' in r

    def test_ad(self, ohlcv):
        r = AD(ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'])
        assert len(r) == 100

    def test_cmf(self, ohlcv):
        r = CMF(ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'])
        valid = r.dropna()
        assert valid.min() >= -1.1
        assert valid.max() <= 1.1


# ---------------------------------------------------------------------------
# New strategies
# ---------------------------------------------------------------------------

class TestNewStrategies:

    def test_supertrend_strategy(self):
        data = make_ohlcv(100)
        result = SupertrendStrategy().backtest(
            data, fee_model=ZeroFeeModel(), include_benchmark=False)
        assert isinstance(result, BacktestResult)

    def test_ichimoku_strategy(self):
        data = make_ohlcv(100)
        result = IchimokuStrategy().backtest(
            data, fee_model=ZeroFeeModel(), include_benchmark=False)
        assert isinstance(result, BacktestResult)

    def test_adx_trend_following(self):
        data = make_ohlcv(100)
        result = ADXTrendFollowing().backtest(
            data, fee_model=ZeroFeeModel(), include_benchmark=False)
        assert isinstance(result, BacktestResult)

    def test_donchian_breakout(self):
        data = make_ohlcv(100)
        result = DonchianBreakout().backtest(
            data, fee_model=ZeroFeeModel(), include_benchmark=False)
        assert isinstance(result, BacktestResult)

    def test_mean_reversion_bollinger(self):
        data = make_ohlcv(100)
        result = MeanReversionBollinger().backtest(
            data, fee_model=ZeroFeeModel(), include_benchmark=False)
        assert isinstance(result, BacktestResult)

    def test_vwap_reversion(self):
        data = make_ohlcv(100)
        result = VWAPReversion().backtest(
            data, fee_model=ZeroFeeModel(), include_benchmark=False)
        assert isinstance(result, BacktestResult)

    def test_multi_indicator_strategy(self):
        data = make_ohlcv(100)
        result = MultiIndicatorStrategy().backtest(
            data, fee_model=ZeroFeeModel(), include_benchmark=False)
        assert isinstance(result, BacktestResult)


# ---------------------------------------------------------------------------
# Runtime params (foundation for v1.2 MetaController)
# ---------------------------------------------------------------------------

class TestRuntimeParams:

    def test_params_property(self):
        s = MACrossover(short=5, long=20)
        assert s.params == {'short': 5, 'long': 20, 'ma_type': 'sma'}

    def test_update_params(self):
        s = MACrossover(short=5, long=20)
        s.update_params(short=10, long=30)
        assert s.short == 10
        assert s.long == 30

    def test_update_params_rejects_unknown(self):
        s = MACrossover()
        with pytest.raises(ValueError, match="Unknown parameter"):
            s.update_params(nonexistent=42)

    def test_update_params_affects_signals(self):
        """Changing params produces different signals."""
        data = make_ohlcv(50)
        s = MACrossover(short=5, long=20)
        sig1 = s.generate_signals(data)

        s.update_params(short=3, long=10)
        sig2 = s.generate_signals(data)

        # Signals should differ (different MA windows)
        assert not sig1.equals(sig2)

    def test_default_param_grid(self):
        s = MACrossover()
        grid = s.default_param_grid()
        assert 'short' in grid
        assert 'long' in grid
        assert isinstance(grid['short'], list)

    def test_default_param_grid_empty_for_vwap(self):
        s = VWAPReversion()
        assert s.default_param_grid() == {}


# ---------------------------------------------------------------------------
# Cross-sectional strategies
# ---------------------------------------------------------------------------

class TestMomentumRank:

    def test_multi_asset_signals(self):
        """MomentumRank produces per-symbol signals from dict data."""
        data = {
            "A": make_ohlcv(60, start_price=100, trend=0.005),
            "B": make_ohlcv(60, start_price=100, trend=-0.002),
            "C": make_ohlcv(60, start_price=100, trend=0.001),
        }
        s = MomentumRank(roc_window=10, top_n=1)
        signals = s.generate_signals(data)
        assert isinstance(signals, dict)
        assert len(signals) == 3

    def test_single_asset_fallback(self):
        """Single DataFrame uses ROC > 0 fallback."""
        data = make_ohlcv(60)
        s = MomentumRank(roc_window=10, top_n=1)
        signals = s.generate_signals(data)
        assert isinstance(signals, pd.Series)


class TestPairsTrading:

    def test_two_asset_signals(self):
        """PairsTrading produces opposite signals for two correlated assets."""
        data = {
            "A": make_ohlcv(100, start_price=100, trend=0.001),
            "B": make_ohlcv(100, start_price=100, trend=0.001),
        }
        s = PairsTrading(window=20, entry_z=1.5, exit_z=0.5)
        signals = s.generate_signals(data)
        assert isinstance(signals, dict)
        assert len(signals) == 2

    def test_single_asset_raises(self):
        """PairsTrading with single DataFrame raises."""
        data = make_ohlcv(50)
        s = PairsTrading()
        with pytest.raises((ValueError, TypeError)):
            s.generate_signals(data)

    def test_wrong_num_assets_raises(self):
        """PairsTrading with != 2 assets raises."""
        data = {
            "A": make_ohlcv(50),
            "B": make_ohlcv(50),
            "C": make_ohlcv(50),
        }
        s = PairsTrading()
        with pytest.raises(ValueError, match="exactly 2"):
            s.generate_signals(data)


class TestStrategyOnLongerData:
    """Verify strategies produce trades on sufficient data."""

    def test_ichimoku_produces_trades(self):
        data = make_ohlcv(200)
        result = IchimokuStrategy().backtest(
            data, fee_model=ZeroFeeModel(), include_benchmark=False)
        assert result.num_trades >= 1

    def test_adx_produces_trades_on_trending(self):
        data = make_ohlcv(200, trend=0.005)
        result = ADXTrendFollowing(adx_threshold=15).backtest(
            data, fee_model=ZeroFeeModel(), include_benchmark=False)
        assert result.num_trades >= 1
