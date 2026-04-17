"""PSR / DSR tests — v1.9.5 additions to aiphaforge.significance."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from aiphaforge import (
    BacktestEngine,
    DSRResult,
    PSRResult,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from aiphaforge.fees import ZeroFeeModel
from aiphaforge.strategies import MACrossover
from tests.conftest import make_ohlcv


def _manual_psr(rets: pd.Series, *, benchmark_per: float = 0.0) -> float:
    sr_per = float(rets.mean() / rets.std())
    skew = float(stats.skew(rets, bias=False))
    kurt_pearson = float(stats.kurtosis(rets, fisher=False, bias=False))
    n = len(rets)
    denom = 1.0 - skew * sr_per + (kurt_pearson - 1.0) / 4.0 * sr_per ** 2
    z = (sr_per - benchmark_per) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


# ---------------------------------------------------------------------------
# PSR — series API
# ---------------------------------------------------------------------------


class TestProbabilisticSharpeRatio:
    def test_normal_returns_matches_manual_formula(self):
        np.random.seed(42)
        rets = pd.Series(np.random.normal(0.001, 0.01, 750))
        res = probabilistic_sharpe_ratio(
            rets, benchmark_sharpe=0.0, trading_days=252)
        assert res.psr == pytest.approx(_manual_psr(rets), rel=1e-9)

    def test_normal_returns_locks_specific_value(self):
        np.random.seed(42)
        rets = pd.Series(np.random.normal(0.001, 0.01, 750))
        res = probabilistic_sharpe_ratio(rets, benchmark_sharpe=0.0)
        assert res.psr == pytest.approx(0.9890164, abs=1e-6)

    def test_returns_psr_result_dataclass(self):
        np.random.seed(1)
        rets = pd.Series(np.random.normal(0.001, 0.01, 500))
        res = probabilistic_sharpe_ratio(rets, benchmark_sharpe=0.5)
        assert isinstance(res, PSRResult)
        assert res.n_obs == 500
        assert res.benchmark_sharpe == 0.5
        assert 0.0 <= res.psr <= 1.0

    def test_higher_benchmark_lowers_psr(self):
        np.random.seed(7)
        rets = pd.Series(np.random.normal(0.001, 0.01, 500))
        p_zero = probabilistic_sharpe_ratio(rets, benchmark_sharpe=0.0).psr
        p_high = probabilistic_sharpe_ratio(rets, benchmark_sharpe=2.0).psr
        assert p_zero > p_high

    def test_short_series_returns_nan(self):
        rets = pd.Series([0.01, -0.01, 0.005])
        res = probabilistic_sharpe_ratio(rets)
        assert math.isnan(res.psr)

    def test_zero_std_returns_nan(self):
        rets = pd.Series([0.001] * 100)
        res = probabilistic_sharpe_ratio(rets)
        assert math.isnan(res.psr)

    def test_fat_tailed_psr_not_above_normal_at_matched_sr(self):
        np.random.seed(1)
        n = 750
        normal = pd.Series(np.random.normal(0.001, 0.01, n))
        sr_normal = float(normal.mean() / normal.std())
        raw_t = pd.Series(np.random.standard_t(3, n) * 0.01 + 0.001)
        raw_t = raw_t - raw_t.mean() + sr_normal * raw_t.std()
        p_normal = probabilistic_sharpe_ratio(normal).psr
        p_t = probabilistic_sharpe_ratio(raw_t).psr
        assert p_t <= p_normal + 1e-9

    def test_strong_signal_is_significant(self):
        np.random.seed(2026)
        rets = pd.Series(np.random.normal(0.0015, 0.005, 750))
        res = probabilistic_sharpe_ratio(rets, benchmark_sharpe=1.0)
        assert res.psr > 0.95

    def test_weak_signal_is_insignificant(self):
        np.random.seed(2027)
        rets = pd.Series(np.random.normal(0.0001, 0.01, 750))
        res = probabilistic_sharpe_ratio(rets, benchmark_sharpe=1.0)
        assert res.psr < 0.30


# ---------------------------------------------------------------------------
# PSR — BacktestResult API + trading_days resolution
# ---------------------------------------------------------------------------


class TestPSRResultAPI:
    def test_accepts_backtest_result(self):
        data = make_ohlcv(300)
        engine = BacktestEngine(mode="vectorized", fee_model=ZeroFeeModel())
        engine.set_strategy(MACrossover(short=10, long=30))
        res = engine.run(data, symbol="X")
        psr = probabilistic_sharpe_ratio(res)
        assert isinstance(psr, PSRResult)
        assert psr.n_obs == len(res.equity_curve) - 1

    def test_trading_days_auto_from_result(self):
        data = make_ohlcv(300)
        engine = BacktestEngine(
            mode="vectorized", fee_model=ZeroFeeModel(), trading_days=365)
        engine.set_strategy(MACrossover(short=10, long=30))
        res = engine.run(data, symbol="X")
        psr_auto = probabilistic_sharpe_ratio(res)
        psr_252 = probabilistic_sharpe_ratio(res, trading_days=252)
        if psr_252.observed_sharpe != 0:
            ratio = psr_auto.observed_sharpe / psr_252.observed_sharpe
            assert ratio == pytest.approx(np.sqrt(365 / 252), rel=1e-9)

    def test_explicit_trading_days_overrides_result(self):
        data = make_ohlcv(300)
        engine = BacktestEngine(
            mode="vectorized", fee_model=ZeroFeeModel(), trading_days=252)
        engine.set_strategy(MACrossover(short=10, long=30))
        res = engine.run(data, symbol="X")
        psr_explicit = probabilistic_sharpe_ratio(res, trading_days=365)
        psr_auto = probabilistic_sharpe_ratio(res)
        if psr_auto.observed_sharpe != 0:
            assert psr_explicit.observed_sharpe / psr_auto.observed_sharpe \
                == pytest.approx(np.sqrt(365 / 252), rel=1e-9)

    def test_invalid_source_type(self):
        with pytest.raises(TypeError):
            probabilistic_sharpe_ratio([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DSR
# ---------------------------------------------------------------------------


class TestDeflatedSharpeRatio:
    def test_more_trials_lowers_dsr(self):
        np.random.seed(0)
        rets = pd.Series(np.random.normal(0.001, 0.01, 750))
        d10 = deflated_sharpe_ratio(rets, n_trials=10)
        d100 = deflated_sharpe_ratio(rets, n_trials=100)
        d1000 = deflated_sharpe_ratio(rets, n_trials=1000)
        assert d10.dsr >= d100.dsr >= d1000.dsr

    def test_returns_dsr_result(self):
        np.random.seed(0)
        rets = pd.Series(np.random.normal(0.001, 0.01, 500))
        d = deflated_sharpe_ratio(rets, n_trials=50)
        assert isinstance(d, DSRResult)
        assert d.n_trials == 50
        assert 0.0 <= d.dsr <= 1.0

    def test_invalid_inputs_return_nan(self):
        short = pd.Series([0.01, -0.01])
        assert math.isnan(deflated_sharpe_ratio(short, n_trials=50).dsr)
        rets = pd.Series(np.random.normal(0.001, 0.01, 100))
        assert math.isnan(deflated_sharpe_ratio(rets, n_trials=0).dsr)

    def test_dsr_accepts_result(self):
        data = make_ohlcv(300)
        engine = BacktestEngine(mode="vectorized", fee_model=ZeroFeeModel())
        engine.set_strategy(MACrossover(short=10, long=30))
        res = engine.run(data, symbol="X")
        d = deflated_sharpe_ratio(res, n_trials=100)
        assert isinstance(d, DSRResult)
