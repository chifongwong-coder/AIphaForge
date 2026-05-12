"""PSR / DSR tests — v1.9.5 additions to aiphaforge.significance.

Requires scipy (lazy-imported by `aiphaforge.significance` itself).
``pip install -e ".[dev]"`` already pulls it in; the importorskip
below makes a `pytest tests/` invocation against a bare install
self-skip rather than crash at collection time.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

stats = pytest.importorskip("scipy.stats")

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
    # v2.2.2 Commit B: track the production default ddof=0 (Bailey-LdP
    # canonical biased σ; pairs with the √(n-1) standardization below).
    sr_per = float(rets.mean() / rets.std(ddof=0))
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
        # v2.2.2 Commit B: re-pinned for the new default ddof=0
        # (Bailey-LdP canonical biased σ). Previous pin was 0.9890164
        # under v2.2.1's ddof=1 default; the ~4e-5 shift is the
        # sqrt(T/(T-1)) bias factor on SR at T=750.
        np.random.seed(42)
        rets = pd.Series(np.random.normal(0.001, 0.01, 750))
        res = probabilistic_sharpe_ratio(rets, benchmark_sharpe=0.0)
        assert res.psr == pytest.approx(0.9890599896, abs=1e-7)

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

    def test_near_constant_returns_all_nan(self):
        """Prior bug: near-constant input (std ~ 1e-19) slipped past the
        ``std <= 0`` guard and leaked a huge observed_sharpe (2e16)
        alongside psr=NaN. Every numeric field must be NaN when the
        statistic is undefined."""
        const_like = pd.Series([0.001] * 100)  # std = 0 exactly
        res = probabilistic_sharpe_ratio(const_like)
        assert math.isnan(res.psr)
        assert math.isnan(res.observed_sharpe)
        assert math.isnan(res.skewness)
        assert math.isnan(res.kurtosis)

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
# v2.2.2 Commit B: canonical-formula PSR/DSR tests (hand-computed)
# ---------------------------------------------------------------------------


class TestPSRCanonicalBaileyLopezDePrado:
    """Hand-computed PSR at T=50 against Bailey & López de Prado 2012, eq. 14.

    WARNING: do NOT regenerate the expected value from production code.
    The point of this test is to verify that production matches the
    paper formula. If this test fails, suspect the production code first,
    not the expected value.

    Fixture: 25 returns of +0.01, 25 of -0.005, interleaved. Chosen so
    skewness is 0 by construction (symmetric deviations from μ) and the
    kurtosis ratio simplifies cleanly. T=50 keeps the bias-corrected
    moment estimators stable (the (n-1)/((n-2)(n-3)) factor in
    `stats.kurtosis(bias=False)` is ill-conditioned for small n).

    Arithmetic:
      μ = 0.0025
      Deviations: +0.0075 (for +0.01 entries), -0.0075 (for -0.005 entries)
      m2 = 0.0075² = 5.625e-5
      σ_biased = √m2 = 0.0075
      SR_biased = μ / σ_biased = 0.0025 / 0.0075 = 1/3
      m3 = 0 by symmetry → skew (bias-corrected G1) = 0
      m4 = 0.0075⁴
      biased kurt ratio m4/m2² = 1.0
      bias-corrected Pearson kurtosis (G2 + 3):
          G2 = (49/(48·47)) · (51·1.0 − 3·49) + 0
             = 49/2256 · (−96)
             = −4704/2256
             = −2.085106382978723
          Pearson_kurt = G2 + 3 = 0.914893617021277
      denom = 1 − γ₃·SR + (γ₄−1)/4 · SR²
            = 1 + (0.914893617... − 1) / 4 · (1/9)
            = 1 + (−0.085106382978723) / 36
            = 1 − 0.002364066194
            = 0.997635933806
      z = SR · √(T−1) / √denom = (1/3) · √49 / √0.997635933806
        = (7/3) / 0.998817217...
        = 2.336094...
      PSR = Φ(2.336094) ≈ 0.9902568887
    """

    def test_psr_t50_matches_bailey_lopez_de_prado_eq_14(self):
        rets = pd.Series(
            [0.01 if i % 2 == 0 else -0.005 for i in range(50)]
        )
        # Pass trading_days=252 to fully exercise the same code path as
        # the BacktestResult API (no impact on PSR — annualization only
        # affects observed_sharpe display).
        res = probabilistic_sharpe_ratio(
            rets, benchmark_sharpe=0.0, trading_days=252,
        )
        assert res.psr == pytest.approx(0.9902568887, abs=1e-7)
        # Sanity on intermediate fields per the hand calc.
        assert res.skewness == pytest.approx(0.0, abs=1e-10)
        assert res.kurtosis == pytest.approx(
            0.914893617021277, abs=1e-9,
        )

    def test_psr_explicit_ddof_one_recovers_v2_2_1_value(self):
        # Same fixture; std_ddof=1 should give a different value
        # related by the sqrt(T/(T-1)) bias factor on the SR numerator.
        rets = pd.Series(
            [0.01 if i % 2 == 0 else -0.005 for i in range(50)]
        )
        res_default = probabilistic_sharpe_ratio(rets, benchmark_sharpe=0.0)
        res_optout = probabilistic_sharpe_ratio(
            rets, benchmark_sharpe=0.0, std_ddof=1,
        )
        # The two values are different (no silent equivalence).
        assert res_default.psr != res_optout.psr
        # And the difference reflects the σ shift: ddof=1 σ is larger
        # by sqrt(T/(T-1)), so SR shrinks, so PSR drops.
        assert res_default.psr > res_optout.psr


class TestDSRCanonicalBaileyLopezDePrado:
    """Hand-computed DSR sanity at T=50, N=100 trials.

    WARNING: do NOT regenerate the expected value from production code.

    Uses the same fixture as the PSR canonical test. The DSR adds the
    Euler-correction term computing the expected max Sharpe under N
    trials (eq. 7) and runs PSR against that threshold. We don't pin
    DSR itself to a high-precision value (the Euler correction
    `(1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(Ne))` is well-tested elsewhere); we
    pin two structural invariants:

      1. expected_max_null_sharpe is positive and finite (the function
         actually fires the formula, not silently NaNs).
      2. DSR < PSR(benchmark=0) by a meaningful margin (the deflation
         actually deflates).
    """

    def test_dsr_t50_n100_deflates_psr(self):
        rets = pd.Series(
            [0.01 if i % 2 == 0 else -0.005 for i in range(50)]
        )
        psr = probabilistic_sharpe_ratio(
            rets, benchmark_sharpe=0.0, trading_days=252,
        )
        dsr = deflated_sharpe_ratio(
            rets, n_trials=100, trading_days=252,
        )
        assert np.isfinite(dsr.expected_max_null_sharpe)
        assert dsr.expected_max_null_sharpe > 0
        # Deflation actually deflates.
        assert dsr.dsr < psr.psr - 0.001


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

    def test_sr_zero_matches_bailey_2014_formula(self):
        """Lock sr_zero against the textbook Bailey-LopezdePrado 2014 eq.6+7.

        A prior bug multiplied sd_SR by std(returns), causing sr_zero to be
        ~100× too small and DSR to lose all deflation power (monotonicity
        alone couldn't catch it). This test locks the absolute value.
        """
        np.random.seed(0)
        T, td, N = 252, 252, 1000
        rets = pd.Series(np.random.normal(0.001, 0.01, T))

        # Manual Bailey 2014 calc (using full skew/kurt expression).
        # v2.2.2 Commit B: ddof=0 to match the production default
        # (Bailey-LdP canonical biased σ).
        sr_per = float(rets.mean() / rets.std(ddof=0))
        skew = float(stats.skew(rets, bias=False))
        kurt_pearson = float(stats.kurtosis(rets, fisher=False, bias=False))
        var_sr_per = (1 - skew * sr_per
                      + (kurt_pearson - 1) / 4 * sr_per ** 2) / (T - 1)
        sd_sr_ann = np.sqrt(var_sr_per) * np.sqrt(td)
        gamma = 0.5772156649
        factor = (
            (1 - gamma) * stats.norm.ppf(1 - 1 / N)
            + gamma * stats.norm.ppf(1 - 1 / (N * np.e))
        )
        sr_zero_expected = sd_sr_ann * factor

        d = deflated_sharpe_ratio(rets, n_trials=N, trading_days=td)
        assert d.expected_max_null_sharpe == pytest.approx(sr_zero_expected, rel=1e-9)
        # And the DSR == PSR(obs, benchmark=sr_zero_expected)
        from aiphaforge import probabilistic_sharpe_ratio
        expected_dsr = probabilistic_sharpe_ratio(
            rets, benchmark_sharpe=sr_zero_expected, trading_days=td).psr
        assert d.dsr == pytest.approx(expected_dsr, rel=1e-9)

    def test_dsr_deflates_meaningfully_vs_psr_at_zero(self):
        """At N=1000 trials on normal returns, DSR should be materially
        below PSR(SR*=0) — if they agree, deflation is broken."""
        np.random.seed(0)
        rets = pd.Series(np.random.normal(0.001, 0.01, 252))
        from aiphaforge import probabilistic_sharpe_ratio
        psr_zero = probabilistic_sharpe_ratio(rets, benchmark_sharpe=0.0).psr
        dsr_n1000 = deflated_sharpe_ratio(rets, n_trials=1000).dsr
        # Deflation should cost at least 30 percentage points (was 0.4pp pre-fix)
        assert psr_zero - dsr_n1000 > 0.30, \
            f"DSR ({dsr_n1000:.4f}) too close to PSR@0 ({psr_zero:.4f}) — deflation broken"

    def test_dsr_n1_is_undefined(self):
        """1 trial = no multiple-comparison problem; Bailey 2014 eq.7 is
        undefined at N=1 (stats.norm.ppf(0) = -∞). The engine returns
        NaN rather than silently falling through to PSR(SR>-∞)=1.0 —
        the earlier version of this test wrongly locked the 1.0 as a
        feature, missing the ppf(0) degeneracy entirely."""
        np.random.seed(0)
        rets = pd.Series(np.random.normal(0.001, 0.01, 252))
        d = deflated_sharpe_ratio(rets, n_trials=1)
        assert math.isnan(d.dsr)
        assert math.isnan(d.expected_max_null_sharpe)

    def test_dsr_n2_is_defined(self):
        """N=2 is the smallest meaningful deflation case; must be a valid number."""
        np.random.seed(0)
        rets = pd.Series(np.random.normal(0.001, 0.01, 252))
        d = deflated_sharpe_ratio(rets, n_trials=2)
        assert not math.isnan(d.dsr)
        assert 0.0 <= d.dsr <= 1.0

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
