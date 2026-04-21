"""Regression tests for v1.9.9 fixes."""

import json

import numpy as np
import pandas as pd
import pytest

from aiphaforge import BacktestEngine
from aiphaforge.performance import PerformanceAnalyzer
from aiphaforge.strategies import MACrossover
from aiphaforge.utils import sharpe_ratio, sortino_ratio


def _synthetic_ohlcv(n=100, seed=0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": 1e6,
        },
        index=idx,
    )


class TestFloatingPointStdGuard:
    """sharpe/sortino/IR must treat near-zero std as zero.

    std of a constant Series is ~7e-18 (floating-point noise), not
    exactly zero. The old `std == 0` guard missed this and returned
    ~2e16. Switch to `std < 1e-12` to match probabilistic_sharpe_ratio.
    """

    def test_sharpe_on_constant_returns_is_zero(self):
        r = pd.Series([0.01] * 100)
        # Pre-fix this returned ~2.28e16
        assert sharpe_ratio(r) == 0.0

    def test_sortino_on_constant_returns_is_zero(self):
        r = pd.Series([0.01] * 100)
        # For all-positive constant returns, clipped is exactly zero
        # (hits the exact-zero branch pre-fix too), but this locks in
        # the contract.
        assert sortino_ratio(r) == 0.0

    def test_sortino_on_near_zero_mixed_noise_is_zero(self):
        # Mixed-sign floating-point noise — this is the case the
        # old exact-zero guard could miss when clipped values are
        # tiny nonzero.
        r = pd.Series([1e-18, -1e-18] * 50)
        assert sortino_ratio(r) == 0.0

    def test_information_ratio_on_matching_benchmark_is_zero(self):
        # excess.std() ~ 1e-18 when returns match the benchmark
        returns = pd.Series([0.01] * 50)
        benchmark = pd.Series([0.01] * 50)
        analyzer = PerformanceAnalyzer.__new__(PerformanceAnalyzer)
        analyzer.returns = returns
        analyzer.trading_days = 252
        # Pre-fix this returned ~infinity / garbage
        assert analyzer.information_ratio(benchmark) == 0.0


class TestSetTargetWeightsSingleAsset:
    """set_target_weights must work on single-asset runs.

    Pre-fix the single-asset `_get_signals` only checked _signals /
    _strategy and raised "Must set either a strategy or signals".
    """

    def test_single_asset_target_weights_runs(self):
        df = _synthetic_ohlcv(n=50)
        eng = BacktestEngine(initial_capital=100_000)
        eng.set_target_weights(
            {
                "2024-01-01": {"default": 0.5},
                "2024-02-01": {"default": 0.0},
            }
        )
        res = eng.run(df)
        assert res.equity_curve is not None
        assert len(res.equity_curve) == len(df)

    def test_single_asset_target_weights_respects_symbol_kwarg(self):
        df = _synthetic_ohlcv(n=50)
        eng = BacktestEngine(initial_capital=100_000)
        eng.set_target_weights(
            {"2024-01-01": {"AAPL": 0.5}, "2024-02-01": {"AAPL": 0.0}}
        )
        res = eng.run(df, symbol="AAPL")
        assert res.equity_curve is not None

    def test_missing_strategy_and_signals_still_raises(self):
        df = _synthetic_ohlcv(n=50)
        eng = BacktestEngine(initial_capital=100_000)
        with pytest.raises(ValueError, match="strategy or signals"):
            eng.run(df)


class TestToDictMultiAssetFields:
    """BacktestResult.to_dict() must include v1.9.5+ fields.

    Pre-fix `symbols`, `per_asset_metrics`, `turnover_history`, and
    `benchmark_name` were dropped on serialization.
    """

    def test_single_asset_to_dict_has_symbols(self):
        df = _synthetic_ohlcv(n=80, seed=1)
        eng = BacktestEngine(initial_capital=100_000).set_strategy(
            MACrossover(5, 20)
        )
        res = eng.run(df, symbol="SYNTH")
        d = res.to_dict()
        assert d.get("symbols") == ["SYNTH"]
        # And the whole thing is JSON-serializable
        json.dumps(d, default=str)

    def test_single_asset_to_dict_has_benchmark_name(self):
        df = _synthetic_ohlcv(n=80, seed=2)
        eng = BacktestEngine(initial_capital=100_000).set_strategy(
            MACrossover(5, 20)
        )
        res = eng.run(df)
        d = res.to_dict()
        # Default benchmark is Buy & Hold; benchmark_metrics populates
        # in single-asset mode too
        if "benchmark_metrics" in d:
            assert d.get("benchmark_name") == "Buy & Hold"

    def test_multi_asset_to_dict_has_per_asset_metrics(self):
        df_a = _synthetic_ohlcv(n=60, seed=3)
        df_b = _synthetic_ohlcv(n=60, seed=4)
        eng = BacktestEngine(initial_capital=100_000).set_strategy(
            MACrossover(5, 20)
        )
        res = eng.run({"AAA": df_a, "BBB": df_b})
        d = res.to_dict()
        assert set(d.get("symbols", [])) == {"AAA", "BBB"}
        # per_asset_metrics is populated for multi-asset
        assert "per_asset_metrics" in d
