"""v2.0 M4 — tests for the A/B probe runner."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from aiphaforge.probes import (
    ABScenario,
    MACrossBaseline,
    MeanRevBaseline,
    MetricConfig,
    run_ab_probe,
)
from aiphaforge.probes.abtest import (
    DEFAULT_METRIC_CONFIG,
    _capacity_parity_warning,
    _check_agent_determinism,
    _compute_relative_drop,
    _extract_metric,
)
from aiphaforge.probes.transforms import (
    BlockBootstrap as BB,
)
from aiphaforge.probes.transforms import (
    SymbolMasker as SM,
)
from aiphaforge.results import BacktestResult


def _ohlcv(n: int = 120, seed: int = 0, start: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = start * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n))
    opens = closes * (1.0 + rng.normal(0.0, 0.003, size=n))
    spreads = np.abs(rng.normal(0.0, 0.005, size=n)) * closes
    highs = np.maximum(opens, closes) + spreads
    lows = np.minimum(opens, closes) - spreads
    vol = rng.integers(1_000_000, 10_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vol},
        index=pd.bdate_range("2024-01-01", periods=n),
    )


def _ai_factory():
    """Toy AI: same MA crossover as baseline (zero excess by design)."""
    return MACrossBaseline(short=5, long=20)


def _baseline_factory():
    return MACrossBaseline(short=5, long=20)


def _alt_baseline_factory():
    return MeanRevBaseline(window=20)


# ---------- Metric extraction ----------

class TestMetricExtraction:
    def test_extract_total_return(self):
        # Build a tiny BacktestResult by hand.
        idx = pd.bdate_range("2024-01-01", periods=10)
        result = BacktestResult(
            equity_curve=pd.Series([100.0, 110.0] * 5, index=idx),
            trades=[],
            positions=pd.DataFrame(),
            metrics={"sharpe_ratio": 1.5, "max_drawdown": 0.05},
            initial_capital=100.0,
            final_capital=110.0,
        )
        assert _extract_metric(result, "total_return") == pytest.approx(0.10)
        assert _extract_metric(result, "sharpe_ratio") == pytest.approx(1.5)
        assert _extract_metric(result, "max_drawdown") == pytest.approx(0.05)
        assert _extract_metric(result, "trade_count") == 0.0

    def test_missing_metric_returns_nan(self):
        idx = pd.bdate_range("2024-01-01", periods=2)
        result = BacktestResult(
            equity_curve=pd.Series([100.0, 100.0], index=idx),
            trades=[], positions=pd.DataFrame(), metrics={},
            initial_capital=100.0, final_capital=100.0,
        )
        v = _extract_metric(result, "no_such_metric")
        assert np.isnan(v)


# ---------- Drop math ----------

class TestRelativeDrop:
    def _cfg(self, **kw):
        defaults = dict(higher_is_better=True, normalization_floor=1e-3,
                        low_anchor_threshold=0.01)
        defaults.update(kw)
        return MetricConfig(**defaults)

    def test_symmetric_normalization(self):
        # raw=1.0, test=0.8 → drop=0.2; denom=max(1.0, 0.8, floor)=1.0; rel=0.2
        abs_d, rel, low = _compute_relative_drop(1.0, 0.8, self._cfg())
        assert abs_d == pytest.approx(0.2)
        assert rel == pytest.approx(0.2)
        assert low is False

    def test_low_anchor_returns_none_rel(self):
        # raw=0.001, test=0.0005, threshold=0.01 → low-anchor.
        abs_d, rel, low = _compute_relative_drop(0.001, 0.0005, self._cfg())
        assert low is True
        assert rel is None

    def test_higher_is_better_false_inverts(self):
        # max_drawdown: positive magnitude where larger is worse.
        # raw=0.05, test=0.10 → AI degraded under transform.
        cfg = self._cfg(higher_is_better=False)
        abs_d, rel, low = _compute_relative_drop(0.05, 0.10, cfg)
        # Oriented: -0.05 vs -0.10; abs_drop = -0.05 - (-0.10) = +0.05.
        # Positive abs_drop means "got worse" under transform.
        assert abs_d == pytest.approx(0.05)
        assert rel == pytest.approx(0.5)


# ---------- Capacity parity ----------

class TestCapacityParity:
    def _make_result(self, n_trades: int) -> BacktestResult:
        from aiphaforge.results import Trade
        idx = pd.bdate_range("2024-01-01", periods=10)
        trades = [
            Trade(trade_id=i, symbol="X", direction=1,
                  entry_time=idx[0], exit_time=idx[1],
                  entry_price=100.0, exit_price=101.0,
                  size=1.0, pnl=1.0, pnl_pct=0.01)
            for i in range(n_trades)
        ]
        return BacktestResult(
            equity_curve=pd.Series([100.0] * 10, index=idx),
            trades=trades, positions=pd.DataFrame(), metrics={},
            initial_capital=100.0, final_capital=100.0,
        )

    def test_within_band_no_warning(self):
        ai = self._make_result(10)
        base = self._make_result(15)
        assert _capacity_parity_warning(ai, base) is None

    def test_outside_band_warns(self):
        ai = self._make_result(2)
        base = self._make_result(20)
        msg = _capacity_parity_warning(ai, base)
        assert msg is not None
        assert "capacity_parity_warning" in msg


# ---------- End-to-end (small smoke runs) ----------

class TestRunABProbe:
    def test_market_level_with_symbolmasker_runs(self):
        data = _ohlcv(n=80)
        scenarios = [
            ABScenario(
                scenario_id="metadata_only",
                mode="market_level",
                transforms=[SM(symbols=["AAPL"], seed=0)],
            ),
        ]
        result = run_ab_probe(
            ai_factory=_ai_factory,
            baseline_factory=_baseline_factory,
            data=data,
            scenarios=scenarios,
            metrics=("total_return", "sharpe_ratio", "trade_count"),
            n_repeat=3,
            min_valid_repeats=1,
            engine_kwargs={"include_benchmark": False},
        )
        assert len(result.scenarios) == 1
        rep = result.scenarios[0]
        assert rep.scenario_id == "metadata_only"
        assert rep.n_repeat_requested == 3
        # SymbolMasker.apply is a no-op on data, so AI=baseline (same
        # strategy) and excess_drop ≈ 0.
        for m in ("total_return", "sharpe_ratio", "trade_count"):
            s = rep.metric_summaries[m]
            assert len(s.ai_raw) == 3
            assert len(s.excess_drop) == 3
        # Detectability warning for SymbolMasker should be present.
        assert any("transform_detectability_warning" in w for w in rep.warnings)

    def test_view_only_with_strategy_runs(self):
        data = _ohlcv(n=80)
        scenarios = [
            ABScenario(
                scenario_id="metadata_view_only",
                mode="view_only",
                transforms=[SM(symbols=["AAPL"], seed=0)],
            ),
        ]
        result = run_ab_probe(
            ai_factory=_ai_factory,
            baseline_factory=_baseline_factory,
            data=data,
            scenarios=scenarios,
            metrics=("total_return", "trade_count"),
            n_repeat=2,
            min_valid_repeats=1,
            engine_kwargs={"include_benchmark": False},
        )
        # Strategy-based factory in view_only is the supported v2.0 path.
        assert result.scenarios[0].n_repeat_requested == 2

    def test_min_valid_repeats_gates_summary_scalars(self):
        data = _ohlcv(n=80)
        scenarios = [
            ABScenario(
                scenario_id="m",
                mode="market_level",
                transforms=[SM(symbols=["AAPL"], seed=0)],
            ),
        ]
        # Force every repeat to be low-anchor by setting an absurdly
        # high low_anchor_threshold for total_return.
        cfg = dict(DEFAULT_METRIC_CONFIG)
        cfg["total_return"] = MetricConfig(
            higher_is_better=True,
            normalization_floor=1e-3,
            low_anchor_threshold=1e10,
        )
        result = run_ab_probe(
            ai_factory=_ai_factory,
            baseline_factory=_baseline_factory,
            data=data,
            scenarios=scenarios,
            metrics=("total_return",),
            n_repeat=3,
            min_valid_repeats=2,
            metric_config=cfg,
            engine_kwargs={"include_benchmark": False},
        )
        s = result.scenarios[0].metric_summaries["total_return"]
        assert s.n_low_anchor_ai == 3
        assert s.n_valid_relative == 0
        # Summary scalars must be None when below min_valid_repeats.
        assert s.dominance_rate is None
        assert s.mean_excess_drop is None
        assert s.median_excess_drop is None

    def test_seeds_length_must_match_n_repeat(self):
        data = _ohlcv(n=40)
        with pytest.raises(ValueError, match="seeds length"):
            run_ab_probe(
                ai_factory=_ai_factory,
                baseline_factory=_baseline_factory,
                data=data,
                scenarios=[ABScenario(
                    scenario_id="x", mode="market_level",
                    transforms=[SM(symbols=["AAPL"], seed=0)],
                )],
                n_repeat=3,
                seeds=[0, 1],  # length mismatch
                metrics=("total_return",),
                engine_kwargs={"include_benchmark": False},
            )

    def test_unknown_metric_raises(self):
        data = _ohlcv(n=40)
        with pytest.raises(KeyError, match="MetricConfig"):
            run_ab_probe(
                ai_factory=_ai_factory,
                baseline_factory=_baseline_factory,
                data=data,
                scenarios=[ABScenario(
                    scenario_id="x", mode="market_level",
                    transforms=[SM(symbols=["AAPL"], seed=0)],
                )],
                n_repeat=2,
                metrics=("no_such_metric_at_all",),
                engine_kwargs={"include_benchmark": False},
            )

    def test_view_only_with_hook_factory_raises(self):
        # Hook-based agent in view_only mode is deferred to v2.0.x;
        # the runner must surface a clear NotImplementedError rather
        # than silently misbehave.
        from aiphaforge.hooks import BacktestHook

        class _DummyHook(BacktestHook):
            def on_pre_signal(self, ctx):
                pass

        data = _ohlcv(n=40)
        with pytest.raises(NotImplementedError, match="broker-proxy wrapper"):
            run_ab_probe(
                ai_factory=lambda: _DummyHook(),
                baseline_factory=_baseline_factory,
                data=data,
                scenarios=[ABScenario(
                    scenario_id="x", mode="view_only",
                    transforms=[SM(symbols=["AAPL"], seed=0)],
                )],
                metrics=("total_return",),
                n_repeat=1,
                min_valid_repeats=1,
                engine_kwargs={"include_benchmark": False},
            )

    def test_manifest_captures_seeds_and_data_hash(self):
        data = _ohlcv(n=40)
        result = run_ab_probe(
            ai_factory=_ai_factory,
            baseline_factory=_baseline_factory,
            data=data,
            scenarios=[ABScenario(
                scenario_id="x", mode="market_level",
                transforms=[SM(symbols=["AAPL"], seed=0)],
            )],
            metrics=("total_return",),
            n_repeat=2,
            seeds=[7, 11],
            min_valid_repeats=1,
            provider_config={"model": "stub", "temperature": 0.0},
            engine_kwargs={"include_benchmark": False},
        )
        m = result.manifest
        assert m["seeds_used"] == [7, 11]
        assert m["data_hash"]  # non-empty
        assert m["provider_config"]["model"] == "stub"
        assert "metric_config" in m
        assert m["scenarios"][0]["scenario_id"] == "x"

    def test_noise_control_skipped_on_deterministic_pipeline(self):
        # enable_ai_noise_control=True on a deterministic transform
        # pipeline must hard-disable the control and emit a warning.
        data = _ohlcv(n=40)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_ab_probe(
                ai_factory=_ai_factory,
                baseline_factory=_baseline_factory,
                data=data,
                scenarios=[ABScenario(
                    scenario_id="det", mode="market_level",
                    transforms=[SM(symbols=["AAPL"], seed=0)],
                )],
                metrics=("total_return",),
                n_repeat=2,
                min_valid_repeats=1,
                enable_ai_noise_control=True,
                engine_kwargs={"include_benchmark": False},
            )
        assert any("vacuous" in str(w.message) for w in caught)
        s = result.scenarios[0].metric_summaries["total_return"]
        assert s.ai_noise_abs is None
        assert any("ai_noise_control_skipped" in w for w in result.scenarios[0].warnings)

    def test_noise_control_runs_on_stochastic_pipeline(self):
        data = _ohlcv(n=80)
        result = run_ab_probe(
            ai_factory=_ai_factory,
            baseline_factory=_baseline_factory,
            data=data,
            scenarios=[ABScenario(
                scenario_id="stoch", mode="market_level",
                transforms=[BB(block_size=20)],  # stochastic
            )],
            metrics=("total_return",),
            n_repeat=3,
            min_valid_repeats=1,
            enable_ai_noise_control=True,
            engine_kwargs={"include_benchmark": False},
        )
        s = result.scenarios[0].metric_summaries["total_return"]
        # AI-on-AI noise lists must be populated for stochastic pipelines.
        assert s.ai_noise_abs is not None
        assert s.ai_noise_rel is not None
        assert len(s.ai_noise_abs) == 3
        assert s.mean_ai_noise_abs is not None
        assert s.iqr_ai_noise_rel is not None

    def test_paired_seed_reproducibility(self):
        # Same seeds → byte-identical results across two runs.
        data = _ohlcv(n=80)
        cfg_kwargs = dict(
            ai_factory=_ai_factory,
            baseline_factory=_baseline_factory,
            data=data,
            scenarios=[ABScenario(
                scenario_id="r", mode="market_level",
                transforms=[BB(block_size=20)],
            )],
            metrics=("total_return",),
            n_repeat=3,
            seeds=[1, 2, 3],
            min_valid_repeats=1,
            engine_kwargs={"include_benchmark": False},
        )
        r1 = run_ab_probe(**cfg_kwargs)
        r2 = run_ab_probe(**cfg_kwargs)
        s1 = r1.scenarios[0].metric_summaries["total_return"]
        s2 = r2.scenarios[0].metric_summaries["total_return"]
        np.testing.assert_array_equal(s1.ai_raw, s2.ai_raw)
        np.testing.assert_array_equal(s1.ai_test, s2.ai_test)
        np.testing.assert_array_equal(s1.baseline_raw, s2.baseline_raw)
        np.testing.assert_array_equal(s1.baseline_test, s2.baseline_test)


# ---------- Determinism check ----------

class TestDeterminismCheck:
    def test_deterministic_strategy_passes(self):
        data = _ohlcv(n=40)
        ok = _check_agent_determinism(
            _baseline_factory, data, seed=0,
            engine_kwargs={"include_benchmark": False},
        )
        assert ok is True

    def test_nondeterministic_strategy_flagged(self):
        # Wrap a strategy that adds genuine randomness on each call.
        class _RandomTilt:
            def __init__(self):
                self._rng = np.random.default_rng()

            def generate_signals(self, data):
                signals = pd.Series(
                    self._rng.choice([-1, 0, 1], size=len(data)),
                    index=data.index, dtype=float,
                )
                return signals

        data = _ohlcv(n=40)
        ok = _check_agent_determinism(
            lambda: _RandomTilt(), data, seed=0,
            engine_kwargs={"include_benchmark": False},
        )
        assert ok is False
