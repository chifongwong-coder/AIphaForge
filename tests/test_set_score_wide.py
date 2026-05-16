"""v2.7 Commit B — set_score_wide(score, rule) entry point.

Covers the 8 tests required by v2.7 plan §"Commit B":

- threshold rule (multi-asset) / cross-sectional rule
- invalid rule rejection / non-DataFrame rejection
- rule returning Series → blames the rule, not set_signals_wide
- strict promotes Inf to error
- strict + warn_on_inf=False raises (v3-decision #20)
- atomicity: rule.transform raise leaves no stale _signals
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.engine import BacktestEngine
from aiphaforge.signal_rules import (
    CrossSectionalQuantileRule,
    ThresholdScoreRule,
)


def _bdays(n: int = 30) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def _ohlcv(idx: pd.DatetimeIndex, base: float = 100.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = base + np.cumsum(rng.normal(0, 1, len(idx)))
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": 1_000_000.0,
        },
        index=idx,
    )


def _multi_data(symbols=("AAPL", "MSFT", "TSLA"), n: int = 30):
    idx = _bdays(n)
    return {
        sym: _ohlcv(idx, base=100.0 + 10.0 * i, seed=42 + i)
        for i, sym in enumerate(symbols)
    }


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_set_score_wide_with_threshold_rule_multi_asset():
    # Contract: set_score_wide(scores, rule) == set_signals_wide(rule.transform(scores))
    data = _multi_data()
    idx = next(iter(data.values())).index
    rng = np.random.default_rng(123)
    scores = pd.DataFrame(
        rng.uniform(0, 1, size=(len(idx), len(data))),
        index=idx, columns=list(data.keys()),
    )
    rule = ThresholdScoreRule(long_threshold=0.7, short_threshold=0.3)

    engine_a = BacktestEngine(initial_capital=100_000.0)
    engine_a.set_signals_wide(rule.transform(scores))
    baseline = engine_a.run(data)

    engine_b = BacktestEngine(initial_capital=100_000.0)
    engine_b.set_score_wide(scores, rule)
    result = engine_b.run(data)

    assert result.total_return == pytest.approx(baseline.total_return, abs=1e-12)
    assert len(result.trades) == len(baseline.trades)


def test_set_score_wide_with_cross_sectional_rule():
    # Equivalence: set_score_wide(scores, CrossSec) ≡
    # set_signals_wide(CrossSec.transform(scores)).
    data = _multi_data()
    idx = next(iter(data.values())).index
    rng = np.random.default_rng(7)
    scores = pd.DataFrame(
        rng.normal(0, 1, size=(len(idx), len(data))),
        index=idx, columns=list(data.keys()),
    )
    rule = CrossSectionalQuantileRule(long_quantile=0.33, short_quantile=0.33)

    engine_baseline = BacktestEngine(initial_capital=100_000.0)
    engine_baseline.set_signals_wide(rule.transform(scores))
    baseline = engine_baseline.run(data)

    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_score_wide(scores, rule)
    result = engine.run(data)

    assert result.total_return == pytest.approx(baseline.total_return, abs=1e-12)
    assert len(result.trades) == len(baseline.trades)


def test_set_score_wide_cross_sectional_rule_single_column_degenerate():
    # Coverage gap (plan §"CrossSec single-asset"): CrossSec ranks ACROSS
    # columns per bar. A 1-col DF has nothing to rank against → output is
    # the rule's neutral_action (default "flat" → 0) on every bar.
    # Test pins the degenerate behavior so it doesn't silently change.
    idx = _bdays(20)
    scores = pd.DataFrame({"AAPL": np.linspace(0.1, 1.0, 20)}, index=idx)
    rule = CrossSectionalQuantileRule(long_quantile=0.33, short_quantile=0.33)
    transformed = rule.transform(scores)
    # All values should be the neutral_action (0 by default).
    assert (transformed.fillna(0) == 0).all().all(), (
        f"1-col CrossSec should be all-neutral; got: {transformed.iloc[0].tolist()}"
    )


# ---------------------------------------------------------------------------
# Validation rejection
# ---------------------------------------------------------------------------


def test_set_score_wide_rejects_invalid_rule():
    engine = BacktestEngine()
    idx = _bdays()
    scores = pd.DataFrame({"AAPL": 0.5}, index=idx)
    # Rule lacks .transform()
    for bad_rule in [lambda x: x, None, 42, "not a rule"]:
        with pytest.raises(TypeError, match="set_score_wide rule must have"):
            engine.set_score_wide(scores, bad_rule)


def test_set_score_wide_rejects_non_dataframe():
    engine = BacktestEngine()
    rule = ThresholdScoreRule(long_threshold=0.7)
    with pytest.raises(TypeError, match="set_score_wide expects pd.DataFrame"):
        engine.set_score_wide(pd.Series([0.5, 0.8]), rule)
    with pytest.raises(TypeError, match="set_score_wide expects pd.DataFrame"):
        engine.set_score_wide({"AAPL": pd.Series([0.5])}, rule)


def test_set_score_wide_rule_returning_series_raises_pointing_at_rule():
    # Custom rule whose .transform returns a Series instead of DataFrame.
    class BadSeriesRule:
        def transform(self, scores):
            return pd.Series([1.0] * len(scores), index=scores.index)

    engine = BacktestEngine()
    idx = _bdays()
    scores = pd.DataFrame({"AAPL": 0.5}, index=idx)
    with pytest.raises(TypeError, match="BadSeriesRule.transform returned Series"):
        engine.set_score_wide(scores, BadSeriesRule())


# ---------------------------------------------------------------------------
# strict semantics (v3-decision #20)
# ---------------------------------------------------------------------------


def test_set_score_wide_strict_promotes_inf_to_error():
    # Rule passes through Inf → strict mode raises at run().
    class IdentityRule:
        def transform(self, scores):
            return scores  # passes Inf through unchanged

    data = _multi_data(symbols=("AAPL",))
    idx = next(iter(data.values())).index
    scores = pd.DataFrame({"AAPL": 1.0}, index=idx)
    scores.iloc[0] = np.inf
    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_score_wide(scores, IdentityRule(), strict=True)
    with pytest.raises(ValueError, match="strict=True.*Inf"):
        engine.run(data)


def test_set_score_wide_strict_with_warn_on_inf_false_raises():
    # v3-decision #20: strict + explicit conflicting kwarg → setter raises.
    idx = _bdays()
    scores = pd.DataFrame({"AAPL": 0.5}, index=idx)
    rule = ThresholdScoreRule(long_threshold=0.7)
    engine = BacktestEngine()
    with pytest.raises(
        ValueError,
        match="strict=True.*incompatible.*warn_on_inf=False",
    ):
        engine.set_score_wide(scores, rule, strict=True, warn_on_inf=False)


# ---------------------------------------------------------------------------
# Atomicity (v3-decision #26 — rule raise leaves no stale state)
# ---------------------------------------------------------------------------


def test_set_score_wide_rule_raise_leaves_no_stale_signals():
    # Set a prior signal, then call set_score_wide with a rule that raises.
    # Verify _signals was zeroed by atomicity step BEFORE the transform
    # was invoked, so no stale signal survives.
    class RaisingRule:
        def transform(self, scores):
            raise RuntimeError("boom")

    idx = _bdays()
    engine = BacktestEngine()
    engine.set_signals(pd.Series(1.0, index=idx))  # prior state
    assert engine._signals is not None

    scores = pd.DataFrame({"AAPL": 0.5}, index=idx)
    with pytest.raises(RuntimeError, match="RaisingRule.transform raised: boom"):
        engine.set_score_wide(scores, RaisingRule())

    # Atomicity invariant: prior _signals MUST be zeroed even though
    # the rule raised. Otherwise a subsequent run() would silently
    # use the stale signal.
    assert engine._signals is None, (
        "atomicity violated: rule.transform raise left stale _signals; "
        "set_score_wide must zero the 5 other state fields BEFORE "
        "invoking rule.transform"
    )
