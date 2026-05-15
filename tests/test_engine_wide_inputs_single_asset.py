"""v2.7 Commit G — single-asset edge cases for the 3 new wide setters.

A 1-column wide DF is technically wide-layout but semantically
single-asset. All 3 setters must accept it and produce results
equivalent to the corresponding non-wide single-asset call.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.engine import BacktestEngine
from aiphaforge.signal_rules import ThresholdScoreRule


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


def test_set_signals_wide_single_asset_dataframe_input():
    idx = pd.bdate_range("2024-01-01", periods=30)
    data = _ohlcv(idx)
    series = pd.Series(1.0, index=idx)
    series.iloc[0] = 0.0
    series.iloc[15] = 0.0  # mid-period exit so a closed trade is recorded

    # Series baseline
    engine_series = BacktestEngine(initial_capital=100_000.0)
    engine_series.set_signals(series)
    baseline = engine_series.run(data)

    # 1-column wide DF
    engine_wide = BacktestEngine(initial_capital=100_000.0)
    engine_wide.set_signals_wide(series.to_frame("default"))
    result = engine_wide.run(data)

    assert result.total_return == pytest.approx(baseline.total_return, abs=1e-12)
    assert len(result.trades) == len(baseline.trades)


def test_set_score_wide_single_asset_with_threshold_rule():
    # Per v3-decision #10: set_score_wide must work with single-column
    # wide DF + ThresholdScoreRule. (CrossSectionalQuantileRule
    # explicitly does NOT apply to single-asset — see signal_rules.py
    # docstring.)
    idx = pd.bdate_range("2024-01-01", periods=30)
    data = _ohlcv(idx)
    rng = np.random.default_rng(7)
    scores = pd.DataFrame({"default": rng.uniform(0, 1, len(idx))}, index=idx)
    rule = ThresholdScoreRule(long_threshold=0.7, short_threshold=0.3)

    # Equivalence: set_score_wide vs set_signals(rule.transform(scores)["default"])
    transformed = rule.transform(scores)["default"]
    engine_baseline = BacktestEngine(initial_capital=100_000.0)
    engine_baseline.set_signals(transformed)
    baseline = engine_baseline.run(data)

    engine_wide = BacktestEngine(initial_capital=100_000.0)
    engine_wide.set_score_wide(scores, rule)
    result = engine_wide.run(data)

    assert result.total_return == pytest.approx(baseline.total_return, abs=1e-12)
    assert len(result.trades) == len(baseline.trades)


def test_set_target_weights_wide_single_asset_passthrough():
    idx = pd.bdate_range("2024-01-01", periods=30)
    data = _ohlcv(idx)
    weights = pd.DataFrame({"default": [1.0] * len(idx)}, index=idx)

    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_target_weights_wide(weights)
    result = engine.run(data)
    assert result.total_return is not None
