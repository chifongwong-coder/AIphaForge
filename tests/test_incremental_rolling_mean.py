"""v2.6 Commit C — RollingMeanIncremental tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.incremental_factors import RollingMeanIncremental
from tests._helpers.incremental import assert_batch_incremental_equivalent


def _data(periods=200, seed=42):
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    return pd.DataFrame(
        {"close": closes},
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


def test_batch_equivalence_against_pandas_rolling_mean():
    # Closed-form running sum / count vs pandas batch rolling.
    # No v2.4 batch counterpart for this factor; reference is a
    # pandas one-liner wrapped as a callable.
    factor = RollingMeanIncremental(window=20)
    assert_batch_incremental_equivalent(
        lambda d: d["close"].rolling(20).mean(),
        factor,
        _data(),
        rtol=1e-12, atol=1e-15,
    )


def test_per_bar_update_against_hand_traced_5_bars():
    # window=3, closes=[10,20,30,40,50]
    # bar 0: warmup → NaN
    # bar 1: warmup → NaN
    # bar 2: mean(10,20,30) = 20
    # bar 3: mean(20,30,40) = 30
    # bar 4: mean(30,40,50) = 40
    factor = RollingMeanIncremental(window=3)
    state = factor.initial_state()
    closes = [10.0, 20.0, 30.0, 40.0, 50.0]
    expected = [float("nan"), float("nan"), 20.0, 30.0, 40.0]
    for c, exp in zip(closes, expected):
        v, state = factor.update(pd.Series({"close": c}), state)
        if np.isnan(exp):
            assert np.isnan(v)
        else:
            assert v == pytest.approx(exp, rel=1e-12)


def test_warmup_bars_are_nan():
    # First (window - 1) = 9 bars NaN, bar window-1=9 first non-NaN.
    factor = RollingMeanIncremental(window=10)
    out = factor.run_all(_data(periods=20))
    assert out.iloc[:9].isna().all()
    assert not pd.isna(out.iloc[9])


def test_invalid_window_raises():
    with pytest.raises(ValueError, match="window"):
        RollingMeanIncremental(window=0)


def test_name_includes_window_param():
    assert RollingMeanIncremental(window=20).name == "rolling_mean_20"
