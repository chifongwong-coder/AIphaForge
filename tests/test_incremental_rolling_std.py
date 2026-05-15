"""v2.6 Commit D — RollingStdIncremental tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.incremental_factors import RollingStdIncremental
from tests._helpers.incremental import assert_batch_incremental_equivalent


def _data(periods=200, seed=42):
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    return pd.DataFrame(
        {"close": closes},
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


def test_batch_equivalence_against_pandas_rolling_std():
    factor = RollingStdIncremental(window=20, ddof=1)
    assert_batch_incremental_equivalent(
        lambda d: d["close"].rolling(20).std(ddof=1),
        factor,
        _data(),
        rtol=1e-9, atol=1e-12,
    )


def test_per_bar_update_against_hand_traced_5_bars():
    # window=3, closes=[10,20,30,40,50]
    # bar 0,1: NaN (warmup)
    # bar 2: std([10,20,30], ddof=1) = sqrt(((10-20)² + 0 + (30-20)²)/2) = 10
    # bar 3: std([20,30,40], ddof=1) = 10
    # bar 4: std([30,40,50], ddof=1) = 10
    factor = RollingStdIncremental(window=3, ddof=1)
    state = factor.initial_state()
    closes = [10.0, 20.0, 30.0, 40.0, 50.0]
    expected = [float("nan"), float("nan"), 10.0, 10.0, 10.0]
    for c, exp in zip(closes, expected):
        v, state = factor.update(pd.Series({"close": c}), state)
        if np.isnan(exp):
            assert np.isnan(v)
        else:
            assert v == pytest.approx(exp, rel=1e-9, abs=1e-12)


def test_warmup_bars_are_nan():
    # First (window - 1) = 9 bars NaN; bar 9 first non-NaN.
    factor = RollingStdIncremental(window=10, ddof=1)
    out = factor.run_all(_data(periods=20))
    assert out.iloc[:9].isna().all()
    assert not pd.isna(out.iloc[9])


def test_invalid_params_raise():
    with pytest.raises(ValueError, match="window"):
        RollingStdIncremental(window=0)
    with pytest.raises(ValueError, match="ddof"):
        RollingStdIncremental(window=10, ddof=-1)
