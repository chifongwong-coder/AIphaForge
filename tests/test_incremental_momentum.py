"""v2.6 Commit E — MomentumIncremental tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.factor_library import MomentumFactor
from aiphaforge.incremental_factors import MomentumIncremental
from tests._helpers.incremental import assert_batch_incremental_equivalent


def _data(periods=200, seed=42):
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    return pd.DataFrame(
        {"close": closes},
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


def test_batch_equivalence_against_v2_4_momentum_factor():
    # MomentumFactor.compute returns a 1-column DataFrame; extract.
    factor = MomentumIncremental(window=20)
    assert_batch_incremental_equivalent(
        lambda d: MomentumFactor(window=20).compute(d).iloc[:, 0],
        factor,
        _data(),
        rtol=1e-12, atol=1e-15,
    )


def test_per_bar_update_against_hand_traced_5_bars():
    # window=2, closes=[10,20,30,40,50]
    # bar 0: NaN (need close[t-window]=close[-2])
    # bar 1: NaN
    # bar 2: 30/10 - 1 = 2.0
    # bar 3: 40/20 - 1 = 1.0
    # bar 4: 50/30 - 1 = 2/3
    factor = MomentumIncremental(window=2)
    state = factor.initial_state()
    closes = [10.0, 20.0, 30.0, 40.0, 50.0]
    expected = [float("nan"), float("nan"), 2.0, 1.0, 2.0 / 3.0]
    for c, exp in zip(closes, expected):
        v, state = factor.update(pd.Series({"close": c}), state)
        if np.isnan(exp):
            assert np.isnan(v)
        else:
            assert v == pytest.approx(exp, rel=1e-12)


def test_warmup_bars_are_nan():
    # Per design doc §I: first `window` bars NaN; bar `window` first non-NaN.
    factor = MomentumIncremental(window=10)
    out = factor.run_all(_data(periods=20))
    assert out.iloc[:10].isna().all()
    assert not pd.isna(out.iloc[10])


def test_invalid_window_raises():
    with pytest.raises(ValueError, match="window"):
        MomentumIncremental(window=0)
