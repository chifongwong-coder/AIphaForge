"""v2.6 Commit G — VolumeZScoreIncremental tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.factor_library import VolumeZScoreFactor
from aiphaforge.incremental_factors import VolumeZScoreIncremental
from tests._helpers.incremental import assert_batch_incremental_equivalent


def _data(periods=200, seed=42):
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    volumes = rng.lognormal(mean=14.0, sigma=0.5, size=periods)
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": volumes,
        },
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


def test_batch_equivalence_against_v2_4_volume_zscore_factor():
    factor = VolumeZScoreIncremental(window=20)
    assert_batch_incremental_equivalent(
        lambda d: VolumeZScoreFactor(window=20).compute(d).iloc[:, 0],
        factor,
        _data(),
        rtol=1e-9, atol=1e-12,
    )


def test_per_bar_update_against_hand_traced_5_bars():
    # window=3, volumes=[10,20,30,40,50]
    # bar 0,1: NaN
    # bar 2: mean(10,20,30)=20, std=10, z = (30-20)/10 = 1.0
    # bar 3: mean(20,30,40)=30, std=10, z = (40-30)/10 = 1.0
    # bar 4: mean(30,40,50)=40, std=10, z = (50-40)/10 = 1.0
    factor = VolumeZScoreIncremental(window=3)
    state = factor.initial_state()
    volumes = [10.0, 20.0, 30.0, 40.0, 50.0]
    expected = [float("nan"), float("nan"), 1.0, 1.0, 1.0]
    for v, exp in zip(volumes, expected):
        out, state = factor.update(pd.Series({"volume": v}), state)
        if np.isnan(exp):
            assert np.isnan(out)
        else:
            assert out == pytest.approx(exp, rel=1e-9, abs=1e-12)


def test_warmup_bars_are_nan():
    factor = VolumeZScoreIncremental(window=10)
    out = factor.run_all(_data(periods=20))
    assert out.iloc[:9].isna().all()
    assert not pd.isna(out.iloc[9])


def test_invalid_window_raises():
    with pytest.raises(ValueError, match="window"):
        VolumeZScoreIncremental(window=1)
