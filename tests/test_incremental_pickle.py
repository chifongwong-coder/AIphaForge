"""v2.6 Commit J — public exports + parametrized cross-factor pickle tests."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

import aiphaforge


def _data(periods=100, seed=42):
    rng = np.random.default_rng(seed)
    closes = 100.0 + np.cumsum(rng.normal(0, 1, periods))
    volumes = rng.lognormal(mean=14.0, sigma=0.5, size=periods)
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": volumes,
        },
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


# ---------------------------------------------------------------------------
# Top-level export sentinels
# ---------------------------------------------------------------------------


def test_incremental_factor_top_level_export():
    assert hasattr(aiphaforge, "IncrementalFactor")
    assert "IncrementalFactor" in aiphaforge.__all__


def test_factor_state_top_level_export():
    assert hasattr(aiphaforge, "FactorState")
    assert "FactorState" in aiphaforge.__all__


def test_5_concrete_factors_top_level_exports():
    expected = [
        "RollingMeanIncremental",
        "RollingStdIncremental",
        "MomentumIncremental",
        "RSIIncremental",
        "VolumeZScoreIncremental",
    ]
    for name in expected:
        assert hasattr(aiphaforge, name), f"missing top-level export: {name}"
        assert name in aiphaforge.__all__, f"{name} not in __all__"


# ---------------------------------------------------------------------------
# Parametrized cross-factor pickle round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factor_factory",
    [
        lambda: aiphaforge.RollingMeanIncremental(window=10),
        lambda: aiphaforge.RollingStdIncremental(window=10),
        lambda: aiphaforge.MomentumIncremental(window=10),
        lambda: aiphaforge.RSIIncremental(period=14),
        lambda: aiphaforge.VolumeZScoreIncremental(window=10),
    ],
    ids=[
        "RollingMeanIncremental",
        "RollingStdIncremental",
        "MomentumIncremental",
        "RSIIncremental",
        "VolumeZScoreIncremental",
    ],
)
def test_concrete_factor_state_picklable(factor_factory):
    # Pickle round-trip on a Python float is bit-exact (IEEE 754 double
    # via pickle protocol >=4) and resuming the same per-bar recursion
    # is deterministic, so split-via-pickle MUST equal single-pass at
    # the last value bit-for-bit. A loose tolerance here would silently
    # mask a serializer regression that drops a digit.
    data = _data(periods=100)
    first_half = data.iloc[:50]
    second_half = data.iloc[50:]

    full_out = factor_factory().run_all(data)
    expected = full_out.iloc[-1]

    split_factor = factor_factory()
    s = split_factor.initial_state()
    for _, bar in first_half.iterrows():
        _, s = split_factor.update(bar, s)
    s = pickle.loads(pickle.dumps(s))
    assert s.bar_count == len(first_half)
    for _, bar in second_half.iterrows():
        v, s = split_factor.update(bar, s)

    if np.isnan(expected):
        assert np.isnan(v)
    else:
        assert v == expected
