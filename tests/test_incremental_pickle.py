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
    # Run 50 bars on a fresh factor; pickle the state; unpickle into
    # a second factor instance; run 50 more bars; assert the result
    # equals running all 100 bars on a single instance.
    data = _data(periods=100)
    first_half = data.iloc[:50]
    second_half = data.iloc[50:]

    factor_a = factor_factory()
    state = factor_a.initial_state()
    for _, bar in first_half.iterrows():
        _, state = factor_a.update(bar, state)

    blob = pickle.dumps(state)
    restored = pickle.loads(blob)
    assert restored.bar_count == state.bar_count

    factor_b = factor_factory()
    for _, bar in second_half.iterrows():
        _, restored = factor_b.update(bar, restored)

    factor_full = factor_factory()
    expected = factor_full.run_all(data).iloc[-1]
    actual_after_pickle, _ = factor_b.update(
        second_half.iloc[-1], restored,
    )
    # Approximate compare — the recursive Welford / Wilder factors
    # have a small floating-point drift between split-and-recombine
    # vs single-pass; tolerance loosely matches per-factor §R15 atol.
    if np.isnan(expected):
        assert np.isnan(actual_after_pickle)
    else:
        # The factor_b state after second_half processing has already
        # consumed the last bar; we need to compare last value WITHIN
        # the second-half pass. Simpler: compare full-recompute vs
        # split-via-pickle at the LAST bar of second_half.
        # Re-run cleanly via run_all to get the final value.
        full_out = factor_factory().run_all(data)
        split_factor = factor_factory()
        s = split_factor.initial_state()
        for _, bar in first_half.iterrows():
            _, s = split_factor.update(bar, s)
        s = pickle.loads(pickle.dumps(s))
        for _, bar in second_half.iterrows():
            v, s = split_factor.update(bar, s)
        assert v == pytest.approx(
            full_out.iloc[-1], rel=1e-9, abs=1e-12,
        )
