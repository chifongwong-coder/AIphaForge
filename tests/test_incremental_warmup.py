"""v2.6 Commit I — per-factor warmup count tests.

Pins each factor's specific warmup count rather than a generic
``window-1``. Catches off-by-one drift on factor-specific math.

Note: RSIIncremental warmup is `period - 1` (not `period` as plan
v1 said). This matches pandas ewm(min_periods=period) semantics
where the first emit is at index period-1 once `period` non-NaN
inputs (including bar 0) have been observed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.incremental_factors import (
    MomentumIncremental,
    RollingMeanIncremental,
    RollingStdIncremental,
    RSIIncremental,
    VolumeZScoreIncremental,
)


def _ohlcv(periods=50, seed=0):
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


@pytest.mark.parametrize(
    ("factory", "warmup_n"),
    [
        (lambda: RollingMeanIncremental(window=10), 9),
        (lambda: RollingStdIncremental(window=10, ddof=1), 9),
        (lambda: MomentumIncremental(window=10), 10),
        (lambda: RSIIncremental(period=14), 13),
        (lambda: VolumeZScoreIncremental(window=10), 9),
    ],
    ids=[
        "RollingMean(window=10)",
        "RollingStd(window=10)",
        "Momentum(window=10)",
        "RSI(period=14)",
        "VolumeZScore(window=10)",
    ],
)
def test_per_factor_warmup_count(factory, warmup_n):
    factor = factory()
    out = factor.run_all(_ohlcv(periods=50))
    assert out.iloc[:warmup_n].isna().all(), (
        f"{factor.name}: bars 0..{warmup_n - 1} must be NaN; "
        f"got first non-NaN at "
        f"{int(out.notna().idxmax() == out.index[0]) if out.notna().any() else 'never'}"
    )
    assert not pd.isna(out.iloc[warmup_n]), (
        f"{factor.name}: bar {warmup_n} must be the first non-NaN value"
    )
