"""v2.6 Commit F — RSIIncremental tests including SMA-seed transition."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.factor_library import RSIFactor
from aiphaforge.incremental_factors import RSIIncremental
from tests._helpers.incremental import assert_batch_incremental_equivalent


def _data(periods=200, seed=42):
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    return pd.DataFrame(
        {"close": closes},
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


def test_batch_equivalence_against_v2_4_rsi_factor():
    factor = RSIIncremental(period=14)
    assert_batch_incremental_equivalent(
        lambda d: RSIFactor(period=14).compute(d).iloc[:, 0],
        factor,
        _data(),
        rtol=1e-9, atol=1e-12,
    )


def test_per_bar_update_output_is_in_zero_one_hundred_range():
    factor = RSIIncremental(period=14)
    out = factor.run_all(_data())
    non_nan = out.dropna()
    assert (non_nan >= 0).all() and (non_nan <= 100).all()


def test_warmup_bars_are_nan():
    # Pandas ewm(min_periods=period) emits at index period-1 (after
    # observing `period` non-NaN inputs including bar 0). RSI matches.
    factor = RSIIncremental(period=14)
    out = factor.run_all(_data(periods=30))
    assert out.iloc[:13].isna().all()
    assert not pd.isna(out.iloc[13])


def test_invalid_period_raises():
    with pytest.raises(ValueError, match="period"):
        RSIIncremental(period=0)


# Seed-transition cliff: bars period-2, period-1, period vs v2.4
# RSIFactor. This is the cliff where SMA seed completes and the
# recursive Wilder formula starts. Most likely drift point.

@pytest.fixture
def _rsi_pair():
    period = 14
    data = _data()
    incremental = RSIIncremental(period=period).run_all(data)
    batch = RSIFactor(period=period).compute(data).iloc[:, 0]
    return incremental, batch, period


def test_rsi_incremental_matches_batch_at_period_minus_2(_rsi_pair):
    inc, batch, period = _rsi_pair
    # Last NaN bar — both should be NaN. (pandas first emits at
    # period-1, so period-2 is the last NaN.)
    assert pd.isna(inc.iloc[period - 2])
    assert pd.isna(batch.iloc[period - 2])


def test_rsi_incremental_matches_batch_at_period_minus_1(_rsi_pair):
    inc, batch, period = _rsi_pair
    # First non-NaN bar — pandas ewm seed transition lands here.
    assert inc.iloc[period - 1] == pytest.approx(
        batch.iloc[period - 1], rel=1e-9, abs=1e-12,
    )


def test_rsi_incremental_matches_batch_at_period(_rsi_pair):
    inc, batch, period = _rsi_pair
    # First fully-recursive Wilder bar — most likely drift point if
    # the seed-vs-recursion handoff is wrong.
    assert inc.iloc[period] == pytest.approx(
        batch.iloc[period], rel=1e-9, abs=1e-12,
    )
