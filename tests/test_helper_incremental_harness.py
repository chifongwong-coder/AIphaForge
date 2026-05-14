"""v2.6 Commit B — tests for the batch-equivalence harness itself.

Sanity-check the helper before relying on it in Commits C-G.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.incremental_factors import FactorState, IncrementalFactor
from tests._helpers.incremental import assert_batch_incremental_equivalent


def _data(periods=20):
    rng = np.random.default_rng(0)
    closes = 100.0 + np.cumsum(rng.normal(0, 1, periods))
    return pd.DataFrame(
        {"close": closes},
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


class _IdentityIncremental(IncrementalFactor):
    """Returns close[t] every bar."""

    name = "identity"

    def initial_state(self):
        return FactorState()

    def update(self, bar_row, state):
        return float(bar_row["close"]), FactorState(
            bar_count=state.bar_count + 1,
        )


class _OffByOneIncremental(IncrementalFactor):
    """Returns close[t] + 1 every bar — should fail equivalence."""

    name = "off_by_one"

    def initial_state(self):
        return FactorState()

    def update(self, bar_row, state):
        return float(bar_row["close"]) + 1.0, FactorState(
            bar_count=state.bar_count + 1,
        )


class _NaNWarmupIncremental(IncrementalFactor):
    """NaN for first 3 bars, then close."""

    name = "nan_warmup"

    def initial_state(self):
        return FactorState()

    def update(self, bar_row, state):
        if state.bar_count < 3:
            value = float("nan")
        else:
            value = float(bar_row["close"])
        return value, FactorState(bar_count=state.bar_count + 1)


def test_helper_passes_when_factors_match():
    data = _data()
    batch_fn = lambda d: d["close"].astype(float)  # noqa: E731
    assert_batch_incremental_equivalent(
        batch_fn, _IdentityIncremental(), data,
        rtol=1e-12, atol=1e-15,
    )


def test_helper_fails_with_descriptive_diff_on_mismatch():
    data = _data()
    batch_fn = lambda d: d["close"].astype(float)  # noqa: E731
    with pytest.raises(AssertionError):
        assert_batch_incremental_equivalent(
            batch_fn, _OffByOneIncremental(), data,
            rtol=1e-12, atol=1e-15,
        )


def test_helper_accepts_callable_batch_reference():
    # Same identity check, just exercising the callable code path
    # explicitly (the dispatch picks function over BaseFactor).
    data = _data()
    assert_batch_incremental_equivalent(
        lambda d: d["close"].astype(float),
        _IdentityIncremental(),
        data,
        rtol=1e-12,
    )


def test_helper_handles_nan_warmup_period():
    # When both batch and incremental output NaN at the same bars,
    # assert_series_equal treats NaN==NaN so the helper passes.
    data = _data()

    def batch_fn(d):
        out = d["close"].astype(float).copy()
        out.iloc[:3] = np.nan
        return out

    assert_batch_incremental_equivalent(
        batch_fn, _NaNWarmupIncremental(), data,
        rtol=1e-12, atol=1e-15,
    )
