"""v2.6 Commit H — update_params + rewarmup tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.incremental_factors import RollingMeanIncremental


def _data(periods=50, seed=0):
    rng = np.random.default_rng(seed)
    closes = 100.0 + np.cumsum(rng.normal(0, 1, periods))
    return pd.DataFrame(
        {"close": closes},
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


def test_update_params_resets_state():
    factor = RollingMeanIncremental(window=10)
    factor.rewarmup(_data(periods=15))  # warmed
    pre = factor._state.bar_count
    assert pre > 0

    factor.update_params(window=20)
    assert factor._state.bar_count == 0  # state was reset
    assert factor.window == 20


def test_update_params_unknown_key_raises():
    factor = RollingMeanIncremental(window=10)
    with pytest.raises(ValueError, match="not_a_param"):
        factor.update_params(not_a_param=42)


def test_state_after_update_params_matches_fresh_instance():
    factor = RollingMeanIncremental(window=10)
    factor.rewarmup(_data())
    factor.update_params(window=15)
    fresh = RollingMeanIncremental(window=15)
    # Both should have empty initial state.
    assert factor._state.bar_count == fresh.initial_state().bar_count == 0


def test_update_immediately_after_update_params_returns_nan():
    # Behavior check (NOT prose-sentinel): pre-warm, mutate window,
    # next update() MUST return NaN — proving the warmup actually
    # reset rather than carrying over stale state.
    factor = RollingMeanIncremental(window=5)
    factor.rewarmup(_data(periods=20))  # well past warmup

    # Sanity: next update would emit non-NaN.
    sample_bar = pd.Series({"close": 100.0})
    val_before, _ = factor.update(sample_bar, factor._state)
    assert not np.isnan(val_before)

    factor.update_params(window=5)  # same window, but state cleared
    val_after, _ = factor.update(sample_bar, factor._state)
    assert np.isnan(val_after), (
        "post-update_params first update must return NaN — warmup reset"
    )


def test_rewarmup_restores_state_to_match_run_all():
    # rewarmup(history) followed by a fresh update should give the
    # same result as run_all on (history + 1 fresh bar).
    factor = RollingMeanIncremental(window=10)
    history = _data(periods=20)
    fresh_bar = pd.Series({"close": 999.0})

    factor.rewarmup(history)
    val_via_rewarmup, _ = factor.update(fresh_bar, factor._state)

    combined = pd.concat([
        history,
        fresh_bar.to_frame().T,
    ], ignore_index=False)
    out = factor.run_all(combined)
    val_via_run_all = out.iloc[-1]

    assert val_via_rewarmup == pytest.approx(val_via_run_all, rel=1e-12)


def test_rewarmup_with_history_shorter_than_window_emits_nan():
    # Real startup footgun: rewarmup with insufficient history.
    # Next update() MUST return NaN because warmup is not yet met.
    factor = RollingMeanIncremental(window=10)
    factor.rewarmup(_data(periods=5))  # 5 < window=10

    sample_bar = pd.Series({"close": 100.0})
    val, _ = factor.update(sample_bar, factor._state)
    assert np.isnan(val), (
        "rewarmup with insufficient history must not lie about state "
        "being warm"
    )
