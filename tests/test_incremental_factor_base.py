"""v2.6 Commit A — IncrementalFactor ABC + FactorState contract tests."""
from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pandas as pd
import pytest

from aiphaforge.incremental_factors import (
    FactorState,
    IncrementalFactor,
)

# ---------------------------------------------------------------------------
# FactorState dataclass + pickle contract
# ---------------------------------------------------------------------------


def test_factor_state_is_dataclass():
    assert dataclasses.is_dataclass(FactorState)
    state = FactorState()
    assert state.bar_count == 0


def test_factor_state_is_picklable():
    state = FactorState(bar_count=42)
    blob = pickle.dumps(state)
    restored = pickle.loads(blob)
    assert restored.bar_count == 42


def test_factor_state_pickle_version_class_attr():
    # __pickle_version__ is a sentinel for cross-version breakage
    # detection. Bumping it on a breaking change lets callers
    # detect stale state instead of silently corrupting.
    assert FactorState.__pickle_version__ == 1


# ---------------------------------------------------------------------------
# ABC contract
# ---------------------------------------------------------------------------


def test_incremental_factor_is_abc():
    with pytest.raises(TypeError, match="abstract"):
        IncrementalFactor()  # type: ignore[abstract]


def test_concrete_subclass_must_implement_initial_state():
    class _MissingInitialState(IncrementalFactor):
        def update(self, bar_row, state):
            return 0.0, state
    with pytest.raises(TypeError, match="abstract"):
        _MissingInitialState()  # type: ignore[abstract]


def test_concrete_subclass_must_implement_update():
    class _MissingUpdate(IncrementalFactor):
        def initial_state(self):
            return FactorState()
    with pytest.raises(TypeError, match="abstract"):
        _MissingUpdate()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# bar_count semantics: incremented AFTER update, warmup uses strict <
# ---------------------------------------------------------------------------


class _StubWarmupFactor(IncrementalFactor):
    """Returns NaN while bar_count < window; non-NaN after."""

    name = "stub_warmup"

    def __init__(self, window: int = 3):
        self.window = window

    def initial_state(self) -> FactorState:
        return FactorState()

    def update(self, bar_row, state):
        if state.bar_count < self.window:
            value = float("nan")
        else:
            value = float(bar_row["close"])
        new_state = FactorState(bar_count=state.bar_count + 1)
        return value, new_state


def test_bar_count_incremented_after_update():
    # Pins both that bar_count is incremented AFTER update returns
    # AND that the warmup check uses strict `<`. Returning NaN
    # while warmup is unmet matches the §R18 convention (NOT 0).
    factor = _StubWarmupFactor(window=3)
    state = factor.initial_state()
    assert state.bar_count == 0

    # First bar: bar_count == 0 < window=3 → NaN, new_state.bar_count == 1
    bar = pd.Series({"close": 100.0})
    value, state = factor.update(bar, state)
    assert np.isnan(value), "warmup not met → factor must emit NaN"
    assert state.bar_count == 1

    # Bars 2, 3: still warming
    for expected_count in (2, 3):
        value, state = factor.update(bar, state)
        assert np.isnan(value)
        assert state.bar_count == expected_count

    # Bar 4: bar_count == 3 (not < 3) → emit non-NaN
    value, state = factor.update(bar, state)
    assert value == 100.0
    assert state.bar_count == 4
