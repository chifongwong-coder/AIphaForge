"""v2.5 Commit F — PriorityCascade 3-mode rewrite tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.signals import SignalSpec
from aiphaforge.strategies import (
    _FALLBACK_WARNED,
    BaseStrategy,
    MACrossover,
    PriorityCascade,
    RSIMeanReversion,
)


@pytest.fixture(autouse=True)
def _clear_fallback_state():
    _FALLBACK_WARNED.clear()
    yield
    _FALLBACK_WARNED.clear()


def _data(periods=60, seed=42):
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": [1e6] * periods,
        },
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


class _ConstantModern(BaseStrategy):
    name = "constant_modern"

    def __init__(self, value=1.0):
        self.value = value

    def generate_signals(self, data):
        if isinstance(data, dict):
            return {sym: pd.Series(self.value, index=df.index, dtype=float)
                    for sym, df in data.items()}
        return pd.Series(self.value, index=data.index, dtype=float)

    def _compute(self, df):
        return pd.Series(self.value, index=df.index, dtype=float)


class _SeriesShapeChild(BaseStrategy):
    """Always returns Series even for dict input — wrong shape for multi."""

    name = "series_shape_child"

    def generate_signals(self, data):
        idx = data.index if not isinstance(data, dict) else (
            list(data.values())[0].index
        )
        return pd.Series(1.0, index=idx)

    def _compute(self, df):
        return pd.Series(1.0, index=df.index)


class _NonDirectionStub(BaseStrategy):
    name = "non_direction_stub"
    spec = SignalSpec(kind="target_weight")

    def _compute(self, df):
        return pd.Series(0.5, index=df.index)


class TestPriorityCascadeV2_5:
    def test_legacy_mode_matches_baseline_snapshot(self):
        from tests.test_strategy_node_legacy_snapshots import (
            _EXPECTED_PC,
            _expected_from_transitions,
        )
        from tests.test_strategy_node_legacy_snapshots import (
            _make_seeded_data as _snap,
        )
        data = _snap()
        comp = PriorityCascade(
            children=[MACrossover(5, 20), RSIMeanReversion(14)],
            mode="legacy_compute",
        )
        expected = _expected_from_transitions(data, _EXPECTED_PC)
        actual = comp.generate_signals(data)
        pd.testing.assert_series_equal(
            actual, expected, check_exact=True, check_names=False,
        )

    def test_generate_signals_with_modern_child(self):
        comp = PriorityCascade(
            children=[MACrossover(5, 20), _ConstantModern(value=1.0)],
            mode="generate_signals",
        )
        actual = comp.generate_signals(_data())
        # Primary (MA) signals NaN most bars; fallback (modern 1.0) covers.
        non_nan = actual.dropna()
        assert len(non_nan) >= 1, "expected at least 1 transition"

    def test_eager_shape_validation_rejects_mixed_shapes(self):
        # PriorityCascade entered with dict input; one child returns Series
        # (wrong shape). Eager validation must reject BEFORE merge loop
        # so the user sees the offending child name in the error.
        data = {"A": _data(seed=1)}
        comp = PriorityCascade(
            children=[_SeriesShapeChild(), _ConstantModern(value=1.0)],
            mode="generate_signals",
        )
        with pytest.raises(TypeError, match="multi-asset"):
            comp.generate_signals(data)

    def test_rejects_non_direction_spec_child(self):
        comp = PriorityCascade(
            children=[MACrossover(5, 20), _NonDirectionStub()],
        )
        with pytest.raises(ValueError, match="direction-kind"):
            comp.generate_signals(_data())
