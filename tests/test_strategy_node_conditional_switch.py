"""v2.5 Commit H — ConditionalSwitch 3-mode rewrite tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.signals import SignalSpec
from aiphaforge.strategies import (
    _FALLBACK_WARNED,
    BaseStrategy,
    ConditionalSwitch,
    MACrossover,
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


class _DictOnlyChild(BaseStrategy):
    """Returns dict for single-asset input — wrong shape."""

    name = "dict_only_child"

    def generate_signals(self, data):
        return {"X": pd.Series(1.0, index=range(5))}

    def _compute(self, df):
        return pd.Series(1.0, index=df.index)


class _NonDirectionStub(BaseStrategy):
    name = "non_direction_stub"
    spec = SignalSpec(kind="target_weight")

    def _compute(self, df):
        return pd.Series(0.5, index=df.index)


def _regime_alternating_5(df):
    arr = np.array([(i // 5) % 2 for i in range(len(df))], dtype=int)
    return pd.Series(arr, index=df.index)


class TestConditionalSwitchV2_5:
    def test_legacy_mode_matches_baseline_snapshot(self):
        from tests.test_strategy_node_legacy_snapshots import (
            _EXPECTED_CS,
            _expected_from_transitions,
        )
        from tests.test_strategy_node_legacy_snapshots import (
            _make_seeded_data as _snap,
        )
        data = _snap()
        comp = ConditionalSwitch(
            children=[MACrossover(5, 20), RSIMeanReversion(14)],
            condition_fn=_regime_alternating_5,
            mode="legacy_compute",
        )
        expected = _expected_from_transitions(data, _EXPECTED_CS)
        actual = comp.generate_signals(data)
        pd.testing.assert_series_equal(
            actual, expected, check_exact=True, check_names=False,
        )

    def test_generate_signals_with_modern_child(self):
        # Modern child returns 1.0; regime fn alternates so each child
        # gets bars where it's selected.
        comp = ConditionalSwitch(
            children=[_ConstantModern(value=1.0), _ConstantModern(value=-1.0)],
            condition_fn=_regime_alternating_5,
            mode="generate_signals",
        )
        actual = comp.generate_signals(_data())
        # transitions_only emits both 1.0 and -1.0 at regime boundaries.
        non_nan = set(actual.dropna().tolist())
        assert non_nan == {1.0, -1.0}, (
            f"expected both directions to appear, got {non_nan}"
        )

    def test_eager_shape_validation_rejects_mixed_shapes(self):
        comp = ConditionalSwitch(
            children=[_DictOnlyChild(), _ConstantModern(value=1.0)],
            condition_fn=_regime_alternating_5,
            mode="generate_signals",
        )
        with pytest.raises(TypeError, match="single-asset"):
            comp.generate_signals(_data())

    def test_rejects_non_direction_spec_child(self):
        comp = ConditionalSwitch(
            children=[MACrossover(5, 20), _NonDirectionStub()],
            condition_fn=_regime_alternating_5,
        )
        with pytest.raises(ValueError, match="direction-kind"):
            comp.generate_signals(_data())
