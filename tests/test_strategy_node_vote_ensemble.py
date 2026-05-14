"""v2.5 Commit G — VoteEnsemble 3-mode rewrite tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.signals import SignalSpec
from aiphaforge.strategies import (
    _FALLBACK_WARNED,
    BaseStrategy,
    MACrossover,
    RSIMeanReversion,
    VoteEnsemble,
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
    """Returns dict for everything — wrong shape on single-asset input."""

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


class TestVoteEnsembleV2_5:
    def test_legacy_mode_matches_baseline_snapshot(self):
        from tests.test_strategy_node_legacy_snapshots import (
            _EXPECTED_VE,
            _expected_from_transitions,
        )
        from tests.test_strategy_node_legacy_snapshots import (
            _make_seeded_data as _snap,
        )
        data = _snap()
        comp = VoteEnsemble(
            children=[
                MACrossover(5, 20),
                MACrossover(10, 30),
                RSIMeanReversion(14),
            ],
            mode="legacy_compute",
        )
        expected = _expected_from_transitions(data, _EXPECTED_VE)
        actual = comp.generate_signals(data)
        pd.testing.assert_series_equal(
            actual, expected, check_exact=True, check_names=False,
        )

    def test_generate_signals_with_modern_child(self):
        # Need a strict majority: |vote_sum| > n_valid/2. With 3 children,
        # 2 longs vs 1 short gives |vote_sum|=1, which is NOT > 1.5 → no
        # majority. Use 3 longs to force a clear majority emit at bar 0.
        comp = VoteEnsemble(
            children=[
                _ConstantModern(value=1.0),
                _ConstantModern(value=1.0),
                _ConstantModern(value=1.0),
            ],
            mode="generate_signals",
        )
        actual = comp.generate_signals(_data())
        # transitions_only: NaN→1 transition emits 1.0 at bar 0 only.
        assert (actual.dropna() == 1.0).any()

    def test_eager_shape_validation_rejects_mixed_shapes(self):
        # Single-asset data, but one child returns dict (wrong shape).
        comp = VoteEnsemble(
            children=[_DictOnlyChild(), _ConstantModern(value=1.0)],
            mode="generate_signals",
        )
        with pytest.raises(TypeError, match="single-asset"):
            comp.generate_signals(_data())

    def test_rejects_non_direction_spec_child(self):
        comp = VoteEnsemble(
            children=[MACrossover(5, 20), _NonDirectionStub()],
        )
        with pytest.raises(ValueError, match="direction-kind"):
            comp.generate_signals(_data())
