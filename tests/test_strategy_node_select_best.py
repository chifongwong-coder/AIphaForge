"""v2.5 Commit E — SelectBest 3-mode rewrite tests."""
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
    SelectBest,
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


class _NonDirectionStub(BaseStrategy):
    name = "non_direction_stub"
    spec = SignalSpec(kind="target_weight")

    def _compute(self, df):
        return pd.Series(0.5, index=df.index)


class TestSelectBestV2_5:
    def test_legacy_mode_matches_baseline_snapshot(self):
        from tests.test_strategy_node_legacy_snapshots import (
            _EXPECTED_SB,
            _expected_from_transitions,
        )
        from tests.test_strategy_node_legacy_snapshots import (
            _make_seeded_data as _snap,
        )
        data = _snap()
        comp = SelectBest(
            children=[MACrossover(5, 20), RSIMeanReversion(14)],
            mode="legacy_compute",
        )
        expected = _expected_from_transitions(data, _EXPECTED_SB)
        actual = comp.generate_signals(data)
        pd.testing.assert_series_equal(
            actual, expected, check_exact=True, check_names=False,
        )

    def test_generate_signals_with_modern_child(self):
        comp = SelectBest(
            children=[_ConstantModern(value=1.0), MACrossover(5, 20)],
            mode="generate_signals",
        )
        actual = comp.generate_signals(_data())
        # Modern child returns 1.0 everywhere; abs=1.0 wins on most bars.
        non_nan = actual.dropna()
        assert len(non_nan) >= 1

    def test_multi_asset_dict_input(self):
        data = {"A": _data(seed=1), "B": _data(seed=2)}
        comp = SelectBest(
            children=[_ConstantModern(value=1.0), _ConstantModern(value=-2.0)],
            mode="generate_signals",
        )
        out = comp.generate_signals(data)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"A", "B"}
        # Child 1 has |signal|=2 > |1|, so output picks child 1's -1.0 sign
        # but transitions_only sparsifies. Just check shape + index.
        for sym, sig in out.items():
            assert sig.index.equals(data[sym].index)

    def test_rejects_non_direction_spec_child(self):
        comp = SelectBest(
            children=[MACrossover(5, 20), _NonDirectionStub()],
        )
        with pytest.raises(ValueError, match="direction-kind"):
            comp.generate_signals(_data())
