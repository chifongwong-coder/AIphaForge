"""v2.5 Commit D — WeightedBlend 3-mode rewrite tests."""
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
    WeightedBlend,
)


@pytest.fixture(autouse=True)
def _clear_fallback_state():
    _FALLBACK_WARNED.clear()
    yield
    _FALLBACK_WARNED.clear()


def _make_seeded_data(periods: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": [1e6] * periods,
        },
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


class _ConstantModernChild(BaseStrategy):
    """Returns a fixed long signal everywhere via generate_signals."""

    name = "constant_modern_child"

    def __init__(self, value: float = 1.0):
        self.value = value

    def generate_signals(self, data):
        if isinstance(data, dict):
            return {
                sym: pd.Series(self.value, index=df.index, dtype=float)
                for sym, df in data.items()
            }
        return pd.Series(self.value, index=data.index, dtype=float)

    def _compute(self, df):
        return pd.Series(self.value, index=df.index, dtype=float)


class _NonDirectionStubChild(BaseStrategy):
    """Test fixture: child carrying a non-direction SignalSpec.

    Per plan v2.5 D.4 — the validator must reject this child at
    composite generate_signals entry, BEFORE any child compute runs.
    """

    name = "non_direction_stub"
    spec = SignalSpec(kind="target_weight")

    def _compute(self, df):
        return pd.Series(0.5, index=df.index)


class TestWeightedBlendV2_5:
    def test_legacy_mode_matches_baseline_snapshot(self):
        # Re-run Commit B's WeightedBlend snapshot via mode="legacy_compute".
        from tests.test_strategy_node_legacy_snapshots import (
            _EXPECTED_WB,
            _expected_from_transitions,
        )
        from tests.test_strategy_node_legacy_snapshots import (
            _make_seeded_data as _snap_data,
        )
        data = _snap_data()
        composite = WeightedBlend(
            children=[
                MACrossover(short=5, long=20),
                RSIMeanReversion(period=14),
            ],
            weights=[0.6, 0.4],
            mode="legacy_compute",
        )
        expected = _expected_from_transitions(data, _EXPECTED_WB)
        actual = composite.generate_signals(data)
        pd.testing.assert_series_equal(
            actual, expected, check_exact=True, check_names=False,
        )

    def test_generate_signals_with_modern_child(self):
        data = _make_seeded_data()
        composite = WeightedBlend(
            children=[
                _ConstantModernChild(value=1.0),
                MACrossover(short=5, long=20),
            ],
            weights=[0.5, 0.5],
            mode="generate_signals",
        )
        # Modern child returns 1.0 everywhere; MACrossover returns NaN
        # most bars. Per-bar weight renorm: where MA is NaN, only the
        # modern child contributes (weight renorms to 1.0). After
        # rounding to signal_precision=2 and transitions_only, the
        # output should emit 1.0 once near the start (NaN→1 transition)
        # and remain NaN thereafter unless MA flips the blended value.
        actual = composite.generate_signals(data)
        non_nan = actual.dropna()
        assert len(non_nan) >= 1, "modern child should emit at least 1 transition"
        # Default mode would normally use auto; explicit generate_signals
        # ensures we exercise the modern path even when MA might fail.
        assert composite.mode == "generate_signals"

    def test_multi_asset_dict_input(self):
        data = {
            "A": _make_seeded_data(seed=1),
            "B": _make_seeded_data(seed=2),
        }
        composite = WeightedBlend(
            children=[
                _ConstantModernChild(value=1.0),
                _ConstantModernChild(value=-1.0),
            ],
            weights=[0.5, 0.5],
            mode="generate_signals",
        )
        out = composite.generate_signals(data)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"A", "B"}
        for sym, sig in out.items():
            assert isinstance(sig, pd.Series)
            assert sig.index.equals(data[sym].index)

    def test_mismatched_symbol_keys_raises(self):
        data = {
            "A": _make_seeded_data(seed=1),
            "B": _make_seeded_data(seed=2),
        }

        class _PartialUniverseChild(BaseStrategy):
            name = "partial_universe"

            def generate_signals(self, data):
                # Only returns symbol A, missing B.
                return {"A": pd.Series(1.0, index=data["A"].index)}

            def _compute(self, df):
                return pd.Series(1.0, index=df.index)

        composite = WeightedBlend(
            children=[
                _ConstantModernChild(value=1.0),
                _PartialUniverseChild(),
            ],
            weights=[0.5, 0.5],
            mode="generate_signals",
        )
        with pytest.raises(ValueError, match="mismatched symbols"):
            composite.generate_signals(data)

    def test_rejects_non_direction_spec_child(self):
        composite = WeightedBlend(
            children=[
                MACrossover(short=5, long=20),
                _NonDirectionStubChild(),
            ],
            weights=[0.5, 0.5],
        )
        df = _make_seeded_data()
        with pytest.raises(ValueError, match="direction-kind"):
            composite.generate_signals(df)
