"""v2.5 Commit B — snapshot baseline for the 5 StrategyNode composites.

Pins v2.4 output as ground truth so subsequent v2.5 commits (C-H)
can verify ``mode="legacy_compute"`` produces bit-equal results.

WARNING (applies to every test): do NOT regenerate expected
values from production code. The point is to verify v2.5
preserves v2.4 behavior. If a test fails, suspect the v2.5
rewrite, not the snapshot.

Snapshots captured against the codebase at the v2.5 branch base
(after v2.4 ship, before any v2.5 rewrite of composites).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiphaforge.strategies import (
    ConditionalSwitch,
    MACrossover,
    PriorityCascade,
    RSIMeanReversion,
    SelectBest,
    VoteEnsemble,
    WeightedBlend,
)


def _make_seeded_data(periods: int = 60, seed: int = 42) -> pd.DataFrame:
    """Single shared deterministic OHLCV builder.

    Uses ``np.random.default_rng`` (not legacy ``np.random.seed``)
    for reproducibility across NumPy versions.
    """
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": [1e6] * periods,
        },
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


def _expected_from_transitions(
    data: pd.DataFrame, transitions: dict[int, float],
) -> pd.Series:
    """Build the expected Series from a sparse {bar_idx: value} dict.

    All other bars are NaN (the v2.4 ``_transitions_only`` filter
    output is sparse-by-construction).
    """
    out = pd.Series(np.nan, index=data.index, dtype=float)
    for bar, val in transitions.items():
        out.iloc[bar] = val
    return out


# Inline-literal expected snapshots captured at v2.5 branch base.
# Format: {bar_index: signal_value}; all other bars = NaN.

_EXPECTED_WB = {19: 1.0, 50: -1.0}
_EXPECTED_SB = {19: 1.0, 50: -1.0}
_EXPECTED_PC = {19: 1.0, 50: -1.0}
_EXPECTED_VE = {19: 1.0, 50: -1.0}
_EXPECTED_CS: dict[int, float] = {}  # all NaN under this regime fixture


class TestLegacySnapshotsV2_4:
    def test_weighted_blend_snapshot_v2_4(self):
        data = _make_seeded_data()
        composite = WeightedBlend(
            children=[
                MACrossover(short=5, long=20),
                RSIMeanReversion(period=14),
            ],
            weights=[0.6, 0.4],
        )
        expected = _expected_from_transitions(data, _EXPECTED_WB)
        actual = composite._compute(data)
        pd.testing.assert_series_equal(
            actual, expected, check_exact=True, check_names=False,
        )

    def test_select_best_snapshot_v2_4(self):
        data = _make_seeded_data()
        composite = SelectBest(
            children=[
                MACrossover(short=5, long=20),
                RSIMeanReversion(period=14),
            ],
        )
        expected = _expected_from_transitions(data, _EXPECTED_SB)
        actual = composite._compute(data)
        pd.testing.assert_series_equal(
            actual, expected, check_exact=True, check_names=False,
        )

    def test_priority_cascade_snapshot_v2_4(self):
        data = _make_seeded_data()
        composite = PriorityCascade(
            children=[
                MACrossover(short=5, long=20),
                RSIMeanReversion(period=14),
            ],
        )
        expected = _expected_from_transitions(data, _EXPECTED_PC)
        actual = composite._compute(data)
        pd.testing.assert_series_equal(
            actual, expected, check_exact=True, check_names=False,
        )

    def test_vote_ensemble_snapshot_v2_4(self):
        data = _make_seeded_data()
        composite = VoteEnsemble(
            children=[
                MACrossover(short=5, long=20),
                MACrossover(short=10, long=30),
                RSIMeanReversion(period=14),
            ],
        )
        expected = _expected_from_transitions(data, _EXPECTED_VE)
        actual = composite._compute(data)
        pd.testing.assert_series_equal(
            actual, expected, check_exact=True, check_names=False,
        )

    def test_conditional_switch_snapshot_v2_4(self):
        data = _make_seeded_data()

        def _regime_alternating_5(df: pd.DataFrame) -> pd.Series:
            arr = np.array(
                [(i // 5) % 2 for i in range(len(df))], dtype=int,
            )
            return pd.Series(arr, index=df.index)

        composite = ConditionalSwitch(
            children=[
                MACrossover(short=5, long=20),
                RSIMeanReversion(period=14),
            ],
            condition_fn=_regime_alternating_5,
        )
        expected = _expected_from_transitions(data, _EXPECTED_CS)
        actual = composite._compute(data)
        pd.testing.assert_series_equal(
            actual, expected, check_exact=True, check_names=False,
        )
