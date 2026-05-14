"""v2.5 Commit C — mode= parameter is wired through (no behavioral change).

Per plan v2 #6: Commit C is parameter-only. The composite ``__init__``
methods accept and store ``mode`` but no dispatch logic uses it yet.
Snapshot tests from Commit B continue to pass under default ``auto``
mode because ``_compute`` paths run unchanged.
"""
from __future__ import annotations

import pytest

from aiphaforge.strategies import (
    ConditionalSwitch,
    MACrossover,
    PriorityCascade,
    RSIMeanReversion,
    SelectBest,
    StrategyNode,
    VoteEnsemble,
    WeightedBlend,
)


def _children():
    return [
        MACrossover(short=5, long=20),
        RSIMeanReversion(period=14),
    ]


class TestModeParam:
    def test_mode_default_is_auto(self):
        assert WeightedBlend(_children()).mode == "auto"

    def test_mode_invalid_value_raises(self):
        with pytest.raises(ValueError, match="mode='invalid'"):
            WeightedBlend(_children(), mode="invalid")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "composite_factory",
        [
            lambda mode: WeightedBlend(_children(), mode=mode),
            lambda mode: SelectBest(_children(), mode=mode),
            lambda mode: PriorityCascade(_children(), mode=mode),
            lambda mode: VoteEnsemble(_children(), mode=mode),
            lambda mode: ConditionalSwitch(
                _children(),
                lambda df: __import__("pandas").Series(0, index=df.index),
                mode=mode,
            ),
        ],
        ids=["WeightedBlend", "SelectBest", "PriorityCascade",
             "VoteEnsemble", "ConditionalSwitch"],
    )
    def test_mode_propagates_to_subclasses(self, composite_factory):
        comp = composite_factory("legacy_compute")
        assert isinstance(comp, StrategyNode)
        assert comp.mode == "legacy_compute"

    def test_mode_present_in_params_view(self):
        comp = WeightedBlend(_children(), mode="generate_signals")
        assert comp.params.get("mode") == "generate_signals"

    def test_baseline_snapshots_unchanged_after_mode_addition(self):
        # The Commit B snapshot tests still pass under default auto mode
        # because _compute paths run unchanged in Commit C. Re-import
        # and run them programmatically as a sentinel.
        from tests.test_strategy_node_legacy_snapshots import (
            TestLegacySnapshotsV2_4,
        )
        instance = TestLegacySnapshotsV2_4()
        instance.test_weighted_blend_snapshot_v2_4()
        instance.test_select_best_snapshot_v2_4()
        instance.test_priority_cascade_snapshot_v2_4()
        instance.test_vote_ensemble_snapshot_v2_4()
        instance.test_conditional_switch_snapshot_v2_4()

    def test_subclass_positional_args_preserved(self):
        # Sentinel for the "subclass __init__ signature preservation"
        # contract from plan v2.1: positional weights / condition_fn
        # MUST keep working after `mode` is appended as keyword-only.
        # No shipped test uses positional args, so without this
        # sentinel the Commit C signature change could silently break
        # downstream user code.
        kids = _children()
        wb = WeightedBlend(kids, [0.6, 0.4])  # positional weights
        assert wb.weights == [0.6, 0.4]
        wb2 = WeightedBlend(kids, [0.6, 0.4], 3)  # positional precision
        assert wb2.signal_precision == 3

        import pandas as pd

        def regime(df):
            return pd.Series(0, index=df.index)

        cs = ConditionalSwitch(kids, regime)  # positional condition_fn
        assert cs.condition_fn is regime
