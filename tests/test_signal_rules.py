"""v2.3 Commit D — signal_rules tests.

Pinned semantics:
- ThresholdScoreRule default neutral_action="hold" (NaN).
- CrossSectionalQuantileRule default neutral_action="flat" (0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.signal_rules import (
    CrossSectionalQuantileRule,
    ThresholdScoreRule,
)

# ---------------------------------------------------------------------------
# ThresholdScoreRule
# ---------------------------------------------------------------------------


class TestThresholdScoreRule:
    def test_outputs_nan_hold_by_default(self):
        idx = pd.date_range("2024-01-01", periods=4)
        scores = pd.Series([0.85, 0.50, 0.10, 0.50], index=idx)
        rule = ThresholdScoreRule(long_threshold=0.7, short_threshold=0.3)
        out = rule.transform(scores)
        # transitions_only is on by default — only state changes emit.
        # Sequence: bar0 long, bar1 hold (stays long via ffill of NaN),
        #           bar2 short, bar3 hold.
        assert out.iloc[0] == 1.0
        assert pd.isna(out.iloc[1])
        assert out.iloc[2] == -1.0
        assert pd.isna(out.iloc[3])

    def test_flat_mode_outputs_zero(self):
        idx = pd.date_range("2024-01-01", periods=4)
        scores = pd.Series([0.85, 0.50, 0.10, 0.50], index=idx)
        rule = ThresholdScoreRule(
            long_threshold=0.7, short_threshold=0.3,
            neutral_action="flat",
        )
        out = rule.transform(scores)
        # bar0: long(1). bar1: neutral(0)=transition→0. bar2: short(-1).
        # bar3: neutral(0) again — transition from short→flat→0.
        assert out.iloc[0] == 1.0
        assert out.iloc[1] == 0.0
        assert out.iloc[2] == -1.0
        assert out.iloc[3] == 0.0

    def test_long_only_mode_when_short_threshold_none(self):
        idx = pd.date_range("2024-01-01", periods=3)
        scores = pd.Series([0.85, 0.10, 0.85], index=idx)
        rule = ThresholdScoreRule(long_threshold=0.7, short_threshold=None)
        out = rule.transform(scores)
        # bar0: long(1). bar1: hold (NaN-of-prior=1, stays). bar2: long
        # again — but state didn't change, so transitions_only emits NaN.
        # Wait: with neutral_action="hold" default, bar1 emits NaN
        # (no instruction), ffill keeps state at 1. bar2 explicit
        # 1.0 also stays at 1 → transitions_only emits NaN.
        assert out.iloc[0] == 1.0
        assert pd.isna(out.iloc[1])
        assert pd.isna(out.iloc[2])

    def test_dataframe_input_applied_per_column(self):
        idx = pd.date_range("2024-01-01", periods=3)
        scores = pd.DataFrame(
            {"A": [0.85, 0.50, 0.10], "B": [0.10, 0.50, 0.85]},
            index=idx,
        )
        rule = ThresholdScoreRule(
            long_threshold=0.7, short_threshold=0.3,
            neutral_action="flat",
        )
        out = rule.transform(scores)
        # Column A: long, flat, short.
        # Column B: short, flat, long.
        assert out["A"].iloc[0] == 1.0 and out["A"].iloc[2] == -1.0
        assert out["B"].iloc[0] == -1.0 and out["B"].iloc[2] == 1.0

    def test_invalid_input_type_raises(self):
        rule = ThresholdScoreRule(long_threshold=0.7, short_threshold=0.3)
        with pytest.raises(TypeError, match="must be pd.Series or pd.DataFrame"):
            rule.transform([0.5, 0.6, 0.7])


# ---------------------------------------------------------------------------
# CrossSectionalQuantileRule
# ---------------------------------------------------------------------------


class TestCrossSectionalQuantileRule:
    def test_uses_rowwise_quantiles_with_flat_default(self):
        # With long_quantile=0.8, short_quantile=0.2 on 5 names,
        # only the strict-top and strict-bottom names trigger.
        # Default neutral_action="flat" means the middle 3 names
        # get 0 (closed), not NaN (held).
        idx = pd.date_range("2024-01-01", periods=2)
        scores = pd.DataFrame(
            {"A": [10, 50], "B": [20, 40], "C": [30, 30],
             "D": [40, 20], "E": [50, 10]},
            index=idx,
        )
        rule = CrossSectionalQuantileRule(
            long_quantile=0.8, short_quantile=0.2,
        )
        out = rule.transform(scores)
        # Bar 0: E (50) is top, A (10) is bottom. With strict gt/lt
        # the 0.8-quantile is 42, scores >42 → only E (50). 0.2-quantile
        # is 18, scores <18 → only A (10). Middle = 0.
        assert out.loc[idx[0], "E"] == 1.0
        assert out.loc[idx[0], "A"] == -1.0
        for sym in ["B", "C", "D"]:
            assert out.loc[idx[0], sym] == 0.0
        # Bar 1: roles reverse. A (50) is top, E (10) is bottom.
        assert out.loc[idx[1], "A"] == 1.0
        assert out.loc[idx[1], "E"] == -1.0

    def test_hold_mode_emits_nan_for_middle_quantiles(self):
        # Override default to verify the opt-in "hold" behavior
        # works (Alphalens-divergent but valid for sparse rebalances).
        idx = pd.date_range("2024-01-01", periods=1)
        scores = pd.DataFrame(
            {"A": [10], "B": [20], "C": [30], "D": [40], "E": [50]},
            index=idx,
        )
        rule = CrossSectionalQuantileRule(
            long_quantile=0.8, short_quantile=0.2,
            neutral_action="hold",
            transition_only=False,  # see middle as NaN, not transitioned
        )
        out = rule.transform(scores)
        for sym in ["B", "C", "D"]:
            assert pd.isna(out.loc[idx[0], sym])

    def test_rejects_series_input(self):
        scores = pd.Series([10, 20, 30, 40, 50])
        rule = CrossSectionalQuantileRule()
        with pytest.raises(TypeError, match="wide pd.DataFrame"):
            rule.transform(scores)

    def test_long_only_when_short_quantile_none(self):
        idx = pd.date_range("2024-01-01", periods=1)
        scores = pd.DataFrame(
            {"A": [10], "B": [20], "C": [30], "D": [40], "E": [50]},
            index=idx,
        )
        rule = CrossSectionalQuantileRule(
            long_quantile=0.8, short_quantile=None,
            transition_only=False,
        )
        out = rule.transform(scores)
        assert out.loc[idx[0], "E"] == 1.0
        # Without short threshold, middle + bottom all flat (default).
        for sym in ["A", "B", "C", "D"]:
            assert out.loc[idx[0], sym] == 0.0

    def test_nan_input_propagates_as_nan(self):
        # A name with NaN score for a given bar (e.g. listing gap)
        # must NOT be silently flattened to 0 by the neutral_action;
        # the missing data should remain missing.
        idx = pd.date_range("2024-01-01", periods=1)
        scores = pd.DataFrame(
            {"A": [10], "B": [np.nan], "C": [30], "D": [40], "E": [50]},
            index=idx,
        )
        rule = CrossSectionalQuantileRule(
            long_quantile=0.8, short_quantile=0.2,
            transition_only=False,
        )
        out = rule.transform(scores)
        assert pd.isna(out.loc[idx[0], "B"]), (
            "NaN input must propagate to NaN output — neutral_action "
            "should not silently flatten missing observations"
        )
