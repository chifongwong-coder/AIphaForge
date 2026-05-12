"""v2.2 M3 tests — ContinuationProbe + 3 templates."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aiphaforge.probes._continuation import (
    CONTINUATION_TEMPLATES,
    ContinuationProbe,
    NextCloseContinuation,
    NextRangeContinuation,
    NextReturnContinuation,
)


def _make_bars(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": [1e6] * n,
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )


# ---------- ContinuationProbe constructor ----------


class TestContinuationProbeConstructor:
    def test_rejects_missing_context_bars(self):
        with pytest.raises(TypeError):
            ContinuationProbe(  # type: ignore[call-arg]
                symbol="AAPL", forward_horizon=1,
            )

    def test_rejects_missing_forward_horizon(self):
        with pytest.raises(TypeError):
            ContinuationProbe(  # type: ignore[call-arg]
                symbol="AAPL", context_bars=20,
            )

    def test_rejects_context_bars_lt_2(self):
        with pytest.raises(ValueError):
            ContinuationProbe(
                symbol="AAPL", context_bars=1, forward_horizon=1,
            )

    def test_rejects_forward_horizon_lt_1(self):
        with pytest.raises(ValueError):
            ContinuationProbe(
                symbol="AAPL", context_bars=20, forward_horizon=0,
            )

    def test_accepts_template_classes_and_instances(self):
        p = ContinuationProbe(
            symbol="AAPL", context_bars=20, forward_horizon=1,
            templates=[NextCloseContinuation, NextReturnContinuation()],
        )
        assert all(
            isinstance(t, NextCloseContinuation) or
            isinstance(t, NextReturnContinuation)
            for t in p.templates
        )


# ---------- ContinuationProbe.build ----------


class TestContinuationProbeBuild:
    def test_emits_one_question_per_template_per_anchor_per_horizon(self):
        data = _make_bars(n=60)
        probe = ContinuationProbe(
            symbol="AAPL", context_bars=10, forward_horizon=3,
            templates=CONTINUATION_TEMPLATES,
        )
        # Pick anchors well away from edges
        anchors = list(data.index[20:25])
        qs = probe.build(data, anchors)
        # 5 anchors * 3 templates * 3 horizons = 45
        assert len(list(qs)) == 5 * 3 * 3

    def test_skips_anchors_too_close_to_data_edges(self):
        data = _make_bars(n=30)
        probe = ContinuationProbe(
            symbol="AAPL", context_bars=10, forward_horizon=1,
            templates=[NextCloseContinuation],
        )
        # Anchor at index 5 → context_bars=10 needs index 5-9 = -4
        # → skipped silently.
        anchors = [data.index[5]]
        qs = probe.build(data, anchors)
        assert len(list(qs)) == 0

    def test_question_spec_carries_context_window_in_metadata(self):
        data = _make_bars(n=60)
        probe = ContinuationProbe(
            symbol="AAPL", context_bars=15, forward_horizon=1,
            templates=[NextCloseContinuation],
        )
        anchors = [data.index[30]]
        qs = list(probe.build(data, anchors))
        spec = qs[0]
        ctx = spec.metadata["context_window"]
        assert isinstance(ctx, list)
        assert len(ctx) == 15
        # All context timestamps strictly <= anchor timestamp.
        for row in ctx:
            assert pd.Timestamp(row["index"]) <= spec.timestamp

    def test_question_spec_no_lookahead_in_context(self):
        # v2.2.1 #8: was `< spec.timestamp + Timedelta(days=1)` which
        # let the next business day leak through (within 24h on a
        # bdate index). Now `<= spec.timestamp` per the docstring of
        # the sibling test test_question_spec_carries_context_window_in_metadata.
        data = _make_bars(n=60)
        probe = ContinuationProbe(
            symbol="AAPL", context_bars=15, forward_horizon=2,
            templates=[NextCloseContinuation],
        )
        anchors = [data.index[30]]
        qs = list(probe.build(data, anchors))
        for spec in qs:
            ctx = spec.metadata["context_window"]
            for row in ctx:
                assert pd.Timestamp(row["index"]) <= spec.timestamp

    def test_continuation_context_unchanged_when_future_bar_mutated(self):
        # v2.2.1 #8 companion: explicitly mutate a bar AFTER the
        # anchor and assert the context window is unaffected.
        data = _make_bars(n=60)
        anchor = data.index[30]
        probe = ContinuationProbe(
            symbol="AAPL", context_bars=15, forward_horizon=1,
            templates=[NextCloseContinuation],
        )
        qs_a = list(probe.build(data, [anchor]))
        ctx_a = qs_a[0].metadata["context_window"]
        # Mutate bar at index 35 (well past anchor at 30).
        data_mod = data.copy()
        data_mod.iloc[35, data_mod.columns.get_loc("close")] = 99999.0
        qs_b = list(probe.build(data_mod, [anchor]))
        ctx_b = qs_b[0].metadata["context_window"]
        # Context windows should be identical.
        for ra, rb in zip(ctx_a, ctx_b):
            assert ra["close"] == rb["close"]

    def test_context_window_size_matches_context_bars_kwarg(self):
        data = _make_bars(n=60)
        probe = ContinuationProbe(
            symbol="AAPL", context_bars=25, forward_horizon=1,
            templates=[NextCloseContinuation],
        )
        anchors = [data.index[30]]
        qs = list(probe.build(data, anchors))
        ctx = qs[0].metadata["context_window"]
        assert len(ctx) == 25


# ---------- Per-template correctness ----------


class TestNextCloseContinuation:
    def test_truth_value_is_close_at_t_plus_h(self):
        data = _make_bars(n=60)
        anchor = data.index[30]
        spec = NextCloseContinuation().build(
            data, "AAPL", anchor, context_bars=10, forward_horizon=1,
        )
        target_close = float(data.iloc[31]["close"])
        assert spec.truth_value == pytest.approx(target_close)

    def test_horizon_2_targets_t_plus_2(self):
        data = _make_bars(n=60)
        anchor = data.index[30]
        spec = NextCloseContinuation().build(
            data, "AAPL", anchor, context_bars=10, forward_horizon=2,
        )
        target_close = float(data.iloc[32]["close"])
        assert spec.truth_value == pytest.approx(target_close)


class TestNextRangeContinuation:
    def test_truth_value_is_low_high_tuple(self):
        data = _make_bars(n=60)
        anchor = data.index[30]
        spec = NextRangeContinuation().build(
            data, "AAPL", anchor, context_bars=10, forward_horizon=1,
        )
        low, high = spec.truth_value
        assert low == pytest.approx(float(data.iloc[31]["low"]))
        assert high == pytest.approx(float(data.iloc[31]["high"]))


class TestNextReturnContinuation:
    def test_truth_value_is_log_return(self):
        data = _make_bars(n=60)
        anchor = data.index[30]
        spec = NextReturnContinuation().build(
            data, "AAPL", anchor, context_bars=10, forward_horizon=1,
        )
        c1 = float(data.iloc[31]["close"])
        c0 = float(data.iloc[30]["close"])
        assert spec.truth_value == pytest.approx(math.log(c1 / c0))

    def test_tolerance_is_sign_sensitive(self):
        spec = NextReturnContinuation().build(
            _make_bars(n=60), "AAPL",
            _make_bars(n=60).index[30],
            context_bars=10, forward_horizon=1,
        )
        assert spec.tolerance.sign_sensitive is True


# ---------- Default vol_scale on templates ----------


class TestContinuationDefaultVolScale:
    def test_next_close_template_has_vol_scale(self):
        spec = NextCloseContinuation().build(
            _make_bars(n=60), "AAPL", _make_bars(n=60).index[30],
            context_bars=10, forward_horizon=1,
        )
        assert spec.tolerance.vol_scale is not None

    def test_next_return_template_has_vol_scale(self):
        spec = NextReturnContinuation().build(
            _make_bars(n=60), "AAPL", _make_bars(n=60).index[30],
            context_bars=10, forward_horizon=1,
        )
        assert spec.tolerance.vol_scale is not None
