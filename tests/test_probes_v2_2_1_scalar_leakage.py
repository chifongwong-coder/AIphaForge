"""v2.2.1 #7 — scalar_leakage_index + h0_se + z typed sentinel."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from aiphaforge.probes.anchors import build_synthetic_anchor
from aiphaforge.probes.models import AnswerRecord, AttestedAnswers
from aiphaforge.probes.orchestrator import (
    _compute_scalar_leakage_index_and_z,
    knowledge_check,
)
from aiphaforge.probes.questions import (
    DEFAULT_TEMPLATES,
    KnowledgeProbe,
)


def _provider_config():
    return {
        "temperature": 0.0, "top_p": 1.0,
        "model_id": "test", "model_version": "1",
        "max_output_tokens": 256, "tokenizer_id": "test",
        "seed": 42, "reasoning_effort": None, "stop_sequences": None,
    }


def _make_data(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.01, n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01,
            "low": closes * 0.99, "close": closes,
            "volume": [1e6] * n,
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )


def _build_perfect_attested(question_set):
    return AttestedAnswers(
        answers=tuple(
            AnswerRecord(
                question_id=q.question_id, raw_answer="x",
                parsed_answer=q.truth_value, parse_status="valid",
            )
            for q in question_set
        ),
        parsing_schema_hash="a" * 64,
        parsing_schema_description="x",
        prompt_template_hash="b" * 64,
        prompt_template_description="y",
    )


# ---------- Pure helper unit tests ----------


class TestComputeScalarLeakageIndex:
    def test_empty_pairs_returns_all_none(self):
        i, se, z = _compute_scalar_leakage_index_and_z([])
        assert i is None and se is None and z is None

    def test_perfect_agreement_returns_zero_index_none_z(self):
        # All d_i = 0 → index = 0, SE = 0 → z = None (0/0)
        pairs = [("exact", "exact"), ("near", "near"), ("miss", "miss")]
        i, se, z = _compute_scalar_leakage_index_and_z(pairs)
        assert i == 0.0
        assert se == 0.0
        assert z is None  # 0/0 indeterminate

    def test_deterministic_positive_diff_returns_positive_infinite(self):
        # All real=exact, all anchor=miss → d_i = 4-1 = 3 (constant)
        # index = 3, SE = 0 → z = "positive_infinite"
        pairs = [("exact", "miss")] * 5
        i, se, z = _compute_scalar_leakage_index_and_z(pairs)
        assert i == pytest.approx(3.0)
        assert se == 0.0
        assert z == "positive_infinite"

    def test_deterministic_negative_diff_returns_negative_infinite(self):
        pairs = [("invalid", "exact")] * 5
        i, se, z = _compute_scalar_leakage_index_and_z(pairs)
        assert i == pytest.approx(-4.0)
        assert se == 0.0
        assert z == "negative_infinite"

    def test_finite_z_for_mixed_pairs(self):
        # Mix produces nonzero SE → finite z.
        pairs = [
            ("exact", "miss"),  # +3
            ("near", "rough"),  # +1
            ("miss", "exact"),  # -3
            ("rough", "near"),  # -1
        ]
        i, se, z = _compute_scalar_leakage_index_and_z(pairs)
        assert i == pytest.approx(0.0)
        assert se > 0
        assert z == 0.0  # index is 0, but SE > 0 → finite 0.0

    def test_single_record_returns_none_se(self):
        # N=1 → SE undefined; index defined.
        pairs = [("exact", "miss")]
        i, se, z = _compute_scalar_leakage_index_and_z(pairs)
        assert i == pytest.approx(3.0)
        assert se is None
        assert z is None

    def test_paired_se_uses_paired_differences(self):
        # Hand-computed: pairs of differences [3, 1, -1, -3]
        # mean = 0, variance(ddof=1) = (9+1+1+9)/3 = 20/3 ≈ 6.667
        # SE = sqrt(20/3 / 4) = sqrt(5/3) ≈ 1.291
        pairs = [
            ("exact", "miss"),  # 4-1 = +3
            ("near", "rough"),  # 3-2 = +1
            ("rough", "near"),  # 2-3 = -1
            ("miss", "exact"),  # 1-4 = -3
        ]
        i, se, z = _compute_scalar_leakage_index_and_z(pairs)
        expected_se = math.sqrt((20.0 / 3.0) / 4.0)
        assert se == pytest.approx(expected_se)


# ---------- Integration tests via knowledge_check ----------


class TestScalarLeakageInReport:
    def _run(self, anchor: bool = True):
        data = _make_data()
        probe = KnowledgeProbe(
            symbol="AAPL", templates=DEFAULT_TEMPLATES,
        )
        anchors = list(data.index[10:18])
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        kwargs = dict(
            probe=probe, data=data, timestamps=anchors,
            answers=attested, provider_config=_provider_config(),
        )
        if anchor:
            anchor_data = build_synthetic_anchor(
                data, seed=1, method="random_walk_volmatched",
            )
            anchor_qs = probe.build(anchor_data, anchors)
            kwargs["anchor"] = anchor_data
            kwargs["anchor_answers"] = _build_perfect_attested(anchor_qs)
        return knowledge_check(**kwargs)

    def test_scalar_present_with_anchor(self):
        report = self._run(anchor=True)
        # Both perfect answers → bucket distributions identical →
        # scalar = 0 → z = None (0/0)
        if report.anchor_validity == "OK":
            assert report.scalar_leakage_index is not None
            assert report.scalar_leakage_index_h0_se is not None
            assert report.leakage_index_caveat is not None
            assert "ordinal weighting" in report.leakage_index_caveat

    def test_scalar_none_without_anchor(self):
        report = self._run(anchor=False)
        assert report.scalar_leakage_index is None
        assert report.scalar_leakage_index_h0_se is None
        assert report.scalar_leakage_index_z is None
        assert report.leakage_index_caveat is None


# ---------- Removed bootstrap params ----------


class TestRemovedBootstrapParams:
    def _setup(self):
        data = _make_data()
        probe = KnowledgeProbe(
            symbol="AAPL", templates=DEFAULT_TEMPLATES,
        )
        anchors = list(data.index[10:15])
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        return data, probe, anchors, attested

    def test_n_anchor_bootstrap_kwarg_removed(self):
        data, probe, anchors, attested = self._setup()
        with pytest.raises(TypeError, match="n_anchor_bootstrap"):
            knowledge_check(
                probe, data, anchors, attested,
                provider_config=_provider_config(),
                n_anchor_bootstrap=10,  # type: ignore[call-arg]
            )

    def test_bootstrap_seed_kwarg_removed(self):
        data, probe, anchors, attested = self._setup()
        with pytest.raises(TypeError, match="bootstrap_seed"):
            knowledge_check(
                probe, data, anchors, attested,
                provider_config=_provider_config(),
                bootstrap_seed=42,  # type: ignore[call-arg]
            )

    def test_bootstrap_unit_kwarg_removed(self):
        data, probe, anchors, attested = self._setup()
        with pytest.raises(TypeError, match="bootstrap_unit"):
            knowledge_check(
                probe, data, anchors, attested,
                provider_config=_provider_config(),
                bootstrap_unit="anchor",  # type: ignore[call-arg]
            )
