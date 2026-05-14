"""v2.2.1 #5 — Pair real/anchor scores by question_id with soft warning."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiphaforge.probes.anchors import build_synthetic_anchor
from aiphaforge.probes.models import (
    AnswerRecord,
    AttestedAnswers,
    QAProbeReport,
    QuestionScore,
)
from aiphaforge.probes.orchestrator import (
    _pair_scores_by_question_id,
    knowledge_check,
)
from aiphaforge.probes.questions import (
    DEFAULT_TEMPLATES,
    KnowledgeProbe,
)


def _empty_template_aggregate():
    return None  # by_template can be None


def _make_report(qids: list[str], bands: list[str]) -> QAProbeReport:
    """Minimal QAProbeReport with one QuestionScore per qid."""
    assert len(qids) == len(bands)
    scores = [
        QuestionScore(
            question_id=qid, validity="valid", band=band,
            truth_value=None, parsed_answer=None,
            relative_error=None, contains_truth=None,
            range_width_ratio=None, max_range_width_exceeded=None,
        )
        for qid, band in zip(qids, bands)
    ]
    return QAProbeReport(
        total_questions=len(qids),
        submitted_answers=len(qids),
        valid_answers=len(qids),
        invalid_answers=0,
        missing_answers=0,
        refusal_answers=0,
        coverage_rate=1.0,
        parse_success_rate=1.0,
        exact_rate=0.0,
        near_rate=0.0,
        rough_rate=0.0,
        miss_rate=0.0,
        band_index_arbitrary=None,
        bands_breakdown={
            "exact": bands.count("exact"),
            "near": bands.count("near"),
            "rough": bands.count("rough"),
            "miss": bands.count("miss"),
            "invalid": bands.count("invalid"),
        },
        mean_range_width_ratio=None,
        median_range_width_ratio=None,
        max_range_width_exceeded_count=0,
        by_template=None,
        by_symbol=None,
        by_period=None,
        question_scores=scores,
        manifest={},
    )


class TestPairByQuestionId:
    def test_full_overlap_no_warning(self):
        real = _make_report(["q1", "q2", "q3"], ["exact", "near", "miss"])
        anchor = _make_report(["q1", "q2", "q3"], ["near", "rough", "miss"])
        pairs, notes = _pair_scores_by_question_id(real, anchor)
        assert pairs == [("exact", "near"), ("near", "rough"), ("miss", "miss")]
        assert notes == []

    def test_pairs_aligned_when_anchor_reordered(self):
        # If anchor returns its scores in different order, qid-based
        # pairing still aligns correctly (position-based would misalign).
        real = _make_report(["q1", "q2", "q3"], ["exact", "near", "miss"])
        anchor = _make_report(["q3", "q1", "q2"], ["exact", "miss", "rough"])
        pairs, notes = _pair_scores_by_question_id(real, anchor)
        # sorted by qid: q1, q2, q3
        assert pairs == [
            ("exact", "miss"),  # q1: real exact, anchor miss
            ("near", "rough"),  # q2: real near, anchor rough
            ("miss", "exact"),  # q3: real miss, anchor exact
        ]
        assert notes == []

    def test_disjoint_qids_returns_empty_and_warning(self):
        real = _make_report(["q1", "q2"], ["exact", "near"])
        anchor = _make_report(["q3", "q4"], ["exact", "near"])
        pairs, notes = _pair_scores_by_question_id(real, anchor)
        assert pairs == []
        assert len(notes) == 1
        assert "anchor-pairing contract is violated" in notes[0]

    def test_partial_overlap_returns_common_and_warning(self):
        real = _make_report(["q1", "q2", "q3"], ["exact", "near", "miss"])
        anchor = _make_report(["q2", "q3", "q4"], ["rough", "miss", "exact"])
        pairs, notes = _pair_scores_by_question_id(real, anchor)
        # Common: q2, q3
        assert pairs == [("near", "rough"), ("miss", "miss")]
        assert len(notes) == 1
        assert "partially overlap" in notes[0]


# ---------- Integration with knowledge_check ----------


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


class TestPairingFailedTag:
    def test_disjoint_anchor_emits_pairing_failed_tag(
        self, monkeypatch,
    ):
        # v2.2.1 audit-fix Commit F: when
        # ``_pair_scores_by_question_id`` returns empty pairs (the
        # M5 §6.5 anchor-pairing contract is violated), the
        # orchestrator should now tag the report PAIRING_FAILED —
        # distinct from REFUSAL_SUSPECTED, which means "the model
        # effectively refused on one side".
        #
        # End-to-end forcing of disjoint qids is awkward because
        # both real and anchor reports flow through
        # ``aggregate_scores``, which synthesizes "missing"
        # placeholders keyed by the question_set qids — so the two
        # reports end up sharing qids even when the answer records
        # don't. We test the orchestrator gate directly by stubbing
        # the pairing helper to return empty pairs, then confirming
        # the validity tag flips.
        from aiphaforge.probes import orchestrator as orch_mod

        def _empty_pairs(_real, _anchor):
            return [], [
                "real and anchor question sets share NO question_ids; "
                "test stub.",
            ]

        monkeypatch.setattr(
            orch_mod, "_pair_scores_by_question_id", _empty_pairs,
        )

        data = _make_data()
        probe = KnowledgeProbe(
            symbol="AAPL", templates=DEFAULT_TEMPLATES,
        )
        anchors = list(data.index[10:15])
        question_set = probe.build(data, anchors)
        real_attested = _build_perfect_attested(question_set)
        anchor_data = build_synthetic_anchor(
            data, seed=1, method="random_walk_volmatched",
        )
        anchor_qs = probe.build(anchor_data, anchors)
        anchor_attested = _build_perfect_attested(anchor_qs)
        report = knowledge_check(
            probe, data, anchors, real_attested,
            anchor=anchor_data, anchor_answers=anchor_attested,
            provider_config=_provider_config(),
        )
        assert report.anchor_validity == "PAIRING_FAILED"
        # Derived stats remain suppressed — same gate as
        # REFUSAL_SUSPECTED, only the tag changes.
        assert report.bucket_delta is None
        assert report.bucket_delta_ci is None
        # v2.2.1 audit-fix Commit J: pin the suppression contract
        # for the scalar leakage index too. The orchestrator only
        # populates scalar_leakage_index when
        # anchor_validity == "OK"; PAIRING_FAILED must leave it None.
        assert report.scalar_leakage_index is None
        # Sanity: validity tag is genuinely distinct, not aliased.
        assert report.anchor_validity != "REFUSAL_SUSPECTED"


class TestDeprecatedPairScoresByPositionRemovalScheduled:
    def test_docstring_announces_removal_in_v2_8(self):
        # v2.2.1 Commit F set this to v2.3.0; v2.3 Commit A retargeted
        # to v2.8.0 to match the master plan v1.0 §5 roadmap (v2.x
        # line ends at v2.8 with v3.0 reserved for a separate major
        # reformulation). The removal intent must remain visible in
        # the source docstring so audit cycles don't lose it.
        from aiphaforge.probes.orchestrator import (
            _pair_scores_by_position,
        )
        doc = _pair_scores_by_position.__doc__ or ""
        assert "Removal scheduled: v2.8.0" in doc


class TestKnowledgeCheckPairByQid:
    def test_anchor_pairing_uses_qid_not_position(self):
        # End-to-end: real and anchor probes share qids by
        # construction (same probe config + same anchors). The
        # qid-based pairing should produce identical results to
        # position-based here, but the contract is now qid-based.
        data = _make_data()
        probe = KnowledgeProbe(
            symbol="AAPL", templates=DEFAULT_TEMPLATES,
        )
        anchors = list(data.index[10:15])
        question_set = probe.build(data, anchors)
        real_attested = _build_perfect_attested(question_set)
        anchor_data = build_synthetic_anchor(
            data, seed=1, method="random_walk_volmatched",
        )
        anchor_qs = probe.build(anchor_data, anchors)
        anchor_attested = _build_perfect_attested(anchor_qs)
        report = knowledge_check(
            probe, data, anchors, real_attested,
            anchor=anchor_data, anchor_answers=anchor_attested,
            provider_config=_provider_config(),
        )
        # Should not produce a "partially overlap" or "violated" note.
        joined = " ".join(report.notes).lower()
        assert "violated" not in joined
        assert "partially overlap" not in joined
