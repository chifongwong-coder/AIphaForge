"""v2.2 M1 tests — AttestedAnswers + score_attested_answers + frozen
migration of AnswerRecord/QuestionScore.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from aiphaforge.probes._hash_utils import (
    attest_parsing,
    attest_prompt_template,
)
from aiphaforge.probes.models import (
    AnswerRecord,
    AttestedAnswers,
    ContextSerializationSpec,
    ExemplarSpec,
    QuestionScore,
)
from aiphaforge.probes.questions import (
    DEFAULT_TEMPLATES,
    KnowledgeProbe,
)
from aiphaforge.probes.scoring import (
    score_answer_file,
    score_attested_answers,
    serialize_answer_records,
)


def _make_context_spec() -> ContextSerializationSpec:
    return ContextSerializationSpec(
        format="csv",
        columns=("t", "o", "h", "l", "c", "v"),
        timestamp_format="iso_utc_seconds",
        timestamp_timezone="UTC",
        price_precision=4,
        volume_precision=0,
        nan_rendering="empty",
        line_terminator="lf",
        decimal_separator=".",
        rounding_mode="half_even",
        trailing_zero_policy="preserve",
        scientific_notation_threshold=None,
        negative_format="minus_prefix",
    )


# ---------- frozen migration ----------


class TestAnswerRecordFrozen:
    def test_answer_record_is_frozen(self):
        assert AnswerRecord.__dataclass_params__.frozen is True

    def test_assigning_to_frozen_field_raises(self):
        rec = AnswerRecord(
            question_id="q1", raw_answer="x",
            parsed_answer=1.0, parse_status="valid",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.parse_status = "invalid"

    def test_question_score_is_frozen(self):
        assert QuestionScore.__dataclass_params__.frozen is True

    def test_assigning_to_question_score_raises(self):
        qs = QuestionScore(
            question_id="q1", validity="valid", band="exact",
            truth_value=1.0, parsed_answer=1.0,
            relative_error=0.0, contains_truth=None,
            range_width_ratio=None, max_range_width_exceeded=None,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            qs.band = "miss"


# ---------- ExemplarSpec ----------


class TestExemplarSpec:
    def test_construction_with_defaults(self):
        ex = ExemplarSpec(prompt_text="Q", answer_text="A")
        assert ex.source_symbol is None
        assert ex.metadata == ""

    def test_is_frozen(self):
        ex = ExemplarSpec(prompt_text="Q", answer_text="A")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ex.prompt_text = "X"


# ---------- AttestedAnswers ----------


class TestAttestedAnswers:
    def _make(self, n: int = 3) -> AttestedAnswers:
        recs = tuple(
            AnswerRecord(
                question_id=f"q{i}", raw_answer=str(i),
                parsed_answer=float(i), parse_status="valid",
            )
            for i in range(n)
        )
        return AttestedAnswers(
            answers=recs,
            parsing_schema_hash="a" * 64,
            parsing_schema_description="test parser",
            prompt_template_hash="b" * 64,
            prompt_template_description="test template",
        )

    def test_is_frozen(self):
        a = self._make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.parsing_schema_hash = "x" * 64

    def test_carries_records(self):
        a = self._make(n=5)
        assert len(a.answers) == 5
        assert all(isinstance(r, AnswerRecord) for r in a.answers)


# ---------- score_attested_answers ----------


@pytest.fixture
def small_question_set(tmp_path):
    """Build a tiny single-symbol question set for scoring tests."""
    n = 30
    df = pd.DataFrame(
        {
            "open": np.linspace(100.0, 110.0, n),
            "high": np.linspace(101.0, 111.0, n),
            "low": np.linspace(99.0, 109.0, n),
            "close": np.linspace(100.5, 110.5, n),
            "volume": [1e6] * n,
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )
    probe = KnowledgeProbe(symbol="AAPL", templates=DEFAULT_TEMPLATES)
    qs = probe.build(df, timestamps=list(df.index[5:15]))
    return qs


class TestScoreAttestedAnswers:
    def test_returns_three_tuple(self, small_question_set):
        # Build perfect-answers attested bundle
        recs = []
        for q in small_question_set:
            recs.append(
                AnswerRecord(
                    question_id=q.question_id,
                    raw_answer=str(q.truth_value),
                    parsed_answer=q.truth_value,
                    parse_status="valid",
                )
            )
        attested = AttestedAnswers(
            answers=tuple(recs),
            parsing_schema_hash="a" * 64,
            parsing_schema_description="perfect oracle",
            prompt_template_hash="b" * 64,
            prompt_template_description="oracle prompt",
        )
        report, p_hash, t_hash = score_attested_answers(
            small_question_set, attested
        )
        assert p_hash == "a" * 64
        assert t_hash == "b" * 64

    def test_report_carries_both_hashes(self, small_question_set):
        attested = AttestedAnswers(
            answers=tuple(
                AnswerRecord(
                    question_id=q.question_id, raw_answer="?",
                    parsed_answer=q.truth_value, parse_status="valid",
                )
                for q in small_question_set
            ),
            parsing_schema_hash="c" * 64,
            parsing_schema_description="d",
            prompt_template_hash="e" * 64,
            prompt_template_description="f",
        )
        report, _, _ = score_attested_answers(
            small_question_set, attested
        )
        assert report.parsing_schema_hash == "c" * 64
        assert report.prompt_template_hash == "e" * 64

    def test_silently_skips_unknown_question_ids(self, small_question_set):
        recs = [
            AnswerRecord(
                question_id="not_in_set",
                raw_answer="x", parsed_answer=99.0,
                parse_status="valid",
            ),
        ]
        # Add a real one too so submitted_count > 0
        q0 = next(iter(small_question_set))
        recs.append(
            AnswerRecord(
                question_id=q0.question_id, raw_answer="x",
                parsed_answer=q0.truth_value, parse_status="valid",
            )
        )
        attested = AttestedAnswers(
            answers=tuple(recs),
            parsing_schema_hash="a" * 64,
            parsing_schema_description="",
            prompt_template_hash="b" * 64,
            prompt_template_description="",
        )
        report, _, _ = score_attested_answers(
            small_question_set, attested
        )
        # Unknown question_id was silently dropped → submitted = 1
        assert report.submitted_answers == 1

    def test_score_answer_file_emits_warning(
        self, small_question_set, tmp_path
    ):
        # Back-compat path emits a warning entry in manifest
        recs = [
            AnswerRecord(
                question_id=q.question_id, raw_answer="x",
                parsed_answer=q.truth_value, parse_status="valid",
            )
            for q in small_question_set
        ]
        path = tmp_path / "answers.jsonl"
        serialize_answer_records(recs, str(path))
        report = score_answer_file(small_question_set, str(path))
        assert report.parsing_schema_hash is None
        assert report.prompt_template_hash is None
        warnings = report.manifest.get("warnings", [])
        assert any(
            "score_attested_answers" in w for w in warnings
        ), warnings


# ---------- Cross-helper integration: real hashes round-trip ----------


class TestHashesRoundTripIntoReport:
    def test_real_hashes_populate_report(self, small_question_set):
        # Compute real hashes from the helpers.
        p_hash = attest_parsing(
            "parse_numeric_answer",
            "2.2.0",  # pinned release string
            {"loose": True},
        )
        t_hash = attest_prompt_template(
            system_prompt="You are a market analyst.",
            user_template="Symbol: {symbol}; date: {date}; close?",
            context_serialization=_make_context_spec(),
            answer_format_instructions="Reply with one number.",
            few_shot_exemplars=(),
        )
        recs = tuple(
            AnswerRecord(
                question_id=q.question_id, raw_answer="x",
                parsed_answer=q.truth_value, parse_status="valid",
            )
            for q in small_question_set
        )
        attested = AttestedAnswers(
            answers=recs,
            parsing_schema_hash=p_hash,
            parsing_schema_description="parse_numeric_answer/loose",
            prompt_template_hash=t_hash,
            prompt_template_description="market-analyst v1",
        )
        report, p_out, t_out = score_attested_answers(
            small_question_set, attested
        )
        assert p_out == p_hash
        assert t_out == t_hash
        assert report.parsing_schema_hash == p_hash
        assert report.prompt_template_hash == t_hash
