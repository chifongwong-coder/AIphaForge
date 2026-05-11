"""v2.2.1 #12 Commit C: parser_used_distribution per-side + drift-note."""
from __future__ import annotations

import pandas as pd
import pytest

from aiphaforge.probes._rank import RankContinuationProbe
from aiphaforge.probes.anchors import build_synthetic_anchor
from aiphaforge.probes.models import AnswerRecord, AttestedAnswers
from aiphaforge.probes.orchestrator import (
    _maybe_emit_parser_used_drift_note,
    _parser_used_distribution_is_non_degenerate,
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


def _multi_data(n: int = 25, anchor_idx: int = 19):
    idx = pd.bdate_range("2024-01-01", periods=n)
    base = pd.DataFrame(
        {
            "open": [100.0] * n, "high": [101.0] * n,
            "low": [99.0] * n, "close": [100.0] * n,
            "volume": [1e6] * n,
        },
        index=idx,
    )
    a = base.copy()
    a.iloc[anchor_idx + 1, a.columns.get_loc("close")] = 105.0
    b = base.copy()
    b.iloc[anchor_idx + 1, b.columns.get_loc("close")] = 102.0
    c = base.copy()
    c.iloc[anchor_idx + 1, c.columns.get_loc("close")] = 99.0
    return {"A": a, "B": b, "C": c}, idx[anchor_idx]


# ---------- Predicate ----------


class TestPredicate:
    def test_single_value_degenerate(self):
        assert not _parser_used_distribution_is_non_degenerate(
            ["user_provided"]
        )

    def test_two_distinct_values_non_degenerate(self):
        assert _parser_used_distribution_is_non_degenerate(
            ["user_provided", "json_array"]
        )

    def test_raw_text_unparseable_counts_as_text_format(self):
        # Mix of user_provided + raw_text_unparseable → non-degenerate
        assert _parser_used_distribution_is_non_degenerate(
            ["user_provided", "raw_text_unparseable"]
        )

    def test_unrecognized_type_does_not_count(self):
        # Mix of one valid path + unrecognized_parsed_answer_type
        # → degenerate (the unrecognized is type-dispatch failure)
        assert not _parser_used_distribution_is_non_degenerate(
            ["user_provided", "unrecognized_parsed_answer_type"]
        )

    def test_empty_degenerate(self):
        assert not _parser_used_distribution_is_non_degenerate([])


# ---------- Drift note ----------


class TestDriftNote:
    def test_no_emit_when_both_degenerate(self):
        notes: list[str] = []
        _maybe_emit_parser_used_drift_note(
            {"user_provided": 5}, {"user_provided": 5}, notes,
        )
        assert notes == []

    def test_emit_when_real_non_degenerate(self):
        notes: list[str] = []
        _maybe_emit_parser_used_drift_note(
            {"user_provided": 3, "json_array": 2},
            {"user_provided": 5},
            notes,
        )
        assert len(notes) == 1
        assert "['real']" in notes[0]

    def test_emit_when_anchor_non_degenerate(self):
        notes: list[str] = []
        _maybe_emit_parser_used_drift_note(
            {"user_provided": 5},
            {"user_provided": 3, "numbered": 2},
            notes,
        )
        assert len(notes) == 1
        assert "['anchor']" in notes[0]

    def test_emit_when_both_non_degenerate(self):
        notes: list[str] = []
        _maybe_emit_parser_used_drift_note(
            {"user_provided": 3, "json_array": 2},
            {"user_provided": 3, "numbered": 2},
            notes,
        )
        assert len(notes) == 1
        assert "['real', 'anchor']" in notes[0]

    def test_note_text_includes_provider_config_pointer(self):
        notes: list[str] = []
        _maybe_emit_parser_used_drift_note(
            {"a": 1, "b": 1}, None, notes,
        )
        assert "manifest.provider_config" in notes[0]


# ---------- Integration with KnowledgeCheckReport ----------


class TestParserUsedDistributionField:
    def test_present_for_rank_probe(self):
        data, ts = _multi_data()
        probe = RankContinuationProbe(
            symbols=["A", "B", "C"], context_bars=5,
            forward_horizon=1,
        )
        qs = probe.build(data, [ts])
        qid = list(qs)[0].question_id
        attested = AttestedAnswers(
            answers=(AnswerRecord(
                question_id=qid, raw_answer=None,
                parsed_answer=["A", "B", "C"],
                parse_status="valid",
            ),),
            parsing_schema_hash="a" * 64,
            parsing_schema_description="x",
            prompt_template_hash="b" * 64,
            prompt_template_description="y",
        )
        report = knowledge_check(
            probe, data, [ts], attested,
            provider_config=_provider_config(),
        )
        assert report.parser_used_distribution_real is not None
        # Counter sums to N=1 (one record submitted)
        assert sum(report.parser_used_distribution_real.values()) == 1
        assert (
            report.parser_used_distribution_real["user_provided_list"]
            == 1
        )

    def test_anchor_distribution_present_when_anchor_supplied(self):
        data, ts = _multi_data()
        probe = RankContinuationProbe(
            symbols=["A", "B", "C"], context_bars=5,
            forward_horizon=1,
        )
        qs = probe.build(data, [ts])
        qid = list(qs)[0].question_id
        real_attested = AttestedAnswers(
            answers=(AnswerRecord(
                question_id=qid, raw_answer=None,
                parsed_answer=["A", "B", "C"],
                parse_status="valid",
            ),),
            parsing_schema_hash="a" * 64,
            parsing_schema_description="x",
            prompt_template_hash="b" * 64,
            prompt_template_description="y",
        )
        anchor_data = {
            sym: build_synthetic_anchor(
                df, seed=1, method="random_walk_volmatched",
            )
            for sym, df in data.items()
        }
        anchor_qs = probe.build(anchor_data, [ts])
        if not list(anchor_qs):
            pytest.skip("anchor build produced no questions")
        anchor_qid = list(anchor_qs)[0].question_id
        anchor_truth = list(anchor_qs)[0].truth_value
        anchor_attested = AttestedAnswers(
            answers=(AnswerRecord(
                question_id=anchor_qid, raw_answer=None,
                parsed_answer=list(anchor_truth),
                parse_status="valid",
            ),),
            parsing_schema_hash="a" * 64,
            parsing_schema_description="x",
            prompt_template_hash="b" * 64,
            prompt_template_description="y",
        )
        report = knowledge_check(
            probe, data, [ts], real_attested,
            anchor=anchor_data["A"],
            anchor_answers=anchor_attested,
            provider_config=_provider_config(),
        )
        assert report.parser_used_distribution_anchor is not None
        assert sum(
            report.parser_used_distribution_anchor.values()
        ) == 1

    def test_none_for_non_rank_probe(self):
        idx = pd.bdate_range("2024-01-01", periods=20)
        df = pd.DataFrame(
            {
                "open": [100.0] * 20, "high": [101.0] * 20,
                "low": [99.0] * 20, "close": [100.0] * 20,
                "volume": [1e6] * 20,
            },
            index=idx,
        )
        probe = KnowledgeProbe(
            symbol="AAPL", templates=DEFAULT_TEMPLATES,
        )
        anchors = list(idx[10:13])
        question_set = probe.build(df, anchors)
        attested = AttestedAnswers(
            answers=tuple(
                AnswerRecord(
                    question_id=q.question_id, raw_answer="x",
                    parsed_answer=q.truth_value,
                    parse_status="valid",
                )
                for q in question_set
            ),
            parsing_schema_hash="a" * 64,
            parsing_schema_description="x",
            prompt_template_hash="b" * 64,
            prompt_template_description="y",
        )
        report = knowledge_check(
            probe, df, anchors, attested,
            provider_config=_provider_config(),
        )
        assert report.parser_used_distribution_real is None
        assert report.parser_used_distribution_anchor is None

    def test_distribution_is_immutable_mapping_proxy(self):
        import types
        data, ts = _multi_data()
        probe = RankContinuationProbe(
            symbols=["A", "B", "C"], context_bars=5,
            forward_horizon=1,
        )
        qs = probe.build(data, [ts])
        qid = list(qs)[0].question_id
        attested = AttestedAnswers(
            answers=(AnswerRecord(
                question_id=qid, raw_answer=None,
                parsed_answer=["A", "B", "C"],
                parse_status="valid",
            ),),
            parsing_schema_hash="a" * 64,
            parsing_schema_description="x",
            prompt_template_hash="b" * 64,
            prompt_template_description="y",
        )
        report = knowledge_check(
            probe, data, [ts], attested,
            provider_config=_provider_config(),
        )
        assert isinstance(
            report.parser_used_distribution_real,
            types.MappingProxyType,
        )
        with pytest.raises(TypeError):
            report.parser_used_distribution_real["x"] = 1  # type: ignore[index]
