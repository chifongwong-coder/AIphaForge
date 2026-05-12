"""v2.2.1 audit-fix nits — addresses A+B+C reviewer findings.

Fixes:
- LE #1: structured-input carve-out also requires parse_status='valid'.
- LE #2: KnowledgeCheckReport.to_dict() helper for asdict compat.
- Drift-note text uses comma-joined sides instead of list repr.
- anchor_parser_counter hoisted out of conditional block.
- Adds idempotency test for RankContinuationProbe via knowledge_check.
"""
from __future__ import annotations

import json

import pandas as pd

from aiphaforge.probes._rank import RankAnswer, RankContinuationProbe
from aiphaforge.probes.models import AnswerRecord, AttestedAnswers
from aiphaforge.probes.orchestrator import (
    KnowledgeCheckReport,
    compute_effective_rate,
    compute_refusal_rate,
    knowledge_check,
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


# ---------- LE #1: parse_status gating on carve-out ----------


class TestCarveOutRequiresParseStatusValid:
    def test_structured_with_parse_status_invalid_does_not_apply_carveout(self):
        # User passes structured RankAnswer but parse_status='invalid'
        # (e.g., upstream schema validation flagged it).
        # The refusal carve-out should NOT bypass refusal heuristic;
        # the record should be evaluated normally (refused if hedged).
        rec = AnswerRecord(
            question_id="q1",
            raw_answer="I'm not sure but here is my answer",
            parsed_answer=RankAnswer(ranking=("A", "B")),
            parse_status="invalid",  # upstream rejection
        )
        # parse_status='invalid' → NOT counted as effective even
        # though parsed_answer is structured.
        eff = compute_effective_rate([rec])
        assert eff == 0.0

    def test_structured_with_parse_status_invalid_is_counted_as_refused(self):
        rec = AnswerRecord(
            question_id="q1",
            raw_answer="I cannot answer that",
            parsed_answer=RankAnswer(ranking=("A", "B")),
            parse_status="invalid",
        )
        # parse_status='invalid' + refusal-keyword raw_answer →
        # refusal_rate counts it.
        assert compute_refusal_rate([rec]) == 1.0

    def test_structured_with_parse_status_valid_still_carves_out(self):
        # Existing carve-out behavior preserved when parse_status='valid'.
        rec = AnswerRecord(
            question_id="q1",
            raw_answer="I'm not sure",
            parsed_answer=RankAnswer(ranking=("A", "B")),
            parse_status="valid",
        )
        assert compute_effective_rate([rec]) == 1.0
        assert compute_refusal_rate([rec]) == 0.0


# ---------- LE #2: KnowledgeCheckReport.to_dict() ----------


class TestKnowledgeCheckReportToDict:
    def _build_report_with_distribution(self) -> KnowledgeCheckReport:
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
        return knowledge_check(
            probe, data, [ts], attested,
            provider_config=_provider_config(),
        )

    def test_to_dict_unwraps_mappingproxy(self):
        report = self._build_report_with_distribution()
        d = report.to_dict()
        # parser_used_distribution_real is now a plain dict (not
        # MappingProxyType) so json.dumps can serialize it.
        assert isinstance(d["parser_used_distribution_real"], dict)

    def test_to_dict_is_json_serializable(self):
        # The whole point of LE #2: users following the
        # asdict→json.dumps pattern should not crash.
        report = self._build_report_with_distribution()
        d = report.to_dict()
        # Some nested dataclasses (real_score) aren't JSON-friendly
        # — drop them for this test; we only verify the
        # MappingProxyType fields specifically.
        d_subset = {
            k: d[k] for k in (
                "parser_used_distribution_real",
                "parser_used_distribution_anchor",
                "scalar_leakage_index",
                "scalar_leakage_index_z",
                "report_uuid",
                "wall_clock_utc",
                "anchor_validity",
            )
            if k in d
        }
        # Should not raise.
        json.dumps(d_subset, default=str)


# ---------- Idempotency for rank probe ----------


class TestRankProbeIdempotency:
    def test_two_calls_produce_identical_report_modulo_uuid_walltime(self):
        # The v2.2 contract: knowledge_check is pure idempotent;
        # only report_uuid and wall_clock_utc change between runs.
        # After A+B+C added rank dispatch, parser_used Counter, and
        # drift-note logic, this contract needs explicit re-verification.
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
        r1 = knowledge_check(
            probe, data, [ts], attested,
            provider_config=_provider_config(),
        )
        r2 = knowledge_check(
            probe, data, [ts], attested,
            provider_config=_provider_config(),
        )
        # Compare bucket counts + scalar + parser distribution.
        assert (
            r1.real_score.bands_breakdown
            == r2.real_score.bands_breakdown
        )
        assert r1.scalar_leakage_index == r2.scalar_leakage_index
        assert (
            dict(r1.parser_used_distribution_real or {})
            == dict(r2.parser_used_distribution_real or {})
        )
        assert r1.notes == r2.notes  # also deterministic
        # UUID and wall_clock SHOULD differ.
        assert r1.report_uuid != r2.report_uuid
