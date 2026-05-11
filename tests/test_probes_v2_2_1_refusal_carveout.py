"""v2.2.1 #12 Commit B: structured-input refusal carve-out.

Plan §2.5: when user passes a structured parsed_answer (RankAnswer,
list[str], dict-with-ranking), the refusal heuristic on raw_answer
should be SKIPPED — the user has already done the parsing work.

Applies across ALL probe types (not rank-only) because
compute_effective_rate / compute_refusal_rate are shared helpers.
"""
from __future__ import annotations

from aiphaforge.probes._rank import RankAnswer
from aiphaforge.probes.models import AnswerRecord
from aiphaforge.probes.orchestrator import (
    _is_structured_rank_input,
    compute_effective_rate,
    compute_refusal_rate,
)


class TestIsStructuredRankInput:
    def test_rankanswer_instance(self):
        ra = RankAnswer(ranking=("A", "B"))
        assert _is_structured_rank_input(ra) is True

    def test_list_of_str(self):
        assert _is_structured_rank_input(["A", "B", "C"]) is True

    def test_tuple_of_str(self):
        assert _is_structured_rank_input(("A", "B")) is True

    def test_dict_with_ranking(self):
        assert _is_structured_rank_input(
            {"ranking": ["A", "B"]}
        ) is True

    def test_bare_string_rejected(self):
        assert _is_structured_rank_input("AAPL") is False

    def test_none_rejected(self):
        assert _is_structured_rank_input(None) is False

    def test_int_rejected(self):
        assert _is_structured_rank_input(42) is False

    def test_dict_without_ranking_rejected(self):
        assert _is_structured_rank_input({"foo": ["A"]}) is False

    def test_mixed_type_list_rejected(self):
        assert _is_structured_rank_input([1, "A"]) is False


class TestEffectiveRateCarveOut:
    def test_structured_rankanswer_counted_effective_despite_hedged_raw(self):
        # Plan §2.5: structured input bypasses refusal heuristic.
        rec = AnswerRecord(
            question_id="q1",
            raw_answer="I'm not sure but here is my answer",
            parsed_answer=RankAnswer(ranking=("A", "B")),
            parse_status="valid",
        )
        eff = compute_effective_rate([rec])
        assert eff == 1.0

    def test_structured_list_counted_effective_despite_hedged_raw(self):
        rec = AnswerRecord(
            question_id="q1",
            raw_answer="I don't have access to this data",
            parsed_answer=["A", "B"],
            parse_status="valid",
        )
        eff = compute_effective_rate([rec])
        assert eff == 1.0

    def test_text_only_hedge_counted_as_refused(self):
        # Existing behavior preserved when no structured payload.
        rec = AnswerRecord(
            question_id="q1",
            raw_answer="I'm not sure",
            parsed_answer=None,
            parse_status="valid",
        )
        eff = compute_effective_rate([rec])
        assert eff == 0.0

    def test_mixed_batch(self):
        recs = [
            AnswerRecord(
                question_id="q1", raw_answer="I'm not sure",
                parsed_answer=RankAnswer(ranking=("A",)),
                parse_status="valid",
            ),  # structured → counted
            AnswerRecord(
                question_id="q2", raw_answer="I don't have access",
                parsed_answer=None, parse_status="valid",
            ),  # refused → not counted
            AnswerRecord(
                question_id="q3", raw_answer="42",
                parsed_answer=42.0, parse_status="valid",
            ),  # plain numeric → counted
        ]
        eff = compute_effective_rate(recs)
        # 2 of 3 counted (q1 structured + q3 valid; q2 refused)
        assert eff == 2.0 / 3.0


class TestRefusalRateCarveOut:
    def test_structured_input_not_counted_as_refused(self):
        rec = AnswerRecord(
            question_id="q1",
            raw_answer="I cannot answer that",  # refusal-keyword
            parsed_answer=RankAnswer(ranking=("A",)),
            parse_status="valid",
        )
        # Even though raw_answer looks like refusal, structured
        # parsed_answer skips the heuristic → refusal_rate = 0.
        assert compute_refusal_rate([rec]) == 0.0

    def test_text_only_refusal_still_counted(self):
        rec = AnswerRecord(
            question_id="q1",
            raw_answer="I cannot answer that",
            parsed_answer=None, parse_status="valid",
        )
        assert compute_refusal_rate([rec]) == 1.0
