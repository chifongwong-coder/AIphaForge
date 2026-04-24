"""v2.0 M3 — tests for Q&A scoring (binary, choice, numeric, range)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from aiphaforge.probes import (
    AnswerRecord,
    BarRangePct,
    CloseQuestion,
    CloseVsOpen,
    OpenQuestion,
    QuestionSpec,
    ToleranceProfile,
    aggregate_scores,
    build_question_set,
    normalize_binary,
    normalize_direction,
    sample_dates,
    score_answer_file,
    score_question,
    serialize_answer_records,
)
from tests.conftest import make_probe_ohlcv as _ohlcv  # noqa: E402

# ---------- Normalizers ----------

class TestNormalize:
    @pytest.mark.parametrize("text,expected", [
        ("yes", True), ("Y", True), ("true", True), ("1", True),
        ("no", False), ("N", False), ("false", False), ("0", False),
    ])
    def test_binary(self, text, expected):
        assert normalize_binary(text) is expected

    def test_binary_unparseable_returns_none(self):
        assert normalize_binary("maybe") is None

    @pytest.mark.parametrize("text", ["up", "down", "higher", "lower"])
    def test_binary_does_not_overload_direction_aliases(self, text):
        # Directional aliases live in normalize_direction, not
        # normalize_binary — mapping "up" → True for an arbitrary
        # binary template ("Was the close BELOW $60?") would silently
        # mis-score the answer.
        assert normalize_binary(text) is None

    @pytest.mark.parametrize("text,expected", [
        ("up", "up"), ("Higher", "up"), ("rose", "up"),
        ("DOWN", "down"), ("fell", "down"),
        ("flat", "unchanged"), ("Same", "unchanged"),
    ])
    def test_direction(self, text, expected):
        assert normalize_direction(text) == expected

    def test_direction_unparseable_returns_none(self):
        assert normalize_direction("sideways") is None


# ---------- Choice scoring ----------

class TestChoiceScoring:
    def test_correct_close_vs_open(self):
        data = _ohlcv()
        ts = next(t for t in data.index if data.loc[t, "close"] > data.loc[t, "open"])
        qs = CloseVsOpen().build(data, "X", ts)
        ans = AnswerRecord(qs.question_id, raw_answer="Higher",
                           parsed_answer="up", parse_status="valid")
        s = score_question(qs, ans)
        assert s.band == "exact"
        assert s.validity == "valid"

    def test_wrong_choice_scores_miss(self):
        data = _ohlcv()
        ts = next(t for t in data.index if data.loc[t, "close"] > data.loc[t, "open"])
        qs = CloseVsOpen().build(data, "X", ts)
        ans = AnswerRecord(qs.question_id, raw_answer="lower",
                           parsed_answer="down", parse_status="valid")
        s = score_question(qs, ans)
        assert s.band == "miss"

    def test_unparseable_choice_invalid(self):
        data = _ohlcv()
        qs = CloseVsOpen().build(data, "X", data.index[5])
        ans = AnswerRecord(qs.question_id, raw_answer="???",
                           parsed_answer="???", parse_status="valid")
        s = score_question(qs, ans)
        assert s.band == "invalid"
        assert s.validity == "invalid"


# ---------- Numeric scalar scoring ----------

class TestScalarNumericScoring:
    def _qs(self, truth: float, tol: ToleranceProfile = None) -> QuestionSpec:
        return QuestionSpec(
            question_id="q1",
            symbol="X",
            timestamp=pd.Timestamp("2024-01-15"),
            template_id="close",
            answer_type="numeric",
            prompt_text="?",
            choices=None,
            truth_value=truth,
            tolerance=tol or ToleranceProfile(
                absolute_floor=1e-6,
                exact_threshold=0.005,
                near_threshold=0.02,
                rough_threshold=0.05,
                exact_range_width=0.005,
                near_range_width=0.02,
                rough_range_width=0.05,
                max_range_width=0.20,
            ),
        )

    def _ans(self, parsed) -> AnswerRecord:
        return AnswerRecord("q1", raw_answer=str(parsed),
                            parsed_answer=parsed, parse_status="valid")

    def test_exact_band(self):
        s = score_question(self._qs(100.0), self._ans(100.1))
        assert s.band == "exact"

    def test_near_band(self):
        s = score_question(self._qs(100.0), self._ans(101.5))
        assert s.band == "near"

    def test_rough_band(self):
        s = score_question(self._qs(100.0), self._ans(104.0))
        assert s.band == "rough"

    def test_miss_band(self):
        s = score_question(self._qs(100.0), self._ans(150.0))
        assert s.band == "miss"

    def test_relative_error_recorded(self):
        s = score_question(self._qs(100.0), self._ans(105.0))
        assert s.relative_error == pytest.approx(0.05 / 1.05, rel=1e-6)

    def test_zero_truth_uses_floor_scale(self):
        # truth=0, answer=tiny → relative error scales by max(|t|,|a|,floor)
        s = score_question(self._qs(0.0), self._ans(1e-9))
        # |1e-9 - 0| / max(0, 1e-9, 1e-6) = 1e-9/1e-6 = 1e-3 → "exact"
        assert s.band == "exact"

    def test_sign_sensitive_flag_band_miss_and_metadata(self):
        tol = ToleranceProfile(
            absolute_floor=1e-6, exact_threshold=0.005, near_threshold=0.02,
            rough_threshold=0.05, exact_range_width=0.005,
            near_range_width=0.02, rough_range_width=0.05,
            max_range_width=0.20, sign_sensitive=True, sign_epsilon=1e-4,
        )
        # Truth +0.01, answer -0.01 → sign error.
        s = score_question(self._qs(0.01, tol), self._ans(-0.01))
        assert s.band == "miss"
        assert s.metadata.get("sign_error") is True


# ---------- Numeric range scoring ----------

class TestRangeNumericScoring:
    def _qs(self, truth: float, *, max_w: float = 0.20) -> QuestionSpec:
        return QuestionSpec(
            question_id="q1", symbol="X",
            timestamp=pd.Timestamp("2024-01-15"),
            template_id="close", answer_type="numeric",
            prompt_text="?", choices=None, truth_value=truth,
            tolerance=ToleranceProfile(
                absolute_floor=1e-6,
                exact_threshold=0.005, near_threshold=0.02,
                rough_threshold=0.05,
                exact_range_width=0.005, near_range_width=0.02,
                rough_range_width=0.05,
                max_range_width=max_w,
            ),
        )

    def _ans(self, lo: float, hi: float) -> AnswerRecord:
        return AnswerRecord("q1", raw_answer=f"[{lo}, {hi}]",
                            parsed_answer=(lo, hi), parse_status="valid")

    def test_inside_truth_narrow_range_exact(self):
        s = score_question(self._qs(100.0), self._ans(99.7, 100.2))
        assert s.contains_truth is True
        assert s.band == "exact"

    def test_inside_truth_medium_range_near(self):
        s = score_question(self._qs(100.0), self._ans(99.0, 101.0))
        assert s.contains_truth is True
        assert s.band == "near"

    def test_inside_truth_wide_range_rough(self):
        s = score_question(self._qs(100.0), self._ans(95.0, 105.0))
        assert s.contains_truth is True
        assert s.band == "rough"

    def test_max_range_width_demotes_to_miss(self):
        # The anti-gaming cap: an enormous range that contains the
        # truth must be a miss, not "rough".
        s = score_question(self._qs(100.0, max_w=0.20),
                           self._ans(-1_000_000.0, 1_000_000.0))
        assert s.contains_truth is True
        assert s.max_range_width_exceeded is True
        assert s.band == "miss"

    def test_outside_close_rough(self):
        s = score_question(self._qs(100.0), self._ans(102.0, 104.0))
        assert s.contains_truth is False
        assert s.band == "rough"

    def test_outside_far_miss(self):
        s = score_question(self._qs(100.0), self._ans(200.0, 300.0))
        assert s.contains_truth is False
        assert s.band == "miss"

    def test_lo_gt_hi_invalid(self):
        s = score_question(self._qs(100.0), self._ans(105.0, 95.0))
        assert s.band == "invalid"

    def test_lo_eq_hi_treated_as_scalar(self):
        # Degenerate range == scalar → routed through scalar scoring,
        # not auto-promoted to "exact" by virtue of width=0.
        s = score_question(self._qs(100.0), self._ans(100.0, 100.0))
        assert s.band == "exact"
        # An obviously-wrong "scalar range" should still miss.
        s2 = score_question(self._qs(100.0), self._ans(150.0, 150.0))
        assert s2.band == "miss"


# ---------- Binary + status handling ----------

class TestStatusAndAggregation:
    def test_refusal_yields_validity_refusal(self):
        data = _ohlcv()
        qs = OpenQuestion().build(data, "X", data.index[5])
        ans = AnswerRecord(qs.question_id, raw_answer=None,
                           parsed_answer=None, parse_status="refusal")
        s = score_question(qs, ans)
        assert s.validity == "refusal"
        assert s.band == "invalid"

    def test_missing_yields_validity_missing(self):
        data = _ohlcv()
        qs = OpenQuestion().build(data, "X", data.index[5])
        ans = AnswerRecord(qs.question_id, raw_answer=None,
                           parsed_answer=None, parse_status="missing")
        s = score_question(qs, ans)
        assert s.validity == "missing"

    def test_user_invalid_kept_as_invalid(self):
        data = _ohlcv()
        qs = OpenQuestion().build(data, "X", data.index[5])
        ans = AnswerRecord(qs.question_id, raw_answer="garbage",
                           parsed_answer=None, parse_status="invalid")
        s = score_question(qs, ans)
        assert s.validity == "invalid"

    def test_question_id_mismatch_raises(self):
        # Regression: pairing contract is enforced eagerly so off-by-one
        # in a list-zip cannot silently produce wrong scores.
        data = _ohlcv()
        qs = OpenQuestion().build(data, "X", data.index[5])
        ans = AnswerRecord(
            "WRONG_ID", raw_answer="x", parsed_answer=100.0,
            parse_status="valid",
        )
        with pytest.raises(ValueError, match="id mismatch"):
            score_question(qs, ans)


class TestMixedCaseChoiceTemplate:
    def test_uppercase_allowed_set_normalized_correctly(self):
        # Regression: a future direction-style template using
        # ["UP", "DOWN", "UNCHANGED"] (uppercase) must still resolve
        # canonical-direction normalization back into the template's
        # exact casing — otherwise `parsed not in allowed` rejects the
        # answer as `invalid`.
        from aiphaforge.probes.scoring import _normalize_choice
        result = _normalize_choice("Higher", ["UP", "DOWN", "UNCHANGED"])
        assert result == "UP"
        result = _normalize_choice("fell", ["UP", "DOWN", "UNCHANGED"])
        assert result == "DOWN"


# ---------- Aggregator ----------

class TestAggregate:
    def test_missing_question_counted(self):
        data = _ohlcv()
        qs = build_question_set(
            data, "X", [data.index[5], data.index[10]],
            [CloseQuestion()],
        )
        # Only answer the first.
        scores = [
            score_question(
                qs.questions[0],
                AnswerRecord(qs.questions[0].question_id, raw_answer="100",
                             parsed_answer=qs.questions[0].truth_value,
                             parse_status="valid"),
            ),
        ]
        report = aggregate_scores(qs, scores)
        assert report.total_questions == 2
        assert report.submitted_answers == 1
        assert report.missing_answers == 1
        assert report.coverage_rate == 0.5

    def test_band_index_arbitrary_present(self):
        data = _ohlcv()
        qs = build_question_set(
            data, "X", [data.index[5]], [CloseQuestion()],
        )
        scores = [
            score_question(
                qs.questions[0],
                AnswerRecord(qs.questions[0].question_id, raw_answer="x",
                             parsed_answer=qs.questions[0].truth_value,
                             parse_status="valid"),
            ),
        ]
        report = aggregate_scores(qs, scores)
        # Single exact answer → band_index == 1.0.
        assert report.band_index_arbitrary == pytest.approx(1.0)
        assert report.bands_breakdown.get("exact") == 1

    def test_range_width_aggregates_none_when_no_ranges(self):
        data = _ohlcv()
        qs = build_question_set(
            data, "X", [data.index[5]], [CloseQuestion()],
        )
        scores = [
            score_question(
                qs.questions[0],
                AnswerRecord(qs.questions[0].question_id, raw_answer="x",
                             parsed_answer=100.0, parse_status="valid"),
            ),
        ]
        report = aggregate_scores(qs, scores)
        assert report.mean_range_width_ratio is None
        assert report.median_range_width_ratio is None
        assert report.max_range_width_exceeded_count == 0

    def test_range_width_aggregates_populated(self):
        data = _ohlcv()
        qs = build_question_set(
            data, "X", [data.index[5], data.index[10]],
            [CloseQuestion()],
        )
        # Two range answers, one inside truth → narrow, one wide.
        truth_a = qs.questions[0].truth_value
        truth_b = qs.questions[1].truth_value
        scores = [
            score_question(
                qs.questions[0],
                AnswerRecord(qs.questions[0].question_id, raw_answer="r1",
                             parsed_answer=(truth_a - 0.5, truth_a + 0.5),
                             parse_status="valid"),
            ),
            score_question(
                qs.questions[1],
                AnswerRecord(qs.questions[1].question_id, raw_answer="r2",
                             parsed_answer=(truth_b - 5, truth_b + 5),
                             parse_status="valid"),
            ),
        ]
        report = aggregate_scores(qs, scores)
        assert report.mean_range_width_ratio is not None
        assert report.median_range_width_ratio is not None


# ---------- File-driven workflow ----------

class TestFileWorkflow:
    def test_round_trip_score_answer_file(self, tmp_path):
        data = _ohlcv(n=60)
        ts_list = sample_dates(data, n=5, seed=0, start=1)
        qs = build_question_set(
            data, "X", ts_list,
            [OpenQuestion(), CloseQuestion(), CloseVsOpen(), BarRangePct()],
        )
        # Build answers: half correct, half wrong.
        answers: list[AnswerRecord] = []
        for i, q in enumerate(qs):
            if q.answer_type == "numeric":
                parsed = float(q.truth_value) if i % 2 == 0 else 0.0
            elif q.answer_type == "choice":
                parsed = q.truth_value if i % 2 == 0 else "down"
            else:
                parsed = q.truth_value
            answers.append(AnswerRecord(
                question_id=q.question_id,
                raw_answer=str(parsed),
                parsed_answer=parsed,
                parse_status="valid",
            ))
        ans_path = tmp_path / "answers.jsonl"
        serialize_answer_records(answers, str(ans_path))
        report = score_answer_file(qs, str(ans_path))
        assert report.total_questions == len(qs)
        assert report.submitted_answers == len(qs)
        assert report.valid_answers == len(qs)
        # Some exacts and some misses by construction.
        assert report.bands_breakdown.get("exact", 0) > 0
        assert report.bands_breakdown.get("miss", 0) > 0

    def test_extra_rows_in_file_ignored(self, tmp_path):
        data = _ohlcv(n=60)
        qs = build_question_set(
            data, "X", [data.index[5]], [CloseQuestion()],
        )
        ans_path = tmp_path / "answers.jsonl"
        # Write one matching row and one orphan row.
        with open(ans_path, "w") as f:
            f.write(json.dumps({
                "question_id": qs.questions[0].question_id,
                "raw_answer": "x",
                "parsed_answer": qs.questions[0].truth_value,
                "parse_status": "valid",
                "metadata": {},
            }) + "\n")
            f.write(json.dumps({
                "question_id": "ORPHAN|2024-01-01|x",
                "raw_answer": "x",
                "parsed_answer": 1.0,
                "parse_status": "valid",
                "metadata": {},
            }) + "\n")
        report = score_answer_file(qs, str(ans_path))
        assert report.total_questions == 1
        assert report.submitted_answers == 1
        assert report.valid_answers == 1

    def test_range_round_trips_through_jsonl(self, tmp_path):
        data = _ohlcv(n=60)
        qs = build_question_set(
            data, "X", [data.index[5]], [CloseQuestion()],
        )
        truth = float(qs.questions[0].truth_value)
        ans = AnswerRecord(
            qs.questions[0].question_id,
            raw_answer="[t-1, t+1]",
            parsed_answer=(truth - 1.0, truth + 1.0),
            parse_status="valid",
        )
        ans_path = tmp_path / "answers.jsonl"
        serialize_answer_records([ans], str(ans_path))
        report = score_answer_file(qs, str(ans_path))
        assert report.valid_answers == 1
        s = report.question_scores[0]
        assert s.contains_truth is True
        assert s.range_width_ratio is not None


# ---------- v2.0.1 M5: by_template aggregation ----------

class TestByTemplate:
    def _build_multi(self, *, all_correct: bool):
        from aiphaforge.probes.scoring import score_question
        data = _ohlcv(n=60)
        qs = build_question_set(
            data, "X", [data.index[10], data.index[20]],
            [OpenQuestion(), CloseQuestion(), CloseVsOpen()],
        )
        scores = []
        for q in qs:
            if all_correct:
                ans = AnswerRecord(
                    q.question_id, raw_answer=str(q.truth_value),
                    parsed_answer=q.truth_value, parse_status="valid",
                )
            else:
                # Mix: parse_status valid but with deliberately wrong
                # numeric to force band="miss".
                wrong = q.truth_value
                if isinstance(wrong, (int, float)):
                    wrong = float(wrong) * 1000.0
                ans = AnswerRecord(
                    q.question_id, raw_answer=str(wrong),
                    parsed_answer=wrong, parse_status="valid",
                )
            scores.append(score_question(q, ans))
        return qs, scores

    def test_by_template_populated_with_one_entry_per_template(self):
        qs, scores = self._build_multi(all_correct=True)
        report = aggregate_scores(qs, scores)
        assert report.by_template is not None
        # OpenQuestion, CloseQuestion, CloseVsOpen → 3 distinct templates.
        assert set(report.by_template.keys()) == {
            q.template_id for q in qs.questions
        }
        assert len(report.by_template) == 3

    def test_by_template_inner_keys_match_schema(self):
        qs, scores = self._build_multi(all_correct=True)
        report = aggregate_scores(qs, scores)
        # r5 §6.1 TemplateAggregate shape.
        expected = {
            "n_questions", "submitted_answers", "valid_answers",
            "invalid_answers", "missing_answers", "refusal_answers",
            "coverage_rate", "parse_success_rate",
            "exact_rate", "near_rate", "rough_rate", "miss_rate",
            "band_index_arbitrary", "bands_breakdown",
            "mean_range_width_ratio", "median_range_width_ratio",
            "max_range_width_exceeded_count",
        }
        for tid, entry in report.by_template.items():
            assert set(entry.keys()) == expected
            assert entry["n_questions"] >= 1
            # All-correct fixture: every band rate is non-None and bounded.
            assert 0.0 <= entry["exact_rate"] <= 1.0

    def test_by_template_n_questions_sum_equals_total(self):
        qs, scores = self._build_multi(all_correct=True)
        report = aggregate_scores(qs, scores)
        total = sum(e["n_questions"] for e in report.by_template.values())
        assert total == report.total_questions

    def test_per_template_rates_pass_through_when_all_correct(self):
        qs, scores = self._build_multi(all_correct=True)
        report = aggregate_scores(qs, scores)
        for entry in report.by_template.values():
            # exact + near + rough + miss == 1.0 over the valid bucket.
            assert entry["exact_rate"] + entry["near_rate"] \
                + entry["rough_rate"] + entry["miss_rate"] \
                == pytest.approx(1.0)

    def test_single_template_report_has_one_entry(self):
        data = _ohlcv(n=30)
        qs = build_question_set(
            data, "X", [data.index[5], data.index[10]], [OpenQuestion()],
        )
        scores = []
        for q in qs:
            ans = AnswerRecord(
                q.question_id, raw_answer=str(q.truth_value),
                parsed_answer=q.truth_value, parse_status="valid",
            )
            scores.append(score_question(q, ans))
        report = aggregate_scores(qs, scores)
        assert report.by_template is not None
        assert len(report.by_template) == 1
        only = next(iter(report.by_template.values()))
        assert only["n_questions"] == 2

    def test_empty_question_set_yields_none(self):
        from aiphaforge.probes.questions import QuestionSet
        empty = QuestionSet(questions=[])
        report = aggregate_scores(empty, [])
        assert report.by_template is None

    def test_band_index_per_template_independent_of_global(self):
        # Selective-memorization signal: OpenQuestion all-correct vs
        # CloseQuestion all-wrong should yield distinct band_index
        # values per template even when the global rate is mixed.
        data = _ohlcv(n=40)
        qs = build_question_set(
            data, "X", [data.index[10], data.index[20]],
            [OpenQuestion(), CloseQuestion()],
        )
        scores = []
        for q in qs:
            if q.template_id == "open":
                pa = q.truth_value
            else:
                pa = float(q.truth_value) * 1000.0
            scores.append(score_question(
                q, AnswerRecord(
                    q.question_id, raw_answer=str(pa),
                    parsed_answer=pa, parse_status="valid",
                ),
            ))
        report = aggregate_scores(qs, scores)
        assert report.by_template["open"]["exact_rate"] == pytest.approx(1.0)
        assert report.by_template["close"]["miss_rate"] == pytest.approx(1.0)

    # ---- r5 §6 mandated tests ----

    def test_by_template_zero_valid_answers_uses_none_band_rates(self):
        # r5 §6.2: when valid_answers == 0, band rates must be None,
        # not 0.0, so a "no scorable answers" template is not visually
        # indistinguishable from an "all-miss" template.
        data = _ohlcv(n=30)
        qs = build_question_set(
            data, "X", [data.index[5], data.index[10]], [OpenQuestion()],
        )
        # Submit all answers as parse_status="invalid" so valid_answers is 0
        # but submitted_answers > 0 — exercises the None-band-rate path.
        scores = []
        for q in qs:
            scores.append(score_question(
                q, AnswerRecord(
                    q.question_id, raw_answer="garbage",
                    parsed_answer=None, parse_status="invalid",
                ),
            ))
        report = aggregate_scores(qs, scores)
        entry = report.by_template["open"]
        assert entry["valid_answers"] == 0
        assert entry["exact_rate"] is None
        assert entry["near_rate"] is None
        assert entry["rough_rate"] is None
        assert entry["miss_rate"] is None
        # Coverage and parse-success use different denominators and
        # remain numeric even when no answer scored as valid.
        assert entry["coverage_rate"] == pytest.approx(1.0)
        assert entry["parse_success_rate"] == pytest.approx(0.0)
        # band_index_arbitrary should also be None when no valid answers.
        assert entry["band_index_arbitrary"] is None

    def test_by_template_type_shape_includes_counts_and_breakdown(self):
        # r5 §6.1: TemplateAggregate must carry counts (n_questions,
        # submitted/valid/invalid/missing/refusal), a bands_breakdown
        # dict, range-width aggregates, and the max-range-exceeded
        # count alongside the rate fields.
        qs, scores = self._build_multi(all_correct=True)
        report = aggregate_scores(qs, scores)
        for entry in report.by_template.values():
            # Counts.
            for key in (
                "n_questions", "submitted_answers", "valid_answers",
                "invalid_answers", "missing_answers", "refusal_answers",
            ):
                assert isinstance(entry[key], int)
            # Counts must add up: submitted+missing == n_questions.
            assert (
                entry["submitted_answers"] + entry["missing_answers"]
                == entry["n_questions"]
            )
            # Breakdown is a real dict.
            assert isinstance(entry["bands_breakdown"], dict)
            # Range-width aggregates and exceeded count exist.
            assert "mean_range_width_ratio" in entry
            assert "median_range_width_ratio" in entry
            assert isinstance(entry["max_range_width_exceeded_count"], int)
