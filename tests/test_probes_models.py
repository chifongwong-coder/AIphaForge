"""v2.0 M1 — dataclass shape and instantiation tests for probes.models.

These tests verify the dataclass contracts (fields, defaults, typed
enums). Scoring / runner logic ships in later milestones and has its
own tests.
"""
from __future__ import annotations

import pandas as pd
import pytest

from aiphaforge.probes import (
    ABProbeResult,
    ABScenario,
    AnswerKeyRecord,
    AnswerRecord,
    MetricConfig,
    MetricDropSummary,
    QAProbeReport,
    QuestionPromptRecord,
    QuestionScore,
    QuestionSpec,
    ScenarioABReport,
    ToleranceProfile,
)
from aiphaforge.probes.models import RECOMMENDED_PROVIDER_CONFIG_KEYS


class TestToleranceProfile:
    def test_minimal_construction(self):
        tp = ToleranceProfile(
            absolute_floor=1e-8,
            exact_threshold=0.005,
            near_threshold=0.02,
            rough_threshold=0.05,
            exact_range_width=0.005,
            near_range_width=0.02,
            rough_range_width=0.05,
        )
        assert tp.max_range_width is None
        assert tp.sign_sensitive is False
        assert tp.sign_epsilon == 0.0

    def test_with_max_range_width(self):
        tp = ToleranceProfile(
            absolute_floor=1e-8,
            exact_threshold=0.005,
            near_threshold=0.02,
            rough_threshold=0.05,
            exact_range_width=0.005,
            near_range_width=0.02,
            rough_range_width=0.05,
            max_range_width=0.2,
        )
        assert tp.max_range_width == 0.2

    def test_sign_sensitive(self):
        tp = ToleranceProfile(
            absolute_floor=1e-8,
            exact_threshold=0.005,
            near_threshold=0.02,
            rough_threshold=0.05,
            exact_range_width=0.005,
            near_range_width=0.02,
            rough_range_width=0.05,
            sign_sensitive=True,
            sign_epsilon=1e-4,
        )
        assert tp.sign_sensitive is True
        assert tp.sign_epsilon == 1e-4


class TestQuestionSpec:
    def test_numeric_question(self):
        q = QuestionSpec(
            question_id="q001",
            symbol="AAPL",
            timestamp=pd.Timestamp("2020-03-12"),
            template_id="high",
            answer_type="numeric",
            prompt_text="What was the high of AAPL on 2020-03-12?",
            choices=None,
            truth_value=62.06,
            tolerance=None,
            metadata={"source": "test"},
        )
        assert q.answer_type == "numeric"
        assert q.choices is None
        assert q.metadata["source"] == "test"

    def test_choice_question(self):
        q = QuestionSpec(
            question_id="q002",
            symbol="AAPL",
            timestamp=pd.Timestamp("2020-03-12"),
            template_id="return_sign",
            answer_type="choice",
            prompt_text="Direction of return for AAPL on 2020-03-12?",
            choices=["up", "down", "unchanged"],
            truth_value="down",
            tolerance=None,
        )
        assert q.choices == ["up", "down", "unchanged"]
        assert q.truth_value == "down"


class TestExportSchemas:
    def test_question_prompt_record_has_no_truth(self):
        # Defensive check — the export dataclass MUST NOT contain the
        # truth_value field. The exporter relies on this structural
        # separation to guarantee answer-key non-leakage.
        fields = QuestionPromptRecord.__dataclass_fields__.keys()
        assert "truth_value" not in fields
        assert "tolerance" not in fields

    def test_answer_key_record_has_truth(self):
        fields = AnswerKeyRecord.__dataclass_fields__.keys()
        assert "truth_value" in fields
        assert "tolerance" in fields
        assert "prompt_text" not in fields  # prompt stays on the sheet


class TestAnswerRecord:
    def test_refusal_status_supported(self):
        a = AnswerRecord(
            question_id="q001",
            raw_answer=None,
            parsed_answer=None,
            parse_status="refusal",
        )
        assert a.parse_status == "refusal"

    def test_invalid_status_supported(self):
        a = AnswerRecord(
            question_id="q001",
            raw_answer="not a number",
            parsed_answer=None,
            parse_status="invalid",
        )
        assert a.parse_status == "invalid"


class TestMetricConfig:
    def test_higher_is_better(self):
        mc = MetricConfig(
            higher_is_better=True,
            normalization_floor=0.01,
            low_anchor_threshold=0.1,
        )
        assert mc.higher_is_better is True

    def test_max_drawdown_is_positive_magnitude(self):
        # Plan §3.5 pins max_drawdown as positive magnitude where
        # higher is worse.
        mc = MetricConfig(
            higher_is_better=False,
            normalization_floor=0.001,
            low_anchor_threshold=0.005,
        )
        assert mc.higher_is_better is False


class TestABScenario:
    def test_minimal(self):
        s = ABScenario(
            scenario_id="metadata_only",
            mode="view_only",
            transforms=[],
        )
        assert s.scenario_id == "metadata_only"
        assert s.notes is None

    def test_market_level(self):
        s = ABScenario(
            scenario_id="jitter_20bps",
            mode="market_level",
            transforms=[],
            notes="20bps OHLC jitter",
        )
        assert s.mode == "market_level"
        assert s.notes == "20bps OHLC jitter"


class TestMetricDropSummary:
    def test_low_anchor_populated_and_none_contract(self):
        # Populated/None contract: summary scalars are None when
        # n_valid_relative < min_valid_repeats.
        s = MetricDropSummary(
            metric="sharpe_ratio",
            ai_raw=[1.0, 1.0, 1.0],
            ai_test=[0.9, 0.9, 0.9],
            baseline_raw=[1.0, 1.0, 1.0],
            baseline_test=[0.95, 0.95, 0.95],
            ai_abs_drop=[0.1, 0.1, 0.1],
            baseline_abs_drop=[0.05, 0.05, 0.05],
            ai_rel_drop=[None, None, None],  # all low-anchor
            baseline_rel_drop=[None, None, None],
            excess_drop=[None, None, None],
            n_valid_relative=0,
            min_valid_repeats=5,
            n_low_anchor_ai=3,
            n_low_anchor_baseline=3,
            dominance_rate=None,
            mean_excess_drop=None,
            median_excess_drop=None,
            std_excess_drop=None,
            p10_excess_drop=None,
            p90_excess_drop=None,
        )
        assert s.dominance_rate is None
        assert s.mean_excess_drop is None
        # List lengths match n_repeat_requested (= 3 here)
        assert len(s.ai_raw) == 3
        assert len(s.excess_drop) == 3

    def test_noise_control_fields_default_none(self):
        s = MetricDropSummary(
            metric="total_return",
            ai_raw=[0.1],
            ai_test=[0.08],
            baseline_raw=[0.1],
            baseline_test=[0.09],
            ai_abs_drop=[0.02],
            baseline_abs_drop=[0.01],
            ai_rel_drop=[0.2],
            baseline_rel_drop=[0.1],
            excess_drop=[0.1],
            n_valid_relative=1,
            min_valid_repeats=1,
            n_low_anchor_ai=0,
            n_low_anchor_baseline=0,
            dominance_rate=1.0,
            mean_excess_drop=0.1,
            median_excess_drop=0.1,
            std_excess_drop=0.0,
            p10_excess_drop=0.1,
            p90_excess_drop=0.1,
        )
        assert s.ai_noise_abs is None
        assert s.mean_ai_noise_abs is None
        assert s.iqr_ai_noise_rel is None


class TestScenarioAndResultShape:
    def test_scenario_ab_report_default_warnings_empty(self):
        r = ScenarioABReport(
            scenario_id="x",
            mode="view_only",
            n_repeat_requested=10,
            n_unique_transform_realizations=10,
            metric_summaries={},
            per_repeat_table=None,
        )
        assert r.warnings == []

    def test_ab_probe_result(self):
        r = ABProbeResult(scenarios=[], manifest={"run_id": "r1"})
        assert r.manifest["run_id"] == "r1"


class TestQuestionScore:
    def test_numeric_inside_range(self):
        qs = QuestionScore(
            question_id="q001",
            validity="valid",
            band="near",
            truth_value=62.06,
            parsed_answer=(61.0, 63.0),
            relative_error=None,
            contains_truth=True,
            range_width_ratio=0.03,
            max_range_width_exceeded=False,
        )
        assert qs.contains_truth is True
        assert qs.max_range_width_exceeded is False

    def test_scalar_miss(self):
        qs = QuestionScore(
            question_id="q002",
            validity="valid",
            band="miss",
            truth_value=62.06,
            parsed_answer=100.0,
            relative_error=0.61,
            contains_truth=None,
            range_width_ratio=None,
            max_range_width_exceeded=None,
        )
        assert qs.band == "miss"
        assert qs.relative_error == pytest.approx(0.61)


class TestQAProbeReport:
    def test_empty_shape(self):
        r = QAProbeReport(
            total_questions=0,
            submitted_answers=0,
            valid_answers=0,
            invalid_answers=0,
            missing_answers=0,
            refusal_answers=0,
            coverage_rate=0.0,
            parse_success_rate=0.0,
            exact_rate=0.0,
            near_rate=0.0,
            rough_rate=0.0,
            miss_rate=0.0,
            band_index_arbitrary=None,
            bands_breakdown={},
            mean_range_width_ratio=None,
            median_range_width_ratio=None,
            max_range_width_exceeded_count=0,
            by_template=None,
            by_symbol=None,
            by_period=None,
            question_scores=[],
            manifest={},
        )
        assert r.total_questions == 0
        assert r.band_index_arbitrary is None


class TestRecommendedProviderConfigKeys:
    def test_keys_present(self):
        # These are non-enforced but shipped as a documented list for
        # cross-paper manifest comparability.
        assert "model" in RECOMMENDED_PROVIDER_CONFIG_KEYS
        assert "snapshot_id" in RECOMMENDED_PROVIDER_CONFIG_KEYS
        assert "temperature" in RECOMMENDED_PROVIDER_CONFIG_KEYS
        assert "prompt_template_hash" in RECOMMENDED_PROVIDER_CONFIG_KEYS
        assert "tool_policy" in RECOMMENDED_PROVIDER_CONFIG_KEYS
