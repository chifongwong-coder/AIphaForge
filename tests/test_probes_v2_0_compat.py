"""v2.0.1 M6 — backward-compatibility regression suite.

Mechanically asserts that every public name in
``aiphaforge.probes.__all__`` still imports, and that the v2.0
"Day-One" example still runs end-to-end without v2.0.1 keyword
arguments. This is the "no v2.0 import or call site breaks" success
criterion in machine-checkable form.
"""
from __future__ import annotations

import importlib

import pytest

import aiphaforge.probes as probes_pkg
from aiphaforge.probes import (
    AnswerRecord,
    KnowledgeProbe,
    OpenQuestion,
    aggregate_scores,
    build_question_set,
    sample_dates,
    score_answer_file,
    serialize_answer_records,
)
from tests.conftest import make_probe_ohlcv as _ohlcv

# ---------- Public-surface stability ----------

class TestPublicSurface:
    def test_every_name_in_all_resolves(self):
        # Mechanical: every entry in __all__ must be a real attribute,
        # not a stale string left behind after a refactor.
        missing = [
            name for name in probes_pkg.__all__
            if not hasattr(probes_pkg, name)
        ]
        assert missing == [], f"missing names in aiphaforge.probes: {missing}"

    def test_v2_0_required_names_still_exported(self):
        # Frozen list of names that v2.0 README + docs depended on.
        # If any one of these disappears we have broken back-compat.
        v2_0_names = {
            "ABProbeResult", "ABScenario", "AgentContract",
            "AnswerKeyRecord", "AnswerRecord",
            "BarRangePct", "CloseQuestion", "CloseVsOpen",
            "DEFAULT_METRIC_CONFIG", "DEFAULT_METRICS",
            "DEFAULT_TEMPLATES",
            "GapVsPrevClose", "HighQuestion", "KnowledgeProbe",
            "LowQuestion", "MACrossBaseline", "MeanRevBaseline",
            "MetricConfig", "MetricDropSummary", "MomentumBaseline",
            "OpenQuestion", "QAProbeReport", "QuestionPromptRecord",
            "QuestionScore", "QuestionSet", "QuestionSpec",
            "QuestionTemplate", "ReturnSign", "ScenarioABReport",
            "ToleranceProfile",
            "aggregate_scores", "build_question_set",
            "build_question_sets_multi",
            "normalize_binary", "normalize_direction",
            "run_ab_probe", "sample_dates",
            "score_answer_file", "score_question",
            "serialize_answer_records",
        }
        actual = set(probes_pkg.__all__)
        missing = v2_0_names - actual
        assert not missing, f"v2.0 names dropped: {missing}"

    def test_models_module_still_importable(self):
        # Several v2.0 internal users imported from probes.models
        # directly. The submodule must remain importable.
        m = importlib.import_module("aiphaforge.probes.models")
        for name in ("QAProbeReport", "MetricConfig", "ABScenario",
                     "ToleranceProfile"):
            assert hasattr(m, name)


# ---------- v2.0 Day-One example, unchanged ----------

class TestV2_0_DayOneExample:
    """The v2.0 README's two-paragraph "10-line probe" example,
    mechanically reproduced. None of the v2.0.1 kwargs appear here on
    purpose — if a v2.0 user upgrades the package without changing
    code, this is what their pipeline does.
    """

    def test_v2_0_qa_pipeline_runs_unchanged(self, tmp_path):
        data = _ohlcv(n=50)
        ts_list = sample_dates(data, n=3, seed=0, start=1)
        qs = build_question_set(data, "AAPL", ts_list, [OpenQuestion()])
        # Simulate a "perfect responder" by submitting truth values.
        answers = [
            AnswerRecord(
                question_id=q.question_id,
                raw_answer=str(q.truth_value),
                parsed_answer=q.truth_value,
                parse_status="valid",
            )
            for q in qs
        ]
        ans_path = tmp_path / "answers.jsonl"
        serialize_answer_records(answers, str(ans_path))
        # No provider_config, no manifest — exactly the v2.0 call shape.
        report = score_answer_file(qs, str(ans_path))
        assert report.total_questions == len(qs)
        assert report.valid_answers == len(qs)
        # v2.0 contract: provider_config must NOT appear when none was
        # passed. (M2 collision rule must not silently inject it.)
        assert "provider_config" not in (report.manifest or {})

    def test_v2_0_aggregate_scores_unchanged_signature(self):
        from inspect import signature
        sig = signature(aggregate_scores)
        # The v2.0 signature was (question_set, scores, *, manifest=None).
        # M2 deliberately did NOT add provider_config here.
        params = list(sig.parameters)
        assert params[0] == "question_set"
        assert params[1] == "scores"
        assert "manifest" in sig.parameters
        assert "provider_config" not in sig.parameters

    def test_v2_0_knowledge_probe_score_without_provider_config(self, tmp_path):
        data = _ohlcv(n=40)
        probe = KnowledgeProbe(symbol="X", templates=[OpenQuestion()])
        ts_list = sample_dates(data, n=2, seed=0, start=1)
        qs = probe.build(data, ts_list)
        answers = [
            AnswerRecord(
                question_id=q.question_id,
                raw_answer=str(q.truth_value),
                parsed_answer=q.truth_value,
                parse_status="valid",
            )
            for q in qs
        ]
        ans_path = tmp_path / "answers.jsonl"
        serialize_answer_records(answers, str(ans_path))
        # v2.0 call shape — no provider_config kwarg.
        report = probe.score(str(ans_path), question_set=qs)
        assert report.valid_answers == len(qs)


# ---------- v2.0 ToleranceProfile defaults unchanged ----------

class TestV2_0_ToleranceDefaults:
    def test_v2_0_default_price_tolerance_unchanged(self):
        # The shipped default _PRICE_TOLERANCE governs the OHLC
        # templates' built-in scoring; users who never construct a
        # ToleranceProfile rely on this. v2.0.1 must NOT shift it
        # (the strict variant is opt-in only).
        from aiphaforge.probes.questions import _PRICE_TOLERANCE
        assert _PRICE_TOLERANCE.exact_threshold == pytest.approx(0.005)
        assert _PRICE_TOLERANCE.near_threshold == pytest.approx(0.02)
        assert _PRICE_TOLERANCE.rough_threshold == pytest.approx(0.05)
        assert _PRICE_TOLERANCE.max_range_width == pytest.approx(0.20)


# ---------- v2.0 A/B probe call shape ----------

class TestV2_0_ABCallShape:
    def test_v2_0_run_ab_probe_works_without_v2_0_1_kwargs(self):
        from aiphaforge.probes import (
            ABScenario,
            MACrossBaseline,
            run_ab_probe,
        )
        from aiphaforge.probes.transforms import PriceScale

        data = _ohlcv(n=40)
        scen = ABScenario(
            scenario_id="s", mode="market_level",
            transforms=[PriceScale(factor=2.0)],
        )

        def factory():
            return MACrossBaseline(short=5, long=20)

        # Pure v2.0 call — no agent_determinism_check, no
        # determinism_metrics, no determinism_rel_tol, no provider_config.
        result = run_ab_probe(
            ai_factory=factory, baseline_factory=factory,
            data=data, scenarios=[scen],
            n_repeat=2, seeds=[0, 1],
            metrics=("total_return", "trade_count"),
            min_valid_repeats=1,
            engine_kwargs={"include_benchmark": False},
        )
        # v2.0 fields still present and meaningful.
        m = result.manifest
        assert m["ai_determinism_check_passed"] is True
        assert m["baseline_determinism_check_passed"] is True
        # v2.0.1 r5 default: agent_determinism_check="raw_only" with
        # determinism_profile="auto" must resolve to v2-compatible
        # raw determinism behavior.
        assert m["agent_determinism_check_mode"] == "raw_only"
        assert m["determinism_profile"] == "v2_compat"
        assert m["determinism_metrics"] == ["total_return"]
        assert m["determinism_rel_tol"] == 1e-12

    def test_v2_0_default_resolves_to_v2_compat(self):
        # r5 §7.2: explicit assertion that the canonical resolver
        # returns v2_compat for v2.0 callers.
        from aiphaforge.probes.abtest import resolve_determinism_config
        r = resolve_determinism_config(
            mode="raw_only", profile="auto",
            determinism_metrics=None, determinism_rel_tol=None,
        )
        assert r.profile == "v2_compat"
        assert r.determinism_metrics == ("total_return",)
        assert r.determinism_rel_tol == 1e-12

    def test_v2_0_models_agent_contract_still_order_shape(self):
        # r5 §7.2: existing models.AgentContract must still resolve to
        # the order-shape literal (not shadowed by the new
        # AgentImplementationContract).
        from aiphaforge.probes import models as probes_models
        assert probes_models.AgentContract.__args__ == (
            "signal_only", "market_orders_only", "price_orders_allowed",
        )
