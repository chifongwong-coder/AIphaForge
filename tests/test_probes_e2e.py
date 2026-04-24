"""v2.0.1 M6 — end-to-end smoke combining M1..M5.

Exercises every public surface introduced or touched in v2.0.1 in a
single round-trip:

- ``KnowledgeProbe(symbol="AAPL", templates=DEFAULT_TEMPLATES)``
- ``parse_numeric_answer`` on synthetic LLM-style replies
- ``KnowledgeProbe.score`` with ``provider_config=...``
- ``run_ab_probe`` with ``agent_determinism_check="per_scenario"``,
  ``ToleranceProfile.us_equity_price_strict()`` consumed indirectly via
  the question-set tolerance, and the new ``determinism_metrics`` /
  ``determinism_rel_tol`` knobs

The test asserts all returns are populated, the new manifest fields
exist, and no surprising warnings fire.
"""
from __future__ import annotations

import pytest

from aiphaforge.probes import (
    DEFAULT_TEMPLATES,
    ABScenario,
    AnswerRecord,
    KnowledgeProbe,
    MACrossBaseline,
    OpenQuestion,
    ToleranceProfile,
    parse_numeric_answer,
    run_ab_probe,
    sample_dates,
    serialize_answer_records,
)
from aiphaforge.probes.transforms import PriceScale
from tests.conftest import make_probe_ohlcv as _ohlcv


@pytest.fixture
def fixture_data():
    return _ohlcv(n=80)


# ---------- Q&A end-to-end ----------

class TestQAEndToEnd:
    def test_full_qa_roundtrip_with_provider_config(self, fixture_data, tmp_path):
        # Use the strict-mode preset on one template to exercise the
        # tolerance-preset wiring path.
        templates = list(DEFAULT_TEMPLATES) + [
            OpenQuestion(tolerance=ToleranceProfile.us_equity_price_strict()),
        ]
        probe = KnowledgeProbe(symbol="AAPL", templates=templates)
        ts_list = sample_dates(fixture_data, n=4, seed=0, start=2)
        qs = probe.build(fixture_data, ts_list)
        assert len(qs) > 0

        # Synthesize replies using parse_numeric_answer for numeric
        # questions; pass-through for categorical.
        answers = []
        for q in qs:
            if q.answer_type == "numeric":
                # "$172.34 final" — exercises the parser's $ + suffix path.
                raw = f"${float(q.truth_value):.2f} final"
                parsed = parse_numeric_answer(raw)
            else:
                raw = str(q.truth_value)
                parsed = q.truth_value
            answers.append(AnswerRecord(
                question_id=q.question_id,
                raw_answer=raw, parsed_answer=parsed,
                parse_status="valid",
            ))
        ans_path = tmp_path / "answers.jsonl"
        serialize_answer_records(answers, str(ans_path))

        report = probe.score(
            str(ans_path), question_set=qs,
            provider_config={
                "model": "claude-opus-4-7",
                "temperature": 0.0,
                "prompt_cache_disclosed": True,
                "seed_attestation": False,
            },
        )

        # Q&A scoring populated.
        assert report.total_questions == len(qs)
        assert report.submitted_answers == len(qs)
        assert report.valid_answers >= 1

        # by_template populated (M5).
        assert report.by_template is not None
        assert len(report.by_template) >= 1
        for entry in report.by_template.values():
            assert "exact_rate" in entry

        # provider_config round-tripped (M2).
        pc = report.manifest["provider_config"]
        assert pc["model"] == "claude-opus-4-7"
        assert pc["prompt_cache_disclosed"] is True
        assert pc["seed_attestation"] is False


# ---------- A/B end-to-end ----------

class TestABEndToEnd:
    def test_ab_per_scenario_check_with_new_kwargs(self, fixture_data):
        scen = ABScenario(
            scenario_id="price_scale_2x", mode="market_level",
            transforms=[PriceScale(factor=2.0)],
        )

        def factory():
            return MACrossBaseline(short=5, long=20)

        result = run_ab_probe(
            ai_factory=factory, baseline_factory=factory,
            data=fixture_data, scenarios=[scen],
            n_repeat=2, seeds=[0, 1],
            metrics=("total_return", "trade_count"),
            min_valid_repeats=1,
            agent_determinism_check="per_scenario",
            determinism_metrics=("total_return", "num_trades"),
            determinism_rel_tol=1e-3,
            engine_kwargs={"include_benchmark": False},
            provider_config={
                "model": "claude-opus-4-7", "temperature": 0.0,
            },
        )

        # All scenario reports present.
        assert len(result.scenarios) == 1
        s = result.scenarios[0]
        assert s.scenario_id == "price_scale_2x"

        # r5 manifest keys (M4).
        m = result.manifest
        assert m["agent_determinism_check_mode"] == "per_scenario"
        assert m["determinism_metrics"] == ["total_return", "num_trades"]
        assert m["determinism_rel_tol"] == pytest.approx(1e-3)
        # Per-scenario entry is the rich dict, not a naked bool.
        ps = m["ai_determinism_check_per_scenario"]["price_scale_2x"]
        assert ps["transformed_passed"] is True
        assert ps["transformed_status"] == "passed"
        # Canonical schema present and versioned.
        canonical = m["determinism_check"]
        assert canonical["schema_version"] == "2.0.1"
        assert canonical["mode"] == "per_scenario"
        # Canonical and legacy mirrors agree.
        assert m["determinism_metrics"] == canonical["resolved"]["determinism_metrics"]

        # provider_config in manifest (non-empty bucket).
        assert m["provider_config"]["model"] == "claude-opus-4-7"

        # No transform_detectability_warning expected (PriceScale is
        # not in the detectable list); identical AI/baseline → no
        # capacity_parity warning either; deterministic factory → no
        # determinism failure.
        for w in s.warnings:
            assert "agent_determinism_check_failed" not in w

    def test_ab_run_without_provider_config_does_not_inject_buckets(
        self, fixture_data,
    ):
        # Plan §7.1: empty provider_config buckets must not be added
        # to A/B manifests when the caller did not supply one.
        scen = ABScenario(
            scenario_id="x", mode="market_level",
            transforms=[PriceScale(factor=2.0)],
        )

        def factory():
            return MACrossBaseline(short=5, long=20)

        result = run_ab_probe(
            ai_factory=factory, baseline_factory=factory,
            data=fixture_data, scenarios=[scen],
            n_repeat=2, seeds=[0, 1],
            metrics=("total_return",),
            min_valid_repeats=1,
            engine_kwargs={"include_benchmark": False},
            # No provider_config kwarg.
        )
        assert "provider_config" not in result.manifest
        assert "provider_config_provenance" not in result.manifest
