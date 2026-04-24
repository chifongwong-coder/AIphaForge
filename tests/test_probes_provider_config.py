"""v2.0.1 M2 — tests for provider_config plumbing on Q&A scoring."""
from __future__ import annotations

import pytest

from aiphaforge.probes import (
    AnswerRecord,
    KnowledgeProbe,
    OpenQuestion,
    build_question_set,
    sample_dates,
    score_answer_file,
    serialize_answer_records,
)
from aiphaforge.probes.models import RECOMMENDED_PROVIDER_CONFIG_KEYS
from aiphaforge.probes.scoring import _merge_provider_config
from tests.conftest import make_probe_ohlcv as _ohlcv

# ---------- Recommended-keys schema ----------

class TestRecommendedKeys:
    def test_v2_0_keys_still_present(self):
        # Backward compat: every v2.0 key must still be there.
        for key in (
            "model", "snapshot_id", "temperature", "top_p",
            "max_tokens", "seed", "prompt_template_hash",
            "system_prompt_hash", "tool_policy", "notes",
        ):
            assert key in RECOMMENDED_PROVIDER_CONFIG_KEYS

    def test_v2_0_1_additions_present(self):
        # All 9 new keys identified by the LLM-eval review.
        for key in (
            "system_fingerprint", "cache_control", "stop_sequences",
            "response_format", "n_intra_replicates", "pooling_strategy",
            "n_retries_per_question", "seed_attestation",
            "prompt_cache_disclosed",
        ):
            assert key in RECOMMENDED_PROVIDER_CONFIG_KEYS


# ---------- _merge_provider_config rules ----------

class TestMergeRule:
    def test_both_none_returns_none(self):
        result, prov = _merge_provider_config(None, None)
        assert result is None
        assert prov is None

    def test_only_kwarg_populates_manifest(self):
        result, prov = _merge_provider_config(
            None, {"model": "X", "temperature": 0.0},
        )
        assert result["provider_config"] == {"model": "X", "temperature": 0.0}
        assert prov == {"model": "kwarg", "temperature": "kwarg"}

    def test_only_manifest_pc_passes_through(self):
        result, prov = _merge_provider_config(
            {"provider_config": {"model": "X"}}, None,
        )
        assert result["provider_config"] == {"model": "X"}
        assert prov == {"model": "manifest"}

    def test_partial_overlap_merges_silently(self):
        # Common case: stable IDs in manifest, per-run knobs in kwarg.
        result, prov = _merge_provider_config(
            {"provider_config": {"model": "X", "snapshot_id": "abc"}},
            {"temperature": 0.0, "seed": 42},
        )
        assert result["provider_config"] == {
            "model": "X", "snapshot_id": "abc",
            "temperature": 0.0, "seed": 42,
        }
        assert prov == {
            "model": "manifest", "snapshot_id": "manifest",
            "temperature": "kwarg", "seed": "kwarg",
        }

    def test_same_key_same_value_marked_both(self):
        result, prov = _merge_provider_config(
            {"provider_config": {"model": "X"}},
            {"model": "X"},
        )
        assert result["provider_config"] == {"model": "X"}
        assert prov == {"model": "both"}

    def test_same_key_different_value_raises(self):
        with pytest.raises(ValueError, match="provider_config collision"):
            _merge_provider_config(
                {"provider_config": {"model": "X"}},
                {"model": "Y"},
            )

    def test_collision_message_names_both_values(self):
        with pytest.raises(ValueError) as exc_info:
            _merge_provider_config(
                {"provider_config": {"temperature": 0.0}},
                {"temperature": 0.5},
            )
        msg = str(exc_info.value)
        assert "temperature" in msg
        assert "0.0" in msg
        assert "0.5" in msg

    def test_provenance_recorded_in_manifest(self):
        result, _ = _merge_provider_config(
            {"provider_config": {"model": "X"}},
            {"temperature": 0.0},
        )
        assert "provider_config_provenance" in result
        assert result["provider_config_provenance"] == {
            "model": "manifest", "temperature": "kwarg",
        }


# ---------- v2.0.1 r5 §3.4: empty-bucket rule ----------

class TestEmptyBucketRule:
    def test_manifest_only_does_not_create_empty_buckets(self):
        # The classic gap: caller passes a manifest with no provider
        # config and no kwarg; the merge result must not inject
        # provider_config={} or provider_config_provenance={}.
        result, prov = _merge_provider_config({"run_id": "abc"}, None)
        assert result == {"run_id": "abc"}
        assert "provider_config" not in result
        assert "provider_config_provenance" not in result
        assert prov is None

    def test_empty_kwarg_does_not_create_empty_buckets(self):
        result, prov = _merge_provider_config({"run_id": "abc"}, {})
        assert "provider_config" not in result
        assert "provider_config_provenance" not in result
        assert prov is None

    def test_input_empty_bucket_is_stripped_from_output(self):
        # Plan §3.4 explicit case: literal {"provider_config": {}} in
        # the input manifest must not round-trip into the output.
        result, prov = _merge_provider_config(
            {"run_id": "abc", "provider_config": {}}, None,
        )
        assert result == {"run_id": "abc"}
        assert prov is None

    def test_input_empty_bucket_with_empty_kwarg_stripped(self):
        result, prov = _merge_provider_config(
            {"run_id": "abc", "provider_config": {}}, {},
        )
        assert result == {"run_id": "abc"}
        assert prov is None

    def test_input_empty_bucket_and_provenance_stripped(self):
        # Idempotency: a manifest that already carries both literal
        # empty buckets must come back without them.
        result, prov = _merge_provider_config(
            {
                "run_id": "abc",
                "provider_config": {},
                "provider_config_provenance": {},
            },
            None,
        )
        assert result == {"run_id": "abc"}
        assert prov is None

    def test_non_empty_kwarg_still_writes_buckets(self):
        # Sanity: the strip rule must not break the populated path.
        result, prov = _merge_provider_config(
            {"run_id": "abc"}, {"model": "X"},
        )
        assert result["provider_config"] == {"model": "X"}
        assert result["provider_config_provenance"] == {"model": "kwarg"}


# ---------- score_answer_file integration ----------

class TestScoreAnswerFile:
    def _round_trip(self, tmp_path, *, manifest=None, provider_config=None):
        data = _ohlcv(n=60)
        ts_list = sample_dates(data, n=3, seed=0, start=1)
        qs = build_question_set(data, "AAPL", ts_list, [OpenQuestion()])
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
        return score_answer_file(
            qs, str(ans_path),
            manifest=manifest, provider_config=provider_config,
        )

    def test_provider_config_round_trips_through_manifest(self, tmp_path):
        report = self._round_trip(
            tmp_path, provider_config={
                "model": "claude-opus-4-7",
                "temperature": 0.0,
                "prompt_cache_disclosed": True,
            },
        )
        pc = report.manifest["provider_config"]
        assert pc["model"] == "claude-opus-4-7"
        assert pc["temperature"] == 0.0
        assert pc["prompt_cache_disclosed"] is True

    def test_no_provider_config_no_field_added(self, tmp_path):
        report = self._round_trip(tmp_path)
        # Manifest may be {} or have no provider_config key.
        assert "provider_config" not in (report.manifest or {})

    def test_collision_raises_in_score_answer_file(self, tmp_path):
        with pytest.raises(ValueError, match="provider_config collision"):
            self._round_trip(
                tmp_path,
                manifest={"provider_config": {"model": "X"}},
                provider_config={"model": "Y"},
            )


# ---------- KnowledgeProbe.score plumbing ----------

class TestKnowledgeProbeScorePlumbing:
    def test_provider_config_kwarg(self, tmp_path):
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
        report = probe.score(
            str(ans_path), question_set=qs,
            provider_config={"model": "stub", "seed_attestation": False},
        )
        assert report.manifest["provider_config"]["model"] == "stub"
        assert report.manifest["provider_config"]["seed_attestation"] is False


# ---------- aggregate_scores deliberately UNCHANGED ----------

class TestAggregateScoresNoProviderConfig:
    def test_aggregate_scores_does_not_take_provider_config(self):
        # Per the v2.0.1 plan: provider_config plumbing is on the user-
        # facing entry points (score_answer_file + KnowledgeProbe.score),
        # not on the inner aggregation. Users wanting it on aggregate
        # stuff it into manifest themselves.
        from inspect import signature

        from aiphaforge.probes.scoring import aggregate_scores
        sig = signature(aggregate_scores)
        assert "provider_config" not in sig.parameters
