"""v2.2.1 audit-fix Commit E + Commit G followups.

Commit E:
- LEAKAGE_INDEX_BUCKET_WEIGHTS public export.
- bucket_delta_ci canonical alias for bucket_delta_tango_ci.

Commit G adds further pre-merge tests in this same file (Arrow dtype,
JSON sentinel round-trip, inline comma, anchor sigma strict).
"""
from __future__ import annotations

import types

import pandas as pd

from aiphaforge.probes import LEAKAGE_INDEX_BUCKET_WEIGHTS
from aiphaforge.probes._rank import RankContinuationProbe
from aiphaforge.probes.anchors import build_synthetic_anchor
from aiphaforge.probes.models import AnswerRecord, AttestedAnswers
from aiphaforge.probes.orchestrator import (
    _BUCKET_ORDINAL_WEIGHTS,
    KnowledgeCheckReport,
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


# ---------- Commit E: LEAKAGE_INDEX_BUCKET_WEIGHTS export ----------


class TestLeakageIndexBucketWeightsExport:
    def test_constant_is_exported_at_package_level(self):
        # Paper authors should be able to import the locked weights
        # rather than transcribe them. The export name is descriptive
        # and includes the version-locked semantics.
        assert LEAKAGE_INDEX_BUCKET_WEIGHTS["exact"] == 4.0
        assert LEAKAGE_INDEX_BUCKET_WEIGHTS["near"] == 3.0
        assert LEAKAGE_INDEX_BUCKET_WEIGHTS["rough"] == 2.0
        assert LEAKAGE_INDEX_BUCKET_WEIGHTS["miss"] == 1.0
        assert LEAKAGE_INDEX_BUCKET_WEIGHTS["invalid"] == 0.0

    def test_constant_is_immutable_mapping_proxy(self):
        # The locked weights are §14-versioned; mutating them in
        # place would silently invalidate cross-release comparisons.
        assert isinstance(
            LEAKAGE_INDEX_BUCKET_WEIGHTS, types.MappingProxyType,
        )

    def test_internal_alias_points_to_same_object(self):
        # _BUCKET_ORDINAL_WEIGHTS is the historic internal name; it
        # must remain a reference to the same proxy so internal
        # callers don't fork a private copy.
        assert _BUCKET_ORDINAL_WEIGHTS is LEAKAGE_INDEX_BUCKET_WEIGHTS


# ---------- Commit E: bucket_delta_ci alias ----------


class TestBucketDeltaCiAlias:
    def _build_report_with_anchor(self) -> KnowledgeCheckReport:
        data, ts = _multi_data()
        probe = RankContinuationProbe(
            symbols=["A", "B", "C"], context_bars=5,
            forward_horizon=1,
        )
        qs = probe.build(data, [ts])
        qid = list(qs)[0].question_id
        real = AttestedAnswers(
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
        return knowledge_check(
            probe, data, [ts], real,
            anchor=anchor_data["A"],
            anchor_answers=anchor_attested,
            provider_config=_provider_config(),
        )

    def test_alias_populated_when_anchor_present(self):
        report = self._build_report_with_anchor()
        # Anchor is present → both fields should be populated and
        # point to the SAME dict object (not just equal dicts).
        assert report.bucket_delta_ci is not None
        assert report.bucket_delta_tango_ci is not None
        assert report.bucket_delta_ci is report.bucket_delta_tango_ci

    def test_alias_both_none_when_no_anchor(self):
        # Without anchor, both should remain None — the alias only
        # mirrors a present value.
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
        assert report.bucket_delta_ci is None
        assert report.bucket_delta_tango_ci is None

    def test_alias_back_fills_historic_name_from_canonical(self):
        # Forward-compat: if a downstream caller produces a report
        # carrying only the canonical bucket_delta_ci (e.g., a v2.2.2
        # path that drops the "tango" suffix), __post_init__ should
        # back-fill bucket_delta_tango_ci so existing readers keep
        # working. Use dataclasses.replace to trigger __post_init__
        # on a fully-populated report.
        import dataclasses
        base = self._build_report_with_anchor()
        ci = base.bucket_delta_ci
        # Clear the historic name; expect back-fill via post-init.
        replaced = dataclasses.replace(
            base,
            bucket_delta_tango_ci=None,
            bucket_delta_ci=ci,
        )
        assert replaced.bucket_delta_ci is ci
        assert replaced.bucket_delta_tango_ci is ci
