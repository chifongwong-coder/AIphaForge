"""v2.2.1 audit-fix Commit E + Commit G followups.

Commit E:
- LEAKAGE_INDEX_BUCKET_WEIGHTS public export.
- bucket_delta_ci canonical alias for bucket_delta_tango_ci.

Commit G adds further pre-merge tests in this same file (Arrow dtype,
optional pyarrow, JSON sentinel round-trip, numbered inline comma,
anchor sigma strict).
"""
from __future__ import annotations

import builtins
import types

import numpy as np
import pandas as pd
import pytest

from aiphaforge.probes import LEAKAGE_INDEX_BUCKET_WEIGHTS
from aiphaforge.probes._continuation import (
    ContinuationProbe,
    NextCloseContinuation,
)
from aiphaforge.probes._hash_utils import _canonical_json
from aiphaforge.probes._rank import (
    RankContinuationProbe,
    _is_sequence_of_str,
    parse_rank_answer,
)
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

    def test_bucket_delta_ci_mutation_shows_in_tango_alias(self):
        # v2.2.1 audit-fix Commit J: pin the documented shared-
        # reference semantics. The two field names point at the
        # SAME dict object — mutating one (against the documented
        # "read-only" advice) must show up in the other. The
        # existing test_alias_populated_when_anchor_present asserts
        # `is`-identity; this test exercises the contract via an
        # actual mutation so a future regression that switches
        # __post_init__ to `dict(...)` (copy) is caught.
        report = self._build_report_with_anchor()
        assert report.bucket_delta_ci is not None
        before = dict(report.bucket_delta_ci)
        # Mutate via the canonical name.
        report.bucket_delta_ci["__probe__"] = (9.0, 9.0)
        # The historic name reflects it.
        assert report.bucket_delta_tango_ci is not None
        assert report.bucket_delta_tango_ci["__probe__"] == (9.0, 9.0)
        # Clean up so we don't pollute report state for any
        # later assertion in the build helper's shared fixture.
        del report.bucket_delta_ci["__probe__"]
        assert dict(report.bucket_delta_ci) == before

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


class TestBucketDeltaCiAliasInvariant:
    """v2.2.2 Commit E: hard-enforce the shared-reference invariant.

    The docstring promises both alias fields always point at the same
    dict object. v2.2.1's __post_init__ cross-populated only when
    exactly one side was None; constructing with two distinct dicts
    silently violated the invariant.
    """

    def _valid_base(self) -> KnowledgeCheckReport:
        # Build a real valid report so we can use dataclasses.replace
        # to construct test states.
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
        from aiphaforge.probes.anchors import build_synthetic_anchor
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

    def test_distinct_dicts_for_both_fields_raises_valueerror(self):
        import dataclasses
        base = self._valid_base()
        d1 = {"exact": (-0.1, 0.2)}
        d2 = {"exact": (-0.5, 0.5)}  # DIFFERENT dict object
        with pytest.raises(ValueError) as exc_info:
            dataclasses.replace(
                base,
                bucket_delta_ci=d1,
                bucket_delta_tango_ci=d2,
            )
        # Error message points at the v2.3 migration path.
        assert "bucket_delta_ci" in str(exc_info.value)
        assert "bucket_delta_tango_ci" in str(exc_info.value)

    def test_same_dict_for_both_fields_is_accepted(self):
        import dataclasses
        base = self._valid_base()
        shared = {"exact": (-0.1, 0.2)}
        # Same object identity → no exception.
        replaced = dataclasses.replace(
            base,
            bucket_delta_ci=shared,
            bucket_delta_tango_ci=shared,
        )
        assert replaced.bucket_delta_ci is shared
        assert replaced.bucket_delta_tango_ci is shared


# ---------- Commit G: Arrow dtype acceptance in _is_sequence_of_str ----------


class TestIsSequenceOfStrArrowDtype:
    def test_accepts_pandas_arrow_string_series(self):
        # r7 §15 said path-2 acceptance must cover Arrow-backed
        # string series too — mirroring the StringDtype branch so
        # callers using `dtype=pd.ArrowDtype(pa.string())` (the
        # pandas 2.x recommended path for high-throughput pipelines)
        # don't fall through to the all-False return.
        pa = pytest.importorskip("pyarrow")
        s = pd.Series(
            ["AAPL", "MSFT", "GOOG"],
            dtype=pd.ArrowDtype(pa.string()),
        )
        assert _is_sequence_of_str(s)

    def test_falls_through_when_pyarrow_missing(self, monkeypatch):
        # The Arrow branch wraps `import pyarrow` in try/except so
        # a deployment without pyarrow still loads. Simulate the
        # missing-pyarrow case by hijacking the import: any
        # well-formed list[str] must still be accepted via the
        # sequence path, and an object-dtype Series with all-string
        # values must still be accepted via the object-dtype branch.
        #
        # Scope note: this test covers the OUTER invariant — module
        # load and non-Arrow paths must not eager-import pyarrow.
        # It cannot construct a `pd.ArrowDtype(...)` Series with
        # pyarrow blocked (pandas needs pyarrow to construct the
        # dtype in the first place), so the inner `except ImportError`
        # branch at _rank.py:_is_sequence_of_str — the line that
        # catches a runtime failure of `import pyarrow` while
        # inspecting an ArrowDtype series — is structurally
        # untestable here. That branch is a 1-line defensive guard;
        # treat its correctness as inspection-verified, not
        # test-verified.
        real_import = builtins.__import__

        def _block_pyarrow(name, *args, **kwargs):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ImportError("pyarrow disabled for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_pyarrow)
        # Plain list still accepted regardless of pyarrow state.
        assert _is_sequence_of_str(["AAPL", "MSFT"])
        # Object-dtype Series with all-string values accepted via
        # the object-dtype branch — pyarrow not needed for that.
        s = pd.Series(["AAPL", "MSFT"], dtype=object)
        assert _is_sequence_of_str(s)


# ---------- Commit G: canonical_json sentinel round-trip ----------


class TestCanonicalJsonSentinelRoundTrip:
    def test_positive_infinite_sentinel_serializes_under_allow_nan_false(
        self,
    ):
        # _canonical_json pins allow_nan=False to keep hashes
        # reproducible across releases. The orchestrator stores
        # divide-by-zero z as the literal string sentinels
        # "positive_infinite"/"negative_infinite" instead of float
        # inf so this serialization stays clean. Round-trip both
        # sentinels through _canonical_json + json.loads.
        import json
        payload = {
            "scalar_leakage_index": 0.5,
            "scalar_leakage_index_z": "positive_infinite",
            "nested": {
                "other_z": "negative_infinite",
            },
        }
        encoded = _canonical_json(payload)
        decoded = json.loads(encoded)
        assert decoded["scalar_leakage_index_z"] == "positive_infinite"
        assert decoded["nested"]["other_z"] == "negative_infinite"

    def test_float_inf_still_rejected_under_allow_nan_false(self):
        # Sanity guard: the locked allow_nan=False kwarg must keep
        # rejecting raw float infinities. Otherwise users could
        # silently bypass the sentinel discipline.
        with pytest.raises(ValueError):
            _canonical_json({"z": float("inf")})


# ---------- Commit G: numbered list does NOT split on inline commas ----------


class TestParseRankAnswerNumberedInlineComma:
    def test_inline_comma_in_numbered_does_not_silently_split(self):
        # A numbered-list line that happens to contain a comma
        # (e.g., "1. AAPL, 2. MSFT" was emitted as ONE line by a
        # model that forgot to insert newlines) must NOT silently
        # produce a 2-element ranking — that would invent intent
        # the user never expressed. The numbered regex captures the
        # whole tail as a single token; the result either fails to
        # resolve against expected_symbols (returns None) or yields
        # a single-token ranking that does not equal ["AAPL","MSFT"].
        ans = parse_rank_answer(
            "1. AAPL, 2. MSFT",
            expected_symbols=["AAPL", "MSFT"],
            normalization_preset="passthrough",
        )
        if ans is None:
            # Acceptable: parser refused to commit to an ambiguous
            # interpretation. This is the strongest guarantee.
            return
        # If a RankAnswer was produced, it MUST NOT be the silently
        # split ["AAPL", "MSFT"] interpretation.
        assert tuple(ans.ranking) != ("AAPL", "MSFT")


# ---------- Commit G: continuation anchor sigma test (strict variant) ----------


def _make_continuation_data(
    n: int, sigma: float, seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, sigma, n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    # Scale H/L spread with sigma so per-day Garman-Klass picks up
    # the difference (constant H/L would mask the sigma input).
    intraday_range = sigma * 2.0
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * (1.0 + intraday_range),
            "low": closes * (1.0 - intraday_range),
            "close": closes,
            "volume": [1e6] * n,
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )


def _build_perfect_continuation_attested(question_set):
    return AttestedAnswers(
        answers=tuple(
            AnswerRecord(
                question_id=q.question_id, raw_answer="x",
                parsed_answer=q.truth_value, parse_status="valid",
            )
            for q in question_set
        ),
        parsing_schema_hash="a" * 64,
        parsing_schema_description="x",
        prompt_template_hash="b" * 64,
        prompt_template_description="y",
    )


class TestContinuationAnchorSigmaStrict:
    def test_anchor_sigma_strictly_differs_from_real_sigma(self):
        # Strict variant of the existing wiring test: the original
        # silently `return`-ed when no qid had both sides recorded,
        # which let a regression where the anchor pipeline fails to
        # populate provenance slip through. Here we assert at least
        # one qid has both sides AND the anchor sigma is at least
        # 3× the real sigma (the data is built with σ_anchor ≈ 10×
        # σ_real).
        real_data = _make_continuation_data(n=60, sigma=0.005, seed=1)
        anchor_data = _make_continuation_data(n=60, sigma=0.05, seed=2)
        probe = ContinuationProbe(
            symbol="AAPL", context_bars=20, forward_horizon=1,
            templates=[NextCloseContinuation],
        )
        anchors = list(real_data.index[30:35])
        question_set = probe.build(real_data, anchors)
        attested = _build_perfect_continuation_attested(question_set)
        anchor_qs = probe.build(anchor_data, anchors)
        anchor_attested = _build_perfect_continuation_attested(anchor_qs)
        report = knowledge_check(
            probe, real_data, anchors, attested,
            anchor=anchor_data, anchor_answers=anchor_attested,
            provider_config=_provider_config(),
        )
        prov = report.vol_scale_provenance
        assert prov is not None
        any_both_sides = next(
            (
                qid for qid, sides in prov.items()
                if "real" in sides and "anchor" in sides
            ),
            None,
        )
        # Strict assertion (replaces the silent-return path).
        assert any_both_sides is not None, (
            "expected at least one question to record both real and "
            "anchor sigma in vol_scale_provenance"
        )
        real_sigma = prov[any_both_sides]["real"]["sigma"]
        anchor_sigma = prov[any_both_sides]["anchor"]["sigma"]
        assert anchor_sigma > real_sigma * 3
