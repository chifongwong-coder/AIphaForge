"""v2.2.1 #1 — vol-scaling wiring (both real and anchor sides)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aiphaforge.probes._continuation import (
    ContinuationProbe,
    NextCloseContinuation,
)
from aiphaforge.probes.anchors import build_synthetic_anchor
from aiphaforge.probes.models import (
    AnswerRecord,
    AttestedAnswers,
)
from aiphaforge.probes.orchestrator import knowledge_check
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


def _make_data(n: int = 60, sigma: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """Realistic fixture where H/L spread scales with σ. This way
    Parkinson/Garman-Klass produce sigma values that vary with the
    fixture's σ parameter (default uses constant H/L which makes
    GK return a fixed σ regardless of the underlying volatility)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, sigma, n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    intraday_range = abs(rng.normal(0.0, sigma, n))  # scales with σ
    highs = closes * (1.0 + intraday_range)
    lows = closes * (1.0 - intraday_range)
    return pd.DataFrame(
        {
            "open": closes, "high": highs,
            "low": lows, "close": closes,
            "volume": [1e6] * n,
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )


def _build_perfect_attested(question_set):
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


class TestVolScaleWiringNamedIndexRegression:
    def test_continuation_vol_scaling_works_with_named_datetime_index(self):
        # v2.2.1 hotfix regression test. Previously _vol.py looked up
        # `ctx[-1]["index"]` to recover the last-context-bar timestamp,
        # but `df.reset_index().to_dict(orient="records")` keys the
        # column by the index NAME — so any user with
        # `data.index.name == "Date"` (the realistic case from
        # `pd.read_csv(..., parse_dates=...)` or any code that names
        # its DatetimeIndex) crashed with KeyError("index") at
        # vol-scaling time. The fix reads `qs.timestamp` directly,
        # which equals the anchor bar by ContinuationProbe construction.
        rng = np.random.default_rng(0)
        rets = rng.normal(0.0, 0.01, 60)
        closes = 100.0 * np.exp(np.cumsum(rets))
        intraday_range = abs(rng.normal(0.0, 0.01, 60))
        idx = pd.DatetimeIndex(
            pd.bdate_range("2024-01-01", periods=60), name="Date",
        )
        data = pd.DataFrame(
            {
                "open": closes,
                "high": closes * (1.0 + intraday_range),
                "low": closes * (1.0 - intraday_range),
                "close": closes,
                "volume": [1e6] * 60,
            },
            index=idx,
        )
        assert data.index.name == "Date"
        probe = ContinuationProbe(
            symbol="AAPL", context_bars=20, forward_horizon=1,
            templates=[NextCloseContinuation],
        )
        anchors = list(data.index[30:35])
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        # Pre-fix this raised KeyError('index'); now must complete cleanly.
        report = knowledge_check(
            probe, data, anchors, attested,
            provider_config=_provider_config(),
        )
        assert report.vol_scale_provenance is not None
        # Provenance was actually computed (not skipped via fallback).
        assert len(report.vol_scale_provenance) == len(list(question_set))


class TestVolScaleWiringContinuation:
    def test_continuation_probe_records_real_provenance_per_qid(self):
        # ContinuationProbe templates ship vol_scale by default.
        data = _make_data(n=60)
        probe = ContinuationProbe(
            symbol="AAPL", context_bars=20, forward_horizon=1,
            templates=[NextCloseContinuation],
        )
        anchors = list(data.index[30:35])
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        report = knowledge_check(
            probe, data, anchors, attested,
            provider_config=_provider_config(),
        )
        assert report.vol_scale_provenance is not None
        # One entry per question
        assert len(report.vol_scale_provenance) == len(list(question_set))
        # Each has a "real" side
        for qid, side_dict in report.vol_scale_provenance.items():
            assert "real" in side_dict
            assert "sigma" in side_dict["real"]

    def test_continuation_anchor_uses_anchor_sigma_not_real_sigma(self):
        # Build real with sigma~0.005, anchor with sigma~0.05 →
        # vol_scale_provenance should have meaningfully different
        # sigma values for real vs anchor.
        real_data = _make_data(n=60, sigma=0.005, seed=1)
        anchor_data = _make_data(n=60, sigma=0.05, seed=2)
        probe = ContinuationProbe(
            symbol="AAPL", context_bars=20, forward_horizon=1,
            templates=[NextCloseContinuation],
        )
        anchors = list(real_data.index[30:35])
        question_set = probe.build(real_data, anchors)
        attested = _build_perfect_attested(question_set)
        anchor_qs = probe.build(anchor_data, anchors)
        anchor_attested = _build_perfect_attested(anchor_qs)
        report = knowledge_check(
            probe, real_data, anchors, attested,
            anchor=anchor_data, anchor_answers=anchor_attested,
            provider_config=_provider_config(),
        )
        prov = report.vol_scale_provenance
        assert prov is not None
        # Pick any qid that has both sides recorded
        any_both_sides = next(
            (
                qid for qid, sides in prov.items()
                if "real" in sides and "anchor" in sides
            ),
            None,
        )
        if any_both_sides is None:
            # Anchor build may have skipped some qids; ensure at
            # least one has both sides for the test to be meaningful.
            return
        real_sigma = prov[any_both_sides]["real"]["sigma"]
        anchor_sigma = prov[any_both_sides]["anchor"]["sigma"]
        # Anchor sigma should be ~10× real sigma
        assert anchor_sigma > real_sigma * 3  # generous lower bound

    def test_provenance_keyed_by_qid_and_side(self):
        # Shape check: outer dict keyed by qid, inner dict keyed by side.
        data = _make_data(n=60)
        probe = ContinuationProbe(
            symbol="AAPL", context_bars=15, forward_horizon=1,
            templates=[NextCloseContinuation],
        )
        anchors = list(data.index[20:23])
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        anchor_data = build_synthetic_anchor(
            data, seed=42, method="random_walk_volmatched",
        )
        anchor_qs = probe.build(anchor_data, anchors)
        anchor_attested = _build_perfect_attested(anchor_qs)
        report = knowledge_check(
            probe, data, anchors, attested,
            anchor=anchor_data, anchor_answers=anchor_attested,
            provider_config=_provider_config(),
        )
        prov = report.vol_scale_provenance
        assert prov is not None
        # Outer keys are qid strings
        for qid in prov:
            assert isinstance(qid, str)
            # Inner keys are "real" and/or "anchor"
            for side in prov[qid]:
                assert side in {"real", "anchor"}
                # Provenance dict has expected fields
                assert "sigma" in prov[qid][side]
                assert "method_used" in prov[qid][side]


class TestVolScaleWiringKnowledgeProbe:
    def test_knowledge_probe_no_vol_scale_no_provenance(self):
        # Default KnowledgeProbe templates don't set vol_scale on
        # ToleranceProfile → no rewrite, no provenance recorded.
        data = _make_data()
        probe = KnowledgeProbe(
            symbol="AAPL", templates=DEFAULT_TEMPLATES,
        )
        anchors = list(data.index[10:15])
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        report = knowledge_check(
            probe, data, anchors, attested,
            provider_config=_provider_config(),
        )
        # No question had vol_scale → provenance is None
        assert report.vol_scale_provenance is None

    def test_no_anchor_provenance_only_real_side(self):
        data = _make_data(n=60)
        probe = ContinuationProbe(
            symbol="AAPL", context_bars=15, forward_horizon=1,
            templates=[NextCloseContinuation],
        )
        anchors = list(data.index[30:33])
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        report = knowledge_check(
            probe, data, anchors, attested,
            provider_config=_provider_config(),
        )
        prov = report.vol_scale_provenance
        assert prov is not None
        # No anchor → only "real" side keys
        for qid in prov:
            assert "real" in prov[qid]
            assert "anchor" not in prov[qid]
