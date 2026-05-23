"""v2.2 M6 tests — knowledge_check orchestrator + helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.probes._continuation import (
    ContinuationProbe,
    NextCloseContinuation,
)
from aiphaforge.probes.anchors import build_synthetic_anchor
from aiphaforge.probes.models import AnswerRecord, AttestedAnswers
from aiphaforge.probes.orchestrator import (
    KnowledgeCheckReport,
    compute_effective_rate,
    compute_refusal_rate,
    knowledge_check,
    looks_like_refusal,
    sign_test_p,
    wald_paired_diff_ci,
)
from aiphaforge.probes.questions import (
    DEFAULT_TEMPLATES,
    KnowledgeProbe,
)


def _provider_config(**overrides):
    base = {
        "temperature": 0.0,
        "top_p": 1.0,
        "model_id": "claude-opus-4-7",
        "model_version": "20260101",
        "max_output_tokens": 256,
        "tokenizer_id": "anthropic-tokenizer-v2",
        "seed": 42,
        "reasoning_effort": None,
        "stop_sequences": None,
    }
    base.update(overrides)
    return base


def _make_real_data(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01,
            "low": closes * 0.99, "close": closes,
            "volume": [1e6] * n,
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )


# ---------- looks_like_refusal ----------


class TestLooksLikeRefusal:
    # v2.8.3 Commit D (M9): leading window widened 50 → 80 characters.
    def test_keyword_in_leading_80_chars_returns_true(self):
        assert looks_like_refusal("I don't have access to that data.")

    def test_keyword_at_position_66_now_caught_under_widen_80(self):
        # "i don't know" starts at position 66 in the prefix below.
        # Under the prior 50-char window this missed the refusal;
        # under the M9-widened 80-char window it is correctly caught.
        prefix = "The price was approximately 100. " * 2  # 66 chars
        text = prefix + "I don't know more."
        assert len(prefix) == 66
        assert looks_like_refusal(text)

    def test_keyword_past_position_80_returns_false(self):
        # Push the keyword far enough that it lies past the widened
        # 80-char window. The digit ratio is intentionally high so
        # the length-based fallback also does not trigger.
        prefix = "Price 100.55 close 100.60 open 100.40 vol 1000. " * 2
        text = prefix + "I don't know more."
        assert len(prefix) > 80
        assert not looks_like_refusal(text)

    def test_in_quote_refusal_phrase_past_window_still_excluded(self):
        # Regression: the leading-window restriction's whole purpose
        # is to avoid flagging numeric answers that quote a
        # refusal-shaped phrase mid-reply. Widening to 80 must not
        # break that property — a quoted refusal that begins after
        # position 80 inside a digit-rich answer must still return
        # False.
        prefix = "Open 100.55, high 100.80, low 100.20, close 100.65. " * 2
        text = (
            prefix
            + "The article says 'I don't have access' but the "
            + "close was 100.65."
        )
        assert len(prefix) > 80
        assert not looks_like_refusal(text)

    def test_short_numeric_answer_returns_false(self):
        assert not looks_like_refusal("147.20")

    def test_long_prose_with_few_digits_returns_true(self):
        text = (
            "This is a long answer with very few digits "
            "and lots of words explaining context that "
            "doesn't really commit to a number at all."
        ) * 2
        assert looks_like_refusal(text)

    def test_no_longer_triggers_on_as_an_ai(self):
        # r5 had "as an ai" as a keyword; r6 removed it.
        assert not looks_like_refusal("As an AI, I estimate 147.20")

    def test_im_not_sure_in_leading_chars_returns_true(self):
        assert looks_like_refusal("I'm not sure, but try 100")

    def test_empty_string_returns_false(self):
        assert not looks_like_refusal("")

    def test_synthetic_ticker_phrase(self):
        assert looks_like_refusal(
            "appears to be a synthetic ticker, no data"
        )


# ---------- sign_test_p ----------


class TestSignTestP:
    def test_n_zero_returns_one(self):
        p, variant = sign_test_p(0, 0)
        assert p == 1.0
        assert variant == "trivial_n_zero"

    def test_small_n_uses_exact_binomial(self):
        p, variant = sign_test_p(5, 5)
        assert variant == "exact_binomial"
        assert 0 <= p <= 1

    def test_n_25_uses_continuity_correction(self):
        p, variant = sign_test_p(15, 10)
        assert variant == "normal_continuity_corrected"
        assert 0 <= p <= 1

    def test_n_40_uses_normal_approx(self):
        p, variant = sign_test_p(25, 15)
        assert variant == "normal_approximation"

    def test_extreme_imbalance_small_p(self):
        p, _ = sign_test_p(20, 0)
        assert p < 1e-3


# ---------- wald_paired_diff_ci ----------


class TestTangoPairedDiffCi:
    def test_perfect_agreement_returns_point_zero(self):
        lo, hi = wald_paired_diff_ci(
            n_both=10, n_real_only=0, n_anchor_only=0, n_neither=20,
        )
        assert lo == 0.0
        assert hi == 0.0

    def test_empty_bucket_raises(self):
        with pytest.raises(ValueError, match="empty bucket"):
            wald_paired_diff_ci(0, 0, 0, 0)

    def test_real_advantage_yields_positive_ci(self):
        # 20 pairs where real succeeds 18, anchor succeeds 2
        lo, hi = wald_paired_diff_ci(
            n_both=2, n_real_only=16, n_anchor_only=0, n_neither=2,
        )
        assert lo > 0  # CI excludes zero on the positive side


# ---------- compute_effective_rate / refusal_rate ----------


class TestEffectiveRate:
    def test_all_valid_no_refusals_yields_one(self):
        recs = [
            AnswerRecord(
                question_id=f"q{i}", raw_answer="100",
                parsed_answer=100.0, parse_status="valid",
            )
            for i in range(5)
        ]
        assert compute_effective_rate(recs) == 1.0
        assert compute_refusal_rate(recs) == 0.0

    def test_paragraph_refusal_with_stray_number_caught(self):
        recs = [
            AnswerRecord(
                question_id="q0",
                raw_answer=(
                    "I don't have access to data about SYN_AB12CD "
                    "because it appears to be a synthetic ticker "
                    "generated on 2026-04-26."
                ),
                parsed_answer=2026.0,  # spurious extracted number
                parse_status="valid",
            )
        ]
        # parsed_answer is "valid" but raw_answer triggers refusal
        # heuristic → effective_rate = 0
        assert compute_effective_rate(recs) == 0.0
        assert compute_refusal_rate(recs) == 1.0

    def test_joint_not_product(self):
        # 10 records: 5 valid+non-refusal, 5 invalid+refusal.
        # Product of marginals: 0.5 * 0.5 = 0.25
        # Joint: P(valid AND not refused) = 5/10 = 0.5
        recs = []
        for i in range(5):
            recs.append(AnswerRecord(
                question_id=f"q{i}", raw_answer="100",
                parsed_answer=100.0, parse_status="valid",
            ))
        for i in range(5, 10):
            recs.append(AnswerRecord(
                question_id=f"q{i}",
                raw_answer="I don't know the answer",
                parsed_answer=None, parse_status="invalid",
            ))
        eff = compute_effective_rate(recs)
        # Product would be: parse_success(0.5) * (1-refusal(0.5))
        #                 = 0.5 * 0.5 = 0.25
        # Joint: 5/10 = 0.5
        assert eff == 0.5


# ---------- knowledge_check end-to-end (KnowledgeProbe) ----------


def _build_perfect_attested(question_set):
    recs = tuple(
        AnswerRecord(
            question_id=q.question_id, raw_answer="x",
            parsed_answer=q.truth_value, parse_status="valid",
        )
        for q in question_set
    )
    return AttestedAnswers(
        answers=recs,
        parsing_schema_hash="a" * 64,
        parsing_schema_description="oracle",
        prompt_template_hash="b" * 64,
        prompt_template_description="oracle prompt",
    )


class TestKnowledgeCheckRealOnly:
    def test_round_trip(self):
        data = _make_real_data()
        probe = KnowledgeProbe(symbol="AAPL",
                                templates=DEFAULT_TEMPLATES)
        anchors = list(data.index[10:15])
        # Build once to get truth values, then construct attested
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        report = knowledge_check(
            probe, data, anchors, attested,
            provider_config=_provider_config(),
        )
        assert isinstance(report, KnowledgeCheckReport)
        assert report.probe_kind == "knowledge"
        assert report.anchor_score is None
        assert report.anchor_validity == "NO_ANCHOR"
        assert report.parsing_schema_hash == "a" * 64
        assert report.prompt_template_hash == "b" * 64

    def test_carries_non_transitivity_note_in_notes(self):
        data = _make_real_data()
        probe = KnowledgeProbe(symbol="AAPL",
                                templates=DEFAULT_TEMPLATES)
        anchors = list(data.index[10:15])
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        report = knowledge_check(
            probe, data, anchors, attested,
            provider_config=_provider_config(),
        )
        assert any(
            "non-transitivity" in n.lower() for n in report.notes
        )

    def test_carries_uuid_and_wall_clock(self):
        data = _make_real_data()
        probe = KnowledgeProbe(symbol="AAPL",
                                templates=DEFAULT_TEMPLATES)
        anchors = list(data.index[10:15])
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        report = knowledge_check(
            probe, data, anchors, attested,
            provider_config=_provider_config(),
        )
        assert len(report.report_uuid) == 36  # UUID4 string
        assert "T" in report.wall_clock_utc  # ISO-8601


# ---------- knowledge_check provider_config validation ----------


class TestProviderConfigValidation:
    @pytest.fixture
    def common_args(self):
        data = _make_real_data()
        probe = KnowledgeProbe(symbol="AAPL",
                                templates=DEFAULT_TEMPLATES)
        anchors = list(data.index[10:15])
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        return dict(
            probe=probe, data=data, timestamps=anchors,
            answers=attested,
        )

    def test_refuses_missing_temperature(self, common_args):
        cfg = _provider_config()
        del cfg["temperature"]
        with pytest.raises(ValueError, match="missing required keys"):
            knowledge_check(**common_args, provider_config=cfg)

    def test_refuses_missing_seed(self, common_args):
        cfg = _provider_config()
        del cfg["seed"]
        with pytest.raises(ValueError, match="missing required keys"):
            knowledge_check(**common_args, provider_config=cfg)

    def test_refuses_missing_reasoning_effort(self, common_args):
        cfg = _provider_config()
        del cfg["reasoning_effort"]
        with pytest.raises(ValueError, match="missing required keys"):
            knowledge_check(**common_args, provider_config=cfg)

    def test_refuses_missing_stop_sequences(self, common_args):
        cfg = _provider_config()
        del cfg["stop_sequences"]
        with pytest.raises(ValueError, match="missing required keys"):
            knowledge_check(**common_args, provider_config=cfg)

    def test_refuses_missing_provider_config_entirely(self, common_args):
        with pytest.raises(ValueError, match="provider_config"):
            knowledge_check(**common_args, provider_config=None)

    def test_nullable_seed_can_be_none(self, common_args):
        cfg = _provider_config(seed=None)
        # Should not raise
        report = knowledge_check(**common_args, provider_config=cfg)
        assert isinstance(report, KnowledgeCheckReport)

    def test_typed_none_for_reasoning_effort_accepted(self, common_args):
        cfg = _provider_config(reasoning_effort=None)
        report = knowledge_check(**common_args, provider_config=cfg)
        assert isinstance(report, KnowledgeCheckReport)

    def test_n_a_string_not_accepted_for_reasoning_effort(self, common_args):
        cfg = _provider_config(reasoning_effort=123)  # wrong type
        with pytest.raises(TypeError, match="reasoning_effort"):
            knowledge_check(**common_args, provider_config=cfg)


# ---------- knowledge_check refuses bad combos ----------


class TestKnowledgeCheckBadCombos:
    def _setup(self):
        data = _make_real_data()
        probe = KnowledgeProbe(symbol="AAPL",
                                templates=DEFAULT_TEMPLATES)
        anchors = list(data.index[10:15])
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        return data, probe, anchors, attested

    def test_anchor_without_anchor_answers_raises(self):
        data, probe, anchors, attested = self._setup()
        anchor_data = build_synthetic_anchor(
            data, seed=1, method="random_walk_volmatched",
        )
        with pytest.raises(ValueError, match="both"):
            knowledge_check(
                probe, data, anchors, attested,
                anchor=anchor_data, anchor_answers=None,
                provider_config=_provider_config(),
            )

    def test_n_bootstrap_kwargs_removed_in_v2_2_1(self):
        # v2.2.1 #7: n_anchor_bootstrap / bootstrap_seed /
        # bootstrap_unit removed from knowledge_check signature.
        # The dead parameters were never wired to any actual
        # bootstrap loop. Passing them now raises TypeError
        # (unexpected keyword argument).
        data, probe, anchors, attested = self._setup()
        with pytest.raises(TypeError, match="n_anchor_bootstrap"):
            knowledge_check(
                probe, data, anchors, attested,
                provider_config=_provider_config(),
                n_anchor_bootstrap=10,  # type: ignore[call-arg]
            )

    def test_bootstrap_seed_kwarg_removed_in_v2_2_1(self):
        data, probe, anchors, attested = self._setup()
        with pytest.raises(TypeError, match="bootstrap_seed"):
            knowledge_check(
                probe, data, anchors, attested,
                provider_config=_provider_config(),
                bootstrap_seed=42,  # type: ignore[call-arg]
            )

    def test_overlapping_exemplar_pairs_raises(self):
        data, probe, anchors, attested = self._setup()
        # An exemplar at one of the test anchors → overlap
        with pytest.raises(ValueError, match="overlap"):
            knowledge_check(
                probe, data, anchors, attested,
                provider_config=_provider_config(),
                exemplar_pairs=[("AAPL", anchors[0])],
            )

    def test_disjoint_exemplar_pairs_accepted(self):
        data, probe, anchors, attested = self._setup()
        # Different symbol or different timestamp → ok
        report = knowledge_check(
            probe, data, anchors, attested,
            provider_config=_provider_config(),
            exemplar_pairs=[("OTHER_SYMBOL", anchors[0])],
        )
        assert isinstance(report, KnowledgeCheckReport)


# ---------- knowledge_check with anchor ----------


class TestKnowledgeCheckWithAnchor:
    def test_anchor_emits_per_bucket_delta(self):
        data = _make_real_data(n=80)
        probe = KnowledgeProbe(symbol="AAPL",
                                templates=DEFAULT_TEMPLATES)
        anchors = list(data.index[10:18])
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        anchor_data = build_synthetic_anchor(
            data, seed=3, method="random_walk_volmatched",
        )
        anchor_qs = probe.build(anchor_data, anchors)
        anchor_attested = _build_perfect_attested(anchor_qs)
        report = knowledge_check(
            probe, data, anchors, attested,
            anchor=anchor_data, anchor_answers=anchor_attested,
            provider_config=_provider_config(),
        )
        if report.anchor_validity == "OK":
            assert report.bucket_delta is not None
            assert "exact" in report.bucket_delta
            assert report.bucket_delta_ci is not None

    def test_refusal_suspected_when_anchor_all_refusals(self):
        data = _make_real_data(n=80)
        probe = KnowledgeProbe(symbol="AAPL",
                                templates=DEFAULT_TEMPLATES)
        anchors = list(data.index[10:15])
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        anchor_data = build_synthetic_anchor(
            data, seed=4, method="random_walk_volmatched",
        )
        anchor_qs = probe.build(anchor_data, anchors)
        # Build all-refusal anchor answers
        anchor_recs = tuple(
            AnswerRecord(
                question_id=q.question_id,
                raw_answer="I don't have access to that data",
                parsed_answer=None, parse_status="invalid",
            )
            for q in anchor_qs
        )
        anchor_attested = AttestedAnswers(
            answers=anchor_recs,
            parsing_schema_hash="a" * 64,
            parsing_schema_description="x",
            prompt_template_hash="b" * 64,
            prompt_template_description="y",
        )
        report = knowledge_check(
            probe, data, anchors, attested,
            anchor=anchor_data, anchor_answers=anchor_attested,
            provider_config=_provider_config(),
        )
        assert report.anchor_validity == "REFUSAL_SUSPECTED"
        assert report.bucket_delta is None
        assert report.paired_sign_test_p is None


# ---------- KnowledgeCheckReport invariants ----------


class TestKnowledgeCheckReportInvariants:
    def test_post_init_enforces_pillar_summary_false(self):
        data = _make_real_data()
        probe = KnowledgeProbe(symbol="AAPL",
                                templates=DEFAULT_TEMPLATES)
        anchors = list(data.index[10:15])
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        report = knowledge_check(
            probe, data, anchors, attested,
            provider_config=_provider_config(),
        )
        # Try to construct with is_pillar_summary=True via replace
        from dataclasses import replace
        with pytest.raises(ValueError, match="is_pillar_summary"):
            replace(report, is_pillar_summary=True)


# ---------- Persistence baseline (continuation only) ----------


class TestPersistenceBaseline:
    def test_present_for_continuation_probe(self):
        data = _make_real_data(n=80)
        probe = ContinuationProbe(
            symbol="AAPL", context_bars=10, forward_horizon=1,
            templates=[NextCloseContinuation],
        )
        anchors = list(data.index[20:25])
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        report = knowledge_check(
            probe, data, anchors, attested,
            provider_config=_provider_config(),
        )
        assert report.persistence_baseline_score is not None
        assert report.persistence_caveat is not None
        assert "deterministic" in report.persistence_caveat

    def test_absent_for_pointintime_probe(self):
        data = _make_real_data()
        probe = KnowledgeProbe(symbol="AAPL",
                                templates=DEFAULT_TEMPLATES)
        anchors = list(data.index[10:15])
        question_set = probe.build(data, anchors)
        attested = _build_perfect_attested(question_set)
        report = knowledge_check(
            probe, data, anchors, attested,
            provider_config=_provider_config(),
        )
        assert report.persistence_baseline_score is None
        assert report.persistence_caveat is None
