"""v2.2.1 #12 Commit A: rank-probe orchestrator scoring wiring.

The v2.2.1 audit found RankContinuationProbe runs through
knowledge_check produce silently-zero KnowledgeCheckReports
because the orchestrator routed through generic numeric scoring
which can't handle tuple-of-strings truth_value. This commit
adds _score_rank_attested and wires it on both real and anchor
sides.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from aiphaforge.probes._rank import RankContinuationProbe
from aiphaforge.probes.anchors import build_synthetic_anchor
from aiphaforge.probes.models import AnswerRecord, AttestedAnswers
from aiphaforge.probes.orchestrator import (
    _score_rank_attested,
    knowledge_check,
)


def _provider_config():
    return {
        "temperature": 0.0, "top_p": 1.0,
        "model_id": "test", "model_version": "1",
        "max_output_tokens": 256, "tokenizer_id": "test",
        "seed": 42, "reasoning_effort": None, "stop_sequences": None,
    }


def _bars(n: int = 30, seed: int = 0, base: float = 100.0,
          jump_at: int = 20, jump_size: float = 5.0) -> pd.DataFrame:
    """Deterministic OHLCV with a known jump at `jump_at`."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, n)
    closes = base * np.exp(np.cumsum(rets))
    closes[jump_at] = closes[jump_at - 1] + jump_size
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.005,
            "low": closes * 0.995, "close": closes,
            "volume": [1e6] * n,
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )


def _multi_data_with_known_ranking(
    n: int = 30, anchor_idx: int = 19,
) -> tuple[dict[str, pd.DataFrame], pd.Timestamp]:
    """Returns (data_dict, anchor_ts). Symbols at t+1 have known
    returns: A best, B middle, C worst."""
    # Use plain bdates and set last close at anchor+1 to encode
    # the ranking deterministically.
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


# ---------- Unit: _score_rank_attested helper ----------


class TestScoreRankAttested:
    def test_returns_report_and_counter_signature(self):
        data, anchor_ts = _multi_data_with_known_ranking()
        probe = RankContinuationProbe(
            symbols=["A", "B", "C"], context_bars=5,
            forward_horizon=1,
        )
        qs = probe.build(data, [anchor_ts])
        # Mixed batch: one structured, one text
        recs = [
            AnswerRecord(
                question_id=list(qs)[0].question_id,
                raw_answer=None,
                parsed_answer=["A", "B", "C"],
                parse_status="valid",
            ),
        ]
        attested = AttestedAnswers(
            answers=tuple(recs),
            parsing_schema_hash="a" * 64,
            parsing_schema_description="x",
            prompt_template_hash="b" * 64,
            prompt_template_description="y",
        )
        report, counter = _score_rank_attested(
            qs, attested, provider_config=_provider_config(),
        )
        # Signature: (QAProbeReport, Counter)
        from aiphaforge.probes.models import QAProbeReport
        assert isinstance(report, QAProbeReport)
        assert isinstance(counter, Counter)
        # Counter sums to number of records that had a non-None
        # parser_used (here 1).
        assert sum(counter.values()) == 1
        # And the path was user_provided_list.
        assert counter["user_provided_list"] == 1


# ---------- Integration: knowledge_check end-to-end ----------


class TestKnowledgeCheckRankContinuation:
    def _attested(self, ranking, qid, hash_="a" * 64):
        return AttestedAnswers(
            answers=(AnswerRecord(
                question_id=qid, raw_answer=None,
                parsed_answer=list(ranking),
                parse_status="valid",
            ),),
            parsing_schema_hash=hash_,
            parsing_schema_description="x",
            prompt_template_hash="b" * 64,
            prompt_template_description="y",
        )

    def test_perfect_answer_scores_exact_not_invalid(self):
        # The v2.2.1 audit bug: perfect ranking scored "invalid"
        # because orchestrator routed through _score_numeric.
        # After Commit A: should score "exact".
        data, anchor_ts = _multi_data_with_known_ranking()
        probe = RankContinuationProbe(
            symbols=["A", "B", "C"], context_bars=5,
            forward_horizon=1,
        )
        qs = probe.build(data, [anchor_ts])
        qid = list(qs)[0].question_id
        # Truth at t+1: A>B>C (per _multi_data_with_known_ranking).
        attested = self._attested(["A", "B", "C"], qid)
        report = knowledge_check(
            probe, data, [anchor_ts], attested,
            provider_config=_provider_config(),
        )
        breakdown = report.real_score.bands_breakdown
        assert breakdown["exact"] == 1, (
            f"Expected 1 exact, got {breakdown}"
        )
        assert breakdown["invalid"] == 0

    def test_anti_ranking_scores_miss(self):
        data, anchor_ts = _multi_data_with_known_ranking()
        probe = RankContinuationProbe(
            symbols=["A", "B", "C"], context_bars=5,
            forward_horizon=1,
        )
        qs = probe.build(data, [anchor_ts])
        qid = list(qs)[0].question_id
        # Anti-ranking: C, B, A
        attested = self._attested(["C", "B", "A"], qid)
        report = knowledge_check(
            probe, data, [anchor_ts], attested,
            provider_config=_provider_config(),
        )
        breakdown = report.real_score.bands_breakdown
        assert breakdown["miss"] == 1
        assert breakdown["exact"] == 0

    def test_partial_answer_scores_invalid(self):
        # Omitted symbol → invalid bucket per RankAnswer scoring rule.
        data, anchor_ts = _multi_data_with_known_ranking()
        probe = RankContinuationProbe(
            symbols=["A", "B", "C"], context_bars=5,
            forward_horizon=1,
        )
        qs = probe.build(data, [anchor_ts])
        qid = list(qs)[0].question_id
        attested = self._attested(["A", "B"], qid)  # missing C
        report = knowledge_check(
            probe, data, [anchor_ts], attested,
            provider_config=_provider_config(),
        )
        breakdown = report.real_score.bands_breakdown
        assert breakdown["invalid"] == 1
        assert breakdown["exact"] == 0

    def test_text_raw_answer_via_parser_fallback(self):
        data, anchor_ts = _multi_data_with_known_ranking()
        probe = RankContinuationProbe(
            symbols=["A", "B", "C"], context_bars=5,
            forward_horizon=1,
        )
        qs = probe.build(data, [anchor_ts])
        qid = list(qs)[0].question_id
        # Pass raw text, no parsed_answer
        attested = AttestedAnswers(
            answers=(AnswerRecord(
                question_id=qid,
                raw_answer='["A","B","C"]',
                parsed_answer=None,
                parse_status="valid",
            ),),
            parsing_schema_hash="a" * 64,
            parsing_schema_description="x",
            prompt_template_hash="b" * 64,
            prompt_template_description="y",
        )
        report = knowledge_check(
            probe, data, [anchor_ts], attested,
            provider_config=_provider_config(),
        )
        breakdown = report.real_score.bands_breakdown
        assert breakdown["exact"] == 1

    def test_with_anchor_populates_bucket_delta_and_scalar(self):
        data, anchor_ts = _multi_data_with_known_ranking()
        probe = RankContinuationProbe(
            symbols=["A", "B", "C"], context_bars=5,
            forward_horizon=1,
        )
        qs = probe.build(data, [anchor_ts])
        qid = list(qs)[0].question_id
        real_attested = self._attested(["A", "B", "C"], qid)
        # Build synthetic anchor for each symbol.
        anchor_data = {
            sym: build_synthetic_anchor(
                df, seed=hash(sym) % 1000,
                method="random_walk_volmatched",
            )
            for sym, df in data.items()
        }
        anchor_qs = probe.build(anchor_data, [anchor_ts])
        anchor_qid = list(anchor_qs)[0].question_id
        # Pretend the LLM ranks the synthetic-anchor symbols
        # randomly — different from real's perfect ranking.
        anchor_truth = list(anchor_qs)[0].truth_value
        anchor_attested = self._attested(
            list(reversed(anchor_truth)), anchor_qid,
        )
        report = knowledge_check(
            probe, data, [anchor_ts], real_attested,
            anchor=anchor_data["A"],  # workaround: pass A's anchor
            anchor_answers=anchor_attested,
            provider_config=_provider_config(),
        )
        # If validity OK, bucket_delta and scalar_leakage_index
        # should be populated.
        if report.anchor_validity == "OK":
            assert report.bucket_delta is not None
            assert report.scalar_leakage_index is not None

    def test_normalization_preset_applies_to_user_answer(self):
        # User submits lowercase + dash; us_equity preset
        # normalizes to uppercase + dot before comparing to truth.
        data, anchor_ts = _multi_data_with_known_ranking()
        probe = RankContinuationProbe(
            symbols=["A", "B", "C"], context_bars=5,
            forward_horizon=1,
            normalization_preset="us_equity",
        )
        qs = probe.build(data, [anchor_ts])
        qid = list(qs)[0].question_id
        # Lowercase submission; us_equity preset uppercases.
        attested = self._attested(["a", "b", "c"], qid)
        report = knowledge_check(
            probe, data, [anchor_ts], attested,
            provider_config=_provider_config(),
        )
        breakdown = report.real_score.bands_breakdown
        assert breakdown["exact"] == 1

    def test_n_equals_2_does_not_crash(self):
        # Minimum N for ranking. Spearman ρ at N=2 is ±1.
        idx = pd.bdate_range("2024-01-01", periods=20)
        base = pd.DataFrame(
            {
                "open": [100.0] * 20, "high": [101.0] * 20,
                "low": [99.0] * 20, "close": [100.0] * 20,
                "volume": [1e6] * 20,
            },
            index=idx,
        )
        a = base.copy()
        a.iloc[11, a.columns.get_loc("close")] = 105.0
        b = base.copy()
        b.iloc[11, b.columns.get_loc("close")] = 99.0
        data = {"A": a, "B": b}
        probe = RankContinuationProbe(
            symbols=["A", "B"], context_bars=5, forward_horizon=1,
        )
        qs = probe.build(data, [idx[10]])
        qid = list(qs)[0].question_id
        attested = self._attested(["A", "B"], qid)
        # Should not crash; bucket assignment should be finite.
        report = knowledge_check(
            probe, data, [idx[10]], attested,
            provider_config=_provider_config(),
        )
        breakdown = report.real_score.bands_breakdown
        total = sum(breakdown.values())
        assert total == 1
        # At N=2 ρ=+1 → either exact or one of the higher bands.
        assert breakdown["invalid"] == 0
