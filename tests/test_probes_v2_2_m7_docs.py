"""v2.2 M7 tests — README non-transitivity + warning notes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aiphaforge.probes.models import AnswerRecord, AttestedAnswers
from aiphaforge.probes.orchestrator import knowledge_check
from aiphaforge.probes.questions import (
    DEFAULT_TEMPLATES,
    KnowledgeProbe,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"


# ---------- README ----------


class TestReadmeNonTransitivity:
    def test_section_exists(self):
        text = README.read_text()
        assert "Pillar non-transitivity" in text, (
            "README must carry the 'Pillar non-transitivity' "
            "section per v2.2 M7"
        )

    def test_bidirectional_example_present(self):
        text = README.read_text().lower()
        # The README must show BOTH directions of pillar disagreement
        # so a reader cannot read the section as advocating one
        # ordering. Per LE-F6 r3 review.
        assert "knowledge-check pass + obfuscation-bootstrap fail" in text
        assert "knowledge-check fail + obfuscation-bootstrap pass" in text

    def test_no_aggregate_verdict_promised(self):
        text = README.read_text().lower()
        # The README must not promise a single 🟢/🟡/🔴 verdict.
        # Forbidden phrases that would imply one:
        forbidden = (
            "single verdict", "overall verdict",
            "combined leakage score",
        )
        for f in forbidden:
            assert f not in text, (
                f"README must NOT promise an aggregate verdict; "
                f"found {f!r}"
            )


# ---------- KnowledgeCheckReport carries the warning ----------


def _provider_config(**overrides):
    base = {
        "temperature": 0.0, "top_p": 1.0,
        "model_id": "test-model", "model_version": "1",
        "max_output_tokens": 256, "tokenizer_id": "test-tokenizer",
        "seed": 42, "reasoning_effort": None, "stop_sequences": None,
    }
    base.update(overrides)
    return base


def _make_data(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
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


class TestReportCarriesWarning:
    def test_notes_carry_non_transitivity_warning(self):
        data = _make_data()
        probe = KnowledgeProbe(
            symbol="AAPL", templates=DEFAULT_TEMPLATES,
        )
        anchors = list(data.index[10:15])
        question_set = probe.build(data, anchors)
        attested = AttestedAnswers(
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
        report = knowledge_check(
            probe, data, anchors, attested,
            provider_config=_provider_config(),
        )
        joined = " ".join(report.notes).lower()
        assert "pillar non-transitivity" in joined
        assert "obfuscation_bootstrap" in joined


# ---------- Version metadata ----------


class TestVersion:
    def test_version_is_current_release(self):
        # Updated by v2.8.1 patch bump (8 HIGH fixes from post-v2.8 audit).
        import aiphaforge
        assert aiphaforge.__version__ == "2.8.1"
