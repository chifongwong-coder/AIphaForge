"""v2.2.1 #11 — QuestionSpec frozen migration."""
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from aiphaforge.probes.models import QuestionSpec, ToleranceProfile


def _make_spec() -> QuestionSpec:
    return QuestionSpec(
        question_id="q0",
        symbol="AAPL",
        timestamp=pd.Timestamp("2024-01-02"),
        template_id="close",
        answer_type="numeric",
        prompt_text="Predict close",
        choices=None,
        truth_value=100.5,
        tolerance=ToleranceProfile.us_equity_price(),
    )


class TestQuestionSpecFrozen:
    def test_question_spec_is_frozen(self):
        assert QuestionSpec.__dataclass_params__.frozen is True

    def test_assigning_to_frozen_field_raises(self):
        spec = _make_spec()
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.timestamp = pd.Timestamp("2025-01-01")

    def test_dataclasses_replace_works_on_frozen(self):
        # The M6 vol-scaling wiring uses dataclasses.replace; verify
        # it works against frozen QuestionSpec.
        spec = _make_spec()
        new_tol = ToleranceProfile.crypto_price()
        replaced = dataclasses.replace(spec, tolerance=new_tol)
        assert replaced.tolerance.preset_name == "crypto_price"
        assert spec.tolerance.preset_name == "us_equity_price"  # original unchanged

    def test_metadata_dict_remains_shallow_mutable(self):
        # v2.2.1 documented: metadata dict is shallow-mutable
        # (deep immutability deferred to v2.3+). Lock the current
        # behavior so future v2.3 changes are intentional.
        spec = _make_spec()
        spec.metadata["foo"] = "bar"  # does NOT raise
        assert spec.metadata["foo"] == "bar"
