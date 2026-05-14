"""v2.5 Commit M — extract_strategy_factors anti-recursion preservation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import aiphaforge.strategy_factors as sf_module
from aiphaforge.factors import FactorSet
from aiphaforge.strategies import (
    MACrossover,
    RSIMeanReversion,
    WeightedBlend,
)
from aiphaforge.strategy_factors import extract_strategy_factors


def _data(periods=60, seed=42):
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": [1e6] * periods,
        },
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


def test_extract_factors_from_v2_5_composite_still_returns_empty():
    composite = WeightedBlend(
        children=[MACrossover(short=5, long=20), RSIMeanReversion(period=14)],
    )
    result = extract_strategy_factors(composite, _data())
    assert isinstance(result, FactorSet)
    assert result.values == {}
    assert result.specs == {}


@pytest.mark.parametrize(
    "mode", ["auto", "generate_signals", "legacy_compute"],
)
def test_extract_factors_independent_of_mode(mode):
    composite = WeightedBlend(
        children=[MACrossover(short=5, long=20), RSIMeanReversion(period=14)],
        mode=mode,
    )
    result = extract_strategy_factors(composite, _data())
    assert result.values == {}, (
        f"extract_strategy_factors output must be independent of "
        f"composite.mode (got non-empty FactorSet for mode={mode!r})"
    )


def test_anti_recursion_documented_in_strategy_factors_docstring():
    doc = extract_strategy_factors.__doc__ or ""
    # Loose match — survives doc edits as long as the v2.5 contract or
    # the mode-irrelevance is mentioned.
    assert "v2.5" in doc or "mode" in doc, (
        "extract_strategy_factors docstring must mention the v2.5 "
        "anti-recursion preservation rationale or mode-independence"
    )


def test_v3_0_recursive_extraction_not_yet_available():
    # Sentinel: future v3.0 might add an opt-in recursive extractor.
    # Verify the absence today so a v3.0 addition is intentional, not
    # a silent rename / re-export.
    assert not hasattr(sf_module, "extract_composite_factors"), (
        "extract_composite_factors is a v3.0 opt-in feature; if it "
        "shipped, update this sentinel and add coverage."
    )
