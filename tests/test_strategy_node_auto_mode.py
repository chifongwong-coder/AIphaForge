"""v2.5 Commit I — cross-cutting auto-mode dispatch tests.

Verifies that auto-mode picks generate_signals when ALL children
support it, across all 5 composites uniformly. Catches per-composite
divergence in dispatch policy.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from aiphaforge.strategies import (
    _FALLBACK_WARNED,
    BaseStrategy,
    ConditionalSwitch,
    PriorityCascade,
    SelectBest,
    VoteEnsemble,
    WeightedBlend,
)


@pytest.fixture(autouse=True)
def _clear_fallback_state():
    _FALLBACK_WARNED.clear()
    yield
    _FALLBACK_WARNED.clear()


def _data(periods=30, seed=0):
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": [1e6] * periods,
        },
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


class _TrackedModernChild(BaseStrategy):
    """Records which method was called on it."""

    name = "tracked_modern"

    def __init__(self):
        self.calls: list[str] = []

    def generate_signals(self, data):
        self.calls.append("generate_signals")
        if isinstance(data, dict):
            return {sym: pd.Series(1.0, index=df.index, dtype=float)
                    for sym, df in data.items()}
        return pd.Series(1.0, index=data.index, dtype=float)

    def _compute(self, df):
        self.calls.append("_compute")
        return pd.Series(1.0, index=df.index, dtype=float)


class _BrokenModernChild(BaseStrategy):
    """generate_signals raises; _compute returns flat zeros."""

    name = "broken_modern"

    def generate_signals(self, data):
        raise KeyError("simulated_failure")

    def _compute(self, df):
        return pd.Series(0.0, index=df.index, dtype=float)


def _build_composite(composite_cls, children):
    if composite_cls is ConditionalSwitch:
        return composite_cls(
            children=children,
            condition_fn=lambda df: pd.Series(0, index=df.index, dtype=int),
            mode="auto",
        )
    if composite_cls is WeightedBlend:
        return composite_cls(
            children=children,
            weights=[1.0 / len(children)] * len(children),
            mode="auto",
        )
    return composite_cls(children=children, mode="auto")


@pytest.mark.parametrize(
    "composite_cls",
    [WeightedBlend, SelectBest, PriorityCascade, VoteEnsemble, ConditionalSwitch],
    ids=["WeightedBlend", "SelectBest", "PriorityCascade",
         "VoteEnsemble", "ConditionalSwitch"],
)
def test_auto_picks_generate_signals_when_all_children_modern(composite_cls):
    children = [_TrackedModernChild(), _TrackedModernChild()]
    comp = _build_composite(composite_cls, children)
    comp.generate_signals(_data())
    for child in children:
        assert child.calls == ["generate_signals"], (
            f"{composite_cls.__name__} should have routed children via "
            f"generate_signals; got {child.calls}"
        )


def test_auto_per_child_fallback_emits_warning_first_time(caplog):
    comp = WeightedBlend(
        children=[_TrackedModernChild(), _BrokenModernChild()],
        mode="auto",
    )
    with caplog.at_level(logging.WARNING, logger="aiphaforge.strategies"):
        comp.generate_signals(_data())
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and "_BrokenModernChild" in r.message
    ]
    assert len(warnings) == 1, (
        f"expected exactly 1 WARNING for first fallback, got "
        f"{[r.message for r in warnings]}"
    )
