"""v2.5 Commit K — modern composition + nested-composite mode independence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.factor_library import RSIFactor
from aiphaforge.factor_strategy import FactorRuleStrategy
from aiphaforge.signal_rules import ThresholdScoreRule
from aiphaforge.signal_strategy import DirectSignalStrategy
from aiphaforge.strategies import (
    _FALLBACK_WARNED,
    BaseStrategy,
    MACrossover,
    RSIMeanReversion,
    SelectBest,
    VoteEnsemble,
    WeightedBlend,
)


@pytest.fixture(autouse=True)
def _clear_fallback_state():
    _FALLBACK_WARNED.clear()
    yield
    _FALLBACK_WARNED.clear()


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


class _TrackedChild(BaseStrategy):
    """Records which dispatch method was called on each invocation."""

    name = "tracked"

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


def test_weighted_blend_with_direct_signal_strategy_child():
    df = _data()
    direct_signals = pd.Series(1.0, index=df.index)
    direct = DirectSignalStrategy(signals=direct_signals)
    comp = WeightedBlend(
        children=[direct, MACrossover(short=5, long=20)],
        weights=[0.5, 0.5],
        mode="auto",
    )
    out = comp.generate_signals(df)
    assert isinstance(out, pd.Series)
    assert len(out.dropna()) >= 1


def test_select_best_with_factor_rule_strategy_child():
    df = _data()
    factor_strategy = FactorRuleStrategy(
        factor=RSIFactor(period=14),
        rule=ThresholdScoreRule(long_threshold=70.0, short_threshold=30.0),
    )
    comp = SelectBest(
        children=[factor_strategy, MACrossover(short=5, long=20)],
        mode="auto",
    )
    out = comp.generate_signals(df)
    assert isinstance(out, pd.Series)


def test_priority_cascade_with_legacy_plus_modern_children():
    df = _data()
    direct_signals = pd.Series(np.nan, index=df.index)
    direct_signals.iloc[5] = 1.0  # one early non-NaN entry
    direct_primary = DirectSignalStrategy(signals=direct_signals)
    legacy_fallback = MACrossover(short=5, long=20)
    from aiphaforge.strategies import PriorityCascade
    comp = PriorityCascade(
        children=[direct_primary, legacy_fallback],
        mode="auto",
    )
    out = comp.generate_signals(df)
    assert isinstance(out, pd.Series)
    # Primary covered bar 5; cascade should emit 1.0 at bar 5.
    assert out.iloc[5] == 1.0


def test_vote_ensemble_with_three_modern_children():
    df = _data()
    children = [_TrackedChild(), _TrackedChild(), _TrackedChild()]
    comp = VoteEnsemble(children=children, mode="auto")
    out = comp.generate_signals(df)
    assert isinstance(out, pd.Series)
    # All 3 children agree on +1 → majority condition
    # |3| > 3/2=1.5 → emit 1.0 at bar 0 via transitions_only.
    assert (out.dropna() == 1.0).any()


def test_engine_can_run_composite_of_modern_children():
    """End-to-end: full backtest with composite-of-modern-children."""
    df = _data()
    direct_signals = pd.Series(1.0, index=df.index)
    composite = WeightedBlend(
        children=[
            DirectSignalStrategy(signals=direct_signals),
            MACrossover(short=5, long=20),
        ],
        weights=[0.5, 0.5],
        mode="auto",
    )
    # backtest() returns a BacktestResult; we just assert it runs without
    # raising and returns a result object with the standard attributes.
    result = composite.backtest(df, fee_model="zero")
    assert hasattr(result, "equity_curve")
    assert hasattr(result, "trades")


def test_nested_composite_modes_are_independent():
    """WeightedBlend(mode="auto") wrapping SelectBest(mode="legacy_compute"):

    The inner SelectBest runs its children via legacy_compute regardless
    of the parent's auto mode. Verified by tracking which dispatch
    method the innermost child sees.
    """
    df = _data()
    inner_child = _TrackedChild()
    inner = SelectBest(
        children=[inner_child, RSIMeanReversion(period=14)],
        mode="legacy_compute",
    )
    outer = WeightedBlend(
        children=[inner, MACrossover(short=5, long=20)],
        weights=[0.5, 0.5],
        mode="auto",
    )
    outer.generate_signals(df)
    # The inner child must have been routed via _compute (because the
    # inner SelectBest runs in legacy_compute mode), NOT generate_signals
    # (which is what auto would route through if parent's mode leaked).
    assert inner_child.calls == ["_compute"], (
        f"nested SelectBest's child should have been called via _compute "
        f"(inner mode=legacy_compute), got {inner_child.calls}. Parent's "
        "auto mode must NOT propagate to child composite's children."
    )
