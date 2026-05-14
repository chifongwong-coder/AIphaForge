"""v2.5 Commit L — multi-asset composition with cross-sectional probes.

Verifies that cross-sectional strategies (MomentumRank, PairsTrading)
which produce dict[str, Series] outputs can nest inside StrategyNode
composites under the v2.5 generate_signals dispatch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.strategies import (
    _FALLBACK_WARNED,
    BaseStrategy,
    MomentumRank,
    PairsTrading,
    PriorityCascade,
    SelectBest,
    WeightedBlend,
)


@pytest.fixture(autouse=True)
def _clear_fallback_state():
    _FALLBACK_WARNED.clear()
    yield
    _FALLBACK_WARNED.clear()


def _ohlcv(periods=60, seed=0):
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": [1e6] * periods,
        },
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


def _multi_data(symbols=4, periods=60):
    return {f"S{i}": _ohlcv(periods=periods, seed=i + 1)
            for i in range(symbols)}


class _ConstantMultiChild(BaseStrategy):
    """Multi-asset child returning constant 1.0 per symbol."""

    name = "constant_multi"

    def generate_signals(self, data):
        if isinstance(data, dict):
            return {sym: pd.Series(1.0, index=df.index, dtype=float)
                    for sym, df in data.items()}
        return pd.Series(1.0, index=data.index, dtype=float)

    def _compute(self, df):
        return pd.Series(1.0, index=df.index, dtype=float)


def test_weighted_blend_with_momentum_rank_multi_asset():
    data = _multi_data(symbols=4)
    comp = WeightedBlend(
        children=[
            MomentumRank(roc_window=10, top_n=2),
            _ConstantMultiChild(),
        ],
        weights=[0.5, 0.5],
        mode="generate_signals",
    )
    out = comp.generate_signals(data)
    assert isinstance(out, dict)
    assert set(out.keys()) == set(data.keys())
    for sym in data:
        assert isinstance(out[sym], pd.Series)
        assert out[sym].index.equals(data[sym].index)


def test_select_best_with_two_cross_sectional_strategies():
    data = _multi_data(symbols=4)
    comp = SelectBest(
        children=[
            MomentumRank(roc_window=10, top_n=2),
            MomentumRank(roc_window=20, top_n=1),
        ],
        mode="generate_signals",
    )
    out = comp.generate_signals(data)
    assert set(out.keys()) == set(data.keys())


def test_priority_cascade_with_pairs_trading_in_modern_mode():
    data = {"A": _ohlcv(seed=1), "B": _ohlcv(seed=2)}
    comp = PriorityCascade(
        children=[
            PairsTrading(window=20, entry_z=2.0, exit_z=0.5),
            _ConstantMultiChild(),
        ],
        mode="generate_signals",
    )
    out = comp.generate_signals(data)
    assert set(out.keys()) == {"A", "B"}


def test_engine_can_run_multi_asset_composite_end_to_end():
    """Smoke test: composite of multi-asset strategies runs through
    BaseStrategy.backtest without raising."""
    data = _multi_data(symbols=3, periods=40)
    comp = WeightedBlend(
        children=[
            MomentumRank(roc_window=5, top_n=1),
            _ConstantMultiChild(),
        ],
        weights=[0.7, 0.3],
        mode="generate_signals",
    )
    # The .backtest helper accepts dict[str, DataFrame] for multi-asset.
    result = comp.backtest(data, fee_model="zero")
    assert hasattr(result, "equity_curve")
