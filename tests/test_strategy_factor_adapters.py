"""v2.4 Commit J — extract_strategy_factors tests."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from aiphaforge.factors import FactorSet
from aiphaforge.strategies import (
    MACrossover,
    MomentumRank,
    PairsTrading,
    RSIMeanReversion,
    VWAPReversion,
    WeightedBlend,
)
from aiphaforge.strategy_factors import extract_strategy_factors
from tests._helpers.ohlcv import make_ohlcv


def _multi(symbols: int = 4, periods: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        f"S{i}": make_ohlcv(periods=periods, seed=int(rng.integers(0, 9999)))
        for i in range(symbols)
    }


# ---------------------------------------------------------------------------
# Per-strategy adapter round-trips
# ---------------------------------------------------------------------------


class TestPerStrategyAdapters:
    def test_ma_crossover_adapter(self):
        data = make_ohlcv()
        strat = MACrossover(short=5, long=20)
        before = strat.generate_signals(data).copy()
        fs = extract_strategy_factors(strat, data)
        assert {"ma_short", "ma_long", "ma_spread"} <= set(fs.values.keys())
        # Snapshot must NOT mutate strategy state.
        after = strat.generate_signals(data)
        pd.testing.assert_series_equal(before, after, check_names=False)

    def test_rsi_adapter(self):
        data = make_ohlcv()
        strat = RSIMeanReversion(period=14)
        fs = extract_strategy_factors(strat, data)
        assert "rsi" in fs.values

    def test_vwap_adapter(self):
        data = make_ohlcv()
        strat = VWAPReversion()
        fs = extract_strategy_factors(strat, data)
        assert {"vwap", "vwap_distance"} <= set(fs.values.keys())

    def test_momentum_rank_adapter(self):
        data = _multi(symbols=4)
        strat = MomentumRank(roc_window=20, top_n=2)
        fs = extract_strategy_factors(strat, data)
        assert "roc" in fs.values
        # Wide layout: 4 columns matching input symbols.
        assert set(fs.values["roc"].columns) == set(data.keys())

    def test_pairs_adapter(self):
        data = {"A": make_ohlcv(seed=1), "B": make_ohlcv(seed=2)}
        strat = PairsTrading(window=20, entry_z=2.0, exit_z=0.5)
        fs = extract_strategy_factors(strat, data)
        assert {"spread", "spread_zscore"} <= set(fs.values.keys())


# ---------------------------------------------------------------------------
# Dispatch + composite anti-recursion + snapshot semantic
# ---------------------------------------------------------------------------


class TestDispatchSemantics:
    def test_isinstance_dispatch_inherits_subclass(self):
        # User subclass of MACrossover gets the same adapter.
        class MyMA(MACrossover):
            pass

        data = make_ohlcv()
        strat = MyMA(short=5, long=20)
        fs = extract_strategy_factors(strat, data)
        assert "ma_spread" in fs.values

    def test_returns_empty_for_composite_strategy(self):
        # WeightedBlend is a StrategyNode composite; adapter
        # explicitly returns FactorSet.empty(), no recursion into
        # children.
        data = make_ohlcv()
        composite = WeightedBlend(
            children=[MACrossover(short=5, long=20)],
            weights=[1.0],
        )
        fs = extract_strategy_factors(composite, data)
        assert isinstance(fs, FactorSet)
        assert fs.values == {}
        assert fs.specs == {}

    def test_extract_uses_current_params_not_history(self):
        # Snapshot-semantic contract: changing strategy params
        # via update_params(...) changes the extracted factors.
        data = make_ohlcv()
        strat = MACrossover(short=5, long=20)
        fs1 = extract_strategy_factors(strat, data)
        ma_short_before = fs1.values["ma_short"]["_default"]

        # Mutate params.
        strat.update_params(short=10)
        fs2 = extract_strategy_factors(strat, data)
        ma_short_after = fs2.values["ma_short"]["_default"]

        # The two snapshots differ because the lookback changed.
        assert not ma_short_before.equals(ma_short_after)

    def test_unknown_strategy_returns_empty(self):
        from aiphaforge.strategies import BaseStrategy

        class MyMLStrategy(BaseStrategy):
            def generate_signals(self, data):
                return pd.Series(dtype=float)

        fs = extract_strategy_factors(MyMLStrategy(), make_ohlcv())
        assert fs.values == {}


# ---------------------------------------------------------------------------
# Architectural firewall
# ---------------------------------------------------------------------------


class TestStrategyFactorsFirewall:
    def test_strategy_factors_does_not_import_engine(self):
        path = (
            Path(__file__).resolve().parent.parent
            / "src/aiphaforge/strategy_factors.py"
        )
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert "aiphaforge.engine" not in name, (
                    f"strategy_factors.py imports {name!r}"
                )
