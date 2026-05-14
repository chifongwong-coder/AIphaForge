"""v2.4 Commit I — FactorRuleStrategy tests."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aiphaforge import BacktestEngine
from aiphaforge.diagnostics import assert_signal_no_lookahead
from aiphaforge.factor_library import MomentumFactor, RSIFactor
from aiphaforge.factor_strategy import FactorRuleStrategy
from aiphaforge.factors import FactorSet
from aiphaforge.fees import ZeroFeeModel
from aiphaforge.signal_rules import (
    CrossSectionalQuantileRule,
    ThresholdScoreRule,
)


def _ohlcv(periods: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes,
            "volume": [1e6] * periods,
        },
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


def _multi(symbols: int = 4, periods: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        f"S{i}": _ohlcv(
            periods=periods, seed=int(rng.integers(0, 10000)),
        )
        for i in range(symbols)
    }


class TestFactorRuleStrategySingleAsset:
    def test_single_asset_round_trip(self):
        data = _ohlcv()
        rule = ThresholdScoreRule(long_threshold=70, short_threshold=30)
        strategy = FactorRuleStrategy(factor=RSIFactor(period=14), rule=rule)
        out = strategy.generate_signals(data)
        assert isinstance(out, pd.Series)


class TestFactorRuleStrategyMultiAsset:
    def test_multi_asset_round_trip(self):
        data = _multi()
        rule = CrossSectionalQuantileRule(long_quantile=0.75, short_quantile=0.25)
        strategy = FactorRuleStrategy(
            factor=MomentumFactor(window=10), rule=rule,
        )
        out = strategy.generate_signals(data)
        assert isinstance(out, dict)
        assert set(out.keys()) == set(data.keys())


class TestFactorRuleStrategyComputeFactors:
    def test_compute_factors_returns_single_factor_factor_set(self):
        data = _ohlcv()
        strategy = FactorRuleStrategy(
            factor=RSIFactor(period=14),
            rule=ThresholdScoreRule(long_threshold=70, short_threshold=30),
        )
        fs = strategy.compute_factors(data)
        assert isinstance(fs, FactorSet)
        assert "rsi_14" in fs.values
        assert "rsi_14" in fs.specs


class TestFactorRuleStrategyEngineIntegration:
    def test_engine_can_run_factor_rule_strategy(self):
        data = _ohlcv()
        strategy = FactorRuleStrategy(
            factor=MomentumFactor(window=10),
            rule=ThresholdScoreRule(
                long_threshold=0.02, short_threshold=-0.02,
            ),
        )
        engine = BacktestEngine(mode="vectorized", fee_model=ZeroFeeModel())
        engine.set_strategy(strategy)
        result = engine.run(data, symbol="X")
        assert result.equity_curve is not None


class TestFactorRuleStrategyNoLookahead:
    def test_no_lookahead_via_diagnostics(self):
        data = _ohlcv()
        strategy = FactorRuleStrategy(
            factor=MomentumFactor(window=10),
            rule=ThresholdScoreRule(
                long_threshold=0.02, short_threshold=-0.02,
            ),
        )
        # Use the v2.3 diagnostic on the strategy.
        assert_signal_no_lookahead(strategy, data)


class TestFactorRuleStrategyValidation:
    def test_raises_on_multi_factor_construction(self):
        # Anti-leak guard for v2.5+ multi-factor surface.
        with pytest.raises(ValueError, match="single-factor MVP"):
            FactorRuleStrategy(
                factor=[RSIFactor(), MomentumFactor()],  # type: ignore
                rule=ThresholdScoreRule(long_threshold=70),
            )

    def test_raises_on_non_factor_input(self):
        with pytest.raises(TypeError, match="BaseFactor"):
            FactorRuleStrategy(factor="not a factor", rule=object())  # type: ignore

    def test_raises_on_rule_without_transform(self):
        with pytest.raises(TypeError, match="transform"):
            FactorRuleStrategy(
                factor=RSIFactor(), rule=object(),
            )


class TestFactorStrategyFirewall:
    def test_factor_strategy_does_not_import_engine(self):
        # AST guard. factor_strategy.py imports BaseStrategy from
        # aiphaforge.strategies (upstream of engine — strategies
        # only does a LAZY local import of engine inside the
        # backtest method); module-level engine import is forbidden.
        path = (
            Path(__file__).resolve().parent.parent
            / "src/aiphaforge/factor_strategy.py"
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
                    f"factor_strategy.py imports {name!r}"
                )
