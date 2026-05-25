"""v2.3 Commit F — DirectSignalStrategy tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge import BacktestEngine
from aiphaforge.fees import ZeroFeeModel
from aiphaforge.signal_rules import (
    ThresholdScoreRule,
)
from aiphaforge.signal_strategy import DirectSignalStrategy
from tests._helpers.ohlcv import make_ohlcv

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestDirectSignalStrategyShapes:
    def test_single_series(self):
        data = make_ohlcv(periods=30)
        sig = pd.Series(
            [1.0, np.nan, 0.0, -1.0, np.nan] * 6, index=data.index,
        )
        strategy = DirectSignalStrategy(signals=sig)
        out = strategy.generate_signals(data)
        assert isinstance(out, pd.Series)
        assert out.iloc[0] == 1.0
        assert out.iloc[3] == -1.0

    def test_multi_dict(self):
        data = {"A": make_ohlcv(periods=30), "B": make_ohlcv(periods=30)}
        sigs = {
            "A": pd.Series([1.0] + [np.nan] * 29, index=data["A"].index),
            "B": pd.Series([0.0, -1.0] + [np.nan] * 28, index=data["B"].index),
        }
        strategy = DirectSignalStrategy(signals=sigs)
        out = strategy.generate_signals(data)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"A", "B"}

    def test_wide_dataframe(self):
        data = {"A": make_ohlcv(periods=30), "B": make_ohlcv(periods=30)}
        wide = pd.DataFrame(
            {"A": [1.0] + [np.nan] * 29,
             "B": [0.0, -1.0] + [np.nan] * 28},
            index=data["A"].index,
        )
        strategy = DirectSignalStrategy(signals=wide)
        out = strategy.generate_signals(data)
        assert isinstance(out, dict)
        assert out["A"].iloc[0] == 1.0
        assert out["B"].iloc[1] == -1.0

    def test_scores_with_rule(self):
        data = make_ohlcv(periods=30)
        scores = pd.Series(
            [0.85] + [0.5] * 14 + [0.10] + [0.5] * 14, index=data.index,
        )
        rule = ThresholdScoreRule(long_threshold=0.7, short_threshold=0.3)
        strategy = DirectSignalStrategy(scores=scores, rule=rule)
        out = strategy.generate_signals(data)
        assert isinstance(out, pd.Series)
        assert out.iloc[0] == 1.0
        assert out.iloc[15] == -1.0

    def test_missing_symbol_returns_nan(self):
        # Multi-asset data has B but signals dict only has A.
        data = {"A": make_ohlcv(periods=30), "B": make_ohlcv(periods=30)}
        sigs = {"A": pd.Series([1.0] * 30, index=data["A"].index)}
        strategy = DirectSignalStrategy(signals=sigs)
        out = strategy.generate_signals(data)
        assert "B" in out
        assert out["B"].isna().all()


# ---------------------------------------------------------------------------
# Live-update API
# ---------------------------------------------------------------------------


class TestUpdateSignalsAndScores:
    def test_update_signals_replaces_underlying(self):
        data = make_ohlcv(periods=30)
        sig1 = pd.Series([1.0] * 30, index=data.index)
        strategy = DirectSignalStrategy(signals=sig1)
        out1 = strategy.generate_signals(data)
        assert out1.iloc[0] == 1.0
        # Replace.
        sig2 = pd.Series([-1.0] * 30, index=data.index)
        strategy.update_signals(sig2)
        out2 = strategy.generate_signals(data)
        assert out2.iloc[0] == -1.0

    def test_update_scores_requires_rule_to_be_set(self):
        # Strategy constructed via signals path → no rule → cannot
        # accept scores.
        data = make_ohlcv(periods=30)
        sig = pd.Series([1.0] * 30, index=data.index)
        strategy = DirectSignalStrategy(signals=sig)
        scores = pd.Series([0.85] * 30, index=data.index)
        with pytest.raises(ValueError, match="requires a rule"):
            strategy.update_scores(scores)

    def test_update_scores_after_setting_rule_via_update_params(self):
        # Per docstring: update_params(rule=...) then update_scores
        # is the supported workflow for retrofitting a signals-path
        # strategy with score conversion.
        data = make_ohlcv(periods=30)
        sig = pd.Series([1.0] * 30, index=data.index)
        strategy = DirectSignalStrategy(signals=sig)
        rule = ThresholdScoreRule(long_threshold=0.7, short_threshold=0.3)
        strategy.update_params(rule=rule)
        scores = pd.Series(
            [0.85] + [0.5] * 14 + [0.10] + [0.5] * 14, index=data.index,
        )
        strategy.update_scores(scores)  # must NOT raise
        out = strategy.generate_signals(data)
        assert out.iloc[0] == 1.0
        assert out.iloc[15] == -1.0


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestConstructionValidation:
    def test_neither_signals_nor_scores_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            DirectSignalStrategy()

    def test_both_signals_and_scores_raises(self):
        sig = pd.Series([1.0])
        scores = pd.Series([0.85])
        rule = ThresholdScoreRule(long_threshold=0.7)
        with pytest.raises(ValueError, match="exactly one"):
            DirectSignalStrategy(signals=sig, scores=scores, rule=rule)

    def test_scores_without_rule_raises(self):
        scores = pd.Series([0.85])
        with pytest.raises(ValueError, match="requires a rule"):
            DirectSignalStrategy(scores=scores)


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_engine_can_run_direct_signal_strategy_single_asset(self):
        data = make_ohlcv(periods=30)
        # Long on day 5, flatten on day 20.
        sig = pd.Series(np.nan, index=data.index)
        sig.iloc[5] = 1.0
        sig.iloc[20] = 0.0
        strategy = DirectSignalStrategy(signals=sig)
        engine = BacktestEngine(mode="vectorized", fee_model=ZeroFeeModel())
        engine.set_strategy(strategy)
        result = engine.run(data, symbol="X")
        # Sanity: result has equity_curve and trade-related metrics.
        assert result.equity_curve is not None
        assert len(result.equity_curve) == len(data)

    def test_engine_can_run_direct_signal_strategy_multi_asset(self):
        data = {"A": make_ohlcv(periods=30), "B": make_ohlcv(periods=30)}
        wide = pd.DataFrame(
            np.nan, index=data["A"].index, columns=["A", "B"], dtype=float,
        )
        wide.iloc[5, 0] = 1.0  # long A
        wide.iloc[5, 1] = -1.0  # short B
        wide.iloc[20, 0] = 0.0  # close A
        wide.iloc[20, 1] = 0.0  # close B
        strategy = DirectSignalStrategy(signals=wide)
        engine = BacktestEngine(mode="vectorized", fee_model=ZeroFeeModel())
        engine.set_strategy(strategy)
        result = engine.run(data)
        assert result.equity_curve is not None


# ---------------------------------------------------------------------------
# update_params scope
# ---------------------------------------------------------------------------


class TestUpdateParamsScope:
    def test_unknown_param_raises_with_helpful_message(self):
        sig = pd.Series([1.0])
        strategy = DirectSignalStrategy(signals=sig)
        with pytest.raises(ValueError, match="update_signals"):
            strategy.update_params(long_threshold=0.7)

    def test_update_params_rule_none_clears_rule(self):
        # Document the clearing semantic: update_params(rule=None) is
        # legal and disables future update_scores() calls. Subsequent
        # update_scores raises with the standard "requires a rule"
        # message — verifying the cleared state is detectable.
        sig = pd.Series([1.0])
        rule = ThresholdScoreRule(long_threshold=0.7)
        strategy = DirectSignalStrategy(signals=sig)
        strategy.update_params(rule=rule)
        assert strategy.rule is rule
        # Now clear it.
        strategy.update_params(rule=None)
        assert strategy.rule is None
        # update_scores must now raise.
        scores = pd.Series([0.85])
        with pytest.raises(ValueError, match="requires a rule"):
            strategy.update_scores(scores)

    def test_name_and_rule_allowed(self):
        sig = pd.Series([1.0])
        strategy = DirectSignalStrategy(signals=sig)
        new_rule = ThresholdScoreRule(long_threshold=0.7)
        strategy.update_params(name="renamed", rule=new_rule)
        assert strategy.name == "renamed"
        assert strategy.rule is new_rule
