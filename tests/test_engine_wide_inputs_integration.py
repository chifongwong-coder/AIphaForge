"""v2.7 Commit F — multi-symbol integration + chained-run + transitivity.

5 tests required by v2.7 plan §"Commit F":

- 3-symbol 60-bar end-to-end integration for each new setter
- chained run (different data per call) re-materializes wide inputs
- 6-step transitivity chain — only the LAST setter's state survives
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.engine import BacktestEngine
from aiphaforge.signal_rules import ThresholdScoreRule


def _ohlcv(idx: pd.DatetimeIndex, base: float = 100.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = base + np.cumsum(rng.normal(0, 1, len(idx)))
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": 1_000_000.0,
        },
        index=idx,
    )


def _multi_data(symbols=("AAPL", "MSFT", "TSLA"), n: int = 60):
    idx = pd.bdate_range("2024-01-01", periods=n)
    return {
        sym: _ohlcv(idx, base=100.0 + 10.0 * i, seed=42 + i)
        for i, sym in enumerate(symbols)
    }


# ---------------------------------------------------------------------------
# End-to-end integration (3-symbol, 60 bars)
# ---------------------------------------------------------------------------


def test_set_signals_wide_integration_3_symbol_60_bar():
    data = _multi_data()
    idx = next(iter(data.values())).index
    # Hand-traced: long all 3 from bar 5 onward, flat on bar 30, long on bar 35.
    signal = pd.DataFrame(np.nan, index=idx, columns=list(data.keys()))
    signal.iloc[5] = 1.0
    signal.iloc[30] = 0.0
    signal.iloc[35] = 1.0

    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_signals_wide(signal)
    result = engine.run(data)

    # Sanity: at least one trade per symbol (entry + flatten + entry).
    symbols_traded = {t.symbol for t in result.trades}
    assert symbols_traded == set(data.keys()), (
        f"expected all 3 symbols traded, got {symbols_traded}"
    )


def test_set_score_wide_integration_with_threshold_rule_3_symbol():
    # Equivalence vs explicit set_signals_wide(rule.transform) baseline.
    data = _multi_data()
    idx = next(iter(data.values())).index
    rng = np.random.default_rng(13)
    scores = pd.DataFrame(
        rng.uniform(0, 1, size=(len(idx), len(data))),
        index=idx, columns=list(data.keys()),
    )
    rule = ThresholdScoreRule(long_threshold=0.7, short_threshold=0.3)

    engine_baseline = BacktestEngine(initial_capital=100_000.0)
    engine_baseline.set_signals_wide(rule.transform(scores))
    baseline = engine_baseline.run(data)

    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_score_wide(scores, rule)
    result = engine.run(data)

    assert result.total_return == pytest.approx(baseline.total_return, abs=1e-12)
    assert len(result.trades) == len(baseline.trades)


def test_set_target_weights_wide_integration_quarterly_rebalance():
    # Verify rebalance_dates actually fires the schedule ONLY on those
    # 3 dates (not every daily bar). Compare schedule key set vs the
    # 3-element rebalance_dates list.
    from aiphaforge.signals import target_weight_wide_to_schedule
    data = _multi_data(n=60)
    idx = next(iter(data.values())).index
    weights = pd.DataFrame(
        {sym: [1.0 / len(data)] * len(idx) for sym in data},
        index=idx,
    )
    rebalance_dates = [idx[0], idx[20], idx[40]]
    schedule = target_weight_wide_to_schedule(
        weights, data, rebalance_dates=rebalance_dates,
    )
    assert set(schedule.keys()) == set(rebalance_dates), (
        f"rebalance_dates must produce schedule keys exactly matching "
        f"the input dates; got {sorted(schedule.keys())} vs "
        f"{sorted(rebalance_dates)}"
    )
    # End-to-end through the engine: same result.
    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_target_weights_wide(
        weights, rebalance_dates=rebalance_dates,
    )
    engine.run(data)
    assert set(engine._target_weights.keys()) == set(rebalance_dates)


# ---------------------------------------------------------------------------
# Chained-run + transitivity
# ---------------------------------------------------------------------------


def test_chained_run_re_materializes_wide_inputs():
    # Chained run on different data per call must re-materialize against
    # each call's data_dict. Compare against fresh-engine baseline.
    sig_idx = pd.bdate_range("2024-01-01", periods=60)
    signal = pd.DataFrame({"AAPL": 1.0, "MSFT": 1.0}, index=sig_idx)
    signal.iloc[0] = 0.0

    data_q1 = {
        "AAPL": _ohlcv(sig_idx[:30], base=100, seed=1),
        "MSFT": _ohlcv(sig_idx[:30], base=200, seed=2),
    }
    data_q2 = {
        "AAPL": _ohlcv(sig_idx[30:], base=100, seed=3),
        "MSFT": _ohlcv(sig_idx[30:], base=200, seed=4),
    }

    # Independent baseline
    engine_a = BacktestEngine(initial_capital=100_000.0)
    engine_a.set_signals_wide(signal)
    base_q1 = engine_a.run(data_q1)
    engine_b = BacktestEngine(initial_capital=100_000.0)
    engine_b.set_signals_wide(signal)
    base_q2 = engine_b.run(data_q2)

    # Chained
    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_signals_wide(signal)
    chained_q1 = engine.run(data_q1)
    chained_q2 = engine.run(data_q2)

    assert chained_q1.total_return == pytest.approx(base_q1.total_return, abs=1e-12)
    assert chained_q2.total_return == pytest.approx(base_q2.total_return, abs=1e-12)


def test_six_setter_chain_only_last_survives():
    """Apply all 6 setters in sequence; verify only the LAST setter's
    state survives. This is a stronger assertion than the 30-pair
    sentinel — it asserts that 5 sequential overrides each correctly
    zero the prior state, transitively."""
    idx = pd.bdate_range("2024-01-01", periods=10)

    class _Dummy:
        name = "dummy"
        def generate_signals(self, data):
            if isinstance(data, dict):
                return {sym: pd.Series(1.0, index=df.index) for sym, df in data.items()}
            return pd.Series(1.0, index=data.index)

    engine = BacktestEngine()
    # Apply all 6 in order.
    engine.set_strategy(_Dummy())
    engine.set_signals(pd.Series(1.0, index=idx))
    engine.set_target_weights({"2024-01-01": {"AAPL": 1.0}})
    engine.set_signals_wide(pd.DataFrame({"AAPL": 1.0}, index=idx))
    engine.set_score_wide(
        pd.DataFrame({"AAPL": 0.8}, index=idx),
        ThresholdScoreRule(long_threshold=0.7),
    )
    engine.set_target_weights_wide(pd.DataFrame({"AAPL": 1.0}, index=idx))

    # Only set_target_weights_wide's owned fields should be populated.
    assert engine._strategy is None
    assert engine._signals is None
    assert engine._target_weights is None
    assert engine._signals_wide is None
    assert engine._target_weights_wide is not None
    assert engine._target_weights_wide_config is not None
