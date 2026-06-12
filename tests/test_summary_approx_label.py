"""v2.8.6 Commit I — vectorized cost lines labeled (approx).

The once-per-process vectorized cost warning drowns in parameter
sweeps; the ``metadata['cost_model']`` marker travels with each
result so ``summary()`` can label the cost lines themselves.
"""
from __future__ import annotations

import pandas as pd

from aiphaforge import BacktestEngine

from .conftest import make_ohlcv


def _summary(mode: str) -> str:
    data = make_ohlcv(40)
    signals = pd.Series(0.0, index=data.index)
    signals.iloc[0:20] = 1.0
    engine = BacktestEngine(mode=mode)
    engine.set_signals(signals)
    return engine.run(data).summary()


def test_vectorized_summary_costs_marked_approx():
    summary = _summary("vectorized")
    assert "Total Commission" in summary
    commission_line = next(
        line for line in summary.splitlines()
        if "Total Commission" in line)
    slippage_line = next(
        line for line in summary.splitlines()
        if "Total Slippage" in line)
    assert commission_line.endswith("(approx)")
    assert slippage_line.endswith("(approx)")
    # Vectorized never shows a Total Spread line (folded into returns).
    assert "Total Spread" not in summary


def test_event_driven_summary_no_approx_label():
    assert "(approx)" not in _summary("event_driven")
