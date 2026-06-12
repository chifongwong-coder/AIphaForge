"""v2.8.6 Commit I/N — vectorized cost reporting in summary().

Vectorized costs are subtracted from returns and never attributed to
reconstructed trades, so the per-trade totals are hard zeros; printing
"0.00 (approx)" would mislead. The summary replaces the cost lines
with an embedded-in-returns note, driven by the per-result
``metadata['cost_model']`` marker (the once-per-process warning
drowns in parameter sweeps).
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
    assert "[Costs]" in summary
    assert "Embedded in returns (vectorized approx)" in summary
    assert "event_driven" in summary  # actionable pointer
    # No misleading hard-zero attribution lines.
    assert "Total Commission" not in summary
    assert "Total Slippage" not in summary
    assert "Total Spread" not in summary


def test_event_driven_summary_no_approx_label():
    summary = _summary("event_driven")
    assert "Embedded in returns" not in summary
    assert "(approx)" not in summary
    assert "Total Commission" in summary
    assert "Total Slippage" in summary
