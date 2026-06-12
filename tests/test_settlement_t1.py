"""v2.8.6 Commits B/C — T+1 settlement constraint.

Commit B: config/engine plumbing and validation.
Commit C: broker-side enforcement (bought-today bucket, fill-time
sellable check, IOC/FOK/exit-rule/margin interactions).

T+1 is the SSE/SZSE cash-equity rule: shares bought today cannot be
sold the same calendar day. Intraday fixtures build their indexes
inline with ``pd.date_range`` (the shared ``tests/_helpers/ohlcv.py``
constructors are business-daily only).
"""
from __future__ import annotations

import pandas as pd
import pytest

from aiphaforge import BacktestEngine

from .conftest import make_ohlcv

# ---------------------------------------------------------------------------
# Commit B — config + engine plumbing
# ---------------------------------------------------------------------------


def test_invalid_settlement_value_raises():
    with pytest.raises(ValueError, match=r"t\+0.*t\+1"):
        BacktestEngine(settlement="t+2")


def test_t1_with_vectorized_mode_raises():
    data = make_ohlcv(30)
    signals = pd.Series(1.0, index=data.index)
    engine = BacktestEngine(settlement="t+1")  # default mode: vectorized
    engine.set_signals(signals)
    with pytest.raises(ValueError, match="event_driven"):
        engine.run(data)


def test_asset_settlements_t1_vectorized_raises():
    data = make_ohlcv(30)
    signals = pd.Series(1.0, index=data.index)
    engine = BacktestEngine(
        settlement="t+0", asset_settlements={"X": "t+1"})
    engine.set_signals(signals)
    with pytest.raises(ValueError, match="event_driven"):
        engine.run(data)
