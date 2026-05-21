"""v2.8.1 H2 — validate_ohlcv hard-fails on duplicate timestamps.

v2.8.0 emitted a warning for duplicate-index OHLCV in 'warn' mode and
let the data through. The event-driven core then crashed deep inside
the loop with a TypeError because df.loc[ts, "close"] returns a Series
on a duplicate index. v2.8.1 raises at validation time with an
actionable dedupe recipe, regardless of validation_level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge import BacktestEngine
from aiphaforge.utils import validate_ohlcv


def _dup_ohlcv() -> pd.DataFrame:
    """Build an OHLCV frame with a duplicate timestamp in the middle."""
    idx = pd.DatetimeIndex(
        [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-04",  # duplicate
            "2024-01-05",
        ]
    )
    close = np.array([100.0, 101.0, 102.0, 102.5, 103.0])
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": 1e6},
        index=idx,
    )


@pytest.mark.parametrize("validation_level", ["strict", "warn", "none"])
def test_validate_ohlcv_duplicate_index_raises_at_all_levels(validation_level):
    """v2.8.1 H2: duplicate timestamps raise ValueError regardless of
    validation_level — the recipe is included in the error message."""
    df = _dup_ohlcv()
    with pytest.raises(ValueError, match="duplicate"):
        validate_ohlcv(df, validation_level=validation_level)


def test_engine_event_driven_rejects_dup_data_at_validation_not_loop():
    """Engine event-driven on dup data raises ValueError at validation,
    NOT TypeError mid-loop (the v2.8.0 failure mode).

    v2.8.2 M3 added set_signals boundary validation; to keep this test
    focused on validate_ohlcv (the H2 contract), the signals get a
    clean (non-dup) DatetimeIndex; the OHLCV data carries the dup.
    Engine.run validates the data and raises before the signal index
    matters.
    """
    df = _dup_ohlcv()
    clean_idx = pd.date_range("2024-01-02", periods=5, freq="D")
    eng = BacktestEngine(
        initial_capital=100_000, mode="event_driven", data_validation="warn",
    )
    eng.set_signals(pd.Series(0.0, index=clean_idx))
    with pytest.raises(ValueError, match="duplicate"):
        eng.run(df)
