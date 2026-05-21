"""v2.8.2 M3 — BacktestEngine.set_signals validates at the boundary.

v2.8.1's set_signals accepted any pd.Series or dict[str, pd.Series]
without validation. Duplicate-index, non-DatetimeIndex, or non-numeric
Series surfaced as confusing crashes deep in the run loop.

v2.8.2 calls validate_signal_series at the boundary via try/except +
re-raise pattern that interpolates the offending symbol name on the
dict path. Respects engine data_validation="none" for parity with
validate_ohlcv convention.

Authoritative rejection set:
`tests/_helpers/expected_rejections.md` (committed in this commit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge import BacktestEngine


def _good_signals(n: int = 20) -> pd.Series:
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    s = pd.Series(np.nan, index=idx, dtype=float)
    s.iloc[0] = 1.0
    return s


# --------------------------------------------------------------------
# Test 1 — duplicate-index Series rejected at boundary
# --------------------------------------------------------------------

def test_set_signals_rejects_duplicate_index_series():
    eng = BacktestEngine(initial_capital=100_000)
    idx = pd.DatetimeIndex([
        "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04",
    ])
    bad = pd.Series([1.0, np.nan, np.nan, np.nan], index=idx, dtype=float)
    with pytest.raises(ValueError, match="duplicate"):
        eng.set_signals(bad)


# --------------------------------------------------------------------
# Test 2 — non-DatetimeIndex Series rejected at boundary
# --------------------------------------------------------------------

def test_set_signals_rejects_non_datetime_index_series():
    eng = BacktestEngine(initial_capital=100_000)
    bad = pd.Series([1.0, np.nan, np.nan, np.nan])  # RangeIndex
    with pytest.raises(TypeError, match="DatetimeIndex"):
        eng.set_signals(bad)


# --------------------------------------------------------------------
# Test 3 — dict path validates each Series; error names the symbol
# --------------------------------------------------------------------

def test_set_signals_dict_validates_each_series_with_symbol_name():
    eng = BacktestEngine(initial_capital=100_000)
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-03"])
    good = _good_signals()
    bad = pd.Series([1.0, np.nan, np.nan], index=idx, dtype=float)
    with pytest.raises(ValueError, match="signals\\['BAD'\\]"):
        eng.set_signals({"GOOD": good, "BAD": bad})


# --------------------------------------------------------------------
# Test 4 — data_validation="none" skips validation (escape hatch)
# --------------------------------------------------------------------

def test_set_signals_respects_data_validation_none():
    """Parity with validate_ohlcv convention: when the engine is
    constructed with data_validation='none', boundary validation
    is bypassed (legitimate edge cases can opt out)."""
    eng = BacktestEngine(
        initial_capital=100_000, data_validation="none",
    )
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-03"])
    bad = pd.Series([1.0, np.nan, np.nan], index=idx, dtype=float)
    # Should NOT raise — validation is skipped.
    eng.set_signals(bad)


# --------------------------------------------------------------------
# Test 5 — non-Series / non-dict input rejected with clean TypeError
# --------------------------------------------------------------------

def test_set_signals_rejects_non_series_non_dict():
    eng = BacktestEngine(initial_capital=100_000)
    with pytest.raises(TypeError, match="pd.Series or dict"):
        eng.set_signals([1, 0, 1, 0])  # list, not Series/dict
    with pytest.raises(TypeError, match="pd.Series or dict"):
        eng.set_signals(np.array([1, 0, 1, 0]))  # ndarray
