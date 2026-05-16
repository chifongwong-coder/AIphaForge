"""v2.7 Commit D — set_signals strict refusal of DataFrame input.

Plan v2.7 §5.5 anti-list: set_signals continues to accept ONLY
pd.Series (single-asset) or dict[str, pd.Series] (multi-asset).
NEVER pd.DataFrame. Wide-layout input goes through set_signals_wide.

Previously (v2.6) a DataFrame would crash deep inside the engine at
_get_signals's .reindex call with a confusing AttributeError. v2.7
tightens to a clean TypeError at the boundary.
"""
from __future__ import annotations

import pandas as pd
import pytest

from aiphaforge.engine import BacktestEngine


def test_set_signals_rejects_dataframe():
    engine = BacktestEngine()
    df = pd.DataFrame(
        {"AAPL": [1.0, 0.0], "MSFT": [0.0, 1.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    with pytest.raises(TypeError) as excinfo:
        engine.set_signals(df)
    msg = str(excinfo.value)
    assert "set_signals_wide" in msg, f"error must point to set_signals_wide; got: {msg}"
    assert "breaking change" in msg, f"error must flag breaking change from v2.6; got: {msg}"


def test_set_signals_series_still_works():
    # Regression: existing single-asset Series call path unchanged.
    engine = BacktestEngine()
    series = pd.Series(
        [1.0, 0.0, 1.0, 0.0],
        index=pd.date_range("2024-01-01", periods=4),
    )
    engine.set_signals(series)
    assert engine._signals is series


def test_set_signals_dict_still_works():
    # Regression: existing multi-asset dict call path unchanged.
    engine = BacktestEngine()
    idx = pd.date_range("2024-01-01", periods=3)
    sig_dict = {
        "AAPL": pd.Series([1.0, 0.0, 1.0], index=idx),
        "MSFT": pd.Series([0.0, 1.0, 0.0], index=idx),
    }
    engine.set_signals(sig_dict)
    assert engine._signals is sig_dict
