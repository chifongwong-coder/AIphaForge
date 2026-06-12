"""v2.8.6 Commit G — annualization mismatch warning.

``trading_days`` is bars-per-year throughout the codebase; the engine
warns (never auto-corrects) when the data's observed bar density is
more than 3x off the effective value. Density-based inference
(len / years spanned), not median-interval — see
``utils.infer_bars_per_year``. Tests match by message substring (repo
convention); intraday fixtures build indexes inline with
``pd.date_range``.
"""
from __future__ import annotations

import warnings

import pandas as pd

from aiphaforge import BacktestEngine
from aiphaforge.utils import infer_bars_per_year

from .conftest import make_ohlcv

_MISMATCH = "annualization mismatch"


def _hourly_24_7(periods: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq="h")
    closes = pd.Series([100.0] * periods, index=idx)
    return pd.DataFrame({
        "open": closes, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "volume": [1e6] * periods,
    })


def _run(data: pd.DataFrame, **engine_kwargs) -> list:
    signals = pd.Series(1.0, index=data.index)
    engine = BacktestEngine(mode="event_driven", **engine_kwargs)
    engine.set_signals(signals)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.run(data)
    return [w for w in caught if _MISMATCH in str(w.message)]


def test_hourly_crypto_default_252_warns():
    data = _hourly_24_7(24 * 15)  # 15 days of 24/7 hourly bars
    mismatches = _run(data)  # default trading_days=252
    assert len(mismatches) == 1
    message = str(mismatches[0].message)
    assert "trading_days=252" in message
    # Suggested value is ~8760 bars/year for 24/7 hourly data.
    assert "trading_days=8" in message


def test_daily_data_default_no_warning():
    data = make_ohlcv(60)  # business-daily: density ~265/year
    assert _run(data) == []


def test_hourly_explicit_correct_no_warning():
    data = _hourly_24_7(24 * 15)
    assert _run(data, trading_days=8760) == []


def test_short_index_no_opinion():
    idx = pd.date_range("2024-01-01", periods=20, freq="h")
    assert infer_bars_per_year(idx) is None
    data = _hourly_24_7(20)
    assert _run(data) == []
