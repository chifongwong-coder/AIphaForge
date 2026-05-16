"""v2.7 Commit H — wide-input lookahead-safety + edge cases.

3 tests required by v2.7 plan §"Commit H":

- Wide DF index extends BEYOND data → engine truncates, no lookahead
- Wide DF index COARSER than data → NaN-as-hold ffills (existing engine
  contract carries through)
- tz-aware wide DF + tz-naive data → validate_signal_wide(forbid_tz=True)
  raises clean ValueError at the engine setter
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.engine import BacktestEngine


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


def test_wide_df_extends_beyond_data_truncates_no_lookahead():
    # Data covers 30 bars; wide DF covers 60 bars (data + 30 extra future).
    # Engine should truncate the materialized signal to data's range.
    sig_idx = pd.bdate_range("2024-01-01", periods=60)
    data_idx = sig_idx[:30]
    data = {"AAPL": _ohlcv(data_idx)}
    signal = pd.DataFrame({"AAPL": [1.0] * 60}, index=sig_idx)
    signal.iloc[0] = 0.0

    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_signals_wide(signal)
    result = engine.run(data)

    # No trade dates beyond data_idx[-1] — truncation enforced by reindex
    # inside wide_to_signal_dict (data_dict drives the per-symbol index).
    if result.trades:
        max_trade_date = max(t.entry_date for t in result.trades)
        assert max_trade_date <= data_idx[-1], (
            f"wide DF should truncate at data range; got trade at "
            f"{max_trade_date} > data end {data_idx[-1]}"
        )


def test_wide_df_coarser_than_data_holds_via_NaN():
    # Wide DF is monthly (3 dates); data is daily.
    # On non-month-end days the materialized signal must be NaN — the
    # reindex against data's daily index introduces NaN on every bar
    # not in the wide DF's index. The engine treats NaN as hold.
    from aiphaforge.signals import wide_to_signal_dict
    daily_idx = pd.bdate_range("2024-01-01", periods=60)
    monthly_idx = pd.DatetimeIndex(["2024-01-02", "2024-02-01", "2024-03-01"])
    data = {"AAPL": _ohlcv(daily_idx)}
    signal = pd.DataFrame({"AAPL": [1.0, 0.0, 1.0]}, index=monthly_idx)

    # Direct adapter behavior: only the 3 monthly dates carry non-NaN.
    signal_dict = wide_to_signal_dict(signal, data)
    aapl_signals = signal_dict["AAPL"]
    non_nan_dates = aapl_signals.dropna().index
    assert list(non_nan_dates) == list(monthly_idx), (
        f"materialized signal must be non-NaN ONLY on the 3 monthly "
        f"dates; got non-NaN on: {list(non_nan_dates)[:6]}..."
    )
    assert aapl_signals.isna().sum() == len(daily_idx) - 3

    # End-to-end via engine: no crash, signal series propagates NaN-hold.
    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_signals_wide(signal)
    engine.run(data)


def test_wide_df_tz_aware_raises_at_setter():
    # tz-aware wide DF + tz-naive data → set_signals_wide raises cleanly
    # via validate_signal_wide(forbid_tz=True).
    tz_idx = pd.date_range("2024-01-01", periods=10, tz="US/Eastern")
    signal = pd.DataFrame({"AAPL": [1.0] * 10}, index=tz_idx)
    engine = BacktestEngine()
    with pytest.raises(ValueError, match="tz-aware.*tz_localize"):
        engine.set_signals_wide(signal)


def test_target_weights_wide_tz_aware_raises_at_setter():
    # Same forbid_tz protection on the target-weights setter.
    tz_idx = pd.date_range("2024-01-01", periods=10, tz="US/Eastern")
    weights = pd.DataFrame({"AAPL": [0.5] * 10}, index=tz_idx)
    engine = BacktestEngine()
    with pytest.raises(ValueError, match="tz-aware.*tz_localize"):
        engine.set_target_weights_wide(weights)
