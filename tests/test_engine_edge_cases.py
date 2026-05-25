"""v2.8.5 L2 — engine edge-case contracts.

Three deliberately small tests that pin behaviors which were
previously unasserted anywhere in the suite:

* N=1 bar input — engine runs without raising, returns a 1-bar
  equity curve with ``total_return == 0`` and zero trades.
* All-NaN signal series — under the NaN-as-hold contract, no trades
  execute and final equity equals initial capital.
* Constant-drift price path — buy-and-hold total return matches the
  closed-form ``(price_end - price_start) / price_start`` value to
  floating-point precision.

These tests use the shared OHLCV helpers from
:mod:`tests._helpers.ohlcv` introduced in commit B.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge import BacktestEngine
from aiphaforge.fees import ZeroFeeModel
from tests._helpers.ohlcv import make_close_only, make_ohlcv


def test_engine_n_eq_1_bar_behavior() -> None:
    """Pin the engine's N=1 contract.

    With a single-bar OHLCV frame and a flat-long signal, the engine
    must complete without raising, emit a 1-element equity curve
    equal to the initial capital, report zero total return, and
    record no trades (a single bar leaves no room for entry +
    settlement).
    """
    data = pd.DataFrame(
        {
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1_000_000.0],
        },
        index=pd.bdate_range("2024-01-01", periods=1),
    )
    eng = BacktestEngine(
        fee_model=ZeroFeeModel(), initial_capital=100_000.0,
    )
    signals = pd.Series([1.0], index=data.index, dtype=float)
    eng.set_signals(signals)
    result = eng.run(data)

    assert len(result.equity_curve) == 1
    assert result.equity_curve.iloc[0] == 100_000.0
    assert result.total_return == 0.0
    assert len(result.trades) == 0


def test_engine_all_nan_signal_produces_no_trades() -> None:
    """All-NaN signal series ==> zero trades + flat equity curve.

    The NaN-as-hold contract states that NaN signal values mean
    "do not change the current position". Starting flat and never
    receiving a non-NaN signal must therefore produce zero trades
    and leave equity equal to initial capital across every bar.
    """
    data = make_ohlcv(periods=60)
    signals = pd.Series(np.nan, index=data.index, dtype=float)
    eng = BacktestEngine(
        fee_model=ZeroFeeModel(), initial_capital=100_000.0,
    )
    eng.set_signals(signals)
    result = eng.run(data)

    assert len(result.trades) == 0
    assert result.total_return == 0.0
    # Every bar's equity equals the starting capital because no trade
    # ever shifts the position off cash.
    assert (result.equity_curve == 100_000.0).all()


def test_engine_constant_drift_total_return_matches_closed_form() -> None:
    """Buy-and-hold on a deterministic drift path ==> closed-form.

    With prices ``base * (1 + 0.001 * arange(periods))``, a long-only
    buy at bar 0 followed by passive hold to the last bar must
    produce a ``total_return`` equal to
    ``(price_last - price_first) / price_first`` up to floating-point
    accumulation noise.
    """
    periods = 252
    data = make_close_only(periods=periods)
    signals = pd.Series(np.nan, index=data.index, dtype=float)
    signals.iloc[0] = 1.0
    eng = BacktestEngine(
        fee_model=ZeroFeeModel(), initial_capital=100_000.0,
    )
    eng.set_signals(signals)
    result = eng.run(data)

    price_first = float(data["close"].iloc[0])
    price_last = float(data["close"].iloc[-1])
    expected = (price_last - price_first) / price_first
    assert result.total_return == pytest.approx(expected, abs=1e-9)
