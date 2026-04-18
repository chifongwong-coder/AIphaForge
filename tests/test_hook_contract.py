"""Hook lifecycle contract tests (v1.9.6).

Asserts:
* `BacktestHook.on_backtest_start` and `on_backtest_end` fire exactly
  once per backtest, regardless of symbol count (B2 + R5).
* The new ``LifecycleContext``-based signature is the supported one.
* User subclasses still using the legacy ``(data, symbol, *, config)``
  signature continue to work via a runtime-detected adapter, which
  emits a one-time ``DeprecationWarning`` per subclass.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from aiphaforge import (
    BacktestEngine,
    BacktestHook,
    LifecycleContext,
)


def _make_data(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )
    return df


class _RecordingHook(BacktestHook):
    def __init__(self) -> None:
        self.start_calls: list[LifecycleContext] = []
        self.end_calls: list[LifecycleContext] = []

    def on_backtest_start(self, ctx: LifecycleContext) -> None:
        self.start_calls.append(ctx)

    def on_backtest_end(self, ctx: LifecycleContext) -> None:
        self.end_calls.append(ctx)


class _LegacyHook(BacktestHook):
    """Pre-v1.9.6 subclass that hasn't migrated."""

    def __init__(self) -> None:
        self.start_calls: list[tuple] = []
        self.end_calls = 0

    def on_backtest_start(self, data, symbol, *, config=None):
        self.start_calls.append((data, symbol, config))

    def on_backtest_end(self):
        self.end_calls += 1


def test_lifecycle_fires_once_single_asset():
    hook = _RecordingHook()
    data = _make_data()
    signals = pd.Series(np.nan, index=data.index, dtype=float)
    signals.iloc[5] = 1.0
    eng = BacktestEngine(mode="event_driven", hooks=[hook])
    eng.set_signals(signals)
    eng.run(data)

    assert len(hook.start_calls) == 1
    assert len(hook.end_calls) == 1
    start_ctx = hook.start_calls[0]
    end_ctx = hook.end_calls[0]
    assert start_ctx.phase == "start"
    assert end_ctx.phase == "end"
    assert start_ctx.timestamp == data.index[0]
    assert end_ctx.timestamp == data.index[-1]
    assert start_ctx.primary_symbol == start_ctx.symbols[0]


def test_lifecycle_fires_once_multi_asset():
    hook = _RecordingHook()
    data_a = _make_data(seed=1)
    data_b = _make_data(seed=2)
    signals_a = pd.Series(np.nan, index=data_a.index, dtype=float)
    signals_b = pd.Series(np.nan, index=data_b.index, dtype=float)
    signals_a.iloc[5] = 1.0
    signals_b.iloc[5] = 1.0

    eng = BacktestEngine(mode="event_driven", hooks=[hook])
    eng.set_signals({"AAA": signals_a, "BBB": signals_b})
    eng.run({"AAA": data_a, "BBB": data_b})

    assert len(hook.start_calls) == 1
    assert len(hook.end_calls) == 1
    assert hook.start_calls[0].symbols == ["AAA", "BBB"]
    assert hook.start_calls[0].primary_symbol == "AAA"


def test_legacy_signature_still_works_with_warning():
    hook = _LegacyHook()
    data = _make_data()
    signals = pd.Series(np.nan, index=data.index, dtype=float)
    signals.iloc[5] = 1.0
    eng = BacktestEngine(mode="event_driven", hooks=[hook])
    eng.set_signals(signals)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eng.run(data)

    deprecations = [
        w for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "LifecycleContext" in str(w.message)
    ]
    # Exactly one warning per legacy method (start + end), once for the
    # whole subclass — even though subsequent calls might re-fire, only
    # this single run hit each method once.
    methods_warned = {w.message.args[0].split(".")[1].split(" ")[0]
                      for w in deprecations}
    assert "on_backtest_start" in methods_warned
    assert "on_backtest_end" in methods_warned
    assert hook.start_calls and hook.start_calls[0][1] == "default"
    assert hook.end_calls == 1


def test_legacy_deprecation_warning_emitted_once_per_subclass():
    # Use a fresh subclass so the module-level dedup set is clean.
    class _Once(BacktestHook):
        def on_backtest_start(self, data, symbol, *, config=None):
            pass

        def on_backtest_end(self):
            pass

    hook1 = _Once()
    hook2 = _Once()
    data = _make_data()
    signals = pd.Series(np.nan, index=data.index, dtype=float)
    signals.iloc[5] = 1.0

    eng = BacktestEngine(mode="event_driven", hooks=[hook1])
    eng.set_signals(signals)
    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter("always")
        eng.run(data)

    eng2 = BacktestEngine(mode="event_driven", hooks=[hook2])
    eng2.set_signals(signals)
    with warnings.catch_warnings(record=True) as second:
        warnings.simplefilter("always")
        eng2.run(data)

    first_dep = [
        w for w in first
        if issubclass(w.category, DeprecationWarning)
        and "LifecycleContext" in str(w.message)
    ]
    second_dep = [
        w for w in second
        if issubclass(w.category, DeprecationWarning)
        and "LifecycleContext" in str(w.message)
    ]
    assert len(first_dep) == 2  # start + end
    assert len(second_dep) == 0  # already warned for this class


@pytest.mark.parametrize("phase", ["start", "end"])
def test_lifecycle_context_provides_data_dict(phase):
    hook = _RecordingHook()
    data = _make_data()
    signals = pd.Series(np.nan, index=data.index, dtype=float)
    signals.iloc[5] = 1.0
    eng = BacktestEngine(mode="event_driven", hooks=[hook])
    eng.set_signals(signals)
    eng.run(data)

    calls = hook.start_calls if phase == "start" else hook.end_calls
    ctx = calls[0]
    assert ctx.primary_symbol in ctx.data_dict
    assert ctx.data_dict[ctx.primary_symbol] is ctx.primary_data
