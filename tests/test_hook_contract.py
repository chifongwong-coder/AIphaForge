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


def test_vectorized_lifecycle_fires_once_single_asset():
    hook = _RecordingHook()
    data = _make_data()
    signals = pd.Series(np.nan, index=data.index, dtype=float)
    signals.iloc[5] = 1.0
    eng = BacktestEngine(mode="vectorized", hooks=[hook])
    eng.set_signals(signals)
    eng.run(data)

    assert len(hook.start_calls) == 1
    assert len(hook.end_calls) == 1


def test_vectorized_lifecycle_fires_once_multi_asset():
    hook = _RecordingHook()
    data_a = _make_data(seed=1)
    data_b = _make_data(seed=2)
    signals_a = pd.Series(np.nan, index=data_a.index, dtype=float)
    signals_b = pd.Series(np.nan, index=data_b.index, dtype=float)
    signals_a.iloc[5] = 1.0
    signals_b.iloc[5] = 1.0
    eng = BacktestEngine(mode="vectorized", hooks=[hook])
    eng.set_signals({"AAA": signals_a, "BBB": signals_b})
    eng.run({"AAA": data_a, "BBB": data_b})

    assert len(hook.start_calls) == 1
    assert len(hook.end_calls) == 1
    assert hook.start_calls[0].symbols == ["AAA", "BBB"]


def test_event_driven_end_hook_fires_on_mid_loop_exception():
    """v1.9.7: end-hook fires even when the engine raises mid-run.

    Pre-fix: post-loop end-hook was unreachable on exception, so
    LatencyHook queues / open file handles / dashboards leaked.
    """

    class _CrashingHook(BacktestHook):
        def __init__(self) -> None:
            self.start_calls = 0
            self.end_calls = 0

        def on_backtest_start(self, ctx: LifecycleContext) -> None:
            self.start_calls += 1

        def on_pre_signal(self, context):
            # Crash on bar 3 to simulate a mid-loop failure.
            if context.bar_index == 3:
                raise RuntimeError("simulated mid-loop failure")

        def on_backtest_end(self, ctx: LifecycleContext) -> None:
            self.end_calls += 1

    hook = _CrashingHook()
    data = _make_data()
    signals = pd.Series(np.nan, index=data.index, dtype=float)
    signals.iloc[1] = 1.0
    eng = BacktestEngine(mode="event_driven", hooks=[hook])
    eng.set_signals(signals)
    with pytest.raises(RuntimeError, match="simulated mid-loop failure"):
        eng.run(data)
    assert hook.start_calls == 1
    assert hook.end_calls == 1, (
        "end-hook should fire even when the loop raises (try/finally)")


def test_event_driven_end_hook_fires_for_all_hooks_when_one_raises():
    """v1.9.7 commit 5 probe: with multiple hooks, if one raises mid-loop,
    ALL hooks (including the crashing one and any that came before/after
    in the dispatch list) get on_backtest_end. This guarantees other
    hooks' cleanup runs even when a peer crashes.
    """

    class _CrashOnBar3Hook(BacktestHook):
        def __init__(self) -> None:
            self.start_calls = 0
            self.end_calls = 0

        def on_backtest_start(self, ctx: LifecycleContext) -> None:
            self.start_calls += 1

        def on_pre_signal(self, context):
            if context.bar_index == 3:
                raise RuntimeError("crash")

        def on_backtest_end(self, ctx: LifecycleContext) -> None:
            self.end_calls += 1

    class _PassiveHook(BacktestHook):
        def __init__(self) -> None:
            self.start_calls = 0
            self.end_calls = 0

        def on_backtest_start(self, ctx: LifecycleContext) -> None:
            self.start_calls += 1

        def on_backtest_end(self, ctx: LifecycleContext) -> None:
            self.end_calls += 1

    crash = _CrashOnBar3Hook()
    passive_before = _PassiveHook()
    passive_after = _PassiveHook()

    data = _make_data()
    signals = pd.Series(np.nan, index=data.index, dtype=float)
    signals.iloc[1] = 1.0
    eng = BacktestEngine(
        mode="event_driven",
        hooks=[passive_before, crash, passive_after],
    )
    eng.set_signals(signals)
    with pytest.raises(RuntimeError, match="crash"):
        eng.run(data)

    # All three hooks must see exactly one start AND one end.
    for h, name in [(passive_before, "passive_before"),
                    (crash, "crash"),
                    (passive_after, "passive_after")]:
        assert h.start_calls == 1, f"{name}.start_calls={h.start_calls}"
        assert h.end_calls == 1, (
            f"{name}.end_calls={h.end_calls} — must fire even when "
            f"a peer hook crashed mid-loop")


def test_event_driven_end_hook_does_NOT_fire_when_start_raises():
    """v1.9.7: if a start hook itself raises, end should NOT fire.

    The started_hooks flag is only flipped after the start dispatch
    loop completes; the finally block checks the flag.
    """

    class _CrashingStartHook(BacktestHook):
        def __init__(self) -> None:
            self.end_calls = 0

        def on_backtest_start(self, ctx: LifecycleContext) -> None:
            raise RuntimeError("start crash")

        def on_backtest_end(self, ctx: LifecycleContext) -> None:
            self.end_calls += 1

    hook = _CrashingStartHook()
    data = _make_data()
    signals = pd.Series(np.nan, index=data.index, dtype=float)
    signals.iloc[1] = 1.0
    eng = BacktestEngine(mode="event_driven", hooks=[hook])
    eng.set_signals(signals)
    with pytest.raises(RuntimeError, match="start crash"):
        eng.run(data)
    assert hook.end_calls == 0


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
