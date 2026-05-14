"""v2.6: stateful per-bar factor computation.

Parallel API to ``aiphaforge.factors.BaseFactor`` (batch). Each
concrete ``IncrementalFactor`` advances a per-(factor, symbol)
``FactorState`` through ``update(bar, state)`` calls and emits one
value per bar. Designed for live trading and event-driven
simulation; ``run_all(data)`` is provided as a batch-mode
convenience for testing and replay.

Engine integration ships in v2.7+; v2.6 users drive ``update()``
manually or use ``run_all()`` for full-data replay.

Design doc: ``docs/AIphaForge_Incremental_Factor_Design_v1.0.md``
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class FactorState:
    """Per-(factor, symbol) state carried across bars.

    bar_count semantics (PINNED):
        Incremented AFTER ``update()`` returns. So during the
        i-th call to update():
            entry: state.bar_count == i
            exit:  new_state.bar_count == i + 1
        Warmup check is therefore ``bar_count < window`` (strict);
        a factor needing W bars of data emits NaN until
        ``state.bar_count >= window``.

    Subclasses extend with factor-specific fields (rolling window
    deque, running sum, Welford mean+M2, etc). Subclass MUST be
    decorated ``@dataclass`` and MUST give defaults to any new
    fields (because ``bar_count`` already has a default — Python
    dataclass inheritance forbids non-default fields after
    default fields).

    Pickle:
        In-process pickling for parallel backtests IS a contract.
        Cross-version pickling is NOT — adding/renaming a field
        in v2.7 will break v2.6-pickled state. The
        ``__pickle_version__`` class attr lets callers detect
        such breakage rather than silently corrupt.
    """

    bar_count: int = 0
    __pickle_version__: int = 1


class IncrementalFactor(ABC):
    """Stateful per-bar factor computation.

    Each concrete factor must:
      1. Expose a ``name`` (for logging + factor registry).
      2. Implement ``initial_state()`` returning a fresh
         ``FactorState`` (or subclass thereof).
      3. Implement ``update(bar_row, state)`` returning
         ``(value, new_state)`` where
         ``new_state.bar_count == state.bar_count + 1``.
      4. Ship a batch-equivalence test using
         ``tests._helpers.incremental.assert_batch_incremental_equivalent``
         with the per-factor ``(rtol, atol)`` from the design doc §3.
    """

    name: str = "incremental_factor"

    @abstractmethod
    def initial_state(self) -> FactorState:
        """Return a fresh state for a new (factor, symbol) pair."""

    @abstractmethod
    def update(
        self, bar_row: pd.Series, state: FactorState,
    ) -> Tuple[float, FactorState]:
        """Advance state with one bar; return (value, new_state).

        ``value`` is NaN while warmup is not yet satisfied (per
        the §R18 convention). ``new_state.bar_count`` MUST equal
        ``state.bar_count + 1``.
        """

    def run_all(self, data: pd.DataFrame) -> pd.Series:
        """Drive ``update()`` across an entire DataFrame.

        Used by the batch-equivalence harness and by single-symbol
        replay workflows. For large fixtures this is O(N) Python
        overhead; a v2.7+ perf pass may vectorize hot paths.
        """
        state = self.initial_state()
        out = []
        for _, row in data.iterrows():
            value, state = self.update(row, state)
            out.append(value)
        return pd.Series(out, index=data.index, dtype=float)

    # ------------------------------------------------------------------
    # Stateful convenience API — for callers (typically strategy
    # authors) who treat the factor as if it owned its own state. The
    # main update(bar, state) API remains stateless; these helpers
    # manage ``self._state`` for the common single-symbol use case.
    # ------------------------------------------------------------------

    def update_params(self, **kwargs) -> None:
        """Apply runtime parameter changes; reset internal state.

        Per the v2.6 design doc Q4 decision: clear + rebuild (no
        local invalidation). After ``update_params``, ``self._state``
        is a clean slate; the caller MUST call ``rewarmup(history)``
        before resuming live ``update()`` calls, otherwise the next
        ``window`` bars emit NaN.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError(
                    f"Unknown param '{k}' for {type(self).__name__}"
                )
            setattr(self, k, v)
        self._state = self.initial_state()

    def rewarmup(self, history: pd.DataFrame) -> None:
        """Replay history to rebuild ``self._state`` under current params.

        Convenience method to avoid every caller hand-rolling the
        same per-bar update loop after ``update_params(...)``.
        Equivalent to manually calling ``initial_state()`` then
        feeding each row of ``history`` through ``update()``.

        If ``history`` is shorter than the factor's warmup window,
        rewarmup completes but ``self._state`` is still in warmup —
        subsequent ``update()`` calls will return NaN until enough
        bars accumulate.
        """
        self._state = self.initial_state()
        for _, bar in history.iterrows():
            _, self._state = self.update(bar, self._state)


# ---------------------------------------------------------------------------
# Concrete v2.6 MVP factors (Commits C-G)
# ---------------------------------------------------------------------------


@dataclass
class _RollingMeanState(FactorState):
    window_buf: tuple = ()
    running_sum: float = 0.0


class RollingMeanIncremental(IncrementalFactor):
    """Rolling mean of ``close`` over a fixed window.

    Algorithm: running sum + tuple-backed sliding window. Emits
    NaN until ``window`` bars have been observed (matches
    ``pd.Series.rolling(window).mean()`` warmup convention).
    """

    def __init__(self, window: int):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window
        self.name = f"rolling_mean_{window}"

    def initial_state(self) -> _RollingMeanState:
        return _RollingMeanState()

    def update(
        self, bar_row: pd.Series, state: _RollingMeanState,
    ) -> Tuple[float, _RollingMeanState]:
        x = float(bar_row["close"])
        if len(state.window_buf) < self.window:
            new_buf = state.window_buf + (x,)
            new_sum = state.running_sum + x
        else:
            popped = state.window_buf[0]
            new_buf = state.window_buf[1:] + (x,)
            new_sum = state.running_sum + x - popped
        value = new_sum / self.window if len(new_buf) >= self.window else float("nan")
        new_state = _RollingMeanState(
            bar_count=state.bar_count + 1,
            window_buf=new_buf,
            running_sum=new_sum,
        )
        return value, new_state


@dataclass
class _VolumeZScoreState(FactorState):
    window_buf: tuple = ()


class VolumeZScoreIncremental(IncrementalFactor):
    """Per-symbol rolling z-score of ``volume`` over a fixed window.

    Matches v2.4 ``VolumeZScoreFactor``: ``(v - rolling_mean) /
    rolling_std`` with ``ddof=1`` and ``min_periods=window``. This
    is a per-symbol time-series transform (NOT cross-sectional).
    Cross-sectional z-score is deferred to v2.7+ / v3.0.
    """

    def __init__(self, window: int):
        if window < 2:
            raise ValueError(
                f"window must be >= 2 for sample std (ddof=1), got {window}"
            )
        self.window = window
        self.name = f"volume_zscore_{window}"

    def initial_state(self) -> _VolumeZScoreState:
        return _VolumeZScoreState()

    def update(
        self, bar_row: pd.Series, state: _VolumeZScoreState,
    ) -> Tuple[float, _VolumeZScoreState]:
        v = float(bar_row["volume"])
        if len(state.window_buf) < self.window:
            new_buf = state.window_buf + (v,)
        else:
            new_buf = state.window_buf[1:] + (v,)
        if len(new_buf) < self.window:
            value = float("nan")
        else:
            mean = sum(new_buf) / self.window
            std = float(np.std(new_buf, ddof=1))
            value = (v - mean) / std if std != 0 else float("nan")
        new_state = _VolumeZScoreState(
            bar_count=state.bar_count + 1,
            window_buf=new_buf,
        )
        return value, new_state


@dataclass
class _RSIState(FactorState):
    avg_gain: float = 0.0
    avg_loss: float = 0.0
    prev_close: float = float("nan")


class RSIIncremental(IncrementalFactor):
    """Relative Strength Index (Wilder smoothing via EWM, ``adjust=False``).

    Matches v2.4 ``RSIFactor`` (which delegates to ``indicators.RSI``)
    byte-for-byte at the SMA-seed → recursive-Wilder transition cliff.

    The v2.4 reference uses ``ewm(alpha=1/period, min_periods=period,
    adjust=False).mean()`` on the per-bar gains and losses. The
    incremental form is the same recursion driven one bar at a time:

        avg_gain[t] = alpha * gain[t] + (1 - alpha) * avg_gain[t-1]
        avg_loss[t] = alpha * loss[t] + (1 - alpha) * avg_loss[t-1]

    where ``gain[0] = loss[0] = 0`` (delta is undefined on the first
    bar). Output is NaN for bars 0..(period - 2); bar ``period - 1``
    is the first non-NaN value (matches pandas
    ``ewm(min_periods=period).mean()`` semantics: emit once
    ``period`` non-NaN inputs have been observed, including bar 0).

    avg_loss == 0 edge case (post-warmup): RSI = 100 (matches the
    explicit fill in v2.4 ``indicators.RSI``).
    """

    def __init__(self, period: int = 14):
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        self.period = period
        self.alpha = 1.0 / period
        self.name = f"rsi_{period}"

    def initial_state(self) -> _RSIState:
        return _RSIState()

    def update(
        self, bar_row: pd.Series, state: _RSIState,
    ) -> Tuple[float, _RSIState]:
        x = float(bar_row["close"])
        if state.bar_count == 0:
            # First bar: no prev_close, delta is NaN → coerce to (0, 0)
            # to match v2.4's `delta.where(delta > 0, 0.0)` behavior.
            gain = 0.0
            loss = 0.0
        else:
            delta = x - state.prev_close
            if delta > 0:
                gain, loss = delta, 0.0
            elif delta < 0:
                gain, loss = 0.0, -delta
            else:
                gain, loss = 0.0, 0.0

        new_avg_gain = self.alpha * gain + (1.0 - self.alpha) * state.avg_gain
        new_avg_loss = self.alpha * loss + (1.0 - self.alpha) * state.avg_loss
        new_bar_count = state.bar_count + 1

        if new_bar_count < self.period:
            value = float("nan")
        elif new_avg_loss == 0.0:
            value = 100.0
        else:
            rs = new_avg_gain / new_avg_loss
            value = 100.0 - 100.0 / (1.0 + rs)

        new_state = _RSIState(
            bar_count=new_bar_count,
            avg_gain=new_avg_gain,
            avg_loss=new_avg_loss,
            prev_close=x,
        )
        return value, new_state


@dataclass
class _MomentumState(FactorState):
    window_buf: tuple = ()


class MomentumIncremental(IncrementalFactor):
    """Gross simple return over the past ``window`` bars.

    Formula: ``close[t] / close[t - window] - 1`` (matches v2.4
    ``MomentumFactor`` byte-for-byte at the closed-form arithmetic
    level). Warmup: first ``window`` bars NaN — bar ``window`` is
    the first non-NaN (because we need ``close[t-window]``,
    indexed at bar 0 when t == window).
    """

    def __init__(self, window: int):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window
        self.name = f"momentum_{window}"

    def initial_state(self) -> _MomentumState:
        return _MomentumState()

    def update(
        self, bar_row: pd.Series, state: _MomentumState,
    ) -> Tuple[float, _MomentumState]:
        x = float(bar_row["close"])
        # Maintain a (window + 1)-element buffer: head is close[t-window].
        if len(state.window_buf) < self.window + 1:
            new_buf = state.window_buf + (x,)
        else:
            new_buf = state.window_buf[1:] + (x,)
        if len(new_buf) < self.window + 1:
            value = float("nan")
        else:
            value = new_buf[-1] / new_buf[0] - 1.0
        new_state = _MomentumState(
            bar_count=state.bar_count + 1,
            window_buf=new_buf,
        )
        return value, new_state


@dataclass
class _RollingStdState(FactorState):
    window_buf: tuple = ()


class RollingStdIncremental(IncrementalFactor):
    """Rolling sample std of ``close`` over a fixed window.

    Algorithm: maintain a sliding window tuple; each update
    recomputes std on the current window via ``numpy.std`` (which
    uses Welford internally since NumPy 1.x and matches pandas
    ``Series.rolling(window).std(ddof=...)`` numerically).

    The per-update complexity is O(window). True O(1) rolling
    Welford requires approximate removal arithmetic that
    accumulates floating-point error over thousands of bars; the
    O(window) per-update cost is the v2.6 MVP trade-off in favor
    of bit-stable equivalence with the pandas batch reference.
    Implementation MUST short-circuit to NaN while
    ``bar_count < window`` to avoid the n=1 / divide-by-zero edge
    case in the variance formula at ddof=1.
    """

    def __init__(self, window: int, ddof: int = 1):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if ddof < 0:
            raise ValueError(f"ddof must be >= 0, got {ddof}")
        self.window = window
        self.ddof = ddof
        self.name = f"rolling_std_{window}"

    def initial_state(self) -> _RollingStdState:
        return _RollingStdState()

    def update(
        self, bar_row: pd.Series, state: _RollingStdState,
    ) -> Tuple[float, _RollingStdState]:
        x = float(bar_row["close"])
        if len(state.window_buf) < self.window:
            new_buf = state.window_buf + (x,)
        else:
            new_buf = state.window_buf[1:] + (x,)
        if len(new_buf) < self.window:
            value = float("nan")
        else:
            value = float(np.std(new_buf, ddof=self.ddof))
        new_state = _RollingStdState(
            bar_count=state.bar_count + 1,
            window_buf=new_buf,
        )
        return value, new_state
