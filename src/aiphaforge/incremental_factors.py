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
