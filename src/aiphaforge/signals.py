"""v2.3 Signal Layer foundation.

Pure utility module for the signal contract. Defines the canonical
``transitions_only`` helper, layout adapters between the engine's
per-symbol ``dict[str, Series]`` and research-friendly wide
``DataFrame[index=datetime, columns=symbol]``, and lightweight
validation.

This module is **engine-agnostic**: it does NOT import
``aiphaforge.engine`` (verified by an AST guard test). The
``BacktestEngine`` continues to consume ``pd.Series`` /
``dict[str, pd.Series]`` via its existing ``set_signals`` / strategy
APIs; the wide-DataFrame conversions live here so engine code stays
unchanged through the v2.x line.

Signal contract (from master plan v1.0 §0.3):

    NaN = hold (no new instruction; engine forward-fills position)
    0   = explicit flatten (close any position)
    1   = long
    -1  = short
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Union

import numpy as np
import pandas as pd


def transitions_only(raw: pd.Series) -> pd.Series:
    """Suppress repeated identical signals; keep state transitions only.

    Converts a per-bar instruction series (where each bar carries the
    desired position state) into a transition series where only state
    changes carry a signal. Bars where the state is unchanged from the
    previous bar's forward-filled value emit NaN (hold).

    Semantic preservation (signal contract):

        NaN = hold (no new instruction)
        0   = explicit flatten (engine closes position)
        1   = long
        -1  = short

    The 0 (flatten) value is treated as a valid TRANSITION — going
    from long → flat or flat → long both produce signals on the
    transition bar.

    Parameters
    ----------
    raw
        Per-bar position-instruction series. May contain NaN (no
        instruction yet), 0 (flat), 1 (long), or -1 (short).

    Returns
    -------
    pd.Series
        Same index as ``raw``. NaN everywhere except at transition
        bars, which carry the new state value.

    Notes
    -----
    Originally lived as ``aiphaforge.strategies._transitions_only``
    (private). Promoted to a public utility in v2.3 so factor-aware
    strategies, ML-direct strategies, and AI-agent signal pipelines
    can all reuse the same transition-suppression logic without
    importing through ``strategies.py``. The private name remains
    as a thin alias (v2.3 Commit C) and is removed in v3.0.
    """
    filled = raw.ffill()
    changed = filled != filled.shift(1)
    signals = pd.Series(np.nan, index=raw.index, dtype=float)
    signals[changed] = filled[changed]
    return signals


def dict_to_signal_wide(
    signals: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Stack per-symbol signal Series into wide DataFrame layout.

    The output index is the union of all per-symbol indices. Symbols
    missing on a given index date receive NaN.

    Parameters
    ----------
    signals
        Per-symbol signal Series. Symbol order in the input mapping
        determines column order in the output.

    Returns
    -------
    pd.DataFrame
        ``index=datetime, columns=symbol``. dtype is float (NaN-able).

    Examples
    --------
    >>> import pandas as pd
    >>> idx = pd.date_range("2024-01-01", periods=3)
    >>> sig = {"A": pd.Series([1, 0, -1], index=idx),
    ...        "B": pd.Series([0, 1, 0], index=idx)}
    >>> dict_to_signal_wide(sig)
                  A    B
    2024-01-01  1.0  0.0
    2024-01-02  0.0  1.0
    2024-01-03 -1.0  0.0
    """
    if not signals:
        return pd.DataFrame()
    return pd.DataFrame(dict(signals))


def wide_to_signal_dict(
    signal_wide: pd.DataFrame,
    data_dict: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, pd.Series]:
    """Split wide signal DataFrame into per-symbol Series.

    Parameters
    ----------
    signal_wide
        Wide-layout signal: ``index=datetime, columns=symbol``.
    data_dict
        If supplied, each output Series is reindexed to that
        symbol's data index. Symbols present in ``data_dict`` but
        absent from ``signal_wide.columns`` receive an all-NaN
        Series at the data's own index. Symbols present in
        ``signal_wide`` but absent from ``data_dict`` are dropped
        with no warning (the caller is presumably routing only the
        relevant subset to the engine).

    Returns
    -------
    dict[str, pd.Series]
        Per-symbol signal Series, ready for
        ``engine.set_signals({...})``.

    Notes
    -----
    Inf and -Inf values in ``signal_wide`` are replaced with NaN
    before splitting, since the signal contract has no defined
    behavior for infinite values and engine code uses NaN as the
    "hold" sentinel.
    """
    cleaned = signal_wide.replace([np.inf, -np.inf], np.nan)
    if data_dict is None:
        return {sym: cleaned[sym].copy() for sym in cleaned.columns}
    out: dict[str, pd.Series] = {}
    for sym, df in data_dict.items():
        if sym in cleaned.columns:
            out[sym] = cleaned[sym].reindex(df.index)
        else:
            out[sym] = pd.Series(np.nan, index=df.index, dtype=float)
    return out


def validate_signal_series(
    signal: pd.Series,
    *,
    allow_fractional: bool = True,
) -> None:
    """Lightweight validation of a single-asset signal Series.

    Checks (cheap, no engine state):
      - ``signal.index`` is a ``pd.DatetimeIndex``.
      - No duplicate timestamps.
      - All values are numeric (or NaN).

    Does NOT check:
      - Whether the signal is "good" (returns, hit rate, etc.).
      - Whether the signal contains lookahead (use
        ``aiphaforge.diagnostics.assert_signal_no_lookahead`` for that).

    Parameters
    ----------
    signal
        Signal Series.
    allow_fractional
        If True, fractional values (e.g. 0.5 target weight) are
        allowed. If False, only the canonical ``{NaN, 0, 1, -1}``
        values pass.

    Raises
    ------
    TypeError
        If the index is not a DatetimeIndex.
    ValueError
        On duplicate timestamps or out-of-range values when
        ``allow_fractional=False``.
    """
    if not isinstance(signal.index, pd.DatetimeIndex):
        raise TypeError(
            f"signal index must be a DatetimeIndex, got "
            f"{type(signal.index).__name__}"
        )
    if signal.index.has_duplicates:
        dup = signal.index[signal.index.duplicated()].unique()
        raise ValueError(
            f"signal index has {len(dup)} duplicate timestamp(s): "
            f"{list(dup[:5])}{' ...' if len(dup) > 5 else ''}"
        )
    if not pd.api.types.is_numeric_dtype(signal):
        raise TypeError(
            f"signal values must be numeric, got dtype {signal.dtype}"
        )
    if not allow_fractional:
        valid = {0.0, 1.0, -1.0}
        non_canonical = signal.dropna()
        non_canonical = non_canonical[~non_canonical.isin(valid)]
        if len(non_canonical) > 0:
            raise ValueError(
                f"allow_fractional=False but signal contains "
                f"{len(non_canonical)} non-{{NaN, 0, ±1}} values; "
                f"sample: {list(non_canonical.head(5))}"
            )


def validate_signal_wide(signal_wide: pd.DataFrame) -> None:
    """Wide-form analog of :func:`validate_signal_series`.

    Checks the index + numeric dtype across all columns. Per-symbol
    semantic checks are performed by ``validate_signal_series``
    after a ``wide_to_signal_dict`` split.

    Raises
    ------
    TypeError
        If the index is not a DatetimeIndex or any column has
        non-numeric dtype.
    ValueError
        On duplicate timestamps.
    """
    if not isinstance(signal_wide.index, pd.DatetimeIndex):
        raise TypeError(
            f"signal_wide index must be a DatetimeIndex, got "
            f"{type(signal_wide.index).__name__}"
        )
    if signal_wide.index.has_duplicates:
        dup = signal_wide.index[signal_wide.index.duplicated()].unique()
        raise ValueError(
            f"signal_wide index has {len(dup)} duplicate timestamp(s)"
        )
    non_numeric_cols = [
        col for col in signal_wide.columns
        if not pd.api.types.is_numeric_dtype(signal_wide[col])
    ]
    if non_numeric_cols:
        raise TypeError(
            f"signal_wide columns must be numeric; non-numeric: "
            f"{non_numeric_cols}"
        )


# ---------------------------------------------------------------------------
# v2.4 Commit A: SignalSpec / SignalFrame typed wrappers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalSpec:
    """Typed metadata for the signal contract.

    Wraps the canonical NaN/0/±1 contract plus optional per-pipeline
    semantics (target-weight kind, transition-only behaviour, signal
    shift) so factor → rule → signal pipelines can carry intent
    through the stack without resorting to ad-hoc conventions.

    The defaults match v2.3's de facto contract — direction signals,
    NaN=hold, 0=flat, ±1=direction, transitions-only output, no
    additional shift (engine handles its own positioning lag).

    Pickle stability:
        SignalSpec is ``frozen=True``. Adding or renaming fields in
        a future minor version (v2.5+) WILL BREAK pickled instances.
        For cross-version persistence, serialize the contained
        values via JSON (using a manual converter) rather than
        pickle. The v2.x dataclass shape is NOT a stable
        persistence schema.
    """

    kind: Literal["direction", "target_weight", "score"] = "direction"
    hold_value: float = float("nan")
    flat_value: float = 0.0
    long_value: float = 1.0
    short_value: float = -1.0
    transition_only: bool = True
    signal_shift: int = 0   # default 0; engine handles its own lag


@dataclass(frozen=True)
class SignalFrame:
    """Typed wrapper around the raw signal payload.

    Carries the signal values + a SignalSpec describing the contract
    + free-form metadata (source attribution, generation timestamp,
    upstream model hash, etc.).

    Pickle stability:
        Same caveat as :class:`SignalSpec`. Persist via JSON for
        cross-version stability.
    """

    values: Union[pd.Series, Mapping[str, pd.Series], pd.DataFrame]
    spec: SignalSpec
    source: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Wrap metadata in MappingProxyType so the read-only Mapping
        # contract holds. Mirrors KnowledgeCheckReport's pattern in
        # probes/orchestrator.py — `frozen=True` blocks attribute
        # rebinding but the dict-default would otherwise still be
        # mutable.
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(
                self, "metadata",
                MappingProxyType(dict(self.metadata)),
            )

    def to_engine_input(
        self,
        data: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],
    ) -> Union[pd.Series, dict[str, pd.Series]]:
        """Downgrade to engine-compatible signal.

        Calls :func:`prepare_signals_for_engine` internally to
        resolve the wide-vs-dict-vs-series shape against the engine's
        data shape. Takes ``data`` at call time (NOT at SignalFrame
        construction) so a single SignalFrame can be reused across
        multiple engine runs / data shapes.

        The ``broadcast=False`` default is preserved — callers wanting
        to broadcast a single Series across a multi-asset universe
        should call :func:`prepare_signals_for_engine` directly with
        ``broadcast=True``.
        """
        return prepare_signals_for_engine(self.values, data)


def prepare_signals_for_engine(
    signals: pd.Series | dict[str, pd.Series] | pd.DataFrame,
    data: pd.DataFrame | dict[str, pd.DataFrame],
    *,
    broadcast: bool = False,
) -> pd.Series | dict[str, pd.Series]:
    """Resolve user-supplied signals to engine-compatible shape.

    `BacktestEngine.set_signals` accepts only ``pd.Series`` (single
    asset) or ``dict[str, pd.Series]`` (multi asset). Research and
    ML pipelines often produce wide ``pd.DataFrame`` signals or a
    single Series intended to apply across a multi-asset universe.
    This adapter resolves the user input to the engine-compatible
    shape, raising on ambiguous combinations.

    Routing:

    +---------------------------------------+----------------------------+
    | Input                                 | Output                     |
    +=======================================+============================+
    | single ``data`` + ``Series``          | ``Series`` (validated)     |
    +---------------------------------------+----------------------------+
    | single ``data`` + 1-col ``DataFrame`` | ``Series`` (the column)    |
    +---------------------------------------+----------------------------+
    | single ``data`` + multi-col ``DF``    | ``ValueError`` (ambiguous) |
    +---------------------------------------+----------------------------+
    | multi ``data_dict`` + ``dict``        | aligned ``dict``           |
    +---------------------------------------+----------------------------+
    | multi ``data_dict`` + wide ``DF``     | ``dict[str, Series]``      |
    +---------------------------------------+----------------------------+
    | multi ``data_dict`` + ``Series``,     |                            |
    |   ``broadcast=False`` (default)       | ``ValueError``             |
    +---------------------------------------+----------------------------+
    | multi ``data_dict`` + ``Series``,     | ``dict``: same Series for  |
    |   ``broadcast=True``                  | each symbol (per-symbol    |
    |                                       | reindex)                   |
    +---------------------------------------+----------------------------+

    Parameters
    ----------
    signals
        User-supplied signal in any of the accepted shapes.
    data
        ``pd.DataFrame`` (single asset) or
        ``Mapping[str, pd.DataFrame]`` (multi asset). Used only to
        determine the routing shape and the per-symbol target index.
    broadcast
        Opt-in escape hatch: when ``True`` and the input is a
        single Series with multi-asset ``data``, the same Series is
        replicated across every symbol in ``data`` (each reindexed
        to that symbol's data index). The canonical use case is
        a global market-wide signal (e.g. "go long all symbols
        when VIX < 15"). Default ``False`` makes the case raise so
        the user has to acknowledge the broadcast intent
        explicitly.

    Returns
    -------
    pd.Series | dict[str, pd.Series]
        Shape-matched, index-aligned, NaN/0/±1 semantics preserved.

    Raises
    ------
    TypeError
        If ``signals`` or ``data`` is a type the routing doesn't
        recognise.
    ValueError
        On the ambiguous routing cases listed in the table.
    """
    multi = isinstance(data, dict)
    # ---- single-asset branch ----
    if not multi:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"data must be pd.DataFrame (single) or "
                f"Mapping[str, pd.DataFrame] (multi); got "
                f"{type(data).__name__}"
            )
        if isinstance(signals, pd.Series):
            return signals.copy()
        if isinstance(signals, pd.DataFrame):
            if signals.shape[1] == 1:
                return signals.iloc[:, 0].copy()
            raise ValueError(
                f"single-asset data was supplied but signals is a "
                f"multi-column DataFrame ({signals.shape[1]} columns: "
                f"{list(signals.columns)[:5]}{'...' if signals.shape[1] > 5 else ''}). "
                f"Routing is ambiguous — slice the column you want "
                f"explicitly: signals['<col>']."
            )
        if isinstance(signals, dict):
            raise ValueError(
                "single-asset data was supplied but signals is a "
                "dict[str, Series]. Pick one symbol's series or pass "
                "the multi-asset data_dict."
            )
        raise TypeError(
            f"signals must be pd.Series, pd.DataFrame, or "
            f"dict[str, pd.Series]; got {type(signals).__name__}"
        )
    # ---- multi-asset branch ----
    if isinstance(signals, pd.DataFrame):
        return wide_to_signal_dict(signals, data)
    if isinstance(signals, dict):
        out: dict[str, pd.Series] = {}
        for sym, df in data.items():
            if sym in signals:
                out[sym] = signals[sym].reindex(df.index)
            else:
                out[sym] = pd.Series(np.nan, index=df.index, dtype=float)
        return out
    if isinstance(signals, pd.Series):
        if not broadcast:
            raise ValueError(
                "multi-asset data was supplied but signals is a "
                "single pd.Series. To broadcast the same signal "
                "across all symbols (e.g. a market-wide regime "
                "indicator), pass broadcast=True; otherwise supply a "
                "wide DataFrame or dict[str, Series]."
            )
        return {
            sym: signals.reindex(df.index).copy()
            for sym, df in data.items()
        }
    raise TypeError(
        f"signals must be pd.Series, pd.DataFrame, or "
        f"dict[str, pd.Series]; got {type(signals).__name__}"
    )


def target_weight_wide_to_schedule(
    weights: pd.DataFrame,
    data_dict: Mapping[str, pd.DataFrame],
    *,
    rebalance_dates: Iterable[pd.Timestamp] | None = None,
    snap: Literal["exact", "next", "previous"] = "exact",
    universe_alignment: Literal["union", "intersection"] = "union",
    on_collision: Literal["warn", "raise", "first", "last"] = "warn",
) -> dict:
    """Convert a wide target-weight DataFrame to engine schedule format.

    The output is suitable for ``engine.set_target_weights(schedule)``,
    which accepts both ``str`` and ``pd.Timestamp`` keys via the
    coercion at ``engine.py:725``. This adapter returns
    ``pd.Timestamp`` keys for type safety.

    Parameters
    ----------
    weights
        Wide-layout target-weight DataFrame:
        ``index=datetime, columns=symbol``.
    data_dict
        Per-symbol OHLCV mapping. Used to determine the
        valid-trading-day set per symbol (see ``universe_alignment``).
    rebalance_dates
        Specific dates to fire rebalances on. If None, every row in
        ``weights`` is treated as a candidate rebalance date.
    snap
        - ``"exact"``: candidate rebalance dates that are not present
          in the per-symbol valid-date set are silently dropped (with
          a count warning).
        - ``"next"``: snap to the next valid trading day.
        - ``"previous"``: snap to the previous valid trading day.
    universe_alignment
        - ``"union"`` (DEFAULT): per-symbol valid-date sets are
          independent. A staggered listing date does NOT invalidate
          rebalances for already-listed symbols. Missing-symbol-at-
          date receives weight 0.
        - ``"intersection"``: a rebalance date is valid only if ALL
          symbols have that date in their data index. Use when the
          strategy requires simultaneous rebalances of every name.
    on_collision
        Policy when ``snap="next"`` / ``"previous"`` pushes multiple
        requested rebalance dates onto the same target date:

        - ``"warn"`` (DEFAULT): last-write-wins, emit ``UserWarning``.
        - ``"raise"``: ``ValueError``.
        - ``"first"``: keep the first-requested write.
        - ``"last"``: silently last-write-wins (no warning).

    Returns
    -------
    dict[pd.Timestamp, dict[str, float]]
        Engine-compatible target-weight schedule.

    Notes
    -----
    Engine compatibility:
        ``BacktestEngine.set_target_weights(schedule)`` (defined at
        ``engine.py:363``) accepts both ``dict[str, ...]`` (date-string
        keys) and ``dict[pd.Timestamp, ...]``. Coercion via
        ``pd.Timestamp(date_str)`` happens at ``engine.py:725`` for
        every schedule key. v2.3 returns Timestamp keys; the engine
        side is unchanged.
    """
    import warnings

    if not isinstance(weights, pd.DataFrame):
        raise TypeError(
            f"weights must be pd.DataFrame, got {type(weights).__name__}"
        )
    if rebalance_dates is None:
        candidates = list(weights.index)
    else:
        candidates = [pd.Timestamp(d) for d in rebalance_dates]

    # Per-symbol valid-date sets.
    per_symbol_valid: dict[str, pd.DatetimeIndex] = {
        sym: pd.DatetimeIndex(df.index) for sym, df in data_dict.items()
    }
    if universe_alignment == "intersection":
        if per_symbol_valid:
            common = per_symbol_valid[next(iter(per_symbol_valid))]
            for v in per_symbol_valid.values():
                common = common.intersection(v)
            universe_valid = common
        else:
            universe_valid = pd.DatetimeIndex([])
    elif universe_alignment == "union":
        if per_symbol_valid:
            universe_valid = per_symbol_valid[next(iter(per_symbol_valid))]
            for v in per_symbol_valid.values():
                universe_valid = universe_valid.union(v)
        else:
            universe_valid = pd.DatetimeIndex([])
    else:
        raise ValueError(
            f"universe_alignment must be 'union' or 'intersection', "
            f"got {universe_alignment!r}"
        )

    def _snap_to(ts: pd.Timestamp) -> pd.Timestamp | None:
        if ts in universe_valid:
            return ts
        if snap == "exact":
            return None
        if snap == "next":
            after = universe_valid[universe_valid > ts]
            return after[0] if len(after) > 0 else None
        if snap == "previous":
            before = universe_valid[universe_valid < ts]
            return before[-1] if len(before) > 0 else None
        raise ValueError(
            f"snap must be 'exact', 'next', or 'previous', got {snap!r}"
        )

    snapped: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    n_dropped = 0
    for ts in candidates:
        target = _snap_to(ts)
        if target is None:
            n_dropped += 1
            continue
        snapped.append((ts, target))

    if snap == "exact" and n_dropped > 0:
        warnings.warn(
            f"target_weight_wide_to_schedule dropped {n_dropped} "
            f"rebalance date(s) not present in the universe valid "
            f"set (snap='exact'). Use snap='next' or 'previous' "
            f"to absorb non-aligned dates.",
            UserWarning,
            stacklevel=2,
        )

    # Detect collisions: target → list[requested]
    by_target: dict[pd.Timestamp, list[pd.Timestamp]] = {}
    for requested, target in snapped:
        by_target.setdefault(target, []).append(requested)
    collisions = {
        t: reqs for t, reqs in by_target.items() if len(reqs) > 1
    }
    if collisions:
        if on_collision == "raise":
            sample = next(iter(collisions.items()))
            raise ValueError(
                f"snap='{snap}' produced {len(collisions)} collision(s) "
                f"where multiple requested rebalance dates landed on the "
                f"same target. Sample: target={sample[0]}, requested="
                f"{sample[1][:3]}{'...' if len(sample[1]) > 3 else ''}. "
                f"Use on_collision='first' / 'last' / 'warn' to absorb."
            )
        if on_collision == "warn":
            warnings.warn(
                f"target_weight_wide_to_schedule snap='{snap}' produced "
                f"{len(collisions)} collision(s); using last-write-wins. "
                f"Pass on_collision='raise'/'first'/'last' to choose "
                f"a different policy.",
                UserWarning,
                stacklevel=2,
            )

    out: dict[pd.Timestamp, dict[str, float]] = {}
    for requested, target in snapped:
        if target in out:
            if on_collision == "first":
                continue  # keep the existing (earlier) write
            # "last" / "warn" → overwrite (last-write-wins)
        if requested in weights.index:
            row = weights.loc[requested]
        else:
            # Snapping forward/backward but the *requested* row isn't
            # in weights (only happens if rebalance_dates supplied
            # values not in the weights frame). Fall back to all-zero.
            row = pd.Series(0.0, index=weights.columns)
        symbol_weights: dict[str, float] = {}
        for sym in data_dict:
            if sym in weights.columns:
                val = row[sym]
                if pd.isna(val):
                    symbol_weights[sym] = 0.0
                else:
                    symbol_weights[sym] = float(val)
            else:
                symbol_weights[sym] = 0.0
        out[target] = symbol_weights
    return out


__all__ = [
    "transitions_only",
    "dict_to_signal_wide",
    "wide_to_signal_dict",
    "validate_signal_series",
    "validate_signal_wide",
    "prepare_signals_for_engine",
    "target_weight_wide_to_schedule",
    # v2.4 Commit A
    "SignalSpec",
    "SignalFrame",
]
