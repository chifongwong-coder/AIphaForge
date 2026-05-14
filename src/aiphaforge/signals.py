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

from typing import Mapping

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


__all__ = [
    "transitions_only",
    "dict_to_signal_wide",
    "wide_to_signal_dict",
    "validate_signal_series",
    "validate_signal_wide",
]
