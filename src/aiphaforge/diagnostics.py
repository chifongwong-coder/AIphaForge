"""v2.3 Commit I — No-lookahead diagnostics.

Test / research tools for verifying that a factor's or strategy's
output at time t is not contaminated by data from t' > t.

Both functions use **prefix-slice semantics**:

    For each check point t:
        full       = provider(data)
        truncated  = provider(data.loc[:t])
        assert     full.loc[:t] ≈ truncated.loc[:t] within atol

Why prefix-slice instead of NaN-tail (master plan §6.2):
    NaN-tail (overwriting data[t+1:] with NaN and recomputing) is
    BIT-IDENTICAL to prefix-slice on simple windowed rolling/ewm
    indicators (RSI(14), SMA, ewm(span=20), rolling.std). For
    those, both approaches catch lookahead bugs equally well. But
    prefix-slice is the strictly stronger formal definition of
    "no future data", catching three real failure modes that
    NaN-tail silently misses:

    1. **Full-sample normalizations** (e.g. z-score over the whole
       series): NaN-tail computes mean/std ignoring NaN, so the per-t
       output is identical to the no-truncation case — a genuine
       lookahead bug goes undetected. Prefix-slice computes over
       [:t] only and catches the divergence.
    2. **Length-aware factors** (anything reading len(series) or
       indexing series.iloc[-1] as a boundary anchor): NaN-tail
       preserves the full length, prefix-slice exposes the
       truncation.
    3. **min_periods boundary**: rolling with min_periods=N returns
       NaN for the first N-1 bars; under NaN-tail this boundary
       silently moves because the NaN tail looks like missing
       observations.

Usage notes:
    These are TEST / RESEARCH diagnostics, NOT runtime checks. Cost
    is O(check_points × prefix compute) — each call passes
    data.loc[:t] (a prefix slice), not the full series. For 3 default
    check points, average cost is ~0.5× full compute.

    Engine-agnostic by design — neither function imports
    aiphaforge.engine.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence, Union

import numpy as np
import pandas as pd

DEFAULT_ATOL = 1e-12

DataLike = Union[pd.DataFrame, Mapping[str, pd.DataFrame]]


def _slice_data(data: DataLike, t: pd.Timestamp) -> DataLike:
    """Prefix-slice each DataFrame in the input at timestamp t."""
    if isinstance(data, pd.DataFrame):
        return data.loc[:t]
    return {sym: df.loc[:t] for sym, df in data.items()}


def _default_check_points(data: DataLike) -> list[pd.Timestamp]:
    """Default check points: 25%, 50%, 75% of the data index."""
    if isinstance(data, pd.DataFrame):
        idx = data.index
    else:
        # Use the first symbol's index as the reference.
        first_sym = next(iter(data))
        idx = data[first_sym].index
    n = len(idx)
    if n < 4:
        # Too short to pick non-trivial check points; pick the middle.
        return [idx[n // 2]] if n > 0 else []
    return [idx[n // 4], idx[n // 2], idx[3 * n // 4]]


def _compare_prefix(
    full: pd.DataFrame | pd.Series,
    truncated: pd.DataFrame | pd.Series,
    t: pd.Timestamp,
    atol: float,
) -> str | None:
    """Return None if full.loc[:t] ≈ truncated.loc[:t], else an error message."""
    if type(full) is not type(truncated):
        return (
            f"output type mismatch: full={type(full).__name__}, "
            f"truncated={type(truncated).__name__}"
        )
    full_prefix = full.loc[:t]
    trunc_prefix = truncated.loc[:t]
    if isinstance(full_prefix, pd.DataFrame):
        if not full_prefix.columns.equals(trunc_prefix.columns):
            return (
                f"column mismatch at t={t!r}: full="
                f"{list(full_prefix.columns)}, truncated="
                f"{list(trunc_prefix.columns)}"
            )
    if not full_prefix.index.equals(trunc_prefix.index):
        return f"index mismatch at t={t!r}"
    # v2.4 Commit C: dtype precheck. The diagnostic is numeric-only;
    # categorical / string-valued factors raise a confusing
    # "could not convert string to float" deep inside numpy. Catch
    # them here with a typed message naming the offending column.
    if isinstance(full_prefix, pd.DataFrame):
        non_numeric = [
            c for c in full_prefix.columns
            if not pd.api.types.is_numeric_dtype(full_prefix[c])
        ]
        if non_numeric:
            return (
                f"non-numeric column dtype detected at t={t!r}: "
                f"{[(c, str(full_prefix[c].dtype)) for c in non_numeric]}. "
                f"assert_*_no_lookahead is numeric-only — convert "
                f"categorical / string factors to integer codes "
                f"before checking, or write a custom check."
            )
    elif isinstance(full_prefix, pd.Series):
        if not pd.api.types.is_numeric_dtype(full_prefix):
            return (
                f"non-numeric series dtype detected at t={t!r}: "
                f"{full_prefix.dtype}. assert_*_no_lookahead is "
                f"numeric-only."
            )
    arr_full = full_prefix.to_numpy(dtype=float)
    arr_trunc = trunc_prefix.to_numpy(dtype=float)
    # Treat NaN-vs-finite asymmetry as divergence (catches shift(-N)
    # leakage where the truncated edge becomes NaN while the full
    # version still carries a future value). np.subtract gives NaN
    # for both-NaN and mixed-NaN; we differentiate the two cases.
    nan_full = np.isnan(arr_full)
    nan_trunc = np.isnan(arr_trunc)
    nan_mismatch = nan_full ^ nan_trunc  # XOR: exactly one is NaN
    if np.any(nan_mismatch):
        n = int(nan_mismatch.sum())
        return (
            f"prefix NaN-state mismatch at t={t!r} on {n} cell(s) — "
            f"the full-data factor produced finite values where the "
            f"prefix-data factor produced NaN (or vice versa). This "
            f"is the signature of shift(-N) future-leak or similar."
        )
    # Both finite (or both NaN): compare numerically.
    diff = np.where(
        nan_full & nan_trunc, 0.0, np.abs(arr_full - arr_trunc),
    )
    if np.any(diff > atol):
        max_diff = float(np.nanmax(diff))
        return (
            f"prefix divergence at t={t!r}: max abs diff "
            f"{max_diff:.3e} > atol {atol:.3e}"
        )
    return None


def _coerce_to_frame(
    out: object,
) -> pd.DataFrame:
    """Normalise a provider's return value to a DataFrame for comparison.

    Single-asset providers may return Series; this lifts them to a
    one-column DataFrame for uniform .loc / .columns handling.
    """
    if isinstance(out, pd.Series):
        return out.to_frame()
    if isinstance(out, pd.DataFrame):
        return out
    raise TypeError(
        f"expected pd.Series or pd.DataFrame from provider, got "
        f"{type(out).__name__}"
    )


def assert_factor_no_lookahead(
    factor_provider: Callable[[DataLike], object],
    data: DataLike,
    *,
    check_points: Sequence[pd.Timestamp] | None = None,
    atol: float = DEFAULT_ATOL,
) -> None:
    """Verify a factor function does not use future data.

    Parameters
    ----------
    factor_provider
        Callable taking ``data`` and returning a per-bar factor
        value (Series or DataFrame). For multi-asset, the input
        is a ``dict[str, DataFrame]`` and the output is typically
        a wide DataFrame.
    data
        Input data, single ``DataFrame`` or per-symbol mapping.
    check_points
        Timestamps to test. Defaults to 25%, 50%, 75% of the data
        index.
    atol
        Absolute tolerance for the per-bar comparison. Default
        ``1e-12`` — strict enough to catch any meaningful drift,
        loose enough to absorb float-op-order noise from
        ``groupby().transform`` and similar.

    Raises
    ------
    AssertionError
        If any check point's prefix output diverges by more than
        ``atol``.
    """
    if check_points is None:
        check_points = _default_check_points(data)
    full = _coerce_to_frame(factor_provider(data))
    for t in check_points:
        trunc = _coerce_to_frame(factor_provider(_slice_data(data, t)))
        msg = _compare_prefix(full, trunc, pd.Timestamp(t), atol)
        if msg is not None:
            raise AssertionError(
                f"factor lookahead detected: {msg}. The factor "
                f"depends on data after t — probable cause: full-sample "
                f"normalisation, length-aware indexing, or shift(-N)."
            )


def assert_signal_no_lookahead(
    signal_provider,
    data: DataLike,
    *,
    check_points: Sequence[pd.Timestamp] | None = None,
    atol: float = DEFAULT_ATOL,
) -> None:
    """Verify a signal-producing object does not use future data.

    ``signal_provider`` may be:
      - any object with ``generate_signals(data)`` (e.g. BaseStrategy),
      - or a plain callable taking ``data``.

    See :func:`assert_factor_no_lookahead` for parameter semantics.
    """
    if check_points is None:
        check_points = _default_check_points(data)

    def _call(d: DataLike) -> object:
        if hasattr(signal_provider, "generate_signals"):
            return signal_provider.generate_signals(d)
        return signal_provider(d)

    full = _coerce_to_frame(_call(data))
    for t in check_points:
        trunc = _coerce_to_frame(_call(_slice_data(data, t)))
        msg = _compare_prefix(full, trunc, pd.Timestamp(t), atol)
        if msg is not None:
            raise AssertionError(
                f"signal lookahead detected: {msg}. The signal "
                f"provider depends on data after t — probable cause: "
                f"future labels, full-sample stats, or shift(-N)."
            )


__all__ = [
    "assert_factor_no_lookahead",
    "assert_signal_no_lookahead",
]
