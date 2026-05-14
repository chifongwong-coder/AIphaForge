"""v2.4 Commit K — Signal-level metrics for signal-only strategies.

Master plan §0.4: signal-only strategies (ML direct signals, AI
agent decisions) are first-class — they don't expose factors and
deserve research-layer metrics anyway. This module ships three
metrics defined directly on the {NaN, 0, ±1} signal contract.
"""
from __future__ import annotations

import pandas as pd

from aiphaforge.alpha.labels import forward_returns


def _series_to_position(signal: pd.Series) -> pd.Series:
    """Per signal contract: NaN=hold, 0=flat, ±1=direction.

    Position derivation: ``signal.ffill().fillna(0.0)``. Initial
    NaN window before the first explicit signal becomes 0 (no
    position). Subsequent NaN bars carry forward the prior
    instruction.
    """
    return signal.ffill().fillna(0.0)


def signal_forward_return(
    signal: pd.Series,
    prices: pd.Series,
    *,
    horizon: int = 1,
    entry_lag: int = 1,
) -> pd.Series:
    """Per-bar realised return of taking the signal's position.

    position[t] * forward_returns(prices, horizon, entry_lag)[t].
    Uses simple gross returns from :func:`alpha.labels.forward_returns`.
    """
    if not isinstance(signal, pd.Series):
        raise TypeError(
            f"signal must be pd.Series, got {type(signal).__name__}"
        )
    if not isinstance(prices, pd.Series):
        raise TypeError(
            f"prices must be pd.Series, got {type(prices).__name__}"
        )
    position = _series_to_position(signal)
    fr = forward_returns(
        prices.to_frame(name="_p"), horizon=horizon, entry_lag=entry_lag,
    )["_p"]
    # Reindex position to fr.index (same as prices.index by construction).
    return position.reindex(fr.index) * fr


def signal_hit_rate(
    signal: pd.Series,
    prices: pd.Series,
    *,
    horizon: int = 1,
    entry_lag: int = 1,
) -> float:
    """Fraction of NON-FLAT bars where position * forward_return > 0.

    Numerator: count of bars with ``position[t] != 0`` AND
    ``position[t] * forward_return[t] > 0``.
    Denominator: count of bars with ``position[t] != 0`` (the
    non-flat trading universe).

    Strict ``> 0`` excludes ties at exactly 0 from the numerator.
    Returns NaN if there are no non-flat bars.
    """
    position = _series_to_position(signal)
    fr = forward_returns(
        prices.to_frame(name="_p"), horizon=horizon, entry_lag=entry_lag,
    )["_p"]
    pnl = position.reindex(fr.index) * fr
    non_flat = position.reindex(fr.index) != 0
    valid = non_flat & ~pnl.isna()
    if int(valid.sum()) == 0:
        return float("nan")
    wins = (valid & (pnl > 0)).sum()
    return float(wins) / float(valid.sum())


def signal_turnover(signal: pd.Series) -> float:
    """Average per-bar position-change magnitude.

    Defined as ``position.diff().abs().mean()``. A long↔short
    flip contributes 2 to the diff (|-1 - 1| = 2); a long↔flat
    transition contributes 1.

    The first bar has NaN diff and is excluded from the mean
    (pandas `.mean()` skips NaN by default).
    """
    position = _series_to_position(signal)
    return float(position.diff().abs().mean())


__all__ = [
    "signal_forward_return",
    "signal_hit_rate",
    "signal_turnover",
]
