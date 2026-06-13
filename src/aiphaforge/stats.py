"""Shared numerical primitives.

Neutral home for low-level statistical building blocks that are used by
more than one pillar of the library (``significance``, ``probes``,
``alpha``, ``calendars``). Hoisting them here removes the cross-pillar
``_``-prefixed reach-ins that previously forced the ``alpha.rank_stats``
shim and a ``calendars`` -> ``probes`` reverse dependency.

This module depends only on numpy/pandas and the standard library; it
imports nothing from the rest of ``aiphaforge`` and is therefore safe to
import from anywhere without risking a cycle.

Contents:
    - ``IntegrityCheckResult`` — OHLC integrity check outcome.
    - ``midranks`` / ``tie_corrected_spearman`` — rank-correlation primitives.
    - ``_stationary_block_bootstrap`` / ``_block_bootstrap_indices`` /
      ``_reconstruct_ohlcv`` — stationary block-bootstrap primitives
      (Politis-Romano), private but importable for internal reuse.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "IntegrityCheckResult",
    "midranks",
    "tie_corrected_spearman",
]


# ---------------------------------------------------------------------------
# OHLC integrity
# ---------------------------------------------------------------------------


@dataclass
class IntegrityCheckResult:
    """Outcome of running OHLC integrity checks on a transformed frame."""

    passed: bool
    errors: list[str]


# ---------------------------------------------------------------------------
# Rank-correlation primitives
# ---------------------------------------------------------------------------


def tie_corrected_spearman(
    x_ranks: np.ndarray,
    y_ranks: np.ndarray,
) -> float:
    """Compute the tie-corrected Spearman rho.

    Uses the standard Kendall (1948) / Olds (1949) form, equivalent
    to scipy's spearmanr under the hood. Matches scipy to ~1e-12.

    Formula:
      rho = ((N^3 - N) - 6*sum(d^2) - (T_x + T_y)/2)
            / sqrt( ((N^3 - N) - T_x) * ((N^3 - N) - T_y) )

    where T = sum_g (t_g^3 - t_g) over tie groups in each ranking.
    """
    n = len(x_ranks)
    if n != len(y_ranks):
        raise ValueError("rank arrays must have equal length")
    if n < 2:
        return 0.0

    def _tie_term(ranks: np.ndarray) -> float:
        # Count tie-group sizes by integer bucketing of mid-ranks.
        # Mid-ranks of tied pairs share the same value.
        unique, counts = np.unique(ranks, return_counts=True)
        return float(
            np.sum((counts.astype(float) ** 3) - counts.astype(float))
        )

    d = x_ranks.astype(float) - y_ranks.astype(float)
    sum_d2 = float(np.sum(d ** 2))
    n3_n = float(n ** 3 - n)
    t_x = _tie_term(x_ranks)
    t_y = _tie_term(y_ranks)
    numerator = n3_n - 6.0 * sum_d2 - 0.5 * (t_x + t_y)
    denom = math.sqrt(max((n3_n - t_x), 0.0) * max((n3_n - t_y), 0.0))
    if denom == 0:
        return 0.0
    return numerator / denom


def midranks(values: Sequence[float]) -> np.ndarray:
    """Return mid-ranks of values (1-indexed, ties get average rank)."""
    arr = np.asarray(values, dtype=float)
    return pd.Series(arr).rank(method="average").to_numpy()


# ---------------------------------------------------------------------------
# Stationary block-bootstrap primitives (Politis-Romano)
# ---------------------------------------------------------------------------


def _stationary_block_bootstrap(
    returns: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One stationary block bootstrap replication (Politis-Romano).

    Block lengths are drawn from a Geometric(1/block_size) distribution.
    Start positions are sampled uniformly. Indices wrap circularly (mod n).

    Parameters:
        returns: Original return series as a numpy array.
        block_size: Expected block length (mean of geometric distribution).
        rng: NumPy random generator instance.

    Returns:
        Resampled return series of the same length as input.
    """
    n = len(returns)
    result = []
    while len(result) < n:
        start = rng.integers(0, n)
        length = rng.geometric(1.0 / block_size)
        for j in range(length):
            if len(result) >= n:
                break
            result.append(returns[(start + j) % n])
    return np.array(result)


def _block_bootstrap_indices(
    n_bars: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one set of block-bootstrapped bar indices.

    Wraps _stationary_block_bootstrap applied to np.arange(n_bars).
    The bootstrap function resamples values in blocks with circular
    wrapping — when applied to sequential integers, it produces a
    block-structured sequence of valid indices.

    Returns integer ndarray of length n_bars.
    """
    # _stationary_block_bootstrap operates on values via indexing:
    # returns[(start + j) % n]. For input [0, 1, 2, ...], the output
    # is those integer values resampled in blocks. No arithmetic is
    # performed on the values, so converting back to int is exact.
    raw = _stationary_block_bootstrap(
        np.arange(n_bars, dtype=float), block_size, rng)
    return raw.astype(int)


def _reconstruct_ohlcv(
    data: pd.DataFrame,
    indices: np.ndarray,
) -> pd.DataFrame:
    """Reconstruct OHLCV from resampled bar indices.

    Algorithm:
        1. Look up the original bars at the bootstrapped indices.
        2. Compute close-to-close returns from the resampled sequence:
           return[i] = original_close[indices[i]] / original_close[indices[i-1]]
           (For i=0, use 1.0 — anchor bar has no return.)
        3. Anchor at data.iloc[0]['close'], apply returns cumulatively
           to build a new close price series.
        4. For each bar, scale open/high/low proportionally:
           ratio = new_close[i] / original_close[indices[i]]
           new_open[i] = original_open[indices[i]] * ratio
           new_high[i] = original_high[indices[i]] * ratio
           new_low[i]  = original_low[indices[i]]  * ratio
           This preserves each bar's OHLC relationships exactly.
        5. Volume: copy directly from the resampled bars.
        6. Attach the ORIGINAL DatetimeIndex (same dates, new prices).
        7. Price guard: if cumulative close drops below epsilon (1e-8),
           clamp to epsilon. This prevents zero/negative prices from
           extreme return sequences.
    """
    orig_close = data["close"].values
    orig_open = data["open"].values
    orig_high = data["high"].values
    orig_low = data["low"].values
    orig_volume = data["volume"].values

    n = len(indices)

    # Step 2: each bar's actual historical return from the original series
    # (avoids spurious returns at block boundaries)
    returns = np.ones(n)
    mask = indices > 0
    returns[mask] = orig_close[indices[mask]] / orig_close[indices[mask] - 1]

    # Step 3: anchor at original first close, cumulative product
    anchor = data.iloc[0]["close"]
    new_close = anchor * np.cumprod(returns)

    # Step 7: epsilon guard — clamp to prevent zero/negative prices
    new_close = np.maximum(new_close, 1e-8)

    # Step 4: scale OHLC proportionally
    ratio = new_close / orig_close[indices]
    new_open = orig_open[indices] * ratio
    new_high = orig_high[indices] * ratio
    new_low = orig_low[indices] * ratio

    # Step 5: volume from resampled bars
    new_volume = orig_volume[indices]

    # Step 6: original DatetimeIndex
    return pd.DataFrame(
        {
            "open": new_open,
            "high": new_high,
            "low": new_low,
            "close": new_close,
            "volume": new_volume,
        },
        index=data.index,
    )
