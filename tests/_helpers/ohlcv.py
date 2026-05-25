"""Shared OHLCV constructors for the AIphaForge test suite.

v2.8.5 L1: consolidates the previously-duplicated ``_ohlcv`` /
``_synthetic_ohlcv`` / ``_make_data`` fixtures (32 defs across 31 test
files) into 3 reusable constructors. Each constructor matches one of
the three shape families observed during the v2.8.5 audit:

* :func:`make_ohlcv`           - GBM-style OHLCV (largest cluster).
* :func:`make_close_only`      - constant-drift / deterministic OHLCV.
* :func:`make_ohlcv_from_closes` - explicit-list OHLCV for strict
  validation tests.

Method-bound fixtures (``self._ohlcv`` / ``self._make_data``) and
bespoke shapes (e.g. ``_ohlcv_clean`` in calendar-snap tests) are
intentionally preserved in-place; v2.9 may revisit those if a
class-based fixture factory turns out to be worth the churn.

These constructors are test-only and SHOULD NOT be imported from any
``src/aiphaforge/`` module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_ohlcv(
    periods: int = 60,
    seed: int = 0,
    base: float = 100.0,
    vol: float = 0.01,
) -> pd.DataFrame:
    """Build a GBM-style OHLCV frame.

    The close path is ``base * exp(cumsum(N(0, vol, periods)))``.
    High and low scale by ``+/- 1%`` of close; volume is a flat
    ``1e6``-per-bar series; the index is a business-day range
    starting ``2024-01-01``.

    Parameters
    ----------
    periods:
        Number of bars to emit.
    seed:
        Seed for ``numpy.random.default_rng``.
    base:
        Initial price level.
    vol:
        Per-bar log-return standard deviation.

    Returns
    -------
    pd.DataFrame
        Columns ``[open, high, low, close, volume]`` indexed by
        business days.
    """
    rng = np.random.default_rng(seed)
    closes = base * np.exp(np.cumsum(rng.normal(0.0, vol, periods)))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": [1e6] * periods,
        },
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


def make_close_only(periods: int, base: float = 100.0) -> pd.DataFrame:
    """Build a constant-drift OHLCV frame with no randomness.

    Close follows ``base * (1 + 0.001 * arange(periods))``; OHLC
    columns all equal close (no intra-bar range); volume is a flat
    ``1e6``-per-bar series; the index is a business-day range
    starting ``2024-01-01``.

    Useful for closed-form regression checks where stochastic noise
    would obscure the expected total return.
    """
    closes = base * (1.0 + 0.001 * np.arange(periods))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1e6] * periods,
        },
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


def make_ohlcv_from_closes(
    closes: list[float] | np.ndarray,
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    """Build an OHLCV frame from an explicit close-price sequence.

    Open / high / low all equal the close column. Volumes default to
    a flat ``1_000_000`` per bar; pass ``volumes`` to override. The
    index is a business-day range starting ``2024-01-01``. The shape
    is intended for strict-validation tests that need to inject
    specific non-finite or non-positive prices at chosen bars.
    """
    n = len(closes)
    if volumes is None:
        volumes = [1_000_000.0] * n
    arr = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"open": arr, "high": arr, "low": arr, "close": arr, "volume": volumes},
        index=pd.bdate_range("2024-01-01", periods=n),
    )
