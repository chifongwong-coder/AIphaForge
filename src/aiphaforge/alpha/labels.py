"""v2.4 Commit F — Forward-return labels for factor research.

Pairs a per-bar factor value at time t with the realized return
of entering at t+entry_lag and exiting at t+entry_lag+horizon.
Default entry_lag=1 mirrors the engine's T+1 fill convention
(decision at t, fill at t+1).

Two return types ship:
    "simple" (default) — gross simple return ``exit / entry - 1``.
                         Rank-invariant; use for IC / RankIC.
    "log"              — log return ``log(exit / entry)``. Compounds
                         additively across non-overlapping windows;
                         closer to Gaussian for distributional stats
                         (Sharpe, t-tests).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def forward_returns(
    prices: pd.DataFrame,
    *,
    horizon: int,
    entry_lag: int = 1,
    return_type: Literal["simple", "log"] = "simple",
) -> pd.DataFrame:
    """Compute t -> t+entry_lag -> t+entry_lag+horizon forward returns.

    At index t, the value is the return of entering the position at
    bar ``t + entry_lag`` and exiting at bar ``t + entry_lag + horizon``.
    Bars near the end of the series produce NaN (insufficient lookforward).

    Parameters
    ----------
    prices
        Wide ``pd.DataFrame[index=datetime, columns=symbol]`` of
        prices (typically close).
    horizon
        Number of bars between entry and exit (positive integer).
    entry_lag
        Bars between the factor's decision time t and the entry
        bar. Default 1 (decision at t, fill at t+1) matches the
        engine's T+1 convention.
    return_type
        ``"simple"`` (default) returns ``exit / entry - 1`` (gross
        simple return). ``"log"`` returns ``log(exit / entry)`` —
        additive across non-overlapping windows and closer to
        Gaussian for distributional statistics. Choose ``"simple"``
        for rank-invariant statistics (IC / RankIC); choose
        ``"log"`` for multi-horizon aggregation or distributional
        analysis.

    Returns
    -------
    pd.DataFrame
        Same shape as ``prices``. NaN for bars where lookforward is
        insufficient.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError(
            f"prices must be pd.DataFrame, got {type(prices).__name__}"
        )
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if entry_lag < 0:
        raise ValueError(f"entry_lag must be >= 0, got {entry_lag}")

    entry = prices.shift(-entry_lag)
    exit_ = prices.shift(-(entry_lag + horizon))

    if return_type == "simple":
        return exit_ / entry - 1.0
    if return_type == "log":
        return np.log(exit_ / entry)
    raise ValueError(
        f"return_type must be 'simple' or 'log', got {return_type!r}"
    )


__all__ = ["forward_returns"]
