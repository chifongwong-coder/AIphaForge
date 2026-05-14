"""v2.4 Commit H — FactorReport typed dataclass + AlphaScreenConfig."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class AlphaScreenConfig:
    """Config for :class:`AlphaScreener.evaluate`.

    Attributes
    ----------
    horizons
        Forward-return horizons to evaluate.
    entry_lag
        Decision-to-fill lag in bars (default 1 mirrors engine T+1).
    quantiles
        Number of quantile buckets for the quantile-returns table.
    """

    horizons: tuple[int, ...] = (1, 5, 10)
    entry_lag: int = 1
    quantiles: int = 5


@dataclass(frozen=True)
class FactorReport:
    """Per-factor research report.

    Attributes
    ----------
    factor_name
        Identifier (matches the factor's column name in the input
        DataFrame, or the FactorSet key).
    metrics
        Aggregate scalars: ``ic_mean``, ``ic_ir``, ``rank_ic_mean``,
        ``rank_ic_ir``, ``coverage_mean``.
    rank_ic
        Per-horizon per-bar Spearman RankIC series.
    ic
        Per-horizon per-bar Pearson IC series.
    quantile_returns
        Per-horizon DataFrame of per-quantile-id per-period mean
        forward returns. Columns are ``range(config.quantiles)``;
        rows are timestamps. NaN columns where the per-timestamp
        quantile bucket collapsed (e.g. heavy-tie scores).
    """

    factor_name: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    rank_ic: Mapping[int, pd.Series] = field(default_factory=dict)
    ic: Mapping[int, pd.Series] = field(default_factory=dict)
    quantile_returns: Mapping[int, pd.DataFrame] = field(default_factory=dict)


__all__ = ["AlphaScreenConfig", "FactorReport"]
