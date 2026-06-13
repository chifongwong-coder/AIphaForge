"""v2.4 alpha — Factor research subpackage.

Architectural firewall (master plan v1.0 §1.2): NO module under
``aiphaforge.alpha.*`` may import ``aiphaforge.engine`` (or any
execution-layer module — fees, broker, market_impact, etc.).
This is a research-only namespace; backtesting lives in the
engine layer.

Rank-correlation primitives (``tie_corrected_spearman``, ``midranks``)
are re-exported from ``aiphaforge.stats``; the v2.4 ``alpha.rank_stats``
shim was removed in v2.9.0 once those primitives moved to the neutral
stats module.

v2.4 ships:
    labels.py           — forward_returns
    metrics.py          — ic, rank_ic, coverage
    evaluator.py        — AlphaScreener
    report.py           — FactorReport typed dataclass
    signal_analysis.py  — signal-level metrics for signal-only
                          strategies that don't expose factors
"""
from __future__ import annotations

from aiphaforge.alpha.evaluator import AlphaScreener
from aiphaforge.alpha.labels import forward_returns
from aiphaforge.alpha.metrics import coverage, ic, rank_ic
from aiphaforge.alpha.report import AlphaScreenConfig, FactorReport
from aiphaforge.alpha.signal_analysis import (
    signal_forward_return,
    signal_hit_rate,
    signal_turnover,
)
from aiphaforge.stats import midranks, tie_corrected_spearman

__all__ = [
    "AlphaScreener",
    "AlphaScreenConfig",
    "FactorReport",
    "coverage",
    "forward_returns",
    "ic",
    "rank_ic",
    "midranks",
    "tie_corrected_spearman",
    "signal_forward_return",
    "signal_hit_rate",
    "signal_turnover",
]
