"""v2.4 alpha — Factor research subpackage.

Architectural firewall (master plan v1.0 §1.2): NO module under
``aiphaforge.alpha.*`` may import ``aiphaforge.engine`` (or any
execution-layer module — fees, broker, market_impact, etc.).
This is a research-only namespace; backtesting lives in the
engine layer.

v2.4 ships:
    rank_stats.py       — re-exports probes._rank primitives
                          (tie_corrected_spearman, midranks)
    labels.py           — forward_returns
    metrics.py          — ic, rank_ic, coverage
    evaluator.py        — AlphaScreener
    report.py           — FactorReport typed dataclass
    signal_analysis.py  — signal-level metrics for signal-only
                          strategies that don't expose factors
"""
from __future__ import annotations
