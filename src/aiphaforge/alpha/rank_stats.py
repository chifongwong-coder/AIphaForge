"""v2.4 Commit E — Rank-statistics primitives for the alpha layer.

Re-exports rank-correlation primitives from ``probes._rank`` so
``alpha`` users don't reach into the probes namespace directly.
The probes module is the canonical implementation; this module
is a thin shim that lives under the alpha firewall.
"""
from __future__ import annotations

from aiphaforge.probes._rank import midranks, tie_corrected_spearman

__all__ = [
    "midranks",
    "tie_corrected_spearman",
]
