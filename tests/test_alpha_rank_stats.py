"""v2.4 Commit E — alpha.rank_stats tests."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from aiphaforge.alpha.rank_stats import midranks, tie_corrected_spearman
from aiphaforge.probes import _rank as probes_rank


class TestAlphaRankStatsExports:
    def test_re_exports_match_probes_rank(self):
        # Same callable identity — alpha.rank_stats is a shim,
        # not a re-implementation.
        assert midranks is probes_rank.midranks
        assert tie_corrected_spearman is probes_rank.tie_corrected_spearman

    def test_callable_round_trip(self):
        # Sanity: can be called via the alpha import path.
        ranks_x = midranks(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        ranks_y = midranks(np.array([5.0, 4.0, 3.0, 2.0, 1.0]))
        rho = tie_corrected_spearman(ranks_x, ranks_y)
        assert rho == -1.0  # perfect negative monotone


class TestAlphaRankStatsFirewall:
    def test_alpha_rank_stats_does_not_import_engine(self):
        # AST guard per master plan v1.0 §1.2.
        path = (
            Path(__file__).resolve().parent.parent
            / "src/aiphaforge/alpha/rank_stats.py"
        )
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert "aiphaforge.engine" not in name, (
                    f"alpha/rank_stats.py imports {name!r} — breaks alpha "
                    f"firewall"
                )
