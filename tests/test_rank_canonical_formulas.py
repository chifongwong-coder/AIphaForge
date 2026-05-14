"""v2.4 Commit M — Canonical-formula tests for probes/_rank.py.

Pin the tie-corrected Spearman ρ implementation against
independent hand computation derived from the Kendall (1948)
formula, NOT a re-run of scipy.stats.spearmanr (which would
just check 'production matches scipy', not 'production matches
the paper formula').

WARNING: do NOT regenerate expected values from production code.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aiphaforge.probes._rank import midranks, tie_corrected_spearman


class TestSpearmanCanonicalFormula:
    def test_tie_free_5_element_matches_hand_computation(self):
        # 5 distinct values, no ties.
        # x_ranks: [1, 2, 3, 4, 5]
        # y_ranks: [5, 4, 3, 2, 1] (perfect negative correlation)
        x_ranks = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_ranks = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        # Hand-compute via Kendall:
        #   d = x - y = [-4, -2, 0, 2, 4]
        #   sum(d²) = 16 + 4 + 0 + 4 + 16 = 40
        #   N = 5; N³ - N = 125 - 5 = 120
        #   t_x = 0, t_y = 0 (no ties)
        #   numerator = 120 - 6·40 - 0 = 120 - 240 = -120
        #   denom = √(120 · 120) = 120
        #   ρ = -120 / 120 = -1.0
        rho = tie_corrected_spearman(x_ranks, y_ranks)
        assert rho == pytest.approx(-1.0, abs=1e-12)

    def test_one_tied_pair_matches_kendall_tie_correction(self):
        # 5 values where x has a tied pair (positions 2 and 3 both
        # at midrank 2.5). y is perfectly negatively ordered.
        # Use midranks helper to be sure x_ranks has the documented
        # 1-indexed midranks shape.
        x_values = np.array([10.0, 20.0, 20.0, 40.0, 50.0])
        y_values = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
        x_ranks = midranks(x_values)
        y_ranks = midranks(y_values)
        # x_ranks: [1, 2.5, 2.5, 4, 5]; y_ranks: [5, 4, 3, 2, 1].
        # Hand compute Kendall tie-corrected:
        #   d = x - y = [-4, -1.5, -0.5, 2, 4]
        #   sum(d²) = 16 + 2.25 + 0.25 + 4 + 16 = 38.5
        #   N = 5; N³ - N = 120
        #   t_x = 2³ - 2 = 6 (one tie-group of size 2)
        #   t_y = 0
        #   numerator = 120 - 6·38.5 - 0.5·(6 + 0)
        #             = 120 - 231 - 3 = -114
        #   denom = √((120 - 6) · (120 - 0)) = √(114 · 120) = √13680
        #         ≈ 116.95
        #   ρ ≈ -114 / 116.95 ≈ -0.97478
        d = x_ranks - y_ranks
        sum_d2 = float(np.sum(d ** 2))
        n_cubed_minus_n = 5 ** 3 - 5
        t_x = 2 ** 3 - 2
        t_y = 0
        numerator = n_cubed_minus_n - 6.0 * sum_d2 - 0.5 * (t_x + t_y)
        denom = math.sqrt(
            (n_cubed_minus_n - t_x) * (n_cubed_minus_n - t_y)
        )
        expected = numerator / denom
        rho = tie_corrected_spearman(x_ranks, y_ranks)
        assert rho == pytest.approx(expected, abs=1e-12)
