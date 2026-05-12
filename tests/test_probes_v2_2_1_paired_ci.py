"""v2.2.1 #3 — paired-difference CI tests (Wald approximation).

The function tango_paired_diff_ci ships as a Wald-style score CI
in v2.2.1; the proper Tango (1998) constrained-MLE implementation
is a v2.2.2 follow-up requiring R + PropCIs to generate the
reference fixture. These tests verify the Wald implementation
behaves correctly as a Wald CI: monotone in confidence level,
contains delta_hat, perfect-agreement collapses to (0, 0), etc.
"""
from __future__ import annotations

import pytest

from aiphaforge.probes.orchestrator import tango_paired_diff_ci


class TestPairedDiffCi:
    def test_perfect_agreement_returns_point_zero(self):
        lo, hi = tango_paired_diff_ci(
            n_both=10, n_real_only=0, n_anchor_only=0, n_neither=10,
        )
        assert lo == 0.0
        assert hi == 0.0

    def test_empty_bucket_raises(self):
        with pytest.raises(ValueError, match="empty bucket"):
            tango_paired_diff_ci(0, 0, 0, 0)

    def test_ci_contains_delta_hat(self):
        n_both, b, c, d = 20, 30, 5, 15
        n = n_both + b + c + d
        delta_hat = (b - c) / n
        lo, hi = tango_paired_diff_ci(n_both, b, c, d)
        assert lo <= delta_hat <= hi

    def test_ci_widens_with_confidence(self):
        n_both, b, c, d = 20, 30, 5, 15
        lo_95, hi_95 = tango_paired_diff_ci(
            n_both, b, c, d, confidence=0.95,
        )
        lo_99, hi_99 = tango_paired_diff_ci(
            n_both, b, c, d, confidence=0.99,
        )
        assert lo_99 <= lo_95
        assert hi_99 >= hi_95

    def test_real_advantage_yields_positive_ci_lower(self):
        # 30 discordant pairs with real-only=25, anchor-only=5
        lo, hi = tango_paired_diff_ci(
            n_both=10, n_real_only=25, n_anchor_only=5,
            n_neither=10,
        )
        assert lo > 0  # CI excludes zero

    def test_handles_all_discordant(self):
        # n_both = n_neither = 0; only off-diagonal cells
        lo, hi = tango_paired_diff_ci(
            n_both=0, n_real_only=15, n_anchor_only=5,
            n_neither=0,
        )
        # delta_hat = (15-5)/20 = 0.5; CI brackets it
        assert lo <= 0.5 <= hi

    def test_ci_in_unit_interval(self):
        n_both, b, c, d = 5, 10, 8, 7
        lo, hi = tango_paired_diff_ci(n_both, b, c, d)
        assert -1.0 <= lo <= 1.0
        assert -1.0 <= hi <= 1.0


class TestWaldImplementationDocumentation:
    """v2.2.1 #3: docstring honestly labels this as Wald, not Tango."""

    def test_docstring_acknowledges_wald_approximation(self):
        doc = tango_paired_diff_ci.__doc__ or ""
        # Honest docstring per r7: notes "Wald-style", "approximation
        # to Tango", and "v2.2.2 follow-up".
        assert "wald" in doc.lower() or "Wald" in doc
        assert "v2.2.2" in doc or "PropCIs" in doc
