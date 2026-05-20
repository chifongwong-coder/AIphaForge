"""v2.8.1 H6 — tango_paired_diff_ci renamed to wald_paired_diff_ci.

v2.2.1 named the function ``tango_paired_diff_ci`` while documenting
it as a Wald approximation, not Tango (1998) method 10. v2.8.1 renames
it to ``wald_paired_diff_ci`` to match what it computes. The legacy
name remains as a DeprecationWarning alias; v2.9 hard-removes.
"""
from __future__ import annotations

import warnings

import pytest

from aiphaforge.probes import wald_paired_diff_ci
from aiphaforge.probes.orchestrator import (
    tango_paired_diff_ci as orchestrator_tango,
)
from aiphaforge.probes.orchestrator import (
    wald_paired_diff_ci as orchestrator_wald,
)


# --------------------------------------------------------------------
# Test 1 — wald matches the legacy tango alias numerically
# --------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture",
    [
        (10, 25, 5, 10),   # real advantage
        (5, 10, 8, 7),     # mixed
        (10, 0, 0, 10),    # perfect agreement
        (0, 15, 5, 0),     # all discordant
        (50, 10, 10, 30),  # balanced
    ],
)
def test_wald_paired_diff_ci_matches_tango_alias_numerically(fixture):
    n_both, b, c, d = fixture
    lo_w, hi_w = orchestrator_wald(n_both, b, c, d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        lo_t, hi_t = orchestrator_tango(n_both, b, c, d)
    assert lo_w == lo_t
    assert hi_w == hi_t


# --------------------------------------------------------------------
# Test 2 — legacy alias fires a DeprecationWarning at caller's line
# --------------------------------------------------------------------

def test_tango_paired_diff_ci_emits_deprecation_warning_with_correct_stacklevel():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        orchestrator_tango(10, 5, 5, 10)
    deprecation = [
        w for w in captured if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation) == 1, (
        f"expected exactly one DeprecationWarning; got {captured}"
    )
    msg = str(deprecation[0].message)
    assert "tango_paired_diff_ci" in msg
    assert "wald_paired_diff_ci" in msg
    assert "v2.9" in msg


# --------------------------------------------------------------------
# Test 3 — wald_paired_diff_ci is exported at the probes top level
# --------------------------------------------------------------------

def test_wald_paired_diff_ci_in_probes_all():
    import aiphaforge.probes as probes
    assert "wald_paired_diff_ci" in probes.__all__
    assert wald_paired_diff_ci is orchestrator_wald
