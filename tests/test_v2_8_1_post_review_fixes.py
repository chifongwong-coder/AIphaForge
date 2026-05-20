"""v2.8.1 post-review fixes (Commit J).

Three real bugs surfaced by the 4-reviewer post-implementation audit:

- HIGH-1 (architect): ``BacktestEngine._resolve_representative_trade``
  silently dropped a user-passed ``representative_size`` when
  ``representative_notional`` was unset. The user-wins precedence was
  half-implemented.
- HIGH-2 (architect): ``KnowledgeCheckReport.__setstate__`` did not
  translate the legacy ``bucket_delta_tango_ci`` field name. Old
  pickles loaded with a ghost field set and the canonical
  ``bucket_delta_ci`` missing — any downstream read raised
  AttributeError.
- MED-3 (architect): ``build_synthetic_anchor`` derives H/L from the
  real bar's high/low columns, but did not pre-validate their
  existence. A real_data frame missing those columns produced a deep
  KeyError instead of a clean ValueError at the entry point.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from aiphaforge import BacktestEngine
from aiphaforge.probes.anchors import build_synthetic_anchor
from aiphaforge.probes.orchestrator import KnowledgeCheckReport


# --------------------------------------------------------------------
# HIGH-1 — user-passed representative_size survives when notional is None
# --------------------------------------------------------------------

def test_user_passed_representative_size_alone_survives_resolve():
    """User passes ONLY representative_size (no representative_notional).
    Engine must NOT overwrite with sizer-derived notional."""
    eng = BacktestEngine(
        initial_capital=100_000,
        position_size=0.95,  # FractionSizer → engine default would set notional
        representative_size=77.0,
    )
    notional, size = eng._resolve_representative_trade()
    assert notional is None, (
        f"engine overwrote with sizer notional despite user-passed "
        f"representative_size: got notional={notional}"
    )
    assert size == 77.0


def test_user_passed_both_representative_kwargs_pass_through():
    eng = BacktestEngine(
        initial_capital=100_000,
        position_size=0.95,
        representative_notional=42_000.0,
        representative_size=200.0,
    )
    notional, size = eng._resolve_representative_trade()
    assert notional == 42_000.0
    assert size == 200.0


def test_no_user_kwargs_falls_through_to_sizer_dispatch():
    eng = BacktestEngine(initial_capital=100_000, position_size=0.95)
    notional, size = eng._resolve_representative_trade()
    assert notional == pytest.approx(95_000.0)  # FractionSizer default path
    assert size is None


# --------------------------------------------------------------------
# HIGH-2 — __setstate__ translates legacy bucket_delta_tango_ci
# --------------------------------------------------------------------

def _baseline_state(report: KnowledgeCheckReport) -> dict:
    return report.__getstate__()


def _make_report() -> KnowledgeCheckReport:
    """Smallest valid report — re-uses the H7 baseline-kwargs helper."""
    from tests.test_v2_8_1_h7_report_pickle import _baseline_kwargs
    return KnowledgeCheckReport(**_baseline_kwargs())


def test_setstate_translates_legacy_bucket_delta_tango_ci_kwarg():
    """Simulate loading a v2.7.x pickle: state dict has
    bucket_delta_tango_ci set and bucket_delta_ci unset.

    Before the fix: bucket_delta_tango_ci becomes a ghost attribute,
    bucket_delta_ci is missing, downstream reads raise AttributeError.
    After the fix: legacy alias is translated to bucket_delta_ci with
    a DeprecationWarning.
    """
    state = _baseline_state(_make_report())
    # Strip the canonical field, inject the legacy field:
    state.pop("bucket_delta_ci", None)
    state["bucket_delta_tango_ci"] = {"all": (-0.05, 0.05)}

    blank = KnowledgeCheckReport.__new__(KnowledgeCheckReport)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        blank.__setstate__(state)

    dep_warnings = [w for w in caught
                    if issubclass(w.category, DeprecationWarning)
                    and "bucket_delta_tango_ci" in str(w.message)]
    assert len(dep_warnings) == 1, (
        f"expected DeprecationWarning naming bucket_delta_tango_ci; "
        f"got {caught}"
    )
    # Canonical field populated:
    assert blank.bucket_delta_ci == {"all": (-0.05, 0.05)}
    # Ghost field NOT set on the revived instance:
    assert not hasattr(blank, "bucket_delta_tango_ci")


def test_setstate_translation_preserves_explicit_bucket_delta_ci():
    """If the state dict has BOTH bucket_delta_ci (canonical) and
    bucket_delta_tango_ci (legacy ghost), the canonical wins; the
    legacy alias only fills when the canonical is absent."""
    state = _baseline_state(_make_report())
    state["bucket_delta_ci"] = {"canonical": (0.0, 0.1)}
    state["bucket_delta_tango_ci"] = {"legacy": (-1.0, 1.0)}

    blank = KnowledgeCheckReport.__new__(KnowledgeCheckReport)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        blank.__setstate__(state)
    assert blank.bucket_delta_ci == {"canonical": (0.0, 0.1)}
    assert not hasattr(blank, "bucket_delta_tango_ci")


# --------------------------------------------------------------------
# MED-3 — build_synthetic_anchor pre-validates real_data columns
# --------------------------------------------------------------------

@pytest.mark.parametrize("missing", ["high", "low", "volume", "close"])
def test_build_synthetic_anchor_missing_column_raises_clean_value_error(missing):
    n = 30
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    closes = np.linspace(100.0, 110.0, n)
    cols = {
        "open": closes, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "volume": 1e6,
    }
    cols.pop(missing)
    df = pd.DataFrame(cols, index=idx)
    with pytest.raises(ValueError, match="missing required column"):
        build_synthetic_anchor(df, seed=1, method="block_bootstrap")


def test_build_synthetic_anchor_complete_columns_does_not_raise():
    """Sanity baseline — the column validation does NOT false-positive
    on a complete OHLCV frame."""
    n = 30
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    closes = np.linspace(100.0, 110.0, n)
    df = pd.DataFrame(
        {"open": closes, "high": closes * 1.01, "low": closes * 0.99,
         "close": closes, "volume": 1e6},
        index=idx,
    )
    # Should not raise:
    build_synthetic_anchor(df, seed=1, method="block_bootstrap")
