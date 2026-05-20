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


# --------------------------------------------------------------------
# Commit L (Option D) — both kwargs supplied → warn + notional wins
# --------------------------------------------------------------------

def test_apply_vectorized_warns_when_both_representative_kwargs_set():
    """When the user passes BOTH representative_notional and
    representative_size, the fee model's commission-rate query can use
    only one (price, size) pair. Notional wins per documented
    precedence; size is dropped with a one-time warning."""
    from aiphaforge import USStockFeeModel
    from aiphaforge.costs import DefaultTradeCost
    import aiphaforge.costs as costs_module

    n = 50
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    close = 100.0
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1e6},
        index=idx,
    )
    positions = pd.Series(0.0, index=idx)
    positions.iloc[5:10] = 1.0
    returns = pd.Series(0.0, index=idx)

    # Reset module-level once-flag so this test reliably observes the
    # warning regardless of test ordering.
    costs_module._BOTH_KWARGS_WARNED = False

    tc = DefaultTradeCost()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        tc.apply_vectorized(
            returns, positions, data,
            USStockFeeModel(commission_per_share=0.005, min_commission=1.0),
            initial_capital=100_000,
            representative_notional=50_000.0,
            representative_size=200.0,
        )
    msgs = [str(w.message) for w in captured
            if issubclass(w.category, UserWarning)]
    assert any("both representative_notional and representative_size" in m
               and "notional takes precedence" in m for m in msgs), (
        f"expected both-kwargs UserWarning; got {msgs!r}"
    )


def test_apply_vectorized_both_kwargs_notional_wins_numerically():
    """When both kwargs are supplied, the resulting cost rate must
    match the notional-only path (NOT the size-only path)."""
    from aiphaforge import USStockFeeModel
    from aiphaforge.costs import DefaultTradeCost
    import aiphaforge.costs as costs_module

    n = 50
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    close = 100.0
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1e6},
        index=idx,
    )
    positions = pd.Series(0.0, index=idx)
    positions.iloc[5:10] = 1.0  # one trade in, one out
    returns = pd.Series(0.0, index=idx)

    tc = DefaultTradeCost()
    fee = USStockFeeModel(commission_per_share=0.005, min_commission=1.0)

    # Path 1: notional only.
    net_notional_only = tc.apply_vectorized(
        returns.copy(), positions, data, fee,
        initial_capital=100_000, representative_notional=50_000.0,
    )
    # Path 2: both supplied — size should be dropped, behavior must
    # match notional-only.
    costs_module._BOTH_KWARGS_WARNED = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        net_both = tc.apply_vectorized(
            returns.copy(), positions, data, fee,
            initial_capital=100_000,
            representative_notional=50_000.0,
            representative_size=200.0,
        )
    pd.testing.assert_series_equal(net_notional_only, net_both)


def test_apply_vectorized_no_warning_when_only_one_kwarg_set():
    """Single-kwarg paths must NOT emit the both-set warning. Pins
    that the warning only fires in the both-set ambiguity case."""
    from aiphaforge import USStockFeeModel
    from aiphaforge.costs import DefaultTradeCost
    import aiphaforge.costs as costs_module

    n = 30
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    close = 100.0
    data = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1e6},
        index=idx,
    )
    positions = pd.Series(0.5, index=idx)
    positions.iloc[0] = 0.0
    returns = pd.Series(0.0, index=idx)
    tc = DefaultTradeCost()
    fee = USStockFeeModel(commission_per_share=0.005, min_commission=1.0)

    for kwargs in [
        {"representative_notional": 50_000.0},
        {"representative_size": 500.0},
    ]:
        costs_module._BOTH_KWARGS_WARNED = False
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            tc.apply_vectorized(returns.copy(), positions, data, fee,
                                initial_capital=100_000, **kwargs)
        both_warns = [w for w in captured
                      if issubclass(w.category, UserWarning)
                      and "both representative" in str(w.message)]
        assert not both_warns, (
            f"unexpected both-kwargs warning when only {list(kwargs)} "
            f"was set: {[str(w.message) for w in both_warns]}"
        )
