"""v2.8.2 Commit F — v2.8.1 deferred test-hardening items.

The v2.8.1 post-implementation 4-reviewer chain surfaced 5 test
robustness items. v2.8.2 lands 4 of them; the 5th (reverse-lock
skip-rule tightening) is documented in
``tests/_helpers/reverse_lock_allowlist.md`` and deferred to v2.8.5
test hygiene release because the audit surfaced ~200 typing-primitive
false positives that need an allowlist before tightening can ship.

This file:
- **TestRev-1**: pins that `DefaultTradeCost.apply_vectorized` calls
  `fee_model.estimate_commission_rate` with named kwargs matching the
  v2.8.1 H1 fix shape — protects against a regression that reverts to
  no-args.
- **TestRev-2**: pins that `tango_paired_diff_ci` deprecation warning
  has `stacklevel=2`, so the warning's filename is the CALLER's site
  not the alias's site inside aiphaforge.
- **TestRev-3-doc**: documents the known reverse-lock escape hatch
  (foreign-class instances pass the lock). v2.8.5 will tighten and
  this pin will be flipped to assert that the lock catches such
  instances.
- **TestRev-4**: pins the ABC additive break — a v2.8.0-shape
  custom `BaseTradeCost` subclass raises a clear TypeError that
  names `representative_notional`.
"""
from __future__ import annotations

import inspect
import warnings
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aiphaforge import BacktestEngine, USStockFeeModel, ZeroFeeModel
from aiphaforge.costs import BaseTradeCost, DefaultTradeCost
from aiphaforge.probes.orchestrator import tango_paired_diff_ci

# --------------------------------------------------------------------
# TestRev-1 — H1 dispatch call_args pin
# --------------------------------------------------------------------

def test_h1_apply_vectorized_calls_estimate_commission_rate_with_named_kwargs():
    """Verify the v2.8.1 H1 fix shape: apply_vectorized must call
    estimate_commission_rate with named (price, size, side) kwargs,
    NOT positional defaults (the v2.8.0 over-billing bug)."""
    n = 30
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
         "volume": 1e6},
        index=idx,
    )
    positions = pd.Series(0.5, index=idx)
    positions.iloc[0] = 0.0
    returns = pd.Series(0.0, index=idx)

    fake_fee = MagicMock(spec=USStockFeeModel)
    fake_fee.estimate_commission_rate.return_value = 0.0005
    fake_fee.slippage_pct = 0.001

    tc = DefaultTradeCost()
    tc.apply_vectorized(
        returns, positions, data, fake_fee,
        initial_capital=100_000, representative_notional=50_000.0,
    )

    # Asserts: estimate_commission_rate was called at least once and
    # EVERY call used the explicit price=/size=/side= named kwargs.
    # (Not `assert_called_once` — a future refactor that queries
    # per-side or per-trade would legitimately produce multiple calls,
    # and we want the test to keep passing as long as the call SHAPE
    # is right. The PIN is on the call shape — protect against a
    # regression that drops kwargs and reverts to v2.8.0 no-args bug.)
    assert fake_fee.estimate_commission_rate.call_count >= 1
    for call in fake_fee.estimate_commission_rate.call_args_list:
        assert set(call.kwargs.keys()) == {"price", "size", "side"}, (
            f"unexpected call signature: args={call.args}, "
            f"kwargs={call.kwargs}"
        )
        assert call.kwargs["side"] == "average"


# --------------------------------------------------------------------
# TestRev-2 — H6 deprecation warning stacklevel
# --------------------------------------------------------------------

def test_h6_tango_paired_diff_ci_deprecation_warning_originates_outside_aiphaforge():
    """The DeprecationWarning emitted by the legacy alias should
    originate at the CALLER's frame (this test file), NOT inside
    aiphaforge — that's what stacklevel=2 guarantees. A regression
    to stacklevel=1 would point the warning at orchestrator.py."""
    import aiphaforge
    aiphaforge_root = inspect.getfile(aiphaforge).rsplit("/", 1)[0]

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        tango_paired_diff_ci(10, 5, 5, 10)

    dep_warnings = [w for w in captured
                    if issubclass(w.category, DeprecationWarning)
                    and "tango_paired_diff_ci" in str(w.message)]
    assert len(dep_warnings) == 1
    fname = dep_warnings[0].filename
    # The warning's filename should be OUTSIDE the aiphaforge src
    # tree (i.e., this test file or somewhere in the user's code).
    # If the warning is fired with stacklevel=1, fname would be
    # inside src/aiphaforge/probes/orchestrator.py.
    assert not fname.startswith(aiphaforge_root), (
        f"DeprecationWarning originated INSIDE aiphaforge at {fname!r}; "
        f"stacklevel should be 2 so it points at the user's call site."
    )


# --------------------------------------------------------------------
# TestRev-3-doc — REMOVED in v2.8.5 Commit J.
#
# The v2.8.2-era pin ``test_reverse_lock_currently_passes_foreign_class_instances``
# documented the rule-2 blind spot (instances of foreign classes
# silently skipped). v2.8.5 Commit J ships the tightener
# (``inspect.isclass`` / ``isfunction`` predicate + typing/future
# allowlist), so the pin no longer applies. See
# ``tests/test_v2_8_public_api_lock.py::test_reverse_lock_other_category_remains_zero``
# for the new regression guard.
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# TestRev-4 — ABC additive break TypeError surface
# --------------------------------------------------------------------

def test_base_trade_cost_legacy_subclass_raises_typeerror_with_actionable_message():
    """v2.8.1 widened the BaseTradeCost.apply_vectorized ABC with two
    optional kwargs (representative_notional, representative_size).
    External subclasses overriding with the v2.8.0 (kwargs-less)
    signature will TypeError on the first vectorized run. Pin that
    the error message names the offending kwarg so users can fix it
    quickly."""

    class LegacyTradeCost(BaseTradeCost):
        # v2.8.0 signature — no kwargs after initial_capital
        def apply_vectorized(
            self,
            returns,
            positions,
            data,
            fee_model,
            initial_capital,
        ):
            return returns

    n = 20
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
         "volume": 1e6},
        index=idx,
    )

    eng = BacktestEngine(
        fee_model=ZeroFeeModel(),
        initial_capital=100_000,
        mode="vectorized",
        include_benchmark=False,
    )
    # Inject the legacy subclass:
    eng._trade_cost = LegacyTradeCost()
    eng.set_signals(pd.Series(1.0, index=idx, dtype=float))

    with pytest.raises(TypeError) as exc_info:
        eng.run(data)

    msg = str(exc_info.value)
    # The user-facing TypeError surfaces the offending kwarg name —
    # so the fix is obvious (add representative_notional / _size or
    # **kwargs to the override).
    assert "representative_notional" in msg or "representative_size" in msg, (
        f"TypeError should name the offending kwarg; got: {msg!r}"
    )
