"""v2.8.2 M4 — MetaContext.adjust_strategy_params in-place mutation docs.

MetaContext.adjust_strategy_params calls strategy.update_params which
mutates the strategy instance in place. If a user shares a strategy
instance across parallel-backtest workers, the adjustment from one
run becomes visible to ALL other callers — silent cross-contamination
of results.

v2.8.2 documents the pattern + ships the factory-per-worker recipe in
the docstring. Performance optimization (partial-timeline regen) is
deferred to v2.8.3, designed jointly with the IncrementalFactor
engine integration.

These tests pin:
1. The docstring contract (factory recipe + warning text).
2. A demonstrator that a shared strategy DOES leak adjustment across
   independent MetaContext instances — concrete regression that, if
   v2.8.3 changes mutation semantics, the test breaks loudly so docs
   stay accurate.
3. The factory pattern isolates adjustments correctly.
"""
from __future__ import annotations

from aiphaforge.meta import MetaContext
from aiphaforge.strategies import MACrossover

# --------------------------------------------------------------------
# Test 1 — Docstring carries the warning text + factory recipe
# --------------------------------------------------------------------

def test_adjust_strategy_params_documents_in_place_mutation():
    doc = MetaContext.adjust_strategy_params.__doc__ or ""
    # Must say it mutates:
    assert "MUTATES" in doc
    # Must surface the factory-per-worker pattern:
    assert "factory per worker" in doc.lower() or "factory-per-worker" in doc.lower()
    # Must mention parallel/Pool to ground the migration recipe:
    assert "Pool" in doc or "parallel" in doc.lower()


# --------------------------------------------------------------------
# Test 2 — Shared strategy instance leaks adjustment across MetaContexts
# --------------------------------------------------------------------

def test_shared_strategy_instance_leaks_adjustment_across_runs():
    """Concrete regression: build one MACrossover instance, set it on
    two independent MetaContext objects, adjust on the first, observe
    the second's strategy state has changed.

    Pins the documented behavior so a v2.8.3 fix (e.g. cloning the
    strategy on attach) would visibly break this test as a signal to
    update the docs.
    """
    strategy = MACrossover(short=5, long=20)

    ctx_a = MetaContext(config=None)
    ctx_a.set_strategy(strategy)

    ctx_b = MetaContext(config=None)
    ctx_b.set_strategy(strategy)  # same instance

    # Mutate through ctx_a:
    ctx_a.adjust_strategy_params(short=10)

    # ctx_b sees the mutation because both reference the same object:
    assert ctx_b._strategy is strategy
    assert ctx_b._strategy.short == 10, (
        "shared-strategy contamination expected — if this fails, "
        "v2.8.3 may have introduced isolation; update the docstring "
        "to match the new behavior."
    )


# --------------------------------------------------------------------
# Test 3 — Factory pattern isolates adjustments across runs
# --------------------------------------------------------------------

def test_factory_pattern_isolates_adjustments_across_runs():
    """Recommended migration: a factory function per run produces a
    fresh strategy instance, so adjustments don't leak."""
    def make_strategy():
        return MACrossover(short=5, long=20)

    ctx_a = MetaContext(config=None)
    ctx_a.set_strategy(make_strategy())

    ctx_b = MetaContext(config=None)
    ctx_b.set_strategy(make_strategy())

    ctx_a.adjust_strategy_params(short=10)

    # ctx_b's strategy is a separate instance — unaffected by ctx_a's
    # adjustment.
    assert ctx_a._strategy is not ctx_b._strategy
    assert ctx_a._strategy.short == 10
    assert ctx_b._strategy.short == 5  # pristine
