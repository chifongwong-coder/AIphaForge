"""v2.8 Commit I.2 — public API lock sentinel.

Parametrized over 32 modules:
- 30 newly-locked in v2.8 Commits F/G/H/I.1/I.2.
- 2 already-locked cross-checks (factor_strategy, signal_strategy
  declared __all__ in v2.3/v2.4).

For each, assert:
1. __all__ is declared at module level.
2. __all__ is a list/tuple of strings.
3. __all__ is non-empty.
4. Every entry resolves to a real module attribute via getattr.

This sentinel pins the v2.8 "public surface declared" contract. Any
v3.0 reorganization that adds a public module or renames a public
symbol WILL break this test unless the rename is reflected in
__all__.
"""
from __future__ import annotations

import importlib

import pytest

# 30 newly-locked + 2 already-locked cross-checks = 32 cases.
_LOCKED_MODULES = [
    # Commit F — core engine (8)
    "engine", "broker", "core_event_driven", "core_vectorized",
    "config", "orders", "hooks", "latency",
    # Commit G — execution & costs (7)
    "fees", "costs", "margin", "market_impact", "position_sizing",
    "capital_allocator", "corporate_actions",
    # Commit H — analysis & results (8)
    "results", "performance", "risk", "exit_rules", "portfolio",
    "portfolio_optimizer", "optimizer", "significance",
    # Commit I.1 — utilities (6)
    "data", "utils", "indicators", "meta", "plotting",
    "incremental_factors",
    # Commit I.2 — strategies (1)
    "strategies",
    # Already-locked cross-checks (2 — declared in v2.3/v2.4)
    "factor_strategy", "signal_strategy",
]


@pytest.mark.parametrize("module_name", _LOCKED_MODULES)
def test_module_declares_valid_public_all(module_name: str) -> None:
    mod = importlib.import_module(f"aiphaforge.{module_name}")
    assert hasattr(mod, "__all__"), (
        f"aiphaforge.{module_name} must declare __all__ per v2.8 "
        f"public API lock"
    )
    all_attr = mod.__all__
    assert isinstance(all_attr, (list, tuple)), (
        f"aiphaforge.{module_name}.__all__ must be list/tuple; "
        f"got {type(all_attr).__name__}"
    )
    assert len(all_attr) > 0, (
        f"aiphaforge.{module_name}.__all__ must be non-empty"
    )
    for name in all_attr:
        assert isinstance(name, str), (
            f"aiphaforge.{module_name}.__all__ entries must be str; "
            f"got {type(name).__name__}: {name!r}"
        )
        assert hasattr(mod, name), (
            f"aiphaforge.{module_name}.__all__ declares {name!r} but "
            f"the module has no such attribute"
        )
