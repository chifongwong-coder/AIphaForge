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

v2.8.1 H8 extends this with a REVERSE-direction sentinel
(``test_module_no_accidentally_public_symbols``): every non-`_`
module-level attribute MUST appear in __all__. Catches the
"accidentally public" failure mode where a private helper without
the `_` prefix gets imported by external code and becomes part of
the de-facto contract.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil

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


def _all_locked_modules() -> list[str]:
    """Auto-discover every aiphaforge module (incl. package __init__.py)
    that declares __all__.

    v2.8.1 H8 / v2.1.1 fix: walk package __init__.py files too. The
    top-level ``aiphaforge`` package and the per-subpackage __init__
    are also public surface and deserve the same lock as plain modules.
    """
    import aiphaforge
    found: list[str] = []
    # Top-level package itself.
    if hasattr(aiphaforge, "__all__"):
        found.append("aiphaforge")
    # All submodules + subpackage __init__.py.
    for _finder, name, _ispkg in pkgutil.walk_packages(
        aiphaforge.__path__, prefix="aiphaforge."
    ):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(mod, "__all__"):
            found.append(name)
    return sorted(found)


@pytest.mark.parametrize("module_name", _all_locked_modules())
def test_module_no_accidentally_public_symbols(module_name: str) -> None:
    """v2.8.1 H8 — reverse-direction lock.

    For every module with ``__all__``, every public (non-`_`) module-
    level attribute MUST appear in ``__all__``. Catches the
    accidentally-public failure mode where a helper without the `_`
    prefix slips into the de-facto public surface via external import.
    """
    mod = importlib.import_module(module_name)
    declared = set(mod.__all__)
    for name in dir(mod):
        if name.startswith("_"):
            continue
        val = getattr(mod, name)
        # Skip re-exported modules (these are import-side artifacts,
        # not API surface — Python convention).
        if inspect.ismodule(val):
            continue
        # Skip symbols imported from a different module — they are
        # not THIS module's API surface. Classes / functions /
        # exceptions defined elsewhere carry __module__ pointing
        # back to their origin module; only originals get locked
        # by this module's __all__.
        origin = getattr(val, "__module__", None)
        if origin is not None and origin != module_name:
            continue
        # Primitive constants (bool, int, float, str, None) without
        # __module__ are almost always cross-module imports
        # (TRADING_DAYS_STOCK pulled from utils into engine, the
        # `TYPE_CHECKING` constant from typing, etc.). Skip them
        # rather than force every importing module to re-export.
        if origin is None and isinstance(val, (bool, int, float, str)):
            continue
        assert name in declared, (
            f"{module_name}.{name} is public (no '_' prefix) but not "
            f"in {module_name}.__all__ — either rename to _{name} or "
            f"add to __all__."
        )


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
