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

# v2.8.5 reverse-lock tightening: when rule 2 of
# ``test_module_no_accidentally_public_symbols`` short-circuits on
# foreign ``__module__``, it now requires the value to be either a
# class / function (the canonical re-export shape) or carry one of the
# allowlisted typing/future origins. Foreign-class-instance leaks fall
# through to the lock assertion.
#
# Allowlist source: kept in lockstep with
# ``tests/_helpers/reverse_lock_audit.py::_TYPING_ORIGIN_MODULES`` plus
# the ``__future__`` module. The lockstep is asserted by
# ``test_reverse_lock_allowlist_matches_audit_script`` below.
_ALLOWED_FOREIGN_ORIGINS = frozenset({
    "typing",
    "typing_extensions",
    "collections.abc",
    "_collections_abc",
    "__future__",
})


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
    # v2.8.6 — new public module, locked at introduction (1)
    "spread",
    # v2.9.0 — neutral shared-primitives module (hoist), locked at introduction (1)
    "stats",
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
        #
        # v2.8.5 tightening: only short-circuit on foreign origin if
        # the symbol is a class or function (the canonical re-export
        # shape) or if its origin module is on the typing / __future__
        # allowlist (the 205-symbol noise floor at v2.8.5 release).
        # Foreign-class-instance leaks fall through to the lock check.
        origin = getattr(val, "__module__", None)
        if origin is not None and origin != module_name:
            if inspect.isclass(val) or inspect.isfunction(val):
                continue
            if origin in _ALLOWED_FOREIGN_ORIGINS:
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


def test_reverse_lock_allowlist_matches_audit_script() -> None:
    """v2.8.5 reverse-lock tightening: the lock test's
    ``_ALLOWED_FOREIGN_ORIGINS`` must equal the audit script's
    ``_TYPING_ORIGIN_MODULES`` union ``{_FUTURE_MODULE}``.

    The two surfaces hold the same allowlist for the same reason and
    must move together. If a future audit run grows the typing-generic
    family or the project adopts a new ``__future__``-style module,
    update both sites in the same commit.
    """
    from tests._helpers.reverse_lock_audit import (
        _FUTURE_MODULE,
        _TYPING_ORIGIN_MODULES,
    )

    expected = frozenset(_TYPING_ORIGIN_MODULES) | {_FUTURE_MODULE}
    assert _ALLOWED_FOREIGN_ORIGINS == expected, (
        f"_ALLOWED_FOREIGN_ORIGINS ({sorted(_ALLOWED_FOREIGN_ORIGINS)}) "
        f"diverges from the audit script's "
        f"_TYPING_ORIGIN_MODULES ∪ {{_FUTURE_MODULE}} "
        f"({sorted(expected)}). Update either site to match the other."
    )


def test_reverse_lock_other_category_remains_zero() -> None:
    """v2.8.5 reverse-lock tightening: the audit script's "other"
    category (foreign-origin symbols that are NOT classes / functions
    / typing-generics / __future__ imports) must remain empty.

    A non-zero "other" count means a real accidentally-public surface
    has slipped through the lock and needs investigation — typically
    by renaming the symbol with a ``_`` prefix or adding it to the
    relevant ``__all__``.
    """
    from tests._helpers.reverse_lock_audit import audit_reverse_lock

    result = audit_reverse_lock()
    other_count = result["by_category"]["other"]
    other_symbols = [
        f"{s['module']}.{s['name']} (origin={s['origin']!r})"
        for s in result["flagged_symbols"]
        if s["category"] == "other"
    ]
    assert other_count == 0, (
        f"reverse-lock audit found {other_count} 'other'-category "
        f"flagged symbols; investigate:\n  "
        + "\n  ".join(other_symbols)
    )
