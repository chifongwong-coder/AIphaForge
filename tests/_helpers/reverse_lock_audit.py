"""Stand-alone reverse-lock skip-rule audit.

v2.8.3 Commit I item 8 — reproduces the audit findings recorded in
``tests/_helpers/reverse_lock_allowlist.md`` from current source.
The audit applies the v2.8.5 proposed tightening (rule 2 requires
``inspect.isclass(val) or inspect.isfunction(val)`` before skipping
on foreign ``__module__``) and categorizes every flagged symbol so
the project can track how the typing-generics noise floor evolves.

This is a stand-alone helper, not a pytest test. The companion test
``tests/test_v2_8_3_reverse_lock_audit_script.py`` invokes it and
sanity-checks the structure of its output.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

# Recognized typing-generic origin modules. Symbols imported from
# these modules (Dict, List, Optional, Mapping, etc.) are the
# "noise floor" the v2.8.5 tightening must allowlist before it
# can flip on safely.
_TYPING_ORIGIN_MODULES = frozenset({
    "typing",
    "typing_extensions",
    "collections.abc",
    "_collections_abc",
})

# Module-level constants that resolve to None __module__ AND are
# imported from __future__ (annotations, division, etc.).
_FUTURE_MODULE = "__future__"


def _classify(val: Any, origin: str | None) -> str:
    """Bucket a flagged symbol into one of three audit categories."""
    if origin in _TYPING_ORIGIN_MODULES:
        return "typing_generics"
    if origin == _FUTURE_MODULE:
        return "future_imports"
    return "other"


def _walk_aiphaforge_modules() -> list[str]:
    """Mirror the discovery in ``test_v2_8_public_api_lock.py`` —
    return every aiphaforge module / subpackage that declares
    ``__all__``.
    """
    import aiphaforge
    found: list[str] = []
    if hasattr(aiphaforge, "__all__"):
        found.append("aiphaforge")
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


def audit_reverse_lock() -> dict[str, Any]:
    """Run the audit. Returns a structured findings dict suitable
    for comparison with the committed allowlist .md.

    The dict has the shape::

        {
            "total_flagged": int,
            "by_category": {
                "typing_generics": int,
                "future_imports": int,
                "other": int,
            },
            "flagged_symbols": list[
                {"module": str, "name": str, "origin": str | None,
                 "category": str},
                ...
            ],
        }
    """
    flagged: list[dict[str, Any]] = []
    counts: dict[str, int] = {
        "typing_generics": 0,
        "future_imports": 0,
        "other": 0,
    }
    for module_name in _walk_aiphaforge_modules():
        mod = importlib.import_module(module_name)
        declared = set(getattr(mod, "__all__", []))
        for name in dir(mod):
            if name.startswith("_"):
                continue
            val = getattr(mod, name)
            if inspect.ismodule(val):
                continue
            origin = getattr(val, "__module__", None)
            # The v2.8.1 skip rule defers to origin != module_name.
            # The proposed v2.8.5 tightening requires class/function
            # before deferring on origin. Re-classify everything the
            # OLD rule skipped that the NEW rule would flag.
            if origin is not None and origin != module_name:
                if inspect.isclass(val) or inspect.isfunction(val):
                    # Still legitimately skipped under v2.8.5.
                    continue
            elif origin is None and isinstance(
                val, (bool, int, float, str)
            ):
                continue
            elif origin is None or origin == module_name:
                # Either truly local, or a module-private constant —
                # would already be caught by the existing lock.
                continue
            # Reaching here means: foreign-origin, NOT a class /
            # function. These are the symbols the v2.8.5 tightening
            # would surface.
            if name in declared:
                # Allowed via __all__; not a finding.
                continue
            category = _classify(val, origin)
            flagged.append({
                "module": module_name,
                "name": name,
                "origin": origin,
                "category": category,
            })
            counts[category] += 1
    return {
        "total_flagged": len(flagged),
        "by_category": counts,
        "flagged_symbols": flagged,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(audit_reverse_lock(), indent=2, default=str))
