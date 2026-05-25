"""v2.8.5 L7 — meta-test pinning the privacy of three probes helpers.

``_canonical_json``, ``_normalize_choice`` and
``_serialize_collision_examples`` are documented in
``aiphaforge/probes/__init__.py`` as test-internal helpers, subject to
change without notice. This meta-test asserts that no non-test src
module outside the ``probes`` subpackage imports them — protecting the
"NOT promote" decision from accidental drift.
"""
from __future__ import annotations

import ast
from pathlib import Path

_PRIVATE_PROBES_SYMBOLS = (
    "_canonical_json",
    "_normalize_choice",
    "_serialize_collision_examples",
)


def test_probes_private_symbols_not_imported_outside_probes() -> None:
    """Meta-test: the 3 private probes helpers must NOT leak into
    non-probes src modules.

    Walks every ``.py`` file under ``src/aiphaforge/`` excluding the
    ``probes/`` subpackage and asserts that none of them imports
    ``_canonical_json``, ``_normalize_choice`` or
    ``_serialize_collision_examples`` via ``from ... import ...``.

    Note that ``import dotted.module`` syntax cannot reference a
    private symbol by name, hence only ``ast.ImportFrom`` is walked.
    """
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "src" / "aiphaforge"
    probes_root = src_root / "probes"
    violations: list[str] = []
    for py in src_root.rglob("*.py"):
        try:
            py.relative_to(probes_root)
            continue
        except ValueError:
            pass
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in _PRIVATE_PROBES_SYMBOLS:
                        violations.append(
                            f"{py.relative_to(repo_root)}: "
                            f"imports {alias.name}"
                        )
    assert not violations, "\n".join(violations)
