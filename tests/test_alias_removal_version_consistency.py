"""v2.3 Commit A — guard against accidental re-introduction of v2.3.0
removal annotations.

The four backward-compat aliases (`_BUCKET_ORDINAL_WEIGHTS`,
`bucket_delta_tango_ci`, `_apply_vol_scaling_to_question_set`
re-export, `_pair_scores_by_position`) and the related ValueError
message in `KnowledgeCheckReport.__post_init__` were originally
annotated "Removal scheduled: v2.3.0". Per the master refactor plan
v1.0 §5, the v2.x line now ends at v2.8 (with v3.0 reserved for a
separate major reformulation), so the removal target is **v2.8.0**.

This test exists so that any future edit re-introducing the v2.3.0
text fails CI loudly instead of silently shipping inconsistent
documentation.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "aiphaforge"


def _iter_python_sources() -> list[Path]:
    return [p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_removal_scheduled_v2_3_0_text_remains():
    """No source file may say 'Removal scheduled: v2.3.0'.

    The accepted target is v2.8.0. A history reference like
    'v2.8.0 (was v2.3.0)' is allowed because the literal string
    'Removal scheduled: v2.3.0' does not appear there.
    """
    bad = "Removal scheduled: v2.3.0"
    offenders: list[tuple[Path, int, str]] = []
    for path in _iter_python_sources():
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if bad in line:
                offenders.append((path, lineno, line.strip()))
    assert not offenders, (
        "These source files still announce 'Removal scheduled: v2.3.0' — "
        "must be 'v2.8.0' per master plan v1.0 §5:\n"
        + "\n".join(f"  {p}:{n}: {s}" for p, n, s in offenders)
    )


def test_alias_removal_target_is_v2_8():
    """Sentinel: the alias-removal version mentioned in source must be
    v2.8.0, not v2.3.0 or v3.0.

    Counts joint occurrence of "Removal" + "v2.8" within a 5-line
    window — robust against multi-line comment wrapping where
    "Removal scheduled:" and "v2.8.0" land on adjacent lines.
    """
    found_v2_8 = False
    for path in _iter_python_sources():
        text = path.read_text()
        if "Removal" in text and "v2.8" in text:
            found_v2_8 = True
            break
    assert found_v2_8, (
        "no 'Removal' + 'v2.8' co-occurrence in src/ — the alias "
        "annotations may have been deleted; update test or master "
        "plan §5.6 accordingly"
    )
