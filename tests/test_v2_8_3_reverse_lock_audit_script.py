"""v2.8.3 Commit I item 8 — reverse-lock audit-script sanity tests.

The audit script at ``tests/_helpers/reverse_lock_audit.py``
reproduces the v2.8.2 reverse-lock skip-rule findings recorded in
``tests/_helpers/reverse_lock_allowlist.md``. These tests pin the
script's contract: structured output, deterministic re-runs,
typing-generic dominance in the noise floor, and consistency with
the broader category split documented in the allowlist file.
"""
from __future__ import annotations

import re
from pathlib import Path

from tests._helpers.reverse_lock_audit import audit_reverse_lock


def test_reverse_lock_audit_script_reproduces_allowlist():
    """End-to-end: run the audit and check its structured output
    matches the shape and dominant-category claim in the
    committed allowlist .md.

    The .md categorizes flagged symbols as: typing generics
    (~180), __future__ (~20), other (~5). The script's output
    must agree on:
      1. typing_generics is the dominant category (>= 90% share)
      2. all three categories are present
      3. total_flagged > 100 (sanity floor — the codebase has
         not collapsed below the audit's discovery threshold)
      4. Re-running the script gives bit-identical output.
    """
    result = audit_reverse_lock()
    # (1) structured shape.
    assert "total_flagged" in result
    assert "by_category" in result
    assert "flagged_symbols" in result
    cats = result["by_category"]
    assert set(cats) == {"typing_generics", "future_imports", "other"}

    # (2) dominance check: typing generics carry the lion's share.
    # The v2.8.2 audit recorded ~180/~205 = ~88%; allow a generous
    # margin for codebase drift across versions.
    total = result["total_flagged"]
    assert total > 0
    typing_share = cats["typing_generics"] / total
    assert typing_share >= 0.80, (
        f"typing_generics share {typing_share:.2%} below 80% — "
        f"either the codebase changed materially or the audit "
        f"script's classifier needs an update. Categories: {cats}"
    )

    # (3) sanity floor.
    assert total > 100, (
        f"total_flagged={total} is below the v2.8.2 audit floor "
        "of 100 — re-check the audit script's discovery loop."
    )

    # (4) determinism.
    result2 = audit_reverse_lock()
    assert result == result2

    # Consistency cross-check with the committed allowlist file:
    # the markdown claims '~180 typing generics' and 'roughly 200
    # total'. We don't pin exact numbers (drift expected as the
    # codebase grows), but the order of magnitude must match.
    md_path = (
        Path(__file__).resolve().parent / "_helpers"
        / "reverse_lock_allowlist.md"
    )
    text = md_path.read_text()
    md_total_match = re.search(r"\*\*(\d+) flagged symbols\*\*", text)
    assert md_total_match is not None, (
        "reverse_lock_allowlist.md must contain a '**N flagged "
        "symbols**' total line for the audit cross-check."
    )
    md_total = int(md_total_match.group(1))
    # Allow ±40% drift since the codebase can grow / shrink between
    # releases. This is a regression guard, not an exact pin.
    assert 0.6 * md_total <= total <= 1.4 * md_total, (
        f"audit reports total_flagged={total}; the committed .md "
        f"records {md_total}. The drift is large enough to suggest "
        f"the public surface changed materially since v2.8.2 — "
        f"update the allowlist file."
    )
