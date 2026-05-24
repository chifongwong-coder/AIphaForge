"""v2.8.4 M11 — CHANGELOG.md existence and sub-anchor preservation tests."""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHANGELOG = _REPO_ROOT / "CHANGELOG.md"


def test_changelog_root_file_exists() -> None:
    """CHANGELOG.md must live at the repo root per Keep-a-Changelog convention."""
    assert _CHANGELOG.is_file(), (
        f"Expected CHANGELOG.md at repo root ({_CHANGELOG}); not found."
    )


def test_changelog_pins_v2_8_anchors() -> None:
    """Top-level Keep-a-Changelog version headers must be present.

    Per the v2.8.4 M11 plan, the v2.8.1, v2.8.2, and v2.8.3 release notes were
    extracted from README.md into CHANGELOG.md. The CHANGELOG MUST keep the
    top-level `## [2.8.X]` headers so external URL anchors (#283, #282, #281)
    resolve.
    """
    text = _CHANGELOG.read_text(encoding="utf-8")
    for version in ("2.8.3", "2.8.2", "2.8.1"):
        assert f"## [{version}]" in text, (
            f"CHANGELOG.md missing top-level header '## [{version}]'"
        )


def test_changelog_preserves_v2_8_3_sub_anchors() -> None:
    """Verbatim copy must preserve the v2.8.3 sub-section headers.

    Per the v2.8.4 M11 plan step (b), markdown auto-anchors are heading-text
    derived; a verbatim copy preserves the anchor URL inside the new file.
    Spot-check a few canonical v2.8.3 sub-sections (especially the
    Lost-data playbook block called out by R7).
    """
    text = _CHANGELOG.read_text(encoding="utf-8")
    expected_sub_headings = (
        "### Lost-data playbook for v2.8.2 ContinuationProbe users",
        "### M10 pickle migration recipe",
        "### M7 honesty paragraph",
        "### Reverse-pickle caveat",
    )
    for heading in expected_sub_headings:
        assert heading in text, (
            f"CHANGELOG.md missing preserved sub-section heading: {heading!r}"
        )
