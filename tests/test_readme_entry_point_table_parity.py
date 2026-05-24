"""v2.8.4 M12 — README "Pick Your Entry Point" table parity test.

Pins that every public ``set_*`` method on ``BacktestEngine`` is either
named in the README decision table OR explicitly carved out on the
allowlist below.  Keeps the README from silently drifting out of sync
with the engine surface.
"""
from __future__ import annotations

import re
from pathlib import Path

from aiphaforge.engine import BacktestEngine

_REPO_ROOT = Path(__file__).resolve().parent.parent
_README = _REPO_ROOT / "README.md"

# Methods that are intentionally NOT in the entry-point table because
# they configure something other than input shape.
_ALLOWLIST: frozenset[str] = frozenset({"set_fee_model"})


def _extract_entry_point_table_methods() -> set[str]:
    """Return the set of ``set_*`` method names referenced in the table.

    Parses the markdown block that starts at the
    ``### Pick Your Entry Point`` heading and stops at the next heading
    of the same or higher level (``## Features``).  Within that block
    every backtick-wrapped token of shape ``set_<name>(`` is picked up.
    """
    text = _README.read_text(encoding="utf-8")
    start_idx = text.find("### Pick Your Entry Point")
    assert start_idx != -1, "README missing '### Pick Your Entry Point' heading"
    # The h3 block ends when the next h2 (or h3) heading is reached.
    end_idx = text.find("\n## ", start_idx)
    assert end_idx != -1, "README h3 block has no terminating h2 heading"
    block = text[start_idx:end_idx]
    return set(re.findall(r"`(set_[a-z_]+)\(", block))


def test_set_methods_in_readme_table_or_allowlist() -> None:
    """Every public ``BacktestEngine.set_*`` method must be discoverable.

    For each ``set_*`` method on ``BacktestEngine`` (skipping private
    helpers like ``_clear_wide_input_state``), assert it is either named
    in the README "Pick Your Entry Point" table or on the explicit
    allowlist.  This prevents silent drift between the engine API and
    the user-facing decision table.
    """
    engine_methods = {
        name
        for name in dir(BacktestEngine)
        if name.startswith("set_") and not name.startswith("_")
        and callable(getattr(BacktestEngine, name))
    }
    table_methods = _extract_entry_point_table_methods()
    missing = engine_methods - table_methods - _ALLOWLIST
    assert not missing, (
        f"BacktestEngine setters missing from README entry-point table: "
        f"{sorted(missing)}.  Either add a row to README.md "
        f"'### Pick Your Entry Point' or extend the allowlist in "
        f"{__file__}."
    )
