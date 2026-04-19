"""Lock package version against pyproject.toml.

This test is intentionally added in v1.9.6 commit 1 and will fail until
commit 8 bumps both `__version__` and `pyproject.toml` to '1.9.6'.
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import aiphaforge

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_VERSION = "1.9.7"


def _read_pyproject_version() -> str:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def test_version_matches_pyproject():
    assert aiphaforge.__version__ == _read_pyproject_version(), (
        f"aiphaforge.__version__ ({aiphaforge.__version__}) does not "
        f"match pyproject.toml version ({_read_pyproject_version()})"
    )


def test_version_is_current_release():
    assert aiphaforge.__version__ == EXPECTED_VERSION, (
        f"Expected v{EXPECTED_VERSION}; found {aiphaforge.__version__}"
    )
