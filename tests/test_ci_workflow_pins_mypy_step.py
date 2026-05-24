"""v2.8.4 M14 — CI workflow mypy step parity test.

Pins the v2.8.4 M14 Phase 1 contract:
- An advisory ``mypy src/aiphaforge/`` step exists in the test job with
  ``continue-on-error: true`` (the broad type check runs but never
  blocks merge).
- A second scoped-blocking ``mypy src/aiphaforge/alpha/`` step exists
  WITHOUT ``continue-on-error``, so type regressions inside
  ``src/aiphaforge/alpha/`` block merge.
"""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CI_YML = _REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _test_job_steps() -> list[dict]:
    """Return the list of step dicts from the ci.yml test job."""
    config = yaml.safe_load(_CI_YML.read_text(encoding="utf-8"))
    assert "jobs" in config and "test" in config["jobs"], (
        "ci.yml missing 'jobs.test'"
    )
    steps = config["jobs"]["test"].get("steps", [])
    assert steps, "ci.yml test job has no steps"
    return steps


def _find_step_running(steps: list[dict], prefix: str) -> dict | None:
    """Find the first step whose ``run:`` starts with ``prefix``."""
    for step in steps:
        run_cmd = step.get("run", "")
        if isinstance(run_cmd, str) and run_cmd.strip().startswith(prefix):
            return step
    return None


def test_ci_workflow_runs_mypy_with_continue_on_error() -> None:
    """v2.8.4 M14 contract: advisory broad mypy + blocking scoped mypy.

    The advisory step covers ``src/aiphaforge/`` (broad) and must carry
    ``continue-on-error: true``.  The scoped-blocking step covers
    ``src/aiphaforge/alpha/`` (currently 0 errors per v3 MED-C lock)
    and must NOT carry ``continue-on-error: true``.
    """
    steps = _test_job_steps()

    advisory = _find_step_running(steps, "mypy src/aiphaforge/\n") or \
        _find_step_running(steps, "mypy src/aiphaforge/ ") or \
        next(
            (s for s in steps
             if isinstance(s.get("run"), str)
             and s["run"].strip() == "mypy src/aiphaforge/"),
            None,
        )
    assert advisory is not None, (
        "ci.yml missing advisory 'mypy src/aiphaforge/' step"
    )
    assert advisory.get("continue-on-error") is True, (
        "Advisory mypy step must set continue-on-error: true so it "
        "never blocks merge; saw: "
        f"{advisory.get('continue-on-error')!r}"
    )

    scoped = next(
        (s for s in steps
         if isinstance(s.get("run"), str)
         and s["run"].strip() == "mypy src/aiphaforge/alpha/"),
        None,
    )
    assert scoped is not None, (
        "ci.yml missing scoped-blocking 'mypy src/aiphaforge/alpha/' step"
    )
    assert scoped.get("continue-on-error") is not True, (
        "Scoped-blocking mypy step must NOT set continue-on-error: true "
        "(it is the v2.8.4 M14 blocking gate); saw: "
        f"{scoped.get('continue-on-error')!r}"
    )
