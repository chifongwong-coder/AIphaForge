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

    # Advisory step: any mypy run targeting the full src/aiphaforge/
    # tree (post-Commit J adds --config-file / --follow-imports flags
    # for robustness). Identify by the trailing `src/aiphaforge/`
    # without the `alpha/` suffix.
    def _is_advisory(s: dict) -> bool:
        run = s.get("run", "")
        if not isinstance(run, str):
            return False
        run = run.strip()
        return (
            run.startswith("mypy ")
            and "src/aiphaforge/" in run
            and "src/aiphaforge/alpha/" not in run
        )

    advisory = next((s for s in steps if _is_advisory(s)), None)
    assert advisory is not None, (
        "ci.yml missing advisory mypy step targeting src/aiphaforge/"
    )
    assert advisory.get("continue-on-error") is True, (
        "Advisory mypy step must set continue-on-error: true so it "
        "never blocks merge; saw: "
        f"{advisory.get('continue-on-error')!r}"
    )

    # Scoped-blocking step: any mypy run targeting src/aiphaforge/alpha/.
    def _is_scoped(s: dict) -> bool:
        run = s.get("run", "")
        if not isinstance(run, str):
            return False
        return run.strip().startswith("mypy ") and "src/aiphaforge/alpha/" in run

    scoped = next((s for s in steps if _is_scoped(s)), None)
    assert scoped is not None, (
        "ci.yml missing scoped-blocking mypy step on src/aiphaforge/alpha/"
    )
    assert scoped.get("continue-on-error") is not True, (
        "Scoped-blocking mypy step must NOT set continue-on-error: true "
        "(it is the v2.8.4 M14 blocking gate); saw: "
        f"{scoped.get('continue-on-error')!r}"
    )
    # Commit J defensive hardening: the scoped step must include
    # --follow-imports=silent so the gate is isolated to alpha/ source
    # even when mypy's import discovery would otherwise reach errors
    # in transitive modules.
    scoped_cmd = scoped["run"].strip()
    assert "--follow-imports=silent" in scoped_cmd, (
        "Scoped mypy step must use --follow-imports=silent to isolate "
        f"the gate to alpha/ source; saw: {scoped_cmd!r}"
    )
