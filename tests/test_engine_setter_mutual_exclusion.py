"""v2.7 Commit E — 30-pair mutual-exclusion sentinel for the 6 setters.

Per v2.7 plan §"Engine state vector" the engine has 8 mutual-
exclusion fields owned by 5 distinct setter groups:

| Setter | Owns                                                              |
|---|---|
| set_strategy             | _strategy                                  |
| set_signals              | _signals                                   |
| set_target_weights       | _target_weights                            |
| set_signals_wide         | _signals_wide, _signals_wide_inf_action,  |
|                          | _signals_wide_strict                       |
| set_score_wide           | (none distinct — delegates to set_signals_wide) |
| set_target_weights_wide  | _target_weights_wide,                      |
|                          | _target_weights_wide_config                |

Carve-out (per v3-decision #23 + v3.1 B4): `set_signals_wide` and
`set_score_wide` are in the same wide-signal group (score-wide
delegates to signals-wide), so transitions between them keep the
wide-signal triple populated rather than zeroing it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from aiphaforge.engine import BacktestEngine
from aiphaforge.signal_rules import ThresholdScoreRule
from aiphaforge.signals import transitions_only  # noqa: F401 — keep import for completeness

_IDX = pd.bdate_range("2024-01-01", periods=10)


# Setter callable lambdas — each takes a fresh engine and applies a
# minimal valid input. Names match the public API.
SETTERS = {
    "set_strategy": lambda e: e.set_strategy(_DummyStrategy()),
    "set_signals": lambda e: e.set_signals(pd.Series(1.0, index=_IDX)),
    "set_target_weights": lambda e: e.set_target_weights(
        {"2024-01-01": {"AAPL": 1.0}}
    ),
    "set_signals_wide": lambda e: e.set_signals_wide(
        pd.DataFrame({"AAPL": 1.0}, index=_IDX)
    ),
    "set_score_wide": lambda e: e.set_score_wide(
        pd.DataFrame({"AAPL": 0.8}, index=_IDX),
        ThresholdScoreRule(long_threshold=0.7),
    ),
    "set_target_weights_wide": lambda e: e.set_target_weights_wide(
        pd.DataFrame({"AAPL": 1.0}, index=_IDX)
    ),
}


class _DummyStrategy:
    name = "dummy"

    def generate_signals(self, data):
        if isinstance(data, dict):
            return {sym: pd.Series(1.0, index=df.index) for sym, df in data.items()}
        return pd.Series(1.0, index=data.index)


def _state_snapshot(engine: BacktestEngine) -> dict:
    """Snapshot the 8 mutual-exclusion fields as a dict of bools
    (True = populated, False = None / falsy)."""
    return {
        "_strategy": engine._strategy is not None,
        "_signals": engine._signals is not None,
        "_target_weights": engine._target_weights is not None,
        "_signals_wide": engine._signals_wide is not None,
        # The two scalar wide-signal fields default to "warn"/False —
        # treat them as part of the wide-signal triple. They're
        # populated iff _signals_wide is set.
        "_signals_wide_inf_action_set": engine._signals_wide is not None,
        "_signals_wide_strict_set": engine._signals_wide is not None,
        "_target_weights_wide": engine._target_weights_wide is not None,
        "_target_weights_wide_config": (
            engine._target_weights_wide_config is not None
        ),
    }


# Map setter name → which fields it OWNS in the snapshot vocabulary.
OWNS = {
    "set_strategy": {"_strategy"},
    "set_signals": {"_signals"},
    "set_target_weights": {"_target_weights"},
    "set_signals_wide": {
        "_signals_wide",
        "_signals_wide_inf_action_set",
        "_signals_wide_strict_set",
    },
    "set_score_wide": {  # delegates → same triple as set_signals_wide
        "_signals_wide",
        "_signals_wide_inf_action_set",
        "_signals_wide_strict_set",
    },
    "set_target_weights_wide": {
        "_target_weights_wide", "_target_weights_wide_config",
    },
}


_WIDE_SIGNAL_GROUP = {"set_signals_wide", "set_score_wide"}


@pytest.mark.parametrize(
    "setter_a,setter_b",
    [
        (a, b) for a in SETTERS for b in SETTERS if a != b
    ],
    ids=lambda s: s,
)
def test_six_setter_mutual_exclusion_30_ordered_pairs(
    setter_a: str, setter_b: str,
):
    """For each ordered pair (a, b) where a != b: applying a then b
    must leave the engine state with ONLY setter_b's owned fields
    populated, all others zeroed.

    Carve-out: when both setters belong to the wide-signal group
    (set_signals_wide ↔ set_score_wide), the wide-signal triple
    stays populated post-setter_b instead of being asserted None.
    """
    engine = BacktestEngine()
    SETTERS[setter_a](engine)
    state_after_a = _state_snapshot(engine)
    # setter_a's owned fields populated; all others None.
    for field, populated in state_after_a.items():
        if field in OWNS[setter_a]:
            assert populated, f"after {setter_a}: expected {field} populated"
        else:
            assert not populated, (
                f"after {setter_a}: expected {field} cleared, was populated"
            )

    SETTERS[setter_b](engine)
    state_after_b = _state_snapshot(engine)
    # Carve-out: wide-signal-group transitions keep the triple
    # populated (same fields owned by both setters).
    if {setter_a, setter_b} <= _WIDE_SIGNAL_GROUP:
        for field, populated in state_after_b.items():
            if field in OWNS[setter_b]:  # = OWNS[setter_a] = wide triple
                assert populated, (
                    f"wide-signal-group transition {setter_a}→{setter_b}: "
                    f"{field} should remain populated"
                )
            else:
                assert not populated, (
                    f"wide-signal-group transition {setter_a}→{setter_b}: "
                    f"{field} should be cleared"
                )
        return

    for field, populated in state_after_b.items():
        if field in OWNS[setter_b]:
            assert populated, (
                f"after {setter_a} → {setter_b}: expected {field} populated"
            )
        else:
            assert not populated, (
                f"after {setter_a} → {setter_b}: expected {field} cleared, "
                f"was populated"
            )
