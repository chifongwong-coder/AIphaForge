"""v2.5 Commit J — explicit Exception fallback policy tests.

Locks in the master §5.3 fix: catch Exception (not AttributeError),
and the once-per-(composite_cls, child_cls) WARNING policy.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from aiphaforge.strategies import (
    _FALLBACK_WARNED,
    BaseStrategy,
    WeightedBlend,
    _resolve_child_signals,
)


@pytest.fixture(autouse=True)
def _clear_fallback_state():
    _FALLBACK_WARNED.clear()
    yield
    _FALLBACK_WARNED.clear()


def _df(periods=20):
    rng = np.random.default_rng(0)
    closes = 100.0 + np.cumsum(rng.normal(0, 1, periods))
    return pd.DataFrame(
        {
            "open": closes, "high": closes, "low": closes,
            "close": closes, "volume": [1.0] * periods,
        },
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


class _RaisesTypeError(BaseStrategy):
    name = "raises_type_error"

    def generate_signals(self, data):
        raise TypeError("data shape mismatch (simulated)")

    def _compute(self, df):
        return pd.Series(0.0, index=df.index)


class _RaisesKeyError(BaseStrategy):
    name = "raises_key_error"

    def generate_signals(self, data):
        raise KeyError("close")

    def _compute(self, df):
        return pd.Series(0.0, index=df.index)


class _RaisesSystemExit(BaseStrategy):
    name = "raises_system_exit"

    def generate_signals(self, data):
        raise SystemExit(2)

    def _compute(self, df):
        return pd.Series(0.0, index=df.index)


class _CompositeA:
    pass


class _CompositeB:
    pass


def test_auto_catches_typeerror_from_data_dict_to_compute():
    out = _resolve_child_signals(
        _CompositeA, _RaisesTypeError(), _df(), mode="auto",
    )
    assert (out == 0.0).all()


def test_auto_catches_keyerror_from_missing_close_column():
    out = _resolve_child_signals(
        _CompositeA, _RaisesKeyError(), _df(), mode="auto",
    )
    assert (out == 0.0).all()


def test_auto_does_not_catch_systemexit():
    # SystemExit derives from BaseException (not Exception). It must
    # propagate so external orchestration can shut down cleanly.
    with pytest.raises(SystemExit):
        _resolve_child_signals(
            _CompositeA, _RaisesSystemExit(), _df(), mode="auto",
        )


def test_auto_logs_composite_and_child_class_names(caplog):
    with caplog.at_level(logging.WARNING, logger="aiphaforge.strategies"):
        _resolve_child_signals(
            WeightedBlend, _RaisesKeyError(), _df(), mode="auto",
        )
    msg = caplog.records[0].message
    assert "WeightedBlend" in msg
    assert "_RaisesKeyError" in msg
    assert "KeyError" in msg


def test_auto_different_child_classes_each_log_warning_once(caplog):
    # Same composite, two different child classes: each gets its own
    # WARNING (the policy is keyed on the (composite, child) PAIR).
    with caplog.at_level(logging.DEBUG, logger="aiphaforge.strategies"):
        _resolve_child_signals(
            WeightedBlend, _RaisesKeyError(), _df(), mode="auto",
        )
        _resolve_child_signals(
            WeightedBlend, _RaisesTypeError(), _df(), mode="auto",
        )
        # Repeat the first pair: stays at DEBUG.
        _resolve_child_signals(
            WeightedBlend, _RaisesKeyError(), _df(), mode="auto",
        )
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert len(warnings) == 2, (
        f"expected 2 warnings (one per distinct child class); got "
        f"{[r.message for r in warnings]}"
    )
    # The 3rd call (same pair as 1st) should have logged DEBUG.
    assert any(
        "_RaisesKeyError" in r.message for r in debugs
    ), "repeated pair should fall through to DEBUG"
