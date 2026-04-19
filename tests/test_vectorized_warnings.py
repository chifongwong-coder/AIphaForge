"""Vectorized-mode "config silently ignored" warnings (v1.9.7).

The vectorized core consumes only a subset of BacktestEngine's
config. v1.9.7 emits a UserWarning when the user passes any of the
following to vectorized mode with a non-default value:
- (position_sizing, position_size) — composite warning
- take_profit, trailing_stop_rule, impact_model, margin_config,
  periodic_cost_model — one warning per field
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from aiphaforge import (
    BacktestEngine,
    PositionSizing,
)
from aiphaforge.exit_rules import TrailingStopLoss
from aiphaforge.fees import ZeroFeeModel
from aiphaforge.margin import BorrowingCostModel, MarginConfig
from aiphaforge.market_impact import SquareRootImpactModel


def _make_data(n: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "volume": [1e6] * n,
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )


def _make_signals(n: int = 30) -> pd.Series:
    s = pd.Series(np.nan, index=pd.bdate_range("2024-01-01", periods=n),
                  dtype=float)
    s.iloc[5] = 1.0
    return s


def _vectorized_warns(**engine_kwargs) -> list[warnings.WarningMessage]:
    """Construct + run a vectorized engine; return captured warnings."""
    eng = BacktestEngine(
        mode="vectorized", fee_model=ZeroFeeModel(),
        include_benchmark=False, **engine_kwargs,
    )
    eng.set_signals(_make_signals())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eng.run(_make_data())
    return [w for w in caught
            if issubclass(w.category, UserWarning)
            and ("vectorized mode" in str(w.message)
                 or "ignored" in str(w.message))]


class TestDefaultsEmitNoWarnings:
    def test_bare_vectorized_engine_emits_no_warning(self):
        warns = _vectorized_warns()
        assert warns == []


class TestSizingComposite:
    def test_position_sizing_change_fires_one_composite_warning(self):
        warns = _vectorized_warns(
            position_sizing=PositionSizing.FIXED_SIZE,
            position_size=100.0,
        )
        assert len(warns) == 1
        msg = str(warns[0].message)
        assert "position_sizing" in msg
        assert "position_size" in msg
        # The composite message includes BOTH field values, not two separate
        # warnings (clean UX).

    def test_position_size_change_alone_fires_composite(self):
        warns = _vectorized_warns(position_size=0.5)
        assert len(warns) == 1
        assert "position_size" in str(warns[0].message)

    def test_default_sizing_emits_zero(self):
        warns = _vectorized_warns(
            position_sizing=PositionSizing.FIXED_FRACTION,
            position_size=0.95,
        )
        assert warns == []


class TestUnsupportedFieldWarnings:
    def test_take_profit_warns(self):
        warns = _vectorized_warns(take_profit=0.05)
        msgs = [str(w.message) for w in warns]
        assert any("take_profit" in m for m in msgs)

    def test_trailing_stop_warns(self):
        warns = _vectorized_warns(
            trailing_stop_rule=TrailingStopLoss(trail_percent=0.03))
        msgs = [str(w.message) for w in warns]
        assert any("trailing_stop_rule" in m for m in msgs)

    def test_impact_model_warns(self):
        warns = _vectorized_warns(impact_model=SquareRootImpactModel())
        msgs = [str(w.message) for w in warns]
        assert any("impact_model" in m for m in msgs)

    def test_margin_config_warns(self):
        warns = _vectorized_warns(
            margin_config=MarginConfig(
                initial_margin_ratio=0.5,
                maintenance_margin_ratio=0.3,
                borrowing_rate=0.05))
        msgs = [str(w.message) for w in warns]
        assert any("margin_config" in m for m in msgs)

    def test_periodic_cost_model_warns(self):
        warns = _vectorized_warns(periodic_cost_model=BorrowingCostModel())
        msgs = [str(w.message) for w in warns]
        assert any("periodic_cost_model" in m for m in msgs)

    def test_stop_loss_does_NOT_warn(self):
        # stop_loss IS honored in vectorized mode
        # (core_vectorized.py:56-58 calls stop_loss_rule.apply_vectorized)
        warns = _vectorized_warns(stop_loss=0.05)
        assert warns == []


class TestEventDrivenEmitsNothing:
    def test_event_driven_with_all_features_no_warnings(self):
        eng = BacktestEngine(
            mode="event_driven", fee_model=ZeroFeeModel(),
            include_benchmark=False,
            take_profit=0.05,
            impact_model=SquareRootImpactModel(),
            position_size=0.5,
        )
        eng.set_signals(_make_signals())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            eng.run(_make_data())
        # No vectorized-related warnings (the engine ones go to event-driven path)
        for w in caught:
            assert "vectorized mode" not in str(w.message)


class TestSourceOfTruth:
    def test_warn_target_fields_match_init_signature_defaults(self):
        # Future kwarg additions to __init__ shouldn't bypass the warn
        # loop unnoticed: the field list must derive from inspect.signature.
        init_defaults = {
            name: param.default
            for name, param in
            inspect.signature(BacktestEngine.__init__).parameters.items()
            if param.default is not inspect.Parameter.empty
        }
        for field in BacktestEngine._VECTORIZED_UNSUPPORTED_FIELDS:
            assert field in init_defaults, (
                f"{field!r} listed in _VECTORIZED_UNSUPPORTED_FIELDS but "
                f"not present as a kwarg in BacktestEngine.__init__")
        # position_sizing / position_size handled jointly in code, not
        # via the field list.
        for joint_field in ("position_sizing", "position_size"):
            assert joint_field in init_defaults
