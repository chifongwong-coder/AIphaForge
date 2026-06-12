"""v2.8.6 Commit D — spread model unit tests.

``spread_bps`` quotes the FULL bid-ask spread; a taker pays half per
side. ``VolatilitySpread`` consumes the rolling per-bar (unannualized)
Parkinson volatility; ``volatility=None`` (pipeline inactive or
warming up) falls back to the ``min_bps`` floor.
"""
from __future__ import annotations

import pytest

from aiphaforge import BaseSpreadModel, FixedSpread, VolatilitySpread


def test_fixed_spread_half_spread_math():
    model = FixedSpread(10)  # 10 bps full spread
    assert model.half_spread(100.0) == pytest.approx(0.05)
    assert model.half_spread(100.0, volatility=0.5) == pytest.approx(0.05)


def test_fixed_spread_negative_bps_raises():
    with pytest.raises(ValueError, match="spread_bps"):
        FixedSpread(-1)


def test_volatility_spread_scales_with_vol():
    model = VolatilitySpread(k=0.1)
    one = model.half_spread(100.0, volatility=0.01)
    two = model.half_spread(100.0, volatility=0.02)
    assert one == pytest.approx(100.0 * 0.1 * 0.01 / 2)
    assert two == pytest.approx(2 * one)


def test_volatility_spread_clamps_min_max_and_none_vol():
    model = VolatilitySpread(k=0.1, min_bps=5, max_bps=20)
    # Below the floor: k*vol = 1e-4 (1 bp) -> floored at 5 bps full.
    floored = model.half_spread(100.0, volatility=0.001)
    assert floored == pytest.approx(100.0 * 5 / 1e4 / 2)
    # Above the cap: k*vol = 1e-2 (100 bps) -> capped at 20 bps full.
    capped = model.half_spread(100.0, volatility=0.1)
    assert capped == pytest.approx(100.0 * 20 / 1e4 / 2)
    # None (inactive/warmup) -> min_bps floor.
    assert model.half_spread(100.0, volatility=None) == floored


def test_base_spread_model_is_abstract():
    with pytest.raises(TypeError):
        BaseSpreadModel()


def test_requires_volatility_flags():
    assert FixedSpread(10).requires_volatility is False
    assert VolatilitySpread(0.1).requires_volatility is True
