"""v2.8.6 Commit H — FundingRateModel 8h-rate conversion.

``per_bar = funding_rate_8h * bar_interval_seconds / 28800`` (exact
linear proration; 96 x 15m bars accrue the same as 3 x 8h periods).
The parameter is named ``bar_interval_seconds`` to avoid colliding
with the intentionally-ignored ``calculate_cost(bar_seconds=...)``
runtime kwarg.
"""
from __future__ import annotations

import pytest

from aiphaforge import FundingRateModel


def test_funding_8h_conversion_15m_bars():
    model = FundingRateModel(
        funding_rate_8h=0.0001, bar_interval_seconds=900)
    assert model.funding_rate_per_bar == pytest.approx(0.0001 / 32)


def test_both_rates_specified_raises():
    with pytest.raises(ValueError, match="mutually exclusive"):
        FundingRateModel(0.0001, funding_rate_8h=0.0001,
                         bar_interval_seconds=900)


def test_8h_without_bar_interval_raises():
    with pytest.raises(ValueError, match="bar_interval_seconds"):
        FundingRateModel(funding_rate_8h=0.0001)


def test_no_arg_default_unchanged():
    assert FundingRateModel().funding_rate_per_bar == pytest.approx(0.0001)
    # Legacy positional usage keeps working.
    assert FundingRateModel(0.0002).funding_rate_per_bar == pytest.approx(
        0.0002)
