"""v2.0.1 M3 — tests for ToleranceProfile per-asset-class presets."""
from __future__ import annotations

import pytest

from aiphaforge.probes import ToleranceProfile

# Loose / strict pairs to validate symmetrically.
_PAIRS = [
    ("us_equity_price", "us_equity_price_strict"),
    ("crypto_price", "crypto_price_strict"),
    ("futures_price", "futures_price_strict"),
    ("penny_stock_price", "penny_stock_price_strict"),
]


@pytest.mark.parametrize("loose_name,strict_name", _PAIRS)
class TestToleranceProfilePresetPairs:
    def test_loose_returns_profile_with_max_range_width(self, loose_name, strict_name):
        loose = getattr(ToleranceProfile, loose_name)()
        assert isinstance(loose, ToleranceProfile)
        # Anti-gaming cap MUST be on by default per plan §A3.
        assert loose.max_range_width is not None
        assert loose.max_range_width > 0

    def test_strict_returns_profile_with_max_range_width(self, loose_name, strict_name):
        strict = getattr(ToleranceProfile, strict_name)()
        assert isinstance(strict, ToleranceProfile)
        assert strict.max_range_width is not None
        assert strict.max_range_width > 0

    def test_threshold_ordering_holds_loose(self, loose_name, strict_name):
        p = getattr(ToleranceProfile, loose_name)()
        assert 0 < p.exact_threshold < p.near_threshold < p.rough_threshold

    def test_threshold_ordering_holds_strict(self, loose_name, strict_name):
        p = getattr(ToleranceProfile, strict_name)()
        assert 0 < p.exact_threshold < p.near_threshold < p.rough_threshold

    def test_range_width_ordering_holds_loose(self, loose_name, strict_name):
        p = getattr(ToleranceProfile, loose_name)()
        assert 0 < p.exact_range_width <= p.near_range_width <= p.rough_range_width
        assert p.rough_range_width <= p.max_range_width

    def test_range_width_ordering_holds_strict(self, loose_name, strict_name):
        p = getattr(ToleranceProfile, strict_name)()
        assert 0 < p.exact_range_width <= p.near_range_width <= p.rough_range_width
        assert p.rough_range_width <= p.max_range_width

    def test_strict_exact_is_strictly_tighter_than_loose(self, loose_name, strict_name):
        loose = getattr(ToleranceProfile, loose_name)()
        strict = getattr(ToleranceProfile, strict_name)()
        # The discriminative point of the strict variant is a tighter
        # exact threshold — for memorization detection.
        assert strict.exact_threshold < loose.exact_threshold


class TestUSEquityValues:
    """The us_equity_price loose preset MUST match the v2.0 default
    (`_PRICE_TOLERANCE` in questions.py) so users who currently rely
    on the v2.0 OHLC template defaults get the same scoring when
    they switch to the named preset.
    """

    def test_loose_us_equity_matches_v2_0_price_tolerance(self):
        from aiphaforge.probes.questions import _PRICE_TOLERANCE
        loose = ToleranceProfile.us_equity_price()
        assert loose.exact_threshold == _PRICE_TOLERANCE.exact_threshold
        assert loose.near_threshold == _PRICE_TOLERANCE.near_threshold
        assert loose.rough_threshold == _PRICE_TOLERANCE.rough_threshold
        assert loose.max_range_width == _PRICE_TOLERANCE.max_range_width

    def test_strict_us_equity_is_basis_point_scale(self):
        # ~1 bp exact for memorization detection — sub-cent precision
        # for typical equity prices.
        strict = ToleranceProfile.us_equity_price_strict()
        assert strict.exact_threshold == pytest.approx(0.0001)


class TestPresetUsageWithTemplate:
    def test_template_accepts_strict_preset(self):
        # Smoke: a built-in template constructed with a strict preset
        # round-trips into a QuestionSpec without error.
        from aiphaforge.probes import OpenQuestion
        from tests.conftest import make_probe_ohlcv
        data = make_probe_ohlcv(n=20)
        q = OpenQuestion(
            tolerance=ToleranceProfile.us_equity_price_strict()
        ).build(data, "X", data.index[5])
        assert q.tolerance.exact_threshold == pytest.approx(0.0001)
