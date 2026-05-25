"""Stricter validate_ohlcv contract (v1.9.6, B5).

Strict mode now rejects:
* non-finite prices (``inf`` / ``-inf``),
* non-positive prices (``<= 0`` for OHLC columns).

Volume of ``0`` remains valid (delisted bars, no-trade days).
Warn / none modes preserve the legacy lenient behavior.
"""
from __future__ import annotations

import pytest

from aiphaforge.utils import validate_ohlcv
from tests._helpers.ohlcv import make_ohlcv_from_closes as _ohlcv


class TestStrictRejection:
    def test_strict_rejects_inf_price(self):
        df = _ohlcv([100.0, float("inf"), 102.0])
        with pytest.raises(ValueError, match="non-finite"):
            validate_ohlcv(df, validation_level="strict")

    def test_strict_rejects_negative_inf_price(self):
        df = _ohlcv([100.0, float("-inf"), 102.0])
        with pytest.raises(ValueError, match="non-finite"):
            validate_ohlcv(df, validation_level="strict")

    def test_strict_rejects_zero_price(self):
        df = _ohlcv([100.0, 0.0, 102.0])
        with pytest.raises(ValueError, match="non-positive"):
            validate_ohlcv(df, validation_level="strict")

    def test_strict_rejects_negative_price(self):
        df = _ohlcv([100.0, -1.5, 102.0])
        with pytest.raises(ValueError, match="non-positive"):
            validate_ohlcv(df, validation_level="strict")

    def test_strict_accepts_zero_volume(self):
        df = _ohlcv([100.0, 101.0, 102.0], volumes=[1.0, 0.0, 1.0])
        # Should not raise — volume of 0 is allowed.
        validate_ohlcv(df, validation_level="strict")

    def test_strict_accepts_clean_data(self):
        df = _ohlcv([100.0, 101.0, 102.0])
        validate_ohlcv(df, validation_level="strict")


class TestWarnMode:
    def test_warn_does_not_raise_on_inf(self):
        df = _ohlcv([100.0, float("inf"), 102.0])
        with pytest.warns(UserWarning, match="non-finite"):
            validate_ohlcv(df, validation_level="warn")

    def test_warn_does_not_raise_on_zero_price(self):
        df = _ohlcv([100.0, 0.0, 102.0])
        with pytest.warns(UserWarning, match="non-positive"):
            validate_ohlcv(df, validation_level="warn")


class TestNoneMode:
    def test_none_skips_all_quality_checks(self):
        df = _ohlcv([100.0, 0.0, float("inf")])
        # 'none' is the legacy escape hatch; only column existence is checked.
        validate_ohlcv(df, validation_level="none")
