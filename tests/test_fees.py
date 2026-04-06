"""
Tests for fee models.
"""

import pytest

from aiphaforge.fees import (
    ChinaAShareFeeModel,
    ZeroFeeModel,
    get_fee_model,
)


class TestFeeModels:
    """Tests for ZeroFeeModel and ChinaAShareFeeModel behavior."""

    def test_zero_fee_model(self):
        """ZeroFeeModel should return 0 for all cost components."""
        fm = ZeroFeeModel()
        assert fm.calculate_commission(100.0, 1000, "buy") == 0.0
        assert fm.calculate_commission(100.0, 1000, "sell") == 0.0
        assert fm.calculate_slippage(100.0, 1000, "buy") == 0.0
        assert fm.total_cost(100.0, 1000, "buy") == 0.0

    def test_china_ashare_stamp_duty_sell_only(self):
        """China A-share stamp duty should apply on sell only."""
        fm = ChinaAShareFeeModel(
            commission_rate=0.0003,
            min_commission=5.0,
            stamp_duty_rate=0.001,
            transfer_fee_rate=0.0,
            slippage_pct=0.0,
        )
        notional = 10.0 * 1000  # 10,000

        buy_fee = fm.calculate_commission(10.0, 1000, "buy")
        sell_fee = fm.calculate_commission(10.0, 1000, "sell")

        # Buy: commission only (min 5.0), no stamp duty
        expected_buy = max(notional * 0.0003, 5.0)
        assert buy_fee == pytest.approx(expected_buy, abs=1e-6)

        # Sell: commission + stamp duty
        expected_sell = max(notional * 0.0003, 5.0) + notional * 0.001
        assert sell_fee == pytest.approx(expected_sell, abs=1e-6)

        # Sell should cost more than buy
        assert sell_fee > buy_fee


class TestGetFeeModelFactory:

    @pytest.mark.parametrize(
        "alias,expected_name",
        [
            ("china", "china_a_share"),
            ("cn", "china_a_share"),
            ("a_share", "china_a_share"),
            ("us", "us_stock"),
            ("us_stock", "us_stock"),
            ("crypto", "crypto_spot"),
            ("crypto_spot", "crypto_spot"),
            ("crypto_futures", "crypto_futures"),
            ("futures", "crypto_futures"),
            ("simple", "simple"),
            ("zero", "zero"),
            ("none", "zero"),
        ],
    )
    def test_get_fee_model_factory_aliases(self, alias: str, expected_name: str):
        """get_fee_model should resolve all documented aliases."""
        fm = get_fee_model(alias)
        assert fm.name == expected_name
