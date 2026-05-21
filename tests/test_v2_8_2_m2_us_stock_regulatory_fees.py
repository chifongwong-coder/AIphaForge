"""v2.8.2 M2 — USStockFeeModel adds SEC §31 + FINRA TAF (sell-only).

v2.8.1 USStockFeeModel only computed broker commission. SEC §31 fee
(sell-only, regulatory transaction fee) and FINRA TAF (sell-only)
were absent. For HFT-style US backtests this under-bills sell-side
costs by ~$2.3 per $100k.

v2.8.2 adds three configurable kwargs with FY2026 / 2026 schedule
defaults:
- sec_fee_rate = 20.60e-6 (FY2026, effective 2026-04-04 per SEC
  Fee Rate Advisory FY2026-2)
- finra_taf_per_share = 0.000195 (2026 schedule, effective 2026-01-01)
- finra_taf_cap = 9.79

The regulatory fees are added on top of the broker commission,
independent of the broker's min_commission floor.
"""
from __future__ import annotations

import pytest

from aiphaforge import USStockFeeModel

# --------------------------------------------------------------------
# Test 1 — FY2026 default rate pin
# --------------------------------------------------------------------

def test_us_stock_sec_fee_default_matches_fy2026_rate():
    """If SEC publishes a new schedule, this test trips and reminds
    the maintainer to bump the default."""
    fee = USStockFeeModel()
    assert fee.sec_fee_rate == 20.60e-6


def test_us_stock_finra_taf_default_matches_2026_schedule():
    fee = USStockFeeModel()
    assert fee.finra_taf_per_share == 0.000195
    assert fee.finra_taf_cap == 9.79


# --------------------------------------------------------------------
# Test 2 — SEC §31 applied to sell only
# --------------------------------------------------------------------

def test_us_stock_sec_fee_applied_to_sell_only():
    fee = USStockFeeModel()
    price = 100.0
    size = 1000.0  # $100k notional sell

    buy_commission = fee.calculate_commission(price, size, "buy")
    sell_commission = fee.calculate_commission(price, size, "sell")

    # Buy: broker only = max(1000 * 0.005, 1.0) = $5.00
    assert buy_commission == pytest.approx(5.00)

    # Sell adds SEC §31 ($100k * 20.60e-6 = $2.06) and FINRA TAF
    # (1000 * 0.000195 = $0.195, well under cap).
    expected_sell = 5.00 + 100_000 * 20.60e-6 + 1000 * 0.000195
    assert sell_commission == pytest.approx(expected_sell)
    assert sell_commission > buy_commission


# --------------------------------------------------------------------
# Test 3 — FINRA TAF cap pins at large size
# --------------------------------------------------------------------

def test_us_stock_finra_taf_capped_at_9_79():
    fee = USStockFeeModel()
    # Large sell: 100,000 shares × $0.000195 = $19.50 → capped at $9.79.
    sell = fee.calculate_commission(price=10.0, size=100_000, side="sell")
    broker = max(100_000 * 0.005, 1.0)  # = $500
    sec = 10.0 * 100_000 * 20.60e-6     # = $20.60
    finra_capped = 9.79
    assert sell == pytest.approx(broker + sec + finra_capped)


# --------------------------------------------------------------------
# Test 4 — Small sell uses per-share FINRA TAF (not cap)
# --------------------------------------------------------------------

def test_us_stock_finra_taf_per_share_for_small_sell():
    fee = USStockFeeModel()
    # 100 shares × $0.000195 = $0.0195, well under cap.
    sell = fee.calculate_commission(price=50.0, size=100, side="sell")
    broker = max(100 * 0.005, 1.0)  # = $1.00 (min hit)
    sec = 50.0 * 100 * 20.60e-6     # = $0.103
    finra_per_share = 100 * 0.000195  # = $0.0195
    assert sell == pytest.approx(broker + sec + finra_per_share)


# --------------------------------------------------------------------
# Test 5 — Zero-kwarg opt-out matches v2.8.1 behavior bit-for-bit
# --------------------------------------------------------------------

def test_us_stock_regulatory_fees_opt_out_by_zero_kwargs():
    """Backtests in periods before SEC §31 / FINRA TAF existed (or
    users who want the v2.8.1 numerical behavior for a paper) opt
    out by passing zeros."""
    legacy = USStockFeeModel(sec_fee_rate=0, finra_taf_per_share=0)
    price = 150.0
    size = 200.0
    buy = legacy.calculate_commission(price, size, "buy")
    sell = legacy.calculate_commission(price, size, "sell")
    # With zero regulatory rates, sell == buy == broker-only.
    expected = max(size * 0.005, 1.0)
    assert buy == pytest.approx(expected)
    assert sell == pytest.approx(expected)


# --------------------------------------------------------------------
# Test 6 — min_commission and regulatory fees are independent
# --------------------------------------------------------------------

def test_us_stock_reg_fees_independent_of_broker_min_commission():
    """Tiny sell hits broker min_commission floor; regulatory fees
    still accrue on top — they are statutory, not subject to the
    broker's per-trade minimum."""
    fee = USStockFeeModel()
    # 1 share × $0.005 = $0.005 → broker min $1.00 floor hits.
    sell = fee.calculate_commission(price=100.0, size=1, side="sell")
    broker = 1.0                               # min floor
    sec = 100.0 * 1 * 20.60e-6                # $0.00206
    finra = min(1 * 0.000195, 9.79)           # $0.000195
    assert sell == pytest.approx(broker + sec + finra)
    # And explicitly: sell > broker_min alone — reg fees are NOT
    # absorbed by the broker minimum.
    assert sell > broker
