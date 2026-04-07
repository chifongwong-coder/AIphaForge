"""
Tests for v0.4 modules: exit rules, position sizing, fee model improvements,
OHLCV validation, and trade costs.
"""


import numpy as np
import pandas as pd
import pytest

from aiphaforge.broker import Broker
from aiphaforge.costs import DefaultTradeCost
from aiphaforge.exit_rules import PercentageStopLoss
from aiphaforge.fees import ChinaAShareFeeModel, SimpleFeeModel
from aiphaforge.portfolio import Portfolio, Position
from aiphaforge.position_sizing import FixedSizer, FractionSizer
from aiphaforge.utils import validate_ohlcv

from .conftest import make_ohlcv

# ---------- Exit Rules ----------


def test_percentage_stop_loss_event_driven():
    """PercentageStopLoss submits a close order when loss exceeds threshold."""
    stop = PercentageStopLoss(0.05)

    # Build a portfolio with a long position that has lost > 5%
    portfolio = Portfolio(initial_capital=100_000)
    ts_entry = pd.Timestamp("2024-01-02")
    ts_now = pd.Timestamp("2024-01-10")

    # Manually create a position that is down > 5%
    pos = Position(
        symbol="TEST",
        size=100,
        avg_entry_price=100.0,
        current_price=94.0,  # -6% unrealized loss
        open_time=ts_entry,
    )
    portfolio.positions["TEST"] = pos

    # Create a broker with a mock so we can inspect submitted orders
    broker = Broker(check_buying_power=False)
    broker.set_portfolio(portfolio)

    bar = pd.Series(
        {"open": 94.0, "high": 95.0, "low": 93.5, "close": 94.0, "volume": 500_000}
    )

    stop.check_event_driven(portfolio, broker, "TEST", bar, ts_now)

    # There should be exactly one pending sell order for 100 shares
    pending = broker.get_pending_orders("TEST")
    assert len(pending) == 1
    order = pending[0]
    assert order.side.value == "sell"
    assert order.size == 100


# ---------- Position Sizing ----------


def test_fraction_sizer_calculation():
    """FractionSizer(0.5) allocates half of equity."""
    sizer = FractionSizer(0.5)
    # equity=100000, price=100, direction=1, max_position_size=1.0
    result = sizer.calculate(equity=100_000, price=100, direction=1, max_position_size=1.0)
    assert result == 500


def test_fixed_sizer_respects_max():
    """FixedSizer(1000) is capped by max_position_size fraction of equity."""
    sizer = FixedSizer(1000)
    # max_size = 100000 * 0.5 / 100 = 500
    result = sizer.calculate(equity=100_000, price=100, direction=1, max_position_size=0.5)
    assert result == 500


# ---------- Fee Model ----------


def test_estimate_commission_rate_average():
    """ChinaAShareFeeModel average rate includes stamp duty component.

    Buy side: commission + transfer_fee (no stamp duty).
    Sell side: commission + transfer_fee + stamp_duty.
    Average should be higher than buy-only rate because of stamp duty.
    """
    fee = ChinaAShareFeeModel()
    avg_rate = fee.estimate_commission_rate(side="average")
    buy_rate = fee.estimate_commission_rate(side="buy")
    sell_rate = fee.estimate_commission_rate(side="sell")

    # Average should be the midpoint of buy and sell
    assert avg_rate == pytest.approx((buy_rate + sell_rate) / 2, rel=1e-9)
    # Average should be strictly between buy and sell (stamp duty makes sell > buy)
    assert avg_rate > buy_rate
    assert avg_rate < sell_rate


# ---------- OHLCV Validation ----------


def test_validate_ohlcv_strict_high_lt_low():
    """Data with high < low in strict mode raises ValueError."""
    df = make_ohlcv(10)
    # Force high < low on one row
    df.iloc[3, df.columns.get_loc("high")] = df.iloc[3]["low"] - 1.0

    with pytest.raises(ValueError, match="high < low"):
        validate_ohlcv(df, validation_level="strict")


def test_validate_ohlcv_warn_nan():
    """Data with NaN in price columns emits a warning."""
    df = make_ohlcv(10)
    df.iloc[2, df.columns.get_loc("close")] = np.nan

    with pytest.warns(UserWarning, match="NaN"):
        validate_ohlcv(df, validation_level="warn")


def test_validate_ohlcv_unsorted_timestamps():
    """Unsorted (non-monotonic) timestamps emit a warning."""
    df = make_ohlcv(10)
    # Reverse the index to make it non-monotonic
    df = df.iloc[::-1]
    # The index is now reversed but still a DatetimeIndex

    with pytest.warns(UserWarning, match="not monotonically increasing"):
        validate_ohlcv(df, validation_level="warn")


# ---------- Trade Costs ----------


def test_default_trade_cost_uses_diff():
    """DefaultTradeCost.apply_vectorized uses positions.diff().abs() for costs."""
    data = make_ohlcv(10)
    # Positions: flat, then long 1.0, then flat
    positions = pd.Series(0.0, index=data.index)
    positions.iloc[2:7] = 1.0

    returns = pd.Series(0.01, index=data.index)  # uniform 1% return
    fee_model = SimpleFeeModel(commission_rate=0.001, slippage_pct=0.001)
    initial_capital = 100_000.0

    cost_model = DefaultTradeCost()
    net_returns = cost_model.apply_vectorized(
        returns, positions, data, fee_model, initial_capital
    )

    # Bars with no position change should have full 1% return
    # (bars 0, 1, 3, 4, 5, 6, 8, 9 -- indices where diff is 0)
    no_change_mask = positions.diff().abs().fillna(0) == 0
    assert (net_returns[no_change_mask] == 0.01).all()

    # Bars with position change (entry at index 2, exit at index 7) should
    # have returns reduced by costs
    change_mask = positions.diff().abs().fillna(0) > 0
    assert (net_returns[change_mask] < 0.01).all()
