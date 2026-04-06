"""
Tests for Broker order execution simulation.
"""

import pandas as pd
import pytest

from aiphaforge.broker import Broker, FillModel
from aiphaforge.fees import ZeroFeeModel
from aiphaforge.orders import Order, OrderSide, OrderType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset: int = 0) -> pd.Timestamp:
    return pd.Timestamp("2024-01-01") + pd.Timedelta(days=offset)


def _bar(open_: float, high: float, low: float, close: float, volume: float = 500_000) -> pd.Series:
    """Create a bar pd.Series."""
    return pd.Series({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOrderFills:
    """Tests for market, limit, and stop order fill behavior."""

    def test_market_order_fills_at_open(self):
        """NEXT_BAR_OPEN fill model: market order fills at the bar's open."""
        broker = Broker(
            fee_model=ZeroFeeModel(),
            fill_model=FillModel.NEXT_BAR_OPEN,
        )
        order = broker.create_market_order("X", "buy", 100, "test", _ts(0))
        broker.submit_order(order, _ts(0))

        bar = _bar(open_=105.0, high=110.0, low=100.0, close=108.0)
        filled = broker.process_bar(bar, _ts(1), "X")

        assert len(filled) == 1
        assert filled[0].is_filled
        assert filled[0].filled_price == pytest.approx(105.0, abs=1e-6)

    def test_limit_order_not_filled_out_of_range(self):
        """A buy limit below the bar's low should not fill."""
        broker = Broker(fee_model=ZeroFeeModel())
        order = broker.create_limit_order("X", "buy", 100, price=90.0, reason="test", timestamp=_ts(0))
        broker.submit_order(order, _ts(0))

        # Bar range is 100-110, so limit at 90 won't trigger
        bar = _bar(open_=105.0, high=110.0, low=100.0, close=108.0)
        filled = broker.process_bar(bar, _ts(1), "X")

        assert len(filled) == 0
        assert order.is_pending

    def test_stop_order_triggers_when_price_crosses(self):
        """Sell stop at 95 should trigger when bar low <= 95."""
        broker = Broker(
            fee_model=ZeroFeeModel(),
            fill_model=FillModel.NEXT_BAR_OPEN,
        )
        order = broker.create_stop_order("X", "sell", 100, stop_price=95.0, reason="test", timestamp=_ts(0))
        broker.submit_order(order, _ts(0))

        # Bar drops below 95
        bar = _bar(open_=100.0, high=102.0, low=93.0, close=94.0)
        filled = broker.process_bar(bar, _ts(1), "X")

        assert len(filled) == 1
        assert filled[0].is_filled


class TestStopLimitConversion:

    def test_stop_limit_converts_to_limit_after_trigger(self):
        """Regression: stop-limit should convert to limit if stop triggers but limit not fillable."""
        broker = Broker(fee_model=ZeroFeeModel())

        # Create stop-limit order: stop at 95, limit at 94
        order = Order(
            symbol="X",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LIMIT,
            size=100,
            stop_price=95.0,
            price=94.0,
            created_time=_ts(0),
        )
        broker.submit_order(order, _ts(0))

        # Bar 1: low hits 95 (stop triggers) but high doesn't reach 94 for a sell limit
        # For a sell limit, high must be >= limit price. Here high=96 >= 94 so it fills.
        # Instead, make a scenario where stop triggers but limit does NOT fill:
        # For a SELL limit, we need high < limit_price (i.e., the bar never rises to 94).
        # But that contradicts low <= 95 (stop trigger). We need a bar where low <= 95
        # and high < 94 -- impossible since low <= high. So let's use a BUY stop-limit:
        # Buy stop at 105, buy limit at 103. Stop triggers when high >= 105.
        # Buy limit fills when low <= 103. If low > 103 then limit doesn't fill.
        order2 = Order(
            symbol="Y",
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LIMIT,
            size=50,
            stop_price=105.0,
            price=103.0,
            created_time=_ts(0),
        )
        broker.submit_order(order2, _ts(0))

        # Bar where high >= 105 (stop triggers) but low > 103 (limit not fillable)
        bar = _bar(open_=104.0, high=106.0, low=104.0, close=105.5)
        filled = broker.process_bar(bar, _ts(1), "Y")

        # Should not fill, but order should now be a LIMIT order
        assert len(filled) == 0
        assert order2.order_type == OrderType.LIMIT

        # Next bar: low drops to 103, limit should fill
        bar2 = _bar(open_=105.0, high=105.5, low=102.0, close=104.0)
        filled2 = broker.process_bar(bar2, _ts(2), "Y")

        assert len(filled2) == 1
        assert filled2[0].is_filled
        assert filled2[0].filled_price == pytest.approx(103.0, abs=1e-6)
