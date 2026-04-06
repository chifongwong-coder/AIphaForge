"""
Tests for Portfolio and Position management.
"""

import pandas as pd
import pytest

from aiphaforge.portfolio import Portfolio, Position

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_days: int = 0) -> pd.Timestamp:
    """Return a deterministic timestamp offset by *offset_days*."""
    return pd.Timestamp("2024-01-01") + pd.Timedelta(days=offset_days)


# ---------------------------------------------------------------------------
# Tests — Core Portfolio Behavior
# ---------------------------------------------------------------------------

class TestPortfolioBasics:
    """Initial state, reset, and position averaging."""

    def test_portfolio_initial_state(self):
        """Portfolio starts with correct cash, no positions, no trades."""
        pf = Portfolio(initial_capital=50_000)
        assert pf.cash == 50_000
        assert pf.total_equity == 50_000
        assert pf.position_value == 0.0
        assert pf.unrealized_pnl == 0.0
        assert pf.realized_pnl == 0.0
        assert pf.total_return == 0.0
        assert len(pf.trade_history) == 0
        assert len(pf.equity_history) == 0
        assert not pf.has_position("ANY")

    def test_portfolio_reset_clears_state(self):
        """reset() returns portfolio to initial state."""
        pf = Portfolio(initial_capital=100_000)

        # Do some trading
        pf.update_position("X", size_change=50, price=100.0, timestamp=_ts(0))
        pf.update_position("X", size_change=-50, price=110.0, timestamp=_ts(3))
        assert len(pf.trade_history) == 1

        pf.reset()

        assert pf.cash == 100_000
        assert pf.total_equity == 100_000
        assert len(pf.positions) == 0
        assert len(pf.trade_history) == 0
        assert len(pf.equity_history) == 0
        assert pf._trade_counter == 0
        assert len(pf._pending_entries) == 0

    def test_position_add_same_direction_averages_price(self):
        """Adding to a position in the same direction averages the entry price."""
        pos = Position(symbol="X")

        pos.add(100, 100.0, _ts(0))
        assert pos.avg_entry_price == 100.0

        pos.add(100, 120.0, _ts(1))
        assert pos.size == 200
        expected_avg = (100 * 100 + 100 * 120) / 200  # 110
        assert pos.avg_entry_price == pytest.approx(expected_avg, abs=1e-6)

    def test_size_change_zero_returns_none(self):
        """update_position(size_change=0) should return None immediately."""
        pf = Portfolio(initial_capital=100_000)
        result = pf.update_position("X", size_change=0, price=100.0, timestamp=_ts(0))
        assert result is None
        assert not pf.has_position("X")


# ---------------------------------------------------------------------------
# Tests — Round Trip Trades
# ---------------------------------------------------------------------------

class TestRoundTripTrades:
    """Full long and short round-trip PnL tests."""

    def test_long_round_trip_pnl(self):
        """Buy 100 shares @ 100, sell @ 110 with ZeroFeeModel -> PnL = 1000."""
        pf = Portfolio(initial_capital=100_000)

        pf.update_position("X", size_change=100, price=100.0, timestamp=_ts(0))
        assert pf.has_position("X")

        trade = pf.update_position("X", size_change=-100, price=110.0, timestamp=_ts(5))

        assert trade is not None
        assert trade.pnl == 1000.0
        assert trade.direction == 1
        assert not pf.has_position("X")

    def test_short_round_trip_pnl(self):
        """Short 100 shares @ 100, cover @ 90 -> PnL = 1000."""
        pf = Portfolio(initial_capital=100_000)

        pf.update_position("X", size_change=-100, price=100.0, timestamp=_ts(0))
        pos = pf.get_position("X")
        assert pos is not None
        assert pos.is_short

        trade = pf.update_position("X", size_change=100, price=90.0, timestamp=_ts(5))

        assert trade is not None
        assert trade.pnl == 1000.0
        assert trade.direction == -1

    def test_short_position_equity(self):
        """Regression: total_equity should use signed notional_value for shorts.

        When short 100 @ 100 and price rises to 110, equity must decrease.
        """
        pf = Portfolio(initial_capital=100_000)

        pf.update_position("X", size_change=-100, price=100.0, timestamp=_ts(0))
        pf.positions["X"].update_price(110.0)

        assert pf.positions["X"].unrealized_pnl == -1000.0
        assert pf.total_equity < 100_000


# ---------------------------------------------------------------------------
# Tests — Fees
# ---------------------------------------------------------------------------

class TestFees:
    """Fee deduction, allocation, and proportional splitting."""

    def test_fees_deducted_correctly(self):
        """SimpleFeeModel: Trade.pnl includes both entry and exit fees."""
        pf = Portfolio(initial_capital=100_000)

        pf.update_position(
            "X", size_change=100, price=100.0, timestamp=_ts(0),
            commission=10.0, slippage=5.0,
        )

        trade = pf.update_position(
            "X", size_change=-100, price=110.0, timestamp=_ts(5),
            commission=11.0, slippage=5.5,
        )

        assert trade is not None
        gross = 100 * (110 - 100)  # 1000
        total_fees = 10.0 + 5.0 + 11.0 + 5.5  # 31.5
        assert trade.pnl == pytest.approx(gross - total_fees, abs=1e-6)
        assert trade.commission == pytest.approx(10.0 + 11.0, abs=1e-6)
        assert trade.slippage_cost == pytest.approx(5.0 + 5.5, abs=1e-6)

    def test_fee_allocation_proportional(self):
        """Proportional: partial close gets proportional share of entry fees."""
        pf = Portfolio(initial_capital=100_000, fee_allocation="proportional")

        # Entry: buy 100 @ 100, entry commission=10
        pf.update_position(
            "X", size_change=100, price=100.0, timestamp=_ts(0),
            commission=10.0, slippage=0.0,
        )

        # Partial close: sell 50 @ 110 (no exit fees to isolate entry allocation)
        trade = pf.update_position(
            "X", size_change=-50, price=110.0, timestamp=_ts(5),
            commission=0.0, slippage=0.0,
        )

        assert trade is not None
        assert trade.size == 50
        # 50% of entry commission (10) = 5
        assert trade.commission == pytest.approx(5.0, abs=1e-6)

    def test_fee_allocation_first_close(self):
        """first_close: first partial close gets all entry commission."""
        pf = Portfolio(initial_capital=100_000, fee_allocation="first_close")

        # Entry: buy 100 @ 100, entry commission=10
        pf.update_position(
            "X", size_change=100, price=100.0, timestamp=_ts(0),
            commission=10.0, slippage=0.0,
        )

        # First partial close: sell 50
        trade1 = pf.update_position(
            "X", size_change=-50, price=110.0, timestamp=_ts(5),
            commission=0.0, slippage=0.0,
        )

        assert trade1 is not None
        # First close gets all entry commission
        assert trade1.commission == pytest.approx(10.0, abs=1e-6)

        # Second partial close: sell remaining 50
        trade2 = pf.update_position(
            "X", size_change=-50, price=115.0, timestamp=_ts(10),
            commission=0.0, slippage=0.0,
        )

        assert trade2 is not None
        # Entry commission was zeroed after first close, so 0 here
        assert trade2.commission == pytest.approx(0.0, abs=1e-6)

    def test_reversal_fee_split(self):
        """Reversal: exit fees are split proportionally between close and new entry."""
        pf = Portfolio(initial_capital=100_000, fee_allocation="proportional")

        # Entry: buy 100 @ 100, commission=10
        pf.update_position(
            "X", size_change=100, price=100.0, timestamp=_ts(0),
            commission=10.0, slippage=0.0,
        )

        # Reversal: sell 200 @ 110, commission=20
        # close_share = 100/200 = 50%, new_entry_share = 50%
        trade = pf.update_position(
            "X", size_change=-200, price=110.0, timestamp=_ts(5),
            commission=20.0, slippage=0.0,
        )

        assert trade is not None
        assert trade.size == 100  # closed the original 100

        # Close gets: 50% of exit commission (10) + 100% of entry commission (10) = 20
        assert trade.commission == pytest.approx(20.0, abs=1e-6)

        # New entry should have 50% of exit commission = 10
        entry_info = pf._pending_entries["X"]
        assert entry_info['entry_commission'] == pytest.approx(10.0, abs=1e-6)
        assert entry_info['direction'] == -1
        assert entry_info['size'] == 100


# ---------------------------------------------------------------------------
# Tests — Partial Close, Reversal, Add-to-Position
# ---------------------------------------------------------------------------

class TestPositionManagement:
    """Partial close, reversal, and add-to-position scenarios."""

    def test_partial_close_generates_trade(self):
        """Buy 100@100, sell 50@110 -> Trade with size=50, correct PnL."""
        pf = Portfolio(initial_capital=100_000)

        pf.update_position("X", size_change=100, price=100.0, timestamp=_ts(0))

        trade = pf.update_position("X", size_change=-50, price=110.0, timestamp=_ts(5))

        assert trade is not None
        assert trade.size == 50
        assert trade.pnl == pytest.approx(50 * (110 - 100), abs=1e-6)
        assert trade.direction == 1

        # Remaining position should be 50
        pos = pf.get_position("X")
        assert pos is not None
        assert pos.size == 50

    def test_reversal_generates_trade(self):
        """Buy 100@100, sell 200@110 -> close Trade (size=100, PnL=1000), position=-100."""
        pf = Portfolio(initial_capital=100_000)

        pf.update_position("X", size_change=100, price=100.0, timestamp=_ts(0))

        trade = pf.update_position("X", size_change=-200, price=110.0, timestamp=_ts(5))

        assert trade is not None
        assert trade.size == 100
        assert trade.pnl == pytest.approx(100 * (110 - 100), abs=1e-6)
        assert trade.direction == 1

        # Position should now be short 100
        pos = pf.get_position("X")
        assert pos is not None
        assert pos.size == -100

    def test_add_to_position_syncs_entry(self):
        """Buy 100@100, buy 100@120 -> pending entry has avg price ~110, size=200."""
        pf = Portfolio(initial_capital=100_000)

        pf.update_position(
            "X", size_change=100, price=100.0, timestamp=_ts(0),
            commission=5.0, slippage=0.0,
        )
        pf.update_position(
            "X", size_change=100, price=120.0, timestamp=_ts(1),
            commission=6.0, slippage=0.0,
        )

        entry_info = pf._pending_entries["X"]
        expected_avg = (100 * 100 + 100 * 120) / 200  # 110
        assert entry_info['entry_price'] == pytest.approx(expected_avg, abs=1e-6)
        assert entry_info['size'] == 200
        assert entry_info['entry_commission'] == pytest.approx(11.0, abs=1e-6)
