"""
End-to-end tests for v0.8 features: margin/leverage trading.

Each test covers one complete user scenario through the engine.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from aiphaforge import (
    BacktestEngine,
    BacktestResult,
    Broker,
    EqualWeightAllocator,
    OrderStatus,
    Portfolio,
)
from aiphaforge.capital_allocator import MarginAllocator
from aiphaforge.fees import ZeroFeeModel
from aiphaforge.margin import (
    BorrowingCostModel,
    FundingRateModel,
    MarginCallExitRule,
    MarginConfig,
)

from .conftest import make_ohlcv


class TestLeveragedRoundTrip:
    """Complete leveraged long journey: open → hold with costs → close."""

    def test_2x_leverage_with_borrowing_cost(self):
        """Buy at 2x leverage, hold with daily borrowing cost, sell.
        Verify equity curve, compare with cash-only run."""
        data = make_ohlcv(30, start_price=100)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1   # buy
        signals.iloc[25] = -1  # sell

        # Run with 2x leverage + borrowing cost
        engine_margin = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=10_000,
            margin_config=MarginConfig(
                initial_margin_ratio=0.5,
                borrowing_rate=0.10,  # 10% annual
            ),
            periodic_cost_model=BorrowingCostModel(),
            include_benchmark=False,
        )
        engine_margin.set_signals(signals)
        result_margin = engine_margin.run(data)

        # Run cash-only (IMR=1.0, no cost)
        engine_cash = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=10_000,
            include_benchmark=False,
        )
        engine_cash.set_signals(signals)
        result_cash = engine_cash.run(data)

        # Both should complete
        assert len(result_margin.equity_curve) == 30
        assert len(result_cash.equity_curve) == 30

        # Margin run should have lower final equity due to borrowing cost
        # (both use same data, but margin pays interest)
        assert result_margin.final_capital < result_cash.final_capital

        # Cash-only with explicit IMR=1.0 should be identical to default
        engine_explicit = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=10_000,
            margin_config=MarginConfig(initial_margin_ratio=1.0),
            include_benchmark=False,
        )
        engine_explicit.set_signals(signals)
        result_explicit = engine_explicit.run(data)
        pd.testing.assert_series_equal(
            result_cash.equity_curve, result_explicit.equity_curve, atol=0.01)


class TestMultiAssetMargin:
    """Multi-asset backtest with margin, auto allocator, and costs."""

    def test_multi_asset_margin_full_flow(self):
        """Dict data + margin → auto MarginAllocator → buy multiple assets
        → borrowing costs deducted → sell → per-asset results."""
        data = {
            "AAPL": make_ohlcv(20, start_price=150),
            "TSLA": make_ohlcv(20, start_price=200),
        }
        signals = {
            "AAPL": pd.Series(np.nan, index=data["AAPL"].index, dtype=float),
            "TSLA": pd.Series(np.nan, index=data["TSLA"].index, dtype=float),
        }
        signals["AAPL"].iloc[1] = 1
        signals["TSLA"].iloc[1] = 1
        signals["AAPL"].iloc[15] = -1
        signals["TSLA"].iloc[15] = -1

        # No explicit allocator → should auto-select MarginAllocator
        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            margin_config=MarginConfig(
                initial_margin_ratio=0.5,
                borrowing_rate=0.05,
            ),
            periodic_cost_model=BorrowingCostModel(),
            include_benchmark=False,
        )
        engine.set_signals(signals)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = engine.run(data)
            # Should auto-select MarginAllocator (not EqualWeight)
            auto_warns = [x for x in w if "MarginAllocator" in str(x.message)]
            assert len(auto_warns) >= 1

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == 20
        assert result.per_asset_pnl is not None

        # Without margin → auto EqualWeightAllocator
        engine_no_margin = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine_no_margin.set_signals(signals)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine_no_margin.run(data)
            equal_warns = [
                x for x in w if "EqualWeightAllocator" in str(x.message)]
            assert len(equal_warns) >= 1


class TestMarginCallScenario:
    """Margin call: trigger → liquidation → close orders allowed."""

    def test_margin_call_allows_close_blocks_open(self):
        """During margin call: closing positions allowed, new opens blocked."""
        mc = MarginConfig(initial_margin_ratio=0.5,
                          maintenance_margin_ratio=0.3)

        # Portfolio in margin call state with both long and short
        p = Portfolio(initial_capital=100, margin_config=mc)
        p.positions["LONG"] = type('Pos', (), {
            'market_value': 500, 'notional_value': 500,
            'is_flat': False, 'is_long': True, 'is_short': False,
            'size': 10, 'current_price': 50.0, 'avg_entry_price': 100.0,
            'unrealized_pnl': -500,
        })()
        p.positions["SHORT"] = type('Pos', (), {
            'market_value': 800, 'notional_value': -800,
            'is_flat': False, 'is_long': False, 'is_short': True,
            'size': -8, 'current_price': 100.0, 'avg_entry_price': 50.0,
            'unrealized_pnl': -400,
        })()
        p.cash = 500
        # equity = 500 + 500 + (-800) = 200
        # maintenance = (500 + 800) * 0.3 = 390
        # 200 < 390 → margin call
        assert p.is_margin_call

        broker_long = Broker(fee_model=ZeroFeeModel(), assigned_symbol="LONG")
        broker_long.set_portfolio(p)
        broker_short = Broker(fee_model=ZeroFeeModel(), assigned_symbol="SHORT")
        broker_short.set_portfolio(p)
        broker_new = Broker(fee_model=ZeroFeeModel(), assigned_symbol="NEW")
        broker_new.set_portfolio(p)

        # 1. Sell to close long → ALLOWED
        order_close_long = broker_long.create_market_order(
            "LONG", "sell", 10, "margin_call")
        broker_long.submit_order(order_close_long)
        assert order_close_long.status == OrderStatus.PENDING

        # 2. Buy to close short → ALLOWED
        order_close_short = broker_short.create_market_order(
            "SHORT", "buy", 8, "margin_call")
        broker_short.submit_order(order_close_short)
        assert order_close_short.status == OrderStatus.PENDING

        # 3. Buy new symbol → BLOCKED
        order_new_buy = broker_new.create_market_order("NEW", "buy", 10)
        order_new_buy.metadata['estimated_price'] = 100.0
        broker_new.submit_order(order_new_buy)
        assert order_new_buy.status == OrderStatus.REJECTED

    def test_margin_call_exit_rule_dedup(self):
        """MarginCallExitRule: rejected order not stuck in pending set."""
        mc = MarginConfig(initial_margin_ratio=0.5,
                          maintenance_margin_ratio=0.3)
        p = Portfolio(initial_capital=100, margin_config=mc)
        p.positions["X"] = type('Pos', (), {
            'market_value': 1000, 'notional_value': 1000,
            'is_flat': False, 'is_long': True, 'is_short': False,
            'size': 10, 'current_price': 100.0, 'avg_entry_price': 100.0,
            'unrealized_pnl': -900,
        })()
        p.cash = -900
        assert p.is_margin_call

        # Broker with wrong assigned_symbol → rejects liquidation
        broker = Broker(fee_model=ZeroFeeModel(), assigned_symbol="WRONG")
        broker.set_portfolio(p)

        rule = MarginCallExitRule("all")
        rule.check_portfolio(p, {"X": broker}, ["X"], {"X": 100.0}, None)

        # Rejected → NOT in pending → can retry next bar
        assert "X" not in rule._pending_liquidations


class TestBorrowingCostFairness:
    """Borrowing cost must be fair: same entry = same cost regardless
    of current price movement (for longs)."""

    def test_winners_and_losers_same_cost(self):
        """Two long positions with same entry but different P&L
        must pay identical borrowing interest."""
        mc = MarginConfig(initial_margin_ratio=0.5, borrowing_rate=0.365)
        model = BorrowingCostModel()

        make_pos = lambda mv, entry_p: type('Pos', (), {
            'market_value': mv, 'is_short': False, 'is_long': True,
            'size': 100, 'avg_entry_price': entry_p,
        })()

        winner = make_pos(15_000, 100.0)  # rose from $100 to $150
        loser = make_pos(8_000, 100.0)    # fell from $100 to $80

        cost_w = model.calculate_cost(winner, 150.0, None, mc)
        cost_l = model.calculate_cost(loser, 80.0, None, mc)

        # Both borrowed 100 * $100 * 0.5 = $5k → same daily cost
        expected = 100 * 100.0 * 0.5 * (0.365 / 365)
        assert abs(cost_w - expected) < 0.001
        assert abs(cost_l - expected) < 0.001
        assert abs(cost_w - cost_l) < 0.001


class TestCryptoFunding:
    """Crypto perpetual futures funding rate scenario."""

    def test_funding_rate_applied_per_bar(self):
        """FundingRateModel charges per-bar on notional value."""
        data = make_ohlcv(10, start_price=50000)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=10_000,
            margin_config=MarginConfig(initial_margin_ratio=0.1),
            periodic_cost_model=FundingRateModel(funding_rate_per_bar=0.0001),
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert len(result.equity_curve) == 10
        # Funding costs should reduce equity vs a run without costs
        engine_no_fund = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=10_000,
            margin_config=MarginConfig(initial_margin_ratio=0.1),
            include_benchmark=False,
        )
        engine_no_fund.set_signals(signals)
        result_no_fund = engine_no_fund.run(data)

        assert result.final_capital < result_no_fund.final_capital


class TestLotSize:
    """Lot-size rounding for A-share and per-asset isolation."""

    def test_lot_size_rounds_down(self):
        """A-share 100-lot: position sizer wants 950 shares → rounds to 900."""
        data = make_ohlcv(10, start_price=100)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1

        # With lot_size=100, buy signal at ~$100 with $100k capital
        # FractionSizer(0.95) targets 950 shares → rounds to 900
        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            lot_size=100,
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        # Check that all trades have sizes that are multiples of 100
        for trade in result.trades:
            assert trade.size % 100 == 0 or trade.size == 0

    def test_per_asset_lot_size_isolation(self):
        """A-share gets 100-lot, crypto gets fractional (lot=1)."""
        data = {
            "600519.SH": make_ohlcv(10, start_price=100),
            "BTC": make_ohlcv(10, start_price=50000),
        }
        signals = {
            "600519.SH": pd.Series(np.nan, index=data["600519.SH"].index, dtype=float),
            "BTC": pd.Series(np.nan, index=data["BTC"].index, dtype=float),
        }
        signals["600519.SH"].iloc[1] = 1
        signals["BTC"].iloc[1] = 1
        signals["600519.SH"].iloc[7] = -1
        signals["BTC"].iloc[7] = -1

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=200_000,
            capital_allocator=EqualWeightAllocator(),
            lot_size=1,  # default: fractional
            asset_lot_sizes={"600519.SH": 100},  # A-share: 100 lots
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert isinstance(result, BacktestResult)
        # Verify isolation: A-share trades in multiples of 100
        for trade in result.trades:
            if trade.symbol == "600519.SH":
                assert trade.size % 100 == 0, (
                    f"A-share trade size {trade.size} not multiple of 100")
        # BTC should have trades (fractional allowed, no rounding)
        btc_trades = [t for t in result.trades if t.symbol == "BTC"]
        ashare_trades = [t for t in result.trades if t.symbol == "600519.SH"]
        assert len(btc_trades) > 0, "BTC should have trades"
        assert len(ashare_trades) > 0, "A-share should have trades"


class TestPositionLimit:
    """Per-asset position limit as fraction of equity."""

    def test_position_limit_caps_exposure(self):
        """max_position_pct=0.5 → each asset limited to 50% of equity."""
        data = make_ohlcv(20, start_price=100)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[15] = -1

        # Without limit: FractionSizer(0.95) uses ~95% of equity
        engine_no_limit = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine_no_limit.set_signals(signals)
        result_no_limit = engine_no_limit.run(data)

        # With 50% limit
        engine_limited = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            max_position_pct=0.5,
            include_benchmark=False,
        )
        engine_limited.set_signals(signals)
        result_limited = engine_limited.run(data)

        # Limited run should have smaller trades
        if result_no_limit.trades and result_limited.trades:
            no_limit_size = result_no_limit.trades[0].size
            limited_size = result_limited.trades[0].size
            assert limited_size < no_limit_size

    def test_per_asset_position_limit_isolation(self):
        """Different limits per asset: AAPL=10%, TSLA=default(100%)."""
        data = {
            "AAPL": make_ohlcv(15, start_price=150),
            "TSLA": make_ohlcv(15, start_price=200),
        }
        signals = {
            "AAPL": pd.Series(np.nan, index=data["AAPL"].index, dtype=float),
            "TSLA": pd.Series(np.nan, index=data["TSLA"].index, dtype=float),
        }
        signals["AAPL"].iloc[1] = 1
        signals["TSLA"].iloc[1] = 1
        signals["AAPL"].iloc[10] = -1
        signals["TSLA"].iloc[10] = -1

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=200_000,
            capital_allocator=EqualWeightAllocator(),
            asset_max_position_pcts={"AAPL": 0.1},  # AAPL: 10% cap
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert isinstance(result, BacktestResult)
        # AAPL should have smaller position than TSLA
        aapl_trades = [t for t in result.trades if t.symbol == "AAPL"]
        tsla_trades = [t for t in result.trades if t.symbol == "TSLA"]
        if aapl_trades and tsla_trades:
            assert aapl_trades[0].size < tsla_trades[0].size
