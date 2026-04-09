"""
End-to-end tests for v0.6 features: PARTIALLY_EXPIRED portfolio fix,
same-bar IOC/FOK processing, SymbolRoutingLatencyHook.

Each test covers one distinct user path through the engine.
"""

import numpy as np
import pandas as pd
import pytest

from aiphaforge import (
    BacktestEngine,
    BacktestHook,
    BacktestResult,
    HookContext,
    Order,
    OrderStatus,
    SymbolRoutingLatencyHook,
)
from aiphaforge.fees import ZeroFeeModel

from .conftest import make_ohlcv

# ---------------------------------------------------------------------------
# Helper hooks
# ---------------------------------------------------------------------------


class OrderSubmittingHook(BacktestHook):
    """Hook that submits orders on specified bars for testing."""

    def __init__(self, submit_on_bars=None, order_type="market",
                 side="buy", size=100, **order_kwargs):
        self.submit_on_bars = submit_on_bars or [0]
        self.order_type = order_type
        self.side = side
        self.size = size
        self.order_kwargs = order_kwargs
        self.submitted_orders: list[Order] = []

    def on_pre_signal(self, ctx: HookContext) -> None:
        if ctx.bar_index in self.submit_on_bars:
            tif = self.order_kwargs.get("time_in_force", "GTC")
            if self.order_type == "market":
                order = ctx.broker.create_market_order(
                    ctx.symbol, self.side, self.size, "test",
                    ctx.timestamp, time_in_force=tif,
                )
            elif self.order_type == "limit":
                limit_price = self.order_kwargs.get("limit_price", 50.0)
                order = ctx.broker.create_limit_order(
                    ctx.symbol, self.side, self.size, limit_price,
                    "test", ctx.timestamp, time_in_force=tif,
                )
            else:
                raise ValueError(f"Unknown order_type: {self.order_type}")
            ctx.broker.submit_order(order, ctx.timestamp)
            self.submitted_orders.append(order)


class EquityRecordingHook(BacktestHook):
    """Records position size per bar for verification."""

    def __init__(self):
        self.position_by_bar: dict[int, float] = {}

    def on_bar(self, ctx: HookContext) -> None:
        pos = ctx.portfolio.get_position_size(ctx.symbol)
        self.position_by_bar[ctx.bar_index] = pos


def _run_event_driven(data, hooks, **kwargs) -> BacktestResult:
    signals = pd.Series(np.nan, index=data.index, dtype=float)
    engine = BacktestEngine(
        fee_model=ZeroFeeModel(), mode="event_driven",
        initial_capital=kwargs.pop("initial_capital", 100_000),
        hooks=hooks,
        include_benchmark=kwargs.pop("include_benchmark", False),
        **kwargs,
    )
    engine.set_signals(signals)
    return engine.run(data)


# ===========================================================================
# Path 1: IOC partial fill updates portfolio (v0.6 bug fix)
# ===========================================================================


class TestPartialFillPortfolio:

    def test_ioc_partial_fill_updates_portfolio(self):
        """Path: IOC + partial_fills + low volume → PARTIALLY_EXPIRED,
        portfolio cash/position reflect the partial fill."""
        from aiphaforge.broker import Broker
        from aiphaforge.config import BacktestConfig
        from aiphaforge.hooks import HookContext
        from aiphaforge.portfolio import Portfolio

        n = 10
        dates = pd.bdate_range("2024-01-01", periods=n, freq="B")
        data = pd.DataFrame({
            "open": [100.0] * n, "high": [110.0] * n,
            "low": [90.0] * n, "close": [105.0] * n,
            "volume": [100.0] * n,
        }, index=dates)

        hook = OrderSubmittingHook(
            submit_on_bars=[0], size=1000, time_in_force="IOC",
        )
        config = BacktestConfig(
            initial_capital=1_000_000, fee_model=ZeroFeeModel(),
            mode="event_driven", hooks=[hook],
        )
        portfolio = Portfolio(initial_capital=1_000_000)
        broker = Broker(
            fee_model=ZeroFeeModel(), partial_fills=True,
            volume_limit_pct=0.1, immediate_fill_price="close",
        )
        broker.set_portfolio(portfolio)
        for h in config.hooks:
            h.on_backtest_start(data, "default", config=config)

        for i, (ts, bar) in enumerate(data.iterrows()):
            portfolio.update_prices({"default": bar["close"]}, ts, record=False)
            for o in broker.process_bar(bar, ts, "default"):
                portfolio.update_from_order(o, ts)
            ctx = HookContext(bar_index=i, timestamp=ts, bar_data=bar,
                              data=data.iloc[:i+1], portfolio=portfolio,
                              symbol="default", broker=broker)
            for h in config.hooks:
                h.on_pre_signal(ctx)
            for o in broker.process_immediate_orders(bar, ts, "default"):
                portfolio.update_from_order(o, ts)
            portfolio._record_equity(ts)

        order = hook.submitted_orders[0]
        assert order.status == OrderStatus.PARTIALLY_EXPIRED
        assert order.filled_size > 0
        assert portfolio.cash < 1_000_000


# ===========================================================================
# Path 2: Same-bar IOC fill via hook (with fill price check)
# ===========================================================================


class TestSameBarIOCFill:

    def test_same_bar_ioc_fills_at_close(self):
        """Path: hook submits IOC → fills on same bar → fill price = close."""
        n = 10
        dates = pd.bdate_range("2024-01-01", periods=n, freq="B")
        data = pd.DataFrame({
            "open": [100.0] * n, "high": [110.0] * n,
            "low": [90.0] * n, "close": [105.0] * n,
            "volume": [1_000_000.0] * n,
        }, index=dates)

        hook = OrderSubmittingHook(submit_on_bars=[0], time_in_force="IOC")
        equity = EquityRecordingHook()
        _run_event_driven(data, [hook, equity])

        order = hook.submitted_orders[0]
        assert order.is_filled
        assert order.filled_price == pytest.approx(105.0)
        assert equity.position_by_bar[0] == 100  # same-bar position


# ===========================================================================
# Path 3: SymbolRoutingLatencyHook (override + fallback in one test)
# ===========================================================================


class TestSymbolRoutingLatencyHook:

    def test_symbol_routing_override_and_fallback(self):
        """Path: override symbol gets short latency, default gets long.
        Since single-symbol backtest uses 'default', test both paths
        by running two backtests with different symbols in overrides."""
        data = make_ohlcv(20)

        # Run 1: "default" is in overrides → 1-bar latency
        inner1 = OrderSubmittingHook(submit_on_bars=[0])
        hook1 = SymbolRoutingLatencyHook(
            inner_hook=inner1,
            default_latency_model="fixed",
            default_latency_params={"bars": 10},
            symbol_overrides={"default": ("fixed", {"bars": 1})},
        )
        equity1 = EquityRecordingHook()
        _run_event_driven(data, [hook1, equity1])
        # 1-bar latency: position visible by bar 1
        assert inner1.submitted_orders[0].is_filled
        assert equity1.position_by_bar.get(1, 0) > 0

        # Run 2: "default" NOT in overrides → falls back to 10-bar default
        inner2 = OrderSubmittingHook(submit_on_bars=[0])
        hook2 = SymbolRoutingLatencyHook(
            inner_hook=inner2,
            default_latency_model="fixed",
            default_latency_params={"bars": 10},
            symbol_overrides={"AAPL": ("fixed", {"bars": 1})},
        )
        equity2 = EquityRecordingHook()
        _run_event_driven(data, [hook2, equity2])
        # 10-bar latency: no position at bars 0-8
        for i in range(9):
            assert equity2.position_by_bar.get(i, 0) == 0


# ===========================================================================
# Path 4: GTC regression (second pass doesn't affect GTC-only)
# ===========================================================================


class TestGTCRegression:

    def test_gtc_only_unaffected(self):
        """Path: GTC-only backtest → same behavior as pre-v0.6."""
        data = make_ohlcv(30)
        buy = OrderSubmittingHook(submit_on_bars=[0], side="buy", time_in_force="GTC")
        sell = OrderSubmittingHook(submit_on_bars=[15], side="sell", time_in_force="GTC")
        result = _run_event_driven(data, [buy, sell])

        assert buy.submitted_orders[0].is_filled
        assert sell.submitted_orders[0].is_filled
        assert len(result.trades) == 1
