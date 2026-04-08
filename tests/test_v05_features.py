"""
End-to-end tests for v0.5 features: TIF orders, LatencyHook, custom benchmarks.

Each test covers one distinct user path through the engine.
"""

from typing import Optional

import numpy as np
import pandas as pd

from aiphaforge import (
    BacktestEngine,
    BacktestHook,
    BacktestResult,
    HookContext,
    LatencyHook,
    Order,
    OrderStatus,
    SimpleLatencyHook,
    backtest,
)
from aiphaforge.fees import ZeroFeeModel

from .conftest import make_ohlcv

# ---------------------------------------------------------------------------
# Helper hooks
# ---------------------------------------------------------------------------


class OrderSubmittingHook(BacktestHook):
    """Hook that submits orders on specified bars for testing."""

    def __init__(
        self,
        submit_on_bars=None,
        order_type="market",
        side="buy",
        size=100,
        **order_kwargs,
    ):
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


class OrderCollectorHook(BacktestHook):
    """Hook that records filled orders after each bar."""

    def __init__(self):
        self.fills_by_bar: dict[int, list] = {}

    def on_bar(self, ctx: HookContext) -> None:
        filled = ctx.broker.get_filled_orders(ctx.symbol)
        if filled:
            self.fills_by_bar[ctx.bar_index] = list(filled)


def _run_event_driven(data, hooks, **kwargs) -> BacktestResult:
    """Run an event-driven backtest with zero signals and given hooks."""
    signals = pd.Series(0, index=data.index, dtype=float)
    engine = BacktestEngine(
        fee_model=ZeroFeeModel(), mode="event_driven",
        initial_capital=100_000, hooks=hooks,
        include_benchmark=kwargs.pop("include_benchmark", False),
        **kwargs,
    )
    engine.set_signals(signals)
    return engine.run(data)


# ===========================================================================
# TIF — one test per distinct path
# ===========================================================================


class TestTIFEndToEnd:

    def test_gtc_round_trip(self):
        """Path: GTC buy → fills next bar → sell → fills → trades in result."""
        data = make_ohlcv(10)
        buy = OrderSubmittingHook(submit_on_bars=[0], side="buy", time_in_force="GTC")
        sell = OrderSubmittingHook(submit_on_bars=[5], side="sell", time_in_force="GTC")
        result = _run_event_driven(data, [buy, sell])

        assert buy.submitted_orders[0].status == OrderStatus.FILLED
        assert sell.submitted_orders[0].status == OrderStatus.FILLED
        assert len(result.trades) >= 1

    def test_ioc_expires_unfillable(self):
        """Path: IOC limit at unreachable price → EXPIRED, zero trades."""
        data = make_ohlcv(10, start_price=100)
        hook = OrderSubmittingHook(
            submit_on_bars=[0], order_type="limit",
            limit_price=1.0, time_in_force="IOC",
        )
        result = _run_event_driven(data, [hook])

        assert hook.submitted_orders[0].status == OrderStatus.EXPIRED
        assert hook.submitted_orders[0].metadata["expiry_reason"] == "ioc_timeout"
        assert len(result.trades) == 0

    def test_fok_rejects_oversized(self):
        """Path: FOK larger than volume → EXPIRED, equity unchanged."""
        data = make_ohlcv(10)
        hook = OrderSubmittingHook(
            submit_on_bars=[0], size=10_000_000, time_in_force="FOK",
        )
        result = _run_event_driven(data, [hook])

        assert hook.submitted_orders[0].status == OrderStatus.EXPIRED
        assert hook.submitted_orders[0].metadata["expiry_reason"] == "fok_volume"
        assert abs(result.equity_curve.iloc[-1] - 100_000) < 1.0

    def test_day_expires_cross_session(self):
        """Path: DAY limit across calendar days → EXPIRED."""
        data = make_ohlcv(20, start_date="2024-01-01")
        hook = OrderSubmittingHook(
            submit_on_bars=[0], order_type="limit",
            limit_price=1.0, time_in_force="DAY",
        )
        _run_event_driven(data, [hook])

        assert hook.submitted_orders[0].status == OrderStatus.EXPIRED
        assert hook.submitted_orders[0].metadata["expiry_reason"] == "day_session_end"


# ===========================================================================
# LatencyHook — one test per distinct path
# ===========================================================================


class TestLatencyHookEndToEnd:

    def test_latency_delays_fill(self):
        """Path: LatencyHook wraps hook → order delayed N bars → fills."""
        data = make_ohlcv(10)
        inner = OrderSubmittingHook(submit_on_bars=[0])
        hook = LatencyHook(inner_hook=inner, latency_model="fixed",
                           latency_params={"bars": 3})
        collector = OrderCollectorHook()
        result = _run_event_driven(data, [hook, collector])

        # No fills on bars 0-1, fills by bar 3+
        for i in range(2):
            assert len(collector.fills_by_bar.get(i, [])) == 0
        all_fills = sum(
            (collector.fills_by_bar.get(i, []) for i in range(3, 10)), []
        )
        assert len(all_fills) >= 1
        assert len(result.equity_curve) == 10

    def test_simple_latency_hook_subclass(self):
        """Path: SimpleLatencyHook subclass → make_decision → delay → fill."""
        data = make_ohlcv(10)

        class MyAgent(SimpleLatencyHook):
            def __init__(self):
                super().__init__(latency_model="fixed",
                                 latency_params={"bars": 2})
                self.decisions = 0

            def make_decision(self, ctx: HookContext) -> Optional[Order]:
                if ctx.bar_index == 0:
                    self.decisions += 1
                    return ctx.broker.create_market_order(
                        ctx.symbol, "buy", 100, "agent", ctx.timestamp)
                return None

        agent = MyAgent()
        result = _run_event_driven(data, [agent])
        assert agent.decisions == 1
        assert len(result.equity_curve) == 10


# ===========================================================================
# Benchmark — one test per distinct path
# ===========================================================================


class TestBenchmarkEndToEnd:

    def test_default_buy_and_hold(self):
        """Path: no custom benchmark → Buy & Hold name, equity present."""
        data = make_ohlcv(30)
        signals = pd.Series(0, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[20] = -1
        engine = BacktestEngine(fee_model=ZeroFeeModel(), mode="vectorized",
                                include_benchmark=True)
        engine.set_signals(signals)
        result = engine.run(data)

        assert result.benchmark_name == "Buy & Hold"
        assert result.benchmark_equity is not None

    def test_custom_benchmark_via_backtest(self):
        """Path: custom price benchmark via backtest() → normalized equity, custom name."""
        data = make_ohlcv(30)
        signals = pd.Series(0, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[20] = -1
        benchmark = pd.Series(np.linspace(200, 300, len(data)), index=data.index)

        result = backtest(
            data, signals=signals, fee_model=ZeroFeeModel(),
            benchmark=benchmark, benchmark_type="prices",
            benchmark_name="My Index", include_benchmark=True,
        )

        assert result.benchmark_name == "My Index"
        assert abs(result.benchmark_equity.iloc[0] - 100_000) < 1.0
        assert abs(result.benchmark_equity.iloc[-1] - 150_000) < 1.0
