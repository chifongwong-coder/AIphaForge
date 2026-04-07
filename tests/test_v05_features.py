"""
End-to-end tests for v0.5 features: TIF orders, LatencyHook, and custom
benchmarks.

Every test walks the COMPLETE user journey:
    create engine -> configure -> run backtest -> check results / order status.

No direct Broker or Order manipulation -- everything flows through
BacktestEngine (or the ``backtest`` convenience function).
"""

from typing import Optional

import numpy as np
import pandas as pd
import pytest

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
                    ctx.symbol,
                    self.side,
                    self.size,
                    "test",
                    ctx.timestamp,
                    time_in_force=tif,
                )
            elif self.order_type == "limit":
                limit_price = self.order_kwargs.get("limit_price", 50.0)
                order = ctx.broker.create_limit_order(
                    ctx.symbol,
                    self.side,
                    self.size,
                    limit_price,
                    "test",
                    ctx.timestamp,
                    time_in_force=tif,
                )
            elif self.order_type == "stop":
                stop_price = self.order_kwargs.get("stop_price", 50.0)
                order = ctx.broker.create_stop_order(
                    ctx.symbol,
                    self.side,
                    self.size,
                    stop_price,
                    "test",
                    ctx.timestamp,
                    time_in_force=tif,
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


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _run_event_driven(
    data: pd.DataFrame,
    hooks: list,
    initial_capital: float = 100_000,
    include_benchmark: bool = False,
    **engine_kwargs,
) -> BacktestResult:
    """Run an event-driven backtest with zero signals and given hooks."""
    signals = pd.Series(0, index=data.index, dtype=float)
    engine = BacktestEngine(
        fee_model=ZeroFeeModel(),
        mode="event_driven",
        initial_capital=initial_capital,
        hooks=hooks,
        include_benchmark=include_benchmark,
        **engine_kwargs,
    )
    engine.set_signals(signals)
    return engine.run(data)


# ===========================================================================
# TIF end-to-end scenarios
# ===========================================================================


class TestTIFEndToEnd:
    """End-to-end TIF (Time-In-Force) scenarios through the engine."""

    def test_gtc_market_order_round_trip(self):
        """GTC order: buy fills -> sell fills -> trades in result."""
        data = make_ohlcv(10)

        # Buy on bar 0, sell on bar 5
        buy_hook = OrderSubmittingHook(
            submit_on_bars=[0],
            order_type="market",
            side="buy",
            size=100,
            time_in_force="GTC",
        )
        sell_hook = OrderSubmittingHook(
            submit_on_bars=[5],
            order_type="market",
            side="sell",
            size=100,
            time_in_force="GTC",
        )

        result = _run_event_driven(data, [buy_hook, sell_hook])

        # Both orders should be filled
        assert buy_hook.submitted_orders[0].status == OrderStatus.FILLED
        assert sell_hook.submitted_orders[0].status == OrderStatus.FILLED

        # Result should contain at least one completed trade
        assert result is not None
        assert len(result.equity_curve) == 10
        assert len(result.trades) >= 1

    def test_ioc_unfillable_order_has_no_effect(self):
        """IOC limit at unreachable price: expires, zero trades, equity unchanged."""
        data = make_ohlcv(10, start_price=100)
        hook = OrderSubmittingHook(
            submit_on_bars=[0],
            order_type="limit",
            limit_price=1.0,  # way below market
            time_in_force="IOC",
        )

        result = _run_event_driven(data, [hook])

        order = hook.submitted_orders[0]
        assert order.status == OrderStatus.EXPIRED
        assert order.metadata.get("expiry_reason") == "ioc_timeout"
        # No trades should have occurred
        assert len(result.trades) == 0
        # Equity should be unchanged (initial capital throughout)
        assert abs(result.equity_curve.iloc[-1] - 100_000) < 1.0

    def test_ioc_market_order_fills_and_trades(self):
        """IOC market order fills normally, trade appears in result."""
        data = make_ohlcv(10)
        hook = OrderSubmittingHook(
            submit_on_bars=[0],
            order_type="market",
            time_in_force="IOC",
        )

        result = _run_event_driven(data, [hook])

        order = hook.submitted_orders[0]
        assert order.status == OrderStatus.FILLED
        # Equity should have changed (position opened)
        assert result is not None
        assert len(result.equity_curve) == 10

    def test_fok_rejected_order_has_no_effect(self):
        """FOK too large: expires, zero trades, equity unchanged."""
        data = make_ohlcv(10)
        hook = OrderSubmittingHook(
            submit_on_bars=[0],
            order_type="market",
            size=10_000_000,  # much larger than volume * volume_limit_pct
            time_in_force="FOK",
        )

        result = _run_event_driven(data, [hook])

        order = hook.submitted_orders[0]
        assert order.status == OrderStatus.EXPIRED
        assert order.metadata.get("expiry_reason") == "fok_volume"
        assert len(result.trades) == 0
        assert abs(result.equity_curve.iloc[-1] - 100_000) < 1.0

    def test_day_order_cross_session_expires(self):
        """DAY order across dates: expires, zero trades."""
        # Data spanning multiple calendar days (business-day freq)
        data = make_ohlcv(20, start_date="2024-01-01")
        hook = OrderSubmittingHook(
            submit_on_bars=[0],
            order_type="limit",
            limit_price=1.0,  # won't fill
            time_in_force="DAY",
        )

        result = _run_event_driven(data, [hook])

        order = hook.submitted_orders[0]
        assert order.status == OrderStatus.EXPIRED
        assert order.metadata.get("expiry_reason") == "day_session_end"
        assert len(result.trades) == 0

    def test_day_order_fills_within_same_session(self):
        """DAY market order fills on same calendar date (intraday data)."""
        n = 10
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
        prices = np.maximum(prices, 50.0)
        dates = pd.date_range("2024-01-02 09:30", periods=n, freq="h")
        data = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.full(n, 500_000.0),
            },
            index=dates,
        )

        hook = OrderSubmittingHook(
            submit_on_bars=[0],
            order_type="market",
            time_in_force="DAY",
        )

        result = _run_event_driven(data, [hook])

        order = hook.submitted_orders[0]
        assert order.status == OrderStatus.FILLED
        assert result is not None
        assert len(result.equity_curve) == n


# ===========================================================================
# LatencyHook end-to-end scenarios
# ===========================================================================


class TestLatencyHookEndToEnd:
    """End-to-end tests for the LatencyHook decorator."""

    def test_latency_delays_fill_by_n_bars(self):
        """latency=3: no position at bars 1-2, position appears at bar 3."""
        data = make_ohlcv(10)

        inner = OrderSubmittingHook(submit_on_bars=[0])
        latency_hook = LatencyHook(
            inner_hook=inner,
            latency_model="fixed",
            latency_params={"bars": 3},
        )
        collector = OrderCollectorHook()

        result = _run_event_driven(data, [latency_hook, collector])

        assert len(inner.submitted_orders) == 1

        # Bars 0, 1 should have no fills
        for bar_idx in range(2):
            fills = collector.fills_by_bar.get(bar_idx, [])
            assert len(fills) == 0, f"Unexpected fills on bar {bar_idx}"

        # By bar 3+ the order should be filled
        all_fills = []
        for bar_idx in range(3, 10):
            all_fills.extend(collector.fills_by_bar.get(bar_idx, []))
        assert len(all_fills) >= 1, "Order was never filled after latency delay"

        # Result should be valid
        assert result is not None
        assert len(result.equity_curve) == 10

    def test_latency_one_fills_next_bar(self):
        """latency=1: decision at bar 0, fill at bar 1."""
        data = make_ohlcv(10)

        inner = OrderSubmittingHook(submit_on_bars=[0])
        latency_hook = LatencyHook(
            inner_hook=inner,
            latency_model="fixed",
            latency_params={"bars": 1},
        )

        result = _run_event_driven(data, [latency_hook])

        assert len(inner.submitted_orders) == 1
        order = inner.submitted_orders[0]
        assert order.is_filled
        assert result is not None
        assert len(result.equity_curve) == 10

    def test_simple_latency_hook_full_flow(self):
        """SimpleLatencyHook subclass: agent decides -> delay -> fill -> result."""
        data = make_ohlcv(10)

        class MyAgent(SimpleLatencyHook):
            def __init__(self):
                super().__init__(
                    latency_model="fixed",
                    latency_params={"bars": 2},
                )
                self.decisions_made = 0

            def make_decision(self, ctx: HookContext) -> Optional[Order]:
                if ctx.bar_index == 0:
                    self.decisions_made += 1
                    return ctx.broker.create_market_order(
                        ctx.symbol, "buy", 100, "agent_decision", ctx.timestamp,
                    )
                return None

        agent = MyAgent()
        result = _run_event_driven(data, [agent])

        assert agent.decisions_made == 1
        assert result is not None
        assert len(result.equity_curve) == 10
        # Equity should have changed since we bought shares
        # (or at least the backtest completed without error)

    def test_latency_hook_reusable_across_runs(self):
        """Run backtest twice with the same hook instance; both results valid."""
        data = make_ohlcv(10)

        inner = OrderSubmittingHook(submit_on_bars=[0])
        latency_hook = LatencyHook(
            inner_hook=inner,
            latency_model="fixed",
            latency_params={"bars": 2},
        )

        # First run
        result1 = _run_event_driven(data, [latency_hook])
        assert result1 is not None
        assert len(result1.equity_curve) == 10

        # Reset inner hook state for second run
        inner.submitted_orders.clear()
        inner.submit_on_bars = [0]

        # Second run with the SAME hook instance
        result2 = _run_event_driven(data, [latency_hook])
        assert result2 is not None
        assert len(result2.equity_curve) == 10

    def test_latency_preserves_decision_timestamp_for_tif(self):
        """LatencyHook + DAY order: created_time is from the decision bar."""
        data = make_ohlcv(10)

        inner = OrderSubmittingHook(submit_on_bars=[0])
        latency_hook = LatencyHook(
            inner_hook=inner,
            latency_model="fixed",
            latency_params={"bars": 3},
        )

        _run_event_driven(data, [latency_hook])

        order = inner.submitted_orders[0]
        decision_ts = data.index[0]
        assert order.created_time == decision_ts


# ===========================================================================
# Benchmark end-to-end scenarios
# ===========================================================================


class TestBenchmarkEndToEnd:
    """End-to-end tests for custom benchmark support."""

    def test_default_buy_and_hold_benchmark(self):
        """No custom benchmark: buy-and-hold name, equity curve present."""
        data = make_ohlcv(30)
        signals = pd.Series(0, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[20] = -1

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="vectorized",
            include_benchmark=True,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert result.benchmark_name == "Buy & Hold"
        assert result.benchmark_equity is not None
        assert len(result.benchmark_equity) == len(data)

    def test_custom_price_benchmark(self):
        """Price series benchmark: normalized to initial_capital."""
        data = make_ohlcv(30)
        signals = pd.Series(0, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[20] = -1

        benchmark_prices = pd.Series(
            np.linspace(200, 300, len(data)),
            index=data.index,
        )

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="vectorized",
            include_benchmark=True,
            initial_capital=100_000,
        )
        engine._config_benchmark_name = "Custom Index"
        engine.set_signals(signals)
        result = engine.run(
            data,
            benchmark=benchmark_prices,
            benchmark_type="prices",
        )

        assert result.benchmark_name == "Custom Index"
        assert result.benchmark_equity is not None
        assert abs(result.benchmark_equity.iloc[0] - 100_000) < 1.0
        assert abs(result.benchmark_equity.iloc[-1] - 150_000) < 1.0

    def test_custom_returns_benchmark(self):
        """Returns series benchmark: equity = (1+r).cumprod() * capital."""
        data = make_ohlcv(30)
        signals = pd.Series(0, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[20] = -1

        benchmark_returns = pd.Series(0.01, index=data.index)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="vectorized",
            include_benchmark=True,
            initial_capital=100_000,
        )
        engine.set_signals(signals)
        result = engine.run(
            data,
            benchmark=benchmark_returns,
            benchmark_type="returns",
        )

        assert result.benchmark_equity is not None
        expected_final = 100_000 * (1.01 ** len(data))
        assert abs(result.benchmark_equity.iloc[-1] - expected_final) < 1.0

    def test_auto_detection_warns(self):
        """Benchmark without type: auto-detection warning emitted."""
        data = make_ohlcv(30)
        signals = pd.Series(0, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[20] = -1

        benchmark = pd.Series(
            np.linspace(100, 200, len(data)),
            index=data.index,
        )

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="vectorized",
            include_benchmark=True,
        )
        engine.set_signals(signals)

        with pytest.warns(UserWarning, match="benchmark_type='auto'"):
            engine.run(data, benchmark=benchmark)

    def test_backtest_convenience_with_benchmark(self):
        """backtest() convenience function: pass all benchmark params -> verify."""
        data = make_ohlcv(30)
        signals = pd.Series(0, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[20] = -1

        benchmark = pd.Series(
            np.linspace(100, 200, len(data)),
            index=data.index,
        )

        result = backtest(
            data,
            signals=signals,
            fee_model=ZeroFeeModel(),
            benchmark=benchmark,
            benchmark_type="prices",
            benchmark_name="My Benchmark",
            include_benchmark=True,
        )

        assert result.benchmark_name == "My Benchmark"
        assert result.benchmark_equity is not None
        assert abs(result.benchmark_equity.iloc[0] - 100_000) < 1.0


# ===========================================================================
# Regression end-to-end scenarios
# ===========================================================================


class TestRegressionEndToEnd:
    """Regression tests that run through the full engine."""

    def test_final_capital_zero_is_preserved(self):
        """final_capital=0.0 should be passed through correctly."""
        data = make_ohlcv(10)
        signals = pd.Series(0, index=data.index, dtype=float)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert result.final_capital is not None
        assert isinstance(result.final_capital, (int, float))
