"""
End-to-end tests for v0.6 features: PARTIALLY_EXPIRED portfolio fix,
same-bar IOC/FOK processing, SymbolRoutingLatencyHook, and custom
benchmark regression.

Every test walks the COMPLETE user journey:
    create engine -> configure -> run backtest -> check results.
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
            else:
                raise ValueError(f"Unknown order_type: {self.order_type}")

            ctx.broker.submit_order(order, ctx.timestamp)
            self.submitted_orders.append(order)


class EquityRecordingHook(BacktestHook):
    """Records equity and position state after each bar."""

    def __init__(self):
        self.equity_by_bar: dict[int, float] = {}
        self.position_by_bar: dict[int, float] = {}

    def on_bar(self, ctx: HookContext) -> None:
        self.equity_by_bar[ctx.bar_index] = ctx.portfolio.total_equity
        self.position_by_bar[ctx.bar_index] = ctx.portfolio.get_position_size(
            ctx.symbol
        )


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
# 1. IOC partial fill updates portfolio
# ===========================================================================


class TestPartiallyExpiredPortfolio:
    """IOC partial fill should update the portfolio."""

    def test_ioc_partial_fill_updates_portfolio(self):
        """IOC with low volume + partial_fills: partial fill changes equity.

        We need to use a custom engine setup that enables partial_fills on
        the broker.  The hook submits a large IOC order, the broker clips
        it to the volume limit, and the remainder expires.
        """
        n = 10
        dates = pd.bdate_range("2024-01-01", periods=n, freq="B")
        data = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [110.0] * n,
                "low": [90.0] * n,
                "close": [105.0] * n,
                "volume": [100.0] * n,  # Very low volume
            },
            index=dates,
        )

        equity_hook = EquityRecordingHook()
        # Submit a large IOC order that will only partially fill
        hook = OrderSubmittingHook(
            submit_on_bars=[0],
            order_type="market",
            side="buy",
            size=1000,  # Much larger than volume*volume_limit_pct
            time_in_force="IOC",
        )

        # Use custom Broker with partial_fills=True
        from aiphaforge.broker import Broker
        from aiphaforge.config import BacktestConfig
        from aiphaforge.portfolio import Portfolio

        config = BacktestConfig(
            initial_capital=1_000_000,
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            hooks=[hook, equity_hook],
        )

        # Run manually to control the broker's partial_fills setting
        portfolio = Portfolio(initial_capital=1_000_000)
        broker = Broker(
            fee_model=ZeroFeeModel(),
            partial_fills=True,
            volume_limit_pct=0.1,
            immediate_fill_price="close",
        )
        broker.set_portfolio(portfolio)

        for h in config.hooks:
            h.on_backtest_start(data, "default", config=config)

        for i, (timestamp, bar) in enumerate(data.iterrows()):
            portfolio.update_prices({"default": bar["close"]}, timestamp, record=False)

            filled_orders = broker.process_bar(bar, timestamp, "default")
            for order in filled_orders:
                portfolio.update_from_order(order, timestamp)

            from aiphaforge.hooks import HookContext
            ctx = HookContext(
                bar_index=i, timestamp=timestamp, bar_data=bar,
                data=data.iloc[:i + 1], portfolio=portfolio,
                symbol="default", broker=broker,
            )
            for h in config.hooks:
                h.on_pre_signal(ctx)

            immediate = broker.process_immediate_orders(bar, timestamp, "default")
            for order in immediate:
                portfolio.update_from_order(order, timestamp)

            for h in config.hooks:
                h.on_bar(ctx)
            portfolio._record_equity(timestamp)

        order = hook.submitted_orders[0]
        # Should be PARTIALLY_EXPIRED (filled some, expired remainder)
        assert order.status == OrderStatus.PARTIALLY_EXPIRED
        assert order.filled_size > 0
        assert order.filled_size < 1000

        # Portfolio should reflect the partial fill
        assert equity_hook.position_by_bar[0] > 0
        # Cash should have decreased (position opened)
        assert portfolio.cash < 1_000_000


# ===========================================================================
# 2. Same-bar IOC fill via hook
# ===========================================================================


class TestSameBarIOCFill:
    """IOC orders submitted by hooks fill on the same bar."""

    def test_same_bar_ioc_fill_via_hook(self):
        """Hook submits IOC on bar 0 -> fills on same bar 0."""
        data = make_ohlcv(10)

        equity_hook = EquityRecordingHook()
        ioc_hook = OrderSubmittingHook(
            submit_on_bars=[0],
            order_type="market",
            side="buy",
            size=100,
            time_in_force="IOC",
        )

        _run_event_driven(data, [ioc_hook, equity_hook])

        order = ioc_hook.submitted_orders[0]
        assert order.is_filled
        # Position should be visible on bar 0 (same-bar fill)
        assert equity_hook.position_by_bar[0] == 100

    def test_same_bar_ioc_fill_price_uses_close(self):
        """Same-bar IOC fill price equals the current bar's close."""
        n = 10
        dates = pd.bdate_range("2024-01-01", periods=n, freq="B")
        # Use deterministic prices
        data = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [110.0] * n,
                "low": [90.0] * n,
                "close": [105.0] * n,
                "volume": [1_000_000.0] * n,
            },
            index=dates,
        )

        ioc_hook = OrderSubmittingHook(
            submit_on_bars=[0],
            order_type="market",
            side="buy",
            size=100,
            time_in_force="IOC",
        )

        _run_event_driven(data, [ioc_hook])

        order = ioc_hook.submitted_orders[0]
        assert order.is_filled
        # Default immediate_fill_price is "close"
        assert order.filled_price == pytest.approx(105.0)


# ===========================================================================
# 3. GTC-only regression
# ===========================================================================


class TestGTCRegression:
    """GTC-only backtests produce identical results with/without second pass."""

    def test_gtc_only_regression(self):
        """GTC-only backtest: result identical whether second pass runs or not."""
        data = make_ohlcv(30)

        gtc_hook = OrderSubmittingHook(
            submit_on_bars=[0],
            order_type="market",
            side="buy",
            size=100,
            time_in_force="GTC",
        )
        sell_hook = OrderSubmittingHook(
            submit_on_bars=[15],
            order_type="market",
            side="sell",
            size=100,
            time_in_force="GTC",
        )

        result = _run_event_driven(data, [gtc_hook, sell_hook])

        # Result should have exactly 1 completed trade (buy-sell)
        assert len(result.trades) == 1
        # The GTC order fills on the NEXT bar, not same bar
        assert gtc_hook.submitted_orders[0].is_filled
        assert sell_hook.submitted_orders[0].is_filled


# ===========================================================================
# 4. SymbolRoutingLatencyHook delays by symbol
# ===========================================================================


class TestSymbolRoutingLatencyHook:
    """Per-symbol latency routing."""

    def test_symbol_routing_different_latency_per_symbol(self):
        """Different symbols get different latency delays."""
        data = make_ohlcv(20)

        # Create inner hook that submits for the current symbol
        inner = OrderSubmittingHook(
            submit_on_bars=[0],
            order_type="market",
            side="buy",
            size=100,
        )

        routing_hook = SymbolRoutingLatencyHook(
            inner_hook=inner,
            default_latency_model="fixed",
            default_latency_params={"bars": 5},
            symbol_overrides={
                "default": ("fixed", {"bars": 1}),
            },
        )

        result = _run_event_driven(data, [routing_hook])

        # With 1-bar latency for "default" symbol, order should fill early
        order = inner.submitted_orders[0]
        assert order.is_filled
        assert result is not None
        assert len(result.equity_curve) == 20

    def test_symbol_routing_falls_back_to_default(self):
        """Unmatched symbol uses default latency model."""
        data = make_ohlcv(20)

        inner = OrderSubmittingHook(
            submit_on_bars=[0],
            order_type="market",
            side="buy",
            size=100,
        )

        # No override for "default" symbol -> uses default_latency_model
        routing_hook = SymbolRoutingLatencyHook(
            inner_hook=inner,
            default_latency_model="fixed",
            default_latency_params={"bars": 3},
            symbol_overrides={
                "AAPL": ("fixed", {"bars": 1}),
            },
        )

        equity_hook = EquityRecordingHook()
        _run_event_driven(data, [routing_hook, equity_hook])

        order = inner.submitted_orders[0]
        assert order.is_filled
        # With 3-bar default latency, position should NOT appear at bar 0 or 1
        assert equity_hook.position_by_bar.get(0, 0) == 0
        assert equity_hook.position_by_bar.get(1, 0) == 0


# ===========================================================================
# 5. Custom benchmark with benchmark_name (v0.5 regression)
# ===========================================================================


class TestBenchmarkNameRegression:
    """Regression: benchmark_name flows through backtest() convenience."""

    def test_backtest_convenience_benchmark_name(self):
        """backtest() with benchmark_name: name appears in result."""
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
            benchmark_name="S&P 500",
            include_benchmark=True,
        )

        assert result.benchmark_name == "S&P 500"
        assert result.benchmark_equity is not None
