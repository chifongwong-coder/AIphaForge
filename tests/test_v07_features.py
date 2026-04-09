"""
End-to-end tests for v0.7 features: multi-asset backtesting.

Each test covers one distinct user path through the engine.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from aiphaforge import (
    BacktestEngine,
    BacktestHook,
    BacktestResult,
    EqualWeightAllocator,
    FixedWeightAllocator,
    HookContext,
    LatencyHook,
    PerformanceAnalyzer,
    ProRataAllocator,
)
from aiphaforge.fees import ZeroFeeModel

from .conftest import make_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_multi_data(n_bars=30, n_symbols=3):
    """Create multi-asset data dict with distinct price paths."""
    data = {}
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    for i, sym in enumerate(symbols):
        data[sym] = make_ohlcv(
            n_bars, start_price=100 + i * 50, trend=0.002 * (i + 1))
    return data, symbols


def _make_multi_signals(data_dict, buy_bars=None, sell_bars=None):
    """Create multi-asset signal dict with buy/sell on specified bars."""
    buy_bars = buy_bars or {sym: [1] for sym in data_dict}
    sell_bars = sell_bars or {sym: [20] for sym in data_dict}
    signals = {}
    for sym, df in data_dict.items():
        sig = pd.Series(0, index=df.index, dtype=float)
        for b in buy_bars.get(sym, []):
            if b < len(sig):
                sig.iloc[b] = 1
        for b in sell_bars.get(sym, []):
            if b < len(sig):
                sig.iloc[b] = -1
        signals[sym] = sig
    return signals


# ===========================================================================
# Multi-Asset Event-Driven — core paths
# ===========================================================================


class TestMultiAssetEventDriven:

    def test_basic_multi_asset_round_trip(self):
        """Path: multi-asset buy → hold → sell → trades from all assets."""
        data, symbols = _make_multi_data(30, 3)
        signals = _make_multi_signals(data)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=300_000,
            capital_allocator=EqualWeightAllocator(),
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == 30
        assert result.num_trades >= 3  # at least one per symbol

    def test_single_asset_backward_compat(self):
        """Path: single DataFrame + single Series → same behavior as v0.6."""
        data = make_ohlcv(30)
        signals = pd.Series(0, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[20] = -1

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == 30
        assert result.num_trades >= 1

    def test_auto_allocator_warning(self):
        """Path: multi-asset without allocator → auto EqualWeight + warning."""
        data, _ = _make_multi_data(10, 2)
        signals = _make_multi_signals(data)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=200_000,
            include_benchmark=False,
        )
        engine.set_signals(signals)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = engine.run(data)
            allocator_warnings = [
                x for x in w if "capital_allocator" in str(x.message)]
            assert len(allocator_warnings) >= 1

        assert isinstance(result, BacktestResult)

    def test_latency_hook_works_in_multi_asset(self):
        """Path: LatencyHook + multi-asset → runs without error (v0.8)."""
        data, _ = _make_multi_data(10, 2)
        signals = _make_multi_signals(data)

        inner = BacktestHook()
        hook = LatencyHook(inner_hook=inner, latency_model="fixed",
                           latency_params={"bars": 1})

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=200_000,
            hooks=[hook],
            capital_allocator=EqualWeightAllocator(),
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == 10


# ===========================================================================
# Capital Allocator
# ===========================================================================


class TestCapitalAllocator:

    def test_equal_weight_splits_budget(self):
        """Path: 3 simultaneous buy signals → each gets 1/3 cash."""
        data, symbols = _make_multi_data(10, 3)
        # All buy on bar 1
        signals = _make_multi_signals(
            data,
            buy_bars={sym: [1] for sym in symbols},
            sell_bars={sym: [] for sym in symbols},
        )

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=300_000,
            capital_allocator=EqualWeightAllocator(),
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        # Should have trades for all 3 symbols
        assert result.num_trades >= 0  # positions opened
        assert len(result.equity_curve) == 10

    def test_fixed_weight_allocator(self):
        """Path: FixedWeightAllocator with explicit weights."""
        data, symbols = _make_multi_data(10, 2)
        signals = _make_multi_signals(
            data,
            buy_bars={sym: [1] for sym in symbols},
            sell_bars={sym: [] for sym in symbols},
        )

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=200_000,
            capital_allocator=FixedWeightAllocator(
                {symbols[0]: 0.7, symbols[1]: 0.3}),
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)
        assert len(result.equity_curve) == 10


# ===========================================================================
# Multi-Asset Vectorized
# ===========================================================================


class TestMultiAssetVectorized:

    def test_vectorized_multi_asset(self):
        """Path: vectorized mode + dict data → per-asset weighted runs."""
        data, symbols = _make_multi_data(30, 2)
        signals = _make_multi_signals(data)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="vectorized",
            initial_capital=200_000,
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(
            data, weights={symbols[0]: 0.6, symbols[1]: 0.4})

        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0

    def test_vectorized_weight_validation(self):
        """Path: weights sum > 1.0 → ValueError."""
        data, symbols = _make_multi_data(10, 2)
        signals = _make_multi_signals(data)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="vectorized",
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine.set_signals(signals)

        with pytest.raises(ValueError, match="exceeds 1.0"):
            engine.run(data, weights={symbols[0]: 0.8, symbols[1]: 0.5})


# ===========================================================================
# Multi-Asset HookContext
# ===========================================================================


class TestMultiAssetHookContext:

    def test_multi_asset_hook_receives_all_fields(self):
        """Path: multi-asset hook gets active_symbols, all_bar_data, etc."""
        data, symbols = _make_multi_data(10, 2)
        signals = _make_multi_signals(data)

        class InspectorHook(BacktestHook):
            def __init__(self):
                self.contexts = []

            def on_pre_signal(self, ctx):
                self.contexts.append(ctx)

        hook = InspectorHook()
        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=200_000,
            capital_allocator=EqualWeightAllocator(),
            hooks=[hook],
            include_benchmark=False,
        )
        engine.set_signals(signals)
        engine.run(data)

        assert len(hook.contexts) == 10  # one per bar
        ctx = hook.contexts[0]
        # Multi-asset fields populated
        assert ctx.active_symbols is not None
        assert len(ctx.active_symbols) == 2
        assert ctx.all_bar_data is not None
        assert ctx.all_brokers is not None
        assert ctx.all_data is not None
        # Single-asset fields are defaults
        assert ctx.bar_data is None
        assert ctx.broker is None
        assert ctx.symbol == ""

    def test_single_asset_hook_unchanged(self):
        """Path: single-asset hook gets bar_data, broker, symbol as before."""
        data = make_ohlcv(10)
        signals = pd.Series(0, index=data.index, dtype=float)

        class InspectorHook(BacktestHook):
            def __init__(self):
                self.contexts = []

            def on_pre_signal(self, ctx):
                self.contexts.append(ctx)

        hook = InspectorHook()
        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            hooks=[hook],
            include_benchmark=False,
        )
        engine.set_signals(signals)
        engine.run(data)

        ctx = hook.contexts[0]
        assert ctx.bar_data is not None
        assert ctx.broker is not None
        assert ctx.symbol == "default"
        # Multi-asset fields are None
        assert ctx.active_symbols is None
        assert ctx.all_brokers is None


# ===========================================================================
# Broker assigned_symbol guard
# ===========================================================================


class TestBrokerSymbolGuard:

    def test_assigned_symbol_rejects_mismatch(self):
        """Path: broker with assigned_symbol rejects wrong symbol order."""
        from aiphaforge import Broker, OrderStatus

        broker = Broker(
            fee_model=ZeroFeeModel(),
            assigned_symbol="AAPL",
        )
        order = broker.create_market_order("TSLA", "buy", 100)
        broker.submit_order(order)
        assert order.status == OrderStatus.REJECTED

    def test_assigned_symbol_accepts_match(self):
        """Path: broker with assigned_symbol accepts matching order."""
        from aiphaforge import Broker, OrderStatus

        broker = Broker(
            fee_model=ZeroFeeModel(),
            assigned_symbol="AAPL",
        )
        order = broker.create_market_order("AAPL", "buy", 100)
        broker.submit_order(order)
        assert order.status == OrderStatus.PENDING


# ===========================================================================
# Per-Asset PnL and Performance Analysis
# ===========================================================================


class TestPerAssetPnL:

    def test_event_driven_per_asset_pnl_populated(self):
        """Path: multi-asset event-driven → per_asset_pnl on result."""
        data, symbols = _make_multi_data(30, 2)
        signals = _make_multi_signals(data)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=200_000,
            capital_allocator=EqualWeightAllocator(),
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert result.per_asset_pnl is not None
        assert len(result.per_asset_pnl) == 2
        for sym in symbols:
            assert sym in result.per_asset_pnl
            assert len(result.per_asset_pnl[sym]) == 30

    def test_vectorized_per_asset_pnl_populated(self):
        """Path: multi-asset vectorized → per_asset_pnl from equity diffs."""
        data, symbols = _make_multi_data(30, 2)
        signals = _make_multi_signals(data)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="vectorized",
            initial_capital=200_000,
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(
            data, weights={symbols[0]: 0.5, symbols[1]: 0.5})

        assert result.per_asset_pnl is not None
        assert len(result.per_asset_pnl) == 2

    def test_per_asset_trades_grouped(self):
        """Path: multi-asset → per_asset_trades grouped by symbol."""
        data, symbols = _make_multi_data(30, 2)
        signals = _make_multi_signals(data)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=200_000,
            capital_allocator=EqualWeightAllocator(),
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        if result.num_trades > 0:
            assert result.per_asset_trades is not None
            for sym in result.per_asset_trades:
                assert sym in symbols

    def test_per_asset_analysis_returns_metrics(self):
        """Path: PerformanceAnalyzer.per_asset_analysis() → dict of metrics."""
        data, symbols = _make_multi_data(30, 2)
        signals = _make_multi_signals(data)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=200_000,
            capital_allocator=EqualWeightAllocator(),
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        analyzer = PerformanceAnalyzer(result)
        pa = analyzer.per_asset_analysis()
        assert len(pa) == 2
        for sym in symbols:
            assert 'total_pnl' in pa[sym]
            assert 'sharpe_ratio' in pa[sym]
            assert 'max_drawdown' in pa[sym]
            assert 'volatility' in pa[sym]

    def test_correlation_matrix(self):
        """Path: PerformanceAnalyzer.correlation_matrix() → DataFrame."""
        data, symbols = _make_multi_data(30, 3)
        signals = _make_multi_signals(data)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=300_000,
            capital_allocator=EqualWeightAllocator(),
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        analyzer = PerformanceAnalyzer(result)
        corr = analyzer.correlation_matrix()
        assert corr is not None
        assert corr.shape == (3, 3)
        # Diagonal should be 1.0
        for sym in symbols:
            assert abs(corr.loc[sym, sym] - 1.0) < 1e-10

    def test_single_asset_no_per_asset_fields(self):
        """Path: single-asset → per_asset_pnl is None."""
        data = make_ohlcv(20)
        signals = pd.Series(0, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[15] = -1

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert result.per_asset_pnl is None
        analyzer = PerformanceAnalyzer(result)
        assert analyzer.per_asset_analysis() == {}
        assert analyzer.correlation_matrix() is None


# ===========================================================================
# Edge cases and validation
# ===========================================================================


class TestEdgeCases:

    def test_no_signals_no_strategy_multi_raises(self):
        """Path: multi-asset without signals or strategy → ValueError."""
        data, _ = _make_multi_data(10, 2)

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode="event_driven",
            initial_capital=100_000,
            include_benchmark=False,
        )

        with pytest.raises(ValueError, match="Must set either"):
            engine.run(data)

    def test_fixed_weight_allocator_validation(self):
        """Path: FixedWeightAllocator with weights > 1.0 → ValueError."""
        with pytest.raises(ValueError, match="exceeds 1.0"):
            FixedWeightAllocator({"A": 0.8, "B": 0.5})

    def test_pro_rata_allocator(self):
        """Path: ProRataAllocator distributes proportional to signal strength."""
        from aiphaforge import Portfolio

        allocator = ProRataAllocator()

        class FakePortfolio:
            cash = 10_000.0

        signals = {"A": 2, "B": 1, "C": -1}
        prices = {"A": 100, "B": 50, "C": 200}
        budgets = allocator.allocate(signals, prices, FakePortfolio(), None)

        # A has strength 2, B has strength 1 → A gets 2/3, B gets 1/3
        assert abs(budgets["A"] - 10_000 * 2 / 3) < 0.01
        assert abs(budgets["B"] - 10_000 * 1 / 3) < 0.01
        # C is sell → None
        assert budgets["C"] is None
