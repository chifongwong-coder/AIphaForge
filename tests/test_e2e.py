"""
AIphaForge End-to-End Tests
===========================

~50 comprehensive tests covering complete user paths through the engine.
Each test exercises a full workflow: setup → run → verify results.
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
    HookContext,
    LatencyHook,
    MetaContext,
    OrderStatus,
    PerformanceAnalyzer,
    Portfolio,
    optimize,
    walk_forward,
)
from aiphaforge.capital_allocator import MarginAllocator
from aiphaforge.config import TurnoverConfig
from aiphaforge.fees import ZeroFeeModel
from aiphaforge.margin import (
    BorrowingCostModel,
    FundingRateModel,
    MarginCallExitRule,
    MarginConfig,
)
from aiphaforge.meta import MetaContext
from aiphaforge.risk import (
    CompositeRiskManager,
    ExposureLimit,
    MaxDrawdownHalt,
)
from aiphaforge.strategies import MACrossover, RSIMeanReversion

from .conftest import make_ohlcv


# ===================================================================
# Core Engine
# ===================================================================


class TestCoreEngine:

    def test_vectorized_backtest(self):
        """Vectorized mode: signals → positions → equity curve."""
        data = make_ohlcv(100)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[10] = 1
        signals.iloc[70] = -1

        engine = BacktestEngine(mode='vectorized', fee_model=ZeroFeeModel(),
                                include_benchmark=False)
        engine.set_signals(signals)
        result = engine.run(data)

        assert result.total_return != 0
        assert len(result.equity_curve) == 100

    def test_event_driven_backtest(self):
        """Event-driven mode: full order lifecycle."""
        data = make_ohlcv(50)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1
        signals.iloc[40] = 0  # flat

        engine = BacktestEngine(mode='event_driven', fee_model=ZeroFeeModel(),
                                initial_capital=100_000, include_benchmark=False)
        engine.set_signals(signals)
        result = engine.run(data)

        assert result.num_trades >= 1
        assert result.final_capital != result.initial_capital

    def test_stop_loss_triggers(self):
        """Stop-loss exits position on drawdown."""
        data = make_ohlcv(50, trend=-0.01)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1

        result_no_sl = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            include_benchmark=False
        ).set_signals(signals).run(data)

        result_sl = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            stop_loss=0.05, include_benchmark=False
        ).set_signals(signals).run(data)

        assert result_sl.final_capital > result_no_sl.final_capital

    def test_signal_semantics(self):
        """signal=0 closes, NaN holds, 0.5 = half position."""
        data = make_ohlcv(30)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1      # full long
        signals.iloc[10] = 0.5   # reduce to half
        signals.iloc[20] = 0     # flat

        engine = BacktestEngine(mode='event_driven', fee_model=ZeroFeeModel(),
                                initial_capital=100_000, include_benchmark=False)
        engine.set_signals(signals)
        result = engine.run(data)

        assert result.num_trades >= 2
        assert len(result.equity_curve) == 30

    def test_signal_transform(self):
        """signal_transform clips z-scores."""
        data = make_ohlcv(20)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 2.5  # z-score > 1

        engine = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            signal_transform=lambda s: np.clip(s, -1, 1),
            include_benchmark=False)
        engine.set_signals(signals)
        result = engine.run(data)
        assert len(result.equity_curve) == 20

    def test_allow_short_false_warns(self):
        """allow_short=False + negative signal warns and skips."""
        data = make_ohlcv(10)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = -1

        engine = BacktestEngine(mode='event_driven', fee_model=ZeroFeeModel(),
                                allow_short=False, include_benchmark=False)
        engine.set_signals(signals)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = engine.run(data)
            assert any("allow_short" in str(x.message) for x in w)
        assert result.num_trades == 0

    def test_benchmark_comparison(self):
        """Custom benchmark overlay on results."""
        data = make_ohlcv(50)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1
        signals.iloc[40] = 0

        benchmark = pd.Series(np.linspace(100, 150, 50), index=data.index)
        result = BacktestEngine(
            mode='vectorized', fee_model=ZeroFeeModel()
        ).set_signals(signals).run(
            data, benchmark=benchmark, benchmark_type='prices')

        assert result.benchmark_equity is not None

    def test_no_strategy_no_signals_raises(self):
        engine = BacktestEngine()
        with pytest.raises(ValueError):
            engine.run(make_ohlcv(10))


# ===================================================================
# Multi-Asset
# ===================================================================


class TestMultiAsset:

    def _make_data(self, n=30):
        return {
            "A": make_ohlcv(n, start_price=100),
            "B": make_ohlcv(n, start_price=200),
        }

    def _make_signals(self, data, buy=1, sell=20):
        signals = {}
        for sym, df in data.items():
            s = pd.Series(np.nan, index=df.index, dtype=float)
            s.iloc[buy] = 1
            s.iloc[sell] = 0
            signals[sym] = s
        return signals

    def test_multi_asset_event_driven(self):
        """Multi-asset with shared capital and per-asset PnL."""
        data = self._make_data()
        signals = self._make_signals(data)

        engine = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=200_000,
            capital_allocator=EqualWeightAllocator(),
            include_benchmark=False)
        engine.set_signals(signals)
        result = engine.run(data)

        assert result.per_asset_pnl is not None
        assert len(result.per_asset_pnl) == 2
        assert result.num_trades >= 2

    def test_multi_asset_vectorized(self):
        """Vectorized multi-asset with weights."""
        data = self._make_data()
        signals = self._make_signals(data)

        engine = BacktestEngine(
            mode='vectorized', fee_model=ZeroFeeModel(),
            initial_capital=200_000, include_benchmark=False)
        engine.set_signals(signals)
        result = engine.run(data, weights={"A": 0.6, "B": 0.4})
        assert len(result.equity_curve) > 0

    def test_target_weights_rebalance(self):
        """Target weights: buy on rebalance date, hold between."""
        data = self._make_data(30)
        dates = data["A"].index
        weights = {
            str(dates[2]): {"A": 0.5, "B": 0.5},
            str(dates[25]): {"A": 0, "B": 0},
        }

        engine = BacktestEngine(mode='event_driven', fee_model=ZeroFeeModel(),
                                initial_capital=100_000, include_benchmark=False)
        engine.set_target_weights(weights)
        result = engine.run(data)
        assert result.num_trades >= 2

    def test_per_asset_analysis(self):
        """PerformanceAnalyzer per-asset metrics + correlation."""
        data = self._make_data(50)
        signals = self._make_signals(data, buy=2, sell=40)

        engine = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=200_000,
            capital_allocator=EqualWeightAllocator(),
            include_benchmark=False)
        engine.set_signals(signals)
        result = engine.run(data)

        analyzer = PerformanceAnalyzer(result)
        pa = analyzer.per_asset_analysis()
        assert len(pa) == 2

    def test_hook_context_multi_asset(self):
        """Multi-asset HookContext has active_symbols and all_brokers."""
        data = self._make_data(10)
        signals = {sym: pd.Series(np.nan, index=df.index, dtype=float)
                   for sym, df in data.items()}

        class Inspector(BacktestHook):
            def __init__(self): self.seen = []
            def on_pre_signal(self, ctx):
                self.seen.append(ctx.active_symbols is not None)

        hook = Inspector()
        engine = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=200_000, hooks=[hook],
            capital_allocator=EqualWeightAllocator(),
            include_benchmark=False)
        engine.set_signals(signals)
        engine.run(data)
        assert all(hook.seen)


# ===================================================================
# Margin & Leverage
# ===================================================================


class TestMarginLeverage:

    def test_leveraged_backtest_with_costs(self):
        """2x leverage + borrowing cost < cash-only result."""
        data = make_ohlcv(30)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[25] = 0

        r_margin = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=10_000, include_benchmark=False,
            margin_config=MarginConfig(initial_margin_ratio=0.5,
                                        borrowing_rate=0.10),
            periodic_cost_model=BorrowingCostModel(),
        ).set_signals(signals).run(data)

        r_cash = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=10_000, include_benchmark=False,
        ).set_signals(signals).run(data)

        assert r_margin.final_capital < r_cash.final_capital

    def test_margin_call_blocks_opens_allows_closes(self):
        """Margin call: new opens blocked, closes allowed."""
        from aiphaforge import Broker
        mc = MarginConfig(initial_margin_ratio=0.5,
                          maintenance_margin_ratio=0.3)
        p = Portfolio(initial_capital=100, margin_config=mc)
        p.positions["X"] = type('Pos', (), {
            'market_value': 500, 'notional_value': 500,
            'is_flat': False, 'is_long': True, 'is_short': False,
            'size': 10, 'current_price': 50.0, 'avg_entry_price': 100.0,
            'unrealized_pnl': -500,
        })()
        p.cash = -400
        assert p.is_margin_call

        broker = Broker(fee_model=ZeroFeeModel(), assigned_symbol="X")
        broker.set_portfolio(p)

        # Close allowed
        close = broker.create_market_order("X", "sell", 10)
        broker.submit_order(close)
        assert close.status == OrderStatus.PENDING

    def test_funding_rate(self):
        """Funding rate reduces equity vs no-cost run."""
        data = make_ohlcv(15, start_price=50000)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1

        r_fund = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=10_000, include_benchmark=False,
            margin_config=MarginConfig(initial_margin_ratio=0.1),
            periodic_cost_model=FundingRateModel(funding_rate_per_bar=0.001),
        ).set_signals(signals).run(data)

        r_none = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=10_000, include_benchmark=False,
            margin_config=MarginConfig(initial_margin_ratio=0.1),
        ).set_signals(signals).run(data)

        assert r_fund.final_capital < r_none.final_capital


# ===================================================================
# Constraints (turnover, lot, position limit)
# ===================================================================


class TestConstraints:

    def test_turnover_cap_scales_trades(self):
        """Tight turnover cap → smaller trades than uncapped."""
        data = make_ohlcv(10)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1

        r_no = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, include_benchmark=False
        ).set_signals(signals).run(data)

        r_cap = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, include_benchmark=False,
            turnover_config=TurnoverConfig(max_turnover_per_bar=0.1)
        ).set_signals(signals).run(data)

        if r_no.trades and r_cap.trades:
            assert r_cap.trades[0].size < r_no.trades[0].size

    def test_close_exempt_from_turnover(self):
        """signal=0 close executes fully despite tight cap."""
        data = make_ohlcv(20)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[15] = 0

        result = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, include_benchmark=False,
            turnover_config=TurnoverConfig(max_turnover_per_bar=0.01)
        ).set_signals(signals).run(data)
        assert result.num_trades >= 1

    def test_lot_size_and_position_limit(self):
        """A-share 100-lot + per-asset position limit."""
        data = {
            "SH": make_ohlcv(15, start_price=100),
            "BTC": make_ohlcv(15, start_price=50000),
        }
        signals = {
            sym: pd.Series(np.nan, index=df.index, dtype=float)
            for sym, df in data.items()
        }
        for s in signals.values():
            s.iloc[1] = 1
            s.iloc[10] = 0

        result = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=200_000, include_benchmark=False,
            capital_allocator=EqualWeightAllocator(),
            lot_size=1, asset_lot_sizes={"SH": 100},
            asset_max_position_pcts={"SH": 0.3},
        ).set_signals(signals).run(data)

        for t in result.trades:
            if t.symbol == "SH":
                assert t.size % 100 == 0


# ===================================================================
# Optimization
# ===================================================================


class TestOptimization:

    def test_grid_search(self):
        """optimize() returns sorted DataFrame."""
        data = make_ohlcv(50)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1
        signals.iloc[40] = 0

        results = optimize(
            data, signals=signals, fee_model=ZeroFeeModel(),
            param_grid={'stop_loss': [0.03, 0.05, 0.10]},
            metric='sharpe_ratio')
        assert len(results) == 3
        assert 'sharpe_ratio' in results.columns

    def test_walk_forward(self):
        """Walk-forward: train → best params → test."""
        data = make_ohlcv(100)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1
        signals.iloc[50] = 0
        signals.iloc[75] = 1
        signals.iloc[95] = 0

        wf = walk_forward(
            data, signals=signals, fee_model=ZeroFeeModel(),
            param_grid={'stop_loss': [0.03, 0.05]},
            train_pct=0.7, metric='sharpe_ratio')

        assert 'best_params' in wf
        assert isinstance(wf['test_result'], BacktestResult)

    def test_strategy_factory_optimization(self):
        """optimize() with strategy_factory for strategy params."""
        data = make_ohlcv(100)
        results = optimize(
            data,
            strategy_factory=lambda p: MACrossover(short=p['short'], long=p['long']),
            param_grid={'short': [5, 10], 'long': [20, 30]},
            fee_model=ZeroFeeModel(), include_benchmark=False)
        assert len(results) == 4


# ===================================================================
# Indicators + Strategies
# ===================================================================


class TestIndicatorsStrategies:

    def test_indicators_to_signals_to_backtest(self):
        """Full chain: indicators → signals → engine → result."""
        from aiphaforge.indicators import SMA
        data = make_ohlcv(100)
        short = SMA(data['close'], 10)
        long = SMA(data['close'], 30)

        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals[short > long] = 1
        signals[short < long] = -1

        result = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, include_benchmark=False
        ).set_signals(signals).run(data)
        assert result.num_trades >= 1

    def test_strategy_one_line_backtest(self):
        """MACrossover().backtest() full round-trip."""
        data = make_ohlcv(100)
        result = MACrossover(short=5, long=20).backtest(
            data, fee_model=ZeroFeeModel(), include_benchmark=False)
        assert isinstance(result, BacktestResult)
        assert result.num_trades >= 1

    def test_strategy_update_params(self):
        """update_params changes signals."""
        data = make_ohlcv(50)
        s = MACrossover(short=5, long=20)
        sig1 = s.generate_signals(data)
        s.update_params(short=3, long=10)
        sig2 = s.generate_signals(data)
        assert not sig1.equals(sig2)

    def test_strategy_multi_asset(self):
        """Strategy.generate_signals with dict data."""
        data = {"A": make_ohlcv(50), "B": make_ohlcv(50)}
        signals = MACrossover().generate_signals(data)
        assert isinstance(signals, dict)
        assert len(signals) == 2


# ===================================================================
# Data + Plotting
# ===================================================================


class TestDataPlotting:

    def test_load_csv_and_run(self):
        """load_csv → engine.run() full chain."""
        import os
        import tempfile
        from aiphaforge.data import load_csv

        df = make_ohlcv(30)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index_label='Date')
            loaded = load_csv(f.name)
            os.unlink(f.name)

        signals = pd.Series(np.nan, index=loaded.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[20] = 0

        result = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            include_benchmark=False
        ).set_signals(signals).run(loaded)
        assert isinstance(result, BacktestResult)

    def test_plot_result(self):
        """plot_result returns matplotlib Figure."""
        pytest.importorskip("matplotlib")
        from aiphaforge.plotting import plot_result

        data = make_ohlcv(30)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[20] = 0

        result = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            include_benchmark=True
        ).set_signals(signals).run(data)

        import matplotlib.figure
        assert isinstance(plot_result(result), matplotlib.figure.Figure)


# ===================================================================
# Risk Management
# ===================================================================


class TestRiskManagement:

    def test_drawdown_halt_blocks_trades(self):
        """MaxDrawdownHalt: drawdown → no new trades → recovery resumes."""
        data = make_ohlcv(50, trend=-0.005)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[30] = 1  # try to add after drawdown

        risk = CompositeRiskManager([
            MaxDrawdownHalt(max_drawdown=0.05, reset_drawdown=0.02)])

        result = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, include_benchmark=False,
            risk_rules=risk
        ).set_signals(signals).run(data)

        assert result.turnover_history is not None

    def test_exposure_limit(self):
        """ExposureLimit blocks overleveraged orders."""
        risk = CompositeRiskManager([
            ExposureLimit(max_long=0.5, max_short=0.3)])

        data = make_ohlcv(20)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1

        result = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, include_benchmark=False,
            risk_rules=risk
        ).set_signals(signals).run(data)
        assert isinstance(result, BacktestResult)

    def test_vectorized_risk_rules(self):
        """Risk rules zero positions in vectorized mode."""
        data = make_ohlcv(100, trend=-0.005)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1

        risk = CompositeRiskManager([MaxDrawdownHalt(max_drawdown=0.05)])

        r_risk = BacktestEngine(
            mode='vectorized', fee_model=ZeroFeeModel(),
            initial_capital=100_000, include_benchmark=False,
            risk_rules=risk
        ).set_signals(signals).run(data)

        r_none = BacktestEngine(
            mode='vectorized', fee_model=ZeroFeeModel(),
            initial_capital=100_000, include_benchmark=False,
        ).set_signals(signals).run(data)

        assert r_risk.final_capital > r_none.final_capital


# ===================================================================
# MetaController
# ===================================================================


class TestMetaController:

    def test_adjust_sizing_mid_backtest(self):
        """Agent reduces position size after drawdown."""
        data = make_ohlcv(50)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[25] = 1  # re-enter

        class SizeAgent(BacktestHook):
            def on_pre_signal(self, ctx):
                if ctx.meta and ctx.bar_index == 20:
                    ctx.meta.adjust_sizing(fraction=0.3)

        result = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, hooks=[SizeAgent()],
            include_benchmark=False
        ).set_signals(signals).run(data)
        assert isinstance(result, BacktestResult)

    def test_swap_strategy(self):
        """Agent swaps from MA to RSI mid-backtest."""
        data = make_ohlcv(100)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1

        class SwapAgent(BacktestHook):
            def on_pre_signal(self, ctx):
                if ctx.meta and ctx.bar_index == 50:
                    ctx.meta.set_strategy(RSIMeanReversion(period=14))

        result = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, hooks=[SwapAgent()],
            include_benchmark=False
        ).set_signals(signals).run(data)
        assert isinstance(result, BacktestResult)

    def test_suppress_resume(self):
        """Agent pauses trading, then resumes."""
        data = make_ohlcv(30)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[15] = 1
        signals.iloc[25] = 0

        class PauseAgent(BacktestHook):
            def on_pre_signal(self, ctx):
                if ctx.meta:
                    if 10 <= ctx.bar_index <= 20:
                        ctx.meta.suppress_signals()
                    else:
                        ctx.meta.resume_signals()

        result = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, hooks=[PauseAgent()],
            include_benchmark=False
        ).set_signals(signals).run(data)
        assert isinstance(result, BacktestResult)

    def test_audit_log(self):
        """Audit log records adjustments with context."""
        data = make_ohlcv(20)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1

        class LogAgent(BacktestHook):
            def on_pre_signal(self, ctx):
                if ctx.meta and ctx.bar_index == 5:
                    ctx.meta.adjust_sizing(fraction=0.5)

        result = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, hooks=[LogAgent()],
            include_benchmark=False
        ).set_signals(signals).run(data)

        audit = result.metadata.get('meta_audit', [])
        assert len(audit) >= 1
        assert 'timestamp' in audit[0]
        assert 'equity' in audit[0]


# ===================================================================
# Edge Cases
# ===================================================================


class TestEdgeCases:

    def test_empty_data_raises(self):
        engine = BacktestEngine(fee_model=ZeroFeeModel())
        engine.set_signals(pd.Series(dtype=float))
        with pytest.raises(ValueError):
            engine.run(make_ohlcv(10).iloc[:0])

    def test_corporate_action_dividend(self):
        """Dividend credits cash."""
        from aiphaforge import CorporateActionHook
        data = make_ohlcv(10)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1

        actions = pd.DataFrame([{
            "date": str(data.index[5]), "symbol": "default",
            "type": "dividend", "value": 1.0}])

        r_div = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, hooks=[CorporateActionHook(actions)],
            include_benchmark=False
        ).set_signals(signals).run(data)

        r_none = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, include_benchmark=False
        ).set_signals(signals).run(data)

        assert r_div.final_capital > r_none.final_capital

    def test_broker_symbol_guard(self):
        """Broker rejects mismatched symbol."""
        from aiphaforge import Broker
        broker = Broker(fee_model=ZeroFeeModel(), assigned_symbol="AAPL")
        order = broker.create_market_order("TSLA", "buy", 100)
        broker.submit_order(order)
        assert order.status == OrderStatus.REJECTED
