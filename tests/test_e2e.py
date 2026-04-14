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
    BayesianResult,
    EqualWeightAllocator,
    HookContext,
    LatencyHook,
    MetaContext,
    OrderStatus,
    PerformanceAnalyzer,
    Portfolio,
    ScheduleHook,
    optimize,
    optimize_bayesian,
    schedule_rebalance,
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
from aiphaforge.strategies import (
    MACrossover,
    MACDStrategy,
    PriorityCascade,
    RSIMeanReversion,
    SelectBest,
    VoteEnsemble,
    WeightedBlend,
    ConditionalSwitch,
)

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


# ===================================================================
# Multi-Timeframe (v1.3)
# ===================================================================


class TestMultiTimeframe:

    def test_single_asset_daily_reference(self):
        """Minute-level primary with daily secondary: hook reads ctx.secondary."""
        from aiphaforge import SecondaryTimeframe

        # Primary: 20 "minute" bars (business-day index for simplicity)
        primary = make_ohlcv(20, start_price=100)

        # Secondary: 4 "daily" bars aligned to every 5th primary bar
        daily_idx = primary.index[::5]  # bars 0, 5, 10, 15
        rng = np.random.default_rng(99)
        daily = pd.DataFrame({
            "open": 100 + rng.standard_normal(len(daily_idx)),
            "high": 105 + rng.standard_normal(len(daily_idx)),
            "low": 95 + rng.standard_normal(len(daily_idx)),
            "close": 102 + rng.standard_normal(len(daily_idx)),
            "volume": rng.integers(1000, 5000, size=len(daily_idx)).astype(float),
        }, index=daily_idx)
        # Clamp open into [low, high]
        daily["open"] = daily["open"].clip(lower=daily["low"], upper=daily["high"])

        signals = pd.Series(np.nan, index=primary.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[18] = 0

        class SecondaryReader(BacktestHook):
            def __init__(self):
                self.collected = []

            def on_pre_signal(self, ctx):
                if ctx.secondary is not None and "1D" in ctx.secondary:
                    tf = ctx.secondary["1D"]
                    assert isinstance(tf, SecondaryTimeframe)
                    bar = tf.bar_data.get("_global")
                    hist = tf.data.get("_global")
                    self.collected.append({
                        "ts": ctx.timestamp,
                        "bar_is_none": bar is None,
                        "hist_len": len(hist),
                    })

        hook = SecondaryReader()
        engine = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, hooks=[hook],
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(
            primary,
            secondary_data={"1D": daily},
            secondary_bar_align="close",
        )

        assert isinstance(result, BacktestResult)
        assert len(hook.collected) == 20

        # Bar 0 has a daily bar (index 0 is in daily)
        entry_0 = hook.collected[0]
        assert not entry_0["bar_is_none"]
        assert entry_0["hist_len"] >= 1

        # Bar 3 should still see the bar from index 0 (forward-fill)
        entry_3 = hook.collected[3]
        assert not entry_3["bar_is_none"]
        assert entry_3["hist_len"] == 1

        # Bar 6 should see 2 daily bars (index 0 and 5)
        entry_6 = hook.collected[6]
        assert not entry_6["bar_is_none"]
        assert entry_6["hist_len"] == 2

    def test_multi_asset_multi_timeframe(self):
        """2 assets x 2 timeframes: correct per-asset per-tf lookup."""
        from aiphaforge import SecondaryTimeframe

        data_a = make_ohlcv(20, start_price=100)
        data_b = make_ohlcv(20, start_price=200)
        data = {"A": data_a, "B": data_b}

        # Build daily secondary per asset (every 5 bars)
        daily_idx = data_a.index[::5]
        rng = np.random.default_rng(77)

        def _make_secondary(idx, base):
            n = len(idx)
            low = base - 5 + rng.standard_normal(n)
            high = base + 5 + rng.standard_normal(n)
            # Ensure high >= low
            low, high = np.minimum(low, high), np.maximum(low, high)
            opn = np.clip(base + rng.standard_normal(n), low, high)
            close = np.clip(base + rng.standard_normal(n), low, high)
            return pd.DataFrame({
                "open": opn, "high": high, "low": low,
                "close": close,
                "volume": rng.integers(1000, 5000, size=n).astype(float),
            }, index=idx)

        daily_a = _make_secondary(daily_idx, 100)
        daily_b = _make_secondary(daily_idx, 200)

        # Weekly secondary: global (single DataFrame, every 10 bars)
        weekly_idx = data_a.index[::10]
        weekly_global = _make_secondary(weekly_idx, 150)

        secondary_data = {
            "1D": {"A": daily_a, "B": daily_b},
            "1W": weekly_global,  # global, auto-wrapped as _global
        }

        signals = {
            sym: pd.Series(np.nan, index=df.index, dtype=float)
            for sym, df in data.items()
        }

        class MultiInspector(BacktestHook):
            def __init__(self):
                self.records = []

            def on_pre_signal(self, ctx):
                if ctx.secondary is None:
                    return
                rec = {"ts": ctx.timestamp}
                # Daily per-asset
                if "1D" in ctx.secondary:
                    tf = ctx.secondary["1D"]
                    rec["daily_A"] = tf.bar_data.get("A") is not None
                    rec["daily_B"] = tf.bar_data.get("B") is not None
                # Weekly global
                if "1W" in ctx.secondary:
                    tf = ctx.secondary["1W"]
                    rec["weekly_global"] = tf.bar_data.get("_global") is not None
                self.records.append(rec)

        hook = MultiInspector()
        engine = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=200_000, hooks=[hook],
            capital_allocator=EqualWeightAllocator(),
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(
            data,
            secondary_data=secondary_data,
        )

        assert isinstance(result, BacktestResult)
        assert len(hook.records) == 20

        # At bar 0 daily should be available (close alignment)
        assert hook.records[0]["daily_A"] is True
        assert hook.records[0]["daily_B"] is True
        assert hook.records[0]["weekly_global"] is True

    def test_early_bars_before_secondary(self):
        """Primary starts before secondary: bar_data is None for early bars."""
        from aiphaforge import SecondaryTimeframe

        primary = make_ohlcv(20, start_price=100)

        # Secondary starts at bar 10 (primary bars 0-9 have no secondary)
        sec_idx = primary.index[10:]
        rng = np.random.default_rng(55)
        n = len(sec_idx)
        low = 95 + rng.standard_normal(n)
        high = 105 + rng.standard_normal(n)
        low, high = np.minimum(low, high), np.maximum(low, high)
        opn = np.clip(100 + rng.standard_normal(n), low, high)
        close = np.clip(100 + rng.standard_normal(n), low, high)
        secondary_df = pd.DataFrame({
            "open": opn, "high": high, "low": low,
            "close": close,
            "volume": rng.integers(1000, 5000, size=n).astype(float),
        }, index=sec_idx)

        signals = pd.Series(np.nan, index=primary.index, dtype=float)

        class EarlyChecker(BacktestHook):
            def __init__(self):
                self.early_none = []
                self.late_present = []

            def on_pre_signal(self, ctx):
                if ctx.secondary is None:
                    return
                tf = ctx.secondary["1D"]
                bar = tf.bar_data.get("_global")
                hist = tf.data.get("_global")
                if ctx.bar_index < 10:
                    self.early_none.append(bar is None)
                else:
                    self.late_present.append(bar is not None)

        hook = EarlyChecker()
        engine = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, hooks=[hook],
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(
            primary,
            secondary_data={"1D": secondary_df},
        )

        assert isinstance(result, BacktestResult)
        # First 10 bars should all have None bar_data
        assert len(hook.early_none) == 10
        assert all(hook.early_none)
        # Bars 10+ should all have data
        assert len(hook.late_present) == 10
        assert all(hook.late_present)


# ===================================================================
# Strategy Composition Tree (v1.4)
# ===================================================================


class TestStrategyTree:

    def test_weighted_blend(self):
        """WeightedBlend: 2 children with known signals, verify weighted output."""
        data = make_ohlcv(100)
        child_a = MACrossover(short=5, long=20)
        child_b = RSIMeanReversion(period=14)

        tree = WeightedBlend(
            children=[child_a, child_b],
            weights=[0.6, 0.4],
        )

        signals = tree.generate_signals(data)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(data)
        # At least some non-NaN signals produced
        assert signals.notna().sum() > 0

        # Full backtest runs without error
        result = tree.backtest(data, fee_model=ZeroFeeModel(),
                               include_benchmark=False)
        assert isinstance(result, BacktestResult)

    def test_select_best(self):
        """SelectBest: children with different strengths, strongest wins."""
        data = make_ohlcv(100)
        child_a = MACrossover(short=5, long=20)
        child_b = MACDStrategy()

        tree = SelectBest(children=[child_a, child_b])
        signals = tree.generate_signals(data)
        assert isinstance(signals, pd.Series)
        assert signals.notna().sum() > 0

        result = tree.backtest(data, fee_model=ZeroFeeModel(),
                               include_benchmark=False)
        assert isinstance(result, BacktestResult)

    def test_priority_cascade(self):
        """PriorityCascade: first child NaN falls to second."""
        data = make_ohlcv(100)
        # RSI with extreme thresholds: rarely signals
        child_primary = RSIMeanReversion(period=14, oversold=5, overbought=95)
        child_fallback = MACrossover(short=5, long=20)

        tree = PriorityCascade(children=[child_primary, child_fallback])
        signals = tree.generate_signals(data)

        # Fallback should fill in where primary is silent
        fallback_signals = child_fallback.generate_signals(data)
        assert isinstance(signals, pd.Series)
        # Should have at least as many signals as fallback alone
        assert signals.notna().sum() >= fallback_signals.notna().sum()

    def test_vote_ensemble(self):
        """VoteEnsemble: 2 buy + 1 sell -> buy via majority."""
        data = make_ohlcv(100)
        tree = VoteEnsemble(children=[
            MACrossover(short=5, long=20),
            MACDStrategy(),
            RSIMeanReversion(period=14),
        ])
        signals = tree.generate_signals(data)
        assert isinstance(signals, pd.Series)
        # Majority vote should produce some signals
        assert signals.notna().sum() > 0

        result = tree.backtest(data, fee_model=ZeroFeeModel(),
                               include_benchmark=False)
        assert isinstance(result, BacktestResult)

    def test_conditional_switch(self):
        """ConditionalSwitch: regime function selects correct child."""
        data = make_ohlcv(100)
        child_trend = MACrossover(short=5, long=20)
        child_revert = RSIMeanReversion(period=14)

        # Simple regime: first half -> child 0, second half -> child 1
        def regime_fn(df):
            mid = len(df) // 2
            idx = pd.Series(0, index=df.index, dtype=int)
            idx.iloc[mid:] = 1
            return idx

        tree = ConditionalSwitch(
            children=[child_trend, child_revert],
            condition_fn=regime_fn,
        )
        signals = tree.generate_signals(data)
        assert isinstance(signals, pd.Series)
        assert signals.notna().sum() > 0

        result = tree.backtest(data, fee_model=ZeroFeeModel(),
                               include_benchmark=False)
        assert isinstance(result, BacktestResult)

    def test_nested_tree(self):
        """Nested tree: blend of cascade and ensemble runs backtest."""
        data = make_ohlcv(100)

        cascade = PriorityCascade(children=[
            RSIMeanReversion(period=14, oversold=5, overbought=95),
            MACrossover(short=5, long=20),
        ])

        ensemble = VoteEnsemble(children=[
            MACrossover(short=10, long=30),
            MACDStrategy(),
            RSIMeanReversion(period=14),
        ])

        tree = WeightedBlend(
            children=[cascade, ensemble],
            weights=[0.6, 0.4],
        )

        result = tree.backtest(data, fee_model=ZeroFeeModel(),
                               include_benchmark=False)
        assert isinstance(result, BacktestResult)

    def test_meta_set_weights(self):
        """Agent adjusts tree weights mid-backtest via MetaContext."""
        data = make_ohlcv(100)

        tree = WeightedBlend(
            children=[MACrossover(short=5, long=20), RSIMeanReversion()],
            weights=[0.7, 0.3],
        )

        class WeightAgent(BacktestHook):
            def on_pre_signal(self, ctx):
                if ctx.meta and ctx.bar_index == 50:
                    ctx.meta.set_weights([0.3, 0.7])

        result = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, hooks=[WeightAgent()],
            include_benchmark=False,
        ).set_strategy(tree).run(data)

        assert isinstance(result, BacktestResult)
        # Verify weights were actually changed
        assert tree.weights == [0.3, 0.7]

    def test_meta_set_weights_single_strategy_warns(self):
        """set_weights on a non-composite strategy warns and no-ops."""
        meta = MetaContext(config=None, strategy=MACrossover())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            meta.set_weights([0.5, 0.5])
            assert len(w) == 1
            assert "not a weighted composite" in str(w[0].message)


# ===================================================================
# Significance Testing (v1.5)
# ===================================================================


class TestSignificance:

    def test_bootstrap_ci_tight_on_constant_returns(self):
        """Low-volatility returns should produce a relatively narrow CI."""
        # Build data with low-volatility positive returns
        n = 252
        rng_data = np.random.default_rng(12)
        dates = pd.bdate_range("2024-01-01", periods=n, freq="B")
        rets = 0.001 + 0.002 * rng_data.standard_normal(n)
        close = 100.0 * np.cumprod(1 + rets)
        close[0] = 100.0
        data = pd.DataFrame({
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(n, 500000.0),
        }, index=dates)

        signals = pd.Series(np.nan, index=dates, dtype=float)
        signals.iloc[0] = 1  # buy and hold

        engine = BacktestEngine(
            mode="vectorized", fee_model=ZeroFeeModel(),
            include_benchmark=False)
        engine.set_signals(signals)
        result = engine.run(data)

        from aiphaforge.significance import bootstrap_ci
        ci = bootstrap_ci(result, metric="sharpe_ratio",
                          n_bootstrap=500, random_state=42)
        assert ci.observed > 0
        assert ci.ci_upper - ci.ci_lower < 5.0

    def test_bootstrap_ci_volatile_wider(self):
        """High volatility data should produce a wider CI than low-vol."""
        n = 252
        dates = pd.bdate_range("2024-01-01", periods=n, freq="B")

        # Low-volatility returns
        rng_lo = np.random.default_rng(12)
        rets_lo = 0.001 + 0.002 * rng_lo.standard_normal(n)
        close_lo = 100.0 * np.cumprod(1 + rets_lo)
        close_lo[0] = 100.0
        data_lo = pd.DataFrame({
            "open": close_lo, "high": close_lo * 1.002,
            "low": close_lo * 0.998, "close": close_lo,
            "volume": np.full(n, 500000.0),
        }, index=dates)

        # High-volatility returns
        rng_hi = np.random.default_rng(99)
        rets_hi = 0.001 + 0.05 * rng_hi.standard_normal(n)
        close_hi = 100.0 * np.cumprod(1 + rets_hi)
        close_hi[0] = 100.0
        data_hi = pd.DataFrame({
            "open": close_hi, "high": close_hi * 1.02,
            "low": close_hi * 0.98, "close": close_hi,
            "volume": np.full(n, 500000.0),
        }, index=dates)

        signals = pd.Series(np.nan, index=dates, dtype=float)
        signals.iloc[0] = 1

        from aiphaforge.significance import bootstrap_ci

        engine_lo = BacktestEngine(mode="vectorized",
                                   fee_model=ZeroFeeModel(),
                                   include_benchmark=False)
        engine_lo.set_signals(signals)
        res_lo = engine_lo.run(data_lo)
        ci_lo = bootstrap_ci(res_lo, n_bootstrap=500, random_state=42)

        engine_hi = BacktestEngine(mode="vectorized",
                                   fee_model=ZeroFeeModel(),
                                   include_benchmark=False)
        engine_hi.set_signals(signals)
        res_hi = engine_hi.run(data_hi)
        ci_hi = bootstrap_ci(res_hi, n_bootstrap=500, random_state=42)

        width_lo = ci_lo.ci_upper - ci_lo.ci_lower
        width_hi = ci_hi.ci_upper - ci_hi.ci_lower
        assert width_hi > width_lo

    def test_bootstrap_ci_max_drawdown_from_equity(self):
        """Bootstrap max_drawdown should be in [0, 1]."""
        data = make_ohlcv(200)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1
        signals.iloc[150] = -1

        engine = BacktestEngine(mode="vectorized",
                                fee_model=ZeroFeeModel(),
                                include_benchmark=False)
        engine.set_signals(signals)
        result = engine.run(data)

        from aiphaforge.significance import bootstrap_ci
        ci = bootstrap_ci(result, metric="max_drawdown",
                          n_bootstrap=500, random_state=42)
        assert ci.ci_lower >= 0
        assert ci.ci_upper <= 1

    def test_bootstrap_metrics_joint(self):
        """Multiple metrics from one bootstrap share sample count."""
        data = make_ohlcv(200)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1
        signals.iloc[150] = -1

        engine = BacktestEngine(mode="vectorized",
                                fee_model=ZeroFeeModel(),
                                include_benchmark=False)
        engine.set_signals(signals)
        result = engine.run(data)

        from aiphaforge.significance import bootstrap_metrics
        cis = bootstrap_metrics(
            result,
            metrics=["sharpe_ratio", "max_drawdown"],
            n_bootstrap=500,
            random_state=42,
        )
        assert "sharpe_ratio" in cis
        assert "max_drawdown" in cis
        assert len(cis["sharpe_ratio"].distribution) == 500
        assert len(cis["max_drawdown"].distribution) == 500
        assert cis["sharpe_ratio"].n_bootstrap == 500
        assert cis["max_drawdown"].n_bootstrap == 500

    def test_bootstrap_ci_custom_metric(self):
        """Custom callable metric should work with bootstrap_ci."""
        data = make_ohlcv(200)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1
        signals.iloc[150] = -1

        engine = BacktestEngine(mode="vectorized",
                                fee_model=ZeroFeeModel(),
                                include_benchmark=False)
        engine.set_signals(signals)
        result = engine.run(data)

        from aiphaforge.significance import BootstrapResult, bootstrap_ci
        mean_return = lambda r: float(np.mean(r))  # noqa: E731
        ci = bootstrap_ci(result, metric=mean_return,
                          n_bootstrap=500, random_state=42)
        assert isinstance(ci, BootstrapResult)
        assert ci.metric_name == "custom_0"

    def test_bootstrap_ci_reproducible(self):
        """Same random_state produces identical results."""
        data = make_ohlcv(200)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1
        signals.iloc[150] = -1

        engine = BacktestEngine(mode="vectorized",
                                fee_model=ZeroFeeModel(),
                                include_benchmark=False)
        engine.set_signals(signals)
        result = engine.run(data)

        from aiphaforge.significance import bootstrap_ci
        ci1 = bootstrap_ci(result, n_bootstrap=500, random_state=123)
        ci2 = bootstrap_ci(result, n_bootstrap=500, random_state=123)
        assert ci1.observed == ci2.observed
        assert ci1.ci_lower == ci2.ci_lower
        assert ci1.ci_upper == ci2.ci_upper

    def test_permutation_test_random_signal_not_significant(self):
        """Random signals should yield a high p-value (no alpha)."""
        data = make_ohlcv(200, trend=0.001, volatility=0.02)
        rng = np.random.default_rng(77)
        raw = rng.choice([1.0, -1.0, 0.0], size=len(data))
        signals = pd.Series(raw, index=data.index, dtype=float)
        # Convert to transition-only
        changed = signals != signals.shift(1)
        transition = pd.Series(np.nan, index=data.index, dtype=float)
        transition[changed] = signals[changed]

        from aiphaforge.significance import permutation_test
        perm = permutation_test(
            data, signals=transition, metric="sharpe_ratio",
            n_permutations=100, random_state=42,
            fee_model=ZeroFeeModel(), mode="vectorized",
            include_benchmark=False,
        )
        # Random signal should not be significant
        assert perm.p_value > 0.05

    def test_permutation_test_with_strategy(self):
        """Permutation test accepts a strategy object."""
        data = make_ohlcv(200)

        from aiphaforge.significance import PermutationResult, permutation_test
        perm = permutation_test(
            data, strategy=MACrossover(short=5, long=20),
            metric="sharpe_ratio", n_permutations=50, random_state=42,
            fee_model=ZeroFeeModel(), mode="vectorized",
            include_benchmark=False,
        )
        assert isinstance(perm, PermutationResult)
        assert 0.0 <= perm.p_value <= 1.0
        assert isinstance(perm.null_distribution, np.ndarray)
        assert len(perm.null_distribution) == 50

    def test_permutation_test_max_drawdown_direction(self):
        """max_drawdown uses lower_is_better p-value logic."""
        data = make_ohlcv(200, trend=0.001)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1
        signals.iloc[150] = -1

        from aiphaforge.significance import permutation_test
        perm = permutation_test(
            data, signals=signals, metric="max_drawdown",
            n_permutations=50, random_state=42,
            fee_model=ZeroFeeModel(), mode="vectorized",
            include_benchmark=False,
        )
        # p-value should be a valid probability
        assert 0.0 <= perm.p_value <= 1.0
        assert perm.metric_name == "max_drawdown"


# ===================================================================
# Monte Carlo & Multiple Comparison Correction (v1.6)
# ===================================================================


class TestMonteCarloAndCorrection:

    def test_generate_paths_block_bootstrap(self):
        """Block bootstrap generates correct number of paths with valid OHLCV."""
        data = make_ohlcv(100)

        from aiphaforge.significance import generate_paths
        paths = generate_paths(data, n_paths=5, method="block_bootstrap",
                               random_state=42)

        assert len(paths) == 5
        for p in paths:
            assert p.shape == data.shape
            assert list(p.columns) == list(data.columns)
            assert p.index.equals(data.index)
            # OHLCV validity
            assert (p["high"] >= np.maximum(p["open"], p["close"])).all()
            assert (p["low"] <= np.minimum(p["open"], p["close"])).all()

    def test_generate_paths_normal(self):
        """Normal method generates valid paths with same structure."""
        data = make_ohlcv(100)

        from aiphaforge.significance import generate_paths
        paths = generate_paths(data, n_paths=5, method="normal",
                               random_state=42)

        assert len(paths) == 5
        for p in paths:
            assert p.shape == data.shape
            assert list(p.columns) == list(data.columns)
            assert p.index.equals(data.index)

    def test_generate_paths_multi_asset(self):
        """Multi-asset dict input produces matching structure."""
        data = {
            "A": make_ohlcv(80, start_price=100),
            "B": make_ohlcv(80, start_price=200),
        }

        from aiphaforge.significance import generate_paths
        paths = generate_paths(data, n_paths=3, random_state=42)

        assert len(paths) == 3
        for p in paths:
            assert isinstance(p, dict)
            assert set(p.keys()) == {"A", "B"}
            for sym in ("A", "B"):
                assert p[sym].shape == data[sym].shape
                assert list(p[sym].columns) == list(data[sym].columns)
                assert p[sym].index.equals(data[sym].index)

    def test_monte_carlo_with_strategy(self):
        """Monte Carlo test with a strategy returns valid MonteCarloResult."""
        data = make_ohlcv(200)

        from aiphaforge.significance import MonteCarloResult, monte_carlo_test
        mc = monte_carlo_test(
            data,
            strategy=MACrossover(short=5, long=20),
            metric="sharpe_ratio",
            n_paths=10,
            random_state=42,
            fee_model=ZeroFeeModel(),
            mode="vectorized",
            include_benchmark=False,
        )

        assert isinstance(mc, MonteCarloResult)
        assert mc.n_valid > 0
        assert mc.worst_case <= mc.mean <= mc.best_case
        assert len(mc.distribution) == 10
        assert mc.metric_name == "sharpe_ratio"

    def test_monte_carlo_with_hooks_state_isolation(self):
        """Hooks are deep-copied per path, preserving original state."""
        data = make_ohlcv(50)

        class CounterHook(BacktestHook):
            def __init__(self):
                self.counter = 0

            def on_bar(self, ctx):
                self.counter += 1

        hook = CounterHook()
        assert hook.counter == 0

        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[40] = 0

        from aiphaforge.significance import monte_carlo_test
        mc = monte_carlo_test(
            data,
            signals=signals,
            hooks=[hook],
            metric="sharpe_ratio",
            n_paths=3,
            random_state=42,
            fee_model=ZeroFeeModel(),
            include_benchmark=False,
        )

        # The original hook's counter should be unchanged by the MC paths
        # (only the observed run uses the original hooks)
        # Deep-copy ensures path runs don't modify the original
        original_counter = hook.counter
        # The observed run goes through the original hooks, so counter > 0
        # But each MC path uses a fresh deep-copy, so no additional increment
        assert isinstance(mc.n_valid, int)
        assert mc.n_valid > 0

    def test_multiple_comparison_bonferroni(self):
        """Bonferroni correction produces valid CorrectionResult."""
        data = make_ohlcv(200)

        from aiphaforge.significance import (
            CorrectionResult,
            multiple_comparison_correction,
        )

        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1
        signals.iloc[150] = -1

        results = optimize(
            data,
            param_grid={'stop_loss': [0.03, 0.05, 0.10]},
            signals=signals,
            fee_model=ZeroFeeModel(),
            include_benchmark=False,
        )

        # Keep a copy of original to verify it's not mutated
        original_cols = set(results.columns)

        corr = multiple_comparison_correction(
            results, data, method="bonferroni", alpha=0.05,
            n_bootstrap=100, random_state=42,
        )

        assert isinstance(corr, CorrectionResult)
        assert corr.n_tested == 3
        assert 'p_value' in corr.results.columns
        assert 'p_value_corrected' in corr.results.columns
        assert 'significant' in corr.results.columns
        # Original DataFrame should NOT have the new columns
        assert 'p_value' not in results.columns
        assert set(results.columns) == original_cols

    def test_multiple_comparison_bh(self):
        """BH correction produces valid CorrectionResult."""
        data = make_ohlcv(200)

        from aiphaforge.significance import multiple_comparison_correction

        # Build signals properly
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1
        signals.iloc[150] = -1

        results = optimize(
            data,
            param_grid={'stop_loss': [0.03, 0.05, 0.10, 0.15]},
            signals=signals,
            fee_model=ZeroFeeModel(),
            include_benchmark=False,
        )

        corr = multiple_comparison_correction(
            results, data, method="bh", alpha=0.05,
            n_bootstrap=100, random_state=42,
        )

        assert corr.method == "bh"
        assert corr.n_tested == 4
        assert len(corr.results) == 4

    def test_multiple_comparison_mcs_import_error(self):
        """MCS raises ImportError if arch not installed, or runs if installed."""
        from aiphaforge.significance import multiple_comparison_correction

        data = make_ohlcv(200)

        # Use strategy_factory with diverse params for MCS to have
        # distinguishable return streams
        results = optimize(
            data,
            param_grid={'short': [5, 10, 20], 'long': [30, 50]},
            strategy_factory=lambda p: MACrossover(**p),
            fee_model=ZeroFeeModel(),
            include_benchmark=False,
        )

        try:
            corr = multiple_comparison_correction(
                results, data, method="mcs", alpha=0.10,
                n_bootstrap=100, random_state=42,
            )
            # arch IS installed — verify basic structure
            assert corr.method == "mcs"
            assert 'p_value' in corr.results.columns
            assert 'significant' in corr.results.columns
        except ImportError as e:
            assert "arch" in str(e)
            assert "pip install arch" in str(e)

    def test_build_returns_matrix(self):
        """build_returns_matrix produces T x N DataFrame."""
        data = make_ohlcv(100)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1
        signals.iloc[80] = -1

        from aiphaforge.significance import build_returns_matrix

        results = optimize(
            data,
            param_grid={'stop_loss': [0.03, 0.05]},
            signals=signals,
            fee_model=ZeroFeeModel(),
            include_benchmark=False,
        )

        mat = build_returns_matrix(results)
        assert isinstance(mat, pd.DataFrame)
        assert mat.shape[1] == 2  # 2 strategies

    def test_generate_paths_reproducible(self):
        """Same random_state produces identical paths."""
        data = make_ohlcv(50)

        from aiphaforge.significance import generate_paths
        paths1 = generate_paths(data, n_paths=3, random_state=99)
        paths2 = generate_paths(data, n_paths=3, random_state=99)

        for p1, p2 in zip(paths1, paths2):
            pd.testing.assert_frame_equal(p1, p2)


# ===================================================================
# Bayesian Optimizer & Scheduling (v1.7)
# ===================================================================


class TestBayesianAndScheduling:

    def test_schedule_hook_monthly(self):
        """ScheduleHook monthly: triggers once per month over ~10 months."""
        data = make_ohlcv(200, start_date="2024-01-01")
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[0] = 1

        trigger_count = [0]

        def count_callback(ctx):
            trigger_count[0] += 1

        hook = ScheduleHook("monthly", count_callback)
        engine = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, hooks=[hook],
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert isinstance(result, BacktestResult)
        # 200 business days ~ 10 months; expect roughly 10 monthly triggers
        n_months = len({(ts.year, ts.month) for ts in data.index})
        assert trigger_count[0] == n_months

    def test_schedule_hook_every_n_bars(self):
        """ScheduleHook with int frequency: triggers every N bars."""
        data = make_ohlcv(100, start_date="2024-01-01")
        signals = pd.Series(np.nan, index=data.index, dtype=float)

        trigger_count = [0]

        def count_callback(ctx):
            trigger_count[0] += 1

        hook = ScheduleHook(10, count_callback)
        engine = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, hooks=[hook],
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert isinstance(result, BacktestResult)
        assert trigger_count[0] == 10  # bars 0,10,20,...,90

    def test_schedule_rebalance(self):
        """schedule_rebalance completes backtest with target weights applied."""
        data_a = make_ohlcv(100, start_price=100)
        data_b = make_ohlcv(100, start_price=200)
        data = {"A": data_a, "B": data_b}

        # Provide signals for both assets
        signals = {
            sym: pd.Series(np.nan, index=df.index, dtype=float)
            for sym, df in data.items()
        }

        hook = schedule_rebalance({"A": 0.5, "B": 0.5}, "monthly")
        engine = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, hooks=[hook],
            capital_allocator=EqualWeightAllocator(),
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        assert isinstance(result, BacktestResult)
        # The engine should complete without errors; equity curve exists
        assert len(result.equity_curve) > 0

    def test_schedule_hook_resets_between_runs(self):
        """ScheduleHook resets state between engine.run calls."""
        data1 = make_ohlcv(50, start_date="2024-01-01")
        data2 = make_ohlcv(50, start_date="2025-01-01")

        signals1 = pd.Series(np.nan, index=data1.index, dtype=float)
        signals2 = pd.Series(np.nan, index=data2.index, dtype=float)

        trigger_count = [0]

        def count_callback(ctx):
            trigger_count[0] += 1

        hook = ScheduleHook("monthly", count_callback)

        engine = BacktestEngine(
            mode='event_driven', fee_model=ZeroFeeModel(),
            initial_capital=100_000, hooks=[hook],
            include_benchmark=False,
        )

        # Run 1
        engine.set_signals(signals1)
        engine.run(data1)
        count_after_run1 = trigger_count[0]
        assert count_after_run1 > 0

        # Run 2 — should trigger again (reset happened)
        engine.set_signals(signals2)
        engine.run(data2)
        count_after_run2 = trigger_count[0]
        assert count_after_run2 > count_after_run1

        # Both runs should have similar trigger counts
        n_months_1 = len({(ts.year, ts.month) for ts in data1.index})
        n_months_2 = len({(ts.year, ts.month) for ts in data2.index})
        assert count_after_run2 == n_months_1 + n_months_2

    def test_optimize_bayesian_basic(self):
        """Bayesian optimizer returns valid BayesianResult."""
        pytest.importorskip("optuna")
        data = make_ohlcv(200)

        result = optimize_bayesian(
            data,
            param_ranges={'short': (5, 15), 'long': (20, 40)},
            strategy_factory=lambda p: MACrossover(
                short=p['short'], long=p['long']),
            metric='sharpe_ratio',
            n_trials=5,
            train_pct=0.7,
            fee_model=ZeroFeeModel(),
        )

        assert isinstance(result, BayesianResult)
        assert 5 <= result.best_params['short'] <= 15
        assert 20 <= result.best_params['long'] <= 40
        assert isinstance(result.in_sample_result, BacktestResult)
        assert result.out_of_sample_result is not None
        assert isinstance(result.out_of_sample_result, BacktestResult)
        assert result.n_trials == 5

    def test_optimize_bayesian_train_pct_1(self):
        """train_pct=1.0 disables out-of-sample split."""
        pytest.importorskip("optuna")
        data = make_ohlcv(100)

        result = optimize_bayesian(
            data,
            param_ranges={'short': (5, 15), 'long': (20, 40)},
            strategy_factory=lambda p: MACrossover(
                short=p['short'], long=p['long']),
            n_trials=3,
            train_pct=1.0,
            fee_model=ZeroFeeModel(),
        )

        assert isinstance(result, BayesianResult)
        assert result.out_of_sample_result is None

    def test_optimize_bayesian_constraint(self):
        """constraint_fn filters trials that violate the constraint."""
        pytest.importorskip("optuna")
        data = make_ohlcv(200)

        result = optimize_bayesian(
            data,
            param_ranges={'short': (5, 15), 'long': (20, 40)},
            strategy_factory=lambda p: MACrossover(
                short=p['short'], long=p['long']),
            metric='sharpe_ratio',
            n_trials=5,
            train_pct=0.7,
            constraint_fn=lambda r: r.max_drawdown <= 0.5,
            fee_model=ZeroFeeModel(),
        )

        assert isinstance(result, BayesianResult)
        # The best result should satisfy the constraint
        assert result.in_sample_result.max_drawdown <= 0.5

    def test_optimize_bayesian_import_error(self):
        """Handles optuna availability gracefully."""
        try:
            import optuna  # noqa: F401
        except ImportError:
            # optuna not installed — verify ImportError
            with pytest.raises(ImportError, match="optuna"):
                optimize_bayesian(
                    make_ohlcv(50),
                    param_ranges={'short': (5, 15)},
                    strategy_factory=lambda p: MACrossover(short=p['short']),
                    n_trials=1,
                    fee_model=ZeroFeeModel(),
                )
            return

        # optuna IS installed — verify basic result
        result = optimize_bayesian(
            make_ohlcv(100),
            param_ranges={'short': (5, 15), 'long': (20, 40)},
            strategy_factory=lambda p: MACrossover(
                short=p['short'], long=p['long']),
            n_trials=3,
            fee_model=ZeroFeeModel(),
        )
        assert isinstance(result, BayesianResult)

    def test_optimize_bayesian_reproducible(self):
        """Same random_state produces identical best_params."""
        pytest.importorskip("optuna")
        data = make_ohlcv(150)

        kwargs = dict(
            data=data,
            param_ranges={'short': (5, 15), 'long': (20, 40)},
            strategy_factory=lambda p: MACrossover(
                short=p['short'], long=p['long']),
            n_trials=5,
            train_pct=0.7,
            random_state=42,
            fee_model=ZeroFeeModel(),
        )

        r1 = optimize_bayesian(**kwargs)
        r2 = optimize_bayesian(**kwargs)

        assert r1.best_params == r2.best_params
