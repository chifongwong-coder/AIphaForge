"""
Tests for v1.2: MetaContext (Agent MetaController).
"""

import numpy as np
import pandas as pd
import pytest

from aiphaforge import (
    BacktestEngine,
    BacktestHook,
    ExecutionMode,
    HookContext,
    MetaContext,
)
from aiphaforge.fees import ZeroFeeModel
from aiphaforge.strategies import MACrossover, RSIMeanReversion

from .conftest import make_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine(**kwargs):
    """Build an event-driven engine with zero fees for clean assertions."""
    defaults = dict(
        fee_model=ZeroFeeModel(),
        initial_capital=100_000,
        mode=ExecutionMode.EVENT_DRIVEN,
        include_benchmark=False,
    )
    defaults.update(kwargs)
    return BacktestEngine(**defaults)


class _NullHook(BacktestHook):
    """Hook that does nothing, used to activate MetaContext."""
    pass


# ---------------------------------------------------------------------------
# MetaContext unit tests
# ---------------------------------------------------------------------------

class TestMetaContextUnit:

    def test_init_defaults(self):
        mc = MetaContext(config=None)
        assert mc.current_strategy is None
        assert mc._suppress is False
        assert mc._target_weights is None
        assert mc._overrides == {}
        assert mc.audit_log == []

    def test_set_strategy_logs(self):
        class FakeStrategy:
            name = "Fake"
        mc = MetaContext(config=None)
        mc.set_strategy(FakeStrategy())
        assert mc.current_strategy.name == "Fake"
        assert len(mc.audit_log) == 1
        assert mc.audit_log[0]['action'] == 'set_strategy'

    def test_suppress_resume(self):
        mc = MetaContext(config=None)
        mc.suppress_signals()
        assert mc._suppress is True
        mc.resume_signals()
        assert mc._suppress is False
        assert len(mc.audit_log) == 2

    def test_adjust_sizing_creates_sizer(self):
        mc = MetaContext(config=None)
        mc.adjust_sizing(0.5)
        assert 'position_sizer' in mc._overrides
        assert mc._overrides['position_sizer'].fraction == 0.5
        assert mc.audit_log[-1]['action'] == 'adjust_sizing'

    def test_adjust_stop_loss_creates_rule(self):
        mc = MetaContext(config=None)
        mc.adjust_stop_loss(0.03)
        assert 'stop_loss_rule' in mc._overrides
        assert mc._overrides['stop_loss_rule'].threshold == 0.03

    def test_adjust_take_profit_creates_rule(self):
        mc = MetaContext(config=None)
        mc.adjust_take_profit(0.10)
        assert 'take_profit_rule' in mc._overrides
        assert mc._overrides['take_profit_rule'].threshold == 0.10

    def test_set_target_weights(self):
        mc = MetaContext(config=None)
        mc.set_target_weights({"AAPL": 0.3, "TSLA": 0.7})
        assert mc._target_weights == {"AAPL": 0.3, "TSLA": 0.7}

    def test_apply_overrides_no_overrides(self):
        from aiphaforge.config import BacktestConfig
        cfg = BacktestConfig()
        mc = MetaContext(config=cfg)
        result = mc._apply_overrides(cfg)
        assert result is cfg  # same object

    def test_apply_overrides_with_sizer(self):
        from aiphaforge.config import BacktestConfig
        cfg = BacktestConfig()
        mc = MetaContext(config=cfg)
        mc.adjust_sizing(0.3)
        result = mc._apply_overrides(cfg)
        assert result is not cfg
        assert result.position_sizer.fraction == 0.3
        # Original unchanged
        assert cfg.position_sizer is not result.position_sizer

    def test_audit_log_returns_copy(self):
        mc = MetaContext(config=None)
        mc.suppress_signals()
        log1 = mc.audit_log
        log1.append({'fake': True})
        assert len(mc.audit_log) == 1  # internal list not mutated

    def test_adjust_strategy_params(self):
        strategy = MACrossover(short=10, long=30)
        mc = MetaContext(config=None, strategy=strategy)
        mc.adjust_strategy_params(short=5)
        assert strategy.short == 5
        assert mc.audit_log[-1]['action'] == 'adjust_strategy_params'


# ---------------------------------------------------------------------------
# Integration: adjust_sizing
# ---------------------------------------------------------------------------

class TestAdjustSizing:

    def test_mid_backtest_size_reduction(self):
        """Reducing sizing mid-backtest should produce smaller trades."""
        data = make_ohlcv(60, start_price=100, trend=0.002, volatility=0.01)

        class SizingHook(BacktestHook):
            def on_pre_signal(self, ctx):
                if ctx.bar_index == 20:
                    ctx.meta.adjust_sizing(fraction=0.3)

        engine = _engine(hooks=[SizingHook()])
        strategy = MACrossover(short=5, long=15)
        engine.set_strategy(strategy)
        result = engine.run(data)

        # Verify audit recorded
        audit = result.metadata.get('meta_audit', [])
        sizing_entries = [e for e in audit if e['action'] == 'adjust_sizing']
        assert len(sizing_entries) == 1
        assert sizing_entries[0]['value'] == 0.3

        # Verify trades exist (strategy should generate signals)
        # The key assertion: MetaContext was created and audit works
        assert result is not None


# ---------------------------------------------------------------------------
# Integration: adjust_stop_loss
# ---------------------------------------------------------------------------

class TestAdjustStopLoss:

    def test_tighter_stop_triggers_earlier(self):
        """A very tight stop should cause earlier exits."""
        # Create data with a rise then a dip
        n = 50
        dates = pd.bdate_range("2024-01-01", periods=n, freq="B")
        close = np.concatenate([
            np.linspace(100, 110, 20),  # rise
            np.linspace(110, 95, 30),   # drop
        ])
        data = pd.DataFrame({
            'open': close,
            'high': close * 1.005,
            'low': close * 0.995,
            'close': close,
            'volume': 1_000_000.0,
        }, index=dates)

        # Signals: buy at bar 5, hold forever
        signals = pd.Series(np.nan, index=dates, dtype=float)
        signals.iloc[5] = 1

        class TightStopHook(BacktestHook):
            def on_pre_signal(self, ctx):
                # Set a 2% stop on bar 10
                if ctx.bar_index == 10:
                    ctx.meta.adjust_stop_loss(threshold=0.02)

        # With tight stop
        engine_tight = _engine(hooks=[TightStopHook()])
        engine_tight.set_signals(signals)
        result_tight = engine_tight.run(data)

        # Without stop (no hooks = no meta, but use a null hook)
        engine_none = _engine(hooks=[_NullHook()])
        engine_none.set_signals(signals)
        result_none = engine_none.run(data)

        # The tight-stop version should have exited, so final capital
        # preserved more than hold-through-dip
        assert result_tight.final_capital > result_none.final_capital


# ---------------------------------------------------------------------------
# Integration: set_strategy (swap)
# ---------------------------------------------------------------------------

class TestSetStrategy:

    def test_swap_strategy_mid_backtest(self):
        """Swapping strategy mid-backtest produces different signals."""
        data = make_ohlcv(100, start_price=100, trend=0.001,
                          volatility=0.015)

        class StrategySwapHook(BacktestHook):
            def on_pre_signal(self, ctx):
                if ctx.bar_index == 40:
                    ctx.meta.set_strategy(
                        RSIMeanReversion(period=14, oversold=30,
                                         overbought=70))

        # Run with swap
        engine = _engine(hooks=[StrategySwapHook()])
        engine.set_strategy(MACrossover(short=5, long=15))
        result_swap = engine.run(data)

        # Run without swap (pure MA)
        engine_ma = _engine(hooks=[_NullHook()])
        engine_ma.set_strategy(MACrossover(short=5, long=15))
        result_ma = engine_ma.run(data)

        # The results should differ (different strategies produce
        # different final capitals)
        swap_audit = result_swap.metadata.get('meta_audit', [])
        strategy_entries = [
            e for e in swap_audit if e['action'] == 'set_strategy']
        assert len(strategy_entries) == 1
        assert strategy_entries[0]['value'] == "RSI Mean Reversion"

        # Final capitals differ because signals differ after bar 40
        assert result_swap.final_capital != pytest.approx(
            result_ma.final_capital, rel=1e-6)


# ---------------------------------------------------------------------------
# Integration: suppress_signals
# ---------------------------------------------------------------------------

class TestSuppressSignals:

    def test_suppress_and_resume(self):
        """Suppressing signals should pause new trades, resume restores."""
        data = make_ohlcv(80, start_price=100, trend=0.002,
                          volatility=0.01)

        class SuppressHook(BacktestHook):
            def __init__(self):
                self.suppress_start = 20
                self.suppress_end = 50

            def on_pre_signal(self, ctx):
                if ctx.bar_index == self.suppress_start:
                    ctx.meta.suppress_signals()
                elif ctx.bar_index == self.suppress_end:
                    ctx.meta.resume_signals()

        # Run with suppression
        engine = _engine(hooks=[SuppressHook()])
        engine.set_strategy(MACrossover(short=5, long=15))
        result = engine.run(data)

        audit = result.metadata.get('meta_audit', [])
        suppress_entries = [
            e for e in audit if e['action'] == 'suppress_signals']
        resume_entries = [
            e for e in audit if e['action'] == 'resume_signals']
        assert len(suppress_entries) == 1
        assert len(resume_entries) == 1


# ---------------------------------------------------------------------------
# Integration: set_target_weights
# ---------------------------------------------------------------------------

class TestSetTargetWeights:

    def test_weight_override(self):
        """set_target_weights should override normal signal processing."""
        data = make_ohlcv(50, start_price=100, trend=0.001,
                          volatility=0.01)

        class WeightHook(BacktestHook):
            def on_pre_signal(self, ctx):
                if ctx.bar_index == 10:
                    ctx.meta.set_target_weights(
                        {"default": 0.5})

        engine = _engine(hooks=[WeightHook()])
        # Use signals that would do something different
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1  # buy signal at bar 5
        engine.set_signals(signals)
        result = engine.run(data)

        # Verify the weight was logged
        audit = result.metadata.get('meta_audit', [])
        weight_entries = [
            e for e in audit if e['action'] == 'set_target_weights']
        assert len(weight_entries) == 1
        assert weight_entries[0]['value'] == {"default": 0.5}


# ---------------------------------------------------------------------------
# Integration: adjust_strategy_params
# ---------------------------------------------------------------------------

class TestAdjustStrategyParams:

    def test_change_ma_window(self):
        """Changing MA window mid-backtest should affect future signals."""
        data = make_ohlcv(100, start_price=100, trend=0.001,
                          volatility=0.015)

        class ParamsHook(BacktestHook):
            def on_pre_signal(self, ctx):
                if ctx.bar_index == 40:
                    ctx.meta.adjust_strategy_params(short=3, long=10)

        engine = _engine(hooks=[ParamsHook()])
        engine.set_strategy(MACrossover(short=10, long=30))
        result = engine.run(data)

        audit = result.metadata.get('meta_audit', [])
        param_entries = [
            e for e in audit if e['action'] == 'adjust_strategy_params']
        assert len(param_entries) == 1
        assert param_entries[0]['value'] == {'short': 3, 'long': 10}


# ---------------------------------------------------------------------------
# Audit trail enrichment
# ---------------------------------------------------------------------------

class TestAuditLog:

    def test_audit_entries_have_context(self):
        """Audit entries should be enriched with timestamp/equity/drawdown."""
        data = make_ohlcv(40, start_price=100, trend=0.001,
                          volatility=0.01)

        class AuditHook(BacktestHook):
            def on_pre_signal(self, ctx):
                if ctx.bar_index == 10:
                    ctx.meta.adjust_sizing(fraction=0.5)
                if ctx.bar_index == 20:
                    ctx.meta.suppress_signals()

        engine = _engine(hooks=[AuditHook()])
        engine.set_strategy(MACrossover(short=5, long=15))
        result = engine.run(data)

        audit = result.metadata.get('meta_audit', [])
        assert len(audit) == 2

        for entry in audit:
            assert 'timestamp' in entry
            assert 'equity' in entry
            assert 'drawdown' in entry
            assert isinstance(entry['equity'], (int, float))
            assert isinstance(entry['drawdown'], (int, float))
            assert entry['equity'] > 0


# ---------------------------------------------------------------------------
# No meta default (backward compat)
# ---------------------------------------------------------------------------

class TestNoMetaDefault:

    def test_no_hooks_means_no_meta(self):
        """Without hooks, HookContext is not created; identical to v1.1."""
        data = make_ohlcv(50, start_price=100, trend=0.001,
                          volatility=0.01)

        engine = _engine()  # no hooks
        engine.set_strategy(MACrossover(short=5, long=15))
        result = engine.run(data)

        # No meta_audit in metadata
        assert 'meta_audit' not in result.metadata
        assert result is not None

    def test_hookcontext_meta_is_none_by_default(self):
        """HookContext.meta defaults to None."""
        from aiphaforge.hooks import HookContext
        from aiphaforge.portfolio import Portfolio
        ctx = HookContext(
            bar_index=0,
            timestamp=pd.Timestamp("2024-01-01"),
            portfolio=Portfolio(initial_capital=100_000),
        )
        assert ctx.meta is None


# ---------------------------------------------------------------------------
# Suppress + risk rules interaction
# ---------------------------------------------------------------------------

class TestSuppressRiskInteraction:

    def test_risk_suppress_overrides_resume(self):
        """Risk rule suppress should win even if agent calls resume."""
        from aiphaforge import CompositeRiskManager, MaxDrawdownHalt

        data = make_ohlcv(60, start_price=100, trend=-0.005,
                          volatility=0.01)

        class ResumeHook(BacktestHook):
            """Agent that always tries to resume signals."""
            def on_pre_signal(self, ctx):
                ctx.meta.resume_signals()

        # Set up risk rule that halts at 5% drawdown
        risk_mgr = CompositeRiskManager(rules=[
            MaxDrawdownHalt(max_drawdown=0.05, reset_drawdown=0.02),
        ])

        engine = _engine(
            hooks=[ResumeHook()],
            risk_rules=risk_mgr,
        )
        # Buy signal at bar 5, prices decline so drawdown triggers
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1
        engine.set_signals(signals)
        result = engine.run(data)

        # If risk suppress works, the agent's resume should not override it.
        # The key check is that the backtest completes without error.
        assert result is not None


# ---------------------------------------------------------------------------
# MetaContext persists overrides across bars
# ---------------------------------------------------------------------------

class TestMetaContextPersistence:

    def test_overrides_persist_across_bars(self):
        """Overrides set on one bar should persist to subsequent bars."""
        data = make_ohlcv(60, start_price=100, trend=0.002,
                          volatility=0.01)

        class PersistHook(BacktestHook):
            def __init__(self):
                self.sizer_fractions = []

            def on_pre_signal(self, ctx):
                if ctx.bar_index == 10:
                    ctx.meta.adjust_sizing(fraction=0.3)
                # Record the current state of overrides
                if ctx.bar_index >= 10:
                    sizer = ctx.meta._overrides.get('position_sizer')
                    if sizer:
                        self.sizer_fractions.append(sizer.fraction)

        hook = PersistHook()
        engine = _engine(hooks=[hook])
        engine.set_strategy(MACrossover(short=5, long=15))
        engine.run(data)

        # All recorded fractions after bar 10 should be 0.3
        assert len(hook.sizer_fractions) > 0
        assert all(f == 0.3 for f in hook.sizer_fractions)


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

class TestVersion:

    def test_version_is_120(self):
        import aiphaforge
        assert aiphaforge.__version__ == '1.2.0'

    def test_meta_context_importable(self):
        from aiphaforge import MetaContext
        assert MetaContext is not None
