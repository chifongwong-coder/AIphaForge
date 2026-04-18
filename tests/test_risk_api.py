"""Risk API consolidation tests (v1.9.6, B3).

CompositeRiskManager now inherits BaseRiskManager so it can be passed
via either ``BacktestEngine(risk_manager=...)`` or
``BacktestEngine(risk_rules=...)``.

Passing both at once raises ValueError to prevent silent precedence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge import BacktestEngine, MaxDrawdownHalt
from aiphaforge.risk import BaseRiskManager, CompositeRiskManager


def _make_data(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=pd.bdate_range("2024-01-01", periods=n),
    )


class TestCompositeIsBaseRiskManager:
    def test_inherits_base_class(self):
        crm = CompositeRiskManager(rules=[])
        assert isinstance(crm, BaseRiskManager)

    def test_initialize_no_op(self):
        crm = CompositeRiskManager(rules=[])
        crm.initialize(100_000.0)  # should not raise

    def test_sync_from_portfolio_no_op(self):
        crm = CompositeRiskManager(rules=[])
        crm.sync_from_portfolio(object())  # should not raise

    def test_calculate_position_size_passes_signal_through(self):
        crm = CompositeRiskManager(rules=[])
        size = crm.calculate_position_size(
            "AAA", signal=1, current_price=100.0,
            market_data=pd.DataFrame())
        assert size == 1.0


class TestEngineAcceptsBothRoutes:
    def test_risk_manager_with_composite_no_longer_crashes(self):
        """Pre-v1.9.6: BacktestEngine(risk_manager=CompositeRiskManager(...))
        crashed at __init__ because CompositeRiskManager had no initialize().
        """
        crm = CompositeRiskManager(rules=[MaxDrawdownHalt(max_drawdown=0.5)])
        eng = BacktestEngine(
            mode="event_driven",
            initial_capital=100_000.0,
            risk_manager=crm,
        )
        # If we got here, __init__ didn't blow up.
        data = _make_data()
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1.0
        eng.set_signals(signals)
        result = eng.run(data)
        # Result is a regular BacktestResult.
        assert hasattr(result, "equity_curve")

    def test_risk_rules_path_still_works(self):
        eng = BacktestEngine(
            mode="event_driven",
            initial_capital=100_000.0,
            risk_rules=CompositeRiskManager(
                rules=[MaxDrawdownHalt(max_drawdown=0.5)]),
        )
        data = _make_data()
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1.0
        eng.set_signals(signals)
        result = eng.run(data)
        assert hasattr(result, "equity_curve")


class TestEngineRejectsBothSimultaneously:
    def test_both_kwargs_raises(self):
        crm = CompositeRiskManager(rules=[MaxDrawdownHalt(max_drawdown=0.5)])
        with pytest.raises(ValueError, match="not both"):
            BacktestEngine(
                mode="event_driven",
                initial_capital=100_000.0,
                risk_manager=crm,
                risk_rules=CompositeRiskManager(
                    rules=[MaxDrawdownHalt(max_drawdown=0.3)]),
            )
