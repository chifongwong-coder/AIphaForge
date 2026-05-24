"""v2.8.4 M13 — README Quick Start sections import-parity tests.

For each of the three new Quick Start sub-sections (factor research /
hook-driven order submission / `knowledge_check`), verify that every
symbol named in the README example resolves to an importable Python
object.  Catches README rot when a symbol is renamed in code.
"""
from __future__ import annotations

import inspect

from aiphaforge.broker import Broker


def test_factor_quickstart_symbols_importable() -> None:
    """Factor research quickstart symbols must import non-None."""
    from aiphaforge.alpha.evaluator import AlphaScreener
    from aiphaforge.alpha.report import FactorReport
    from aiphaforge.factor_strategy import FactorRuleStrategy
    from aiphaforge.factors import BaseFactor

    for sym in (AlphaScreener, FactorReport, FactorRuleStrategy, BaseFactor):
        assert sym is not None, sym


def test_hook_quickstart_classes_have_documented_attrs() -> None:
    """Hook quickstart contract: BacktestHook has on_pre_signal but NOT
    submit_order; orders flow through ``context.broker.submit_order(...)``.

    Also pins ``Broker.submit_order`` signature so the README example
    is wire-compatible with the actual API.
    """
    from aiphaforge.hooks import BacktestHook

    assert hasattr(BacktestHook, "on_pre_signal"), (
        "BacktestHook missing on_pre_signal; README quickstart out of sync"
    )
    assert not hasattr(BacktestHook, "submit_order"), (
        "BacktestHook unexpectedly grew a submit_order method; "
        "README quickstart's `context.broker.submit_order` example may "
        "be out of date."
    )

    # Broker.submit_order must accept (order, timestamp=...) per the
    # README example.  Pin the parameter names.
    sig = inspect.signature(Broker.submit_order)
    params = list(sig.parameters.keys())
    assert params[0] == "self"
    assert params[1] == "order", params
    assert "timestamp" in params, params


def test_knowledge_quickstart_symbols_importable() -> None:
    """`knowledge_check` quickstart symbols must import non-None.

    All five names appear in the README example and are re-exported
    via ``aiphaforge.probes`` per the v2.8.4 plan §4 M13 site cites.
    """
    from aiphaforge.probes import (
        AttestedAnswers,
        KnowledgeCheckReport,
        KnowledgeProbe,
        knowledge_check,
    )

    for sym in (
        AttestedAnswers,
        KnowledgeCheckReport,
        KnowledgeProbe,
        knowledge_check,
    ):
        assert sym is not None, sym
