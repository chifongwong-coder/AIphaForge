"""v2.3 Commit J — public exports tests."""
from __future__ import annotations


class TestPublicExportsV2_3Importable:
    def test_signal_layer_names_at_top_level(self):
        import aiphaforge

        # Functions
        assert hasattr(aiphaforge, "transitions_only")
        assert hasattr(aiphaforge, "prepare_signals_for_engine")
        assert hasattr(aiphaforge, "dict_to_signal_wide")
        assert hasattr(aiphaforge, "wide_to_signal_dict")
        assert hasattr(aiphaforge, "target_weight_wide_to_schedule")
        # Rules
        assert hasattr(aiphaforge, "ThresholdScoreRule")
        assert hasattr(aiphaforge, "CrossSectionalQuantileRule")
        # Strategy wrapper
        assert hasattr(aiphaforge, "DirectSignalStrategy")

    def test_factor_and_diagnostic_apis_subpackage_only(self):
        # v2.3 plan §J: factor + diagnostic APIs are deliberately
        # NOT exported at top level until v2.4.
        import aiphaforge

        assert not hasattr(aiphaforge, "FactorSpec")
        assert not hasattr(aiphaforge, "FactorSet")
        assert not hasattr(aiphaforge, "assert_factor_no_lookahead")
        # But they ARE importable from their subpackages.
        from aiphaforge.diagnostics import (
            assert_factor_no_lookahead,
            assert_signal_no_lookahead,
        )
        from aiphaforge.factors import FactorSet, FactorSpec
        assert FactorSpec is not None
        assert FactorSet.empty().values == {}
        assert assert_factor_no_lookahead is not None
        assert assert_signal_no_lookahead is not None


class TestNoExistingExportsRemoved:
    def test_v2_2_2_top_level_exports_still_present(self):
        # Sample of v2.2.2-era top-level names that MUST remain
        # accessible — sentinel against accidental removal during
        # the v2.3 export shuffle.
        import aiphaforge

        # Note: legacy strategy classes (MACrossover, RSIMeanReversion)
        # are deliberately NOT top-level exports — users import them
        # via `from aiphaforge.strategies import MACrossover`. This
        # is unchanged from v2.2.x. The sentinel below pins names
        # that ARE top-level today.
        v2_2_2_anchors = [
            "BacktestEngine",
            "BacktestResult",
            "ZeroFeeModel",
            "Portfolio",
            "Position",
            "Order",
            "OrderType",
            "OrderStatus",
            "PerformanceAnalyzer",
            "analyze",
            "probabilistic_sharpe_ratio",
            "deflated_sharpe_ratio",
            "StrategyNode",
            "WeightedBlend",
        ]
        missing = [
            name for name in v2_2_2_anchors if not hasattr(aiphaforge, name)
        ]
        assert not missing, (
            f"v2.2.2 top-level exports went missing in v2.3: {missing}"
        )
