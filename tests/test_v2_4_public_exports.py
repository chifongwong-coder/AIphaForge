"""v2.4 Commit P — public exports + FactorSet JSON helpers."""
from __future__ import annotations

import pandas as pd

from aiphaforge.factors import FactorSet, FactorSpec


class TestPublicExportsV2_4Importable:
    def test_factor_alpha_diagnostic_names_at_top_level(self):
        import aiphaforge

        # SignalSpec / SignalFrame (v2.3 deferred → v2.4)
        assert hasattr(aiphaforge, "SignalSpec")
        assert hasattr(aiphaforge, "SignalFrame")
        # Factor types
        assert hasattr(aiphaforge, "BaseFactor")
        assert hasattr(aiphaforge, "FactorSpec")
        assert hasattr(aiphaforge, "FactorSet")
        assert hasattr(aiphaforge, "FactorProvider")
        # Reference factor library
        assert hasattr(aiphaforge, "RSIFactor")
        assert hasattr(aiphaforge, "MomentumFactor")
        assert hasattr(aiphaforge, "MASpreadFactor")
        assert hasattr(aiphaforge, "VWAPDistanceFactor")
        assert hasattr(aiphaforge, "VolumeZScoreFactor")
        # Factor strategy bridge
        assert hasattr(aiphaforge, "FactorRuleStrategy")
        # Alpha layer
        assert hasattr(aiphaforge, "AlphaScreener")
        assert hasattr(aiphaforge, "AlphaScreenConfig")
        assert hasattr(aiphaforge, "FactorReport")
        assert hasattr(aiphaforge, "forward_returns")
        assert hasattr(aiphaforge, "ic")
        assert hasattr(aiphaforge, "rank_ic")
        assert hasattr(aiphaforge, "coverage")
        # Diagnostics
        assert hasattr(aiphaforge, "assert_factor_no_lookahead")
        assert hasattr(aiphaforge, "assert_signal_no_lookahead")

    def test_alpha_subpackage_imports_still_work(self):
        # Belt-and-suspenders: even after promotion to top level,
        # the subpackage import paths must still work for users who
        # already wrote `from aiphaforge.alpha import ...`.
        # Identity (same object across paths).
        import aiphaforge
        from aiphaforge.alpha import (
            AlphaScreenConfig as ASC_sub,
        )
        from aiphaforge.alpha import (
            AlphaScreener as AS_sub,
        )
        from aiphaforge.alpha.metrics import ic, rank_ic
        from aiphaforge.diagnostics import assert_factor_no_lookahead
        from aiphaforge.factors import FactorSpec
        assert aiphaforge.AlphaScreener is AS_sub
        assert aiphaforge.AlphaScreenConfig is ASC_sub
        assert aiphaforge.ic is ic
        assert aiphaforge.rank_ic is rank_ic
        assert aiphaforge.assert_factor_no_lookahead is assert_factor_no_lookahead
        assert aiphaforge.FactorSpec is FactorSpec

    def test_v2_3_top_level_exports_still_present(self):
        # Sentinel against accidental v2.3 export removal.
        import aiphaforge

        v2_3_anchors = [
            "transitions_only",
            "prepare_signals_for_engine",
            "ThresholdScoreRule",
            "CrossSectionalQuantileRule",
            "DirectSignalStrategy",
            "BacktestEngine",
            "Portfolio",
        ]
        missing = [
            n for n in v2_3_anchors if not hasattr(aiphaforge, n)
        ]
        assert not missing, f"v2.3 exports missing in v2.4: {missing}"


class TestFactorSetToJsonRoundTrip:
    def test_multi_asset_multi_factor_round_trip(self):
        idx = pd.date_range("2024-01-01", periods=4)
        rsi_df = pd.DataFrame(
            {"A": [30.0, 50.0, 70.0, 60.0],
             "B": [40.0, 55.0, 75.0, 65.0]},
            index=idx,
        )
        mom_df = pd.DataFrame(
            {"A": [-0.01, 0.02, 0.03, -0.01],
             "B": [0.01, 0.0, -0.02, 0.04]},
            index=idx,
        )
        fs = FactorSet(
            values={"rsi_14": rsi_df, "momentum_5": mom_df},
            specs={
                "rsi_14": FactorSpec(
                    name="rsi_14", description="RSI",
                    family="momentum", direction=-1,
                    required_columns=("close",), lookback=14,
                    is_primary=True, tags=("rsi", "oscillator"),
                ),
                "momentum_5": FactorSpec(
                    name="momentum_5", lookback=5,
                ),
            },
        )
        payload = fs.to_json()
        assert "version" in payload
        rebuilt = FactorSet.from_json(payload)
        assert set(rebuilt.values.keys()) == {"rsi_14", "momentum_5"}
        # Values round-trip exactly.
        for name in fs.values:
            pd.testing.assert_frame_equal(
                rebuilt.values[name], fs.values[name],
                check_freq=False,
            )
        # Spec field-level equality.
        for name, spec in fs.specs.items():
            r = rebuilt.specs[name]
            assert r.name == spec.name
            assert r.description == spec.description
            assert r.family == spec.family
            assert r.direction == spec.direction
            assert r.required_columns == spec.required_columns
            assert r.lookback == spec.lookback
            assert r.is_primary == spec.is_primary
            assert r.tags == spec.tags

    def test_empty_factor_set_round_trip(self):
        # Edge case per code-review v2: empty FactorSet must
        # round-trip cleanly. Schema regressions on the trivial
        # path get caught here.
        fs = FactorSet.empty()
        payload = fs.to_json()
        assert '"values": {}' in payload
        assert '"specs": {}' in payload
        rebuilt = FactorSet.from_json(payload)
        assert rebuilt.values == {}
        assert rebuilt.specs == {}
