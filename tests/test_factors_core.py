"""v2.3 Commit H — factors.py data-structure tests."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from aiphaforge.factors import (
    FactorProvider,
    FactorSet,
    FactorSpec,
    dict_to_factor_wide,
    validate_factor_wide,
)


class TestFactorSpec:
    def test_factor_spec_is_frozen(self):
        spec = FactorSpec(name="rsi_14", description="RSI", lookback=14)
        with pytest.raises(FrozenInstanceError):
            spec.name = "rsi_21"

    def test_factor_spec_default_fields(self):
        spec = FactorSpec(name="x")
        assert spec.description == ""
        assert spec.family is None
        assert spec.direction is None
        assert spec.required_columns == ()
        assert spec.lookback is None
        assert spec.is_primary is False
        assert spec.tags == ()

    def test_is_primary_field_reserved_for_v2_5(self):
        # Field exists, default False, ignored by v2.4
        # FactorRuleStrategy. v2.5+ multi-factor strategies will
        # use it. The test is here to pin the field's existence so
        # a future delete is detectable.
        spec = FactorSpec(name="x", is_primary=True)
        assert spec.is_primary is True


class TestFactorSet:
    def test_empty_is_valid_sentinel(self):
        empty = FactorSet.empty()
        assert empty.values == {}
        assert empty.specs == {}

    def test_holds_values_and_specs(self):
        idx = pd.date_range("2024-01-01", periods=3)
        values = {"momentum": pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)}
        specs = {"momentum": FactorSpec(name="momentum", lookback=20)}
        fset = FactorSet(values=values, specs=specs)
        assert "momentum" in fset.values
        assert fset.specs["momentum"].lookback == 20

    def test_factor_set_is_frozen(self):
        fset = FactorSet.empty()
        with pytest.raises(FrozenInstanceError):
            fset.values = {"a": 1}


class TestFactorProvider:
    def test_protocol_matches_duck_typed_object(self):
        class MyStrategy:
            def compute_factors(self, data):
                return FactorSet.empty()

        # No inheritance from FactorProvider — Protocol matches via
        # duck typing.
        s = MyStrategy()
        # isinstance against runtime-checkable Protocol requires
        # @runtime_checkable on the Protocol; we don't enable that to
        # keep imports light. Verify structural match instead.
        assert hasattr(s, "compute_factors")
        result = s.compute_factors(None)
        assert isinstance(result, FactorSet)
        # Confirm the Protocol exists for typing.
        assert FactorProvider is not None


class TestLayoutHelpers:
    def test_dict_to_factor_wide_roundtrip(self):
        idx = pd.date_range("2024-01-01", periods=3)
        d = {
            "A": pd.Series([1.0, 2.0, 3.0], index=idx),
            "B": pd.Series([4.0, 5.0, 6.0], index=idx),
        }
        wide = dict_to_factor_wide(d)
        assert list(wide.columns) == ["A", "B"]
        assert wide.loc[idx[1], "B"] == 5.0

    def test_validate_factor_wide_rejects_non_datetime_index(self):
        wide = pd.DataFrame({"A": [1.0, 2.0]}, index=[0, 1])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            validate_factor_wide(wide)
