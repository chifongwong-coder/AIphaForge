"""v2.4 Commit A — SignalSpec / SignalFrame typed wrapper tests."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from aiphaforge.signals import SignalFrame, SignalSpec


class TestSignalSpec:
    def test_default_field_values(self):
        spec = SignalSpec()
        assert spec.kind == "direction"
        assert spec.flat_value == 0.0
        assert spec.long_value == 1.0
        assert spec.short_value == -1.0
        assert spec.transition_only is True
        assert spec.signal_shift == 0
        # NaN is not equal to itself; test via isnan.
        assert np.isnan(spec.hold_value)

    def test_signal_spec_is_frozen(self):
        spec = SignalSpec()
        with pytest.raises(FrozenInstanceError):
            spec.kind = "score"


class TestSignalFrame:
    def test_construction_with_defaults(self):
        idx = pd.date_range("2024-01-01", periods=3)
        sig = pd.Series([1.0, 0.0, -1.0], index=idx)
        spec = SignalSpec()
        frame = SignalFrame(values=sig, spec=spec)
        assert frame.source == "unknown"
        assert frame.spec.kind == "direction"
        assert isinstance(frame.metadata, MappingProxyType)

    def test_metadata_wrapped_in_mapping_proxy(self):
        idx = pd.date_range("2024-01-01", periods=3)
        sig = pd.Series([1.0, 0.0, -1.0], index=idx)
        frame = SignalFrame(
            values=sig, spec=SignalSpec(),
            metadata={"model_id": "abc", "git_sha": "deadbeef"},
        )
        # MappingProxyType is read-only; mutation raises TypeError.
        assert frame.metadata["model_id"] == "abc"
        with pytest.raises(TypeError):
            frame.metadata["x"] = 1  # type: ignore[index]

    def test_signal_frame_is_frozen(self):
        idx = pd.date_range("2024-01-01", periods=3)
        sig = pd.Series([1.0, 0.0, -1.0], index=idx)
        frame = SignalFrame(values=sig, spec=SignalSpec())
        with pytest.raises(FrozenInstanceError):
            frame.source = "ml-model-v2"


class TestToEngineInput:
    def _ohlcv(self, periods: int = 5) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=periods)
        return pd.DataFrame({"close": [100.0] * periods}, index=idx)

    def test_single_series_round_trip(self):
        data = self._ohlcv()
        sig = pd.Series([1.0, np.nan, 0.0, -1.0, np.nan], index=data.index)
        frame = SignalFrame(values=sig, spec=SignalSpec())
        out = frame.to_engine_input(data)
        assert isinstance(out, pd.Series)
        # NaN/0/±1 semantics preserved.
        assert out.iloc[0] == 1.0
        assert pd.isna(out.iloc[1])
        assert out.iloc[2] == 0.0

    def test_multi_dict_round_trip(self):
        data = {"A": self._ohlcv(), "B": self._ohlcv()}
        sigs = {
            "A": pd.Series([1.0] * 5, index=data["A"].index),
            "B": pd.Series([-1.0] * 5, index=data["B"].index),
        }
        frame = SignalFrame(values=sigs, spec=SignalSpec())
        out = frame.to_engine_input(data)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"A", "B"}

    def test_reusable_across_data_shapes(self):
        # Same SignalFrame can be used against multiple data shapes
        # because data is bound at to_engine_input call time, NOT
        # at SignalFrame construction.
        wide = pd.DataFrame(
            {"A": [1.0, 0.0], "B": [-1.0, 0.0]},
            index=pd.date_range("2024-01-01", periods=2),
        )
        frame = SignalFrame(values=wide, spec=SignalSpec())
        # Use 1: short multi-asset data.
        data1 = {
            "A": pd.DataFrame({"close": [100.0, 101.0]}, index=wide.index),
            "B": pd.DataFrame({"close": [200.0, 199.0]}, index=wide.index),
        }
        out1 = frame.to_engine_input(data1)
        assert "A" in out1 and "B" in out1
        # Use 2: longer multi-asset data — same frame, different index.
        long_idx = pd.date_range("2024-01-01", periods=10)
        data2 = {
            "A": pd.DataFrame({"close": [100.0] * 10}, index=long_idx),
            "B": pd.DataFrame({"close": [200.0] * 10}, index=long_idx),
        }
        out2 = frame.to_engine_input(data2)
        # Frame's signal only covers 2 bars; rest gets NaN via reindex.
        assert len(out2["A"]) == 10
        assert pd.isna(out2["A"].iloc[5])
