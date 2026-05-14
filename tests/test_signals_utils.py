"""v2.3 Commit B — signals.py utilities tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.signals import (
    dict_to_signal_wide,
    transitions_only,
    validate_signal_series,
    validate_signal_wide,
    wide_to_signal_dict,
)

# ---------------------------------------------------------------------------
# transitions_only
# ---------------------------------------------------------------------------


class TestTransitionsOnly:
    def test_preserves_zero_flat_as_transition(self):
        # 0 (flatten) is a valid signal value — transitioning long → flat
        # must emit 0 on the transition bar.
        idx = pd.date_range("2024-01-01", periods=5)
        raw = pd.Series([1.0, 1.0, 0.0, 0.0, -1.0], index=idx)
        out = transitions_only(raw)
        # First bar emits because no prior state → state changes from
        # NaN to 1. Long → flat at idx[2]. Flat → short at idx[4].
        assert out.iloc[0] == 1.0
        assert pd.isna(out.iloc[1])
        assert out.iloc[2] == 0.0
        assert pd.isna(out.iloc[3])
        assert out.iloc[4] == -1.0

    def test_nan_means_hold_does_not_break_state(self):
        # NaN bars are "no instruction" and should ffill the prior
        # state — they do NOT count as a transition by themselves.
        idx = pd.date_range("2024-01-01", periods=5)
        raw = pd.Series([1.0, np.nan, 1.0, np.nan, -1.0], index=idx)
        out = transitions_only(raw)
        # bar 0: NaN→1 transition (long).
        # bar 1: NaN ffills to 1; state unchanged → NaN.
        # bar 2: explicit 1; state unchanged → NaN.
        # bar 3: NaN ffills to 1; state unchanged → NaN.
        # bar 4: 1→-1 transition (short).
        assert out.iloc[0] == 1.0
        assert pd.isna(out.iloc[1])
        assert pd.isna(out.iloc[2])
        assert pd.isna(out.iloc[3])
        assert out.iloc[4] == -1.0


# ---------------------------------------------------------------------------
# dict_to_signal_wide / wide_to_signal_dict round-trip
# ---------------------------------------------------------------------------


class TestLayoutAdapters:
    def test_dict_to_signal_wide_roundtrip(self):
        idx = pd.date_range("2024-01-01", periods=4)
        sig = {
            "A": pd.Series([1.0, np.nan, 0.0, -1.0], index=idx),
            "B": pd.Series([0.0, 1.0, np.nan, 0.0], index=idx),
        }
        wide = dict_to_signal_wide(sig)
        round_tripped = wide_to_signal_dict(wide)
        for sym in sig:
            pd.testing.assert_series_equal(
                round_tripped[sym], sig[sym], check_names=False,
            )

    def test_wide_to_signal_dict_aligns_to_data_index(self):
        # When data_dict supplied, each output Series must reindex
        # to that symbol's data index (longer or shorter than the
        # signal_wide index).
        sig_idx = pd.date_range("2024-01-01", periods=3)
        wide = pd.DataFrame(
            {"A": [1.0, 0.0, -1.0]}, index=sig_idx,
        )
        # Data extends 5 bars beyond the signal.
        data_idx = pd.date_range("2024-01-01", periods=5)
        data_dict = {"A": pd.DataFrame({"close": [100] * 5}, index=data_idx)}
        out = wide_to_signal_dict(wide, data_dict)
        assert len(out["A"]) == 5
        # First 3 bars match the signal; last 2 bars are NaN
        # (signal didn't cover them).
        assert out["A"].iloc[0] == 1.0
        assert out["A"].iloc[2] == -1.0
        assert pd.isna(out["A"].iloc[3])
        assert pd.isna(out["A"].iloc[4])

    def test_missing_symbol_returns_all_nan_series(self):
        # Symbol present in data_dict but absent from signal_wide
        # gets an all-NaN Series at the data's index.
        sig_idx = pd.date_range("2024-01-01", periods=3)
        wide = pd.DataFrame(
            {"A": [1.0, 0.0, -1.0]}, index=sig_idx,
        )
        data_dict = {
            "A": pd.DataFrame({"close": [100] * 3}, index=sig_idx),
            "B": pd.DataFrame({"close": [100] * 3}, index=sig_idx),
        }
        out = wide_to_signal_dict(wide, data_dict)
        assert "B" in out
        assert out["B"].isna().all()
        assert len(out["B"]) == 3

    def test_inf_values_replaced_with_nan(self):
        # Signal contract has no defined behavior for Inf; treat as
        # "hold" (NaN) rather than letting it leak into engine math.
        idx = pd.date_range("2024-01-01", periods=3)
        wide = pd.DataFrame(
            {"A": [1.0, np.inf, -np.inf]}, index=idx,
        )
        out = wide_to_signal_dict(wide)
        assert out["A"].iloc[0] == 1.0
        assert pd.isna(out["A"].iloc[1])
        assert pd.isna(out["A"].iloc[2])


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_validate_signal_series_rejects_duplicate_index(self):
        idx = pd.DatetimeIndex(
            ["2024-01-01", "2024-01-02", "2024-01-02"],
        )
        sig = pd.Series([1.0, 0.0, -1.0], index=idx)
        with pytest.raises(ValueError, match="duplicate"):
            validate_signal_series(sig)

    def test_validate_signal_series_rejects_non_datetime_index(self):
        sig = pd.Series([1.0, 0.0, -1.0], index=[0, 1, 2])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            validate_signal_series(sig)

    def test_validate_signal_series_strict_mode_rejects_fractional(self):
        idx = pd.date_range("2024-01-01", periods=3)
        sig = pd.Series([1.0, 0.5, -1.0], index=idx)
        with pytest.raises(ValueError, match="allow_fractional=False"):
            validate_signal_series(sig, allow_fractional=False)

    def test_validate_signal_series_allows_canonical_values(self):
        idx = pd.date_range("2024-01-01", periods=4)
        sig = pd.Series([1.0, 0.0, -1.0, np.nan], index=idx)
        # Should NOT raise under either mode.
        validate_signal_series(sig)
        validate_signal_series(sig, allow_fractional=False)

    def test_validate_signal_wide_rejects_non_numeric_column(self):
        idx = pd.date_range("2024-01-01", periods=3)
        wide = pd.DataFrame(
            {"A": [1.0, 0.0, -1.0], "B": ["long", "flat", "short"]},
            index=idx,
        )
        with pytest.raises(TypeError, match="numeric"):
            validate_signal_wide(wide)
