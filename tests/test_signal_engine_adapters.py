"""v2.3 Commit E — prepare_signals_for_engine routing tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.signals import prepare_signals_for_engine


def _ohlcv(periods: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods)
    return pd.DataFrame(
        {"close": [100.0] * periods}, index=idx,
    )


class TestPrepareSignalsRouting:
    def test_single_series(self):
        data = _ohlcv()
        sig = pd.Series([1.0, 0.0, -1.0, np.nan, 1.0], index=data.index)
        out = prepare_signals_for_engine(sig, data)
        assert isinstance(out, pd.Series)
        pd.testing.assert_series_equal(out, sig, check_names=False)

    def test_single_one_column_df(self):
        data = _ohlcv()
        sig = pd.DataFrame({"X": [1.0, 0.0, -1.0, np.nan, 1.0]}, index=data.index)
        out = prepare_signals_for_engine(sig, data)
        assert isinstance(out, pd.Series)
        assert list(out.values[:3]) == [1.0, 0.0, -1.0]

    def test_single_multi_column_df_raises(self):
        data = _ohlcv()
        sig = pd.DataFrame(
            {"A": [1.0] * 5, "B": [0.0] * 5}, index=data.index,
        )
        with pytest.raises(ValueError, match="ambiguous"):
            prepare_signals_for_engine(sig, data)

    def test_multi_wide_df(self):
        data_dict = {"A": _ohlcv(), "B": _ohlcv()}
        wide = pd.DataFrame(
            {"A": [1.0, 0.0, -1.0, np.nan, 1.0],
             "B": [0.0, 1.0, 0.0, -1.0, 0.0]},
            index=data_dict["A"].index,
        )
        out = prepare_signals_for_engine(wide, data_dict)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"A", "B"}
        assert list(out["A"].values[:3]) == [1.0, 0.0, -1.0]
        assert list(out["B"].values[:3]) == [0.0, 1.0, 0.0]

    def test_single_data_with_signal_dict_raises(self):
        # Symmetric to test_single_multi_column_df_raises — passing a
        # multi-symbol dict with single-asset data is ambiguous and
        # must raise. Caught by the routing's "single data + dict"
        # branch at signals.py:345-350.
        data = _ohlcv()
        sigs = {
            "A": pd.Series([1.0, 0.0, -1.0, np.nan, 1.0], index=data.index),
            "B": pd.Series([0.0, 1.0, 0.0, -1.0, 0.0], index=data.index),
        }
        with pytest.raises(ValueError, match="dict"):
            prepare_signals_for_engine(sigs, data)

    def test_multi_series_raises_without_broadcast(self):
        data_dict = {"A": _ohlcv(), "B": _ohlcv()}
        sig = pd.Series([1.0, 0.0, -1.0, np.nan, 1.0], index=data_dict["A"].index)
        with pytest.raises(ValueError, match="broadcast=True"):
            prepare_signals_for_engine(sig, data_dict)

    def test_multi_series_broadcasts_with_flag(self):
        # Global market-wide signal — same Series applied to every symbol.
        data_dict = {"A": _ohlcv(), "B": _ohlcv()}
        sig = pd.Series([1.0, 0.0, -1.0, np.nan, 1.0], index=data_dict["A"].index)
        out = prepare_signals_for_engine(sig, data_dict, broadcast=True)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"A", "B"}
        # Each symbol gets the same Series content.
        for sym in ["A", "B"]:
            assert list(out[sym].values[:3]) == [1.0, 0.0, -1.0]
            # Per-symbol reindex (values should align to that symbol's
            # data index, even though both indices match here).
            assert out[sym].index.equals(data_dict[sym].index)

    def test_preserves_nan_zero_semantics(self):
        # NaN must not be silently coerced to 0; 0 must not be coerced
        # to NaN. The full signal-contract semantics must round-trip.
        data = _ohlcv()
        sig = pd.Series([np.nan, 0.0, 1.0, 0.0, np.nan], index=data.index)
        out = prepare_signals_for_engine(sig, data)
        assert pd.isna(out.iloc[0])
        assert out.iloc[1] == 0.0
        assert out.iloc[2] == 1.0
        assert out.iloc[3] == 0.0
        assert pd.isna(out.iloc[4])
