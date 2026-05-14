"""v2.3 Commit G — target_weight_wide_to_schedule tests."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from aiphaforge import BacktestEngine
from aiphaforge.fees import ZeroFeeModel
from aiphaforge.signals import target_weight_wide_to_schedule


def _ohlcv(periods: int = 30, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, periods)))
    idx = pd.bdate_range(start, periods=periods)
    return pd.DataFrame(
        {"open": closes, "high": closes * 1.01,
         "low": closes * 0.99, "close": closes,
         "volume": [1e6] * periods},
        index=idx,
    )


class TestTargetWeightSchedule:
    def test_exact_only_aligned_dates_fire(self):
        data = {"A": _ohlcv(), "B": _ohlcv()}
        weights = pd.DataFrame(
            {"A": [0.5, 0.5], "B": [0.5, 0.5]},
            index=[data["A"].index[5], data["A"].index[15]],
        )
        out = target_weight_wide_to_schedule(weights, data, snap="exact")
        assert len(out) == 2
        assert all(isinstance(k, pd.Timestamp) for k in out)
        assert all(isinstance(v, dict) for v in out.values())

    def test_next_snap_pushes_to_next_valid_day(self):
        data = {"A": _ohlcv()}
        # 2024-01-06 is a Saturday — not a business day.
        weights = pd.DataFrame(
            {"A": [0.5]}, index=[pd.Timestamp("2024-01-06")],
        )
        out = target_weight_wide_to_schedule(weights, data, snap="next")
        # Snap to 2024-01-08 (Monday).
        assert pd.Timestamp("2024-01-08") in out

    def test_previous_snap_pushes_to_previous_valid_day(self):
        data = {"A": _ohlcv()}
        weights = pd.DataFrame(
            {"A": [0.5]}, index=[pd.Timestamp("2024-01-06")],
        )
        out = target_weight_wide_to_schedule(weights, data, snap="previous")
        # Snap back to 2024-01-05 (Friday).
        assert pd.Timestamp("2024-01-05") in out

    def test_missing_symbol_defaults_to_zero(self):
        data = {"A": _ohlcv(), "B": _ohlcv()}
        weights = pd.DataFrame(
            {"A": [0.7]}, index=[data["A"].index[5]],
        )
        out = target_weight_wide_to_schedule(weights, data, snap="exact")
        ts = data["A"].index[5]
        assert out[ts]["A"] == 0.7
        assert out[ts]["B"] == 0.0

    def test_union_default_keeps_staggered_listings(self):
        # Symbol B starts later than A. Under universe_alignment="union"
        # (the default), the early A-only rebalance date should still
        # fire — just with B at weight 0.
        a = _ohlcv()
        b = _ohlcv(start="2024-01-15")  # later listing
        data = {"A": a, "B": b}
        # Rebalance at A's day-5 — before B exists.
        weights = pd.DataFrame(
            {"A": [0.7, 0.3], "B": [0.0, 0.7]},
            index=[a.index[5], b.index[5]],
        )
        out = target_weight_wide_to_schedule(weights, data, snap="exact")
        # Under union, BOTH rebalance dates fire.
        assert a.index[5] in out
        assert b.index[5] in out
        # On the early date, B isn't listed yet → weight 0.
        assert out[a.index[5]]["B"] == 0.0

        # Under intersection, the early A-only rebalance is dropped
        # (a.index[5] is not in B's index).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            out_int = target_weight_wide_to_schedule(
                weights, data, snap="exact",
                universe_alignment="intersection",
            )
        assert a.index[5] not in out_int

    def test_collision_warns_and_keeps_last(self):
        # Two requested dates snap to the same target → last-write-wins
        # under default on_collision="warn".
        data = {"A": _ohlcv()}
        weights = pd.DataFrame(
            {"A": [0.3, 0.7]},
            index=[
                pd.Timestamp("2024-01-06"),  # Saturday → snaps to Monday
                pd.Timestamp("2024-01-07"),  # Sunday → also snaps to Monday
            ],
        )
        with pytest.warns(UserWarning, match="collision"):
            out = target_weight_wide_to_schedule(
                weights, data, snap="next",
                rebalance_dates=weights.index,
            )
        # Last-write-wins: 0.7 from the second requested date.
        assert out[pd.Timestamp("2024-01-08")]["A"] == 0.7

    def test_engine_can_run_set_target_weights_from_adapter(self):
        # End-to-end: build wide weights, convert via adapter, feed
        # to engine, run backtest.
        data = {"A": _ohlcv(), "B": _ohlcv()}
        weights = pd.DataFrame(
            {"A": [0.5, 0.0], "B": [0.5, 0.0]},
            index=[data["A"].index[5], data["A"].index[15]],
        )
        schedule = target_weight_wide_to_schedule(
            weights, data, snap="exact",
        )
        engine = BacktestEngine(mode="vectorized", fee_model=ZeroFeeModel())
        engine.set_target_weights(schedule)
        result = engine.run(data)
        assert result.equity_curve is not None
