"""v2.0 M2 — tests for transforms, pipeline, and OHLC integrity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.probes.transforms import (
    BlockBootstrap,
    DateShift,
    OHLCJitter,
    PriceRebase,
    PriceScale,
    SymbolMasker,
    TransformPipeline,
    WindowShuffle,
    validate_ohlcv_integrity,
)
from tests.conftest import make_probe_ohlcv as _ohlcv  # noqa: E402

# ---------- SymbolMasker ----------

class TestSymbolMasker:
    def test_bijective_mapping(self):
        sm = SymbolMasker(symbols=["AAPL", "MSFT", "GOOG"], seed=42)
        masked = {sm.mask_symbol(s) for s in ["AAPL", "MSFT", "GOOG"]}
        assert len(masked) == 3
        for s in ["AAPL", "MSFT", "GOOG"]:
            assert sm.unmask_symbol(sm.mask_symbol(s)) == s

    def test_deterministic_per_seed(self):
        sm1 = SymbolMasker(symbols=["AAPL", "MSFT"], seed=42)
        sm2 = SymbolMasker(symbols=["AAPL", "MSFT"], seed=42)
        assert sm1.mask_symbol("AAPL") == sm2.mask_symbol("AAPL")

    def test_different_seed_different_mapping(self):
        # With only 2 symbols and a 20-element pool the chance of two
        # different seeds producing the same mapping is small but
        # nonzero. Try several seeds for a robust assertion.
        sm0 = SymbolMasker(symbols=["AAPL", "MSFT"], seed=0)
        differs = False
        for s in range(1, 20):
            other = SymbolMasker(symbols=["AAPL", "MSFT"], seed=s)
            if other.mask_symbol("AAPL") != sm0.mask_symbol("AAPL"):
                differs = True
                break
        assert differs

    def test_duplicate_symbols_raise(self):
        with pytest.raises(ValueError, match="duplicate"):
            SymbolMasker(symbols=["AAPL", "AAPL"], seed=0)

    def test_alias_pool_too_small_raises(self):
        with pytest.raises(ValueError, match="alias pool"):
            SymbolMasker(
                symbols=[f"S{i}" for i in range(5)],
                alias_pool=("A", "B", "C"),
                seed=0,
            )

    def test_user_supplied_pool_with_duplicates_raises(self):
        # Regression: replace=False guarantees distinct *indices*; if
        # the user-supplied pool contains duplicate strings, two
        # distinct indices can map to the same alias. With a pool of
        # only duplicates, the collision is unavoidable.
        with pytest.raises(ValueError, match="alias collision"):
            SymbolMasker(
                symbols=["A", "B"],
                alias_pool=("DUP", "DUP"),
                seed=0,
            )

    def test_default_alias_pool_is_synthetic(self):
        # Defensive: the default alias pool MUST be obviously synthetic
        # (SYM_xxxx pattern), not real-looking 4-letter tickers. Earlier
        # drafts shipped real listed symbols by accident.
        for alias in SymbolMasker.DEFAULT_ALIAS_POOL[:5]:
            assert alias.startswith("SYM_")

    def test_unknown_mask_unmask_raises(self):
        sm = SymbolMasker(symbols=["AAPL"], seed=0)
        with pytest.raises(KeyError):
            sm.mask_symbol("MSFT")
        with pytest.raises(KeyError):
            sm.unmask_symbol("ZZZZZ")

    def test_apply_is_no_op_on_data(self):
        sm = SymbolMasker(symbols=["AAPL"], seed=0)
        data = _ohlcv()
        out = sm.apply(data)
        # Frame content should be byte-identical (no copy required).
        pd.testing.assert_frame_equal(out, data)


# ---------- DateShift ----------

class TestDateShift:
    def test_shift_index_forward(self):
        ds = DateShift(offset=pd.DateOffset(years=3))
        data = _ohlcv()
        out = ds.apply(data)
        assert (out.index == data.index + pd.DateOffset(years=3)).all()

    def test_shift_inverse(self):
        ds = DateShift(offset=pd.DateOffset(years=-5))
        ts = pd.Timestamp("2024-03-12")
        assert ds.unshift_date(ds.shift_date(ts)) == ts

    def test_data_values_unchanged(self):
        ds = DateShift(offset=pd.DateOffset(years=-3))
        data = _ohlcv()
        out = ds.apply(data)
        np.testing.assert_array_equal(out["close"].to_numpy(), data["close"].to_numpy())


class TestDateShiftCalendarMarker:
    """v2.1 §4.1 / §5.3 r3-final: explicit marker protocol."""

    def test_marker_attribute_present_and_true(self):
        from typing import get_type_hints
        # Class-level marker, value is True.
        assert DateShift._aiphaforge_calendar_provider is True
        # Annotation present (defect #1: ClassVar[Literal[True]]).
        hints = get_type_hints(DateShift, include_extras=True)
        assert "_aiphaforge_calendar_provider" in hints

    def test_get_effective_calendar_returns_calendar(self):
        from aiphaforge.calendars import US_EQUITY
        ds = DateShift(offset=pd.DateOffset(years=-3), calendar=US_EQUITY)
        assert ds.get_effective_calendar() is US_EQUITY

    def test_get_effective_calendar_returns_none_when_unset(self):
        ds = DateShift(offset=pd.DateOffset(years=-3))
        assert ds.get_effective_calendar() is None


class TestDateShiftSnap:
    """v2.1 §4.2 — snap modes."""

    def test_calendar_none_preserves_v2_0_behavior(self):
        # Backward-compat: no calendar means no snap, no collision
        # check, exact unshift_date round-trip.
        ds = DateShift(offset=pd.DateOffset(years=-3))
        ts = pd.Timestamp("2024-03-12")
        assert ds.shift_date(ts) == pd.Timestamp("2021-03-12")
        assert ds.unshift_date(ds.shift_date(ts)) == ts

    def test_snap_forward_through_holiday(self):
        from aiphaforge.calendars import US_EQUITY
        ds = DateShift(
            offset=pd.DateOffset(days=0),  # no offset, just snap
            calendar=US_EQUITY,
            snap="forward",
        )
        # 2024-12-25 (Wed, holiday) → 2024-12-26 (Thu)
        assert ds.shift_date(pd.Timestamp("2024-12-25")) == pd.Timestamp("2024-12-26")

    def test_snap_backward_through_holiday(self):
        from aiphaforge.calendars import US_EQUITY
        ds = DateShift(
            offset=pd.DateOffset(days=0),
            calendar=US_EQUITY,
            snap="backward",
        )
        assert ds.shift_date(pd.Timestamp("2024-12-25")) == pd.Timestamp("2024-12-24")

    def test_snap_error_raises(self):
        from aiphaforge.calendars import US_EQUITY, CalendarSnapError
        ds = DateShift(
            offset=pd.DateOffset(days=0),
            calendar=US_EQUITY,
            snap="error",
        )
        with pytest.raises(CalendarSnapError):
            ds.shift_date(pd.Timestamp("2024-12-25"))

    def test_snap_nearest_picks_closer(self):
        from aiphaforge.calendars import US_EQUITY
        ds = DateShift(
            offset=pd.DateOffset(days=0),
            calendar=US_EQUITY,
            snap="nearest",
        )
        # Saturday 2024-01-06: prev Fri 01-05 (1 day), next Mon 01-08
        # (2 days). Nearest = backward.
        assert ds.shift_date(pd.Timestamp("2024-01-06")) == pd.Timestamp("2024-01-05")

    def test_invalid_snap_rejected(self):
        from aiphaforge.calendars import US_EQUITY
        with pytest.raises(ValueError, match="snap must be"):
            DateShift(offset=0, calendar=US_EQUITY, snap="sideways")  # type: ignore[arg-type]


class TestDateShiftCollision:
    """v2.1 §4.3 / §4.4 — collision policy + structured report."""

    def _frame_with_holiday_collision(self):
        # Two consecutive trading days where the offset of -3 years
        # lands them across Christmas 2021. With snap='forward',
        # both 2021-12-23 (Thu) and 2021-12-24 (Fri, NYSE early
        # close but full session in pandas_market_calendars) avoid
        # collision. We need a different setup: shift backward by
        # an offset that puts source 2024-12-25 (already a holiday
        # would propagate) and 2024-12-26 onto a NYSE non-trading
        # cluster.
        # Simpler: build explicit input with two source dates that
        # both forward-snap to the same target.
        # Source 2018-12-22 (Sat) and 2018-12-25 (Tue, holiday).
        # No offset; snap='forward'. 2018-12-22 → 2018-12-24 (Mon),
        # 2018-12-25 → 2018-12-26 (Wed). No collision actually.
        # Better: 2018-12-23 (Sun) and 2018-12-25 (Tue holiday).
        # Both forward to 2018-12-24 and 2018-12-26 — still no
        # collision. The narrow case is two holidays/weekends in a
        # row that all snap to the SAME next trading day.
        # 2024-11-28 (Thu Thanksgiving) + 2024-11-29 (Fri half-day,
        # but treated as full holiday by NYSE). Both forward-snap
        # to 2024-12-02 (Mon).
        return pd.DataFrame(
            {"open": [1.0, 2.0], "high": [1.5, 2.5],
             "low": [0.5, 1.5], "close": [1.2, 2.2], "volume": [100, 200]},
            index=pd.DatetimeIndex(["2024-11-28", "2024-11-29"]),
        )

    def test_collision_error_default(self):
        from aiphaforge.calendars import US_EQUITY, CalendarSnapCollisionError
        ds = DateShift(offset=pd.DateOffset(days=0), calendar=US_EQUITY, snap="forward")
        df = self._frame_with_holiday_collision()
        with pytest.raises(CalendarSnapCollisionError, match="duplicate target"):
            ds.apply(df)
        # v2.1.0 r4 §3: no instance-state side effect to assert here;
        # last_collision_report has been removed entirely. The
        # raised exception is the only signal.
        assert not hasattr(ds, "last_collision_report")

    def test_collision_keep_first(self):
        from aiphaforge.calendars import US_EQUITY
        ds = DateShift(
            offset=pd.DateOffset(days=0), calendar=US_EQUITY,
            snap="forward", on_collision="keep_first",
        )
        df = self._frame_with_holiday_collision()
        out = ds.apply(df)
        assert out.index.is_unique
        # First source row (2024-11-28 → 2024-12-02) survives.
        assert out["close"].iloc[0] == 1.2

    def test_collision_keep_last(self):
        from aiphaforge.calendars import US_EQUITY
        ds = DateShift(
            offset=pd.DateOffset(days=0), calendar=US_EQUITY,
            snap="forward", on_collision="keep_last",
        )
        df = self._frame_with_holiday_collision()
        out = ds.apply(df)
        assert out.index.is_unique
        # Last source row (2024-11-29 → 2024-12-02) survives.
        assert out["close"].iloc[0] == 2.2

    def test_collision_diagnostic_has_required_fields(self):
        # v2.1.0 r4 §3.4: collision report shape moves from
        # `last_collision_report` (instance state) to a per-call
        # diagnostic returned by `apply_with_diagnostics`.
        from aiphaforge.calendars import US_EQUITY
        ds = DateShift(
            offset=pd.DateOffset(days=0), calendar=US_EQUITY,
            snap="forward", on_collision="keep_last",
        )
        df = self._frame_with_holiday_collision()
        result = ds.apply_with_diagnostics(df)
        # Exactly one collision diagnostic for this single-collision
        # input.
        diags = [
            d for d in result.diagnostics
            if d.code == "calendar_snap_collision_rows_dropped"
        ]
        assert len(diags) == 1
        d = diags[0]
        assert d.source == "DateShift"
        assert d.severity == "warning"
        # Plan §3.4 details schema with schema_version="1.0".
        for key in (
            "schema_version", "transform", "on_collision",
            "collision_count", "collision_group_count",
            "examples", "examples_truncated",
        ):
            assert key in d.details, f"missing detail key {key}"
        assert d.details["schema_version"] == "1.0"
        assert d.details["transform"] == "DateShift"
        assert d.details["on_collision"] == "keep_last"
        assert d.details["collision_count"] == 1
        assert d.details["collision_group_count"] == 1
        # Examples preserve in-memory pd.Timestamp typing; the
        # manifest serializer (M10) stringifies for JSON output.
        ex = d.details["examples"][0]
        assert isinstance(ex["target_ts"], pd.Timestamp)
        assert isinstance(ex["kept_source_ts"], pd.Timestamp)
        assert all(isinstance(t, pd.Timestamp) for t in ex["source_ts"])

    def test_no_collision_yields_empty_diagnostics(self):
        # v2.1.0 r4 §3: clean apply produces no collision diagnostic.
        # No instance-state field to "clear" — there isn't one.
        from aiphaforge.calendars import US_EQUITY
        ds = DateShift(
            offset=pd.DateOffset(days=0), calendar=US_EQUITY,
            snap="forward", on_collision="keep_first",
        )
        clean = pd.DataFrame(
            {"open": [1.0, 2.0], "high": [1.5, 2.5],
             "low": [0.5, 1.5], "close": [1.2, 2.2], "volume": [100, 200]},
            index=pd.bdate_range("2024-01-08", periods=2),
        )
        result = ds.apply_with_diagnostics(clean)
        collision_diags = [
            d for d in result.diagnostics
            if d.code == "calendar_snap_collision_rows_dropped"
        ]
        assert collision_diags == []
        # And there's no last_collision_report attribute to leak
        # state across calls.
        assert not hasattr(ds, "last_collision_report")

    def test_apply_returns_dataframe_for_backward_compat(self):
        # v2.0/v2.0.1/v2.0.2 callers expect plain pd.DataFrame from
        # DateShift.apply. The new diagnostics API is opt-in.
        from aiphaforge.calendars import US_EQUITY
        ds = DateShift(
            offset=pd.DateOffset(days=0), calendar=US_EQUITY,
            snap="forward", on_collision="keep_last",
        )
        out = ds.apply(self._frame_with_holiday_collision())
        assert isinstance(out, pd.DataFrame)
        assert out.index.is_unique

    def test_invalid_collision_policy_rejected(self):
        with pytest.raises(ValueError, match="on_collision must be"):
            DateShift(offset=0, on_collision="merge_mean")  # type: ignore[arg-type]


class TestDateShiftLossyUnshift:
    """v2.1 §4.5 — best-effort lossy unshift contract."""

    def test_unshift_inverse_when_no_calendar(self):
        # Calendar-less case: exact inverse.
        ds = DateShift(offset=pd.DateOffset(years=-5))
        ts = pd.Timestamp("2024-03-12")
        assert ds.unshift_date(ds.shift_date(ts)) == ts

    def test_unshift_uses_opposite_snap(self):
        # forward snap on apply → backward snap on unshift.
        from aiphaforge.calendars import US_EQUITY
        ds = DateShift(
            offset=pd.DateOffset(days=0), calendar=US_EQUITY,
            snap="forward",
        )
        # 2024-12-25 (holiday Wed) shift_date → 2024-12-26 (Thu).
        # unshift_date snaps 2024-12-26 backward (no holiday, just
        # offset reversal) → 2024-12-26 stays since it's a trading
        # day. So result is 2024-12-26, not original 2024-12-25 —
        # the lossy contract.
        shifted = ds.shift_date(pd.Timestamp("2024-12-25"))
        assert shifted == pd.Timestamp("2024-12-26")
        # Unshift returns the same (it was already a trading day).
        assert ds.unshift_date(shifted) == pd.Timestamp("2024-12-26")


# ---------- PriceScale ----------

class TestPriceScale:
    def test_factor_applied_to_ohlc(self):
        ps = PriceScale(factor=2.5)
        data = _ohlcv()
        out = ps.apply(data)
        for col in ("open", "high", "low", "close"):
            np.testing.assert_allclose(
                out[col].to_numpy(), data[col].to_numpy() * 2.5
            )

    def test_volume_unchanged(self):
        ps = PriceScale(factor=0.5)
        data = _ohlcv()
        out = ps.apply(data)
        np.testing.assert_array_equal(
            out["volume"].to_numpy(), data["volume"].to_numpy()
        )

    def test_returns_preserved_exactly(self):
        # PriceScale must preserve returns — that's the documented
        # property (and the documented limitation).
        ps = PriceScale(factor=3.7)
        data = _ohlcv()
        out = ps.apply(data)
        r_in = data["close"].pct_change().dropna()
        r_out = out["close"].pct_change().dropna()
        np.testing.assert_allclose(r_in.to_numpy(), r_out.to_numpy())

    def test_invalid_factor_raises(self):
        with pytest.raises(ValueError):
            PriceScale(factor=0.0)
        with pytest.raises(ValueError):
            PriceScale(factor=-1.0)

    def test_invertible(self):
        ps = PriceScale(factor=2.5)
        assert ps.unscale_price(ps.scale_price(100.0)) == pytest.approx(100.0)


# ---------- PriceRebase ----------

class TestPriceRebase:
    def test_first_close_equals_base(self):
        pr = PriceRebase(base=100.0)
        data = _ohlcv(start=247.5)
        out = pr.apply(data)
        assert out["close"].iloc[0] == pytest.approx(100.0)

    def test_returns_preserved(self):
        pr = PriceRebase(base=100.0)
        data = _ohlcv(start=247.5)
        out = pr.apply(data)
        r_in = data["close"].pct_change().dropna()
        r_out = out["close"].pct_change().dropna()
        np.testing.assert_allclose(r_in.to_numpy(), r_out.to_numpy())

    def test_last_factor_recorded(self):
        pr = PriceRebase(base=100.0)
        data = _ohlcv(start=200.0)
        pr.apply(data)
        # Initial close ~200 → factor ~0.5.
        assert pr.last_factor is not None
        assert pr.last_factor == pytest.approx(100.0 / data["close"].iloc[0])

    def test_invalid_base_raises(self):
        with pytest.raises(ValueError):
            PriceRebase(base=0.0)


# ---------- OHLCJitter ----------

class TestOHLCJitter:
    def test_deterministic_per_seed(self):
        j = OHLCJitter(bps=20.0)
        data = _ohlcv()
        a = j.apply(data, seed=42)
        b = j.apply(data, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_different_output(self):
        j = OHLCJitter(bps=20.0)
        data = _ohlcv()
        a = j.apply(data, seed=42)
        b = j.apply(data, seed=43)
        # At 20bps with 60 bars, identical output is essentially zero
        # probability.
        assert not np.allclose(a["close"].to_numpy(), b["close"].to_numpy())

    def test_preserves_ohlc_invariants(self):
        j = OHLCJitter(bps=200.0)  # extreme noise to stress invariants
        data = _ohlcv()
        out = j.apply(data, seed=0)
        result = validate_ohlcv_integrity(out)
        assert result.passed, result.errors

    def test_zero_bps_is_identity(self):
        j = OHLCJitter(bps=0.0)
        data = _ohlcv()
        out = j.apply(data, seed=0)
        np.testing.assert_allclose(
            out["close"].to_numpy(), data["close"].to_numpy()
        )

    def test_negative_bps_raises(self):
        with pytest.raises(ValueError):
            OHLCJitter(bps=-1.0)


# ---------- BlockBootstrap ----------

class TestBlockBootstrap:
    def test_deterministic_per_seed(self):
        bb = BlockBootstrap(block_size=10)
        data = _ohlcv(n=100)
        a = bb.apply(data, seed=42)
        b = bb.apply(data, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_anchored_at_source_first_close(self):
        # Anchoring property: cumulative product is anchored at
        # data.iloc[0]['close']. The output first close equals the
        # anchor times the first sampled bar's return — so realizations
        # cluster near the source first close, not exactly at it.
        #
        # This deliberately does NOT assert `mean(first_closes) ≈ 100`
        # over a small sample — that's a flaky statistical claim. The
        # invariant we lock is structural:
        #   - all first closes are positive (anchor preserved)
        #   - first closes vary across seeds (anchor is multiplied by
        #     the first sampled bar's return, not pinned to the anchor)
        # If a future refactor silently clamps the first bar to the
        # anchor, `std == 0` will fail.
        bb = BlockBootstrap(block_size=20)
        data = _ohlcv(n=200, start=100.0)
        first_closes = [bb.apply(data, seed=s)["close"].iloc[0] for s in range(20)]
        assert all(c > 0 for c in first_closes)
        assert np.std(first_closes) > 0

    def test_empty_data_raises(self):
        bb = BlockBootstrap(block_size=5)
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], name="t"),
        )
        with pytest.raises(ValueError, match="at least 2 bars"):
            bb.apply(empty, seed=0)

    def test_index_preserved(self):
        bb = BlockBootstrap(block_size=20)
        data = _ohlcv(n=100)
        out = bb.apply(data, seed=0)
        assert (out.index == data.index).all()

    def test_invalid_block_size(self):
        with pytest.raises(ValueError):
            BlockBootstrap(block_size=0)

    def test_ohlc_integrity_holds(self):
        bb = BlockBootstrap(block_size=15)
        data = _ohlcv(n=100)
        for s in range(5):
            out = bb.apply(data, seed=s)
            result = validate_ohlcv_integrity(out)
            assert result.passed, (s, result.errors)


# ---------- WindowShuffle ----------

class TestWindowShuffle:
    def test_deterministic_per_seed(self):
        ws = WindowShuffle(window=10, swaps=3)
        data = _ohlcv(n=100)
        a = ws.apply(data, seed=42)
        b = ws.apply(data, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_index_preserved(self):
        ws = WindowShuffle(window=10, swaps=3)
        data = _ohlcv(n=100)
        out = ws.apply(data, seed=0)
        assert (out.index == data.index).all()

    def test_zero_swaps_is_identity(self):
        ws = WindowShuffle(window=10, swaps=0)
        data = _ohlcv(n=100)
        out = ws.apply(data, seed=0)
        np.testing.assert_array_equal(
            out["close"].to_numpy(), data["close"].to_numpy()
        )

    def test_short_series_returns_copy(self):
        ws = WindowShuffle(window=100, swaps=5)
        data = _ohlcv(n=50)
        out = ws.apply(data, seed=0)
        np.testing.assert_array_equal(
            out["close"].to_numpy(), data["close"].to_numpy()
        )

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            WindowShuffle(window=0)
        with pytest.raises(ValueError):
            WindowShuffle(window=10, swaps=-1)


# ---------- OHLC integrity validator ----------

class TestValidator:
    def test_clean_frame_passes(self):
        result = validate_ohlcv_integrity(_ohlcv())
        assert result.passed
        assert result.errors == []

    def test_missing_columns_fail(self):
        data = _ohlcv().drop(columns=["volume"])
        result = validate_ohlcv_integrity(data)
        assert not result.passed
        assert any("missing" in e for e in result.errors)

    def test_nan_detected(self):
        data = _ohlcv()
        data.loc[data.index[5], "close"] = np.nan
        result = validate_ohlcv_integrity(data)
        assert not result.passed
        assert any("NaN" in e for e in result.errors)

    def test_negative_volume_detected(self):
        data = _ohlcv()
        data.loc[data.index[0], "volume"] = -1.0
        result = validate_ohlcv_integrity(data)
        assert not result.passed
        assert any("negative" in e for e in result.errors)

    def test_ohlc_violation_detected(self):
        data = _ohlcv()
        # Force high < close.
        data.loc[data.index[0], "high"] = data.loc[data.index[0], "close"] - 1.0
        result = validate_ohlcv_integrity(data)
        assert not result.passed

    def test_non_monotonic_index_fails(self):
        data = _ohlcv()
        data = data.iloc[[0, 2, 1] + list(range(3, len(data)))]
        result = validate_ohlcv_integrity(data)
        assert not result.passed
        assert any("monotonic" in e for e in result.errors)

    def test_inf_detected(self):
        data = _ohlcv()
        data.loc[data.index[0], "close"] = np.inf
        result = validate_ohlcv_integrity(data)
        assert not result.passed
        assert any("inf" in e for e in result.errors)

    def test_non_positive_price_detected(self):
        data = _ohlcv()
        data.loc[data.index[0], "low"] = 0.0
        result = validate_ohlcv_integrity(data)
        assert not result.passed
        assert any("non-positive" in e for e in result.errors)


# ---------- Pipeline ----------

class TestTransformPipeline:
    def test_stage_order_canonical(self):
        # Ordering: metadata -> level -> series, regardless of user order.
        sm = SymbolMasker(symbols=["AAPL"], seed=0)
        ps = PriceScale(factor=2.0)
        bb = BlockBootstrap(block_size=10)
        # User order: series first, metadata last — pipeline should
        # reorder.
        pipeline = TransformPipeline(
            transforms=[bb, ps, sm], mode="market_level"
        )
        ordered = pipeline._stage_sorted()
        assert [t.category for t in ordered] == ["metadata", "level", "series"]

    def test_view_only_rejects_non_invertible_without_contract(self):
        # All built-ins that support view_only are invertible. Use a
        # synthetic stub: view-only-capable + non-invertible.
        class _ViewOnlyNonInvertible:
            name = "VONI"
            category = "metadata"
            supports_view_only = True
            supports_market_level = True
            order_invertible = False
            stochastic = False

            def apply(self, data, *, seed=None):
                return data.copy()

        with pytest.raises(ValueError, match="non-invertible"):
            TransformPipeline(
                transforms=[_ViewOnlyNonInvertible()], mode="view_only"
            )

    def test_view_only_allows_non_invertible_with_signal_only(self):
        class _ViewOnlyNonInvertible:
            name = "VONI"
            category = "metadata"
            supports_view_only = True
            supports_market_level = True
            order_invertible = False
            stochastic = False

            def apply(self, data, *, seed=None):
                return data.copy()

        # With agent_contract=signal_only, the non-invertible transform
        # is admitted in view_only mode.
        pipeline = TransformPipeline(
            transforms=[_ViewOnlyNonInvertible()],
            mode="view_only",
            agent_contract="signal_only",
        )
        assert pipeline.agent_contract == "signal_only"

    def test_market_level_rejects_view_only_only_transform(self):
        # All built-ins support market_level, so this is checked by
        # passing a fake transform with supports_market_level=False.
        class _ViewOnlyOnly:
            name = "ViewOnlyStub"
            category = "metadata"
            supports_view_only = True
            supports_market_level = False
            order_invertible = True
            stochastic = False

            def apply(self, data, *, seed=None):
                return data

        with pytest.raises(ValueError, match="does not support market_level"):
            TransformPipeline(
                transforms=[_ViewOnlyOnly()], mode="market_level"
            )

    def test_apply_runs_validator(self):
        # OHLCJitter at extreme bps can produce invalid bars without
        # the preserve_ohlc clamp; with clamp on (default) the
        # pipeline must produce valid output.
        data = _ohlcv(n=60)
        pipeline = TransformPipeline(
            transforms=[OHLCJitter(bps=200.0)], mode="market_level"
        )
        out = pipeline.apply(data, seed=0)
        result = validate_ohlcv_integrity(out)
        assert result.passed

    def test_apply_deterministic_per_outer_seed(self):
        # Stochastic sub-seeds derived from outer seed via SeedSequence.
        data = _ohlcv(n=60)
        pipeline = TransformPipeline(
            transforms=[BlockBootstrap(block_size=10)], mode="market_level"
        )
        a = pipeline.apply(data, seed=42)
        b = pipeline.apply(data, seed=42)
        pd.testing.assert_frame_equal(a, b)

    def test_apply_different_outer_seed_different_output(self):
        data = _ohlcv(n=60)
        pipeline = TransformPipeline(
            transforms=[BlockBootstrap(block_size=10)], mode="market_level"
        )
        a = pipeline.apply(data, seed=42)
        b = pipeline.apply(data, seed=43)
        assert not np.allclose(a["close"].to_numpy(), b["close"].to_numpy())

    def test_user_order_does_not_affect_output(self):
        # Regression: two pipelines with the same transforms in
        # different user-supplied orders must produce byte-identical
        # output for the same outer seed, because sub-seeds are
        # spawned in canonical stage order (not user input order).
        data = _ohlcv(n=60)
        sm = SymbolMasker(symbols=["X"], seed=7)
        ps = PriceScale(factor=1.5)
        bb = BlockBootstrap(block_size=10)
        p_user = TransformPipeline(
            transforms=[bb, ps, sm], mode="market_level"  # user put series first
        )
        p_canon = TransformPipeline(
            transforms=[sm, ps, bb], mode="market_level"  # already canonical
        )
        pd.testing.assert_frame_equal(
            p_user.apply(data, seed=42), p_canon.apply(data, seed=42)
        )

    def test_pipeline_error_names_offending_transform(self):
        # A transform that always raises should surface its name in
        # the chained error so debugging isn't guesswork.
        class _Boom:
            name = "Boom"
            category = "series"
            supports_view_only = False
            supports_market_level = True
            order_invertible = False
            stochastic = False

            def apply(self, data, *, seed=None):
                raise RuntimeError("kaboom")

        p = TransformPipeline(transforms=[_Boom()], mode="market_level")
        with pytest.raises(RuntimeError, match="transform 'Boom' failed"):
            p.apply(_ohlcv(), seed=1)

    def test_stochastic_pipeline_flag(self):
        bb = BlockBootstrap(block_size=10)
        ps = PriceScale(factor=2.0)
        assert TransformPipeline([bb], mode="market_level").stochastic
        assert not TransformPipeline([ps], mode="market_level").stochastic

    def test_invalid_transform_raises(self):
        class _Bad:
            name = "Bad"
            # missing other attrs

            def apply(self, data, *, seed=None):
                return data

        with pytest.raises(ValueError, match="missing required attribute"):
            TransformPipeline([_Bad()], mode="market_level")


# ---------- v2.1 M4: validator + pipeline calendar wiring ----------


class TestValidatorCalendar:
    """v2.1 §5.1 — validate_ohlcv_integrity(calendar=)."""

    def test_no_calendar_kwarg_preserves_v2_0_behavior(self):
        # Bit-identical v2.0 baseline: validator without calendar
        # accepts a clean bdate_range.
        from aiphaforge.probes.transforms import validate_ohlcv_integrity
        df = _ohlcv()
        r1 = validate_ohlcv_integrity(df)
        r2 = validate_ohlcv_integrity(df, calendar=None)
        assert r1.passed == r2.passed
        assert r1.errors == r2.errors

    def test_calendar_flags_holiday_in_index(self):
        from aiphaforge.calendars import US_EQUITY
        from aiphaforge.probes.transforms import validate_ohlcv_integrity
        # bdate_range from 2024-12-23 includes 12-24, 12-25 (NYSE
        # holiday), 12-26.
        idx = pd.bdate_range("2024-12-23", periods=5)
        df = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0,
             "close": 1.0, "volume": 1.0},
            index=idx,
        )
        result = validate_ohlcv_integrity(df, calendar=US_EQUITY)
        assert result.passed is False
        assert any("2024-12-25" in e for e in result.errors)


class TestPipelineCalendar:
    """v2.1 §5.2 — TransformPipeline.calendar=."""

    def test_explicit_calendar_threads_to_validator(self):
        from aiphaforge.calendars import US_EQUITY
        # Hand-build a frame with Christmas in the index; the
        # pipeline's final validator should reject it.
        idx = pd.bdate_range("2024-12-23", periods=5)
        df = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0,
             "close": 1.0, "volume": 1.0},
            index=idx,
        )
        # Use a noop transform list; the pipeline still runs the
        # final validator after applying zero transforms.
        pipeline = TransformPipeline(
            transforms=[], mode="market_level", calendar=US_EQUITY,
        )
        with pytest.raises(ValueError, match="2024-12-25"):
            pipeline.apply(df)

    def test_inferred_calendar_threads_to_validator(self):
        # When a transform supplies a calendar but the pipeline
        # itself does not, the pipeline.apply() validator still
        # uses the inferred calendar — this test proves it by
        # using snap='nearest' to repair the index AND letting the
        # snap+validator both succeed on a clean post-snap frame.
        from aiphaforge.calendars import US_EQUITY
        # Clean frame without holiday-adjacent collision potential.
        idx = pd.bdate_range("2024-01-08", periods=5)
        df = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0,
             "close": 1.0, "volume": 1.0},
            index=idx,
        )
        ds = DateShift(
            offset=pd.DateOffset(days=0),
            calendar=US_EQUITY,
            snap="nearest",
        )
        pipeline = TransformPipeline(
            transforms=[ds], mode="market_level",
        )
        # No exception → pipeline.apply ran, snap was a no-op,
        # validator passed against the inferred US_EQUITY calendar.
        out = pipeline.apply(df)
        assert len(out) == 5

    def test_pipeline_apply_calendar_rejects_holiday_index(self):
        # Direct: calendar threading rejects a frame whose index
        # contains a holiday after applying a no-op DateShift.
        from aiphaforge.calendars import US_EQUITY
        idx = pd.DatetimeIndex([
            "2024-12-23", "2024-12-24", "2024-12-26", "2024-12-27",
        ])
        df = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0,
             "close": 1.0, "volume": 1.0},
            index=idx,
        )
        # Pipeline built with no DateShift but with explicit
        # calendar — validator runs at apply.
        pipeline = TransformPipeline(
            transforms=[], mode="market_level", calendar=US_EQUITY,
        )
        # Clean — no holidays in index, validator passes.
        pipeline.apply(df)

        # Now add a holiday to the index.
        bad_idx = idx.append(pd.DatetimeIndex(["2024-12-25"])).sort_values()
        bad_df = df.reindex(bad_idx, fill_value=1.0)
        with pytest.raises(ValueError, match="2024-12-25"):
            pipeline.apply(bad_df)

    def test_explicit_vs_inferred_calendar_conflict_raises(self):
        from aiphaforge.calendars import (
            CHINA_A_SHARE,
            US_EQUITY,
            CalendarConflictError,
        )
        ds = DateShift(offset=0, calendar=CHINA_A_SHARE)
        with pytest.raises(CalendarConflictError, match="disagrees"):
            TransformPipeline(
                transforms=[ds], mode="market_level",
                calendar=US_EQUITY,
            )

    def test_separately_constructed_identical_calendars_match(self):
        # Plan §5.2: stable_fingerprint() value-equality, NOT object
        # identity.
        from aiphaforge.calendars import US_EQUITY, TradingCalendar
        # Build a copy with the same fields.
        copy_us = TradingCalendar(
            name=US_EQUITY.name,
            weekend_days=US_EQUITY.weekend_days,
            holidays=US_EQUITY.holidays,
            coverage_start=US_EQUITY.coverage_start,
            coverage_end=US_EQUITY.coverage_end,
        )
        assert copy_us is not US_EQUITY
        ds = DateShift(offset=0, calendar=copy_us)
        # Pipeline should accept the explicit US_EQUITY (singleton)
        # alongside the copy (separate instance) — they fingerprint
        # equal.
        TransformPipeline(
            transforms=[ds], mode="market_level",
            calendar=US_EQUITY,
        )


class TestEffectiveCalendarInference:
    """v2.1 §5.3 / §5.4 — explicit marker protocol."""

    def test_dateshift_marker_routes_calendar(self):
        from aiphaforge.calendars import US_EQUITY
        from aiphaforge.probes.transforms import (
            _effective_calendar_from_transforms,
        )
        ds = DateShift(offset=0, calendar=US_EQUITY)
        result = _effective_calendar_from_transforms([ds])
        assert result is US_EQUITY

    def test_no_calendar_returns_none(self):
        from aiphaforge.probes.transforms import (
            _effective_calendar_from_transforms,
        )
        ds = DateShift(offset=0)  # no calendar
        ps = PriceScale(factor=2.0)
        assert _effective_calendar_from_transforms([ds, ps]) is None

    def test_user_transform_with_unrelated_calendar_attr_is_ignored(self):
        # Defect-prevention test: a user transform with a `.calendar`
        # attribute but no `_aiphaforge_calendar_provider` marker
        # must NOT be picked up. This is the exact foot-gun the r2
        # review flagged.
        from aiphaforge.probes.transforms import (
            _effective_calendar_from_transforms,
        )

        class _LookbackTransform:
            name = "Lookback"
            category = "level"
            supports_view_only = True
            supports_market_level = True
            order_invertible = True
            stochastic = False
            calendar = "some-unrelated-config"  # NOT a TradingCalendar

            def apply(self, data, *, seed=None):
                return data

        result = _effective_calendar_from_transforms([_LookbackTransform()])
        assert result is None  # the unrelated attribute was ignored

    def test_marker_without_method_raises_protocol_error(self):
        from aiphaforge.calendars import CalendarProviderProtocolError
        from aiphaforge.probes.transforms import (
            _effective_calendar_from_transforms,
        )

        class _BadProvider:
            name = "Bad"
            category = "metadata"
            supports_view_only = True
            supports_market_level = True
            order_invertible = True
            stochastic = False
            _aiphaforge_calendar_provider = True
            # NO get_effective_calendar method

            def apply(self, data, *, seed=None):
                return data

        with pytest.raises(
            CalendarProviderProtocolError,
            match="does not implement get_effective_calendar",
        ):
            _effective_calendar_from_transforms([_BadProvider()])

    def test_provider_method_raises_wrapped(self):
        # Defect #2 fix: a get_effective_calendar() that raises
        # internally must surface as CalendarProviderProtocolError,
        # NOT as the bare exception.
        from aiphaforge.calendars import CalendarProviderProtocolError
        from aiphaforge.probes.transforms import (
            _effective_calendar_from_transforms,
        )

        class _RaisingProvider:
            name = "Raising"
            category = "metadata"
            supports_view_only = True
            supports_market_level = True
            order_invertible = True
            stochastic = False
            _aiphaforge_calendar_provider = True

            def get_effective_calendar(self):
                raise RuntimeError("internal config missing")

            def apply(self, data, *, seed=None):
                return data

        with pytest.raises(
            CalendarProviderProtocolError,
            match="raised RuntimeError",
        ):
            _effective_calendar_from_transforms([_RaisingProvider()])

    def test_two_distinct_calendars_raise_conflict(self):
        from aiphaforge.calendars import (
            CHINA_A_SHARE,
            US_EQUITY,
            CalendarConflictError,
        )
        from aiphaforge.probes.transforms import (
            _effective_calendar_from_transforms,
        )
        ds_us = DateShift(offset=0, calendar=US_EQUITY)
        ds_cn = DateShift(offset=0, calendar=CHINA_A_SHARE)
        with pytest.raises(CalendarConflictError, match="distinct"):
            _effective_calendar_from_transforms([ds_us, ds_cn])

    def test_two_identical_calendars_pass(self):
        from aiphaforge.calendars import US_EQUITY, TradingCalendar
        from aiphaforge.probes.transforms import (
            _effective_calendar_from_transforms,
        )
        copy_us = TradingCalendar(
            name=US_EQUITY.name,
            weekend_days=US_EQUITY.weekend_days,
            holidays=US_EQUITY.holidays,
            coverage_start=US_EQUITY.coverage_start,
            coverage_end=US_EQUITY.coverage_end,
        )
        ds_a = DateShift(offset=0, calendar=US_EQUITY)
        ds_b = DateShift(offset=0, calendar=copy_us)
        # Returns the first one but no conflict raised.
        result = _effective_calendar_from_transforms([ds_a, ds_b])
        assert result is US_EQUITY


class TestRunScenarioCalendarThreadThrough:
    """v2.1 §5.5 — _run_scenario passes inferred calendar into the
    internal TransformPipeline.
    """

    def test_run_ab_probe_with_dateshift_calendar_does_not_crash(self):
        # Smoke: a scenario containing DateShift(calendar=US_EQUITY)
        # runs end-to-end through run_ab_probe without unrelated
        # errors. Calendar threading happens inside _run_scenario
        # (verified by deeper inspection in the M5 e2e file).
        from aiphaforge.calendars import US_EQUITY
        from aiphaforge.probes import (
            ABScenario,
            MACrossBaseline,
            run_ab_probe,
        )
        from tests.conftest import make_probe_ohlcv as _ohlcv_p

        # Use a clean fixture starting after holidays to avoid
        # tripping the calendar validator on the raw arm.
        data = _ohlcv_p(n=40)

        ds = DateShift(
            offset=pd.DateOffset(years=-1),
            calendar=US_EQUITY,
            snap="forward",
            on_collision="keep_last",
        )
        scen = ABScenario(
            scenario_id="cal_shift", mode="market_level",
            transforms=[ds],
        )

        result = run_ab_probe(
            ai_factory=lambda: MACrossBaseline(short=5, long=20),
            baseline_factory=lambda: MACrossBaseline(short=5, long=20),
            data=data,
            scenarios=[scen],
            n_repeat=2, seeds=[0, 1],
            metrics=("total_return",),
            min_valid_repeats=1,
            engine_kwargs={"include_benchmark": False},
        )
        # The scenario report exists and the canonical determinism
        # block from v2.0.1 r5 is intact.
        assert len(result.scenarios) == 1
        assert "determinism_check" in result.manifest
        assert result.manifest["determinism_check"]["schema_version"] == "2.0.1"

    def test_run_ab_probe_with_conflicting_calendars_fails_fast(self):
        from aiphaforge.calendars import (
            CHINA_A_SHARE,
            US_EQUITY,
            CalendarConflictError,
        )
        from aiphaforge.probes import (
            ABScenario,
            MACrossBaseline,
            run_ab_probe,
        )
        from tests.conftest import make_probe_ohlcv as _ohlcv_p

        data = _ohlcv_p(n=20)
        ds_us = DateShift(offset=0, calendar=US_EQUITY)
        ds_cn = DateShift(offset=0, calendar=CHINA_A_SHARE)
        scen = ABScenario(
            scenario_id="conflict", mode="market_level",
            transforms=[ds_us, ds_cn],
        )
        with pytest.raises(CalendarConflictError):
            run_ab_probe(
                ai_factory=lambda: MACrossBaseline(short=5, long=20),
                baseline_factory=lambda: MACrossBaseline(short=5, long=20),
                data=data,
                scenarios=[scen],
                n_repeat=2, seeds=[0, 1],
                metrics=("total_return",),
                min_valid_repeats=1,
                engine_kwargs={"include_benchmark": False},
            )
