"""v2.1 M1 — tests for the TradingCalendar primitive.

Plan: ``docs/plans/v2.1-plan.md`` §10.1 (r3-final).

Predefined calendars are NOT exercised here — those land in M2 with
their own test file (`tests/test_calendars_predefined.py`). M1 ships
only the core type plus exception classes, so tests construct
calendars inline.
"""
from __future__ import annotations

import warnings

import pandas as pd
import pytest

from aiphaforge.calendars import (
    CalendarConflictError,
    CalendarProviderProtocolError,
    CalendarSnapCollisionError,
    CalendarSnapError,
    TradingCalendar,
)
from aiphaforge.probes.transforms import IntegrityCheckResult

# ---------- Fixtures ----------


def _us_like_calendar(
    *,
    name: str = "TestUS",
    holidays: list[str] | None = None,
    coverage_start: str | None = "2024-01-01",
    coverage_end: str | None = "2024-12-31",
) -> TradingCalendar:
    return TradingCalendar(
        name=name,
        weekend_days=frozenset({5, 6}),
        holidays=frozenset(
            pd.Timestamp(d) for d in (holidays or ["2024-12-25"])
        ),
        coverage_start=pd.Timestamp(coverage_start) if coverage_start else None,
        coverage_end=pd.Timestamp(coverage_end) if coverage_end else None,
    )


# ---------- Construction / normalization ----------


class TestConstruction:
    def test_basic_construction(self):
        cal = _us_like_calendar()
        assert cal.name == "TestUS"
        assert cal.weekend_days == frozenset({5, 6})
        assert pd.Timestamp("2024-12-25") in cal.holidays

    def test_normalises_tz_aware_holidays(self):
        # Construction should strip tz and normalise to date-only.
        cal = TradingCalendar(
            name="t",
            weekend_days=frozenset({5, 6}),
            holidays=frozenset({
                pd.Timestamp("2024-12-25 14:30", tz="America/New_York"),
            }),
        )
        # The stored holiday must be tz-naive and time-zero.
        h = next(iter(cal.holidays))
        assert h.tzinfo is None
        assert h.hour == 0 and h.minute == 0
        # Date is preserved (after tz round-trip to UTC + normalise).
        # 14:30 ET on 2024-12-25 is 19:30 UTC on the same date, so
        # date-normalised result is still 2024-12-25.
        assert h.date() == pd.Timestamp("2024-12-25").date()

    def test_normalises_time_bearing_holidays(self):
        cal = TradingCalendar(
            name="t",
            weekend_days=frozenset({5, 6}),
            holidays=frozenset({pd.Timestamp("2024-12-25 09:30")}),
        )
        h = next(iter(cal.holidays))
        assert h.hour == 0
        assert h.date() == pd.Timestamp("2024-12-25").date()

    def test_normalises_coverage_window(self):
        cal = TradingCalendar(
            name="t",
            weekend_days=frozenset({5, 6}),
            holidays=frozenset(),
            coverage_start=pd.Timestamp("2024-01-01 12:00"),
            coverage_end=pd.Timestamp("2024-12-31 23:59"),
        )
        assert cal.coverage_start.hour == 0
        assert cal.coverage_end.hour == 0

    def test_rejects_invalid_weekend_days(self):
        with pytest.raises(ValueError, match="out-of-range"):
            TradingCalendar(
                name="bad",
                weekend_days=frozenset({5, 6, 7}),  # 7 invalid
                holidays=frozenset(),
            )

    def test_no_implicit_weekend_default(self):
        # The dataclass requires weekend_days explicitly — there is
        # no implicit "{5, 6}" fallback.
        with pytest.raises(TypeError):
            TradingCalendar(  # type: ignore[call-arg]
                name="missing",
                holidays=frozenset(),
            )

    def test_frozen_attribute_assignment_rejected(self):
        cal = _us_like_calendar()
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            cal.name = "renamed"  # type: ignore[misc]


# ---------- is_trading_day ----------


class TestIsTradingDay:
    def test_weekday_non_holiday(self):
        cal = _us_like_calendar()
        # Monday 2024-01-08
        assert cal.is_trading_day(pd.Timestamp("2024-01-08")) is True

    def test_saturday_is_not_trading_day(self):
        cal = _us_like_calendar()
        assert cal.is_trading_day(pd.Timestamp("2024-01-06")) is False

    def test_sunday_is_not_trading_day(self):
        cal = _us_like_calendar()
        assert cal.is_trading_day(pd.Timestamp("2024-01-07")) is False

    def test_holiday_is_not_trading_day(self):
        cal = _us_like_calendar()
        assert cal.is_trading_day(pd.Timestamp("2024-12-25")) is False

    def test_crypto_calendar_accepts_every_weekend(self):
        crypto = TradingCalendar(
            name="Crypto",
            weekend_days=frozenset(),
            holidays=frozenset(),
        )
        assert crypto.is_trading_day(pd.Timestamp("2024-01-06")) is True
        assert crypto.is_trading_day(pd.Timestamp("2024-12-25")) is True


# ---------- next/prev_trading_day ----------


class TestNextPrevTradingDay:
    def test_next_skips_weekend(self):
        cal = _us_like_calendar()
        # Friday → Monday (skipping Sat/Sun).
        assert cal.next_trading_day(pd.Timestamp("2024-01-05")) == pd.Timestamp("2024-01-08")

    def test_next_skips_holiday(self):
        cal = _us_like_calendar()
        # Tuesday 2024-12-24 → Thursday 2024-12-26 (skipping the
        # holiday on 12-25).
        assert cal.next_trading_day(pd.Timestamp("2024-12-24")) == pd.Timestamp("2024-12-26")

    def test_prev_skips_weekend(self):
        cal = _us_like_calendar()
        # Monday → Friday (skipping Sun/Sat).
        assert cal.prev_trading_day(pd.Timestamp("2024-01-08")) == pd.Timestamp("2024-01-05")

    def test_prev_skips_holiday(self):
        cal = _us_like_calendar()
        # Thursday 2024-12-26 → Tuesday 2024-12-24.
        assert cal.prev_trading_day(pd.Timestamp("2024-12-26")) == pd.Timestamp("2024-12-24")


# ---------- snap ----------


class TestSnap:
    def test_trading_day_passes_through_in_all_directions(self):
        cal = _us_like_calendar()
        ts = pd.Timestamp("2024-01-08")  # Mon, not a holiday
        for direction in ("forward", "backward", "nearest", "error"):
            assert cal.snap(ts, direction) == ts

    def test_forward_snaps_holiday_to_next(self):
        cal = _us_like_calendar()
        assert cal.snap(pd.Timestamp("2024-12-25"), "forward") == pd.Timestamp("2024-12-26")

    def test_backward_snaps_holiday_to_prev(self):
        cal = _us_like_calendar()
        assert cal.snap(pd.Timestamp("2024-12-25"), "backward") == pd.Timestamp("2024-12-24")

    def test_nearest_picks_closer(self):
        # Saturday 2024-01-06: prev=Fri 01-05 (1 day), next=Mon
        # 01-08 (2 days). nearest → backward.
        cal = _us_like_calendar()
        assert cal.snap(pd.Timestamp("2024-01-06"), "nearest") == pd.Timestamp("2024-01-05")
        # Sunday 2024-01-07: prev=Fri 01-05 (2 days), next=Mon
        # 01-08 (1 day). nearest → forward.
        assert cal.snap(pd.Timestamp("2024-01-07"), "nearest") == pd.Timestamp("2024-01-08")

    def test_nearest_ties_resolve_forward(self):
        # Holiday on Wednesday 2024-12-25; gap to Tuesday 12-24 is 1
        # day, gap to Thursday 12-26 is also 1 day. Tie → forward.
        cal = _us_like_calendar()
        assert cal.snap(pd.Timestamp("2024-12-25"), "nearest") == pd.Timestamp("2024-12-26")

    def test_error_raises(self):
        cal = _us_like_calendar()
        with pytest.raises(CalendarSnapError, match="not a trading day"):
            cal.snap(pd.Timestamp("2024-12-25"), "error")

    def test_invalid_direction_raises(self):
        cal = _us_like_calendar()
        with pytest.raises(ValueError, match="snap direction"):
            cal.snap(pd.Timestamp("2024-12-25"), "sideways")  # type: ignore[arg-type]


# ---------- is_conformant ----------


class TestIsConformant:
    def test_clean_index_passes(self):
        cal = _us_like_calendar()
        # All weekdays in a holiday-free week.
        idx = pd.bdate_range("2024-01-08", periods=5)
        result = cal.is_conformant(idx)
        assert isinstance(result, IntegrityCheckResult)
        assert result.passed is True
        assert result.errors == []

    def test_holiday_in_index_flagged(self):
        cal = _us_like_calendar()
        # bdate_range includes 12-25 by default.
        idx = pd.bdate_range("2024-12-23", periods=5)
        result = cal.is_conformant(idx)
        assert result.passed is False
        assert len(result.errors) == 1
        assert "2024-12-25" in result.errors[0]

    def test_weekend_in_index_flagged(self):
        cal = _us_like_calendar()
        # Custom index that includes Saturday.
        idx = pd.DatetimeIndex([
            "2024-01-05", "2024-01-06", "2024-01-08",
        ])
        result = cal.is_conformant(idx)
        assert result.passed is False
        assert "2024-01-06" in result.errors[0]

    def test_error_string_truncates_to_10_dates(self):
        # Fabricate an index with 50 holidays + extra context.
        cal = TradingCalendar(
            name="manyholidays",
            weekend_days=frozenset({5, 6}),
            holidays=frozenset(
                pd.Timestamp(f"2024-{m:02d}-15")
                for m in range(1, 13)
            ),
            coverage_start=pd.Timestamp("2024-01-01"),
            coverage_end=pd.Timestamp("2024-12-31"),
        )
        idx = pd.DatetimeIndex([
            pd.Timestamp(f"2024-{m:02d}-15") for m in range(1, 13)
        ])
        result = cal.is_conformant(idx)
        assert result.passed is False
        msg = result.errors[0]
        # 12 offending dates, capped at 10 with "and 2 more" suffix.
        assert "12 dates" in msg
        assert "and 2 more" in msg


# ---------- coverage / boundary warning ----------


class TestCoverageWarning:
    def test_in_range_query_does_not_warn(self):
        cal = _us_like_calendar()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.is_trading_day(pd.Timestamp("2024-06-15"))
        assert not any(
            "out of range" in str(w.message) for w in caught
        )

    def test_out_of_range_query_warns_once_per_instance(self):
        cal = _us_like_calendar()  # coverage 2024 only
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.is_trading_day(pd.Timestamp("2020-06-15"))
            cal.is_trading_day(pd.Timestamp("2020-07-15"))
            cal.is_trading_day(pd.Timestamp("2030-06-15"))
        out_of_range_warnings = [
            w for w in caught
            if "out of range" in str(w.message) or "weekday-only fallback" in str(w.message)
        ]
        # Exactly one warning despite three out-of-range queries.
        assert len(out_of_range_warnings) == 1

    def test_out_of_range_falls_back_to_weekday(self):
        cal = _us_like_calendar()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 2020-06-15 is a Monday, so weekday-only fallback says
            # it's a trading day even though the calendar has no
            # holiday data for 2020.
            assert cal.is_trading_day(pd.Timestamp("2020-06-15")) is True
            # 2020-06-13 is a Saturday; weekend rule still applies.
            assert cal.is_trading_day(pd.Timestamp("2020-06-13")) is False

    def test_user_calendar_without_coverage_skips_warning(self):
        cal = TradingCalendar(
            name="user",
            weekend_days=frozenset({5, 6}),
            holidays=frozenset(),
            coverage_start=None,
            coverage_end=None,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.is_trading_day(pd.Timestamp("1850-06-15"))
        assert not any(
            "out of range" in str(w.message) for w in caught
        )


# ---------- stable_fingerprint ----------


class TestStableFingerprint:
    def test_returns_tuple(self):
        cal = _us_like_calendar()
        fp = cal.stable_fingerprint()
        assert isinstance(fp, tuple)

    def test_two_separately_constructed_calendars_match(self):
        # Plan §5.2 (r3 final): calendar conflict detection uses
        # value-based equality, not object identity.
        cal_a = _us_like_calendar()
        cal_b = _us_like_calendar()
        assert cal_a is not cal_b
        assert cal_a.stable_fingerprint() == cal_b.stable_fingerprint()

    def test_different_holidays_diverge(self):
        cal_a = _us_like_calendar(holidays=["2024-12-25"])
        cal_b = _us_like_calendar(holidays=["2024-07-04"])
        assert cal_a.stable_fingerprint() != cal_b.stable_fingerprint()

    def test_different_weekend_days_diverge(self):
        cal_a = _us_like_calendar()
        cal_b = TradingCalendar(
            name="TestUS",
            weekend_days=frozenset({4, 5, 6}),  # 3-day weekend
            holidays=frozenset({pd.Timestamp("2024-12-25")}),
            coverage_start=pd.Timestamp("2024-01-01"),
            coverage_end=pd.Timestamp("2024-12-31"),
        )
        assert cal_a.stable_fingerprint() != cal_b.stable_fingerprint()

    def test_warning_state_does_not_affect_fingerprint(self):
        cal = _us_like_calendar()
        fp_before = cal.stable_fingerprint()
        # Trigger the warning to flip the seen-set.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal.is_trading_day(pd.Timestamp("2020-06-15"))
        assert cal._warning_keys_seen, "test setup: warning should have fired"
        assert cal.stable_fingerprint() == fp_before

    def test_holidays_order_invariant(self):
        # frozenset order is non-deterministic; fingerprint must be.
        cal_a = TradingCalendar(
            name="x",
            weekend_days=frozenset({5, 6}),
            holidays=frozenset({
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-12-25"),
            }),
        )
        cal_b = TradingCalendar(
            name="x",
            weekend_days=frozenset({5, 6}),
            holidays=frozenset({
                pd.Timestamp("2024-12-25"),
                pd.Timestamp("2024-01-01"),
            }),
        )
        assert cal_a.stable_fingerprint() == cal_b.stable_fingerprint()


# ---------- error class re-exports ----------


class TestErrorClasses:
    def test_calendar_snap_error_is_value_error(self):
        # Subclassing ValueError lets users catch with the broader
        # pandas/numpy idiom or specifically with the calendar one.
        assert issubclass(CalendarSnapError, ValueError)

    def test_collision_error_is_value_error(self):
        assert issubclass(CalendarSnapCollisionError, ValueError)

    def test_conflict_error_is_value_error(self):
        assert issubclass(CalendarConflictError, ValueError)

    def test_provider_protocol_error_is_type_error(self):
        # TypeError is the conventional choice for protocol-shape
        # violations (it's what duck-typing failures usually surface
        # as in stdlib).
        assert issubclass(CalendarProviderProtocolError, TypeError)
