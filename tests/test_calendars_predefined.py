"""v2.1 M2 — tests for the four predefined trading calendars.

Plan: ``docs/plans/v2.1-plan.md`` §10.2 (r3-final).

These tests exercise the shipped JSON holiday data and the
provenance metadata. Holiday data is sourced offline from
``pandas_market_calendars`` (MIT) by ``scripts/generate_holidays_json.py``
— the upstream package is NOT a runtime dependency of AIphaForge.
"""
from __future__ import annotations

import json
from importlib import resources

import pandas as pd

from aiphaforge.calendars import (
    CHINA_A_SHARE,
    CRYPTO_24_7,
    US_EQUITY,
    US_FUTURES_ES,
    TradingCalendar,
)

# ---------- Package-root re-exports ----------


class TestPackageRootReExports:
    def test_all_four_predefined_at_root(self):
        # Plan §3.1: users should be able to write
        # `from aiphaforge.calendars import US_EQUITY` without
        # reaching into `predefined`.
        from aiphaforge import calendars
        for name in ("US_EQUITY", "CHINA_A_SHARE", "CRYPTO_24_7", "US_FUTURES_ES"):
            assert hasattr(calendars, name), f"missing root re-export {name}"

    def test_trading_calendar_at_root(self):
        from aiphaforge import calendars
        assert hasattr(calendars, "TradingCalendar")

    def test_predefined_are_trading_calendar_instances(self):
        for cal in (US_EQUITY, CHINA_A_SHARE, CRYPTO_24_7, US_FUTURES_ES):
            assert isinstance(cal, TradingCalendar)


# ---------- Coverage window + tz-naive holidays ----------


class TestCoverageAndNormalization:
    def test_coverage_window_is_1990_to_2035(self):
        for cal in (US_EQUITY, CHINA_A_SHARE, US_FUTURES_ES, CRYPTO_24_7):
            assert cal.coverage_start == pd.Timestamp("1990-01-01"), (
                f"{cal.name}: coverage_start should be 1990-01-01"
            )
            assert cal.coverage_end == pd.Timestamp("2035-12-31"), (
                f"{cal.name}: coverage_end should be 2035-12-31"
            )

    def test_all_holidays_are_tz_naive_and_date_normalised(self):
        # Plan §3.4 / §10.2: required for replayability and
        # fingerprint stability across machines / locales.
        for cal in (US_EQUITY, CHINA_A_SHARE, US_FUTURES_ES, CRYPTO_24_7):
            for h in cal.holidays:
                assert h.tzinfo is None, f"{cal.name}: {h} is tz-aware"
                assert h.hour == 0 and h.minute == 0 and h.second == 0, (
                    f"{cal.name}: {h} has time component"
                )

    def test_weekend_days_match_market_convention(self):
        assert US_EQUITY.weekend_days == frozenset({5, 6})
        assert CHINA_A_SHARE.weekend_days == frozenset({5, 6})
        assert US_FUTURES_ES.weekend_days == frozenset({5, 6})
        assert CRYPTO_24_7.weekend_days == frozenset()


# ---------- Known holiday assertions ----------


class TestUSEquityHolidays:
    def test_christmas_2024_is_holiday(self):
        assert US_EQUITY.is_trading_day(pd.Timestamp("2024-12-25")) is False

    def test_july_4_2024_is_holiday(self):
        assert US_EQUITY.is_trading_day(pd.Timestamp("2024-07-04")) is False

    def test_thanksgiving_2024_is_holiday(self):
        # Thursday 2024-11-28
        assert US_EQUITY.is_trading_day(pd.Timestamp("2024-11-28")) is False

    def test_pre_2000_holiday_is_caught(self):
        # July 4, 1995 (Tuesday) — verifies the 1990-2035 extension
        # vs the original plan-r1 2000-2030 window.
        # Note: NYSE didn't observe MLK Day until 1998, so we use
        # July 4 instead.
        assert US_EQUITY.is_trading_day(pd.Timestamp("1995-07-04")) is False
        # Christmas 1990 — also pre-2000.
        assert US_EQUITY.is_trading_day(pd.Timestamp("1990-12-25")) is False

    def test_regular_monday_2024_is_trading_day(self):
        # Monday 2024-01-08, no holiday.
        assert US_EQUITY.is_trading_day(pd.Timestamp("2024-01-08")) is True

    def test_us_federal_only_dates_are_distinguished(self):
        # Columbus Day (US federal holiday) — NYSE typically OPEN.
        # Pinpoint a date: 2024-10-14 (second Monday in October).
        # NYSE was open. This proves the calendar is not just a
        # mirror of US federal holidays.
        assert US_EQUITY.is_trading_day(pd.Timestamp("2024-10-14")) is True


class TestChinaAShareHolidays:
    def test_national_day_2024_is_holiday(self):
        # 2024-10-01 = National Day (Tuesday).
        assert CHINA_A_SHARE.is_trading_day(pd.Timestamp("2024-10-01")) is False

    def test_lunar_new_year_2024_is_holiday(self):
        # Lunar New Year 2024-02-10 (Saturday — weekend anyway).
        # Use 2024-02-12 (Monday) which was a CN national holiday.
        assert CHINA_A_SHARE.is_trading_day(pd.Timestamp("2024-02-12")) is False

    def test_provenance_states_sse(self):
        assert CHINA_A_SHARE.provenance["source_calendar"] == "SSE"

    def test_provenance_carries_known_limitations(self):
        # Plan §3.2: SSE/SZSE divergence note must be present.
        kl = CHINA_A_SHARE.provenance["known_limitations"]
        assert isinstance(kl, list) and len(kl) >= 1
        joined = " ".join(kl)
        assert "SSE" in joined
        assert "SZSE" in joined


class TestUSFuturesESHolidays:
    def test_christmas_2024_is_holiday(self):
        # CME Equity full-session calendar treats 2024-12-25 as a
        # holiday (and the early closes on Christmas Eve are not
        # modeled at this resolution).
        assert US_FUTURES_ES.is_trading_day(pd.Timestamp("2024-12-25")) is False

    def test_new_year_2024_is_holiday(self):
        assert US_FUTURES_ES.is_trading_day(pd.Timestamp("2024-01-01")) is False

    def test_provenance_states_cme_equity(self):
        assert US_FUTURES_ES.provenance["source_calendar"] == "CME_Equity"


class TestCrypto247:
    def test_accepts_every_weekend(self):
        for d in pd.date_range("2024-01-01", "2024-01-31"):
            assert CRYPTO_24_7.is_trading_day(d) is True

    def test_accepts_christmas(self):
        assert CRYPTO_24_7.is_trading_day(pd.Timestamp("2024-12-25")) is True

    def test_holidays_set_is_empty(self):
        assert CRYPTO_24_7.holidays == frozenset()

    def test_no_weekend_days(self):
        assert CRYPTO_24_7.weekend_days == frozenset()


# ---------- JSON provenance ----------


class TestJSONProvenance:
    @classmethod
    def setup_class(cls):
        package = "aiphaforge.calendars._data"
        cls.payload = json.loads(
            resources.files(package).joinpath("holidays.json").read_text()
        )

    def test_schema_version(self):
        assert self.payload["schema_version"] == "1.0"

    def test_source_is_pandas_market_calendars(self):
        prov = self.payload["provenance"]
        assert prov["source"] == "pandas_market_calendars"
        assert prov["source_license"] == "MIT"

    def test_runtime_dependency_is_false(self):
        # Critical: the package MUST NOT depend on
        # pandas_market_calendars at runtime.
        assert self.payload["provenance"]["runtime_dependency"] is False

    def test_source_package_version_is_recorded(self):
        # Real version string, not a placeholder.
        version = self.payload["provenance"]["source_package_version"]
        assert version
        assert version != "<filled by generator>"

    def test_generation_script_sha256_is_recorded(self):
        sha = self.payload["provenance"]["generation_script_sha256"]
        assert sha
        # SHA-256 hex digest is 64 chars.
        assert len(sha) == 64

    def test_coverage_matches_predefined(self):
        prov = self.payload["provenance"]
        assert prov["coverage_start"] == "1990-01-01"
        assert prov["coverage_end"] == "2035-12-31"

    def test_next_refresh_target_present(self):
        target = self.payload["provenance"]["next_refresh_target"]
        # 2 years before 2035 boundary.
        assert target == "2033-12-31"

    def test_all_four_calendars_listed(self):
        names = set(self.payload["calendars"].keys())
        assert names == {
            "US_EQUITY", "CHINA_A_SHARE", "US_FUTURES_ES", "CRYPTO_24_7",
        }

    def test_china_a_share_has_known_limitations(self):
        entry = self.payload["calendars"]["CHINA_A_SHARE"]
        assert "known_limitations" in entry
        assert any("SZSE" in lim for lim in entry["known_limitations"])


# ---------- License notice presence ----------


class TestLicenseNotice:
    def test_license_holidays_file_ships(self):
        package = "aiphaforge.calendars._data"
        text = resources.files(package).joinpath("LICENSE.holidays").read_text()
        assert "MIT" in text
        assert "pandas_market_calendars" in text

    def test_license_notice_includes_copyright(self):
        package = "aiphaforge.calendars._data"
        text = resources.files(package).joinpath("LICENSE.holidays").read_text()
        # Verbatim MIT requires copyright notice.
        assert "Copyright" in text
