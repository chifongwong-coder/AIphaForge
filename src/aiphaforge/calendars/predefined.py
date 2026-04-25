"""Predefined trading calendars for v2.1.

Four exchange calendars with shipped holiday data 1990-2035, loaded
once at import time from the vendored ``_data/holidays.json`` file.

The JSON itself is generated offline from ``pandas_market_calendars``
(MIT) by ``scripts/generate_holidays_json.py``. See
``_data/LICENSE.holidays`` for the upstream attribution.

NEXT REFRESH: 2033-12-31 — extend coverage forward before this date
so users running multi-decade backtests do not hit the boundary
warning. The current refresh target is recorded in the JSON
``provenance.next_refresh_target`` field.

Plan: see ``docs/plans/v2.1-plan.md`` §3 (r3-final).
"""
from __future__ import annotations

import json
from importlib import resources
from typing import Any

import pandas as pd

from aiphaforge.calendars.core import TradingCalendar

_JSON_RESOURCE = "holidays.json"

# US/CN/CME exchange calendars all use Mon-Fri weekend convention.
_MON_FRI_WEEKEND = frozenset({5, 6})

# Crypto trades 24/7.
_NO_WEEKEND = frozenset()


def _load_holidays_payload() -> dict[str, Any]:
    """Load the vendored JSON exactly once.

    Uses ``importlib.resources`` so the data file is found regardless
    of whether the package is installed editable, as a wheel, or
    zipped.
    """
    package = "aiphaforge.calendars._data"
    text = resources.files(package).joinpath(_JSON_RESOURCE).read_text()
    return json.loads(text)


def _build_calendar(
    name: str,
    *,
    weekend_days: frozenset[int],
    payload: dict[str, Any],
) -> TradingCalendar:
    cal_entry = payload["calendars"][name]
    holidays = frozenset(
        pd.Timestamp(s) for s in cal_entry["holidays"]
    )
    coverage_start = pd.Timestamp(payload["provenance"]["coverage_start"])
    coverage_end = pd.Timestamp(payload["provenance"]["coverage_end"])
    # Provenance for downstream consumers — read-only mapping that
    # mirrors the JSON entry plus the global package metadata. Wrap
    # in a plain dict (immutable in spirit; the dataclass is frozen).
    provenance = {
        "source": payload["provenance"]["source"],
        "source_license": payload["provenance"]["source_license"],
        "source_calendar": cal_entry["source_calendar"],
        "market_scope": cal_entry["market_scope"],
        "source_package_version": payload["provenance"]["source_package_version"],
        "generated_at": payload["provenance"]["generated_at"],
        "coverage_start": payload["provenance"]["coverage_start"],
        "coverage_end": payload["provenance"]["coverage_end"],
        "runtime_dependency": payload["provenance"]["runtime_dependency"],
    }
    if "known_limitations" in cal_entry:
        provenance["known_limitations"] = list(cal_entry["known_limitations"])
    return TradingCalendar(
        name=name,
        weekend_days=weekend_days,
        holidays=holidays,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        provenance=provenance,
    )


_PAYLOAD = _load_holidays_payload()

# US equity (NYSE-style) — Mon-Fri, NYSE full-session holidays
# 1990-2035.
US_EQUITY: TradingCalendar = _build_calendar(
    "US_EQUITY",
    weekend_days=_MON_FRI_WEEKEND,
    payload=_PAYLOAD,
)

# Mainland China A-share (SSE-pinned for v2.1; SZSE divergences
# recorded in provenance.known_limitations rather than splitting the
# public schema mid-release).
CHINA_A_SHARE: TradingCalendar = _build_calendar(
    "CHINA_A_SHARE",
    weekend_days=_MON_FRI_WEEKEND,
    payload=_PAYLOAD,
)

# CME Equity futures (ES contract) — Mon-Fri, CME Equity
# full-session holidays. Subset of NYSE on most dates but diverges on
# a few (Good Friday, etc.).
US_FUTURES_ES: TradingCalendar = _build_calendar(
    "US_FUTURES_ES",
    weekend_days=_MON_FRI_WEEKEND,
    payload=_PAYLOAD,
)

# Crypto: every day is a trading day. We deliberately set a coverage
# window matching the others so the boundary warning suppression
# tests apply uniformly — but with no holidays, the window only
# affects the warning for queries far outside the 1990-2035 range.
CRYPTO_24_7: TradingCalendar = _build_calendar(
    "CRYPTO_24_7",
    weekend_days=_NO_WEEKEND,
    payload=_PAYLOAD,
)


__all__ = [
    "CHINA_A_SHARE",
    "CRYPTO_24_7",
    "US_EQUITY",
    "US_FUTURES_ES",
]
