"""Offline holiday-data generator for v2.1 trading calendars.

Run this once per refresh cycle to regenerate
``src/aiphaforge/calendars/_data/holidays.json`` from the
upstream ``pandas_market_calendars`` package.

``pandas_market_calendars`` is NOT a runtime dependency of
AIphaForge — this script is the only place it is touched. The
generated JSON is committed to the repo and shipped with the
package.

Usage:
    python scripts/generate_holidays_json.py

Plan: see ``docs/plans/v2.1-plan.md`` §3.4 (r3-final).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pandas_market_calendars as mcal

# Calendars to extract: (canonical name in JSON, source calendar in mcal,
# market_scope label, optional known_limitations list)
CALENDAR_SPECS = [
    {
        "name": "US_EQUITY",
        "source_calendar": "NYSE",
        "market_scope": "us_equity_daily_full_session",
        "known_limitations": None,
    },
    {
        "name": "CHINA_A_SHARE",
        "source_calendar": "SSE",
        "market_scope": "mainland_china_a_share_daily_full_session",
        "known_limitations": [
            "CHINA_A_SHARE is pinned to SSE for v2.1. SZSE-specific "
            "divergences, if any, are documented here and deferred to "
            "v2.1.x or v2.2.",
        ],
    },
    {
        "name": "US_FUTURES_ES",
        "source_calendar": "CME_Equity",
        "market_scope": "cme_equity_futures_daily_full_session",
        "known_limitations": None,
    },
    # CRYPTO_24_7 is intentionally empty — no holidays, every day a
    # trading day. We emit it anyway so the JSON file enumerates all
    # four calendars consistently.
    {
        "name": "CRYPTO_24_7",
        "source_calendar": "manual_empty",
        "market_scope": "crypto_24_7",
        "known_limitations": None,
    },
]

COVERAGE_START = pd.Timestamp("1990-01-01")
COVERAGE_END = pd.Timestamp("2035-12-31")
NEXT_REFRESH = "2033-12-31"  # 2 years before boundary

OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "src" / "aiphaforge" / "calendars" / "_data" / "holidays.json"
)


def _holidays_for(source_calendar: str) -> list[str]:
    """Pull holidays in [COVERAGE_START, COVERAGE_END] for one source.

    Returns ISO-formatted date strings (YYYY-MM-DD), sorted ascending.
    The CRYPTO sentinel returns an empty list.
    """
    if source_calendar == "manual_empty":
        return []
    calendar = mcal.get_calendar(source_calendar)
    raw_holidays = calendar.holidays().holidays  # numpy datetime64 array
    lo = COVERAGE_START.to_datetime64()
    hi = COVERAGE_END.to_datetime64()
    in_range = [d for d in raw_holidays if lo <= d <= hi]
    out = sorted(
        pd.Timestamp(d).strftime("%Y-%m-%d") for d in in_range
    )
    return out


def _script_sha256() -> str:
    """Hash of THIS file at generation time, for the provenance block.

    The hash is computed against the on-disk script content; the
    in-memory sys.argv[0] resolution handles symlinks.
    """
    script_path = Path(sys.argv[0]).resolve()
    if not script_path.is_file():
        # Fall back to the module __file__ if the script was invoked
        # via `python -m`.
        script_path = Path(__file__).resolve()
    h = hashlib.sha256(script_path.read_bytes()).hexdigest()
    return h


def _build_payload() -> dict[str, Any]:
    today = _dt.date.today().isoformat()
    calendars: dict[str, Any] = {}
    for spec in CALENDAR_SPECS:
        entry: dict[str, Any] = {
            "source_calendar": spec["source_calendar"],
            "market_scope": spec["market_scope"],
            "holidays": _holidays_for(spec["source_calendar"]),
        }
        if spec["known_limitations"]:
            entry["known_limitations"] = spec["known_limitations"]
        calendars[spec["name"]] = entry

    payload = {
        "schema_version": "1.0",
        "provenance": {
            "source": "pandas_market_calendars",
            "source_license": "MIT",
            "source_url": (
                "https://github.com/rsheftel/pandas_market_calendars"
            ),
            "source_package_version": mcal.__version__,
            "generation_script": "scripts/generate_holidays_json.py",
            "generation_script_sha256": _script_sha256(),
            "generated_at": today,
            "last_verified": today,
            "next_refresh_target": NEXT_REFRESH,
            "maintainer": "AIphaForge contributors",
            "coverage_start": COVERAGE_START.strftime("%Y-%m-%d"),
            "coverage_end": COVERAGE_END.strftime("%Y-%m-%d"),
            "runtime_dependency": False,
        },
        "calendars": calendars,
    }
    return payload


def main() -> None:
    payload = _build_payload()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )
    sizes = {
        name: len(payload["calendars"][name]["holidays"])
        for name in payload["calendars"]
    }
    print(f"wrote {OUT_PATH}")
    print(f"holiday counts: {sizes}")
    print(f"source pkg version: {payload['provenance']['source_package_version']}")
    print(f"script sha256: {payload['provenance']['generation_script_sha256']}")


if __name__ == "__main__":
    main()
