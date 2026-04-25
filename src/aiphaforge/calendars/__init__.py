"""Trading-calendar foundation (v2.1).

A self-contained module — no engine dependencies, daily resolution.

Public API at the package root:

- :class:`TradingCalendar` — the primitive (M1)
- :data:`US_EQUITY`, :data:`CHINA_A_SHARE`, :data:`CRYPTO_24_7`,
  :data:`US_FUTURES_ES` — predefined calendars (M2)
- :class:`CalendarSnapError`, :class:`CalendarSnapCollisionError`,
  :class:`CalendarConflictError`,
  :class:`CalendarProviderProtocolError` — error classes (M1)

Plan: see ``docs/plans/v2.1-plan.md`` (r3-final).
"""
from aiphaforge.calendars.core import (
    CalendarConflictError,
    CalendarProviderProtocolError,
    CalendarSnapCollisionError,
    CalendarSnapError,
    TradingCalendar,
)
from aiphaforge.calendars.predefined import (
    CHINA_A_SHARE,
    CRYPTO_24_7,
    US_EQUITY,
    US_FUTURES_ES,
)

__all__ = [
    "CHINA_A_SHARE",
    "CRYPTO_24_7",
    "CalendarConflictError",
    "CalendarProviderProtocolError",
    "CalendarSnapCollisionError",
    "CalendarSnapError",
    "TradingCalendar",
    "US_EQUITY",
    "US_FUTURES_ES",
]
