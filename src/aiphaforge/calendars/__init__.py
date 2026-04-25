"""Trading-calendar foundation (v2.1).

A self-contained module — no engine dependencies, daily resolution.
Predefined calendars (`US_EQUITY`, `CHINA_A_SHARE`, `CRYPTO_24_7`,
`US_FUTURES_ES`) ship in M2; M1 ships only the primitive.

Plan: see ``docs/plans/v2.1-plan.md`` (r3-final).
"""
from aiphaforge.calendars.core import (
    CalendarConflictError,
    CalendarProviderProtocolError,
    CalendarSnapCollisionError,
    CalendarSnapError,
    TradingCalendar,
)

__all__ = [
    "CalendarConflictError",
    "CalendarProviderProtocolError",
    "CalendarSnapCollisionError",
    "CalendarSnapError",
    "TradingCalendar",
]
