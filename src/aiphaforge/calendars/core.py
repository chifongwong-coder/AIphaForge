"""Trading-calendar primitive for the v2.1 calendars module.

Daily-resolution. No intraday session hours, no early closes
(deferred to v2.2). All public types here are forever-stable
public API once shipped, so the field set is intentionally minimal.

The four error classes (``CalendarSnapError``,
``CalendarSnapCollisionError``, ``CalendarConflictError``,
``CalendarProviderProtocolError``) live here too so any caller of
the calendar API has a single import surface for catching them.

Plan: see ``docs/plans/v2.1-plan.md`` (r3-final).
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Optional

import pandas as pd

# `IntegrityCheckResult` is reused from probes.transforms — the
# shape (passed: bool, errors: list[str]) is exactly what
# is_conformant returns. Importing it at runtime is fine because
# transforms.py uses TYPE_CHECKING for its calendar-side type hints
# (no cycle).
from aiphaforge.probes.transforms import IntegrityCheckResult

if TYPE_CHECKING:
    pass


# ---------- Errors ----------


class CalendarSnapError(ValueError):
    """A date snap was requested with ``snap="error"`` and the input
    falls on a non-trading day.
    """


class CalendarSnapCollisionError(ValueError):
    """``DateShift.apply`` produced duplicate output timestamps under
    calendar snapping and ``on_collision="error"``.

    Raised before any frame is returned.
    """


class CalendarConflictError(ValueError):
    """Two distinct trading calendars were inferred for the same
    scenario / pipeline.

    Raised by ``_effective_calendar_from_transforms`` (v2.1 §5.4) and
    by ``TransformPipeline`` when an explicit calendar disagrees with
    the calendar inferred from its transforms.
    """


class CalendarProviderProtocolError(TypeError):
    """A transform declares ``_aiphaforge_calendar_provider = True``
    but does not implement ``get_effective_calendar()`` (or that call
    raised).

    Raised by the inference helper to surface a clear protocol error
    instead of a generic ``AttributeError`` deep in the call stack.
    """


# ---------- Constants ----------

_VALID_DAYOFWEEK = frozenset(range(7))


# ---------- Helpers ----------


def _normalize_to_date(ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    """Coerce input to a tz-naive, time-zero ``pd.Timestamp`` or None.

    Inputs that are already date-only and tz-naive pass through; tz-
    aware inputs are converted to UTC-naive and then date-normalized;
    inputs with a non-zero time component are floored to the day.
    """
    if ts is None:
        return None
    out = pd.Timestamp(ts)
    if out.tzinfo is not None:
        # Strip tz: the calendar lives in date space, not wall-clock.
        out = out.tz_convert("UTC").tz_localize(None)
    return out.normalize()


def _normalize_holidays(
    holidays: Iterable[Any],
) -> frozenset[pd.Timestamp]:
    out: set[pd.Timestamp] = set()
    for h in holidays:
        norm = _normalize_to_date(h)
        if norm is None:
            continue
        out.add(norm)
    return frozenset(out)


def _holidays_hash(holidays: frozenset[pd.Timestamp]) -> str:
    """Deterministic short hash for fingerprint use.

    Two frozensets containing the same dates produce the same hash
    regardless of insertion order, since we sort before hashing.
    """
    h = hashlib.sha256()
    for ts in sorted(holidays):
        h.update(ts.isoformat().encode("ascii"))
        h.update(b";")
    return h.hexdigest()[:16]


def _format_offending_dates(
    dates: list[pd.Timestamp],
    *,
    cap: int = 10,
) -> str:
    """Bounded human-readable date list (plan §2.4 r3)."""
    if not dates:
        return ""
    sorted_dates = sorted(dates)
    shown = sorted_dates[:cap]
    omitted = len(sorted_dates) - len(shown)
    body = ", ".join(d.strftime("%Y-%m-%d") for d in shown)
    if omitted > 0:
        return f"{body}, ... and {omitted} more"
    return body


# ---------- TradingCalendar ----------


@dataclass(frozen=True)
class TradingCalendar:
    """A trading-day calendar with weekend + holiday awareness.

    Daily resolution only. Does NOT model session open/close times,
    early closes, partial holidays, or intraday breaks. Intraday
    semantics are deferred to v2.2.

    Field conventions:
        * ``weekend_days`` uses ``pd.Timestamp.dayofweek``
          (Monday=0, ..., Sunday=6). No implicit default — the four
          predefined US/CN calendars supply ``frozenset({5, 6})``;
          ``CRYPTO_24_7`` supplies ``frozenset()``.
        * ``holidays`` is a frozenset of tz-naive, date-normalised
          ``pd.Timestamp``s. Construction normalises any input
          timestamps.
        * ``coverage_start`` / ``coverage_end`` are optional tz-naive,
          date-only markers for the holiday-data window. Used by the
          per-instance out-of-range warning. User calendars may omit
          them and skip the warning entirely.
        * ``provenance`` mirrors the JSON provenance section for the
          predefined calendars; user calendars typically pass
          ``None``.

    The private ``_warning_keys_seen`` set lets the boundary warning
    fire at most once per instance per warning key. It is excluded
    from repr / equality / hash / ``stable_fingerprint()``.
    """

    name: str
    weekend_days: frozenset[int]
    holidays: frozenset[pd.Timestamp]
    coverage_start: Optional[pd.Timestamp] = None
    coverage_end: Optional[pd.Timestamp] = None
    provenance: Optional[Mapping[str, Any]] = None
    _warning_keys_seen: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    # ---- construction normalization ----

    def __post_init__(self) -> None:
        # Validate weekend_days range.
        bad = [d for d in self.weekend_days if d not in _VALID_DAYOFWEEK]
        if bad:
            raise ValueError(
                f"weekend_days contains out-of-range values {sorted(bad)}; "
                "valid values are 0 (Mon) through 6 (Sun)."
            )

        # Re-normalize holidays / coverage in case the caller passed
        # tz-aware or time-bearing timestamps. We can't reassign on a
        # frozen dataclass via plain attribute syntax, but
        # object.__setattr__ is the documented escape hatch for
        # __post_init__.
        normalized_holidays = _normalize_holidays(self.holidays)
        if normalized_holidays != self.holidays:
            object.__setattr__(self, "holidays", normalized_holidays)

        normalized_start = _normalize_to_date(self.coverage_start)
        if normalized_start != self.coverage_start:
            object.__setattr__(self, "coverage_start", normalized_start)

        normalized_end = _normalize_to_date(self.coverage_end)
        if normalized_end != self.coverage_end:
            object.__setattr__(self, "coverage_end", normalized_end)

    # ---- core membership ----

    def is_trading_day(self, ts: pd.Timestamp) -> bool:
        """True if ``ts`` (date component only) is a trading day.

        A timestamp outside the declared coverage window falls back
        to weekday-only logic and emits the boundary warning at most
        once per warning key per instance.
        """
        date_ts = _normalize_to_date(ts)
        if date_ts is None:
            return False
        if date_ts.dayofweek in self.weekend_days:
            return False
        # Out-of-range fallback.
        if self._is_out_of_range(date_ts):
            self._warn_once(
                "out_of_range_weekday_fallback",
                f"{self.name}: queried date {date_ts.date()} is outside "
                f"declared coverage window "
                f"[{self.coverage_start.date() if self.coverage_start else '-inf'}, "
                f"{self.coverage_end.date() if self.coverage_end else '+inf'}]; "
                "holidays out of range; weekday-only fallback.",
            )
            return True  # weekday + no holiday lookup
        return date_ts not in self.holidays

    def next_trading_day(self, ts: pd.Timestamp) -> pd.Timestamp:
        """Smallest trading day strictly greater than ``ts``."""
        date_ts = _normalize_to_date(ts)
        if date_ts is None:
            raise ValueError("next_trading_day requires a non-None timestamp")
        candidate = date_ts + pd.Timedelta(days=1)
        # 7-day weekend (impossible) would loop forever; bound by 366
        # iterations to guarantee termination even with degenerate
        # weekend_days.
        for _ in range(366):
            if self.is_trading_day(candidate):
                return candidate
            candidate = candidate + pd.Timedelta(days=1)
        raise ValueError(
            f"next_trading_day: no trading day found within 366 days of "
            f"{date_ts.date()} for calendar {self.name!r}; check "
            "weekend_days / holidays for a degenerate definition."
        )

    def prev_trading_day(self, ts: pd.Timestamp) -> pd.Timestamp:
        """Largest trading day strictly less than ``ts``."""
        date_ts = _normalize_to_date(ts)
        if date_ts is None:
            raise ValueError("prev_trading_day requires a non-None timestamp")
        candidate = date_ts - pd.Timedelta(days=1)
        for _ in range(366):
            if self.is_trading_day(candidate):
                return candidate
            candidate = candidate - pd.Timedelta(days=1)
        raise ValueError(
            f"prev_trading_day: no trading day found within 366 days of "
            f"{date_ts.date()} for calendar {self.name!r}."
        )

    def snap(
        self,
        ts: pd.Timestamp,
        direction: Literal["forward", "backward", "nearest", "error"],
    ) -> pd.Timestamp:
        """Return the nearest trading day in the requested direction.

        Trading days pass through unchanged regardless of direction.
        Non-trading days dispatch on ``direction``:

            * ``"forward"``  → ``next_trading_day(ts)``
            * ``"backward"`` → ``prev_trading_day(ts)``
            * ``"nearest"``  → closer of next / prev; tie → forward
              (deterministic).
            * ``"error"``    → raise :class:`CalendarSnapError`.
        """
        date_ts = _normalize_to_date(ts)
        if date_ts is None:
            raise ValueError("snap requires a non-None timestamp")
        if self.is_trading_day(date_ts):
            return date_ts

        if direction == "error":
            raise CalendarSnapError(
                f"{self.name}: {date_ts.date()} is not a trading day "
                "(snap='error')."
            )
        if direction == "forward":
            return self.next_trading_day(date_ts)
        if direction == "backward":
            return self.prev_trading_day(date_ts)
        if direction == "nearest":
            nxt = self.next_trading_day(date_ts)
            prv = self.prev_trading_day(date_ts)
            fwd_gap = (nxt - date_ts).days
            bwd_gap = (date_ts - prv).days
            # Tie: forward, for deterministic behavior (plan §4.2).
            return nxt if fwd_gap <= bwd_gap else prv
        raise ValueError(
            f"snap direction must be 'forward'/'backward'/'nearest'/'error', "
            f"got {direction!r}"
        )

    # ---- conformance ----

    def is_conformant(
        self,
        index: pd.DatetimeIndex,
    ) -> IntegrityCheckResult:
        """Return ``IntegrityCheckResult`` describing whether every
        date in ``index`` is a trading day under this calendar.

        Offending dates are listed in the ``errors`` string, capped
        at the first 10 dates with ``... and K more`` per plan §2.4.
        """
        offending: list[pd.Timestamp] = []
        for ts in index:
            if not self.is_trading_day(ts):
                offending.append(_normalize_to_date(ts))
        if not offending:
            return IntegrityCheckResult(passed=True, errors=[])
        msg = (
            f"calendar {self.name!r} non-conformant on "
            f"{len(offending)} dates: "
            + _format_offending_dates(offending, cap=10)
        )
        return IntegrityCheckResult(passed=False, errors=[msg])

    # ---- fingerprint ----

    def stable_fingerprint(self) -> tuple:
        """Tuple suitable for value-based equality across separately
        constructed calendars.

        Compares by VALUE (Python tuple equality), not by object
        identity, so a user-constructed copy with the same fields as
        a predefined calendar matches. The mutable
        ``_warning_keys_seen`` set is deliberately excluded.
        """
        return (
            self.name,
            tuple(sorted(self.weekend_days)),
            _holidays_hash(self.holidays),
            self.coverage_start,
            self.coverage_end,
        )

    # ---- internals ----

    def _is_out_of_range(self, date_ts: pd.Timestamp) -> bool:
        if self.coverage_start is not None and date_ts < self.coverage_start:
            return True
        if self.coverage_end is not None and date_ts > self.coverage_end:
            return True
        return False

    def _warn_once(self, key: str, message: str) -> None:
        # The set is mutable even on a frozen dataclass because we
        # only mutate its CONTENTS, not the field itself. The
        # standard frozen-dataclass-with-mutable-internal-state
        # pattern.
        if key in self._warning_keys_seen:
            return
        self._warning_keys_seen.add(key)
        warnings.warn(message, UserWarning, stacklevel=3)


__all__ = [
    "CalendarConflictError",
    "CalendarProviderProtocolError",
    "CalendarSnapCollisionError",
    "CalendarSnapError",
    "TradingCalendar",
]
