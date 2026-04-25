"""Data transforms for the v2.0 memory-probe A/B runner.

Three categories of transform (canonical stage order):

1. **metadata** — `SymbolMasker`, `DateShift`. These mutate identity
   surfaces (symbol, timestamp) without changing the price path.
2. **level** — `PriceScale`, `PriceRebase`. Multiplicative rescaling;
   returns are preserved exactly, level presentation changes.
3. **series** — `OHLCJitter`, `BlockBootstrap`, `WindowShuffle`. The
   observed price path itself changes.

A `TransformPipeline` enforces the canonical stage order and runs
full-frame OHLC integrity validation. `BlockBootstrap` reuses the
existing OHLC-preserving primitive in ``aiphaforge.significance``.

Metadata transforms expose explicit ``mask`` / ``unmask`` /
``shift_date`` APIs because the view-only wrapper (later milestone)
needs them for broker translation. ``apply`` itself is documented as
a no-op for SymbolMasker (the symbol identity is metadata, not part
of the OHLCV frame).

See `docs/plans/v2.0-plan.md` §2 for the full contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
)

import numpy as np
import pandas as pd

# Reuse the engine's existing OHLC-preserving block-bootstrap primitive.
from aiphaforge.significance import (
    _block_bootstrap_indices,
    _reconstruct_ohlcv,
)

if TYPE_CHECKING:
    # Calendar types live in aiphaforge.calendars (v2.1). They are
    # loaded lazily via TYPE_CHECKING because calendars/core.py
    # imports IntegrityCheckResult from THIS module — avoiding a
    # cyclic import at module-load time.
    from aiphaforge.calendars.core import TradingCalendar


def _import_calendar_collision_error():
    """Lazy import of CalendarSnapCollisionError to avoid the cycle.

    Called only when DateShift._resolve_collisions actually needs to
    raise. Module-load-time has no calendars import.
    """
    from aiphaforge.calendars.core import CalendarSnapCollisionError
    return CalendarSnapCollisionError


def _import_calendar_conflict_error():
    """Lazy import of CalendarConflictError; same cycle reasoning."""
    from aiphaforge.calendars.core import CalendarConflictError
    return CalendarConflictError


def _import_calendar_provider_protocol_error():
    """Lazy import of CalendarProviderProtocolError; same cycle."""
    from aiphaforge.calendars.core import CalendarProviderProtocolError
    return CalendarProviderProtocolError


# ---------- v2.1.0 r4 stabilization (M8): per-call diagnostics ----------


@dataclass(frozen=True)
class TransformDiagnostic:
    """One structured diagnostic produced by a transform's apply call.

    Manifest warnings derive from these — never from mutable
    transform-instance state. Plan r4-final §3.

    ``code`` and ``source`` identify the diagnostic; ``severity`` is
    one of ``"info"`` / ``"warning"`` / ``"error"``; ``details``
    carries arbitrary JSON-shaped metadata (the manifest serializer
    runs separately and is responsible for stringifying any in-memory
    pd.Timestamp values).
    """

    code: str
    source: str
    severity: Literal["info", "warning", "error"]
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransformApplyResult:
    """Result of applying a transform pipeline to a single arm.

    ``data`` is the transformed frame; ``diagnostics`` is the tuple
    of diagnostics emitted across all transforms during this apply.
    Per-call so cross-arm aliasing (the M5/r3 footgun) is impossible
    by construction.
    """

    data: pd.DataFrame
    diagnostics: tuple[TransformDiagnostic, ...] = ()


def _effective_calendar_from_transforms(
    transforms: Sequence[Any],
) -> Optional["TradingCalendar"]:
    """Infer the effective calendar from a transform sequence.

    Plan §5.4 r3-final: only transforms with the explicit
    ``_aiphaforge_calendar_provider = True`` class attribute AND a
    working ``get_effective_calendar()`` method participate. User
    transforms with an unrelated ``.calendar`` attribute are ignored
    — bare duck-typing would silently route accidental attributes
    into scenario validation and recreate the foot-gun this design
    eliminates.

    Returns:
        The single calendar all participating transforms agree on,
        or ``None`` if no transform participates.

    Raises:
        CalendarProviderProtocolError: a transform declares the
            marker but does not implement ``get_effective_calendar``,
            or that call raises an exception.
        CalendarConflictError: two participating transforms supply
            calendars that differ by ``stable_fingerprint()``.
    """
    calendars: list[Any] = []
    for transform in transforms:
        if not getattr(transform, "_aiphaforge_calendar_provider", False):
            continue
        provider = getattr(transform, "get_effective_calendar", None)
        if provider is None:
            CalendarProviderProtocolError = (
                _import_calendar_provider_protocol_error()
            )
            raise CalendarProviderProtocolError(
                f"{type(transform).__name__} declares "
                "_aiphaforge_calendar_provider=True but does not "
                "implement get_effective_calendar()."
            )
        try:
            calendar = provider()
        except Exception as exc:
            # Wrap so users get a clear protocol-violation error
            # rather than a bare AttributeError or similar deep in
            # the stack (defect #2 fix from r3 review).
            CalendarProviderProtocolError = (
                _import_calendar_provider_protocol_error()
            )
            raise CalendarProviderProtocolError(
                f"{type(transform).__name__}.get_effective_calendar() "
                f"raised {type(exc).__name__}: {exc}"
            ) from exc
        if calendar is None:
            continue
        # v2.1.1 guard: a buggy provider returning a non-calendar
        # object (e.g. a config dict, a string) should fail clearly
        # at the protocol boundary, not later in
        # ``.stable_fingerprint()``.
        from aiphaforge.calendars.core import TradingCalendar
        if not isinstance(calendar, TradingCalendar):
            CalendarProviderProtocolError = (
                _import_calendar_provider_protocol_error()
            )
            raise CalendarProviderProtocolError(
                f"{type(transform).__name__}.get_effective_calendar() "
                f"returned {type(calendar).__name__}, expected "
                "TradingCalendar."
            )
        calendars.append(calendar)

    if not calendars:
        return None

    first = calendars[0]
    for calendar in calendars[1:]:
        # Plan §5.2: match by tuple equality (value-based), not
        # object identity. Two separately-constructed identical
        # calendars match.
        if calendar.stable_fingerprint() != first.stable_fingerprint():
            CalendarConflictError = _import_calendar_conflict_error()
            raise CalendarConflictError(
                f"Two distinct effective calendars inferred from "
                f"transforms: {first.name!r} vs {calendar.name!r}."
            )
    return first

TransformCategory = Literal["metadata", "level", "series"]


class DataTransform(Protocol):
    """Protocol every built-in or user transform must satisfy.

    Implementations declare via class attributes:
    - ``name``: short identifier (also used in manifest)
    - ``category``: one of the three canonical stages
    - ``supports_view_only``: usable when the engine fills at real
      prices and only the agent's view is masked
    - ``supports_market_level``: usable when the transformed data
      becomes the execution market
    - ``order_invertible``: in ``view_only``, can the wrapper map an
      order placed on transformed view back to original execution?
    - ``stochastic``: does ``apply`` produce different output for
      different seeds?
    """

    name: str
    category: TransformCategory
    supports_view_only: bool
    supports_market_level: bool
    order_invertible: bool
    stochastic: bool

    def apply(
        self, data: pd.DataFrame, *, seed: Optional[int] = None,
    ) -> pd.DataFrame: ...


# ---------- Metadata transforms ----------

class SymbolMasker:
    """Map real tickers to plausible obscure aliases.

    The mapping is bijective and computed at construction time from a
    fixed universe. ``collision_policy="fail"`` (the v2.0 default)
    raises on alias collision — auto-rehashing is rejected because it
    would silently change the mapping the user thinks they configured.

    `apply(data)` is a no-op on the OHLCV frame itself: the symbol
    identity travels alongside the data via the `BacktestHook` API,
    not as a column. The view-only wrapper consumes
    ``mask_symbol`` / ``unmask_symbol`` directly.
    """

    name = "SymbolMasker"
    category: TransformCategory = "metadata"
    supports_view_only = True
    supports_market_level = True
    order_invertible = True
    stochastic = False

    # Synthetic alias pool — opaque ``SYM_xxxx`` IDs that are clearly
    # not real listed tickers. Earlier drafts shipped 4-letter
    # "obscure" aliases, but several turned out to be real listed
    # symbols; a frontier LLM would recognize them and the pattern.
    # Users who want plausibly-equity-shaped aliases must supply
    # ``alias_pool=...`` explicitly and accept the responsibility of
    # checking the strings against real-listed databases.
    DEFAULT_ALIAS_POOL: tuple[str, ...] = tuple(
        f"SYM_{i:04d}" for i in range(1000)
    )

    def __init__(
        self,
        symbols: Sequence[str],
        *,
        seed: int = 0,
        alias_pool: Optional[Sequence[str]] = None,
        collision_policy: Literal["fail"] = "fail",
    ):
        self.symbols = tuple(symbols)
        self.seed = seed
        self.alias_pool = tuple(alias_pool) if alias_pool else self.DEFAULT_ALIAS_POOL
        self.collision_policy = collision_policy
        self._mapping = self._compute_mapping()
        self._inverse = {v: k for k, v in self._mapping.items()}

    def _compute_mapping(self) -> dict[str, str]:
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError(
                f"SymbolMasker received duplicate symbols: {self.symbols}"
            )
        if len(self.symbols) > len(self.alias_pool):
            raise ValueError(
                f"SymbolMasker alias pool ({len(self.alias_pool)}) "
                f"smaller than symbol count ({len(self.symbols)})"
            )
        # `replace=False` only guarantees distinct *indices* into the
        # alias pool. If the user-supplied pool itself contains
        # duplicate strings, two distinct indices can map to the same
        # alias — the `seen` check below is the actual collision guard.
        rng = np.random.default_rng(self.seed)
        chosen_indices = rng.choice(
            len(self.alias_pool),
            size=len(self.symbols),
            replace=False,
        )
        mapping: dict[str, str] = {}
        seen: set[str] = set()
        for sym, idx in zip(self.symbols, chosen_indices):
            alias = self.alias_pool[idx]
            if alias in seen:
                raise ValueError(
                    f"SymbolMasker alias collision on '{alias}' "
                    f"(seed={self.seed}); the user-supplied alias_pool "
                    "contains duplicate strings"
                )
            seen.add(alias)
            mapping[sym] = alias
        return mapping

    def mask_symbol(self, symbol: str) -> str:
        if symbol not in self._mapping:
            raise KeyError(f"Symbol '{symbol}' not in SymbolMasker universe")
        return self._mapping[symbol]

    def unmask_symbol(self, masked: str) -> str:
        if masked not in self._inverse:
            raise KeyError(
                f"Masked symbol '{masked}' not produced by this SymbolMasker"
            )
        return self._inverse[masked]

    def apply(
        self, data: pd.DataFrame, *, seed: Optional[int] = None,
    ) -> pd.DataFrame:
        # Symbol identity is not a frame column; SymbolMasker is a
        # no-op at the data level. Returning data unchanged keeps the
        # protocol consistent. The view-only wrapper uses
        # `mask_symbol`/`unmask_symbol` directly.
        return data


class DateShift:
    """Relabel the calendar by a fixed offset.

    ``offset`` may be any value pandas can add to a DatetimeIndex
    (a `pd.DateOffset`, `pd.Timedelta`, or anything coercible).

    Calendar memory leaks (FOMC dates, OPEX Fridays, holiday gaps)
    are NOT addressed by this transform — the price path is
    unchanged so behavioral fingerprints survive. See plan §2.6
    "Known detectability caveat".

    v2.1 calendar integration:
        Pass a ``TradingCalendar`` plus ``snap`` and
        ``on_collision`` policies to make shifted dates land on
        trading days. See ``docs/plans/v2.1-plan.md`` §4 (r3-final).

        - ``snap``: ``"forward"`` (default) snaps non-trading
          shifted dates to the next trading day; ``"backward"`` to
          previous; ``"nearest"`` to closest (ties forward);
          ``"error"`` raises ``CalendarSnapError``.
        - ``on_collision``: when calendar snapping maps multiple
          source dates onto the same target trading day,
          ``"error"`` (default) raises
          ``CalendarSnapCollisionError``; ``"keep_first"`` /
          ``"keep_last"`` drop the duplicates and surface a
          structured collision diagnostic via the per-call
          :class:`TransformApplyResult` returned by
          :meth:`apply_with_diagnostics`. The pipeline forwards
          this diagnostic into the per-scenario manifest.

        ``unshift_date`` is exact when ``calendar=None`` (the v2.0
        contract). With a calendar set it is best-effort and lossy
        — see the docstring on the method.
    """

    name = "DateShift"
    category: TransformCategory = "metadata"
    supports_view_only = True
    supports_market_level = True
    order_invertible = True
    stochastic = False

    # Marker for the v2.1 calendar-aware-transform protocol
    # (plan §4.1 / §5.3 r3-final). Annotated as ClassVar[Literal[True]]
    # so static type checkers can verify Protocol conformance.
    _aiphaforge_calendar_provider: ClassVar[Literal[True]] = True

    def __init__(
        self,
        offset,
        *,
        calendar: Optional["TradingCalendar"] = None,
        snap: Literal["forward", "backward", "nearest", "error"] = "forward",
        on_collision: Literal["error", "keep_first", "keep_last"] = "error",
    ):
        if snap not in ("forward", "backward", "nearest", "error"):
            raise ValueError(
                f"snap must be 'forward'/'backward'/'nearest'/'error', "
                f"got {snap!r}"
            )
        if on_collision not in ("error", "keep_first", "keep_last"):
            raise ValueError(
                f"on_collision must be 'error'/'keep_first'/'keep_last', "
                f"got {on_collision!r}"
            )
        self.offset = offset
        self.calendar = calendar
        self.snap = snap
        self.on_collision = on_collision
        # Note: M8 (r4-final) intentionally removed
        # ``self.last_collision_report``. Collision diagnostics are
        # per-call now — see :meth:`apply_with_diagnostics`. Mutable
        # instance state was the root cause of the cross-arm aliasing
        # foot-gun documented in v2.1 plan §15.2.

    # ---- v2.1 calendar-aware-transform protocol ----

    def get_effective_calendar(self) -> Optional["TradingCalendar"]:
        """Return the calendar this transform participates with.

        Part of the v2.1 explicit marker protocol used by
        ``_effective_calendar_from_transforms`` (M4) — never bare
        attribute access.
        """
        return self.calendar

    # ---- shift / unshift ----

    def shift_date(self, ts: pd.Timestamp) -> pd.Timestamp:
        """Apply offset, then snap if a calendar is configured.

        Scalar input; returns a single timestamp. Collision policy
        does not apply at scalar resolution (no duplicate to detect).
        """
        shifted = ts + self.offset
        if self.calendar is None:
            return shifted
        return self.calendar.snap(shifted, self.snap)

    def unshift_date(self, ts: pd.Timestamp) -> pd.Timestamp:
        """Reverse a previously-applied shift.

        With ``calendar=None`` (the v2.0 contract), this is exact.

        With ``calendar`` set, this is **best-effort and lossy**:
        multiple source dates can snap to the same trading day
        under the forward path, and ``unshift_date`` cannot recover
        which source produced a given output. The implementation
        subtracts the offset and re-snaps in the OPPOSITE direction
        (forward snap on apply → backward snap on unshift), which is
        the closest reversible mapping.

        Users requiring exact invertibility must avoid calendar
        snapping. v2.2 may add a stricter per-run mapping table
        (``{shifted_ts -> original_ts}``) for view-only broker-proxy
        routing — v2.1 deliberately does NOT promise that contract.
        """
        unshifted = ts - self.offset
        if self.calendar is None:
            return unshifted
        opposite = {
            "forward": "backward",
            "backward": "forward",
            "nearest": "nearest",
            "error": "error",
        }[self.snap]
        return self.calendar.snap(unshifted, opposite)

    # ---- frame-level apply ----

    def apply(
        self, data: pd.DataFrame, *, seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Backward-compat wrapper around
        :meth:`apply_with_diagnostics`. Returns just the frame so
        v2.0/v2.0.1/v2.0.2 callers see no API change.
        """
        return self.apply_with_diagnostics(data, seed=seed).data

    def apply_with_diagnostics(
        self, data: pd.DataFrame, *, seed: Optional[int] = None,
    ) -> "TransformApplyResult":
        """Apply the shift; return frame + per-call diagnostics.

        v2.1.0 r4 §3 contract: collision diagnostics are emitted in
        the returned tuple, NOT stored on the transform instance.
        Manifest builders read diagnostics from the per-arm result so
        cross-arm or cross-repeat aliasing is structurally impossible.
        """
        # Capture the original (pre-shift, pre-snap) source index BEFORE
        # any mutation. v2.1.1 #5 fix: previously _resolve_collisions
        # tried to invert the shift+snap by computing
        # ``snapped_target - offset``, which collapses to a single
        # date for every row in a collision group and reports false
        # source dates in the audit manifest. The collision resolver
        # now indexes back into the real source positions.
        source_index = data.index.copy()
        out = data.copy()
        out.index = out.index + self.offset
        diagnostics: list[TransformDiagnostic] = []

        if self.calendar is not None:
            # Snap each shifted date according to the configured
            # snap policy. Vectorise via a list comprehension; the
            # snap is pure-Python per element but daily-frequency
            # frames are small enough that this is fine.
            snapped = pd.DatetimeIndex([
                self.calendar.snap(ts, self.snap) for ts in out.index
            ])
            out.index = snapped

            if not snapped.is_unique:
                out, collision_diag = self._resolve_collisions(
                    out, source_index=source_index, target_index=snapped,
                )
                if collision_diag is not None:
                    diagnostics.append(collision_diag)

        return TransformApplyResult(
            data=out, diagnostics=tuple(diagnostics),
        )

    # ---- collision resolution ----

    def _resolve_collisions(
        self,
        out: pd.DataFrame,
        *,
        source_index: pd.DatetimeIndex,
        target_index: pd.DatetimeIndex,
    ) -> tuple[pd.DataFrame, Optional[TransformDiagnostic]]:
        """Apply ``on_collision`` policy and emit a per-call
        diagnostic.

        v2.1.1 #5: takes both ``source_index`` (pre-shift, pre-snap
        timestamps captured by the caller) and ``target_index`` (the
        snapped output index) so collision examples can correctly
        identify which ORIGINAL source rows collided. The previous
        implementation derived source timestamps via ``idx[p] -
        self.offset`` against the snapped index — but every position
        in a collision group shares the same snapped timestamp, so
        every "source date" collapsed to a single value. The
        manifest's ``source_ts`` / ``kept_source_ts`` /
        ``dropped_source_ts`` fields are audit data and must reflect
        the real pre-snap source rows.

        Returns ``(kept_frame, diagnostic_or_None)``. The diagnostic
        is ``None`` when the collision policy is ``"error"`` (we
        raise before returning) or when no rows were actually
        dropped.

        Diagnostic ``details`` shape (``schema_version="1.0"``):

        - ``on_collision``: the active policy
        - ``collision_count``: number of dropped rows
        - ``collision_group_count``: number of distinct duplicate
          target timestamps
        - ``examples``: list (capped at 10) of
          ``{target_ts, source_ts, kept_source_ts, dropped_source_ts}``;
          values are pd.Timestamp in-memory and stringified by the
          manifest serializer (M10).
        - ``examples_truncated``: bool
        """
        # Find duplicate target timestamps, preserving original input
        # order so keep_first / keep_last semantics are well-defined.
        groups: dict[pd.Timestamp, list[int]] = {}
        for pos, ts in enumerate(target_index):
            groups.setdefault(ts, []).append(pos)

        collision_groups = [
            (target_ts, positions)
            for target_ts, positions in groups.items()
            if len(positions) > 1
        ]

        if not collision_groups:
            return out, None

        if self.on_collision == "error":
            preview = ", ".join(
                target_ts.strftime("%Y-%m-%d")
                for target_ts, _ in collision_groups[:5]
            )
            extra = (
                f" and {len(collision_groups) - 5} more"
                if len(collision_groups) > 5
                else ""
            )
            CalendarSnapCollisionError = _import_calendar_collision_error()
            raise CalendarSnapCollisionError(
                f"DateShift produced {len(collision_groups)} duplicate "
                f"target dates after snapping (preview: {preview}{extra}). "
                "Pass on_collision='keep_first' or 'keep_last' to drop "
                "duplicates explicitly."
            )

        keep_positions: set[int] = set()
        dropped_count = 0
        for target_ts, positions in groups.items():
            if len(positions) == 1:
                keep_positions.add(positions[0])
                continue
            if self.on_collision == "keep_first":
                keep_positions.add(positions[0])
            else:  # keep_last
                keep_positions.add(positions[-1])
            dropped_count += len(positions) - 1

        keep_mask = [
            pos in keep_positions for pos in range(len(target_index))
        ]
        kept = out.iloc[keep_mask]

        examples_capped = collision_groups[:10]
        examples = []
        for target_ts, positions in examples_capped:
            # v2.1.1 #5 fix: index into the REAL source index, never
            # invert from snapped+offset. Multiple positions in one
            # collision group have distinct source rows by definition.
            source_timestamps = [source_index[p] for p in positions]
            if self.on_collision == "keep_first":
                kept_pos = positions[0]
            else:  # keep_last
                kept_pos = positions[-1]
            kept_source = source_index[kept_pos]
            dropped_source = [
                source_index[p] for p in positions if p != kept_pos
            ]
            examples.append({
                "target_ts": target_ts,
                "source_ts": list(source_timestamps),
                "kept_source_ts": kept_source,
                "dropped_source_ts": list(dropped_source),
            })

        diagnostic = TransformDiagnostic(
            code="calendar_snap_collision_rows_dropped",
            source="DateShift",
            severity="warning",
            message=(
                f"DateShift calendar snapping produced "
                f"{len(collision_groups)} duplicate target dates; "
                f"on_collision={self.on_collision!r} dropped "
                f"{dropped_count} source rows. This changes the "
                "transformed sample length."
            ),
            details={
                "schema_version": "1.0",
                "transform": "DateShift",
                "on_collision": self.on_collision,
                "collision_count": dropped_count,
                "collision_group_count": len(collision_groups),
                "examples": examples,
                "examples_truncated": len(collision_groups) > 10,
            },
        )
        return kept, diagnostic


# ---------- Level transforms ----------

class PriceScale:
    """Multiply OHLC prices by a constant factor; volume unchanged.

    Returns are preserved exactly. The plan documents this is *zero*
    leak protection against any agent that reads returns directly.
    Useful as a control / sanity check, not a defense.
    """

    name = "PriceScale"
    category: TransformCategory = "level"
    supports_view_only = True
    supports_market_level = True
    order_invertible = True
    stochastic = False

    def __init__(self, factor: float):
        if factor <= 0:
            raise ValueError(f"PriceScale factor must be > 0, got {factor}")
        self.factor = factor

    def scale_price(self, price: float) -> float:
        return price * self.factor

    def unscale_price(self, masked_price: float) -> float:
        return masked_price / self.factor

    def apply(
        self, data: pd.DataFrame, *, seed: Optional[int] = None,
    ) -> pd.DataFrame:
        out = data.copy()
        for col in ("open", "high", "low", "close"):
            out[col] = out[col] * self.factor
        return out


class PriceRebase:
    """Rescale OHLC so first close equals ``base``.

    A specialization of `PriceScale` parameterized by target base
    rather than factor. The applied factor depends on the source
    data's first close — the user is told via ``last_factor`` after
    the first call, and the plan recommends documenting this so
    downstream consumers know the rebased frame is anchored.

    For ``view_only`` order translation, the factor must be passed
    explicitly (the wrapper holds it after the first apply).
    """

    name = "PriceRebase"
    category: TransformCategory = "level"
    supports_view_only = True
    supports_market_level = True
    order_invertible = True
    stochastic = False

    def __init__(self, base: float = 100.0):
        if base <= 0:
            raise ValueError(f"PriceRebase base must be > 0, got {base}")
        self.base = base
        self.last_factor: Optional[float] = None

    def apply(
        self, data: pd.DataFrame, *, seed: Optional[int] = None,
    ) -> pd.DataFrame:
        first_close = float(data["close"].iloc[0])
        if first_close <= 0:
            raise ValueError(
                f"PriceRebase: first close must be > 0, got {first_close}"
            )
        factor = self.base / first_close
        self.last_factor = factor
        out = data.copy()
        for col in ("open", "high", "low", "close"):
            out[col] = out[col] * factor
        return out


# ---------- Series / noise transforms ----------

class OHLCJitter:
    """Add bp-scale Gaussian noise to OHLC; re-project to preserve
    OHLC ordering invariants.

    For each bar, every price (open, high, low, close) is multiplied
    by ``(1 + N(0, bps/10000))``. After perturbation the bar is
    re-projected so ``high >= max(open, close, low)`` and
    ``low <= min(open, close, high)`` by clamping high upward and
    low downward where needed.

    Volume is unchanged.
    """

    name = "OHLCJitter"
    category: TransformCategory = "series"
    supports_view_only = False
    supports_market_level = True
    order_invertible = False
    stochastic = True

    def __init__(self, bps: float = 5.0, *, preserve_ohlc: bool = True):
        if bps < 0:
            raise ValueError(f"OHLCJitter bps must be >= 0, got {bps}")
        self.bps = float(bps)
        self.preserve_ohlc = preserve_ohlc

    def apply(
        self, data: pd.DataFrame, *, seed: Optional[int] = None,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        sigma = self.bps / 10_000.0
        n = len(data)
        out = data.copy()
        for col in ("open", "high", "low", "close"):
            noise = rng.normal(0.0, sigma, size=n)
            out[col] = out[col] * (1.0 + noise)
        # Guarantee positivity (very small chance of negative under
        # extreme noise; clip to a tiny positive epsilon).
        for col in ("open", "high", "low", "close"):
            out[col] = out[col].clip(lower=1e-8)
        if self.preserve_ohlc:
            o = out["open"].to_numpy()
            c = out["close"].to_numpy()
            h = out["high"].to_numpy()
            low = out["low"].to_numpy()
            # Clamp high up to at least max(o,c,low)
            new_high = np.maximum.reduce([h, o, c, low])
            # Clamp low down to at most min(o,c,high)
            new_low = np.minimum.reduce([low, o, c, new_high])
            out["high"] = new_high
            out["low"] = new_low
        return out


class BlockBootstrap:
    """Stationary block bootstrap of bars, reusing the engine's
    OHLC-reconstruction-aware primitive.

    Delegates to ``aiphaforge.significance._block_bootstrap_indices``
    + ``_reconstruct_ohlcv``.

    Anchoring property: the cumulative-product anchor is
    ``data.iloc[0]['close']``. The output's first close equals the
    anchor multiplied by the first sampled bar's return, so different
    realizations cluster near (but not exactly at) the source's first
    close. The original DatetimeIndex is preserved across all
    realizations.
    """

    name = "BlockBootstrap"
    category: TransformCategory = "series"
    supports_view_only = False
    supports_market_level = True
    order_invertible = False
    stochastic = True

    def __init__(self, block_size: int = 20):
        if block_size < 1:
            raise ValueError(
                f"BlockBootstrap block_size must be >= 1, got {block_size}"
            )
        self.block_size = block_size

    def apply(
        self, data: pd.DataFrame, *, seed: Optional[int] = None,
    ) -> pd.DataFrame:
        if len(data) < 2:
            raise ValueError(
                f"BlockBootstrap requires at least 2 bars, got {len(data)}"
            )
        rng = np.random.default_rng(seed)
        indices = _block_bootstrap_indices(
            n_bars=len(data), block_size=self.block_size, rng=rng
        )
        return _reconstruct_ohlcv(data, indices)


class WindowShuffle:
    """Permute disjoint windows of bars.

    Splits the series into non-overlapping windows of ``window`` bars
    and swaps ``swaps`` random pairs of windows. Within a window the
    bar order is preserved. Trailing partial window (if `n %% window
    != 0`) is left in place.
    """

    name = "WindowShuffle"
    category: TransformCategory = "series"
    supports_view_only = False
    supports_market_level = True
    order_invertible = False
    stochastic = True

    def __init__(self, window: int = 20, swaps: int = 5):
        if window < 1:
            raise ValueError(f"WindowShuffle window must be >= 1, got {window}")
        if swaps < 0:
            raise ValueError(f"WindowShuffle swaps must be >= 0, got {swaps}")
        self.window = window
        self.swaps = swaps

    def apply(
        self, data: pd.DataFrame, *, seed: Optional[int] = None,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = len(data)
        n_full_windows = n // self.window
        if n_full_windows < 2:
            return data.copy()
        # Build the index permutation.
        new_idx = np.arange(n)
        for _ in range(self.swaps):
            i, j = rng.integers(0, n_full_windows, size=2)
            if i == j:
                continue
            si, ei = i * self.window, (i + 1) * self.window
            sj, ej = j * self.window, (j + 1) * self.window
            new_idx[si:ei], new_idx[sj:ej] = (
                new_idx[sj:ej].copy(),
                new_idx[si:ei].copy(),
            )
        out = data.iloc[new_idx].copy()
        # Preserve the original index — only the bar contents are
        # permuted (consistent with BlockBootstrap's contract).
        out.index = data.index
        return out


# ---------- Pipeline + integrity validator ----------

# Canonical stage order per plan §2.4.
_STAGE_ORDER: dict[TransformCategory, int] = {
    "metadata": 0,
    "level": 1,
    "series": 2,
}


@dataclass
class IntegrityCheckResult:
    """Outcome of running OHLC integrity checks on a transformed frame."""

    passed: bool
    errors: list[str]


def validate_ohlcv_integrity(
    data: pd.DataFrame,
    *,
    calendar: Optional["TradingCalendar"] = None,
) -> IntegrityCheckResult:
    """Validate OHLC integrity over the FULL frame (no sampling).

    Per plan §2.4: validation is performed on the **full transformed
    frame**, not on a sampled subset. Even a single bad bar in a
    million invalidates the dataset.

    v2.1 ``calendar=`` (default ``None`` preserves v2.0 contract):
        when set, runs ``calendar.is_conformant`` on the index and
        appends any conformance failures to the returned errors list
        with the same bounded formatting (first 10 dates plus an
        "and K more" tail).
    """
    errors: list[str] = []

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(data.columns)
    if missing:
        errors.append(f"missing OHLCV columns: {sorted(missing)}")
        return IntegrityCheckResult(passed=False, errors=errors)

    for col in ("open", "high", "low", "close", "volume"):
        if data[col].isna().any():
            errors.append(f"column '{col}' contains NaN")
        if np.isinf(data[col].to_numpy()).any():
            errors.append(f"column '{col}' contains inf")

    if not data.index.is_monotonic_increasing:
        errors.append("index is not monotonically increasing")

    o = data["open"].to_numpy()
    h = data["high"].to_numpy()
    low = data["low"].to_numpy()
    c = data["close"].to_numpy()
    v = data["volume"].to_numpy()

    bad_high_mask = h < np.maximum.reduce([o, c, low])
    if bad_high_mask.any():
        errors.append(
            f"OHLC ordering violated on {int(bad_high_mask.sum())} bars: "
            "high < max(open, close, low)"
        )

    bad_low_mask = low > np.minimum.reduce([o, c, h])
    if bad_low_mask.any():
        errors.append(
            f"OHLC ordering violated on {int(bad_low_mask.sum())} bars: "
            "low > min(open, close, high)"
        )

    if (v < 0).any():
        errors.append(f"volume has {int((v < 0).sum())} negative bars")

    # Positive-price guard: any non-positive open/high/low/close is a
    # broken bar (extreme noise after OHLCJitter, bad input, etc.).
    for col, arr in (("open", o), ("high", h), ("low", low), ("close", c)):
        if (arr <= 0).any():
            errors.append(
                f"column '{col}' has {int((arr <= 0).sum())} non-positive values"
            )

    # v2.1 calendar conformance (after the v2.0 invariants so the
    # error list has structural problems first, then calendar.)
    if calendar is not None:
        cal_result = calendar.is_conformant(data.index)
        if not cal_result.passed:
            errors.extend(cal_result.errors)

    return IntegrityCheckResult(passed=not errors, errors=errors)


@dataclass
class TransformPipeline:
    """Validated pipeline of transforms applied in canonical stage
    order.

    Construction-time validation:
    - all transforms must declare valid category / mode / invertibility
    - mode-incompatible transforms are rejected up front
    - non-invertible transforms in ``view_only`` mode require an
      explicit ``agent_contract`` of ``signal_only`` or
      ``market_orders_only``

    Runtime validation:
    - after applying all transforms, the OHLC integrity validator
      runs on the full frame (per §2.4)
    """

    transforms: Sequence[DataTransform]
    mode: Literal["view_only", "market_level"]
    agent_contract: Optional[
        Literal["signal_only", "market_orders_only", "price_orders_allowed"]
    ] = None
    # v2.1 §5.2: optional explicit calendar passed through to the
    # final integrity validator. If callers also include calendar-
    # aware transforms (e.g. DateShift) and the inferred effective
    # calendar disagrees with this explicit one, __post_init__
    # raises CalendarConflictError.
    calendar: Optional["TradingCalendar"] = None

    def __post_init__(self):
        # Validate each transform declares the expected attributes.
        for t in self.transforms:
            for attr in (
                "name", "category", "supports_view_only",
                "supports_market_level", "order_invertible", "stochastic",
            ):
                if not hasattr(t, attr):
                    raise ValueError(
                        f"transform missing required attribute '{attr}': {t!r}"
                    )

        # Mode compatibility.
        for t in self.transforms:
            if self.mode == "view_only" and not t.supports_view_only:
                raise ValueError(
                    f"transform {t.name} does not support view_only mode"
                )
            if self.mode == "market_level" and not t.supports_market_level:
                raise ValueError(
                    f"transform {t.name} does not support market_level mode"
                )

        # In view_only, non-invertible transforms require an explicit
        # agent contract that admits them.
        if self.mode == "view_only":
            non_invertible = [
                t for t in self.transforms if not t.order_invertible
            ]
            if non_invertible:
                if self.agent_contract not in (
                    "signal_only", "market_orders_only",
                ):
                    names = [t.name for t in non_invertible]
                    raise ValueError(
                        f"view_only with non-invertible transforms {names} "
                        "requires agent_contract='signal_only' or "
                        "'market_orders_only'"
                    )

        # v2.1 §5.2: if both an explicit calendar and an inferred
        # effective calendar exist, they must match by stable
        # fingerprint. This catches the case where a user passes
        # `TransformPipeline(calendar=cal_a)` with a transform that
        # also provides `cal_b`. Allow either source individually.
        inferred = _effective_calendar_from_transforms(self.transforms)
        if self.calendar is not None and inferred is not None:
            if self.calendar.stable_fingerprint() != inferred.stable_fingerprint():
                CalendarConflictError = _import_calendar_conflict_error()
                raise CalendarConflictError(
                    f"TransformPipeline.calendar={self.calendar.name!r} "
                    f"disagrees with calendar inferred from transforms "
                    f"({inferred.name!r}). Calendars must match by "
                    "stable_fingerprint() (value equality, not object "
                    "identity)."
                )

    @property
    def stochastic(self) -> bool:
        return any(t.stochastic for t in self.transforms)

    def _stage_sorted(self) -> list[DataTransform]:
        # Stable sort by canonical stage; preserves user order within
        # a stage (so two SymbolMaskers are still applied user-order,
        # though that case is unusual).
        return sorted(self.transforms, key=lambda t: _STAGE_ORDER[t.category])

    def apply(
        self, data: pd.DataFrame, *, seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Apply transforms in canonical stage order, then validate.

        Backward-compatible v2.0/v2.0.1/v2.0.2 surface — returns the
        transformed frame only. New v2.1.0 callers wanting per-call
        diagnostics use :meth:`apply_with_diagnostics`.

        Stochastic transforms each receive a deterministically derived
        sub-seed from the given ``seed``. Sub-seeds are spawned in
        **canonical stage order**, not user input order, so two
        pipelines containing the same transforms in different
        user-supplied orders produce byte-identical output for the
        same outer seed.
        """
        return self.apply_with_diagnostics(data, seed=seed).data

    def apply_with_diagnostics(
        self, data: pd.DataFrame, *, seed: Optional[int] = None,
    ) -> "TransformApplyResult":
        """Apply transforms and collect per-call diagnostics.

        Plan r4-final §3: manifest warnings derive from the returned
        diagnostics tuple, never from mutable transform-instance
        state. Each transform that opts in by implementing
        ``apply_with_diagnostics`` (currently DateShift) contributes
        its diagnostics; other transforms fall back to the bare
        ``apply`` method and contribute nothing extra.
        """
        sorted_transforms = self._stage_sorted()
        rng_seq = np.random.SeedSequence(seed) if seed is not None else None
        children = (
            list(rng_seq.spawn(len(sorted_transforms)))
            if rng_seq is not None
            else [None] * len(sorted_transforms)
        )

        out = data
        diagnostics: list[TransformDiagnostic] = []
        for t, child in zip(sorted_transforms, children):
            sub_seed = (
                int(child.generate_state(1)[0]) if child is not None else None
            )
            try:
                if hasattr(t, "apply_with_diagnostics"):
                    sub_result = t.apply_with_diagnostics(out, seed=sub_seed)
                    out = sub_result.data
                    diagnostics.extend(sub_result.diagnostics)
                else:
                    out = t.apply(out, seed=sub_seed)
            except Exception as e:
                raise type(e)(
                    f"transform '{t.name}' failed on outer seed={seed} "
                    f"sub_seed={sub_seed}: {e}"
                ) from e

        # v2.1: thread the pipeline's effective calendar (explicit
        # or inferred) through to the validator.
        effective_cal = self.calendar
        if effective_cal is None:
            effective_cal = _effective_calendar_from_transforms(self.transforms)
        result = validate_ohlcv_integrity(out, calendar=effective_cal)
        if not result.passed:
            # Identify the last-applied transform so the user knows
            # which stage produced the invalid frame.
            last = sorted_transforms[-1].name if sorted_transforms else "(none)"
            raise ValueError(
                f"Transform pipeline produced invalid OHLCV after "
                f"'{last}' (outer seed={seed}): "
                + "; ".join(result.errors)
            )
        return TransformApplyResult(
            data=out, diagnostics=tuple(diagnostics),
        )
