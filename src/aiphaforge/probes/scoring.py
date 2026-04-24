"""Scoring for Q&A-probe answers.

The engine does not see LLM prompts or reply text — only typed
:class:`AnswerRecord` payloads. Binary and choice answers are scored
by exact match after deterministic normalization; numeric answers are
scored by relative error with per-template tolerance bands; range
answers are scored by containment + width, with a ``max_range_width``
cap that demotes "predict everything" ranges to ``miss``.

Aggregation rules follow `docs/plans/v2.0-plan.md` §1.7:

- ``coverage_rate = submitted / total``
- ``parse_success_rate = valid / submitted``
- ``exact_rate`` etc. computed over valid answers only
- range-width aggregates computed over **valid range-form answers
  only**; ``None`` when no such answers exist
- ``band_index_arbitrary`` is a descriptive weighted score, not a
  probability — users who surface it must render the full
  ``bands_breakdown`` alongside (plan §1.7)
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from statistics import median
from typing import Any, Literal, Optional, Sequence, Union

from aiphaforge.probes.models import (
    AnswerRecord,
    QAProbeReport,
    QuestionScore,
    QuestionSpec,
    TemplateAggregate,
    ToleranceProfile,
)
from aiphaforge.probes.questions import (
    QuestionSet,
    normalize_binary,
    normalize_direction,
)

# ---------- v2.0.1 user-facing parser helpers ----------

# Regex for a signed scientific-notation float. Accepts ASCII '-' and
# Unicode minus '−' (U+2212). Anchored matches happen via .fullmatch
# at use sites; this is the building block.
_NUMBER_RE = re.compile(
    r"[-−]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?"
)

# Currency symbols / unit suffixes to strip before number parsing.
_CURRENCY_SYMBOLS = ("$", "€", "£", "¥", "₿")
_UNIT_SUFFIXES = (
    " usd", " eur", " gbp", " jpy", " btc",
    "usd ", "eur ", "gbp ", "jpy ", "btc ",
)
# Approximation prefixes to strip before number parsing.
_APPROX_PREFIXES = (
    "about", "approximately", "approx", "approx.", "around", "roughly",
    "~", "≈",
)
# "I don't know" / refusal-style replies that always parse to None.
_DONT_KNOW_PATTERNS = frozenset({
    "", "n/a", "na", "none", "null", "unknown", "i don't know",
    "i dont know", "idk", "no answer", "cannot answer", "can't answer",
})


def _normalize_minus(s: str) -> str:
    """Convert Unicode minus (U+2212) and en/em-dash to ASCII '-'.

    Used for the *number-parsing* path only; the range-detection path
    handles em-dash separately via a dedicated split rule.
    """
    return s.replace("\u2212", "-")


def _strip_currency_and_units(s: str) -> str:
    out = s
    for sym in _CURRENCY_SYMBOLS:
        out = out.replace(sym, "")
    lo = out.lower()
    for suf in _UNIT_SUFFIXES:
        lo = lo.replace(suf, " ")
    return lo


def _strip_approximation(s: str) -> str:
    out = s.strip().lower()
    for pref in _APPROX_PREFIXES:
        if out.startswith(pref):
            out = out[len(pref):].lstrip(" \t.")
    return out


def _try_float(token: str) -> Optional[float]:
    """Parse a single numeric token to float, returning None on failure.

    Handles thousands separators and Unicode minus.
    """
    cleaned = _normalize_minus(token).replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_numbers(s: str) -> list[float]:
    """Find all numeric tokens in `s` and return them parsed."""
    out: list[float] = []
    for m in _NUMBER_RE.finditer(_normalize_minus(s)):
        v = _try_float(m.group(0))
        if v is not None:
            out.append(v)
    return out


def _try_range(s: str) -> Optional[tuple[float, float]]:
    """Detect a numeric-range answer; return (lo, hi) or None.

    Conventions (documented in the public docstring):
    - Bracketed: ``[lo, hi]`` or ``(lo, hi)``
    - Worded: ``lo to hi``, ``between lo and hi``, ``from lo to hi``
    - Hyphen with or without spaces: ``lo-hi`` and ``lo - hi`` are
      both ranges (r5 §2.1: parser is an answer parser, not an
      expression evaluator).
    - En/em-dash with or without spaces: ``lo–hi``, ``lo — hi``.
    """
    raw = s.strip()
    norm = raw.lower()

    # Bracketed forms.
    bracket_match = re.fullmatch(
        r"[\[(]\s*([-−]?\d[\d.,eE+\-−]*)\s*,\s*([-−]?\d[\d.,eE+\-−]*)\s*[\])]",
        raw,
    )
    if bracket_match:
        lo = _try_float(bracket_match.group(1))
        hi = _try_float(bracket_match.group(2))
        if lo is not None and hi is not None:
            return (lo, hi)

    # Worded forms.
    for pattern in (
        r"^\s*from\s+(.+?)\s+to\s+(.+?)\s*$",
        r"^\s*between\s+(.+?)\s+and\s+(.+?)\s*$",
        r"^\s*(.+?)\s+to\s+(.+?)\s*$",
    ):
        m = re.match(pattern, norm)
        if m:
            lo = _try_float(m.group(1))
            hi = _try_float(m.group(2))
            if lo is not None and hi is not None:
                return (lo, hi)

    # En/em-dash range (with or without surrounding spaces).
    em_match = re.fullmatch(
        r"\s*([-−]?\d[\d.,eE+\-−]*)\s*[\u2013\u2014]\s*([-−]?\d[\d.,eE+\-−]*)\s*",
        raw,
    )
    if em_match:
        lo = _try_float(em_match.group(1))
        hi = _try_float(em_match.group(2))
        if lo is not None and hi is not None:
            return (lo, hi)

    # Hyphen, with OR without surrounding spaces (r5 §2.1).
    direct = re.fullmatch(
        r"\s*([-−]?\d[\d.,eE+]*(?:\.\d+)?)\s*-\s*(\d[\d.,eE+]*(?:\.\d+)?)\s*",
        raw,
    )
    if direct:
        lo = _try_float(direct.group(1))
        hi = _try_float(direct.group(2))
        if lo is not None and hi is not None:
            return (lo, hi)

    return None


def _strict_raise(stage: str, text: Any, suggestion: str) -> None:
    """Build the standard strict-mode ValueError."""
    raw = repr(text)
    if len(raw) > 200:
        raw = raw[:197] + "..."
    raise ValueError(
        f"parse failed at {stage!r} on input {raw}; {suggestion}"
    )


def parse_numeric_answer(
    text: Optional[str],
    *,
    strict: bool = False,
    permissive: bool = False,
    percent: Literal["reject", "decimal", "number"] = "reject",
) -> Union[float, tuple[float, float], None]:
    """Parse a raw LLM reply string into a typed numeric value.

    Returns ``float`` for scalars, ``tuple[float, float]`` for ranges,
    or ``None`` for unparseable input. Pure function — no side effects,
    no LLM call.

    Hyphen handling (r5 §2.1):
        Hyphen-separated two-number answers are ranges, not
        arithmetic. Both ``"172-175"`` and ``"172 - 175"`` parse to
        ``(172.0, 175.0)``. The parser is an answer parser, not an
        expression evaluator. Callers expecting subtraction must
        pre-evaluate before calling this helper.

    Percent handling (``percent``, r5 §2.1):
        Inputs ending in ``%`` are ambiguous — the downstream
        template may want a decimal (``0.025``) or a number
        (``2.5``). Default ``"reject"`` returns ``None`` (or raises
        in strict mode). ``"decimal"`` divides by 100; ``"number"``
        keeps the raw value. Apply consistently per-template.

    Hedging signal:
        ``"about 172"`` returns ``172.0`` and **drops the hedging
        signal**. Users who want to surface "this answer was hedged"
        must re-scan ``raw_answer`` separately and stash the flag on
        ``AnswerRecord.metadata``.

    Multi-number replies (``permissive``):
        Frontier LLMs frequently emit conversational replies like
        ``"the open was 172.06 and the close was 175.30"``. With the
        default ``permissive=False`` this returns ``None`` — the
        parser refuses to guess. With ``permissive=True``, the first
        numeric token wins.

    Strict mode (``strict``):
        Raises ``ValueError`` instead of returning ``None`` when the
        input cannot be parsed. The error message includes the
        verbatim input (truncated to 200 chars), the parse stage that
        failed, and a one-line suggested fix.
    """
    if text is None:
        if strict:
            _strict_raise("input-presence", text, "input is None")
        return None

    raw = str(text).strip()
    if raw.lower() in _DONT_KNOW_PATTERNS:
        if strict:
            _strict_raise(
                "don't-know-detection", text,
                f"input matches don't-know pattern {raw!r}",
            )
        return None

    # Percent handling — detect a trailing ``%`` and dispatch on the
    # caller's policy. Done early because percent strings should not
    # fall through to currency/range stripping.
    pct_match = re.fullmatch(
        r"\s*([-−]?\d[\d.,eE+\-−]*)\s*%\s*", raw,
    )
    if pct_match:
        if percent == "reject":
            if strict:
                _strict_raise(
                    "percent-policy", text,
                    "percent sign present but percent='reject'; "
                    "pass percent='decimal' (divide by 100) or "
                    "percent='number' (keep raw) to accept",
                )
            return None
        v = _try_float(pct_match.group(1))
        if v is None:
            if strict:
                _strict_raise(
                    "percent-extraction", text,
                    "could not extract a number before %",
                )
            return None
        return v / 100.0 if percent == "decimal" else v

    # 1. Try range detection BEFORE scalar parsing so "172-175" wins
    #    over "172" (first-number-extraction).
    pre_clean = _strip_approximation(_strip_currency_and_units(raw))
    rng = _try_range(pre_clean)
    if rng is None:
        # Also try on the un-stripped raw input (some range forms
        # don't survive currency stripping unchanged).
        rng = _try_range(raw)
    if rng is not None:
        return rng

    # 2. Scalar parsing path.
    cleaned = _strip_approximation(_strip_currency_and_units(raw))
    cleaned = _normalize_minus(cleaned)

    numbers = _extract_numbers(cleaned)
    if len(numbers) == 0:
        if strict:
            _strict_raise(
                "number-extraction", text,
                "no numeric tokens detected",
            )
        return None
    if len(numbers) == 1:
        return numbers[0]

    # Multi-number reply.
    if permissive:
        return numbers[0]
    if strict:
        _strict_raise(
            "multi-token disambiguation", text,
            f"{len(numbers)} numeric tokens detected; pass "
            "permissive=True to take the first or strip the "
            "surrounding context before parsing",
        )
    return None


def parse_choice_answer(
    text: Optional[str],
    allowed: Sequence[str],
    *,
    strict: bool = False,
) -> Optional[str]:
    """Parse a raw LLM reply into one of `allowed`, or None.

    Wraps the internal :func:`_normalize_choice` (which dispatches to
    direction-aware normalisation when ``allowed`` is the canonical
    up/down/unchanged vocabulary) with a public, strict-aware API
    symmetric to :func:`parse_numeric_answer`.

    Hedging tokens (e.g., ``"probably yes"``, ``"likely up"``) parse
    to ``None`` — the user is responsible for upstream pre-stripping
    if they want to honor hedges.
    """
    if text is None:
        if strict:
            _strict_raise("input-presence", text, "input is None")
        return None
    if str(text).strip().lower() in _DONT_KNOW_PATTERNS:
        if strict:
            _strict_raise(
                "don't-know-detection", text,
                "input matches don't-know pattern",
            )
        return None
    out = _normalize_choice(str(text), allowed)
    if out is None and strict:
        _strict_raise(
            "choice-match", text,
            f"input did not match any allowed value in {list(allowed)!r}",
        )
    return out


def parse_binary_answer(
    text: Optional[str],
    *,
    strict: bool = False,
) -> Optional[bool]:
    """Parse a raw LLM reply into True / False / None.

    Wraps :func:`normalize_binary` with a strict-aware API symmetric
    to :func:`parse_numeric_answer` and :func:`parse_choice_answer`.
    Directional aliases (``"up"``, ``"higher"``) deliberately do NOT
    map to True — see the v2.0 fix in
    `aiphaforge.probes.questions.normalize_binary`.
    """
    if text is None:
        if strict:
            _strict_raise("input-presence", text, "input is None")
        return None
    if str(text).strip().lower() in _DONT_KNOW_PATTERNS:
        if strict:
            _strict_raise(
                "don't-know-detection", text,
                "input matches don't-know pattern",
            )
        return None
    out = normalize_binary(str(text))
    if out is None and strict:
        _strict_raise(
            "binary-match", text,
            "input did not normalize to yes/true/1 or no/false/0",
        )
    return out

# Weights for the descriptive `band_index_arbitrary` aggregate.
_BAND_WEIGHTS = {"exact": 1.00, "near": 0.67, "rough": 0.33, "miss": 0.00}


def _normalize_choice(text: str, allowed: Sequence[str]) -> Optional[str]:
    """Normalize a choice answer. Returns the matching `allowed` value or None.

    Dispatches to :func:`normalize_direction` when the allowed set is
    the canonical up/down/unchanged vocabulary; falls back to
    case-insensitive exact match for other choice templates. The
    returned value is always one of the strings in ``allowed`` (or
    ``None``), preserving the template's original case.
    """
    allowed_set = {c.lower() for c in allowed}
    if allowed_set == {"up", "down", "unchanged"}:
        # Map the canonical direction back to the user's exact `allowed`
        # casing so downstream `parsed in allowed` checks succeed even
        # when the template uses, e.g., ["UP", "DOWN", "UNCHANGED"].
        canonical = normalize_direction(text)
        if canonical is None:
            return None
        for c in allowed:
            if c.lower() == canonical:
                return c
        return None
    if text is None:
        return None
    cand = str(text).strip().lower()
    for c in allowed:
        if cand == c.lower():
            return c
    return None


def _score_binary(qs: QuestionSpec, ans: AnswerRecord) -> QuestionScore:
    parsed = normalize_binary(str(ans.parsed_answer)) \
        if ans.parsed_answer is not None else None
    if parsed is None:
        return _invalid_score(qs, ans)
    correct = bool(parsed) == bool(qs.truth_value)
    return QuestionScore(
        question_id=qs.question_id,
        validity="valid",
        band="exact" if correct else "miss",
        truth_value=qs.truth_value,
        parsed_answer=parsed,
        relative_error=None,
        contains_truth=None,
        range_width_ratio=None,
        max_range_width_exceeded=None,
    )


def _score_choice(qs: QuestionSpec, ans: AnswerRecord) -> QuestionScore:
    allowed = qs.choices or []
    parsed = _normalize_choice(str(ans.parsed_answer), allowed) \
        if ans.parsed_answer is not None else None
    if parsed is None or parsed not in allowed:
        return _invalid_score(qs, ans)
    correct = parsed == qs.truth_value
    return QuestionScore(
        question_id=qs.question_id,
        validity="valid",
        band="exact" if correct else "miss",
        truth_value=qs.truth_value,
        parsed_answer=parsed,
        relative_error=None,
        contains_truth=None,
        range_width_ratio=None,
        max_range_width_exceeded=None,
    )


def _score_scalar_numeric(
    qs: QuestionSpec, ans: AnswerRecord, tol: ToleranceProfile, a: float,
) -> QuestionScore:
    t = float(qs.truth_value)
    metadata: dict[str, Any] = {}

    # Sign error is qualitatively different from magnitude error:
    # map it to band "miss" and surface via metadata flag per plan §1.6.
    if tol.sign_sensitive:
        if (
            abs(t) > tol.sign_epsilon
            and abs(a) > tol.sign_epsilon
            and ((t > 0) != (a > 0))
        ):
            metadata["sign_error"] = True
            scale = max(abs(t), abs(a), tol.absolute_floor)
            return QuestionScore(
                question_id=qs.question_id,
                validity="valid",
                band="miss",
                truth_value=t,
                parsed_answer=a,
                relative_error=abs(a - t) / scale,
                contains_truth=None,
                range_width_ratio=None,
                max_range_width_exceeded=None,
                metadata=metadata,
            )

    scale = max(abs(t), abs(a), tol.absolute_floor)
    rel = abs(a - t) / scale
    if rel <= tol.exact_threshold:
        band = "exact"
    elif rel <= tol.near_threshold:
        band = "near"
    elif rel <= tol.rough_threshold:
        band = "rough"
    else:
        band = "miss"
    return QuestionScore(
        question_id=qs.question_id,
        validity="valid",
        band=band,
        truth_value=t,
        parsed_answer=a,
        relative_error=rel,
        contains_truth=None,
        range_width_ratio=None,
        max_range_width_exceeded=None,
        metadata=metadata,
    )


def _score_range_numeric(
    qs: QuestionSpec,
    ans: AnswerRecord,
    tol: ToleranceProfile,
    lo: float,
    hi: float,
) -> QuestionScore:
    if lo > hi:
        return _invalid_score(qs, ans, reason="lo > hi")
    if lo == hi:
        # Degenerate range → treat as scalar.
        return _score_scalar_numeric(qs, ans, tol, lo)
    t = float(qs.truth_value)
    scale = max(abs(t), abs(lo), abs(hi), tol.absolute_floor)
    contains = lo <= t <= hi
    width_ratio = (hi - lo) / scale
    if contains:
        max_exceeded = (
            tol.max_range_width is not None
            and width_ratio > tol.max_range_width
        )
        if max_exceeded:
            band = "miss"
        elif width_ratio <= tol.exact_range_width:
            band = "exact"
        elif width_ratio <= tol.near_range_width:
            band = "near"
        else:
            band = "rough"
        return QuestionScore(
            question_id=qs.question_id,
            validity="valid",
            band=band,
            truth_value=t,
            parsed_answer=(lo, hi),
            relative_error=None,
            contains_truth=True,
            range_width_ratio=width_ratio,
            max_range_width_exceeded=max_exceeded,
        )
    # Truth outside the range.
    outside_distance = min(abs(t - lo), abs(t - hi))
    outside_ratio = outside_distance / scale
    band = "rough" if outside_ratio <= tol.rough_threshold else "miss"
    return QuestionScore(
        question_id=qs.question_id,
        validity="valid",
        band=band,
        truth_value=t,
        parsed_answer=(lo, hi),
        relative_error=outside_ratio,
        contains_truth=False,
        range_width_ratio=width_ratio,
        max_range_width_exceeded=False,
    )


def _score_numeric(qs: QuestionSpec, ans: AnswerRecord) -> QuestionScore:
    if qs.tolerance is None:
        # A numeric template without a tolerance profile is a bug —
        # built-ins must ship one. Don't silently score.
        return _invalid_score(qs, ans, reason="no tolerance profile on numeric question")
    parsed = ans.parsed_answer
    if parsed is None:
        return _invalid_score(qs, ans)
    # Scalar?
    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
        try:
            return _score_scalar_numeric(qs, ans, qs.tolerance, float(parsed))
        except (TypeError, ValueError):
            return _invalid_score(qs, ans)
    # Range? Accept (lo, hi) tuple or 2-element list.
    if (
        isinstance(parsed, (tuple, list))
        and len(parsed) == 2
        and all(isinstance(v, (int, float)) for v in parsed)
    ):
        return _score_range_numeric(
            qs, ans, qs.tolerance, float(parsed[0]), float(parsed[1])
        )
    return _invalid_score(qs, ans, reason="unrecognized numeric answer shape")


def _invalid_score(
    qs: QuestionSpec, ans: AnswerRecord, reason: Optional[str] = None,
) -> QuestionScore:
    metadata: dict[str, Any] = {}
    if reason is not None:
        metadata["invalid_reason"] = reason
    if ans.parse_status == "refusal":
        validity = "refusal"
    elif ans.parse_status == "missing":
        validity = "missing"
    else:
        validity = "invalid"
    return QuestionScore(
        question_id=qs.question_id,
        validity=validity,
        band="invalid",
        truth_value=qs.truth_value,
        parsed_answer=ans.parsed_answer,
        relative_error=None,
        contains_truth=None,
        range_width_ratio=None,
        max_range_width_exceeded=None,
        metadata=metadata,
    )


def score_question(qs: QuestionSpec, ans: AnswerRecord) -> QuestionScore:
    """Score a single question against a user-supplied answer record.

    Raises ``ValueError`` if ``qs.question_id != ans.question_id``: the
    pairing contract is enforced eagerly so an off-by-one in a list-zip
    cannot silently produce wrong scores.
    """
    if qs.question_id != ans.question_id:
        raise ValueError(
            f"score_question: question/answer id mismatch — "
            f"qs.question_id={qs.question_id!r} but "
            f"ans.question_id={ans.question_id!r}"
        )
    if ans.parse_status in ("refusal", "missing"):
        return _invalid_score(qs, ans)
    if ans.parse_status == "invalid":
        return _invalid_score(qs, ans, reason="user marked invalid")
    if qs.answer_type == "binary":
        return _score_binary(qs, ans)
    if qs.answer_type == "choice":
        return _score_choice(qs, ans)
    if qs.answer_type == "numeric":
        return _score_numeric(qs, ans)
    return _invalid_score(qs, ans, reason=f"unknown answer_type {qs.answer_type}")


# ---------- Aggregation ----------

def _band_counts(scores: Sequence[QuestionScore]) -> dict[str, int]:
    counts = {"exact": 0, "near": 0, "rough": 0, "miss": 0, "invalid": 0}
    for s in scores:
        counts[s.band] = counts.get(s.band, 0) + 1
    return counts


def _band_index(counts: dict[str, int], valid_n: int) -> Optional[float]:
    if valid_n <= 0:
        return None
    weighted = sum(_BAND_WEIGHTS[b] * counts.get(b, 0) for b in _BAND_WEIGHTS)
    return weighted / valid_n


def aggregate_scores(
    question_set: QuestionSet,
    scores: Sequence[QuestionScore],
    *,
    manifest: Optional[dict[str, Any]] = None,
) -> QAProbeReport:
    """Aggregate per-question scores into a :class:`QAProbeReport`.

    Consumes every question in ``question_set``; scores for questions
    not present in ``scores`` are treated as ``missing``.
    """
    score_by_id = {s.question_id: s for s in scores}
    total = len(question_set)
    submitted = 0
    valid = 0
    invalid = 0
    missing = 0
    refusal = 0
    all_scores: list[QuestionScore] = []
    range_widths: list[float] = []
    max_range_exceeded_count = 0

    # Per-template / per-symbol / per-period breakdowns are kept as
    # plain dicts (avoiding the optional pandas import for a thin
    # aggregation). Callers wanting a DataFrame can build one from
    # `by_template` / `by_symbol` / `by_period` themselves; v2.0
    # returns None for these DataFrames to keep the report JSON-native.
    for q in question_set:
        s = score_by_id.get(q.question_id)
        if s is None:
            s = QuestionScore(
                question_id=q.question_id,
                validity="missing",
                band="invalid",
                truth_value=q.truth_value,
                parsed_answer=None,
                relative_error=None,
                contains_truth=None,
                range_width_ratio=None,
                max_range_width_exceeded=None,
                metadata={"invalid_reason": "no answer submitted"},
            )
            missing += 1
        else:
            submitted += 1
            if s.validity == "valid":
                valid += 1
            elif s.validity == "refusal":
                refusal += 1
            elif s.validity == "missing":
                missing += 1
            else:
                invalid += 1
            if s.range_width_ratio is not None:
                range_widths.append(s.range_width_ratio)
                if s.max_range_width_exceeded:
                    max_range_exceeded_count += 1
        all_scores.append(s)

    counts = _band_counts([s for s in all_scores if s.validity == "valid"])
    coverage = submitted / total if total else 0.0
    parse_ok = valid / submitted if submitted else 0.0

    def _rate(band: str) -> float:
        return counts.get(band, 0) / valid if valid else 0.0

    mean_width = sum(range_widths) / len(range_widths) if range_widths else None
    median_width = median(range_widths) if range_widths else None

    # Per-template breakdown (v2.0.1 r5, A5). Typed dict keyed by
    # template_id; load-bearing for selective-memorization detection.
    # We pair each question with its score to bucket by template_id.
    by_template: Optional[dict[str, TemplateAggregate]] = None
    if total > 0:
        templates: dict[str, list[QuestionScore]] = {}
        for q, s in zip(question_set, all_scores):
            templates.setdefault(q.template_id, []).append(s)
        by_template = {}
        for tid, tscores in templates.items():
            t_n = len(tscores)
            t_valid_list = [s for s in tscores if s.validity == "valid"]
            t_valid = len(t_valid_list)
            t_invalid = sum(
                1 for s in tscores
                if s.validity not in ("valid", "missing", "refusal")
            )
            t_missing = sum(1 for s in tscores if s.validity == "missing")
            t_refusal = sum(1 for s in tscores if s.validity == "refusal")
            # ``submitted`` excludes the synthetic missing rows we
            # added when no answer was sent at all. The score-side
            # ``validity == "missing"`` covers both the synthetic
            # rows and any caller-supplied missing answers; per
            # plan §6.2 we surface coverage as
            # submitted/n_questions and parse-success as
            # valid/max(submitted, 1).
            t_submitted = t_n - t_missing
            t_coverage = t_submitted / t_n if t_n else 0.0
            t_parse_ok = t_valid / t_submitted if t_submitted else 0.0
            tcounts = _band_counts(t_valid_list)
            t_widths = [
                s.range_width_ratio for s in tscores
                if s.range_width_ratio is not None
            ]
            t_exceeded = sum(
                1 for s in tscores if s.max_range_width_exceeded
            )

            def _trate(band: str, denom: int = t_valid) -> Optional[float]:
                # Plan §6.2: zero-valid templates report None for band
                # rates so they aren't confused with all-miss reports.
                if denom == 0:
                    return None
                return tcounts.get(band, 0) / denom

            by_template[tid] = TemplateAggregate(
                n_questions=t_n,
                submitted_answers=t_submitted,
                valid_answers=t_valid,
                invalid_answers=t_invalid,
                missing_answers=t_missing,
                refusal_answers=t_refusal,
                coverage_rate=t_coverage,
                parse_success_rate=t_parse_ok,
                exact_rate=_trate("exact"),
                near_rate=_trate("near"),
                rough_rate=_trate("rough"),
                miss_rate=_trate("miss"),
                band_index_arbitrary=_band_index(tcounts, t_valid),
                bands_breakdown=tcounts,
                mean_range_width_ratio=(
                    sum(t_widths) / len(t_widths) if t_widths else None
                ),
                median_range_width_ratio=(
                    median(t_widths) if t_widths else None
                ),
                max_range_width_exceeded_count=t_exceeded,
            )

    return QAProbeReport(
        total_questions=total,
        submitted_answers=submitted,
        valid_answers=valid,
        invalid_answers=invalid,
        missing_answers=missing,
        refusal_answers=refusal,
        coverage_rate=coverage,
        parse_success_rate=parse_ok,
        exact_rate=_rate("exact"),
        near_rate=_rate("near"),
        rough_rate=_rate("rough"),
        miss_rate=_rate("miss"),
        band_index_arbitrary=_band_index(counts, valid),
        bands_breakdown=counts,
        mean_range_width_ratio=mean_width,
        median_range_width_ratio=median_width,
        max_range_width_exceeded_count=max_range_exceeded_count,
        by_template=by_template,
        by_symbol=None,
        by_period=None,
        question_scores=all_scores,
        manifest=dict(manifest or {}),
    )


# ---------- File-driven workflow ----------

def _parse_answer_payload(
    answer_type: str, raw_parsed: Any,
) -> Any:
    """Coerce a JSON-decoded `parsed_answer` into the engine's expected shape."""
    if answer_type == "numeric":
        # JSON arrays come back as list; convert 2-element to tuple.
        if isinstance(raw_parsed, list) and len(raw_parsed) == 2:
            return (float(raw_parsed[0]), float(raw_parsed[1]))
    return raw_parsed


def _merge_provider_config(
    manifest: Optional[dict[str, Any]],
    provider_config: Optional[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Merge a `provider_config` kwarg into the report manifest.

    Rules per plan §3 (r5):
    - Keys present in only one source pass through.
    - Keys present in BOTH with the same value → merged silently.
    - Keys present in BOTH with different values → ValueError naming
      the conflicting key + both values.
    - Provenance per key (``"manifest"`` / ``"kwarg"`` / ``"both"``)
      is recorded under ``manifest["provider_config_provenance"]``.
    - Empty-bucket rule: do not write or preserve `provider_config` /
      `provider_config_provenance` unless the merged config has at
      least one key. If the input manifest contains literal empty
      buckets and merge is empty, strip them from the output.

    Returns ``(merged_manifest, provenance)``. Both are ``None`` when
    no manifest material at all was supplied.
    """
    if not manifest and not provider_config:
        return None, None
    out = dict(manifest or {})
    existing_pc = dict(out.get("provider_config") or {})
    incoming_pc = dict(provider_config or {})
    provenance: dict[str, str] = {}
    merged: dict[str, Any] = {}
    all_keys = set(existing_pc) | set(incoming_pc)
    for key in sorted(all_keys):
        in_existing = key in existing_pc
        in_incoming = key in incoming_pc
        if in_existing and in_incoming:
            v_existing = existing_pc[key]
            v_incoming = incoming_pc[key]
            if v_existing != v_incoming:
                raise ValueError(
                    f"provider_config collision on key {key!r}: "
                    f"manifest has {v_existing!r}, kwarg has {v_incoming!r}. "
                    "Same-key conflicts must be resolved by the caller; "
                    "remove one source or align the values."
                )
            merged[key] = v_existing
            provenance[key] = "both"
        elif in_existing:
            merged[key] = existing_pc[key]
            provenance[key] = "manifest"
        else:
            merged[key] = incoming_pc[key]
            provenance[key] = "kwarg"
    if merged:
        out["provider_config"] = merged
        out["provider_config_provenance"] = provenance
        return out, provenance
    out.pop("provider_config", None)
    out.pop("provider_config_provenance", None)
    return out, None


def score_answer_file(
    question_set: QuestionSet,
    answers_path: str,
    *,
    manifest: Optional[dict[str, Any]] = None,
    provider_config: Optional[dict[str, Any]] = None,
) -> QAProbeReport:
    """Score a JSONL file of :class:`AnswerRecord` rows.

    Each line is a JSON object matching the AnswerRecord shape:
    ``{"question_id": ..., "raw_answer": ..., "parsed_answer": ...,
    "parse_status": "valid|invalid|missing|refusal", "metadata": {...}}``.

    Rows with ``question_id`` not present in ``question_set`` are
    silently ignored (they may have come from a different question
    set). Questions in the set with no matching answer row are
    treated as ``missing`` by :func:`aggregate_scores`.

    ``provider_config`` (v2.0.1) lets callers attach the user's
    LLM configuration for cross-paper comparability. Recommended
    keys live in :data:`aiphaforge.probes.models.RECOMMENDED_PROVIDER_CONFIG_KEYS`
    (e.g., ``model``, ``snapshot_id``, ``temperature``,
    ``prompt_template_hash``, ``prompt_cache_disclosed``,
    ``system_fingerprint``). When both ``manifest["provider_config"]``
    and ``provider_config=`` carry the same key with different
    values, a ``ValueError`` is raised — never silent merge.
    """
    merged_manifest, _ = _merge_provider_config(manifest, provider_config)
    qs_by_id = {q.question_id: q for q in question_set}
    scores: list[QuestionScore] = []
    with open(answers_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("question_id")
            if qid not in qs_by_id:
                continue
            qs = qs_by_id[qid]
            parsed = _parse_answer_payload(
                qs.answer_type, obj.get("parsed_answer")
            )
            ans = AnswerRecord(
                question_id=qid,
                raw_answer=obj.get("raw_answer"),
                parsed_answer=parsed,
                parse_status=obj.get("parse_status", "invalid"),
                metadata=obj.get("metadata", {}) or {},
            )
            scores.append(score_question(qs, ans))
    return aggregate_scores(question_set, scores, manifest=merged_manifest)


# Re-export a couple of utilities for users wiring their own scorers.
__all__ = [
    "aggregate_scores",
    "normalize_binary",
    "normalize_direction",
    "parse_binary_answer",
    "parse_choice_answer",
    "parse_numeric_answer",
    "score_answer_file",
    "score_question",
]


# ---------- Serialization helper (answers file) ----------

def serialize_answer_records(
    answers: Sequence[AnswerRecord],
    path: str,
) -> None:
    """Write a JSONL answers file from a list of AnswerRecord.

    Intended as a convenience for users who already have
    AnswerRecord objects in memory (e.g., after running a user-side
    parser); the engine-side workflow only requires that the file
    exist in the right shape for :func:`score_answer_file`.
    """
    with open(path, "w") as f:
        for a in answers:
            d = asdict(a)
            # JSON doesn't support tuples; convert (lo, hi) numeric
            # ranges to a 2-element list for round-trip compatibility.
            pa = d.get("parsed_answer")
            if isinstance(pa, tuple):
                d["parsed_answer"] = list(pa)
            f.write(json.dumps(d) + "\n")
