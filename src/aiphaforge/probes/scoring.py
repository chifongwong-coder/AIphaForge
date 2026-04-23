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
from dataclasses import asdict
from statistics import median
from typing import Any, Optional, Sequence

from aiphaforge.probes.models import (
    AnswerRecord,
    QAProbeReport,
    QuestionScore,
    QuestionSpec,
    ToleranceProfile,
)
from aiphaforge.probes.questions import (
    QuestionSet,
    normalize_binary,
    normalize_direction,
)

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
        by_template=None,
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


def score_answer_file(
    question_set: QuestionSet,
    answers_path: str,
    *,
    manifest: Optional[dict[str, Any]] = None,
) -> QAProbeReport:
    """Score a JSONL file of :class:`AnswerRecord` rows.

    Each line is a JSON object matching the AnswerRecord shape:
    ``{"question_id": ..., "raw_answer": ..., "parsed_answer": ...,
    "parse_status": "valid|invalid|missing|refusal", "metadata": {...}}``.

    Rows with ``question_id`` not present in ``question_set`` are
    silently ignored (they may have come from a different question
    set). Questions in the set with no matching answer row are
    treated as ``missing`` by :func:`aggregate_scores`.
    """
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
    return aggregate_scores(question_set, scores, manifest=manifest)


# Re-export a couple of utilities for users wiring their own scorers.
__all__ = [
    "aggregate_scores",
    "normalize_binary",
    "normalize_direction",
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
