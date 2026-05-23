"""v2.2 M6: knowledge_check() orchestrator.

Pillar non-transitivity: this module produces Q&A memorization
diagnostics only. It does NOT imply or refute results from the
obfuscation_bootstrap (v2.0 anonymization pillar) or the
differential_bootstrap (v2.1) — these are different LLM invocation
patterns measuring different things. The same agent passing one
does not imply it passes another.

See ``docs/plans/v2.2-plan-r6.md`` § 7.
"""
from __future__ import annotations

import math
import uuid
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import pandas as pd

from aiphaforge.probes._continuation import ContinuationProbe
from aiphaforge.probes._hash_utils import _normalize_for_hash
from aiphaforge.probes._rank import (
    RankContinuationProbe,
    resolve_rank_answer,
    score_rank_answer,
)
from aiphaforge.probes._vol import apply_vol_scaling_to_question_set
from aiphaforge.probes.models import (
    AnswerRecord,
    AttestedAnswers,
    QAProbeReport,
    QuestionSpec,
)
from aiphaforge.probes.questions import KnowledgeProbe, QuestionSet
from aiphaforge.probes.scoring import score_attested_answers

# ---------- provider_config validation ----------

# Required keys for paper-grade reproducibility (per § 7.2.1).
_PROVIDER_CONFIG_REQUIRED_KEYS = (
    "temperature",
    "top_p",
    "model_id",
    "model_version",
    "max_output_tokens",
    "tokenizer_id",
    "seed",
    "reasoning_effort",
    "stop_sequences",
)
# Nullable keys accept either their typed value or None (sentinel
# "key intentionally unused"). All other keys are non-nullable.
_PROVIDER_CONFIG_NULLABLE = frozenset({
    "seed", "reasoning_effort", "stop_sequences",
})


def _validate_provider_config(provider_config: dict) -> None:
    """Raise ValueError on missing keys; raise TypeError on
    non-None / non-typed values for nullable keys.
    """
    missing = [
        k for k in _PROVIDER_CONFIG_REQUIRED_KEYS
        if k not in provider_config
    ]
    if missing:
        raise ValueError(
            f"provider_config missing required keys: {missing}. "
            f"All 9 keys are required: "
            f"{list(_PROVIDER_CONFIG_REQUIRED_KEYS)}."
        )
    # Nullable-key type check: must be either None or the typed value.
    for k in _PROVIDER_CONFIG_NULLABLE:
        v = provider_config[k]
        if v is None:
            continue
        if k == "seed" and not isinstance(v, int):
            raise TypeError(
                f"provider_config[{k!r}] must be int or None; "
                f"got {type(v).__name__}={v!r}"
            )
        if k == "reasoning_effort" and not isinstance(v, str):
            raise TypeError(
                f"provider_config[{k!r}] must be str or None; "
                f"got {type(v).__name__}={v!r}"
            )
        if k == "stop_sequences" and not isinstance(v, (tuple, list)):
            raise TypeError(
                f"provider_config[{k!r}] must be tuple/list or None; "
                f"got {type(v).__name__}={v!r}"
            )


# ---------- Refusal heuristic ----------


_REFUSAL_KEYWORDS_LEADING = (
    # canonical refusal openings
    "i don't have access", "i do not have access",
    "i cannot answer", "i can't answer",
    "i don't have information", "i do not have information",
    "i'm not able to", "i am not able to",
    "i'm unable to", "i am unable to",
    "i don't know", "i do not know",
    "i'm not sure", "i am not sure",
    "i couldn't find", "i could not find",
    "no data available",
    "i don't recognize", "unknown ticker", "unknown symbol",
    # synthetic-ticker-specific
    "appears to be a synthetic", "synthetic ticker",
    "fictional ticker", "appears fictional",
    "appears to be fabricated",
    "i should not", "i won't be able to",
)


def looks_like_refusal(raw_text: str) -> bool:
    """Heuristic: True if raw answer text looks like a refusal
    paragraph rather than a numeric/structured answer.

    v2.8.3 Commit D (M9): leading-window widened 50 → 80 characters.
    The 50-char window missed refusals that begin with a polite
    preface (``"As an AI language model, I don't have access to..."``
    — the keyword ``"don't have access"`` starts past character 50).
    80 captures the common preface-then-refusal shape without
    materially increasing false positives from in-quote refusal
    phrases (those typically appear deeper in the reply).

      1. Apply _normalize_for_hash + lowercase.
      2. SHORT-CIRCUIT: if any keyword in _REFUSAL_KEYWORDS_LEADING
         appears in the LEADING 80 CHARACTERS, return True.
         Restricting to leading text removes the false-positive class
         where the LLM quotes a refusal-shaped phrase inside a valid
         answer (e.g., 'the article says "I don't have access..."
         but the close was $147.20').
      3. FALLBACK (no keyword in leading 80): if length > 120 AND
         digit ratio < 5%, return True.
      4. Otherwise return False.

    LIMITATION: keyword list is English-only. Non-English refusals
    will pass through. Documented as a v2.2 known limitation.
    """
    if not raw_text:
        return False
    s = _normalize_for_hash(raw_text).lower()
    leading = s[:80]
    for kw in _REFUSAL_KEYWORDS_LEADING:
        if kw in leading:
            return True
    if len(s) > 120:
        digit_count = sum(1 for c in s if c.isdigit())
        if (digit_count / len(s)) < 0.05:
            return True
    return False


# ---------- Sign test ----------


def sign_test_p(n_pos: int, n_neg: int) -> tuple[float, str]:
    """Two-sided sign test p-value.

    n = n_pos + n_neg (ties dropped per drop-and-reduce-n convention).

    Variant pinning per § 7.8:
      - n < 25: exact binomial (scipy.stats.binomtest).
      - 25 <= n < 40: normal approximation with continuity correction:
          z = (|n_pos - n/2| - 0.5) / sqrt(n/4)
      - n >= 40: normal approximation without correction.

    Returns (p_value, variant_name).
    """
    n = n_pos + n_neg
    if n == 0:
        return 1.0, "trivial_n_zero"
    if n < 25:
        try:
            from scipy.stats import binomtest  # type: ignore[import-not-found]
            res = binomtest(n_pos, n=n, p=0.5, alternative="two-sided")
            return float(res.pvalue), "exact_binomial"
        except ImportError:
            # Fall through to normal-approx fallback
            pass
    if 25 <= n < 40:
        # Normal with continuity correction. Textbook form: floor the
        # |x - n/2| term at 0 BEFORE dividing, so a tied n_pos == n/2
        # produces z = 0 (and therefore p = 1) cleanly instead of a
        # negative z that we'd have to clamp via the result. Same
        # numeric output as the prior post-hoc clamp; cleaner derivation.
        z = max(abs(n_pos - n / 2.0) - 0.5, 0.0) / math.sqrt(n / 4.0)
        p = 2.0 * (1.0 - _phi(z))
        return float(max(0.0, min(1.0, p))), "normal_continuity_corrected"
    if n < 25:
        # v2.2.1 #9: scipy unavailable for n<25 — fall through to
        # continuity-corrected normal approximation (better small-
        # sample coverage than uncorrected). r5/r6 had two
        # identical branches; r7 removes the dead duplication and
        # uses the correction. Same textbook form as above.
        z = max(abs(n_pos - n / 2.0) - 0.5, 0.0) / math.sqrt(n / 4.0)
        p = 2.0 * (1.0 - _phi(z))
        return (
            float(max(0.0, min(1.0, p))),
            "normal_continuity_corrected_fallback",
        )
    # n >= 40
    z = abs(n_pos - n / 2.0) / math.sqrt(n / 4.0)
    p = 2.0 * (1.0 - _phi(z))
    return float(max(0.0, min(1.0, p))), "normal_approximation"


def _phi(z: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------- Tango (1998) paired score interval ----------


def wald_paired_diff_ci(
    n_both: int,
    n_real_only: int,
    n_anchor_only: int,
    n_neither: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wald-style CI for the difference of two paired binomial
    proportions p1 − p2.

    p1 = (n_both + n_real_only) / N         # real success rate
    p2 = (n_both + n_anchor_only) / N       # anchor success rate

    Equivalent to PropCIs::diffpropci.Wald.mp (R reference).
    Numerically equivalent; implementation uses bisection over delta
    rather than closed-form, but variance is delta-independent so
    outputs match exactly.

    Implementation: bisection over delta ∈ [-1, 1] for which the
    score statistic for H0: p1 - p2 = delta is within the critical
    region. The variance estimator uses the standard discordant-
    cell formula (does NOT depend on delta).

    HISTORICAL NOTE: v2.2.1 introduced this function as
    ``tango_paired_diff_ci``. v2.8.1 renamed it to
    ``wald_paired_diff_ci`` to match what it actually computes.
    The Wald CI is conservative-vs-Tango (slightly wider intervals)
    when the discordant-pair count is small relative to N. For
    paper-grade rigor, users should additionally run
    R/PropCIs::scoreci.mp on their (n_both, n_real_only,
    n_anchor_only, n_neither) tables and compare. Do NOT cite this
    function as implementing Tango (1998).

    Edge cases:
      - N = 0: raises ValueError (cannot CI an empty bucket).
      - All concordant (n_real_only = n_anchor_only = 0):
        returns (0.0, 0.0) — point CI at delta=0.
      - All discordant (n_both = n_neither = 0): standard formula
        applies.
    """
    n = n_both + n_real_only + n_anchor_only + n_neither
    if n == 0:
        raise ValueError(
            "wald_paired_diff_ci: empty bucket (N=0)"
        )
    if n_real_only == 0 and n_anchor_only == 0:
        # Perfect agreement → point CI at zero
        return 0.0, 0.0
    z_alpha = _z_for_confidence(confidence)
    delta_hat = (n_real_only - n_anchor_only) / n

    def _score_stat(delta: float) -> float:
        """Score statistic at H0: p1 - p2 = delta.

        Wald-style with discordant-cell variance estimator
        (constant in delta). Honest documentation of what's
        actually implemented; v2.2.2 will swap to constrained-
        MLE Tango per the F2 follow-up.
        """
        var = ((n_real_only + n_anchor_only)
               - (n_real_only - n_anchor_only) ** 2 / n) / (n * n)
        if var <= 0:
            return 0.0
        return ((n_real_only - n_anchor_only) / n - delta) / math.sqrt(var)

    # Bisection bracketing.
    def _stat_squared_minus_z(delta: float) -> float:
        s = _score_stat(delta)
        return s * s - z_alpha * z_alpha

    # Lower bound: search left of delta_hat.
    lo = max(-1.0, delta_hat - 1.0)
    hi = delta_hat
    f_lo = _stat_squared_minus_z(lo)
    f_hi = _stat_squared_minus_z(hi)
    if f_lo * f_hi > 0:
        # No crossing — clamp to -1
        lower_ci = max(-1.0, delta_hat - 1.0)
    else:
        lower_ci = _bisect(_stat_squared_minus_z, lo, hi)
    # Upper bound: search right of delta_hat.
    lo2 = delta_hat
    hi2 = min(1.0, delta_hat + 1.0)
    f_lo2 = _stat_squared_minus_z(lo2)
    f_hi2 = _stat_squared_minus_z(hi2)
    if f_lo2 * f_hi2 > 0:
        upper_ci = min(1.0, delta_hat + 1.0)
    else:
        upper_ci = _bisect(_stat_squared_minus_z, lo2, hi2)
    return float(max(-1.0, lower_ci)), float(min(1.0, upper_ci))


def tango_paired_diff_ci(*args, **kwargs):
    """Deprecated since v2.8.1; renamed to ``wald_paired_diff_ci``.

    The implementation is unchanged (Wald-style, not Tango 1998
    method 10). Hard removal scheduled for v2.9 (the final v2.x
    release).
    """
    warnings.warn(
        "tango_paired_diff_ci is deprecated since v2.8.1; rename to "
        "wald_paired_diff_ci. The implementation is unchanged "
        "(Wald-style, not Tango 1998 method 10). Hard removal: v2.9.",
        DeprecationWarning,
        stacklevel=2,
    )
    return wald_paired_diff_ci(*args, **kwargs)


def _bisect(f, lo: float, hi: float, tol: float = 1e-9,
            max_iter: int = 200) -> float:
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def _z_for_confidence(confidence: float) -> float:
    """z value for two-sided CI at the given confidence level."""
    # Common values: 0.95 → 1.96, 0.99 → 2.576
    try:
        from scipy.stats import norm  # type: ignore[import-not-found]
        return float(norm.ppf((1.0 + confidence) / 2.0))
    except ImportError:
        table = {0.95: 1.96, 0.99: 2.576, 0.90: 1.645}
        return table.get(confidence, 1.96)


# ---------- Joint effective-rate computation ----------


def _is_structured_rank_input(x: Any) -> bool:
    """v2.2.1 #12 Commit B: predicate matching paths 1-3 of
    resolve_rank_answer's priority chain.

    Returns True if ``x`` is a structured answer payload that
    bypasses the refusal heuristic (user has already done the
    parsing work).

    Applies across ALL probe types (not rank-only). For a
    KnowledgeProbe whose parsed_answer happens to be a list[str],
    the carve-out skips refusal detection. Rationale: structured
    input means the user already extracted the answer from raw
    text; the engine's refusal heuristic on raw_answer is moot.
    """
    from aiphaforge.probes._rank import (
        RankAnswer,
        _is_sequence_of_str,
    )
    if isinstance(x, RankAnswer):
        return True
    if _is_sequence_of_str(x):
        return True
    if isinstance(x, dict) and _is_sequence_of_str(x.get("ranking")):
        return True
    return False


def compute_effective_rate(records: Sequence[AnswerRecord]) -> float:
    """Joint P(parsed AND not refused), computed per-AnswerRecord.

    Per § 7.5: replaces the r5 product-of-marginals
    `parse_success * (1 - refusal_rate)` because that formula
    double-discounts answers that BOTH fail parse AND look like
    refusal (the common case for paragraph refusals).

    v2.2.1 #12 Commit B: structured-input carve-out — when
    rec.parsed_answer is a RankAnswer / Sequence[str] / dict-with-
    ranking AND parse_status == "valid", skip the refusal heuristic
    on raw_answer (user already did the parsing).

    v2.2.1 audit-fix nit (LE #1): carve-out also requires
    parse_status == "valid". An upstream parser flagging the
    structured payload as invalid (e.g., schema-validation failure
    above the engine) should NOT be silently overridden into
    "effective" by this carve-out.
    """
    n_total = len(records)
    if n_total == 0:
        return 0.0
    n_eff = 0
    for rec in records:
        if (
            _is_structured_rank_input(rec.parsed_answer)
            and rec.parse_status == "valid"
        ):
            # Structured input AND user-attested valid → counted as
            # effective regardless of raw_answer content.
            n_eff += 1
            continue
        parsed = (rec.parsed_answer is not None
                  and rec.parse_status == "valid")
        refused = looks_like_refusal(rec.raw_answer or "")
        if parsed and not refused:
            n_eff += 1
    return n_eff / n_total


def compute_refusal_rate(records: Sequence[AnswerRecord]) -> float:
    """Fraction of AnswerRecords whose raw_answer looks like refusal.

    v2.2.1 #12 Commit B + audit-fix nit (LE #1): structured-input
    carve-out only fires when parse_status == "valid". Records with
    structured payload but parse_status != "valid" still get the
    refusal heuristic applied.
    """
    if not records:
        return 0.0
    def _carve_out_applies(r: AnswerRecord) -> bool:
        return (
            _is_structured_rank_input(r.parsed_answer)
            and r.parse_status == "valid"
        )
    return sum(
        1 for r in records
        if not _carve_out_applies(r)
        and looks_like_refusal(r.raw_answer or "")
    ) / len(records)


# ---------- Persistence baseline ----------


def _persistence_predict(spec: QuestionSpec) -> Any:
    """Per-template persistence baseline (per § 7.6).

    - NextCloseContinuation: predict close[last context bar] for all
      forward steps.
    - NextRangeContinuation: predict (low[last], high[last]).
    - NextReturnContinuation: predict 0.0 (zero log-return — martingale
      hypothesis; NOT predict-last-return, which is degenerate per
      Q-S4 r3 review).
    - Other templates: not applicable; raises ValueError.
    """
    template_id = spec.template_id
    ctx = spec.metadata.get("context_window", [])
    if not ctx:
        raise ValueError(
            f"persistence baseline requires context_window in "
            f"metadata; spec {spec.question_id!r} has none"
        )
    last_bar = ctx[-1]
    if template_id == "next_close_continuation":
        return float(last_bar["close"])
    if template_id == "next_range_continuation":
        return (float(last_bar["low"]), float(last_bar["high"]))
    if template_id == "next_return_continuation":
        # Martingale hypothesis: predict 0 log-return.
        return 0.0
    raise ValueError(
        f"persistence baseline not defined for template_id="
        f"{template_id!r}"
    )


# ---------- KnowledgeCheckReport ----------


_NON_TRANSITIVITY_NOTE = (
    "Pillar non-transitivity: this report measures Q&A memorization "
    "only. It does not imply or refute results from "
    "obfuscation_bootstrap."
)
_PERSISTENCE_CAVEAT = (
    "persistence_vs_real_sign_test_p compares a deterministic "
    "baseline against a sampled (LLM) predictor; not like-for-like "
    "with anchor_vs_real (LLM-vs-LLM). Interpret p-values within "
    "their respective comparison classes."
)
# v2.2.1 #7: hardcoded ordinal weighting for scalar_leakage_index.
# Locked v2.2.1; changing requires __version__ bump per §14
# stability commitment.
#
# v2.2.1 audit fix Commit E: promoted to a public, MappingProxyType-
# backed constant. Paper authors should `from aiphaforge.probes
# import LEAKAGE_INDEX_BUCKET_WEIGHTS` rather than transcribing the
# numeric literals — that way the citation tracks the locked weights
# across releases.
LEAKAGE_INDEX_BUCKET_WEIGHTS: Mapping[str, float] = MappingProxyType({
    "exact": 4.0, "near": 3.0, "rough": 2.0, "miss": 1.0, "invalid": 0.0,
})
_LEAKAGE_INDEX_CAVEAT = (
    "scalar_leakage_index is a point estimate with hardcoded "
    "ordinal weighting (exact=4, near=3, rough=2, miss=1, "
    "invalid=0). Range [-4, +4]. SE accounts for paired correlation "
    "via per-question differences. Approximate significance: |z| >= "
    "1.96 corresponds to two-sided p < 0.05. For full inferential "
    "rigor, see bucket_delta_ci. COMPARABILITY: comparable "
    "WITHIN a single probe type AND configuration. NOT comparable "
    "across probe types or across ToleranceProfile configurations "
    "even within the same probe type, because bucket calibration "
    "differs (e.g., vol-scaled vs fixed-band exact thresholds map "
    "to different actual band widths). SENTINEL Z: when _z is a "
    "sentinel ('positive_infinite'/'negative_infinite'), the "
    "underlying scalar_leakage_index magnitude is the comparable "
    "quantity; sentinel equality across reports is NOT evidence "
    "of equivalent effect size. SINGLE-RECORD DIAGNOSTIC: when N "
    "== 1 for either side, drift cannot be detected from one "
    "observation; rerun with N >= 2 if cross-run nondeterminism "
    "signal is needed. Comparing across aiphaforge versions is "
    "only valid when the weighting is unchanged — this version: "
    "v2.2.1."
)


def _parser_used_distribution_is_non_degenerate(
    parser_used_values: Sequence[str],
) -> bool:
    """v2.2.1 #12 Commit C: predicate per r7 §15.

    Returns True iff the distribution contains at least TWO distinct
    parser-path values (excluding 'unrecognized_parsed_answer_type'
    which is the type-dispatch failure mode, not a parser path).

    'raw_text_unparseable' IS counted — it's a failed text-parse
    attempt and still represents LLM-nondeterminism signal.

    Used to gate the informational note emission in
    _maybe_emit_parser_used_drift_note.
    """
    successful = {
        pu for pu in parser_used_values
        if pu != "unrecognized_parsed_answer_type"
    }
    return len(successful) >= 2


def _maybe_emit_parser_used_drift_note(
    distribution_real: Optional[Mapping[str, int]],
    distribution_anchor: Optional[Mapping[str, int]],
    notes: list[str],
) -> None:
    """v2.2.1 #12 Commit C (per r7 §15): emit the informational
    LLM-nondeterminism note iff EITHER side's distribution is
    non-degenerate. The note text identifies which side(s)
    drifted so paper readers can interpret without re-deriving.
    """
    drifted_sides: list[str] = []
    if distribution_real is not None and (
        _parser_used_distribution_is_non_degenerate(
            list(distribution_real.keys())
        )
    ):
        drifted_sides.append("real")
    if distribution_anchor is not None and (
        _parser_used_distribution_is_non_degenerate(
            list(distribution_anchor.keys())
        )
    ):
        drifted_sides.append("anchor")
    if drifted_sides:
        # v2.2.1 audit-fix nit: emit comma-joined human-readable
        # form instead of Python list repr ('real, anchor' vs
        # "['real', 'anchor']"). Cleaner for paper readers.
        sides_text = ", ".join(drifted_sides)
        notes.append(
            f"parser_used distribution is informational; LLM "
            f"nondeterminism upstream affects both this distribution "
            f"and bucket counts. Cross-run comparisons should "
            f"disclose LLM sampling parameters (see "
            f"KnowledgeCheckReport.manifest.provider_config). "
            f"Non-degenerate side(s): {sides_text}."
        )


def _score_rank_attested(
    question_set: "QuestionSet",
    attested: AttestedAnswers,
    *,
    manifest: Optional[dict[str, Any]] = None,
    provider_config: Optional[dict] = None,
    normalization_preset: str = "passthrough",
) -> tuple[QAProbeReport, "Counter[str]"]:
    """v2.2.1 #12 Commit A: score a rank-probe attested answer set.

    Returns ``(QAProbeReport, parser_used_counter)``. Commit C
    later surfaces the Counter as a typed field on
    ``KnowledgeCheckReport``; this commit returns it so the
    handoff is concrete.

    Flow per AnswerRecord:
      1. Look up the corresponding ``QuestionSpec`` by question_id.
      2. Derive ``expected_symbols = set(qs.truth_value)`` — the
         canonical normalized symbols (truth_value is a tuple of
         these by ``RankContinuationProbe.build``'s contract).
         NOTE: there is no ``qs.metadata["expected_symbols"]``
         key; the truth_value tuple IS the expected-set source.
      3. Call ``resolve_rank_answer(rec, expected_symbols,
         normalization_preset)`` from ``_rank.py``.
      4. On success: score with ``score_rank_answer`` against
         ``qs.truth_value`` (a tuple in descending-return order).
      5. On failure: produce a ``QuestionScore`` with
         ``band="invalid"`` and ``validity="invalid"``.
      6. Record the resolver-returned ``parser_used`` string into
         the Counter (drop None values — they don't represent a
         valid path).

    Aggregates via ``aggregate_scores`` and returns.
    """
    import collections

    from aiphaforge.probes.models import QuestionScore
    from aiphaforge.probes.scoring import _merge_provider_config, aggregate_scores

    merged_manifest, _ = _merge_provider_config(
        manifest, provider_config,
    )
    qs_by_id = {q.question_id: q for q in question_set}
    counter: "Counter[str]" = collections.Counter()
    scores: list[QuestionScore] = []
    for rec in attested.answers:
        qs = qs_by_id.get(rec.question_id)
        if qs is None:
            # Answer for an unknown question; silently skip (same
            # convention as score_attested_answers).
            continue
        expected_symbols = set(qs.truth_value)
        rank_answer, parse_status, parser_used = resolve_rank_answer(
            rec, expected_symbols, normalization_preset,
        )
        # Record the parser_used path for the Commit C drift note.
        if parser_used is not None:
            counter[parser_used] += 1
        if rank_answer is None:
            scores.append(
                QuestionScore(
                    question_id=qs.question_id,
                    validity="invalid",
                    band="invalid",
                    truth_value=qs.truth_value,
                    parsed_answer=None,
                    relative_error=None,
                    contains_truth=None,
                    range_width_ratio=None,
                    max_range_width_exceeded=None,
                    metadata={"rank_parse_status": parse_status},
                )
            )
            continue
        band, prov = score_rank_answer(rank_answer, qs.truth_value)
        # NOTE: validity is "valid" even when band == "invalid".
        # The two fields measure different things:
        #   validity = "did the LLM produce a well-formed answer?"
        #   band     = "did the scored content fall in which bucket?"
        # A rank answer with omitted_symbols parses fine but scores
        # "invalid" content-wise; that still counts as a valid
        # submission for aggregate denominators. Setting
        # validity="invalid" here would cause aggregate_scores to
        # filter the record out entirely (see scoring.py:730),
        # producing zero in bands_breakdown["invalid"] — the
        # opposite of what we want.
        scores.append(
            QuestionScore(
                question_id=qs.question_id,
                validity="valid",
                band=band,
                truth_value=qs.truth_value,
                parsed_answer=rank_answer.ranking,
                relative_error=None,
                contains_truth=None,
                range_width_ratio=None,
                max_range_width_exceeded=None,
                metadata={
                    "rank_parser_used": parser_used,
                    **prov,
                },
            )
        )
    from dataclasses import replace as _dc_replace
    report = aggregate_scores(
        question_set, scores, manifest=merged_manifest,
    )
    # Attach attestation hashes from the input AttestedAnswers
    # (the standard scoring path does this; we mirror it here).
    report = _dc_replace(
        report,
        parsing_schema_hash=attested.parsing_schema_hash,
        prompt_template_hash=attested.prompt_template_hash,
    )
    return report, counter


def _compute_scalar_leakage_index_and_z(
    paired_bands: list[tuple[str, str]],
) -> tuple[Optional[float], Optional[float], Union[
    float, Literal["positive_infinite", "negative_infinite"], None
]]:
    """Compute (index, h0_se, z) from paired (real_band, anchor_band).

    index = mean(d_i) where d_i = w(real_i) - w(anchor_i).
    h0_se = sqrt(var(d_i, ddof=1) / N) — paired-difference SE.
    z = index / h0_se when SE > eps; else typed sentinel.

    All None when paired_bands is empty.
    """
    import math as _math
    n = len(paired_bands)
    if n == 0:
        return None, None, None
    diffs = [
        LEAKAGE_INDEX_BUCKET_WEIGHTS.get(rb, 0.0)
        - LEAKAGE_INDEX_BUCKET_WEIGHTS.get(ab, 0.0)
        for rb, ab in paired_bands
    ]
    index = sum(diffs) / n
    if n < 2:
        # Single record: SE undefined. Index defined but SE/z None.
        return float(index), None, None
    mean = index
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = _math.sqrt(var / n)
    eps_se = 1e-15
    eps_index = 1e-12
    if se > eps_se:
        z: Union[float, str, None] = float(index / se)
    elif abs(index) < eps_index:
        z = None  # 0/0
    elif index > 0:
        z = "positive_infinite"
    else:
        z = "negative_infinite"
    return float(index), float(se), z


@dataclass(init=False, frozen=True)
class KnowledgeCheckReport:
    """v2.2 (M6) standalone Q&A leakage diagnostic report.

    See ``docs/plans/v2.2-plan-r6.md`` § 7.3.

    v2.8.1 (H7): switched from ``@dataclass(frozen=True)`` to
    ``@dataclass(init=False, frozen=True)`` with a manual ``__init__``.
    Two reasons:

    1. Pickle compatibility. The dataclass holds two
       ``MappingProxyType`` fields wrapped in ``__post_init__``.
       Default pickling cannot round-trip these. ``__getstate__`` /
       ``__setstate__`` below unwrap on dump and re-wrap on load.

    2. Backward-compat for old pickles that still carry the
       ``bucket_delta_tango_ci`` field name (removed in v2.8). The
       manual ``__init__`` translates that legacy kwarg to
       ``bucket_delta_ci`` with a DeprecationWarning.

    Side effect: positional construction now raises ``TypeError``;
    all callers must use keyword arguments. Unknown kwargs (typos,
    obsolete field names not on the legacy-alias list) also raise
    ``TypeError`` so silent field-drops do not happen.
    """

    probe_kind: str  # "knowledge" | "continuation" | "rank_continuation"

    # Primary, per-bucket comparison
    real_score: QAProbeReport
    anchor_score: Optional[QAProbeReport]
    bucket_delta: Optional[dict[str, float]]
    # bucket_delta_ci field declared below. The historic
    # "_tango_ci"-suffixed alias added v2.2.1 was hard-deleted in
    # v2.8 cleanup; external reads must migrate to bucket_delta_ci.

    # Paired sign test (per quant Q3 r2 — replaces Wilcoxon)
    paired_sign_test_p: Optional[float]
    paired_sign_test_n_positive: Optional[int]
    paired_sign_test_n_negative: Optional[int]
    paired_sign_test_n_ties: Optional[int]
    paired_sign_test_variant: Optional[str]

    # Refusal / validity
    real_parse_success_rate: float
    real_refusal_rate: float
    anchor_parse_success_rate: Optional[float]
    anchor_refusal_rate: Optional[float]
    real_effective_rate: float
    anchor_effective_rate: Optional[float]
    # v2.2.1 audit-fix Commit I: tightened from bare `str` so
    # downstream consumers using pyright/mypy can exhaustively
    # dispatch on the tag. Values:
    #   "NO_ANCHOR"          — no anchor supplied; nothing to compare.
    #   "OK"                 — anchor present and pairing succeeded;
    #                          bucket_delta and scalar_leakage_index
    #                          are populated.
    #   "REFUSAL_SUSPECTED"  — one side's effective rate is much
    #                          lower than the other (or both are 0);
    #                          derived stats suppressed.
    #   "PAIRING_FAILED"     — qid sets on real/anchor reports do
    #                          not overlap (v2.2.1 Commit F);
    #                          derived stats suppressed.
    anchor_validity: Literal[
        "NO_ANCHOR", "OK", "REFUSAL_SUSPECTED", "PAIRING_FAILED",
    ]

    # Persistence baseline (continuation only)
    persistence_baseline_score: Optional[QAProbeReport]
    persistence_caveat: Optional[str]
    real_minus_persistence_bucket_delta: Optional[dict[str, float]]
    real_vs_persistence_sign_test_p: Optional[float]

    # Attestation
    parsing_schema_hash: str
    parsing_schema_description: str
    prompt_template_hash: str
    prompt_template_description: str

    # v2.2.1 #7: scalar leakage index + paired SE + z (typed
    # sentinel for SE==0). Range [-4, +4]. Point estimate; for
    # inference use bucket_delta_ci. Computed for ALL
    # probe types (KnowledgeProbe, ContinuationProbe,
    # RankContinuationProbe). None when no anchor.
    scalar_leakage_index: Optional[float] = None
    scalar_leakage_index_h0_se: Optional[float] = None
    scalar_leakage_index_z: Union[
        float,
        Literal["positive_infinite", "negative_infinite"],
        None,
    ] = None
    leakage_index_caveat: Optional[str] = None

    # v2.2.1 #4: normalization preset surfaced for rank probes.
    # Human-readable preset name ("us_equity", "passthrough",
    # etc.). None for non-rank probes.
    normalization_preset: Optional[str] = None

    # v2.2.1 #12 Commit C: per-side Counter of parser_used values
    # for rank probes. None for non-rank probes OR when anchor
    # side absent. Wrapped in MappingProxyType in __post_init__
    # to preserve frozen-dataclass immutability per r7 §2.6.
    parser_used_distribution_real: Optional["Mapping[str, int]"] = None
    parser_used_distribution_anchor: Optional["Mapping[str, int]"] = None

    # v2.2.1 introduced this as the canonical name (the historic
    # "_tango_ci"-suffixed alias was hard-deleted v2.8).
    # Treat as read-only — do not mutate the dict in place.
    bucket_delta_ci: Optional[dict[str, tuple[float, float]]] = None

    # v2.2.1 #1: vol-scaling provenance keyed by (question_id, side).
    # side ∈ {"real", "anchor"}. None when no question used vol_scale
    # (e.g., all probes had vol_scale=None tolerance) or when no
    # anchor was provided AND no real questions had vol_scale.
    vol_scale_provenance: Optional[
        dict[str, dict[str, dict[str, Any]]]
    ] = None

    # Run-to-run drift support
    report_uuid: str = ""
    wall_clock_utc: str = ""

    notes: tuple[str, ...] = ()

    is_pillar_summary: bool = False  # ENFORCED False

    # v2.8.3 Commit E (M10a): persistence-baseline pairing validity,
    # mirrors anchor_validity so downstream consumers can dispatch on
    # whether real_minus_persistence_bucket_delta and
    # real_vs_persistence_sign_test_p carry meaningful values.
    #   "NO_PERSISTENCE" — probe is not a ContinuationProbe; the
    #                      persistence baseline does not apply.
    #   "OK"             — pairing succeeded; downstream stats are
    #                      populated.
    #   "PAIRING_FAILED" — real/persistence question_id sets do not
    #                      overlap; derived stats suppressed.
    #   "UNKNOWN"        — legacy pickle restored without this field;
    #                      backfilled by __setstate__ (Commit F).
    # Conceptually grouped with the persistence_* fields above; placed
    # at the end of the field list because dataclass field ordering
    # interacts with default values across the file's history.
    persistence_validity: Literal[
        "NO_PERSISTENCE", "OK", "PAIRING_FAILED", "UNKNOWN",
    ] = "NO_PERSISTENCE"

    def to_dict(self) -> dict[str, Any]:
        """Convert this report to a plain-dict representation
        suitable for JSON serialization.

        v2.2.1 audit-fix nit (LE #2): the dataclass uses
        MappingProxyType for the parser_used_distribution_*
        fields to preserve frozen-dataclass immutability. But
        Python's stdlib ``dataclasses.asdict()`` raises
        ``TypeError: cannot pickle 'mappingproxy' object`` on
        these fields. Users following the standard
        ``json.dumps(asdict(report))`` pattern hit this. This
        method provides a working alternative — it unwraps any
        MappingProxyType to a plain dict so the result is
        JSON-serializable.

        The output is a shallow copy with field-by-field
        unwrapping; nested dataclasses (real_score etc.) keep
        their original type — callers serializing the full
        report should still walk them with their own
        serialization helpers.

        Nested dataclass guidance (v2.2.1 audit-fix Commit I):
        ``real_score``, ``anchor_score``, and
        ``persistence_baseline_score`` are ``QAProbeReport``
        instances. ``QAProbeReport`` and its component types
        (``QuestionScore``, ``TemplateAggregate``, etc.) contain
        NO ``MappingProxyType`` members — they are safe to pass to
        ``dataclasses.asdict()`` directly. Only THIS top-level
        dataclass holds the proxy-wrapped fields that asdict()
        rejects, so callers don't need bespoke serialization for
        the nested types.
        """
        from dataclasses import fields
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, MappingProxyType):
                out[f.name] = dict(v)
            else:
                out[f.name] = v
        return out

    def __init__(self, **kwargs: Any) -> None:
        # v2.8.1 H7: manual __init__ for pickle compat + legacy-kwarg
        # translation. Kwargs-only by design; positional construction
        # is rejected (Python raises TypeError automatically because
        # we declared **kwargs, no positional params).
        from dataclasses import MISSING, fields

        # Legacy kwarg translation (old pickles / external callers
        # that still use bucket_delta_tango_ci):
        if "bucket_delta_tango_ci" in kwargs:
            warnings.warn(
                "bucket_delta_tango_ci is deprecated since v2.8; "
                "treating as legacy alias for bucket_delta_ci. "
                "Hard removal in v2.9.",
                DeprecationWarning, stacklevel=2,
            )
            legacy = kwargs.pop("bucket_delta_tango_ci")
            kwargs.setdefault("bucket_delta_ci", legacy)

        field_names = {f.name for f in fields(self)}
        # M4: reject unknown kwargs (typos, removed fields not on the
        # legacy-alias whitelist). Without this, the manual __init__
        # would silently drop them — worse than the v2.7 dataclass.
        unknown = set(kwargs) - field_names
        if unknown:
            raise TypeError(
                f"KnowledgeCheckReport got unexpected keyword "
                f"arguments: {sorted(unknown)}"
            )

        for f in fields(self):
            if f.name in kwargs:
                value = kwargs[f.name]
            elif f.default is not MISSING:
                value = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                value = f.default_factory()  # type: ignore[misc]
            else:
                raise TypeError(
                    f"KnowledgeCheckReport: missing required argument "
                    f"{f.name!r}"
                )
            object.__setattr__(self, f.name, value)

        self.__post_init__()

    def __post_init__(self):
        if self.is_pillar_summary:
            raise ValueError(
                "is_pillar_summary must remain False — "
                "KnowledgeCheckReport is a single-pillar diagnostic, "
                "not a cross-pillar verdict."
            )
        # v2.2.1 #12 Commit C: wrap per-side distributions in
        # MappingProxyType so the frozen-dataclass immutability
        # claim extends to these mapping fields. Use
        # object.__setattr__ because frozen blocks direct
        # assignment. Idempotent — re-running on already-wrapped
        # input is a no-op, so __setstate__ can safely call this.
        if (self.parser_used_distribution_real is not None
                and not isinstance(
                    self.parser_used_distribution_real,
                    MappingProxyType,
                )):
            object.__setattr__(
                self, "parser_used_distribution_real",
                MappingProxyType(dict(self.parser_used_distribution_real)),
            )
        if (self.parser_used_distribution_anchor is not None
                and not isinstance(
                    self.parser_used_distribution_anchor,
                    MappingProxyType,
                )):
            object.__setattr__(
                self, "parser_used_distribution_anchor",
                MappingProxyType(dict(self.parser_used_distribution_anchor)),
            )
        # v2.8: removed v2.2.1 cross-population logic for the
        # historic "_tango_ci"-suffixed alias; bucket_delta_ci is
        # now the only name.

    def __getstate__(self) -> dict[str, Any]:
        # Unwrap MappingProxyType for picklability. Pickle cannot
        # serialize mappingproxy objects (they hold a borrowed
        # reference to their underlying dict).
        state = dict(self.__dict__)
        for k in ("parser_used_distribution_real",
                  "parser_used_distribution_anchor"):
            v = state.get(k)
            if isinstance(v, MappingProxyType):
                state[k] = dict(v)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # v2.8.1 post-review fix: legacy alias translation. Old pickles
        # (v2.7.x or earlier) carry `bucket_delta_tango_ci` instead of
        # `bucket_delta_ci`. __init__'s translation block only covers
        # construction; pickle restore goes through __setstate__, so
        # the same translation must happen here or the revived instance
        # has bucket_delta_tango_ci set + bucket_delta_ci missing →
        # AttributeError on any downstream read.
        if "bucket_delta_tango_ci" in state:
            # stacklevel=3 lands at the user's pickle.load() / pickle.loads()
            # site; pickle restore calls __setstate__ via _pickle.c
            # internals, so stacklevel=2 would point at copyreg / pickle
            # rather than the user (round-2 NIT).
            warnings.warn(
                "Loaded a legacy pickle carrying bucket_delta_tango_ci; "
                "translating to bucket_delta_ci. Re-save to silence. "
                "Hard removal in v2.9.",
                DeprecationWarning, stacklevel=3,
            )
            state = dict(state)
            state.setdefault("bucket_delta_ci",
                             state.pop("bucket_delta_tango_ci"))

        # v2.8.3 Commit F (M10b): persistence_validity field was
        # added in v2.8.3 Commit E. Pickles produced by v2.8.2 and
        # earlier carry no such key. Backfill to "UNKNOWN" so the
        # revived instance still satisfies the Literal contract
        # (rather than producing AttributeError on downstream
        # reads). UNKNOWN is a distinct tag — not silently coerced
        # to NO_PERSISTENCE — so consumers can detect the legacy
        # provenance and re-run the orchestrator to upgrade.
        if "persistence_validity" not in state:
            state = dict(state)
            state["persistence_validity"] = "UNKNOWN"

        # Re-wrap MappingProxyType after load + replay __post_init__
        # so any validation / wrapping side effects fire on the
        # revived instance (v2.1.2 M5).
        for k, v in state.items():
            object.__setattr__(self, k, v)
        self.__post_init__()


# ---------- knowledge_check orchestrator ----------

ProbeT = Union[KnowledgeProbe, ContinuationProbe, RankContinuationProbe]


def knowledge_check(
    probe: ProbeT,
    data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
    timestamps: Sequence[pd.Timestamp],
    answers: AttestedAnswers,
    *,
    anchor: Optional[pd.DataFrame] = None,
    anchor_answers: Optional[AttestedAnswers] = None,
    manifest: Optional[dict[str, Any]] = None,
    provider_config: Optional[dict] = None,
    refusal_rate_threshold: float = 0.5,
    exemplar_pairs: Optional[Sequence[tuple[str, pd.Timestamp]]] = None,
) -> KnowledgeCheckReport:
    """Standalone Q&A leakage diagnostic. See § 7.

    The function is **idempotent and pure**: same inputs always
    produce the same report (UUID and wall_clock_utc aside, which
    are metadata not part of the diagnostic).

    v2.2.1 #7: removed the unused ``n_anchor_bootstrap`` /
    ``bootstrap_seed`` / ``bootstrap_unit`` parameters. The r5/r6
    design moved away from a single ``score_minus_anchor`` scalar
    to per-bucket Tango CIs + paired sign test; the bootstrap loop
    was never implemented and the parameters shipped as misleading
    API surface. Inferential primitives are now: per-bucket Tango
    CIs (``bucket_delta_ci``), paired sign test
    (``paired_sign_test_p``), and the new scalar ``scalar_leakage_index``
    + ``scalar_leakage_index_z`` per § 7.2.

    Refuses (raises ValueError) when:
      - provider_config is None or missing any required key.
      - anchor_answers without anchor (or vice versa).
      - parsing_schema_hash or prompt_template_hash mismatch between
        what the probe was built for and what AttestedAnswers carry
        (NOT enforced at this layer — the caller's responsibility).
      - exemplar_pairs overlap with the test (symbol, timestamp) set.
    """
    # ---- validation ----
    if provider_config is None:
        raise ValueError(
            "provider_config is required for paper-grade "
            "reproducibility"
        )
    _validate_provider_config(provider_config)

    if (anchor is None) != (anchor_answers is None):
        raise ValueError(
            "anchor and anchor_answers must both be supplied or "
            "both omitted"
        )

    # ---- exemplar disjointness on (symbol, timestamp) tuples ----
    if exemplar_pairs:
        test_pairs: set[tuple[str, pd.Timestamp]] = set()
        if isinstance(probe, RankContinuationProbe):
            for sym in probe.symbols:
                for ts in timestamps:
                    test_pairs.add((sym, ts))
        else:
            sym = getattr(probe, "symbol", None)
            for ts in timestamps:
                test_pairs.add((sym, ts))
        for ex_sym, ex_ts in exemplar_pairs:
            if (ex_sym, ex_ts) in test_pairs:
                raise ValueError(
                    f"exemplar_pairs overlap with test set on "
                    f"(symbol={ex_sym}, timestamp={ex_ts})"
                )

    # ---- build question set ----
    if isinstance(probe, RankContinuationProbe):
        if not isinstance(data, dict):
            raise TypeError(
                "RankContinuationProbe requires data as dict "
                "of {symbol: DataFrame}"
            )
        question_set = probe.build(data, timestamps)
    elif isinstance(probe, ContinuationProbe):
        question_set = probe.build(data, timestamps)
    elif isinstance(probe, KnowledgeProbe):
        question_set = probe.build(data, timestamps)
    else:
        raise TypeError(
            f"Unsupported probe type: {type(probe).__name__}"
        )

    # ---- v2.2.1 #1: vol-scaling wiring (real side) ----
    # Per question, if its tolerance carries vol_scale, compute
    # sigma causally and rewrite the spec with the vol-scaled
    # tolerance. The vol_scale_provenance dict is keyed by
    # (question_id, side) so anchor-side rewrite (below) doesn't
    # collide.
    vol_scale_provenance: dict[str, dict[str, dict[str, Any]]] = {}
    question_set, real_vol_prov = apply_vol_scaling_to_question_set(
        probe, question_set, data,
    )
    for qid, prov in real_vol_prov.items():
        vol_scale_provenance.setdefault(qid, {})["real"] = prov

    # ---- score real ----
    # v2.2.1 #12 Commit A: route RankContinuationProbe through
    # the dedicated _score_rank_attested helper (uses
    # resolve_rank_answer + score_rank_answer). Other probe
    # types route through the standard numeric scoring path.
    real_parser_counter: Optional[Counter[str]] = None
    if isinstance(probe, RankContinuationProbe):
        real_report, real_parser_counter = _score_rank_attested(
            question_set, answers, manifest=manifest,
            provider_config=provider_config,
            normalization_preset=probe.normalization_preset,
        )
    else:
        real_report, _, _ = score_attested_answers(
            question_set, answers, manifest=manifest,
            provider_config=provider_config,
        )

    real_recs = list(answers.answers)
    real_eff = compute_effective_rate(real_recs)
    real_refusal = compute_refusal_rate(real_recs)

    # ---- score anchor (if provided) ----
    anchor_report: Optional[QAProbeReport] = None
    anchor_eff: Optional[float] = None
    anchor_refusal: Optional[float] = None
    bucket_delta: Optional[dict[str, float]] = None
    bucket_delta_ci: Optional[dict[str, tuple[float, float]]] = None
    sign_test_p_val: Optional[float] = None
    sign_test_n_pos: Optional[int] = None
    sign_test_n_neg: Optional[int] = None
    sign_test_n_ties: Optional[int] = None
    sign_test_variant: Optional[str] = None
    anchor_validity = "NO_ANCHOR"
    notes: list[str] = []
    # v2.2.1 audit-fix nit: hoist anchor_parser_counter init out
    # of the conditional block so the later `try/except NameError`
    # is unnecessary. Mirrors real_parser_counter declaration above.
    anchor_parser_counter: Optional[Counter[str]] = None

    if anchor is not None and anchor_answers is not None:
        if isinstance(probe, RankContinuationProbe):
            anchor_data_dict = {
                sym: anchor for sym in probe.symbols
            }
            anchor_qs = probe.build(anchor_data_dict, timestamps)
        elif isinstance(probe, ContinuationProbe):
            anchor_qs = probe.build(anchor, timestamps)
        elif isinstance(probe, KnowledgeProbe):
            anchor_qs = probe.build(anchor, timestamps)
        else:
            raise TypeError(
                f"Unsupported probe type: {type(probe).__name__}"
            )
        # v2.2.1 #1: vol-scaling wiring (anchor side) — uses
        # ANCHOR sigma, not real sigma. Both sides go through
        # the same rewrite path with their own data.
        anchor_qs, anchor_vol_prov = (
            apply_vol_scaling_to_question_set(
                probe, anchor_qs, anchor,
            )
        )
        for qid, prov in anchor_vol_prov.items():
            vol_scale_provenance.setdefault(qid, {})["anchor"] = prov
        # v2.2.1 #12 Commit A: same RankContinuationProbe routing
        # on anchor side. (anchor_parser_counter was initialized
        # to None at the outer scope per audit-fix nit.)
        if isinstance(probe, RankContinuationProbe):
            anchor_report, anchor_parser_counter = (
                _score_rank_attested(
                    anchor_qs, anchor_answers, manifest=manifest,
                    provider_config=provider_config,
                    normalization_preset=probe.normalization_preset,
                )
            )
        else:
            anchor_report, _, _ = score_attested_answers(
                anchor_qs, anchor_answers, manifest=manifest,
                provider_config=provider_config,
            )
        anchor_recs = list(anchor_answers.answers)
        anchor_eff = compute_effective_rate(anchor_recs)
        anchor_refusal = compute_refusal_rate(anchor_recs)

        # Symmetric refusal threshold with 0/0 edge case (per § 7.5)
        if max(real_eff, anchor_eff) == 0.0:
            anchor_validity = "REFUSAL_SUSPECTED"
            notes.append(
                "Both effective rates are zero; treating as "
                "REFUSAL_SUSPECTED."
            )
        else:
            ratio = (
                min(real_eff, anchor_eff) / max(real_eff, anchor_eff)
            )
            if ratio < refusal_rate_threshold:
                anchor_validity = "REFUSAL_SUSPECTED"
                low_side = (
                    "real" if real_eff < anchor_eff else "anchor"
                )
                notes.append(
                    f"Effective-rate ratio {ratio:.3f} < threshold "
                    f"{refusal_rate_threshold}; suppressing "
                    f"score_minus_anchor and bucket_delta. "
                    f"Low side: {low_side}."
                )
            else:
                anchor_validity = "OK"

        if anchor_validity == "OK":
            # v2.2.1 #5: pair by question_id (was position-based,
            # which silently misaligned on dropped/reordered
            # questions). Hoisted out of the bucket loop so the
            # pairing runs once.
            paired, pairing_notes = _pair_scores_by_question_id(
                real_report, anchor_report,
            )
            notes.extend(pairing_notes)
            if not paired:
                # Disjoint qid sets — degraded report (warning
                # already in notes). Suppress derived statistics.
                # v2.2.1 audit-fix Commit F: distinct validity tag
                # for the pairing failure mode. Previously this
                # case was overloaded onto REFUSAL_SUSPECTED, which
                # confused downstream callers trying to distinguish
                # "model refused" from "anchor qids didn't line up".
                anchor_validity = "PAIRING_FAILED"

        if anchor_validity == "OK":
            # Compute per-bucket delta and Tango paired CI.
            bucket_delta = {}
            bucket_delta_ci = {}
            real_buckets = _bucket_assignments(real_report)
            anchor_buckets = _bucket_assignments(anchor_report)
            for bucket in (
                "exact", "near", "rough", "miss", "invalid",
            ):
                # Bucket-level proportions
                real_share = (
                    real_buckets.get(bucket, 0)
                    / max(1, real_report.submitted_answers)
                )
                anchor_share = (
                    anchor_buckets.get(bucket, 0)
                    / max(1, anchor_report.submitted_answers)
                )
                bucket_delta[bucket] = real_share - anchor_share

                # Tango paired CI: contingency table from `paired`
                # (computed once before the loop, qid-keyed per
                # v2.2.1 #5).
                n_both = sum(
                    1 for (rb, ab) in paired
                    if rb == bucket and ab == bucket
                )
                n_real_only = sum(
                    1 for (rb, ab) in paired
                    if rb == bucket and ab != bucket
                )
                n_anchor_only = sum(
                    1 for (rb, ab) in paired
                    if rb != bucket and ab == bucket
                )
                n_neither = sum(
                    1 for (rb, ab) in paired
                    if rb != bucket and ab != bucket
                )
                if (n_both + n_real_only
                        + n_anchor_only + n_neither) > 0:
                    bucket_delta_ci[bucket] = wald_paired_diff_ci(
                        n_both, n_real_only, n_anchor_only, n_neither,
                    )
                else:
                    bucket_delta_ci[bucket] = (0.0, 0.0)

            # Paired sign test on ordinal bucket diffs
            ordinal_map = {
                "exact": 4, "near": 3, "rough": 2, "miss": 1,
                "invalid": 0,
            }
            paired_pos = sum(
                1 for (rb, ab) in paired
                if ordinal_map.get(rb, 0) > ordinal_map.get(ab, 0)
            )
            paired_neg = sum(
                1 for (rb, ab) in paired
                if ordinal_map.get(rb, 0) < ordinal_map.get(ab, 0)
            )
            paired_ties = sum(
                1 for (rb, ab) in paired
                if ordinal_map.get(rb, 0) == ordinal_map.get(ab, 0)
            )
            sign_test_p_val, sign_test_variant = sign_test_p(
                paired_pos, paired_neg,
            )
            sign_test_n_pos = paired_pos
            sign_test_n_neg = paired_neg
            sign_test_n_ties = paired_ties

    # v2.2.1 #7: bootstrap_unit validation block removed (dead
    # parameter — no bootstrap loop was ever implemented).

    # ---- persistence baseline (continuation only) ----
    persistence_report: Optional[QAProbeReport] = None
    persistence_caveat_field: Optional[str] = None
    real_minus_persistence_delta: Optional[dict[str, float]] = None
    real_vs_persistence_p: Optional[float] = None
    # v2.8.3 Commit E (M10a): pairing-validity tag. Non-continuation
    # probes have no persistence baseline → NO_PERSISTENCE. For
    # ContinuationProbe we compute the baseline below and set OK
    # or PAIRING_FAILED based on the actual qid overlap.
    persistence_validity_field: Literal[
        "NO_PERSISTENCE", "OK", "PAIRING_FAILED", "UNKNOWN",
    ] = "NO_PERSISTENCE"

    if isinstance(probe, ContinuationProbe):
        persistence_report = _compute_persistence_baseline_report(
            question_set, manifest, provider_config,
        )
        persistence_caveat_field = _PERSISTENCE_CAVEAT
        # Per-bucket delta vs persistence
        real_buckets = _bucket_assignments(real_report)
        pers_buckets = _bucket_assignments(persistence_report)
        real_minus_persistence_delta = {}
        for bucket in (
            "exact", "near", "rough", "miss", "invalid",
        ):
            r_share = real_buckets.get(bucket, 0) / max(
                1, real_report.submitted_answers
            )
            p_share = pers_buckets.get(bucket, 0) / max(
                1, persistence_report.submitted_answers
            )
            real_minus_persistence_delta[bucket] = r_share - p_share
        # Sign test vs persistence
        ordinal_map = {
            "exact": 4, "near": 3, "rough": 2, "miss": 1, "invalid": 0,
        }
        # v2.2.1 #5: pair by qid (was position-based).
        paired_rp, persistence_pairing_notes = (
            _pair_scores_by_question_id(
                real_report, persistence_report,
            )
        )
        notes.extend(persistence_pairing_notes)
        # v2.8.3 Commit E (M10a): if the qid-paired list is empty the
        # real and persistence reports share no question_ids; the
        # sign-test would degenerate to sign_test_p(0, 0) = (1.0,
        # "trivial_n_zero"), a meaningless p-value that callers
        # cannot distinguish from a genuine null result. Suppress
        # derived stats and tag PAIRING_FAILED instead.
        if not paired_rp:
            persistence_validity_field = "PAIRING_FAILED"
            real_minus_persistence_delta = None
            real_vs_persistence_p = None
        else:
            persistence_validity_field = "OK"
            rp_pos = sum(
                1 for (rb, pb) in paired_rp
                if ordinal_map.get(rb, 0) > ordinal_map.get(pb, 0)
            )
            rp_neg = sum(
                1 for (rb, pb) in paired_rp
                if ordinal_map.get(rb, 0) < ordinal_map.get(pb, 0)
            )
            real_vs_persistence_p, _ = sign_test_p(rp_pos, rp_neg)

    notes.insert(0, _NON_TRANSITIVITY_NOTE)

    # v2.2.1 #7: scalar leakage index + paired SE + z. Computed
    # only when anchor present AND validity is OK (paired bands
    # available); else all None.
    scalar_index: Optional[float] = None
    scalar_index_se: Optional[float] = None
    scalar_index_z: Union[
        float,
        Literal["positive_infinite", "negative_infinite"],
        None,
    ] = None
    leakage_caveat: Optional[str] = None
    if anchor_report is not None and anchor_validity == "OK":
        # Reuse the qid-paired bands computed for Tango/sign-test.
        scalar_index, scalar_index_se, scalar_index_z = (
            _compute_scalar_leakage_index_and_z(paired)
        )
        leakage_caveat = _LEAKAGE_INDEX_CAVEAT

    # v2.2.1 #12 Commit C: surface per-side parser_used Counters
    # (rank probes only) + emit drift note when distributions are
    # non-degenerate.
    parser_used_dist_real: Optional[Mapping[str, int]] = None
    parser_used_dist_anchor: Optional[Mapping[str, int]] = None
    if isinstance(probe, RankContinuationProbe):
        if real_parser_counter is not None:
            parser_used_dist_real = dict(real_parser_counter)
        if anchor_parser_counter is not None:
            parser_used_dist_anchor = dict(anchor_parser_counter)
        _maybe_emit_parser_used_drift_note(
            parser_used_dist_real, parser_used_dist_anchor, notes,
        )

    return KnowledgeCheckReport(
        probe_kind=_probe_kind(probe),
        real_score=real_report,
        anchor_score=anchor_report,
        bucket_delta=bucket_delta,
        bucket_delta_ci=bucket_delta_ci,
        paired_sign_test_p=sign_test_p_val,
        paired_sign_test_n_positive=sign_test_n_pos,
        paired_sign_test_n_negative=sign_test_n_neg,
        paired_sign_test_n_ties=sign_test_n_ties,
        paired_sign_test_variant=sign_test_variant,
        real_parse_success_rate=real_report.parse_success_rate,
        real_refusal_rate=real_refusal,
        anchor_parse_success_rate=(
            anchor_report.parse_success_rate
            if anchor_report else None
        ),
        anchor_refusal_rate=anchor_refusal,
        real_effective_rate=real_eff,
        anchor_effective_rate=anchor_eff,
        anchor_validity=anchor_validity,
        persistence_baseline_score=persistence_report,
        persistence_caveat=persistence_caveat_field,
        real_minus_persistence_bucket_delta=(
            real_minus_persistence_delta
        ),
        real_vs_persistence_sign_test_p=real_vs_persistence_p,
        persistence_validity=persistence_validity_field,
        parsing_schema_hash=answers.parsing_schema_hash,
        parsing_schema_description=answers.parsing_schema_description,
        prompt_template_hash=answers.prompt_template_hash,
        prompt_template_description=answers.prompt_template_description,
        scalar_leakage_index=scalar_index,
        scalar_leakage_index_h0_se=scalar_index_se,
        scalar_leakage_index_z=scalar_index_z,
        leakage_index_caveat=leakage_caveat,
        normalization_preset=(
            probe.normalization_preset
            if isinstance(probe, RankContinuationProbe)
            else None
        ),
        parser_used_distribution_real=parser_used_dist_real,
        parser_used_distribution_anchor=parser_used_dist_anchor,
        vol_scale_provenance=(
            vol_scale_provenance if vol_scale_provenance else None
        ),
        report_uuid=str(uuid.uuid4()),
        wall_clock_utc=datetime.now(timezone.utc).isoformat(),
        notes=tuple(notes),
    )


def _probe_kind(probe: ProbeT) -> str:
    if isinstance(probe, RankContinuationProbe):
        return "rank_continuation"
    if isinstance(probe, ContinuationProbe):
        return "continuation"
    return "knowledge"


def _bucket_assignments(report: QAProbeReport) -> dict[str, int]:
    return dict(report.bands_breakdown)


def _pair_scores_by_question_id(
    real: QAProbeReport, anchor: QAProbeReport,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Pair (real_score.band, anchor_score.band) by question_id.

    v2.2.1 #5: replaces position-based pairing which silently
    misaligned when either side dropped/reordered questions.

    Returns (pairs, notes). Degraded behavior, not crash:
      - Disjoint qid sets → returns ([], [warning_message]).
      - Partial overlap (len(common) < min(len(real),
        len(anchor))) → returns (pairs_for_common, [warning_message]).
      - Full overlap → returns (pairs, []).

    Pairs are sorted by question_id for deterministic output.
    """
    real_by_id = {s.question_id: s for s in real.question_scores}
    anchor_by_id = {s.question_id: s for s in anchor.question_scores}
    common = set(real_by_id) & set(anchor_by_id)
    notes: list[str] = []
    if not common:
        notes.append(
            "real and anchor question sets share NO question_ids; "
            "the M5 §6.5 anchor-pairing contract is violated. "
            "bucket_delta and paired_sign_test_p suppressed."
        )
        return [], notes
    if len(common) < min(len(real_by_id), len(anchor_by_id)):
        notes.append(
            f"real and anchor question sets only partially overlap "
            f"({len(common)} common qids out of "
            f"{len(real_by_id)} real / {len(anchor_by_id)} anchor); "
            f"pairing only the common qids."
        )
    pairs = [
        (real_by_id[qid].band, anchor_by_id[qid].band)
        for qid in sorted(common)
    ]
    return pairs, notes


def _compute_persistence_baseline_report(
    question_set: QuestionSet,
    manifest: Optional[dict],
    provider_config: dict,
) -> QAProbeReport:
    """Synthesize a persistence-baseline AttestedAnswers and score it
    against the same question_set.
    """
    recs = []
    for q in question_set:
        try:
            pred = _persistence_predict(q)
        except ValueError:
            # Template not eligible; mark as invalid.
            recs.append(
                AnswerRecord(
                    question_id=q.question_id,
                    raw_answer=None,
                    parsed_answer=None,
                    parse_status="invalid",
                )
            )
            continue
        recs.append(
            AnswerRecord(
                question_id=q.question_id,
                raw_answer=str(pred),
                parsed_answer=pred,
                parse_status="valid",
            )
        )
    attested = AttestedAnswers(
        answers=tuple(recs),
        parsing_schema_hash="0" * 64,
        parsing_schema_description="persistence_baseline_synthetic",
        prompt_template_hash="0" * 64,
        prompt_template_description=(
            "persistence_baseline (engine-computed; deterministic)"
        ),
    )
    report, _, _ = score_attested_answers(
        question_set, attested,
        manifest=manifest, provider_config=provider_config,
    )
    return report


__all__ = [
    "KnowledgeCheckReport",
    "LEAKAGE_INDEX_BUCKET_WEIGHTS",
    "compute_effective_rate",
    "compute_refusal_rate",
    "knowledge_check",
    "looks_like_refusal",
    "sign_test_p",
    "tango_paired_diff_ci",  # v2.8.1: deprecation alias; v2.9 removes.
    "wald_paired_diff_ci",
]
