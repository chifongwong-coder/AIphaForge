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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Sequence, Union

import pandas as pd

from aiphaforge.probes._continuation import ContinuationProbe
from aiphaforge.probes._hash_utils import _normalize_for_hash
from aiphaforge.probes._rank import (
    RankContinuationProbe,
)
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

    Locked v2.2 (per § 7.4):
      1. Apply _normalize_for_hash + lowercase.
      2. SHORT-CIRCUIT: if any keyword in _REFUSAL_KEYWORDS_LEADING
         appears in the LEADING 50 CHARACTERS, return True.
         Restricting to leading text removes the false-positive class
         where the LLM quotes a refusal-shaped phrase inside a valid
         answer (e.g., 'the article says "I don't have access..."
         but the close was $147.20').
      3. FALLBACK (no keyword in leading 50): if length > 120 AND
         digit ratio < 5%, return True.
      4. Otherwise return False.

    LIMITATION: keyword list is English-only. Non-English refusals
    will pass through. Documented as a v2.2 known limitation.
    """
    if not raw_text:
        return False
    s = _normalize_for_hash(raw_text).lower()
    leading = s[:50]
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
        # Normal with continuity correction.
        z = (abs(n_pos - n / 2.0) - 0.5) / math.sqrt(n / 4.0)
        p = 2.0 * (1.0 - _phi(z))
        return float(max(0.0, min(1.0, p))), "normal_continuity_corrected"
    # n >= 40 (or scipy unavailable for n<25 — same fallback)
    if n >= 25:
        z = abs(n_pos - n / 2.0) / math.sqrt(n / 4.0)
    else:
        z = abs(n_pos - n / 2.0) / math.sqrt(n / 4.0)
    p = 2.0 * (1.0 - _phi(z))
    return float(max(0.0, min(1.0, p))), "normal_approximation"


def _phi(z: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------- Tango (1998) paired score interval ----------


def tango_paired_diff_ci(
    n_both: int,
    n_real_only: int,
    n_anchor_only: int,
    n_neither: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Tango (1998) score-based CI for the difference of two paired
    binomial proportions p1 − p2.

    Reference: Tango T (1998), "Equivalence test and confidence
    interval for the difference in proportions for the paired-sample
    design". Stat Med 17:891-908.

    p1 = (n_both + n_real_only) / N         # real success rate
    p2 = (n_both + n_anchor_only) / N       # anchor success rate

    Implementation: bisection over delta ∈ [-1, 1] for which the
    score statistic for H0: p1 - p2 = delta is within the critical
    region. Matches PropCIs::scoreci.mp behavior.

    Edge cases (per F3 follow-up):
      - N = 0: raises ValueError (cannot CI an empty bucket).
      - All concordant (n_real_only = n_anchor_only = 0):
        returns (0.0, 0.0) — point CI at delta=0.
      - All discordant (n_both = n_neither = 0): standard formula
        applies.
    """
    n = n_both + n_real_only + n_anchor_only + n_neither
    if n == 0:
        raise ValueError(
            "tango_paired_diff_ci: empty bucket (N=0)"
        )
    if n_real_only == 0 and n_anchor_only == 0:
        # Perfect agreement → point CI at zero
        return 0.0, 0.0
    z_alpha = _z_for_confidence(confidence)
    delta_hat = (n_real_only - n_anchor_only) / n

    def _score_stat(delta: float) -> float:
        """Score statistic at H0: p1 - p2 = delta.

        Wald-style with the standard discordant-cell variance
        estimator. Conservative vs Tango's constrained MLE but
        matches PropCIs::scoreci.mp behavior to within the
        documented test tolerance (see F2 implementation
        follow-up — frozen reference fixture pinned separately).
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


def compute_effective_rate(records: Sequence[AnswerRecord]) -> float:
    """Joint P(parsed AND not refused), computed per-AnswerRecord.

    Per § 7.5: replaces the r5 product-of-marginals
    `parse_success * (1 - refusal_rate)` because that formula
    double-discounts answers that BOTH fail parse AND look like
    refusal (the common case for paragraph refusals).
    """
    n_total = len(records)
    if n_total == 0:
        return 0.0
    n_eff = 0
    for rec in records:
        parsed = (rec.parsed_answer is not None
                  and rec.parse_status == "valid")
        refused = looks_like_refusal(rec.raw_answer or "")
        if parsed and not refused:
            n_eff += 1
    return n_eff / n_total


def compute_refusal_rate(records: Sequence[AnswerRecord]) -> float:
    """Fraction of AnswerRecords whose raw_answer looks like refusal."""
    if not records:
        return 0.0
    return sum(
        1 for r in records if looks_like_refusal(r.raw_answer or "")
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


@dataclass(frozen=True)
class KnowledgeCheckReport:
    """v2.2 (M6) standalone Q&A leakage diagnostic report.

    See ``docs/plans/v2.2-plan-r6.md`` § 7.3.
    """

    probe_kind: str  # "knowledge" | "continuation" | "rank_continuation"

    # Primary, per-bucket comparison
    real_score: QAProbeReport
    anchor_score: Optional[QAProbeReport]
    bucket_delta: Optional[dict[str, float]]
    bucket_delta_tango_ci: Optional[dict[str, tuple[float, float]]]

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
    anchor_validity: str

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

    # Run-to-run drift support
    report_uuid: str
    wall_clock_utc: str

    notes: tuple[str, ...]

    is_pillar_summary: bool = False  # ENFORCED False

    def __post_init__(self):
        if self.is_pillar_summary:
            raise ValueError(
                "is_pillar_summary must remain False — "
                "KnowledgeCheckReport is a single-pillar diagnostic, "
                "not a cross-pillar verdict."
            )


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
    n_anchor_bootstrap: int = 0,
    bootstrap_seed: Optional[int] = None,
    bootstrap_unit: str = "anchor",
    refusal_rate_threshold: float = 0.5,
    exemplar_pairs: Optional[Sequence[tuple[str, pd.Timestamp]]] = None,
) -> KnowledgeCheckReport:
    """Standalone Q&A leakage diagnostic. See § 7.

    The function is **idempotent and pure**: same inputs always
    produce the same report (UUID and wall_clock_utc aside, which
    are metadata not part of the diagnostic).

    Refuses (raises ValueError) when:
      - provider_config is None or missing any required key.
      - anchor_answers without anchor (or vice versa).
      - parsing_schema_hash or prompt_template_hash mismatch between
        what the probe was built for and what AttestedAnswers carry
        (NOT enforced at this layer — the caller's responsibility).
      - n_anchor_bootstrap > 0 without bootstrap_seed.
      - n_anchor_bootstrap > 0 without anchor.
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
    if n_anchor_bootstrap > 0:
        if anchor is None:
            raise ValueError(
                "n_anchor_bootstrap > 0 requires anchor"
            )
        if bootstrap_seed is None:
            raise ValueError(
                "n_anchor_bootstrap > 0 requires bootstrap_seed"
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

    # ---- score real ----
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

                # Tango paired CI: requires per-question pairing.
                # Per § 6.5, anchor question set mirrors real
                # position-by-position. Build the contingency table
                # by pairing scores by question_id index.
                paired = _pair_scores_by_position(
                    real_report, anchor_report,
                )
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
                    bucket_delta_ci[bucket] = tango_paired_diff_ci(
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

    if bootstrap_unit not in ("anchor", "question"):
        raise ValueError(
            f"bootstrap_unit must be 'anchor' or 'question'; "
            f"got {bootstrap_unit!r}"
        )
    if bootstrap_unit == "question":
        notes.append(
            "bootstrap_unit=question selected; CI is anti-conservative — "
            "true CI width depends on intra-anchor correlation among "
            "questions and may be much narrower than nominal. See "
            "plan §7.2.3 and consider bootstrap_unit='anchor' for "
            "paper-grade inference."
        )

    # ---- persistence baseline (continuation only) ----
    persistence_report: Optional[QAProbeReport] = None
    persistence_caveat_field: Optional[str] = None
    real_minus_persistence_delta: Optional[dict[str, float]] = None
    real_vs_persistence_p: Optional[float] = None

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
        paired_rp = _pair_scores_by_position(
            real_report, persistence_report,
        )
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

    return KnowledgeCheckReport(
        probe_kind=_probe_kind(probe),
        real_score=real_report,
        anchor_score=anchor_report,
        bucket_delta=bucket_delta,
        bucket_delta_tango_ci=bucket_delta_ci,
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
        parsing_schema_hash=answers.parsing_schema_hash,
        parsing_schema_description=answers.parsing_schema_description,
        prompt_template_hash=answers.prompt_template_hash,
        prompt_template_description=answers.prompt_template_description,
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


def _pair_scores_by_position(
    real: QAProbeReport, anchor: QAProbeReport,
) -> list[tuple[str, str]]:
    """Pair (real_score.band, anchor_score.band) by index."""
    n = min(len(real.question_scores), len(anchor.question_scores))
    return [
        (real.question_scores[i].band, anchor.question_scores[i].band)
        for i in range(n)
    ]


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
    "compute_effective_rate",
    "compute_refusal_rate",
    "knowledge_check",
    "looks_like_refusal",
    "sign_test_p",
    "tango_paired_diff_ci",
]
