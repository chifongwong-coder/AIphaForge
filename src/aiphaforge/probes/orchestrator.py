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
from typing import Any, Literal, Optional, Sequence, Union

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
    if n < 25:
        # v2.2.1 #9: scipy unavailable for n<25 — fall through to
        # continuity-corrected normal approximation (better small-
        # sample coverage than uncorrected). r5/r6 had two
        # identical branches; r7 removes the dead duplication and
        # uses the correction.
        z = (abs(n_pos - n / 2.0) - 0.5) / math.sqrt(n / 4.0)
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
# v2.2.1 #7: hardcoded ordinal weighting for scalar_leakage_index.
# Locked v2.2.1; changing requires __version__ bump per §14
# stability commitment.
_BUCKET_ORDINAL_WEIGHTS = {
    "exact": 4.0, "near": 3.0, "rough": 2.0, "miss": 1.0, "invalid": 0.0,
}
_LEAKAGE_INDEX_CAVEAT = (
    "scalar_leakage_index is a point estimate with hardcoded "
    "ordinal weighting (exact=4, near=3, rough=2, miss=1, "
    "invalid=0). Range [-4, +4]. SE accounts for paired correlation "
    "via per-question differences. Approximate significance: |z| >= "
    "1.96 corresponds to two-sided p < 0.05. For full inferential "
    "rigor, see bucket_delta_tango_ci. COMPARABILITY: comparable "
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


def _apply_vol_scaling_to_question_set(
    probe: "ProbeT",
    question_set: "QuestionSet",
    data: Union[pd.DataFrame, dict[str, pd.DataFrame]],
) -> tuple["QuestionSet", dict[str, dict[str, Any]]]:
    """v2.2.1 #1: rewrite each QuestionSpec.tolerance to the
    vol-scaled form when its tolerance carries a vol_scale spec.

    Returns (new_question_set, provenance_by_qid).

    Skips:
      - Questions whose tolerance is None.
      - Questions whose tolerance.vol_scale is None.
      - RankContinuationProbe (rank scoring uses Spearman ρ, not
        sigma-bands; vol_scale doesn't apply).

    For each scoped question:
      - KnowledgeProbe → point-in-time mode; sigma from bars
        strictly before qs.timestamp on the side's data.
      - ContinuationProbe → continuation mode; sigma from bars
        ending at the LAST context bar (held fixed across forward
        steps).
    """
    from dataclasses import replace as _dc_replace

    from ._vol import (
        apply_vol_scale,
        estimate_sigma_for_continuation,
        estimate_sigma_for_pointintime,
    )
    from .questions import QuestionSet
    if isinstance(probe, RankContinuationProbe):
        # Rank scoring doesn't use vol-scaled bands.
        return question_set, {}

    rewritten: list = []
    provenance: dict[str, dict[str, Any]] = {}

    for qs in question_set:
        if qs.tolerance is None or qs.tolerance.vol_scale is None:
            rewritten.append(qs)
            continue

        # data may be a DataFrame (single-symbol probes) or a dict
        # (RankContinuationProbe — but we excluded that above).
        if isinstance(data, dict):
            symbol_data = data.get(qs.symbol)
            if symbol_data is None:
                rewritten.append(qs)
                continue
        else:
            symbol_data = data

        spec = qs.tolerance.vol_scale
        if isinstance(probe, ContinuationProbe):
            ctx = qs.metadata.get("context_window")
            if not ctx:
                rewritten.append(qs)
                continue
            last_ctx_ts = pd.Timestamp(ctx[-1]["index"])
            sigma, prov = estimate_sigma_for_continuation(
                symbol_data, last_ctx_ts,
                window=spec.window,
                method=spec.method,
                preset_name=qs.tolerance.preset_name,
                log_returns=spec.log_returns,
            )
        else:
            # Point-in-time probes (KnowledgeProbe).
            sigma, prov = estimate_sigma_for_pointintime(
                symbol_data, qs.timestamp,
                window=spec.window,
                method=spec.method,
                preset_name=qs.tolerance.preset_name,
                log_returns=spec.log_returns,
            )
        scaled_tol = apply_vol_scale(qs.tolerance, sigma)
        rewritten.append(_dc_replace(qs, tolerance=scaled_tol))
        provenance[qs.question_id] = prov

    return QuestionSet(rewritten), provenance


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
        _BUCKET_ORDINAL_WEIGHTS.get(rb, 0.0)
        - _BUCKET_ORDINAL_WEIGHTS.get(ab, 0.0)
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

    # v2.2.1 #7: scalar leakage index + paired SE + z (typed
    # sentinel for SE==0). Range [-4, +4]. Point estimate; for
    # inference use bucket_delta_tango_ci. Computed for ALL
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
    CIs (``bucket_delta_tango_ci``), paired sign test
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
    question_set, real_vol_prov = _apply_vol_scaling_to_question_set(
        probe, question_set, data,
    )
    for qid, prov in real_vol_prov.items():
        vol_scale_provenance.setdefault(qid, {})["real"] = prov

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
        # v2.2.1 #1: vol-scaling wiring (anchor side) — uses
        # ANCHOR sigma, not real sigma. Both sides go through
        # the same rewrite path with their own data.
        anchor_qs, anchor_vol_prov = (
            _apply_vol_scaling_to_question_set(
                probe, anchor_qs, anchor,
            )
        )
        for qid, prov in anchor_vol_prov.items():
            vol_scale_provenance.setdefault(qid, {})["anchor"] = prov
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
                anchor_validity = "REFUSAL_SUSPECTED"  # closest
                # existing validity tag — TODO v2.3 add
                # "PAIRING_FAILED" tag.

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

    # v2.2.1 #7: bootstrap_unit validation block removed (dead
    # parameter — no bootstrap loop was ever implemented).

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
        # v2.2.1 #5: pair by qid (was position-based).
        paired_rp, persistence_pairing_notes = (
            _pair_scores_by_question_id(
                real_report, persistence_report,
            )
        )
        notes.extend(persistence_pairing_notes)
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
        scalar_leakage_index=scalar_index,
        scalar_leakage_index_h0_se=scalar_index_se,
        scalar_leakage_index_z=scalar_index_z,
        leakage_index_caveat=leakage_caveat,
        normalization_preset=(
            probe.normalization_preset
            if isinstance(probe, RankContinuationProbe)
            else None
        ),
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


def _pair_scores_by_position(
    real: QAProbeReport, anchor: QAProbeReport,
) -> list[tuple[str, str]]:
    """v2.2.0 legacy — pairs by positional index.

    DEPRECATED in favor of :func:`_pair_scores_by_question_id`
    (v2.2.1 #5). The position-based pairing silently misaligns when
    either side has dropped or reordered questions; the qid-based
    pairing degrades gracefully and surfaces a warning.

    Kept temporarily so callers can migrate; will be removed in
    v2.3.
    """
    n = min(len(real.question_scores), len(anchor.question_scores))
    return [
        (real.question_scores[i].band, anchor.question_scores[i].band)
        for i in range(n)
    ]


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
    "compute_effective_rate",
    "compute_refusal_rate",
    "knowledge_check",
    "looks_like_refusal",
    "sign_test_p",
    "tango_paired_diff_ci",
]
