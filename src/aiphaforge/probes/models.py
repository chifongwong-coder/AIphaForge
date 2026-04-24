"""Shared dataclasses for the v2.0 memory-probe toolkit.

This module defines the data contracts — no scoring logic, no runner
logic, no transforms. Implementation modules import these types.

The plan for these dataclasses lives in `docs/plans/v2.0-plan.md`
(private). Field semantics, populated/None contracts, and tier-1+
invariants are documented there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

if TYPE_CHECKING:
    import pandas as pd

# `AgentContract` describes what kind of orders the agent emits in a
# scenario; this controls which transforms are admissible in
# `view_only` mode (price-specific orders require invertible level
# transforms).
AgentContract = Literal[
    "signal_only",
    "market_orders_only",
    "price_orders_allowed",
]


# ---------- Q&A probe ----------

@dataclass
class ToleranceProfile:
    """Numeric scoring tolerance for a single question template.

    There is no global default — each built-in numeric template ships
    its own profile. Users may override it per question or per set.

    Built-in numeric templates MUST set a non-None ``max_range_width``
    (recommended default: ``4 * rough_range_width``) so anti-gaming
    protection is on by default.
    """

    absolute_floor: float
    exact_threshold: float
    near_threshold: float
    rough_threshold: float
    exact_range_width: float
    near_range_width: float
    rough_range_width: float
    max_range_width: Optional[float] = None
    sign_sensitive: bool = False
    sign_epsilon: float = 0.0


@dataclass
class QuestionSpec:
    """A single probe question, with truth value, before export."""

    question_id: str
    symbol: str
    timestamp: "pd.Timestamp"
    template_id: str
    answer_type: Literal["binary", "choice", "numeric"]
    prompt_text: str
    choices: Optional[list[str]]
    truth_value: Any
    tolerance: Optional[ToleranceProfile]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionPromptRecord:
    """Export shape for the user-facing question sheet.

    Critical: this record MUST NOT contain ``truth_value``,
    ``tolerance``, or any field that would let the user (or their
    LLM) see the answer key. The exporter validates this.
    """

    question_id: str
    prompt_text: str
    answer_type: Literal["binary", "choice", "numeric"]
    choices: Optional[list[str]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerKeyRecord:
    """Export shape for the engine-facing answer key.

    Users are warned in docs not to feed this file back into the LLM.
    """

    question_id: str
    truth_value: Any
    tolerance: Optional[ToleranceProfile]
    template_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerRecord:
    """User-submitted answer to a single question.

    The engine does not see the LLM prompt or reply text. It only
    consumes typed parsed answers plus a parse-status flag.
    """

    question_id: str
    raw_answer: Optional[str]
    parsed_answer: Any
    parse_status: Literal["valid", "invalid", "missing", "refusal"]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionScore:
    """Per-question scoring result."""

    question_id: str
    validity: Literal["valid", "invalid", "missing", "refusal"]
    band: Literal["exact", "near", "rough", "miss", "invalid"]
    truth_value: Any
    parsed_answer: Any
    relative_error: Optional[float]
    contains_truth: Optional[bool]
    range_width_ratio: Optional[float]
    max_range_width_exceeded: Optional[bool]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAProbeReport:
    """Aggregate Q&A probe report.

    Range-width aggregates (`mean_range_width_ratio`,
    `median_range_width_ratio`) are computed only over valid
    range-form answers. If no submitted answer was a range (or all
    were degenerate/invalid), they are ``None`` and
    ``max_range_width_exceeded_count`` is 0.
    """

    total_questions: int
    submitted_answers: int
    valid_answers: int
    invalid_answers: int
    missing_answers: int
    refusal_answers: int
    coverage_rate: float
    parse_success_rate: float
    exact_rate: float
    near_rate: float
    rough_rate: float
    miss_rate: float
    band_index_arbitrary: Optional[float]
    bands_breakdown: dict[str, int]
    mean_range_width_ratio: Optional[float]
    median_range_width_ratio: Optional[float]
    max_range_width_exceeded_count: int
    by_template: Optional["pd.DataFrame"]
    by_symbol: Optional["pd.DataFrame"]
    by_period: Optional["pd.DataFrame"]
    question_scores: list[QuestionScore]
    manifest: dict[str, Any]


# ---------- A/B probe ----------

@dataclass
class MetricConfig:
    """Per-metric configuration for the A/B runner."""

    higher_is_better: bool
    normalization_floor: float
    low_anchor_threshold: float


@dataclass
class ABScenario:
    """A single A/B scenario specification.

    Each scenario describes a transform pipeline applied to the data
    in either `view_only` or `market_level` mode. The runner sweeps
    a list of scenarios so the user can graduate transform strength.
    """

    scenario_id: str
    mode: Literal["view_only", "market_level"]
    # Transform instances are not in this module to keep `models.py`
    # dependency-light. The runner accepts any object satisfying the
    # `DataTransform` protocol from `transforms.py`.
    transforms: list[Any]
    notes: Optional[str] = None


@dataclass
class MetricDropSummary:
    """Per-metric drop statistics for a single scenario.

    Populated/None contract (see `docs/plans/v2.0-plan.md` §3.10):
    - List fields ``ai_raw``/``ai_test``/``baseline_raw``/
      ``baseline_test``/``ai_abs_drop``/``baseline_abs_drop`` are
      always populated, length equals ``n_repeat_requested``.
    - ``ai_rel_drop[r]``, ``baseline_rel_drop[r]``, ``excess_drop[r]``
      are ``None`` for low-anchor repeats; otherwise floats.
    - ``dominance_rate`` and ``mean_*``/``median_*``/``std_*``/``p10_*``/
      ``p90_*`` are ``None`` when ``n_valid_relative < min_valid_repeats``.
    - ``ai_noise_*``/``baseline_noise_*`` and their aggregates are
      populated only when the AI-on-AI control is enabled AND the
      scenario has at least one stochastic transform.
    """

    metric: str
    ai_raw: list[float]
    ai_test: list[float]
    baseline_raw: list[float]
    baseline_test: list[float]
    ai_abs_drop: list[float]
    baseline_abs_drop: list[float]
    ai_rel_drop: list[Optional[float]]
    baseline_rel_drop: list[Optional[float]]
    excess_drop: list[Optional[float]]
    n_valid_relative: int
    min_valid_repeats: int
    n_low_anchor_ai: int
    n_low_anchor_baseline: int
    dominance_rate: Optional[float]
    mean_excess_drop: Optional[float]
    median_excess_drop: Optional[float]
    std_excess_drop: Optional[float]
    p10_excess_drop: Optional[float]
    p90_excess_drop: Optional[float]
    ai_noise_abs: Optional[list[float]] = None
    ai_noise_rel: Optional[list[float]] = None
    baseline_noise_abs: Optional[list[float]] = None
    baseline_noise_rel: Optional[list[float]] = None
    mean_ai_noise_abs: Optional[float] = None
    median_ai_noise_abs: Optional[float] = None
    iqr_ai_noise_abs: Optional[float] = None
    mean_ai_noise_rel: Optional[float] = None
    median_ai_noise_rel: Optional[float] = None
    iqr_ai_noise_rel: Optional[float] = None


@dataclass
class ScenarioABReport:
    """Per-scenario A/B report."""

    scenario_id: str
    mode: Literal["view_only", "market_level"]
    n_repeat_requested: int
    n_unique_transform_realizations: int
    metric_summaries: dict[str, MetricDropSummary]
    per_repeat_table: Optional["pd.DataFrame"]
    warnings: list[str] = field(default_factory=list)


@dataclass
class ABProbeResult:
    """Top-level A/B probe result, one entry per scenario."""

    scenarios: list[ScenarioABReport]
    manifest: dict[str, Any]


# Recommended (non-enforced) keys for `provider_config` (manifest
# field on Q&A and A/B reports). Two papers using AIphaForge that
# report the same model with different keys produce non-comparable
# manifests; users and the demo page populate these keys when
# available so cross-run comparison is mechanically possible.
RECOMMENDED_PROVIDER_CONFIG_KEYS: Sequence[str] = (
    "model",
    "snapshot_id",
    "temperature",
    "top_p",
    "max_tokens",
    "seed",
    "prompt_template_hash",
    "system_prompt_hash",
    "tool_policy",
    "notes",
)
