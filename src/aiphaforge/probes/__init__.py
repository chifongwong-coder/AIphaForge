"""LLM memory probes — screening and inspection toolkit.

v2.0 of AIphaForge introduces two engine-level probes for measuring
training-data leakage in LLM-driven trading backtests:

- Q&A Probe — generate factual questions from historical data, score
  user-submitted answers against ground truth.
- A/B Probe — compare an AI agent and a baseline strategy on raw vs
  transformed/noised data; report relative sensitivity.

The framework returns descriptive comparisons, distributions, warnings,
and manifests. It does NOT issue verdicts and does NOT certify whether
a model is "honest" or "leaked". Interpretation is the user's
responsibility.

The engine never calls LLMs. Users run their LLM externally and submit
typed answers / hook-driven decisions back into the probe runners.
"""

from aiphaforge.probes._continuation import (
    CONTINUATION_TEMPLATES,
    ContinuationProbe,
    ContinuationTemplate,
    NextCloseContinuation,
    NextRangeContinuation,
    NextReturnContinuation,
)
from aiphaforge.probes._hash_utils import (
    attest_parsing,
    attest_prompt_template,
)
from aiphaforge.probes._rank import (
    NormalizationSpec,
    RankAnswer,
    RankContinuationProbe,
    RankCutoffSpec,
    normalize_symbol,
    score_rank_answer,
    tie_corrected_spearman,
)
from aiphaforge.probes.abtest import (
    DEFAULT_METRIC_CONFIG,
    DEFAULT_METRICS,
    MACrossBaseline,
    MeanRevBaseline,
    MomentumBaseline,
    run_ab_probe,
)
from aiphaforge.probes.anchors import build_synthetic_anchor
from aiphaforge.probes.models import (
    ABProbeResult,
    ABScenario,
    AgentContract,
    AnswerKeyRecord,
    AnswerRecord,
    AttestedAnswers,
    ContextSerializationSpec,
    ExemplarSpec,
    MetricConfig,
    MetricDropSummary,
    QAProbeReport,
    QuestionPromptRecord,
    QuestionScore,
    QuestionSpec,
    ScenarioABReport,
    ToleranceProfile,
    VolScalingSpec,
)
from aiphaforge.probes.questions import (
    DEFAULT_TEMPLATES,
    BarRangePct,
    CloseQuestion,
    CloseVsOpen,
    GapVsPrevClose,
    HighQuestion,
    KnowledgeProbe,
    LowQuestion,
    OpenQuestion,
    QuestionSet,
    QuestionTemplate,
    ReturnSign,
    build_question_set,
    build_question_sets_multi,
    sample_dates,
)
from aiphaforge.probes.scoring import (
    aggregate_scores,
    normalize_binary,
    normalize_direction,
    parse_binary_answer,
    parse_choice_answer,
    parse_numeric_answer,
    score_answer_file,
    score_attested_answers,
    score_question,
    serialize_answer_records,
)

__all__ = [
    "AgentContract",
    "AnswerKeyRecord",
    "AnswerRecord",
    "AttestedAnswers",
    "ABProbeResult",
    "ABScenario",
    "BarRangePct",
    "CONTINUATION_TEMPLATES",
    "CloseQuestion",
    "CloseVsOpen",
    "ContextSerializationSpec",
    "ContinuationProbe",
    "ContinuationTemplate",
    "DEFAULT_TEMPLATES",
    "ExemplarSpec",
    "GapVsPrevClose",
    "HighQuestion",
    "KnowledgeProbe",
    "LowQuestion",
    "MetricConfig",
    "MetricDropSummary",
    "NextCloseContinuation",
    "NextRangeContinuation",
    "NextReturnContinuation",
    "NormalizationSpec",
    "OpenQuestion",
    "QAProbeReport",
    "QuestionPromptRecord",
    "QuestionScore",
    "QuestionSet",
    "QuestionSpec",
    "QuestionTemplate",
    "RankAnswer",
    "RankContinuationProbe",
    "RankCutoffSpec",
    "ReturnSign",
    "ScenarioABReport",
    "ToleranceProfile",
    "VolScalingSpec",
    "DEFAULT_METRIC_CONFIG",
    "DEFAULT_METRICS",
    "MACrossBaseline",
    "MeanRevBaseline",
    "MomentumBaseline",
    "aggregate_scores",
    "attest_parsing",
    "attest_prompt_template",
    "build_synthetic_anchor",
    "normalize_symbol",
    "score_rank_answer",
    "tie_corrected_spearman",
    "build_question_set",
    "build_question_sets_multi",
    "normalize_binary",
    "normalize_direction",
    "parse_binary_answer",
    "parse_choice_answer",
    "parse_numeric_answer",
    "run_ab_probe",
    "sample_dates",
    "score_answer_file",
    "score_attested_answers",
    "score_question",
    "serialize_answer_records",
]
