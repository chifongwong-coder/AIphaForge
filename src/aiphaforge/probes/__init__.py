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

from aiphaforge.probes.abtest import (
    DEFAULT_METRIC_CONFIG,
    DEFAULT_METRICS,
    MACrossBaseline,
    MeanRevBaseline,
    MomentumBaseline,
    run_ab_probe,
)
from aiphaforge.probes.models import (
    ABProbeResult,
    ABScenario,
    AgentContract,
    AnswerKeyRecord,
    AnswerRecord,
    MetricConfig,
    MetricDropSummary,
    QAProbeReport,
    QuestionPromptRecord,
    QuestionScore,
    QuestionSpec,
    ScenarioABReport,
    ToleranceProfile,
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
    score_question,
    serialize_answer_records,
)

__all__ = [
    "AgentContract",
    "AnswerKeyRecord",
    "AnswerRecord",
    "ABProbeResult",
    "ABScenario",
    "BarRangePct",
    "CloseQuestion",
    "CloseVsOpen",
    "DEFAULT_TEMPLATES",
    "GapVsPrevClose",
    "HighQuestion",
    "KnowledgeProbe",
    "LowQuestion",
    "MetricConfig",
    "MetricDropSummary",
    "OpenQuestion",
    "QAProbeReport",
    "QuestionPromptRecord",
    "QuestionScore",
    "QuestionSet",
    "QuestionSpec",
    "QuestionTemplate",
    "ReturnSign",
    "ScenarioABReport",
    "ToleranceProfile",
    "DEFAULT_METRIC_CONFIG",
    "DEFAULT_METRICS",
    "MACrossBaseline",
    "MeanRevBaseline",
    "MomentumBaseline",
    "aggregate_scores",
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
    "score_question",
    "serialize_answer_records",
]
