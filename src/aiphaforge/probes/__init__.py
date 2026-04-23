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

__all__ = [
    "AgentContract",
    "AnswerKeyRecord",
    "AnswerRecord",
    "ABProbeResult",
    "ABScenario",
    "MetricConfig",
    "MetricDropSummary",
    "QAProbeReport",
    "QuestionPromptRecord",
    "QuestionScore",
    "QuestionSpec",
    "ScenarioABReport",
    "ToleranceProfile",
]
