"""Question template system for the v2.0 memory-probe Q&A runner.

Each template is a stateless class with a ``build(data, symbol, ts)``
method that produces a :class:`QuestionSpec` from the dataset. Ground
truth is always read directly from ``data``; the engine never calls
the LLM and has no narrative or commentary layer.

A :class:`QuestionSet` holds the generated questions and exports them
into two structurally-separate files:

- the **question sheet** (``QuestionPromptRecord``) — what the user
  sends to their LLM. No truth values, no tolerance.
- the **answer key** (``AnswerKeyRecord``) — what the engine uses to
  score. Users are warned in docs not to feed this file back to the
  model.

See `docs/plans/v2.0-plan.md` §1 for the full contract.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Literal, Optional, Protocol, Sequence, Union, runtime_checkable

import numpy as np
import pandas as pd

from aiphaforge.probes.models import (
    AnswerKeyRecord,
    QuestionPromptRecord,
    QuestionSpec,
    ToleranceProfile,
)

# ---------- Default tolerance profiles per built-in template ----------

# Price-level OHLC templates: relative tolerance on a price (e.g. AAPL
# close = $62.06; "near" = within 2%).
_PRICE_TOLERANCE = ToleranceProfile(
    absolute_floor=1e-6,
    exact_threshold=0.005,
    near_threshold=0.02,
    rough_threshold=0.05,
    exact_range_width=0.005,
    near_range_width=0.02,
    rough_range_width=0.05,
    max_range_width=0.20,  # 4 × rough_range_width
)

# Percentage templates (bar_range_pct, gap_vs_prev_close): truth is a
# fraction (e.g. 0.012 for 1.2%); tolerance is absolute error on the
# fraction itself.
_PCT_TOLERANCE = ToleranceProfile(
    absolute_floor=1e-6,
    exact_threshold=0.001,
    near_threshold=0.005,
    rough_threshold=0.02,
    exact_range_width=0.001,
    near_range_width=0.005,
    rough_range_width=0.02,
    max_range_width=0.08,  # 4 × rough_range_width
)


# ---------- Normalization / parsing helpers ----------

# Binary aliases are deliberately *not* directional (no up/higher/down/lower).
# Direction-style aliases live in `_DIRECTION_*` and are only routed to a
# binary template by templates whose semantics are explicitly "is X higher
# than Y" (currently none of the built-ins). Mixing directional aliases
# into binary scoring would silently mis-score templates of the form "Was
# the close BELOW $60?" where "down" is the user's hedge and unrelated to
# the truth value.
_BINARY_YES_ALIASES = frozenset({"yes", "y", "true", "t", "1"})
_BINARY_NO_ALIASES = frozenset({"no", "n", "false", "f", "0"})

_DIRECTION_UP_ALIASES = frozenset({
    "up", "higher", "positive", "rose", "rising", "increase", "increased",
})
_DIRECTION_DOWN_ALIASES = frozenset({
    "down", "lower", "negative", "fell", "falling", "decrease", "decreased",
})
_DIRECTION_UNCHANGED_ALIASES = frozenset({
    "unchanged", "flat", "same", "equal", "zero",
})


def _make_question_id(symbol: str, ts: pd.Timestamp, template_id: str) -> str:
    """Build a question_id that uniquely encodes (symbol, timestamp, template).

    Uses the full ISO timestamp, not just the date, so intraday probes do
    not silently collide. Symbols containing '|' are rejected to keep the
    pipe-delimited format unambiguous.
    """
    if "|" in symbol:
        raise ValueError(
            f"symbol must not contain '|' (used as question_id delimiter): {symbol!r}"
        )
    return f"{symbol}|{ts.isoformat()}|{template_id}"


def normalize_binary(text: str) -> Optional[bool]:
    """Normalize a free-text binary answer to True / False / None.

    Returns ``None`` when the text doesn't match either side; callers
    treat ``None`` as a parse failure (band ``invalid``).
    """
    if text is None:
        return None
    t = str(text).strip().lower()
    if t in _BINARY_YES_ALIASES:
        return True
    if t in _BINARY_NO_ALIASES:
        return False
    return None


def normalize_direction(text: str) -> Optional[str]:
    """Normalize a direction answer to ``up``/``down``/``unchanged`` or None."""
    if text is None:
        return None
    t = str(text).strip().lower()
    if t in _DIRECTION_UP_ALIASES:
        return "up"
    if t in _DIRECTION_DOWN_ALIASES:
        return "down"
    if t in _DIRECTION_UNCHANGED_ALIASES:
        return "unchanged"
    return None


# ---------- Template protocol ----------

@runtime_checkable
class QuestionTemplate(Protocol):
    """Every built-in / user template must match this protocol."""

    template_id: str
    answer_type: Literal["binary", "choice", "numeric"]

    def build(
        self, data: pd.DataFrame, symbol: str, ts: pd.Timestamp,
    ) -> QuestionSpec: ...


# ---------- Built-in templates ----------

_OHLC_PROMPT_SUFFIX = (
    " Answer as a single decimal number in the dataset's price units, "
    "with no currency symbol, no thousands separators, and no rounding "
    "below the bar's reported precision."
)


class _OHLCBase:
    """Shared implementation for same-bar OHLC numeric templates."""

    answer_type: Literal["numeric"] = "numeric"
    _column: str  # subclass sets

    def __init__(self, tolerance: Optional[ToleranceProfile] = None):
        self.tolerance = tolerance or _PRICE_TOLERANCE

    def _truth(self, data: pd.DataFrame, ts: pd.Timestamp) -> float:
        if ts not in data.index:
            raise KeyError(f"timestamp {ts} not in data index")
        return float(data.loc[ts, self._column])


class OpenQuestion(_OHLCBase):
    template_id = "open"
    _column = "open"

    def build(self, data, symbol, ts):
        return QuestionSpec(
            question_id=_make_question_id(symbol, ts, self.template_id),
            symbol=symbol,
            timestamp=ts,
            template_id=self.template_id,
            answer_type=self.answer_type,
            prompt_text=f"What was the open of {symbol} on {ts.date()}?" + _OHLC_PROMPT_SUFFIX,
            choices=None,
            truth_value=self._truth(data, ts),
            tolerance=self.tolerance,
            metadata={"source": "same_bar", "column": "open"},
        )


class HighQuestion(_OHLCBase):
    template_id = "high"
    _column = "high"

    def build(self, data, symbol, ts):
        return QuestionSpec(
            question_id=_make_question_id(symbol, ts, self.template_id),
            symbol=symbol,
            timestamp=ts,
            template_id=self.template_id,
            answer_type=self.answer_type,
            prompt_text=f"What was the high of {symbol} on {ts.date()}?" + _OHLC_PROMPT_SUFFIX,
            choices=None,
            truth_value=self._truth(data, ts),
            tolerance=self.tolerance,
            metadata={"source": "same_bar", "column": "high"},
        )


class LowQuestion(_OHLCBase):
    template_id = "low"
    _column = "low"

    def build(self, data, symbol, ts):
        return QuestionSpec(
            question_id=_make_question_id(symbol, ts, self.template_id),
            symbol=symbol,
            timestamp=ts,
            template_id=self.template_id,
            answer_type=self.answer_type,
            prompt_text=f"What was the low of {symbol} on {ts.date()}?" + _OHLC_PROMPT_SUFFIX,
            choices=None,
            truth_value=self._truth(data, ts),
            tolerance=self.tolerance,
            metadata={"source": "same_bar", "column": "low"},
        )


class CloseQuestion(_OHLCBase):
    template_id = "close"
    _column = "close"

    def build(self, data, symbol, ts):
        return QuestionSpec(
            question_id=_make_question_id(symbol, ts, self.template_id),
            symbol=symbol,
            timestamp=ts,
            template_id=self.template_id,
            answer_type=self.answer_type,
            prompt_text=f"What was the closing price of {symbol} on {ts.date()}?" + _OHLC_PROMPT_SUFFIX,
            choices=None,
            truth_value=self._truth(data, ts),
            tolerance=self.tolerance,
            metadata={"source": "same_bar", "column": "close"},
        )


class CloseVsOpen:
    """Same-bar direction: sign(close - open)."""

    template_id = "close_vs_open"
    answer_type: Literal["choice"] = "choice"
    _CHOICES = ["up", "down", "unchanged"]

    def __init__(self, sign_epsilon: float = 1e-6):
        self.sign_epsilon = sign_epsilon

    def build(self, data, symbol, ts):
        if ts not in data.index:
            raise KeyError(f"timestamp {ts} not in data index")
        bar_open = float(data.loc[ts, "open"])
        bar_close = float(data.loc[ts, "close"])
        diff = bar_close - bar_open
        if abs(diff) <= self.sign_epsilon * max(abs(bar_open), 1e-12):
            truth = "unchanged"
        elif diff > 0:
            truth = "up"
        else:
            truth = "down"
        return QuestionSpec(
            question_id=_make_question_id(symbol, ts, self.template_id),
            symbol=symbol,
            timestamp=ts,
            template_id=self.template_id,
            answer_type=self.answer_type,
            prompt_text=(
                f"For {symbol} on {ts.date()}, was the close higher than, "
                "lower than, or roughly equal to the open? "
                "Answer with 'up', 'down', or 'unchanged'."
            ),
            choices=list(self._CHOICES),
            truth_value=truth,
            tolerance=None,
            metadata={"source": "same_bar", "sign_epsilon": self.sign_epsilon},
        )


class GapVsPrevClose:
    """Same-bar open gap vs prior close: (open_t - close_{t-1}) / close_{t-1}."""

    template_id = "gap_vs_prev_close"
    answer_type: Literal["numeric"] = "numeric"

    def __init__(self, tolerance: Optional[ToleranceProfile] = None):
        # Sign matters here: a +1.2% gap up is qualitatively different from
        # a -1.2% gap down. Default to a sign-sensitive profile so a model
        # that gets the magnitude right but the direction wrong is scored
        # `miss` regardless of magnitude band.
        if tolerance is None:
            tolerance = ToleranceProfile(
                absolute_floor=_PCT_TOLERANCE.absolute_floor,
                exact_threshold=_PCT_TOLERANCE.exact_threshold,
                near_threshold=_PCT_TOLERANCE.near_threshold,
                rough_threshold=_PCT_TOLERANCE.rough_threshold,
                exact_range_width=_PCT_TOLERANCE.exact_range_width,
                near_range_width=_PCT_TOLERANCE.near_range_width,
                rough_range_width=_PCT_TOLERANCE.rough_range_width,
                max_range_width=_PCT_TOLERANCE.max_range_width,
                sign_sensitive=True,
                sign_epsilon=1e-4,
            )
        self.tolerance = tolerance

    def build(self, data, symbol, ts):
        if ts not in data.index:
            raise KeyError(f"timestamp {ts} not in data index")
        pos = data.index.get_loc(ts)
        if pos == 0:
            raise ValueError(
                f"GapVsPrevClose template cannot build for the first bar "
                f"(ts={ts}); no previous close available"
            )
        bar_open = float(data.loc[ts, "open"])
        prev_close = float(data.iloc[pos - 1]["close"])
        gap = (bar_open - prev_close) / prev_close
        return QuestionSpec(
            question_id=_make_question_id(symbol, ts, self.template_id),
            symbol=symbol,
            timestamp=ts,
            template_id=self.template_id,
            answer_type=self.answer_type,
            prompt_text=(
                f"What was the open-vs-previous-close gap for {symbol} on "
                f"{ts.date()}, expressed as a decimal fraction "
                "(e.g., 0.012 for +1.2%)?"
            ),
            choices=None,
            truth_value=gap,
            tolerance=self.tolerance,
            metadata={"source": "prev_close_anchored"},
        )


class BarRangePct:
    """Same-bar range as a fraction of close: (high - low) / close."""

    template_id = "bar_range_pct"
    answer_type: Literal["numeric"] = "numeric"

    def __init__(self, tolerance: Optional[ToleranceProfile] = None):
        self.tolerance = tolerance or _PCT_TOLERANCE

    def build(self, data, symbol, ts):
        if ts not in data.index:
            raise KeyError(f"timestamp {ts} not in data index")
        bar_high = float(data.loc[ts, "high"])
        bar_low = float(data.loc[ts, "low"])
        bar_close = float(data.loc[ts, "close"])
        rng = (bar_high - bar_low) / bar_close
        return QuestionSpec(
            question_id=_make_question_id(symbol, ts, self.template_id),
            symbol=symbol,
            timestamp=ts,
            template_id=self.template_id,
            answer_type=self.answer_type,
            prompt_text=(
                f"What was the intraday bar range for {symbol} on "
                f"{ts.date()} as a decimal fraction of the close, "
                "i.e., (high - low) / close?"
            ),
            choices=None,
            truth_value=rng,
            tolerance=self.tolerance,
            metadata={"source": "same_bar", "formula": "(high-low)/close"},
        )


class ReturnSign:
    """Previous-close to current-close direction: sign(close_t - close_{t-1})."""

    template_id = "return_sign"
    answer_type: Literal["choice"] = "choice"
    _CHOICES = ["up", "down", "unchanged"]

    def __init__(self, sign_epsilon: float = 1e-4):
        self.sign_epsilon = sign_epsilon

    def build(self, data, symbol, ts):
        if ts not in data.index:
            raise KeyError(f"timestamp {ts} not in data index")
        pos = data.index.get_loc(ts)
        if pos == 0:
            raise ValueError(
                f"ReturnSign template cannot build for the first bar "
                f"(ts={ts}); no previous close available"
            )
        bar_close = float(data.loc[ts, "close"])
        prev_close = float(data.iloc[pos - 1]["close"])
        diff = bar_close - prev_close
        if abs(diff) <= self.sign_epsilon * max(abs(prev_close), 1e-12):
            truth = "unchanged"
        elif diff > 0:
            truth = "up"
        else:
            truth = "down"
        return QuestionSpec(
            question_id=_make_question_id(symbol, ts, self.template_id),
            symbol=symbol,
            timestamp=ts,
            template_id=self.template_id,
            answer_type=self.answer_type,
            prompt_text=(
                f"For {symbol}, was the close on {ts.date()} higher than, "
                "lower than, or roughly equal to the previous close? "
                "Answer with 'up', 'down', or 'unchanged'."
            ),
            choices=list(self._CHOICES),
            truth_value=truth,
            tolerance=None,
            metadata={"source": "prev_close_anchored", "sign_epsilon": self.sign_epsilon},
        )


DEFAULT_TEMPLATES: tuple[type, ...] = (
    OpenQuestion,
    HighQuestion,
    LowQuestion,
    CloseQuestion,
    CloseVsOpen,
    GapVsPrevClose,
    BarRangePct,
    ReturnSign,
)


# ---------- Sampling helper ----------

def sample_dates(
    data: pd.DataFrame,
    n: int,
    *,
    seed: int = 0,
    start: int = 0,
) -> list[pd.Timestamp]:
    """Randomly pick ``n`` distinct timestamps from ``data.index``.

    The first ``start`` bars are excluded from the sample so templates
    that require a previous close (e.g. ``ReturnSign``,
    ``GapVsPrevClose``) always have a valid lookback.
    """
    if n <= 0:
        raise ValueError(f"sample_dates n must be > 0, got {n}")
    candidates = data.index[start:]
    if len(candidates) < n:
        raise ValueError(
            f"sample_dates requested {n} dates but only {len(candidates)} "
            f"candidates available (len(data)={len(data)}, start={start})"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=n, replace=False)
    return sorted(candidates[i] for i in chosen)


# ---------- Question set + exporter ----------

class QuestionSet:
    """A collection of :class:`QuestionSpec` ready for export and scoring.

    Deduplicates on ``(symbol, timestamp, template_id, answer_type,
    normalized truth_value)`` at construction per plan §1.3.
    """

    def __init__(self, questions: Sequence[QuestionSpec]):
        self.questions: list[QuestionSpec] = self._deduplicate(list(questions))

    @staticmethod
    def _dedup_key(q: QuestionSpec):
        truth = q.truth_value
        # Normalize float truth values so bit-level differences don't
        # defeat dedup; categorical values are already strings.
        if isinstance(truth, float):
            truth = round(truth, 12)
        return (q.symbol, q.timestamp, q.template_id, q.answer_type, truth)

    @classmethod
    def _deduplicate(cls, qs: list[QuestionSpec]) -> list[QuestionSpec]:
        seen: set = set()
        out: list[QuestionSpec] = []
        for q in qs:
            key = cls._dedup_key(q)
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
        return out

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)

    def prompt_records(self) -> list[QuestionPromptRecord]:
        return [
            QuestionPromptRecord(
                question_id=q.question_id,
                prompt_text=q.prompt_text,
                answer_type=q.answer_type,
                choices=list(q.choices) if q.choices else None,
                metadata=dict(q.metadata),
            )
            for q in self.questions
        ]

    def answer_key_records(self) -> list[AnswerKeyRecord]:
        return [
            AnswerKeyRecord(
                question_id=q.question_id,
                truth_value=q.truth_value,
                tolerance=q.tolerance,
                template_id=q.template_id,
                metadata=dict(q.metadata),
            )
            for q in self.questions
        ]

    def export_questions(self, path: str) -> None:
        records = self.prompt_records()
        # Defensive structural check — this invariant is load-bearing
        # (it's the reason we use two separate record types). Use raise
        # rather than assert so it survives `python -O`.
        for r in records:
            d = asdict(r)
            if "truth_value" in d or "tolerance" in d:
                raise RuntimeError(
                    f"QuestionPromptRecord for {r.question_id} unexpectedly "
                    "contains answer-key fields; export aborted to prevent "
                    "leakage"
                )
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(_serialize_prompt(r)) + "\n")

    def export_answer_key(self, path: str) -> None:
        records = self.answer_key_records()
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(_serialize_answer_key(r)) + "\n")


def _serialize_prompt(r: QuestionPromptRecord) -> dict[str, Any]:
    return {
        "question_id": r.question_id,
        "prompt_text": r.prompt_text,
        "answer_type": r.answer_type,
        "choices": r.choices,
        "metadata": r.metadata,
    }


def _serialize_answer_key(r: AnswerKeyRecord) -> dict[str, Any]:
    return {
        "question_id": r.question_id,
        "truth_value": r.truth_value,
        "tolerance": asdict(r.tolerance) if r.tolerance is not None else None,
        "template_id": r.template_id,
        "metadata": r.metadata,
    }


# ---------- Builders ----------

def build_question_set(
    data: pd.DataFrame,
    symbol: str,
    timestamps: Sequence[pd.Timestamp],
    templates: Sequence[QuestionTemplate],
) -> QuestionSet:
    """Build a single-symbol question set across the given templates and
    timestamps.

    Templates that don't have enough lookback at a given timestamp
    (e.g. ``ReturnSign`` on the first bar) are skipped silently per
    template; other templates still produce questions at that bar.
    """
    out: list[QuestionSpec] = []
    for ts in timestamps:
        for t in templates:
            try:
                out.append(t.build(data, symbol, ts))
            except ValueError:
                # lookback not available — skip this (template, ts) pair.
                continue
    return QuestionSet(out)


class KnowledgeProbe:
    """High-level entry point for the Q&A workflow.

    Bundles the full Q&A pipeline behind one named class so the
    common case is a 4-line workflow:

    >>> probe = KnowledgeProbe(symbol="AAPL", templates=DEFAULT_TEMPLATES)
    >>> qs = probe.build(data, timestamps=sample_dates(data, n=50, seed=0))
    >>> qs.export_questions("questions.jsonl")
    >>> qs.export_answer_key("answer_key.jsonl")
    >>> # ... user runs LLM externally, writes answers.jsonl ...
    >>> report = probe.score("answers.jsonl", question_set=qs)

    The class is a thin facade over :func:`build_question_set` /
    :func:`build_question_sets_multi` / :func:`score_answer_file` —
    use those directly when you need finer control (e.g., per-symbol
    timestamp sampling).
    """

    def __init__(
        self,
        symbol: str,
        templates: Sequence[Union[QuestionTemplate, type]] = (),
    ):
        self.symbol = symbol
        # Accept either template instances or template classes (instantiated
        # with default profile). Plan example shows `OpenQuestion()` instances.
        self.templates: list[QuestionTemplate] = [
            t() if isinstance(t, type) else t for t in templates
        ]

    def build(
        self,
        data: pd.DataFrame,
        timestamps: Sequence[pd.Timestamp],
    ) -> QuestionSet:
        """Build the question set for this probe's symbol + templates."""
        return build_question_set(
            data, self.symbol, list(timestamps), self.templates
        )

    @staticmethod
    def score(
        answers_path: str,
        question_set: QuestionSet,
        *,
        manifest: Optional[dict[str, Any]] = None,
        provider_config: Optional[dict[str, Any]] = None,
    ):
        """Score a JSONL answers file. Re-exported for facade convenience.

        Implementation lives in :mod:`aiphaforge.probes.scoring`; this
        method just calls it. Kept on the class so users can write
        ``probe.score(...)`` symmetrically with ``probe.build(...)``.

        ``provider_config`` (v2.0.1) attaches the user's LLM
        configuration for cross-paper comparability. See
        :func:`aiphaforge.probes.score_answer_file` for the merge
        rule and recommended keys.
        """
        # Import here to avoid a circular import (scoring imports from
        # questions for normalize helpers).
        from aiphaforge.probes.scoring import score_answer_file
        return score_answer_file(
            question_set, answers_path,
            manifest=manifest, provider_config=provider_config,
        )


def build_question_sets_multi(
    data_dict: dict[str, pd.DataFrame],
    timestamps_by_symbol: dict[str, Sequence[pd.Timestamp]],
    templates: Sequence[QuestionTemplate],
) -> QuestionSet:
    """Build a question set spanning multiple symbols.

    Each symbol contributes its own (data, timestamps, templates)
    slice; the union is deduplicated at construction time.
    """
    combined: list[QuestionSpec] = []
    for symbol, data in data_dict.items():
        timestamps = timestamps_by_symbol.get(symbol, [])
        per_symbol = build_question_set(data, symbol, timestamps, templates)
        combined.extend(per_symbol.questions)
    return QuestionSet(combined)


# ---------- Convenience types ----------

NumericAnswer = Union[float, tuple[float, float]]
"""Parsed numeric answer: either a scalar or a ``(lo, hi)`` range."""
