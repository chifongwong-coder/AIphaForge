"""v2.2 M3: ContinuationProbe — forward-extrapolation Q&A probe.

Asks the LLM to predict bar[t + forward_horizon] given the prior
``context_bars`` of context. Different from KnowledgeProbe which asks
point-in-time questions about a named bar.

Design contract (per v2.2-plan-r6 § 4):
- ``context_bars`` and ``forward_horizon`` have NO defaults — both
  are task-design choices and silent default-driven drift between
  papers is the failure mode we are preventing.
- QuestionSpec carries the context window in ``metadata["context_window"]``
  as a list-of-dicts (CSV/JSON serialization is the user's choice).
- Numeric scoring path uses M2 vol-scaled tolerance by default for
  continuation templates (continuation-mode causality).
- ``forward_horizon > 1`` produces one QuestionSpec per forward step.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import pandas as pd

from aiphaforge.probes.models import (
    QuestionSpec,
    ToleranceProfile,
    VolScalingSpec,
)
from aiphaforge.probes.questions import (
    QuestionSet,
    _make_question_id,
)

# Default tolerance profiles for continuation templates. Use the
# US-equity vol-scaled preset since continuation probes are most
# common on equity. Users can override per-template.
_CONTINUATION_VOL_SCALE = VolScalingSpec(
    window=20,
    method="auto",
    near_factor=1.0,
    rough_factor=2.0,
    miss_floor_factor=5.0,
)


def _continuation_price_tolerance() -> ToleranceProfile:
    """Vol-scaled price tolerance for next-close / next-range
    continuation templates."""
    base = ToleranceProfile.us_equity_price()
    # Replace with a fresh ToleranceProfile carrying the vol_scale.
    return ToleranceProfile(
        absolute_floor=base.absolute_floor,
        exact_threshold=base.exact_threshold,
        near_threshold=base.near_threshold,
        rough_threshold=base.rough_threshold,
        exact_range_width=base.exact_range_width,
        near_range_width=base.near_range_width,
        rough_range_width=base.rough_range_width,
        max_range_width=base.max_range_width,
        sign_sensitive=base.sign_sensitive,
        sign_epsilon=base.sign_epsilon,
        vol_scale=_CONTINUATION_VOL_SCALE,
        preset_name=base.preset_name,
    )


def _continuation_return_tolerance() -> ToleranceProfile:
    """Tolerance for next-return continuation. Returns are unitless,
    so we use absolute thresholds (not relative) and a vol_scale.
    """
    return ToleranceProfile(
        absolute_floor=1e-6,
        exact_threshold=0.005,   # 50 bps absolute
        near_threshold=0.02,
        rough_threshold=0.05,
        exact_range_width=0.005,
        near_range_width=0.02,
        rough_range_width=0.05,
        max_range_width=0.20,
        sign_sensitive=True,     # direction matters for returns
        sign_epsilon=1e-4,
        vol_scale=_CONTINUATION_VOL_SCALE,
        preset_name="continuation_return",
    )


# ---------- Continuation templates ----------


class ContinuationTemplate:
    """Base for continuation templates. Subclasses override ``build``.

    Every template must populate ``QuestionSpec.metadata["context_window"]``
    with the prior bars so the prompt formatter can be swapped without
    rebuilding the question set, and
    ``QuestionSpec.metadata["forward_horizon"]`` with the integer
    horizon so M6's persistence baseline can dispatch per-template.
    """

    template_id: str

    def build(
        self,
        data: pd.DataFrame,
        symbol: str,
        anchor_ts: pd.Timestamp,
        *,
        context_bars: int,
        forward_horizon: int,
    ) -> Optional[QuestionSpec]:
        raise NotImplementedError


class NextCloseContinuation(ContinuationTemplate):
    """Predict close at anchor_ts + forward_horizon given the prior
    ``context_bars`` of OHLCV data.
    """

    template_id = "next_close_continuation"

    def build(
        self,
        data: pd.DataFrame,
        symbol: str,
        anchor_ts: pd.Timestamp,
        *,
        context_bars: int,
        forward_horizon: int,
    ) -> Optional[QuestionSpec]:
        idx_loc = data.index.get_loc(anchor_ts)
        if idx_loc + forward_horizon >= len(data):
            return None
        if idx_loc - context_bars + 1 < 0:
            return None
        ctx = data.iloc[idx_loc - context_bars + 1: idx_loc + 1]
        target_bar = data.iloc[idx_loc + forward_horizon]
        truth = float(target_bar["close"])
        prompt = (
            f"Given the prior {context_bars} bars of {symbol} ending "
            f"at {anchor_ts.isoformat()}, predict the close price at "
            f"step t+{forward_horizon} (timestamp "
            f"{data.index[idx_loc + forward_horizon].isoformat()})."
        )
        return QuestionSpec(
            question_id=_make_question_id(
                symbol, anchor_ts, f"{self.template_id}_h{forward_horizon}"
            ),
            symbol=symbol,
            timestamp=anchor_ts,
            template_id=self.template_id,
            answer_type="numeric",
            prompt_text=prompt,
            choices=None,
            truth_value=truth,
            tolerance=_continuation_price_tolerance(),
            metadata={
                "context_window": ctx.reset_index().to_dict(
                    orient="records"
                ),
                "context_bars": context_bars,
                "forward_horizon": forward_horizon,
                "anchor_timestamp": anchor_ts.isoformat(),
                "target_timestamp": (
                    data.index[idx_loc + forward_horizon].isoformat()
                ),
            },
        )


class NextRangeContinuation(ContinuationTemplate):
    """Predict (high, low) at anchor_ts + forward_horizon."""

    template_id = "next_range_continuation"

    def build(
        self,
        data: pd.DataFrame,
        symbol: str,
        anchor_ts: pd.Timestamp,
        *,
        context_bars: int,
        forward_horizon: int,
    ) -> Optional[QuestionSpec]:
        idx_loc = data.index.get_loc(anchor_ts)
        if idx_loc + forward_horizon >= len(data):
            return None
        if idx_loc - context_bars + 1 < 0:
            return None
        ctx = data.iloc[idx_loc - context_bars + 1: idx_loc + 1]
        target_bar = data.iloc[idx_loc + forward_horizon]
        # Truth is a (low, high) range; user submits a midpoint or
        # range answer scored via ToleranceProfile range mode.
        truth = (float(target_bar["low"]), float(target_bar["high"]))
        prompt = (
            f"Given the prior {context_bars} bars of {symbol} ending "
            f"at {anchor_ts.isoformat()}, predict the (low, high) "
            f"range at step t+{forward_horizon}."
        )
        return QuestionSpec(
            question_id=_make_question_id(
                symbol, anchor_ts, f"{self.template_id}_h{forward_horizon}"
            ),
            symbol=symbol,
            timestamp=anchor_ts,
            template_id=self.template_id,
            answer_type="numeric",
            prompt_text=prompt,
            choices=None,
            truth_value=truth,
            tolerance=_continuation_price_tolerance(),
            metadata={
                "context_window": ctx.reset_index().to_dict(
                    orient="records"
                ),
                "context_bars": context_bars,
                "forward_horizon": forward_horizon,
                "anchor_timestamp": anchor_ts.isoformat(),
                "target_timestamp": (
                    data.index[idx_loc + forward_horizon].isoformat()
                ),
            },
        )


class NextReturnContinuation(ContinuationTemplate):
    """Predict log return r[t + forward_horizon] = log(close[t+h] /
    close[t+h-1]). Sign-sensitive scoring.
    """

    template_id = "next_return_continuation"

    def build(
        self,
        data: pd.DataFrame,
        symbol: str,
        anchor_ts: pd.Timestamp,
        *,
        context_bars: int,
        forward_horizon: int,
    ) -> Optional[QuestionSpec]:
        idx_loc = data.index.get_loc(anchor_ts)
        if idx_loc + forward_horizon >= len(data):
            return None
        if idx_loc - context_bars + 1 < 0:
            return None
        if idx_loc + forward_horizon - 1 < 0:
            return None
        ctx = data.iloc[idx_loc - context_bars + 1: idx_loc + 1]
        c_t = float(data.iloc[idx_loc + forward_horizon]["close"])
        c_t_minus_1 = float(
            data.iloc[idx_loc + forward_horizon - 1]["close"]
        )
        if c_t_minus_1 <= 0 or c_t <= 0:
            return None
        import math
        truth = math.log(c_t / c_t_minus_1)
        prompt = (
            f"Given the prior {context_bars} bars of {symbol} ending "
            f"at {anchor_ts.isoformat()}, predict the log return "
            f"r[t+{forward_horizon}] = log(close[t+{forward_horizon}] "
            f"/ close[t+{forward_horizon - 1}])."
        )
        return QuestionSpec(
            question_id=_make_question_id(
                symbol, anchor_ts, f"{self.template_id}_h{forward_horizon}"
            ),
            symbol=symbol,
            timestamp=anchor_ts,
            template_id=self.template_id,
            answer_type="numeric",
            prompt_text=prompt,
            choices=None,
            truth_value=truth,
            tolerance=_continuation_return_tolerance(),
            metadata={
                "context_window": ctx.reset_index().to_dict(
                    orient="records"
                ),
                "context_bars": context_bars,
                "forward_horizon": forward_horizon,
                "anchor_timestamp": anchor_ts.isoformat(),
                "target_timestamp": (
                    data.index[idx_loc + forward_horizon].isoformat()
                ),
            },
        )


# ---------- ContinuationProbe facade ----------


class ContinuationProbe:
    """Q&A probe asking the LLM to extrapolate from a context window.

    Both ``context_bars`` and ``forward_horizon`` are
    **task-design choices with NO engine-side default**. The user
    must pass both explicitly so silent drift between papers cannot
    produce incomparable numbers.

    See ``docs/plans/v2.2-plan-r6.md`` § 4.
    """

    def __init__(
        self,
        symbol: str,
        context_bars: int,        # NO DEFAULT
        forward_horizon: int,     # NO DEFAULT
        templates: Sequence[Union[ContinuationTemplate, type]] = (),
    ):
        if not isinstance(context_bars, int) or context_bars < 2:
            raise ValueError(
                "ContinuationProbe.context_bars must be an int >= 2; "
                f"got {context_bars!r}"
            )
        if not isinstance(forward_horizon, int) or forward_horizon < 1:
            raise ValueError(
                "ContinuationProbe.forward_horizon must be an int "
                f">= 1; got {forward_horizon!r}"
            )
        self.symbol = symbol
        self.context_bars = context_bars
        self.forward_horizon = forward_horizon
        self.templates: list[ContinuationTemplate] = [
            t() if isinstance(t, type) else t for t in templates
        ]

    def build(
        self,
        data: pd.DataFrame,
        timestamps: Sequence[pd.Timestamp],
    ) -> QuestionSet:
        """Build the question set for the configured (symbol,
        context_bars, forward_horizon) over the given anchor timestamps.

        For ``forward_horizon > 1``, one QuestionSpec is emitted per
        forward step (1..forward_horizon) per anchor timestamp per
        template. Anchor timestamps that are too close to either edge
        of ``data`` are silently skipped.
        """
        specs: list[QuestionSpec] = []
        for ts in timestamps:
            if ts not in data.index:
                continue
            for template in self.templates:
                for h in range(1, self.forward_horizon + 1):
                    spec = template.build(
                        data, self.symbol, ts,
                        context_bars=self.context_bars,
                        forward_horizon=h,
                    )
                    if spec is not None:
                        specs.append(spec)
        return QuestionSet(specs)


CONTINUATION_TEMPLATES: tuple[type, ...] = (
    NextCloseContinuation,
    NextRangeContinuation,
    NextReturnContinuation,
)


__all__ = [
    "CONTINUATION_TEMPLATES",
    "ContinuationProbe",
    "ContinuationTemplate",
    "NextCloseContinuation",
    "NextRangeContinuation",
    "NextReturnContinuation",
]
