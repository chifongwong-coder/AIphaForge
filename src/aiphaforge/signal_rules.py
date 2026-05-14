"""v2.3 Commit D — Score → signal rules.

R1 (master plan v1.0 §7): ML scores must NOT be auto-treated as
trading signals. This module provides explicit, typed rules that
convert ``pd.Series | DataFrame`` of scores into the canonical
``{NaN, 0, ±1}`` direction-signal contract.

Two rules ship in v2.3:

- :class:`ThresholdScoreRule` — single-asset / per-symbol absolute
  thresholds. Default ``neutral_action="hold"`` (NaN) — a score
  below the long threshold but above the short threshold means
  "ML model is uncertain", which leaves any existing position
  intact.

- :class:`CrossSectionalQuantileRule` — multi-asset rank-based.
  Default ``neutral_action="flat"`` (0) — daily cross-sectional
  rebalances expect every name to be re-ranked each bar; a name
  dropping out of the top quantile must be CLOSED, not held.
  The Alphalens / Qlib convention.

The asymmetric defaults are deliberate. See module section
"Default asymmetry" below for the full rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import pandas as pd

from aiphaforge.signals import transitions_only

# ---------------------------------------------------------------------------
# ThresholdScoreRule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdScoreRule:
    """Convert per-bar absolute scores into direction signals via thresholds.

    Parameters
    ----------
    long_threshold
        Score values >= this are mapped to +1 (long).
    short_threshold
        Score values <= this are mapped to -1 (short). If None, only
        long signals are generated; below-long scores follow
        ``neutral_action``.
    neutral_action
        What to emit for scores in the neutral band:

        - ``"hold"`` (DEFAULT) — emit NaN. Existing positions are
          left intact (engine ffills). Best for ML pipelines where
          the model is "uncertain" rather than asserting a flatten.
        - ``"flat"`` — emit 0. Engine closes any existing position.
    transition_only
        If True (default), pass through :func:`signals.transitions_only`
        to suppress repeated identical instructions.

    Examples
    --------
    Single-asset:

    >>> import pandas as pd
    >>> from aiphaforge.signal_rules import ThresholdScoreRule
    >>> idx = pd.date_range("2024-01-01", periods=4)
    >>> scores = pd.Series([0.85, 0.50, 0.10, 0.50], index=idx)
    >>> rule = ThresholdScoreRule(long_threshold=0.7, short_threshold=0.3)
    >>> rule.transform(scores)
    2024-01-01    1.0
    2024-01-02    NaN
    2024-01-03   -1.0
    2024-01-04    NaN
    Freq: D, dtype: float64
    """

    long_threshold: float
    short_threshold: Union[float, None] = None
    neutral_action: Literal["hold", "flat"] = "hold"
    transition_only: bool = True

    def transform(
        self,
        scores: Union[pd.Series, pd.DataFrame],
    ) -> Union[pd.Series, pd.DataFrame]:
        """Apply the threshold rule.

        Returns the same type as input (Series in / Series out;
        DataFrame in / DataFrame out — column-wise application).
        """
        if isinstance(scores, pd.Series):
            return self._transform_series(scores)
        if isinstance(scores, pd.DataFrame):
            out = scores.apply(self._transform_series, axis=0)
            return out
        raise TypeError(
            f"scores must be pd.Series or pd.DataFrame, got "
            f"{type(scores).__name__}"
        )

    def _transform_series(self, scores: pd.Series) -> pd.Series:
        neutral_value = (
            np.nan if self.neutral_action == "hold" else 0.0
        )
        raw = pd.Series(neutral_value, index=scores.index, dtype=float)
        raw[scores >= self.long_threshold] = 1.0
        if self.short_threshold is not None:
            raw[scores <= self.short_threshold] = -1.0
        if self.transition_only:
            return transitions_only(raw)
        return raw


# ---------------------------------------------------------------------------
# CrossSectionalQuantileRule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossSectionalQuantileRule:
    """Convert wide cross-sectional scores into long/short rank signals.

    For each row (timestamp), names whose score lands in the top
    ``long_quantile`` fraction get +1, names in the bottom
    ``short_quantile`` fraction get -1, and the middle fraction
    follows ``neutral_action``.

    Parameters
    ----------
    long_quantile
        Quantile threshold (in [0, 1]) above which scores get +1.
        Default 0.8 = top 20%.
    short_quantile
        Quantile threshold below which scores get -1. Default 0.2 =
        bottom 20%. If None, only long signals are generated.
    neutral_action
        What to emit for the middle quantiles:

        - ``"flat"`` (DEFAULT) — emit 0. Engine closes positions on
          names that drop out of the top quantile. Matches the
          Alphalens / Qlib daily-rebalance convention.
        - ``"hold"`` — emit NaN. Engine ffills any existing position
          on middle-quantile names. Use with caution: stale long
          positions accumulate and silently inflate top-bucket
          turnover stats.
    transition_only
        If True (default), pass through :func:`signals.transitions_only`
        per column.

    Notes
    -----
    Input MUST be wide ``pd.DataFrame[index=datetime, columns=symbol]``.
    Series input raises ``TypeError`` because cross-sectional ranks
    require a column dimension. Use :class:`ThresholdScoreRule` for
    single-asset score → signal conversion.
    """

    long_quantile: float = 0.8
    short_quantile: Union[float, None] = 0.2
    neutral_action: Literal["hold", "flat"] = "flat"
    transition_only: bool = True

    def transform(self, scores: pd.DataFrame) -> pd.DataFrame:
        if isinstance(scores, pd.Series):
            raise TypeError(
                "CrossSectionalQuantileRule.transform requires a wide "
                "pd.DataFrame[index=datetime, columns=symbol]; got "
                "pd.Series. Use ThresholdScoreRule for single-asset "
                "absolute thresholds, or stack your Series into a "
                "1-column DataFrame to silence this error."
            )
        if not isinstance(scores, pd.DataFrame):
            raise TypeError(
                f"scores must be pd.DataFrame, got "
                f"{type(scores).__name__}"
            )
        long_thr = scores.quantile(self.long_quantile, axis=1)
        long_mask = scores.gt(long_thr, axis=0)
        if self.short_quantile is not None:
            short_thr = scores.quantile(self.short_quantile, axis=1)
            short_mask = scores.lt(short_thr, axis=0)
        else:
            short_mask = pd.DataFrame(
                False, index=scores.index, columns=scores.columns,
            )
        neutral_value = (
            np.nan if self.neutral_action == "hold" else 0.0
        )
        raw = pd.DataFrame(
            neutral_value, index=scores.index, columns=scores.columns,
            dtype=float,
        )
        raw = raw.mask(long_mask, 1.0)
        raw = raw.mask(short_mask, -1.0)
        # NaN scores (no rank) propagate through quantile() as NaN
        # thresholds → masks become False → values stay neutral.
        # Force NaN-input rows to remain NaN regardless of
        # neutral_action so missing data isn't silently flattened.
        raw = raw.where(~scores.isna(), other=np.nan)
        if self.transition_only:
            return raw.apply(transitions_only, axis=0)
        return raw


__all__ = [
    "ThresholdScoreRule",
    "CrossSectionalQuantileRule",
]
