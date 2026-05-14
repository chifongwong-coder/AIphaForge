"""v2.4 Commit I — FactorRuleStrategy: factor + rule -> signal bridge.

The minimum factor-aware strategy: takes a single ``BaseFactor`` and
a rule (anything with ``.transform(scores)``), composes them into a
``BaseStrategy``-compatible object the engine can run via
``engine.set_strategy(strategy)``.

v2.4 MVP enforces single-factor construction (master plan §6.4).
Multi-factor composition (e.g. ``LinearScoreRule`` with weights)
lands in v2.5 once the ``FactorSpec.is_primary`` field is honored
by a multi-factor variant.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Union

import pandas as pd

from aiphaforge.factors import BaseFactor, FactorSet
from aiphaforge.signals import prepare_signals_for_engine
from aiphaforge.strategies import BaseStrategy


class FactorRuleStrategy(BaseStrategy):
    """Single-factor factor-aware strategy.

    Pipeline:
      1. ``factor.compute(data)`` produces wide
         ``DataFrame[index=datetime, columns=symbol]``.
      2. ``rule.transform(values)`` converts to direction signals
         (``ThresholdScoreRule`` for absolute thresholds,
         ``CrossSectionalQuantileRule`` for rank-based).
      3. ``prepare_signals_for_engine`` aligns to engine shape.

    ``compute_factors(data)`` is exposed alongside
    ``generate_signals(data)`` so research workflows can introspect
    the factor without re-running the rule.

    Multi-factor anti-leak guard: passing a list / tuple of factors
    raises at construction time (single-factor MVP).
    """

    def __init__(
        self,
        factor: BaseFactor,
        rule: Any,
        name: Optional[str] = None,
    ):
        if isinstance(factor, (list, tuple)):
            raise ValueError(
                "FactorRuleStrategy enforces single-factor MVP per "
                "master plan §6.4. Multi-factor composition (e.g. "
                "LinearScoreRule with weights) lands in v2.5+ when "
                "FactorSpec.is_primary is honored. For now, pass a "
                "single BaseFactor instance."
            )
        if not isinstance(factor, BaseFactor):
            raise TypeError(
                f"factor must be a BaseFactor instance, got "
                f"{type(factor).__name__}"
            )
        if not hasattr(rule, "transform"):
            raise TypeError(
                f"rule must have a .transform(scores) method; got "
                f"{type(rule).__name__}"
            )
        self.factor = factor
        self.rule = rule
        self.name = name or f"FactorRule({factor.name})"

    def compute_factors(
        self,
        data: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],
    ) -> FactorSet:
        """Return a single-factor FactorSet for research / introspection."""
        values = self.factor.compute(data)
        spec_dict = (
            {self.factor.name: self.factor.spec}
            if self.factor.spec is not None else {}
        )
        return FactorSet(
            values={self.factor.name: values},
            specs=spec_dict,
        )

    def generate_signals(
        self,
        data: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],
    ) -> Union[pd.Series, dict[str, pd.Series]]:
        """factor → rule → signal pipeline, then engine-shape alignment."""
        factor_values = self.factor.compute(data)
        # Heuristic: single-asset (one-column DataFrame) calls
        # rule.transform with a Series so ThresholdScoreRule's
        # Series path fires; multi-asset stays as DataFrame for
        # CrossSectionalQuantileRule's DataFrame requirement.
        if isinstance(data, pd.DataFrame) and factor_values.shape[1] == 1:
            scores = factor_values.iloc[:, 0]
        else:
            scores = factor_values
        signal = self.rule.transform(scores)
        return prepare_signals_for_engine(signal, data)


__all__ = ["FactorRuleStrategy"]
