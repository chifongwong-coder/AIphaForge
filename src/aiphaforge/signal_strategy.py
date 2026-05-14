"""v2.3 Commit F — DirectSignalStrategy.

Wraps precomputed signals (or precomputed scores + a rule) in a
``BaseStrategy``-compatible object so ML / AI / external-source
signal pipelines become first-class strategies that the engine
can run via ``engine.set_strategy(strategy)``.

Two construction paths:

- ``DirectSignalStrategy(signals=...)`` — already-canonical signal
  values flow through ``prepare_signals_for_engine`` for shape
  alignment.
- ``DirectSignalStrategy(scores=..., rule=...)`` — raw scores get
  converted to direction signals by ``rule.transform`` before
  alignment.

Live-update API for MetaController workflows:

- ``strategy.update_signals(new_signals)`` — swap in a refreshed
  signal series mid-backtest. Clears any prior score-path state.
- ``strategy.update_scores(new_scores)`` — swap in refreshed
  scores. Requires that a ``rule`` was supplied at construction
  (or has been updated via ``update_params(rule=...)``).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Union

import pandas as pd

from aiphaforge.signals import prepare_signals_for_engine
from aiphaforge.strategies import BaseStrategy


class DirectSignalStrategy(BaseStrategy):
    """Wrap precomputed signals or scores+rule as a BaseStrategy.

    Parameters
    ----------
    signals
        Precomputed signal in any shape ``prepare_signals_for_engine``
        accepts (``pd.Series``, ``dict[str, pd.Series]``, or wide
        ``pd.DataFrame``). Mutually exclusive with ``scores``.
    scores
        Precomputed scores, either ``pd.Series`` (single asset) or
        wide ``pd.DataFrame``. Requires ``rule`` to be supplied.
    rule
        A ``ThresholdScoreRule`` or ``CrossSectionalQuantileRule``
        (any object with a ``.transform(scores)`` method). Required
        when ``scores`` is supplied; ignored otherwise.
    name
        Display name for the strategy.

    Notes
    -----
    Exactly one of ``signals`` or ``scores`` must be supplied.
    Supplying both, or neither, raises ``ValueError`` at construction.

    For multi-asset signals supplied as a single ``pd.Series``,
    callers must pre-broadcast (e.g. via
    ``prepare_signals_for_engine(sig, data_dict, broadcast=True)``)
    or use the wide-DataFrame / dict shape — this strategy does NOT
    auto-broadcast at ``generate_signals`` time because the engine's
    data shape isn't visible until then.
    """

    def __init__(
        self,
        signals: Optional[
            Union[pd.Series, Mapping[str, pd.Series], pd.DataFrame]
        ] = None,
        scores: Optional[Union[pd.Series, pd.DataFrame]] = None,
        rule: Optional[Any] = None,
        name: str = "Direct Signal",
    ):
        if (signals is None) == (scores is None):
            raise ValueError(
                "DirectSignalStrategy requires exactly one of "
                "`signals=` or `scores=`. Got "
                f"signals={'set' if signals is not None else 'None'}, "
                f"scores={'set' if scores is not None else 'None'}."
            )
        if scores is not None and rule is None:
            raise ValueError(
                "DirectSignalStrategy(scores=..., rule=...) requires a "
                "rule to convert scores to direction signals. Pass "
                "ThresholdScoreRule or CrossSectionalQuantileRule."
            )
        self._signals = signals
        self._scores = scores
        self.rule = rule
        self.name = name

    def generate_signals(
        self,
        data: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],
    ) -> Union[pd.Series, dict[str, pd.Series]]:
        if self._scores is not None:
            converted = self.rule.transform(self._scores)
            return prepare_signals_for_engine(converted, data)
        return prepare_signals_for_engine(self._signals, data)

    # ----- Live-update API (MetaController support) -----

    def update_signals(
        self,
        signals: Union[pd.Series, Mapping[str, pd.Series], pd.DataFrame],
    ) -> None:
        """Replace the underlying precomputed signals.

        For MetaController / live-model workflows where the upstream
        model emits a refreshed signal series mid-backtest. Clears
        any prior score-path state — the strategy is always in
        exactly one path.
        """
        self._signals = signals
        self._scores = None

    def update_scores(
        self,
        scores: Union[pd.Series, pd.DataFrame],
    ) -> None:
        """Replace the underlying scores.

        Requires that a ``rule`` was supplied at construction time
        (or has since been set via ``update_params(rule=...)``).
        Clears any prior signal-path state.
        """
        if self.rule is None:
            raise ValueError(
                "update_scores requires a rule to be set; this "
                "strategy was constructed via the signals path. "
                "Call update_params(rule=...) first or rebuild the "
                "strategy via DirectSignalStrategy(scores=..., rule=...)."
            )
        self._scores = scores
        self._signals = None

    def update_params(self, **kwargs) -> None:
        """Override BaseStrategy.update_params to allow ``name`` and
        ``rule`` updates only.

        For score / signal series updates, use the dedicated
        ``update_signals`` / ``update_scores`` methods — these are
        not "params" in the optimization sense.
        """
        allowed = {"name", "rule"}
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(
                f"DirectSignalStrategy.update_params only supports "
                f"{sorted(allowed)}; got unknown keys: {sorted(unknown)}. "
                f"For signal / score updates, use update_signals() / "
                f"update_scores()."
            )
        for k, v in kwargs.items():
            setattr(self, k, v)


__all__ = ["DirectSignalStrategy"]
