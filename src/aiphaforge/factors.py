"""v2.3 Commit H — Factor data structures (no compute, no engine).

Establishes the typed factor data structures EARLY so v2.4 can
build directly on them. Three pieces ship:

- :class:`FactorSpec` — frozen-dataclass description of a factor
  (name, family, direction, lookback, tags). Reserves an
  ``is_primary`` field for v2.5+ multi-factor strategies.
- :class:`FactorSet` — frozen-dataclass holding a collection of
  factor values keyed by name, with a parallel mapping of specs.
  ``FactorSet.empty()`` is the sentinel returned by signal-only
  strategies that have no factors to expose.
- :class:`FactorProvider` — Protocol for any object that can
  produce a FactorSet via ``compute_factors(data)``. Optional —
  strategies are never required to implement it.

Plus two layout helpers (``dict_to_factor_wide``,
``validate_factor_wide``) mirroring the signal-layer adapters.

Engine-agnostic by construction. No compute logic — that lands
in v2.4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import pandas as pd


@dataclass(frozen=True)
class FactorSpec:
    """Description of a factor — what it is, not what it computed.

    Pickle stability:
        FactorSpec is ``frozen=True``. Adding or renaming fields in
        a future minor version (v2.4+) WILL BREAK pickled FactorSet
        instances created against older versions. If you persist
        FactorSet across ``aiphaforge`` upgrades, serialize via
        JSON (using a manual converter) rather than pickle. The
        v2.x dataclass shape is NOT a stable persistence schema.

    Attributes
    ----------
    name
        Stable identifier used as the key in ``FactorSet.values``.
    description
        Human-readable description of what the factor measures.
    family
        Optional grouping label (e.g. "momentum", "mean_reversion",
        "volume", "carry").
    direction
        Hypothesised sign of the factor's predictive relationship:
        +1 = higher factor → higher forward return; -1 = lower
        factor → higher forward return; None = no prior.
    required_columns
        OHLCV columns the factor reads. Used for early-fail
        validation when the input data lacks them.
    lookback
        Number of bars of history the factor needs before its
        first non-NaN output. Used for warmup checks.
    is_primary
        RESERVED for v2.5+ multi-factor strategies. v2.4's
        FactorRuleStrategy enforces single-factor construction
        and ignores this field. Leave the field on the dataclass
        so the shape is stable across minor versions.
    tags
        Free-form tags for filtering / grouping in factor research.
    """

    name: str
    description: str = ""
    family: str | None = None
    direction: int | None = None
    required_columns: tuple[str, ...] = ()
    lookback: int | None = None
    is_primary: bool = False
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class FactorSet:
    """A collection of computed factor values + parallel specs.

    Pickle stability:
        Same caveat as :class:`FactorSpec`. Persist via JSON for
        cross-version stability.

    Attributes
    ----------
    values
        Per-factor wide DataFrame: ``index=datetime, columns=symbol``.
        Single-asset factors are stored as one-column DataFrame for
        layout uniformity (master plan v1.0 §2.4).
    specs
        Parallel mapping ``name → FactorSpec``. Keys MUST match
        ``values.keys()``.
    """

    values: Mapping[str, pd.DataFrame]
    specs: Mapping[str, FactorSpec]

    @classmethod
    def empty(cls) -> "FactorSet":
        """Sentinel for strategies that don't expose explicit factors.

        Master plan §0.4: signal-only strategies are first-class.
        Calling factor analysis on them returns ``FactorSet.empty()``,
        never raises.
        """
        return cls(values={}, specs={})


class FactorProvider(Protocol):
    """OPTIONAL Protocol for factor-aware objects.

    A strategy / model / data source matches this Protocol structurally
    if it has a ``compute_factors(data) -> FactorSet`` method. NO
    inheritance required — duck typing via :func:`isinstance` against
    the Protocol works for any class implementing the method.
    """

    def compute_factors(
        self,
        data: pd.DataFrame | Mapping[str, pd.DataFrame],
    ) -> FactorSet: ...


def dict_to_factor_wide(
    values: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Stack per-symbol Series into wide factor DataFrame.

    Mirrors :func:`aiphaforge.signals.dict_to_signal_wide` for the
    factor canonical layout.

    Parameters
    ----------
    values
        Per-symbol factor Series. Symbol order in the input mapping
        determines column order in the output.

    Returns
    -------
    pd.DataFrame
        ``index=datetime, columns=symbol``. dtype is float (NaN-able).
    """
    if not values:
        return pd.DataFrame()
    return pd.DataFrame(dict(values))


def validate_factor_wide(values: pd.DataFrame) -> None:
    """Lightweight wide-layout factor validation.

    Checks:
      - DatetimeIndex.
      - No duplicate timestamps.
      - All columns numeric.
    """
    if not isinstance(values, pd.DataFrame):
        raise TypeError(
            f"factor values must be pd.DataFrame, got "
            f"{type(values).__name__}"
        )
    if not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError(
            f"factor values index must be a DatetimeIndex, got "
            f"{type(values.index).__name__}"
        )
    if values.index.has_duplicates:
        n = values.index.duplicated().sum()
        raise ValueError(
            f"factor values index has {n} duplicate timestamp(s)"
        )
    non_numeric = [
        c for c in values.columns
        if not pd.api.types.is_numeric_dtype(values[c])
    ]
    if non_numeric:
        raise TypeError(
            f"factor values columns must be numeric; non-numeric: "
            f"{non_numeric}"
        )


__all__ = [
    "FactorSpec",
    "FactorSet",
    "FactorProvider",
    "dict_to_factor_wide",
    "validate_factor_wide",
]
