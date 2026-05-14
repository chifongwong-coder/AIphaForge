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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Protocol, Union

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

    def to_json(self) -> str:
        """Serialise to JSON via pandas ``orient='split'``.

        v2.4 Commit P — sanctioned cross-version persistence path
        (R10). Pickle is unstable across minor versions because
        FactorSpec / FactorSet are frozen dataclasses; field
        additions break pickled instances. JSON via this helper
        survives v2.x → v3.0.

        Schema (version 1):

        ::

            {
              "version": "1",
              "values": { factor_name: <pandas split dict> },
              "specs":  { factor_name: <FactorSpec field dict> }
            }

        ``orient="split"`` preserves DatetimeIndex dtype + column
        dtype + values dtype across the round-trip.
        """
        import json
        from dataclasses import asdict
        values_payload: dict[str, dict] = {}
        for name, df in self.values.items():
            # Convert the DataFrame to a JSON-friendly dict via
            # split orient. Index is serialised to ISO timestamps
            # by default for DatetimeIndex.
            values_payload[name] = json.loads(df.to_json(orient="split"))
        specs_payload: dict[str, dict] = {}
        for name, spec in self.specs.items():
            d = asdict(spec)
            # tuples → lists in JSON; preserve the field shape.
            d["required_columns"] = list(d.get("required_columns", ()))
            d["tags"] = list(d.get("tags", ()))
            specs_payload[name] = d
        return json.dumps({
            "version": "1",
            "values": values_payload,
            "specs": specs_payload,
        })

    @classmethod
    def from_json(cls, payload: str) -> "FactorSet":
        """Deserialise from :meth:`to_json` output.

        Reconstructs frozen ``FactorSpec`` instances via keyword
        expansion. Tuples (``required_columns``, ``tags``) are
        re-wrapped from JSON lists.
        """
        import json
        from io import StringIO
        obj = json.loads(payload)
        version = obj.get("version", "1")
        if version != "1":
            raise ValueError(
                f"unsupported FactorSet JSON schema version: "
                f"{version!r}. Only version '1' is supported in v2.4."
            )
        values: dict[str, pd.DataFrame] = {}
        for name, split_dict in obj.get("values", {}).items():
            df = pd.read_json(
                StringIO(json.dumps(split_dict)), orient="split",
            )
            df.index = pd.to_datetime(df.index)
            # Factor canonical layout is float64; cast to preserve
            # dtype across the round-trip (read_json infers int64
            # when all values are whole numbers).
            df = df.astype(float)
            values[name] = df
        specs: dict[str, FactorSpec] = {}
        for name, d in obj.get("specs", {}).items():
            d = dict(d)
            d["required_columns"] = tuple(d.get("required_columns", ()))
            d["tags"] = tuple(d.get("tags", ()))
            specs[name] = FactorSpec(**d)
        return cls(values=values, specs=specs)


class BaseFactor(ABC):
    """Abstract base class for compute-side factors (v2.4).

    Subclasses implement :meth:`compute` returning a wide
    ``pd.DataFrame[index=datetime, columns=symbol]``. Both
    single-asset and multi-asset inputs are accepted; the column
    layout is uniform per master plan §2.4.

    Single-vs-multi dispatch (pinned in v2.4 Commit D):
      - If ``data`` is ``pd.DataFrame``: single-asset; output is a
        one-column DataFrame whose column name is ``self.name``
        (the factor's own parametrised identity).
      - If ``data`` is ``Mapping[str, pd.DataFrame]``: multi-asset;
        output columns mirror the input keys preserving the
        mapping's iteration order.

    Subclasses MUST set ``self.name`` (typically encoding the
    parameter values, e.g. ``"rsi_14"``, ``"momentum_20"``) and
    SHOULD set ``self.spec`` to a :class:`FactorSpec` describing
    the factor's metadata.
    """

    name: str = "base_factor"
    spec: "FactorSpec | None" = None

    @abstractmethod
    def compute(
        self,
        data: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],
    ) -> pd.DataFrame:
        """Return canonical wide ``DataFrame[index=datetime, columns=symbol]``."""


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
    "BaseFactor",
    "dict_to_factor_wide",
    "validate_factor_wide",
]
