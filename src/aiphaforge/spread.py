"""
Bid-Ask Spread Models
=====================

Microstructure spread cost for event-driven fills: the bar fill price
(open / close / VWAP per ``FillModel``) is treated as the mid, and a
taker pays half the quoted spread per side — buys fill at
``price + half_spread``, sells at ``price - half_spread``. Orthogonal
to ``FillModel``, slippage, and market impact.

Modeling choices (documented, conservative):

- Passive limit fills pay (up to) the half-spread rather than earning
  it; combined with the broker's limit-price clamp, the effective rule
  is "limit fills pay ``min(half, headroom-to-limit)``".
- The decomposition applies impact before spread; convention measures
  temporary impact from the mid after crossing the spread, but the two
  orders differ only by the second-order impact x half-spread cross
  term — immaterial at modeled magnitudes.
- The default ``slippage_pct`` of most fee models already crudely
  proxies spread-crossing. Stacking a spread model on an unchanged
  ``slippage_pct`` double-counts that friction economically; reduce
  one when adding the other.

Vectorized mode folds :class:`FixedSpread` into the linear cost
approximation; other spread models are ignored there (with a per-run
warning) — use ``mode="event_driven"`` for dynamic spreads.
"""

from abc import ABC, abstractmethod
from typing import Optional

# v2.8.6: public surface lock.
__all__ = [
    "BaseSpreadModel",
    "FixedSpread",
    "VolatilitySpread",
]


class BaseSpreadModel(ABC):
    """Base class for spread models.

    Attributes:
        requires_volatility: When True, the event loop activates the
            rolling-volatility pipeline (shared with the market impact
            infrastructure) and feeds ``half_spread`` a per-bar
            volatility; when the pipeline is inactive or warming up,
            ``volatility`` is None.
    """

    requires_volatility: bool = False

    @abstractmethod
    def half_spread(
        self, price: float, volatility: Optional[float] = None,
    ) -> float:
        """Absolute half-spread in price units.

        Buys fill at ``price + half_spread``, sells at
        ``price - half_spread``.

        Parameters:
            price: Pre-spread fill price (treated as the mid).
            volatility: Rolling PER-BAR volatility fraction at the
                input bar frequency, unannualized (Parkinson estimate
                over ``impact_vol_lookback`` bars). None when the vol
                pipeline is inactive or warming up.
        """


class FixedSpread(BaseSpreadModel):
    """Constant quoted spread.

    Parameters:
        spread_bps: FULL bid-ask spread in basis points; a taker pays
            half per side (``half = price * spread_bps / 1e4 / 2``).
    """

    requires_volatility = False

    def __init__(self, spread_bps: float):
        if spread_bps < 0:
            raise ValueError(
                f"spread_bps must be >= 0, got {spread_bps}")
        self.spread_bps = float(spread_bps)

    def half_spread(
        self, price: float, volatility: Optional[float] = None,
    ) -> float:
        return price * self.spread_bps / 1e4 / 2.0

    def __repr__(self) -> str:
        return f"FixedSpread(spread_bps={self.spread_bps})"


class VolatilitySpread(BaseSpreadModel):
    """Volatility-proportional spread.

    Full spread fraction = ``clip(k * volatility, min_bps/1e4,
    max_bps/1e4)``; half-spread = ``price * full_frac / 2``. The clamp
    band applies to the FULL spread, matching :class:`FixedSpread`'s
    parameterization.

    ``volatility`` is the rolling PER-BAR Parkinson volatility at the
    input bar frequency, UNANNUALIZED — so ``k`` is bar-frequency
    dependent. Calibration examples: daily bars with vol around
    0.01-0.02 and ``k=0.1`` give a 10-20 bps full spread; 15-minute
    bars with vol around 0.002-0.004 and the same ``k`` give 2-4 bps.

    Recommend ``min_bps > 0`` (1-2 bps): the volatility pipeline's
    warmup window feeds ``volatility=None``, which falls back to the
    ``min_bps`` floor — with ``min_bps=0`` those bars are cost-free.

    Parameters:
        k: Multiplier mapping per-bar volatility to full spread
            fraction.
        min_bps: Full-spread floor in basis points (also the
            ``volatility=None`` fallback).
        max_bps: Optional full-spread cap in basis points.
    """

    requires_volatility = True

    def __init__(
        self,
        k: float,
        min_bps: float = 0.0,
        max_bps: Optional[float] = None,
    ):
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        if min_bps < 0:
            raise ValueError(f"min_bps must be >= 0, got {min_bps}")
        if max_bps is not None and max_bps < min_bps:
            raise ValueError(
                f"max_bps ({max_bps}) must be >= min_bps ({min_bps})")
        self.k = float(k)
        self.min_bps = float(min_bps)
        self.max_bps = None if max_bps is None else float(max_bps)

    def half_spread(
        self, price: float, volatility: Optional[float] = None,
    ) -> float:
        if volatility is None:
            full_frac = self.min_bps / 1e4
        else:
            full_frac = max(self.k * volatility, self.min_bps / 1e4)
            if self.max_bps is not None:
                full_frac = min(full_frac, self.max_bps / 1e4)
        return price * full_frac / 2.0

    def __repr__(self) -> str:
        return (f"VolatilitySpread(k={self.k}, min_bps={self.min_bps}, "
                f"max_bps={self.max_bps})")
