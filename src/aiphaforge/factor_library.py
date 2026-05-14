"""v2.4 Commit D — Reference factor library.

Five concrete :class:`BaseFactor` subclasses for v2.4's factor
research and FactorRuleStrategy MVP. Each factor:

- uses rolling windows with ``min_periods=window`` (NaN warmup,
  NEVER ``.expanding()`` / full-sample stats — would be lookahead),
- carries a parametrised ``name`` so multi-instance research
  workflows can hold multiple instances in a single FactorSet
  without column-name collisions,
- produces wide ``DataFrame[index=datetime, columns=symbol]``
  output (one column for single-asset, per-symbol columns for
  multi-asset).

Engine-agnostic: this module does NOT import ``aiphaforge.engine``
(verified by AST guard test).
"""
from __future__ import annotations

from typing import Mapping, Union

import pandas as pd

from aiphaforge.factors import BaseFactor, FactorSpec
from aiphaforge.indicators import EMA, RSI, SMA

DataLike = Union[pd.DataFrame, Mapping[str, pd.DataFrame]]


def _per_symbol(
    data: DataLike,
    fn,
    factor_name: str,
) -> pd.DataFrame:
    """Apply ``fn(df) -> pd.Series`` per symbol; stack into wide DataFrame.

    Single-asset input produces a one-column DataFrame named
    ``factor_name``. Multi-asset input mirrors the input dict keys.
    """
    if isinstance(data, pd.DataFrame):
        return fn(data).to_frame(name=factor_name)
    if isinstance(data, Mapping):
        cols = {sym: fn(df) for sym, df in data.items()}
        return pd.DataFrame(cols)
    raise TypeError(
        f"data must be pd.DataFrame or Mapping[str, pd.DataFrame], "
        f"got {type(data).__name__}"
    )


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


class RSIFactor(BaseFactor):
    """Relative Strength Index over a rolling window.

    Wraps :func:`aiphaforge.indicators.RSI`. Output range [0, 100].
    Warmup: first ``period - 1`` bars NaN.

    Parametrised name format: ``f"rsi_{period}"``.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.name = f"rsi_{period}"
        self.spec = FactorSpec(
            name=self.name,
            description="Relative Strength Index",
            family="momentum",
            direction=-1,  # high RSI → mean-reversion short
            required_columns=("close",),
            lookback=period,
            tags=("rsi", "oscillator"),
        )

    def compute(self, data: DataLike) -> pd.DataFrame:
        return _per_symbol(
            data,
            lambda df: RSI(df["close"], self.period),
            self.name,
        )


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


class MomentumFactor(BaseFactor):
    """Gross simple return over the past ``window`` bars.

    Formula: ``close[t] / close[t - window] - 1``. Dimensionless ratio
    (NOT ROC's 100x-scaled version). Warmup: first ``window`` bars NaN.

    Parametrised name format: ``f"momentum_{window}"``.
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.name = f"momentum_{window}"
        self.spec = FactorSpec(
            name=self.name,
            description="Gross simple return over window bars",
            family="momentum",
            direction=1,
            required_columns=("close",),
            lookback=window,
            tags=("momentum",),
        )

    def compute(self, data: DataLike) -> pd.DataFrame:
        return _per_symbol(
            data,
            lambda df: df["close"] / df["close"].shift(self.window) - 1.0,
            self.name,
        )


# ---------------------------------------------------------------------------
# MA spread
# ---------------------------------------------------------------------------


class MASpreadFactor(BaseFactor):
    """Relative spread between short and long moving averages.

    Formula: ``MA_short(close) / MA_long(close) - 1``. Dimensionless
    ratio. Warmup: first ``long - 1`` bars NaN.

    Parametrised name format: ``f"ma_spread_{short}_{long}_{ma_type}"``.
    """

    def __init__(
        self,
        short: int = 5,
        long: int = 20,
        ma_type: str = "sma",
    ):
        self.short = short
        self.long = long
        self.ma_type = ma_type
        self.name = f"ma_spread_{short}_{long}_{ma_type}"
        self.spec = FactorSpec(
            name=self.name,
            description="Short/long moving-average relative spread",
            family="momentum",
            direction=1,
            required_columns=("close",),
            lookback=long,
            tags=("ma_spread",),
        )

    def compute(self, data: DataLike) -> pd.DataFrame:
        ma_fn = EMA if self.ma_type == "ema" else SMA

        def per(df):
            return ma_fn(df["close"], self.short) / ma_fn(df["close"], self.long) - 1.0

        return _per_symbol(data, per, self.name)


# ---------------------------------------------------------------------------
# VWAP distance (rolling, NOT running cumulative)
# ---------------------------------------------------------------------------


class VWAPDistanceFactor(BaseFactor):
    """Distance from rolling-window VWAP, dimensionless ratio.

    Uses **rolling** VWAP (NOT the running-cumulative
    ``indicators.VWAP``):

    ::

        typical[t] = (high[t] + low[t] + close[t]) / 3
        vwap_t = sum(typical[t-w+1:t+1] * volume[t-w+1:t+1])
                  / sum(volume[t-w+1:t+1])
        distance[t] = (close[t] - vwap_t) / vwap_t

    Zero-volume window edge case: ``0 / 0 → NaN`` naturally via
    pandas/numpy division — no special handling needed.
    Warmup: first ``window - 1`` bars NaN.

    Parametrised name format: ``f"vwap_distance_{window}"``.
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.name = f"vwap_distance_{window}"
        self.spec = FactorSpec(
            name=self.name,
            description="Distance from rolling-window VWAP (ratio)",
            family="mean_reversion",
            direction=-1,
            required_columns=("high", "low", "close", "volume"),
            lookback=window,
            tags=("vwap", "mean_reversion"),
        )

    def compute(self, data: DataLike) -> pd.DataFrame:
        def per(df):
            typical = (df["high"] + df["low"] + df["close"]) / 3.0
            sum_pv = (typical * df["volume"]).rolling(self.window).sum()
            sum_v = df["volume"].rolling(self.window).sum()
            vwap = sum_pv / sum_v   # 0/0 → NaN naturally
            return (df["close"] - vwap) / vwap

        return _per_symbol(data, per, self.name)


# ---------------------------------------------------------------------------
# Volume z-score
# ---------------------------------------------------------------------------


class VolumeZScoreFactor(BaseFactor):
    """Rolling z-score of volume.

    Formula:

    ::

        vz[t] = (volume[t] - rolling_mean(volume, window)[t])
                  / rolling_std(volume, window, ddof=1)[t]

    Uses ``min_periods=window`` (NaN warmup). Explicitly NOT
    ``.expanding()`` mean/std — that would be lookahead via
    full-sample stats. ``ddof=1`` matches the rolling-default
    convention in the codebase (``_vol.py:142``).

    Parametrised name format: ``f"volume_zscore_{window}"``.
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.name = f"volume_zscore_{window}"
        self.spec = FactorSpec(
            name=self.name,
            description="Rolling z-score of volume",
            family="volume",
            direction=None,
            required_columns=("volume",),
            lookback=window,
            tags=("volume", "zscore"),
        )

    def compute(self, data: DataLike) -> pd.DataFrame:
        def per(df):
            v = df["volume"].astype(float)
            mean = v.rolling(self.window, min_periods=self.window).mean()
            std = v.rolling(self.window, min_periods=self.window).std(ddof=1)
            return (v - mean) / std

        return _per_symbol(data, per, self.name)


__all__ = [
    "RSIFactor",
    "MomentumFactor",
    "MASpreadFactor",
    "VWAPDistanceFactor",
    "VolumeZScoreFactor",
]
