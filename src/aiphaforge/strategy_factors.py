"""v2.4 Commit J — Extract implicit factors from built-in strategies.

Lets users analyse the factors implicit in legacy v1.x / v2.x
strategies (MA spread, RSI, VWAP distance, momentum rank, pairs
spread) without rewriting the strategy classes themselves.

POINT-IN-TIME SNAPSHOT WARNING (master plan §6.6):
    The adapter recomputes the strategy's internal factors using
    the strategy's CURRENT params. If a MetaController has updated
    ``strategy.short`` (or any other param) mid-backtest via
    ``update_params(...)``, the extracted factor series will use
    the LATEST param value across the entire history — NOT the
    param-as-of-each-bar. For faithful replay of the param
    trajectory, use v2.6's incremental factor logging (when it
    ships).

Composite strategies (StrategyNode subclasses) explicitly return
``FactorSet.empty()`` — the adapter does NOT recurse into
children. v2.5's StrategyNode rewrite is where composite factor
pass-through lives.
"""
from __future__ import annotations

from typing import Mapping, Union

import numpy as np
import pandas as pd

from aiphaforge.factors import FactorSet, FactorSpec
from aiphaforge.indicators import ROC, RSI, SMA, VWAP
from aiphaforge.strategies import (
    MACrossover,
    MomentumRank,
    PairsTrading,
    RSIMeanReversion,
    StrategyNode,
    VWAPReversion,
)

DataLike = Union[pd.DataFrame, Mapping[str, pd.DataFrame]]


def _wide_from_dict(per_symbol: Mapping[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(dict(per_symbol))


def _ensure_dict(data: DataLike) -> dict[str, pd.DataFrame]:
    if isinstance(data, pd.DataFrame):
        return {"_default": data}
    return dict(data)


def _ma_crossover_factors(strategy: MACrossover, data: DataLike) -> FactorSet:
    d = _ensure_dict(data)
    ma_short = _wide_from_dict({
        sym: SMA(df["close"], strategy.short) for sym, df in d.items()
    })
    ma_long = _wide_from_dict({
        sym: SMA(df["close"], strategy.long) for sym, df in d.items()
    })
    ma_spread = ma_short / ma_long - 1.0
    return FactorSet(
        values={
            "ma_short": ma_short,
            "ma_long": ma_long,
            "ma_spread": ma_spread,
        },
        specs={
            "ma_short": FactorSpec(
                name="ma_short", description="short MA",
                family="momentum", lookback=strategy.short,
            ),
            "ma_long": FactorSpec(
                name="ma_long", description="long MA",
                family="momentum", lookback=strategy.long,
            ),
            "ma_spread": FactorSpec(
                name="ma_spread",
                description="ma_short / ma_long - 1",
                family="momentum", direction=1, lookback=strategy.long,
            ),
        },
    )


def _rsi_factors(strategy: RSIMeanReversion, data: DataLike) -> FactorSet:
    d = _ensure_dict(data)
    rsi = _wide_from_dict({
        sym: RSI(df["close"], strategy.period) for sym, df in d.items()
    })
    return FactorSet(
        values={"rsi": rsi},
        specs={
            "rsi": FactorSpec(
                name="rsi", description="Relative Strength Index",
                family="momentum", direction=-1,
                lookback=strategy.period,
            ),
        },
    )


def _vwap_factors(strategy: VWAPReversion, data: DataLike) -> FactorSet:
    d = _ensure_dict(data)
    vwap = _wide_from_dict({
        sym: VWAP(df["high"], df["low"], df["close"], df["volume"])
        for sym, df in d.items()
    })
    close = _wide_from_dict({sym: df["close"] for sym, df in d.items()})
    distance = (close - vwap) / vwap
    return FactorSet(
        values={"vwap": vwap, "vwap_distance": distance},
        specs={
            "vwap": FactorSpec(
                name="vwap", description="cumulative VWAP",
                family="mean_reversion",
            ),
            "vwap_distance": FactorSpec(
                name="vwap_distance",
                description="(close - vwap) / vwap",
                family="mean_reversion", direction=-1,
            ),
        },
    )


def _momentum_rank_factors(
    strategy: MomentumRank, data: DataLike,
) -> FactorSet:
    d = _ensure_dict(data)
    roc = _wide_from_dict({
        sym: ROC(df["close"], strategy.roc_window) for sym, df in d.items()
    })
    return FactorSet(
        values={"roc": roc},
        specs={
            "roc": FactorSpec(
                name="roc",
                description=f"Rate of Change over {strategy.roc_window} bars",
                family="momentum", direction=1,
                lookback=strategy.roc_window,
            ),
        },
    )


def _pairs_factors(strategy: PairsTrading, data: DataLike) -> FactorSet:
    d = _ensure_dict(data)
    if not isinstance(data, dict) or len(d) != 2:
        return FactorSet.empty()
    syms = list(d.keys())
    spread = d[syms[0]]["close"] - d[syms[1]]["close"]
    spread_mean = SMA(spread, strategy.window)
    spread_std = spread.rolling(
        window=strategy.window, min_periods=strategy.window,
    ).std()
    z = (spread - spread_mean) / spread_std.replace(0, np.nan)
    spread_wide = pd.DataFrame({"_pair": spread})
    z_wide = pd.DataFrame({"_pair": z})
    return FactorSet(
        values={"spread": spread_wide, "spread_zscore": z_wide},
        specs={
            "spread": FactorSpec(
                name="spread",
                description=f"raw price spread {syms[0]} - {syms[1]}",
                family="mean_reversion",
            ),
            "spread_zscore": FactorSpec(
                name="spread_zscore",
                description="rolling z-score of spread",
                family="mean_reversion", direction=-1,
                lookback=strategy.window,
            ),
        },
    )


def extract_strategy_factors(strategy, data: DataLike) -> FactorSet:
    """Return the factors implicit in a built-in strategy.

    Supported types (dispatched via ``isinstance`` so user
    subclasses inherit the adapter):
      - MACrossover      -> ma_short, ma_long, ma_spread
      - RSIMeanReversion -> rsi
      - VWAPReversion    -> vwap, vwap_distance
      - MomentumRank     -> roc
      - PairsTrading     -> spread, spread_zscore
        (returns FactorSet.empty() if the data dict doesn't contain
        exactly 2 symbols, since the strategy itself requires that.)

    Composite strategies (StrategyNode subclasses: WeightedBlend,
    SelectBest, PriorityCascade, VoteEnsemble, ConditionalSwitch)
    explicitly return ``FactorSet.empty()`` — the adapter does NOT
    recurse into children. Composing factor-aware strategies
    requires v2.5's StrategyNode rewrite where the composite
    framework supports factor pass-through.

    Unknown / signal-only types return ``FactorSet.empty()`` per
    master plan §0.4 (signal-only strategies are first-class).

    POINT-IN-TIME SNAPSHOT WARNING:
        Recomputes using the strategy's CURRENT params. If
        MetaController has updated params mid-backtest, the
        extracted factor series uses the LATEST param value
        across the entire history — NOT the param-as-of-each-bar.
        For faithful trajectory replay, use v2.6's incremental
        factor logging (when it ships).
    """
    # Composite strategies first — they are also BaseStrategy
    # subclasses so the isinstance chain matters.
    if isinstance(strategy, StrategyNode):
        return FactorSet.empty()
    if isinstance(strategy, MACrossover):
        return _ma_crossover_factors(strategy, data)
    if isinstance(strategy, RSIMeanReversion):
        return _rsi_factors(strategy, data)
    if isinstance(strategy, VWAPReversion):
        return _vwap_factors(strategy, data)
    if isinstance(strategy, MomentumRank):
        return _momentum_rank_factors(strategy, data)
    if isinstance(strategy, PairsTrading):
        return _pairs_factors(strategy, data)
    return FactorSet.empty()


__all__ = ["extract_strategy_factors"]
