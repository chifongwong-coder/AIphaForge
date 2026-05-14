"""v2.4 Commit H — AlphaScreener MVP.

R11 (master plan v1.0 §7): research-only — MUST NOT import or
reference any execution-layer module (engine, fees, broker,
market_impact, position_sizing, risk, portfolio, optimizer,
portfolio_optimizer, capital_allocator, margin, costs, latency).
Verified by AST guard test in `tests/test_alpha_screener.py`.
"""
from __future__ import annotations

from typing import Mapping, Optional, Union

import numpy as np
import pandas as pd

from aiphaforge.alpha.labels import forward_returns
from aiphaforge.alpha.metrics import coverage, ic, rank_ic
from aiphaforge.alpha.report import AlphaScreenConfig, FactorReport
from aiphaforge.factors import FactorSet, FactorSpec


def _quantile_returns_one_horizon(
    factor: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_quantiles: int,
) -> pd.DataFrame:
    """Per-timestamp per-quantile mean forward return.

    Returns DataFrame[index=timestamp, columns=range(n_quantiles)].
    Columns where the per-timestamp quantile bucket collapsed
    (heavy-tie scores under ``duplicates="drop"``) get NaN — the
    schema stays stable across timestamps via reindex.
    """
    quantile_ids = list(range(n_quantiles))
    out = pd.DataFrame(
        np.nan, index=factor.index, columns=quantile_ids, dtype=float,
    )
    for t in factor.index:
        f_row = factor.loc[t]
        r_row = fwd_ret.loc[t]
        valid = (~f_row.isna()) & (~r_row.isna())
        if int(valid.sum()) < n_quantiles:
            # Not enough points to fill all quantiles meaningfully.
            continue
        f_v = f_row[valid]
        r_v = r_row[valid]
        try:
            binned = pd.qcut(
                f_v, q=n_quantiles, labels=False, duplicates="drop",
            )
        except ValueError:
            # qcut can fail on degenerate input even with duplicates="drop"
            # (all values equal). Skip the row.
            continue
        per_quantile = (
            r_v.groupby(binned).mean().reindex(quantile_ids)
        )
        out.loc[t] = per_quantile.values
    return out


def _compute_one_factor_report(
    factor_name: str,
    factor_values: pd.DataFrame,
    prices: pd.DataFrame,
    spec: Optional[FactorSpec],
    config: AlphaScreenConfig,
) -> FactorReport:
    """Compute a FactorReport for a single factor + price set."""
    sign = 1.0
    if spec is not None and spec.direction is not None:
        sign = float(spec.direction)

    ic_per_h: dict[int, pd.Series] = {}
    rank_ic_per_h: dict[int, pd.Series] = {}
    quantile_per_h: dict[int, pd.DataFrame] = {}
    metrics: dict[str, float] = {}

    cov_series = coverage(factor_values)
    metrics["coverage_mean"] = float(cov_series.mean())

    for h in config.horizons:
        fr = forward_returns(
            prices, horizon=h, entry_lag=config.entry_lag,
        )
        ic_h = ic(factor_values, fr) * sign
        rank_h = rank_ic(factor_values, fr) * sign
        ic_per_h[h] = ic_h
        rank_ic_per_h[h] = rank_h
        quantile_per_h[h] = _quantile_returns_one_horizon(
            factor_values, fr, config.quantiles,
        )

    # Aggregate scalars from the SHORTEST horizon (the standard
    # Alphalens ICIR convention is per-horizon; expose the
    # primary horizon's IC mean / IR as the headline scalars).
    primary_h = config.horizons[0]
    ic_primary = ic_per_h[primary_h].dropna()
    rank_primary = rank_ic_per_h[primary_h].dropna()
    if len(ic_primary):
        metrics["ic_mean"] = float(ic_primary.mean())
        metrics["ic_ir"] = (
            float(ic_primary.mean() / ic_primary.std())
            if ic_primary.std() > 0 else float("nan")
        )
    else:
        metrics["ic_mean"] = float("nan")
        metrics["ic_ir"] = float("nan")
    if len(rank_primary):
        metrics["rank_ic_mean"] = float(rank_primary.mean())
        metrics["rank_ic_ir"] = (
            float(rank_primary.mean() / rank_primary.std())
            if rank_primary.std() > 0 else float("nan")
        )
    else:
        metrics["rank_ic_mean"] = float("nan")
        metrics["rank_ic_ir"] = float("nan")

    return FactorReport(
        factor_name=factor_name,
        metrics=metrics,
        ic=ic_per_h,
        rank_ic=rank_ic_per_h,
        quantile_returns=quantile_per_h,
    )


class AlphaScreener:
    """Per-factor research-layer evaluator.

    MVP scope: IC / RankIC / ICIR / coverage / quantile returns.
    NO transaction costs, NO PnL, NO BacktestEngine integration
    (R11). For trade-cost-aware evaluation, run the user's
    chosen rule through the engine.
    """

    def evaluate(
        self,
        factor_values: Union[pd.DataFrame, FactorSet],
        prices: pd.DataFrame,
        *,
        spec: Optional[FactorSpec] = None,
        config: Optional[AlphaScreenConfig] = None,
    ) -> Union[FactorReport, Mapping[str, FactorReport]]:
        """Evaluate factor(s) against forward returns.

        Input dispatch (master plan v1.0 §6.4 + v2.4 plan H):
          - ``factor_values: pd.DataFrame`` -> single
            :class:`FactorReport` (factor_name from
            ``spec.name`` if supplied, else ``"<factor>"``).
          - ``factor_values: FactorSet`` -> Mapping[str, FactorReport]
            with one report per factor name in the set; per-factor
            spec is read from ``factor_values.specs[name]``.

        ``spec.direction`` (if present): flips the sign of IC for
        reporting. A factor with ``direction=-1`` and raw IC=-0.05
        is reported as IC=+0.05 (the predictive signal is properly
        oriented).

        R9 column alignment: factor and price columns must align
        EXACTLY (order-sensitive equality). Different sets raise;
        same set with different order raises with a "reindex via
        .reindex(columns=prices.columns)" hint.
        """
        cfg = config if config is not None else AlphaScreenConfig()
        if isinstance(factor_values, FactorSet):
            return self._evaluate_factor_set(factor_values, prices, cfg)
        if isinstance(factor_values, pd.DataFrame):
            self._check_columns(factor_values, prices)
            name = (spec.name if spec is not None else "factor")
            return _compute_one_factor_report(
                factor_name=name,
                factor_values=factor_values,
                prices=prices,
                spec=spec,
                config=cfg,
            )
        raise TypeError(
            f"factor_values must be pd.DataFrame or FactorSet, got "
            f"{type(factor_values).__name__}"
        )

    def _evaluate_factor_set(
        self,
        factor_set: FactorSet,
        prices: pd.DataFrame,
        config: AlphaScreenConfig,
    ) -> Mapping[str, FactorReport]:
        out: dict[str, FactorReport] = {}
        for name, values in factor_set.values.items():
            self._check_columns(values, prices)
            spec = factor_set.specs.get(name)
            out[name] = _compute_one_factor_report(
                factor_name=name,
                factor_values=values,
                prices=prices,
                spec=spec,
                config=config,
            )
        return out

    @staticmethod
    def _check_columns(
        factor: pd.DataFrame, prices: pd.DataFrame,
    ) -> None:
        if factor.columns.equals(prices.columns):
            return
        f_set = set(factor.columns)
        p_set = set(prices.columns)
        if f_set == p_set:
            raise ValueError(
                f"factor and price columns share the same set "
                f"({sorted(f_set)}) but the order differs. Reindex "
                f"via `.reindex(columns=prices.columns)` to align "
                f"positions."
            )
        only_f = sorted(f_set - p_set)
        only_p = sorted(p_set - f_set)
        raise ValueError(
            f"factor and price columns do not match. Only in factor: "
            f"{only_f[:5]}; only in prices: {only_p[:5]}"
        )


__all__ = ["AlphaScreener"]
