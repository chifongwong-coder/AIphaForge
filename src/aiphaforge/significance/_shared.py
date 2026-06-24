"""Shared significance helpers: metric-fn factory and p-value direction sets."""
from typing import Callable, Union

import numpy as np
import pandas as pd

from .. import utils

_HIGHER_IS_BETTER = {
    "sharpe_ratio", "annual_return", "total_return",
    "sortino_ratio", "calmar_ratio",
}


_LOWER_IS_BETTER = {
    "max_drawdown",
}


def _make_metric_fn(
    metric: Union[str, Callable],
    initial_capital: float,
    trading_days: int = 252,
) -> Callable[[pd.Series], float]:
    """Convert a metric name or callable into a function(returns) -> float.

    Built-in metrics delegate to functions in utils.py and accept pd.Series.
    Equity-based metrics (max_drawdown, calmar_ratio) reconstruct equity
    from bootstrapped returns: ``initial_capital * cumprod(1 + r)``.

    Custom callables receive pd.Series as input.

    Parameters:
        metric: Metric name string or a callable(pd.Series) -> float.
        initial_capital: Initial capital for equity reconstruction.
        trading_days: Number of trading days per year for annualization.

    Returns:
        A callable that takes a pd.Series of returns and returns a float.

    Raises:
        ValueError: If metric is an unknown string.
    """
    if callable(metric):
        return metric

    if metric == "sharpe_ratio":
        return lambda r: utils.sharpe_ratio(r, trading_days=trading_days)
    if metric == "sortino_ratio":
        return lambda r: utils.sortino_ratio(r, trading_days=trading_days)
    if metric == "max_drawdown":
        return lambda r: utils.max_drawdown(
            pd.Series(initial_capital * np.cumprod(1 + r.values)))
    if metric == "annual_return":
        def _annual_return(r: pd.Series) -> float:
            rv = r.values
            return float(
                (1 + rv).prod() ** (trading_days / max(len(rv), 1)) - 1)
        return _annual_return
    if metric == "calmar_ratio":
        def _calmar(r: pd.Series) -> float:
            rv = r.values
            ann = float(
                (1 + rv).prod() ** (trading_days / max(len(rv), 1)) - 1)
            mdd = utils.max_drawdown(
                pd.Series(initial_capital * np.cumprod(1 + rv)))
            if mdd == 0:
                return float('inf') if ann > 0 else 0.0
            return ann / mdd
        return _calmar
    if metric == "total_return":
        return lambda r: float((1 + r.values).prod() - 1)

    raise ValueError(f"Unknown metric: {metric!r}")
