"""Monte Carlo path simulation and returns-matrix construction."""
import copy
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .. import utils
from ..results import BacktestResult
from ._shared import _make_metric_fn
from .paths import _run_backtest_and_extract, generate_paths


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo path simulation.

    Attributes:
        metric_name: Name of the metric tested.
        observed: Actual metric value on original data.
        mean: Mean across synthetic paths.
        std: Standard deviation across synthetic paths.
        pct_5: 5th percentile of simulation distribution.
        pct_95: 95th percentile of simulation distribution.
        median: Median across synthetic paths.
        n_paths: Total number of synthetic paths generated.
        n_valid: Number of paths that completed successfully.
        distribution: Full array of path metrics (for plotting).
        worst_case: Minimum across valid paths.
        best_case: Maximum across valid paths.
    """
    metric_name: str
    observed: float
    mean: float
    std: float
    pct_5: float
    pct_95: float
    median: float
    n_paths: int
    n_valid: int
    distribution: np.ndarray
    worst_case: float
    best_case: float


def _build_returns_matrix_from_cache(
    results_cache: Dict[int, BacktestResult],
    optimize_results: pd.DataFrame,
) -> pd.DataFrame:
    """Extract T x N returns matrix from results cache.

    Internal helper used by both build_returns_matrix and
    multiple_comparison_correction.
    """
    if '_combo_idx' not in optimize_results.columns:
        raise ValueError(
            "optimize_results must contain '_combo_idx' column. "
            "Use the DataFrame returned by optimize().")
    columns = {}
    for _, row in optimize_results.iterrows():
        combo_idx = int(row['_combo_idx'])
        result = results_cache[combo_idx]
        returns = utils.calculate_returns(result.equity_curve)
        columns[combo_idx] = returns.values

    return pd.DataFrame(columns)


def monte_carlo_test(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    strategy=None,
    signals: Optional[Union[pd.Series, Dict[str, pd.Series]]] = None,
    hooks: Optional[List] = None,
    metric: Union[str, Callable] = "sharpe_ratio",
    n_paths: int = 1000,
    method: str = "block_bootstrap",
    block_size: Optional[int] = None,
    random_state: Optional[int] = None,
    trading_days: int = 252,
    **engine_kwargs,
) -> MonteCarloResult:
    """Run strategy on synthetic market paths, report outcome distribution.

    Three modes of operation:
        1. strategy= : Strategy regenerates signals on each new path.
           Tests: "Is this strategy robust to different market histories?"
        2. signals= : Same signals applied to different price paths.
           Tests: "How sensitive are returns to the specific price path?"
        3. hooks= (with or without strategy): Agent re-executes on each
           path. Tests: "Does this agent perform well on counterfactual
           market histories?"

    IMPORTANT: hooks are deep-copied (copy.deepcopy) for each path to
    ensure agent state is fully reset. Hooks must be picklable.

    Parameters:
        data: Historical OHLCV data (single or multi-asset).
        strategy: Strategy object (signals regenerated per path).
        signals: Pre-computed signals (reused across paths).
        hooks: Hook list. Deep-copied per path for state isolation.
        metric: Metric to report. Same options as bootstrap_ci.
        n_paths: Number of synthetic paths to test.
        method: Path generation method (see generate_paths).
        block_size: Block length (None -> auto sqrt(N)).
        random_state: Seed for reproducibility.
        trading_days: Trading days/year for metric annualization.
        **engine_kwargs: Passed to BacktestEngine (fee_model, mode, etc.).

    Returns:
        MonteCarloResult dataclass.

    Raises:
        ValueError: If hooks and engine_kwargs both contain 'hooks'.
        TypeError: If hooks cannot be deep-copied.
    """
    # 0. Validate hooks vs engine_kwargs conflict
    if hooks is not None and 'hooks' in engine_kwargs:
        raise ValueError(
            "hooks passed both as parameter and in engine_kwargs. "
            "Use one or the other.")

    # 0b. Default mode to event_driven if hooks provided
    if hooks is not None:
        if 'mode' not in engine_kwargs:
            engine_kwargs['mode'] = 'event_driven'
        elif engine_kwargs['mode'] != 'event_driven':
            warnings.warn(
                "hooks are only active in event_driven mode, "
                f"but mode={engine_kwargs['mode']!r} was set.")

    # 1. Build metric function
    initial_capital = engine_kwargs.get('initial_capital', 100000)
    metric_fn = _make_metric_fn(metric, initial_capital, trading_days)

    # Determine metric name
    if isinstance(metric, str):
        metric_name = metric
    else:
        metric_name = "custom"

    # Ensure include_benchmark=False for speed
    if "include_benchmark" not in engine_kwargs:
        engine_kwargs["include_benchmark"] = False

    # 1b. Validate at least one of strategy/signals is provided
    if strategy is None and signals is None:
        raise ValueError(
            "At least one of 'strategy' or 'signals' must be provided. "
            "Hooks alone cannot drive the backtest.")

    # 1c. Save pristine copies BEFORE the observed run mutates state
    if hooks is not None:
        _pristine_hooks = copy.deepcopy(hooks)
    else:
        _pristine_hooks = None
    if strategy is not None:
        _pristine_strategy = copy.deepcopy(strategy)
    else:
        _pristine_strategy = None

    # 1d. Run actual backtest on original data -> observed metric
    observed = _run_backtest_and_extract(
        data, strategy, signals, hooks, metric_fn, engine_kwargs)

    # 2. Generate synthetic paths
    paths = generate_paths(data, n_paths, method, block_size, random_state)

    # 3. Run backtest on each path
    metrics_list: List[float] = []
    n_failures = 0

    for path_data in paths:
        # Deep-copy hooks from pristine (pre-observed-run) state
        if _pristine_hooks is not None:
            try:
                path_hooks = copy.deepcopy(_pristine_hooks)
            except Exception as e:
                raise TypeError(
                    f"Cannot deep-copy hooks for Monte Carlo test: {e}. "
                    f"Hooks must be picklable (no lambdas, file handles, "
                    f"or thread locks as attributes).") from e
            path_kwargs = {**engine_kwargs, 'hooks': path_hooks}
        else:
            path_kwargs = engine_kwargs

        # Deep-copy strategy to reset any state mutated by hooks
        path_strategy = (copy.deepcopy(_pristine_strategy)
                         if _pristine_strategy is not None else None)

        try:
            val = _run_backtest_and_extract(
                path_data, path_strategy, signals,
                path_hooks if hooks is not None else None,
                metric_fn, path_kwargs)
            metrics_list.append(val)
        except Exception:
            metrics_list.append(np.nan)
            n_failures += 1

    # 4. Warn if failures
    if n_failures > 0:
        fail_pct = n_failures / n_paths * 100
        warnings.warn(
            f"monte_carlo_test: {n_failures}/{n_paths} paths "
            f"failed ({fail_pct:.0f}%). Results computed from "
            f"{n_paths - n_failures} valid paths.")

    dist = np.array(metrics_list)
    valid_dist = dist[~np.isnan(dist)]
    n_valid = len(valid_dist)

    if n_valid == 0:
        return MonteCarloResult(
            metric_name=metric_name,
            observed=observed,
            mean=np.nan,
            std=np.nan,
            pct_5=np.nan,
            pct_95=np.nan,
            median=np.nan,
            n_paths=n_paths,
            n_valid=0,
            distribution=dist,
            worst_case=np.nan,
            best_case=np.nan,
        )

    return MonteCarloResult(
        metric_name=metric_name,
        observed=observed,
        mean=float(np.mean(valid_dist)),
        std=float(np.std(valid_dist)),
        pct_5=float(np.percentile(valid_dist, 5)),
        pct_95=float(np.percentile(valid_dist, 95)),
        median=float(np.median(valid_dist)),
        n_paths=n_paths,
        n_valid=n_valid,
        distribution=dist,
        worst_case=float(np.min(valid_dist)),
        best_case=float(np.max(valid_dist)),
    )


def build_returns_matrix(
    optimize_results: pd.DataFrame,
    results_cache: Optional[Dict[int, BacktestResult]] = None,
) -> pd.DataFrame:
    """Extract per-bar returns matrix from optimizer results.

    Returns T x N_strategies DataFrame. Each column is the daily
    return series for one strategy, extracted from its BacktestResult
    equity curve.

    Advanced users can pass this directly to arch.bootstrap.SPA,
    arch.bootstrap.StepM, or other statistical tests.

    Parameters:
        optimize_results: DataFrame from optimize(). Must contain
            '_combo_idx' column.
        results_cache: Explicit cache of BacktestResult objects, keyed
            by combo index. If None, reads from
            optimize_results.attrs['_results_cache'].

    Returns:
        pd.DataFrame with T rows (bar returns) and N columns (strategies).

    Raises:
        ValueError: If results_cache cannot be found.
    """
    if results_cache is None:
        results_cache = optimize_results.attrs.get('_results_cache')
    if results_cache is None:
        raise ValueError(
            "results_cache not found. Pass it explicitly or use the "
            "DataFrame returned by optimize() which stores the cache "
            "in attrs['_results_cache'].")

    return _build_returns_matrix_from_cache(results_cache, optimize_results)
