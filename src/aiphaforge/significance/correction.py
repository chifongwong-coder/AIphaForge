"""Multiple-comparison correction (Bonferroni / BH / MCS)."""
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from .. import utils
from ..results import BacktestResult
from ._shared import _HIGHER_IS_BETTER, _LOWER_IS_BETTER, _make_metric_fn
from .bootstrap import bootstrap_ci
from .monte_carlo import _build_returns_matrix_from_cache


@dataclass
class CorrectionResult:
    """Result of multiple comparison correction.

    Attributes:
        method: Correction method used ('bonferroni', 'bh', or 'mcs').
        alpha: Significance level used.
        results: Copy of optimize_results DataFrame with added columns:
            'p_value', 'p_value_corrected', 'significant'.
        n_tested: Total number of parameter combos tested.
        n_significant: Number of strategies surviving correction.
        best_significant: Parameters of the top surviving strategy,
            or None if none are significant.
    """
    method: str
    alpha: float
    results: pd.DataFrame
    n_tested: int
    n_significant: int
    best_significant: Optional[Dict]


def _compute_benchmark_metric(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    metric: str,
    trading_days: int,
    initial_capital: float,
) -> float:
    """Compute the benchmark metric value for buy-and-hold comparison.

    For single-asset: buy-and-hold equity from close prices.
    For multi-asset: equal-weight buy-and-hold across all assets.

    Returns the metric value (float) for the benchmark strategy.
    """
    metric_fn = _make_metric_fn(metric, initial_capital, trading_days)

    if isinstance(data, dict):
        # Equal-weight buy-and-hold across all assets
        equity_parts = []
        per_asset_capital = initial_capital / len(data)
        for df in data.values():
            equity_parts.append(
                utils.compute_buy_and_hold(df, per_asset_capital))
        # Sum equity curves (assumes aligned index)
        combined = sum(equity_parts)
        returns = utils.calculate_returns(combined)
    else:
        equity = utils.compute_buy_and_hold(data, initial_capital)
        returns = utils.calculate_returns(equity)

    return float(metric_fn(pd.Series(returns.values, dtype=float)))


def _bonferroni(
    p_values: np.ndarray,
    alpha: float,
) -> tuple:
    """Bonferroni correction for multiple comparisons.

    Returns (corrected_p, significant_mask).
    """
    corrected = np.minimum(p_values * len(p_values), 1.0)
    significant = corrected <= alpha
    return corrected, significant


def _benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float,
) -> tuple:
    """Benjamini-Hochberg FDR control with monotonicity enforcement.

    Returns (corrected_p_in_original_order, significant_mask).
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = alpha * np.arange(1, n + 1) / n

    # Step-up: find largest k where p[k] <= threshold[k]
    passing = sorted_p <= thresholds
    if not passing.any():
        corrected_orig = np.minimum(
            sorted_p * n / np.arange(1, n + 1), 1.0)
        # Monotonicity enforcement
        for i in range(n - 2, -1, -1):
            corrected_orig[i] = min(corrected_orig[i], corrected_orig[i + 1])
        result = np.empty(n)
        result[sorted_idx] = corrected_orig
        return result, np.zeros(n, dtype=bool)

    max_k = np.where(passing)[0][-1]

    significant = np.zeros(n, dtype=bool)
    significant[sorted_idx[:max_k + 1]] = True

    # Corrected p-values with monotonicity enforcement
    corrected = np.minimum(sorted_p * n / np.arange(1, n + 1), 1.0)
    for i in range(n - 2, -1, -1):
        corrected[i] = min(corrected[i], corrected[i + 1])
    corrected_orig = np.empty(n)
    corrected_orig[sorted_idx] = corrected

    return corrected_orig, significant


def _arch_mcs(
    returns_matrix: np.ndarray,
    alpha: float,
    n_bootstrap: int,
    block_size: Optional[int],
    random_state: Optional[int],
) -> tuple:
    """Wrapper around arch.bootstrap.MCS (Hansen-Lunde-Nason 2011).

    Parameters:
        returns_matrix: T x N_strategies array of per-bar returns.
            MCS uses loss convention internally: we negate returns
            so that lower loss = higher return = better.
        alpha: Significance level for the confidence set.
        n_bootstrap: Number of bootstrap replications.
        block_size: Block size for stationary bootstrap.
        random_state: Seed for reproducibility.

    Returns:
        p_values: np.ndarray of per-model p-values, in the same
            positional order as columns in returns_matrix.
        included: list of int column indices in the confidence set.
    """
    try:
        from arch.bootstrap import MCS
    except ImportError:
        raise ImportError(
            "method='mcs' requires the arch package. "
            "Install with: pip install arch")

    # MCS uses loss convention (lower = better)
    losses = pd.DataFrame(-returns_matrix)
    bs = block_size or int(np.sqrt(returns_matrix.shape[0]))
    mcs = MCS(losses, size=alpha,
              block_size=bs,
              reps=n_bootstrap, bootstrap='stationary',
              seed=random_state)
    mcs.compute()
    # mcs.pvalues is a DataFrame with column 'Pvalue', indexed by
    # model names (integers 0, 1, 2, ...). Extract via the column.
    n_models = returns_matrix.shape[1]
    p_values = np.ones(n_models)  # default p=1.0 for safety
    for model_name, p_val in mcs.pvalues['Pvalue'].items():
        if isinstance(model_name, (int, np.integer)) and 0 <= model_name < n_models:
            p_values[int(model_name)] = float(p_val)
    # mcs.included is a plain list of model names (integers)
    included = [
        int(idx) for idx in mcs.included
        if isinstance(idx, (int, np.integer)) and 0 <= idx < n_models
    ]
    return p_values, included


def multiple_comparison_correction(
    optimize_results: pd.DataFrame,
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    metric: str = "sharpe_ratio",
    method: str = "bh",
    alpha: float = 0.05,
    benchmark: str = "zero",
    n_bootstrap: int = 10000,
    block_size: Optional[int] = None,
    random_state: Optional[int] = None,
    trading_days: int = 252,
    results_cache: Optional[Dict[int, BacktestResult]] = None,
) -> CorrectionResult:
    """Correct optimizer results for multiple comparisons (data snooping).

    All methods produce per-strategy p-values and significance flags.

    Parameters:
        optimize_results: DataFrame from optimize().
        data: Original OHLCV data used for optimization.
        metric: Metric to test significance on.
        method: Correction method:
            - "bonferroni": Divide alpha by N. Most conservative.
            - "bh": Benjamini-Hochberg FDR control. Less conservative.
            - "mcs": Model Confidence Set (requires arch package).
        alpha: Significance level (default 0.05).
        benchmark: Benchmark for comparison:
            - "zero": Is each strategy better than doing nothing?
            - "buy_hold": Is each strategy better than buy-and-hold?
        n_bootstrap: Bootstrap iterations.
        block_size: Block size for bootstrap. None -> auto.
        random_state: Seed for reproducibility.
        trading_days: Trading days/year for annualization.
        results_cache: Explicit cache of BacktestResult objects.

    Returns:
        CorrectionResult dataclass.

    Raises:
        ImportError: If method='mcs' and arch is not installed.
        ValueError: If method is unknown, alpha out of range, or
            results_cache cannot be found.
    """
    # Input validation
    if '_combo_idx' not in optimize_results.columns:
        raise ValueError(
            "optimize_results must contain '_combo_idx' column. "
            "Use the DataFrame returned by optimize().")
    if method not in ("bonferroni", "bh", "mcs"):
        raise ValueError(
            f"Unknown method: {method!r}. "
            f"Must be 'bonferroni', 'bh', or 'mcs'.")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if benchmark not in ("zero", "buy_hold"):
        raise ValueError(
            f"Unknown benchmark: {benchmark!r}. "
            f"Must be 'zero' or 'buy_hold'.")

    # Get results_cache
    if results_cache is None:
        results_cache = optimize_results.attrs.get('_results_cache')
    if results_cache is None:
        raise ValueError(
            "results_cache not found. Pass it explicitly or use the "
            "DataFrame returned by optimize() which stores the cache "
            "in attrs['_results_cache'].")

    # Work on a copy of the DataFrame
    df = optimize_results.copy()
    n_tested = len(df)

    if method == "mcs":
        # Note: MCS tests based on raw returns, not the specified metric.
        # The 'significant' column for MCS means "in the Model Confidence Set"
        # (statistically indistinguishable from the best model).
        if metric not in ("sharpe_ratio", "total_return"):
            warnings.warn(
                f"MCS tests based on raw returns, not '{metric}'. "
                f"The 'significant' column indicates membership in the "
                f"Model Confidence Set (indistinguishable from the best).")
        # MCS via arch package
        returns_mat = _build_returns_matrix_from_cache(
            results_cache, df).values
        p_vals, _included = _arch_mcs(
            returns_mat, alpha, n_bootstrap, block_size, random_state)
        df['p_value'] = p_vals
        df['p_value_corrected'] = p_vals  # MCS p-values are already corrected
        df['significant'] = p_vals >= alpha  # In MCS, high p = in confidence set
    else:
        # Bootstrap percentile test for p-values
        if metric in _LOWER_IS_BETTER and benchmark == "zero":
            warnings.warn(
                f"metric='{metric}' with benchmark='zero' will always produce "
                f"p=1.0 (drawdown is always >= 0). Use benchmark='buy_hold' "
                f"to compare against buy-and-hold drawdown instead.")
        # Read initial_capital from the first cached result
        first_idx = int(df.iloc[0]['_combo_idx'])
        initial_capital = results_cache[first_idx].initial_capital
        bm_metric = None
        if benchmark == "buy_hold":
            bm_metric = _compute_benchmark_metric(
                data, metric, trading_days, initial_capital)

        p_values = np.zeros(n_tested)
        for i, (_, row) in enumerate(df.iterrows()):
            combo_idx = int(row['_combo_idx'])
            result = results_cache[combo_idx]

            # Per-strategy seed for independent bootstrap samples
            seed_i = (
                (random_state + combo_idx)
                if random_state is not None else None
            )
            ci = bootstrap_ci(
                result, metric=metric, n_bootstrap=n_bootstrap,
                block_size=block_size, trading_days=trading_days,
                random_state=seed_i)

            if benchmark == "zero":
                if metric in _HIGHER_IS_BETTER:
                    p = (ci.distribution <= 0).sum() / len(ci.distribution)
                else:
                    p = (ci.distribution >= 0).sum() / len(ci.distribution)
            else:  # benchmark == "buy_hold" (validated above)
                if metric in _HIGHER_IS_BETTER:
                    p = ((ci.distribution <= bm_metric).sum()
                         / len(ci.distribution))
                else:
                    p = ((ci.distribution >= bm_metric).sum()
                         / len(ci.distribution))

            p_values[i] = float(p)

        df['p_value'] = p_values

        if method == "bonferroni":
            corrected, significant = _bonferroni(p_values, alpha)
        else:  # bh
            corrected, significant = _benjamini_hochberg(p_values, alpha)

        df['p_value_corrected'] = corrected
        df['significant'] = significant

    n_significant = int(df['significant'].sum())

    # Best significant strategy
    best_significant = None
    if n_significant > 0:
        sig_rows = df[df['significant']]
        sort_col = metric if metric in sig_rows.columns else 'sharpe_ratio'
        ascending = sort_col in _LOWER_IS_BETTER
        best_row = sig_rows.sort_values(sort_col, ascending=ascending).iloc[0]
        # Extract params (exclude internal/metric columns)
        internal_cols = {
            '_combo_idx', 'sharpe_ratio', 'total_return',
            'max_drawdown', 'num_trades', 'final_capital',
            'p_value', 'p_value_corrected', 'significant',
        }
        best_significant = {
            k: v for k, v in best_row.items()
            if k not in internal_cols
        }

    return CorrectionResult(
        method=method,
        alpha=alpha,
        results=df,
        n_tested=n_tested,
        n_significant=n_significant,
        best_significant=best_significant,
    )
