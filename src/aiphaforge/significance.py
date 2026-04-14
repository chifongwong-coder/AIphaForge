"""
Statistical Significance Testing
=================================

Bootstrap confidence intervals, permutation tests, Monte Carlo path
simulation, and multiple comparison correction for backtest results.

Answers: "Is this strategy's performance real or luck?"

Four tools:

1. **Bootstrap CI**: Confidence intervals for any performance metric via
   stationary block bootstrap (Politis-Romano).
2. **Permutation Test**: p-value for strategy alpha by shuffling signal timing.
3. **Monte Carlo Path Simulation** (v1.6): Generate synthetic market paths,
   run the full strategy/agent on each. Test robustness across different
   possible histories.
4. **Multiple Comparison Correction** (v1.6): Correct optimizer results for
   data snooping when testing many parameter combos.

Example::

    from aiphaforge.significance import bootstrap_ci, permutation_test

    ci = bootstrap_ci(result, metric="sharpe_ratio", confidence=0.95)
    print(f"Sharpe: {ci.observed:.2f} [{ci.ci_lower:.2f}, {ci.ci_upper:.2f}]")

    perm = permutation_test(data, strategy=my_strategy, metric="sharpe_ratio")
    print(f"p-value: {perm.p_value:.4f}")

    from aiphaforge.significance import monte_carlo_test, generate_paths

    mc = monte_carlo_test(data, strategy=my_strategy, metric="sharpe_ratio")
    print(f"MC 5th/95th: [{mc.pct_5:.2f}, {mc.pct_95:.2f}]")
"""

import copy
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import utils
from .results import BacktestResult

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """Result of a bootstrap confidence interval computation.

    Attributes:
        metric_name: Name of the metric (or "custom_N" for callables).
        observed: Actual metric value from the original backtest.
        mean: Mean of the bootstrap distribution.
        std: Standard deviation of the bootstrap distribution.
        ci_lower: Lower bound of the confidence interval.
        ci_upper: Upper bound of the confidence interval.
        confidence: Confidence level used (e.g. 0.95).
        n_bootstrap: Number of bootstrap replications.
        distribution: Full bootstrap distribution array (for plotting).
    """
    metric_name: str
    observed: float
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    confidence: float
    n_bootstrap: int
    distribution: np.ndarray


@dataclass
class PermutationResult:
    """Result of a permutation test for signal significance.

    Attributes:
        metric_name: Name of the metric tested.
        observed: Actual metric value from the real backtest.
        p_value: Fraction of permutations performing as well as or better
            than observed (direction-aware).
        mean_null: Mean of the null distribution.
        std_null: Standard deviation of the null distribution.
        n_permutations: Number of permutations performed.
        n_valid: Number of valid (non-NaN) permutations used for p-value.
        null_distribution: Full null distribution array (for plotting).
    """
    metric_name: str
    observed: float
    p_value: float
    mean_null: float
    std_null: float
    n_permutations: int
    n_valid: int
    null_distribution: np.ndarray


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


# ---------------------------------------------------------------------------
# p-value direction sets
# ---------------------------------------------------------------------------

_HIGHER_IS_BETTER = {
    "sharpe_ratio", "annual_return", "total_return",
    "sortino_ratio", "calmar_ratio",
}

_LOWER_IS_BETTER = {
    "max_drawdown",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stationary_block_bootstrap(
    returns: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One stationary block bootstrap replication (Politis-Romano).

    Block lengths are drawn from a Geometric(1/block_size) distribution.
    Start positions are sampled uniformly. Indices wrap circularly (mod n).

    Parameters:
        returns: Original return series as a numpy array.
        block_size: Expected block length (mean of geometric distribution).
        rng: NumPy random generator instance.

    Returns:
        Resampled return series of the same length as input.
    """
    n = len(returns)
    result = []
    while len(result) < n:
        start = rng.integers(0, n)
        length = rng.geometric(1.0 / block_size)
        for j in range(length):
            if len(result) >= n:
                break
            result.append(returns[(start + j) % n])
    return np.array(result)


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


def _permute_signals(
    signals: pd.Series,
    rng: np.random.Generator,
) -> pd.Series:
    """Shuffle signal timing by permuting the forward-filled position series.

    Steps:
        1. Forward-fill signals to get a position series (every bar has a value).
        2. Identify leading NaN warmup period and preserve it.
        3. Shuffle non-warmup position values (preserves distribution of values).
        4. Convert back to transition-only format (emit only on change).

    This preserves the signal value distribution and warmup period while
    destroying the timing relationship between signals and prices.

    Parameters:
        signals: Transition-only signal series (NaN = hold, values on changes).
        rng: NumPy random generator instance.

    Returns:
        Permuted transition-only signal series.
    """
    filled = signals.ffill()
    warmup_mask = signals.isna() & filled.isna()  # leading NaNs

    # Shuffle non-warmup values
    non_warmup = ~warmup_mask
    values = filled[non_warmup].values.copy()
    rng.shuffle(values)
    filled = filled.copy()
    filled[non_warmup] = values

    # Convert back to transition-only: emit only on change
    changed = filled != filled.shift(1)
    result = pd.Series(np.nan, index=signals.index, dtype=float)
    result[changed & non_warmup] = filled[changed & non_warmup]
    result[warmup_mask] = np.nan
    return result


# ---------------------------------------------------------------------------
# Public API — Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_metrics(
    result: BacktestResult,
    metrics: Sequence[Union[str, Callable]] = ("sharpe_ratio",),
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    block_size: Optional[int] = None,
    random_state: Optional[int] = None,
    trading_days: int = 252,
) -> Dict[str, BootstrapResult]:
    """Compute confidence intervals for multiple metrics via block bootstrap.

    Primary interface. One set of bootstrap samples is generated; all metrics
    are computed from the same resampled paths. This preserves the joint
    distribution between metrics (e.g. Sharpe and drawdown from the same
    path are correlated).

    Parameters:
        result: Completed backtest result (needs equity_curve).
        metrics: List of metric names or callables. Built-in names:
            ``"sharpe_ratio"``, ``"annual_return"``, ``"max_drawdown"``,
            ``"sortino_ratio"``, ``"calmar_ratio"``, ``"total_return"``.
            Or pass a ``callable(pd.Series) -> float`` for custom metrics.
        n_bootstrap: Number of bootstrap replications.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        block_size: Expected block length for stationary bootstrap.
            ``None`` (default) uses ``max(1, int(sqrt(N)))``.
        random_state: Seed for reproducibility.
        trading_days: Number of trading days per year for annualization.

    Returns:
        Dict mapping metric name (or ``"custom_0"``, ``"custom_1"`` for
        callables) to :class:`BootstrapResult`. All results share the
        same bootstrap samples.

    Raises:
        ValueError: If n_bootstrap, confidence, or block_size are invalid,
            or if the equity curve is too short.
    """
    # Input validation
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1 (exclusive)")
    if block_size is not None and block_size < 1:
        raise ValueError("block_size must be >= 1")

    # Extract returns from equity curve
    returns = utils.calculate_returns(result.equity_curve).values

    # Short equity curve check
    if len(returns) < 2:
        raise ValueError(
            f"equity_curve too short ({len(result.equity_curve)} bars, "
            f"{len(returns)} returns). Need at least 3 bars.")

    # Auto block_size
    if block_size is None:
        block_size = max(1, int(np.sqrt(len(returns))))

    # Build metric names and functions
    custom_idx = 0
    metric_names: List[str] = []
    metric_fns: Dict[str, Callable] = {}

    for m in metrics:
        if callable(m) and not isinstance(m, str):
            name = f"custom_{custom_idx}"
            custom_idx += 1
        else:
            name = str(m)
        metric_names.append(name)
        metric_fns[name] = _make_metric_fn(
            m, result.initial_capital, trading_days=trading_days)

    # Pre-allocate numpy buffer; wrap in Series once per iteration via
    # the buffer (avoids pd.Series construction overhead while staying
    # compatible with pandas Copy-on-Write which makes .values read-only).
    _buf = np.empty(len(returns), dtype=float)

    # Bootstrap loop: one set of samples, all metrics per sample
    rng = np.random.default_rng(random_state)
    distributions: Dict[str, List[float]] = {name: [] for name in metric_names}

    for _ in range(n_bootstrap):
        boot_returns = _stationary_block_bootstrap(returns, block_size, rng)
        _buf[:] = boot_returns
        _reusable_series = pd.Series(_buf, copy=False)
        for name in metric_names:
            distributions[name].append(metric_fns[name](_reusable_series))

    # Build BootstrapResult per metric
    alpha = 1 - confidence
    results: Dict[str, BootstrapResult] = {}

    # Compute observed metrics using pd.Series
    observed_series = pd.Series(returns, dtype=float)
    for name in metric_names:
        dist = np.array(distributions[name])
        results[name] = BootstrapResult(
            metric_name=name,
            observed=float(metric_fns[name](observed_series)),
            mean=float(dist.mean()),
            std=float(dist.std()),
            ci_lower=float(np.percentile(dist, 100 * alpha / 2)),
            ci_upper=float(np.percentile(dist, 100 * (1 - alpha / 2))),
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            distribution=dist,
        )

    return results


def bootstrap_ci(
    result: BacktestResult,
    metric: Union[str, Callable] = "sharpe_ratio",
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    block_size: Optional[int] = None,
    random_state: Optional[int] = None,
    trading_days: int = 252,
) -> BootstrapResult:
    """Single-metric convenience wrapper around :func:`bootstrap_metrics`.

    Same parameters as ``bootstrap_metrics``, but accepts a single metric
    and returns one :class:`BootstrapResult` instead of a dict.
    """
    all_results = bootstrap_metrics(
        result,
        metrics=[metric],
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        block_size=block_size,
        random_state=random_state,
        trading_days=trading_days,
    )
    # Return the single result
    return next(iter(all_results.values()))


# ---------------------------------------------------------------------------
# Public API — Permutation test
# ---------------------------------------------------------------------------


def permutation_test(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    strategy=None,
    signals: Optional[Union[pd.Series, Dict[str, pd.Series]]] = None,
    metric: Union[str, Callable] = "sharpe_ratio",
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
    trading_days: int = 252,
    higher_is_better: bool = True,
    zero_cost: bool = False,
    **engine_kwargs,
) -> PermutationResult:
    """Test whether strategy alpha is statistically significant.

    Shuffles the position series (forward-filled signals) to preserve
    the permutation space, then converts back to transition-only format
    for the engine. Reports p-value = fraction of permutations that
    match or exceed actual performance (Phipson & Smyth +1 correction).

    Either ``strategy`` or ``signals`` must be provided (not both).

    Note: For agent strategies (hook-based, path-dependent), use Monte
    Carlo path simulation instead. Permutation test is appropriate for
    signal-based strategies only.

    Note: When using non-zero transaction costs, the null distribution
    is biased downward because random signals trade more frequently.
    Set ``zero_cost=True`` to isolate signal timing alpha from
    trading frequency effects.

    Parameters:
        data: OHLCV data (single or multi-asset dict).
        strategy: Strategy object with ``generate_signals(data)`` method.
        signals: Pre-computed signals (shuffled directly).
        metric: Metric to compare. Same options as :func:`bootstrap_ci`.
        n_permutations: Number of random permutations.
        random_state: Seed for reproducibility.
        trading_days: Number of trading days per year for annualization.
        higher_is_better: Direction for custom callable metrics. If True
            (default), higher metric values are better. Ignored for
            built-in string metrics which have known directions.
        zero_cost: If True, override fee_model with ZeroFeeModel to
            eliminate transaction cost bias in the null distribution.
        **engine_kwargs: Passed to ``BacktestEngine`` constructor
            (fee_model, mode, initial_capital, etc.).

    Returns:
        :class:`PermutationResult` with p-value and null distribution.

    Raises:
        ValueError: If neither or both of strategy/signals are provided,
            or if n_permutations < 1.
    """
    from .engine import BacktestEngine

    # Validate inputs
    if (strategy is None) == (signals is None):
        raise ValueError(
            "Exactly one of 'strategy' or 'signals' must be provided.")

    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")

    # Apply zero_cost override before creating any engines
    if zero_cost:
        from .fees import ZeroFeeModel
        engine_kwargs['fee_model'] = ZeroFeeModel()

    # Generate signals from strategy if needed
    if strategy is not None:
        signals = strategy.generate_signals(data)

    # Determine metric name for result
    if isinstance(metric, str):
        metric_name = metric
    else:
        metric_name = "custom"

    # Ensure engine_kwargs has reasonable defaults for permutation test
    if "include_benchmark" not in engine_kwargs:
        engine_kwargs["include_benchmark"] = False

    # Get initial_capital from engine_kwargs or use default
    initial_capital = engine_kwargs.get("initial_capital", 100000)

    # Build metric function for extraction from BacktestResult
    metric_fn = _make_metric_fn(
        metric, initial_capital, trading_days=trading_days)

    # Run actual backtest to get observed metric
    engine = BacktestEngine(**engine_kwargs)
    engine.set_signals(signals)
    actual_result = engine.run(data)

    observed_returns = utils.calculate_returns(
        actual_result.equity_curve).values
    observed = float(metric_fn(pd.Series(observed_returns, dtype=float)))

    # Permutation loop
    rng = np.random.default_rng(random_state)
    null_dist_values: List[float] = []
    n_failures = 0

    for _ in range(n_permutations):
        # Permute signals
        if isinstance(signals, dict):
            perm_signals = {
                sym: _permute_signals(sig, rng)
                for sym, sig in signals.items()
            }
        else:
            perm_signals = _permute_signals(signals, rng)

        # Run backtest with permuted signals
        perm_engine = BacktestEngine(**engine_kwargs)
        perm_engine.set_signals(perm_signals)
        try:
            perm_result = perm_engine.run(data)
            perm_returns = utils.calculate_returns(
                perm_result.equity_curve).values
            perm_metric = float(metric_fn(
                pd.Series(perm_returns, dtype=float)))
        except Exception:
            # If a permutation fails (e.g. degenerate signals), use NaN
            perm_metric = np.nan
            n_failures += 1
        null_dist_values.append(perm_metric)

    # Warn if any permutations failed
    if n_failures > 0:
        import warnings
        fail_pct = n_failures / n_permutations * 100
        warnings.warn(
            f"permutation_test: {n_failures}/{n_permutations} permutations "
            f"failed ({fail_pct:.0f}%). p-value computed from "
            f"{n_permutations - n_failures} valid samples.")

    null_dist = np.array(null_dist_values)
    # Filter out NaN values for p-value computation
    valid_null = null_dist[~np.isnan(null_dist)]

    # Determine direction for p-value computation
    if isinstance(metric, str):
        lower_better = metric in _LOWER_IS_BETTER
    else:
        lower_better = not higher_is_better

    # Compute p-value with Phipson & Smyth (2010) +1 correction
    if lower_better:
        p_value = float(
            ((valid_null <= observed).sum() + 1) / (len(valid_null) + 1))
    else:
        p_value = float(
            ((valid_null >= observed).sum() + 1) / (len(valid_null) + 1))

    return PermutationResult(
        metric_name=metric_name,
        observed=observed,
        p_value=p_value,
        mean_null=float(np.nanmean(null_dist)),
        std_null=float(np.nanstd(null_dist)),
        n_permutations=n_permutations,
        n_valid=len(valid_null),
        null_distribution=null_dist,
    )


# ---------------------------------------------------------------------------
# v1.6 Internal helpers — Monte Carlo Path Simulation
# ---------------------------------------------------------------------------


def _block_bootstrap_indices(
    n_bars: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one set of block-bootstrapped bar indices.

    Wraps _stationary_block_bootstrap applied to np.arange(n_bars).
    The bootstrap function resamples values in blocks with circular
    wrapping — when applied to sequential integers, it produces a
    block-structured sequence of valid indices.

    Returns integer ndarray of length n_bars.
    """
    # _stationary_block_bootstrap operates on values via indexing:
    # returns[(start + j) % n]. For input [0, 1, 2, ...], the output
    # is those integer values resampled in blocks. No arithmetic is
    # performed on the values, so converting back to int is exact.
    raw = _stationary_block_bootstrap(
        np.arange(n_bars, dtype=float), block_size, rng)
    return raw.astype(int)


def _reconstruct_ohlcv(
    data: pd.DataFrame,
    indices: np.ndarray,
) -> pd.DataFrame:
    """Reconstruct OHLCV from resampled bar indices.

    Algorithm:
        1. Look up the original bars at the bootstrapped indices.
        2. Compute close-to-close returns from the resampled sequence:
           return[i] = original_close[indices[i]] / original_close[indices[i-1]]
           (For i=0, use 1.0 — anchor bar has no return.)
        3. Anchor at data.iloc[0]['close'], apply returns cumulatively
           to build a new close price series.
        4. For each bar, scale open/high/low proportionally:
           ratio = new_close[i] / original_close[indices[i]]
           new_open[i] = original_open[indices[i]] * ratio
           new_high[i] = original_high[indices[i]] * ratio
           new_low[i]  = original_low[indices[i]]  * ratio
           This preserves each bar's OHLC relationships exactly.
        5. Volume: copy directly from the resampled bars.
        6. Attach the ORIGINAL DatetimeIndex (same dates, new prices).
        7. Price guard: if cumulative close drops below epsilon (1e-8),
           clamp to epsilon. This prevents zero/negative prices from
           extreme return sequences.
    """
    orig_close = data["close"].values
    orig_open = data["open"].values
    orig_high = data["high"].values
    orig_low = data["low"].values
    orig_volume = data["volume"].values

    n = len(indices)

    # Step 2: each bar's actual historical return from the original series
    # (avoids spurious returns at block boundaries)
    returns = np.ones(n)
    mask = indices > 0
    returns[mask] = orig_close[indices[mask]] / orig_close[indices[mask] - 1]

    # Step 3: anchor at original first close, cumulative product
    anchor = data.iloc[0]["close"]
    new_close = anchor * np.cumprod(returns)

    # Step 7: epsilon guard — clamp to prevent zero/negative prices
    new_close = np.maximum(new_close, 1e-8)

    # Step 4: scale OHLC proportionally
    ratio = new_close / orig_close[indices]
    new_open = orig_open[indices] * ratio
    new_high = orig_high[indices] * ratio
    new_low = orig_low[indices] * ratio

    # Step 5: volume from resampled bars
    new_volume = orig_volume[indices]

    # Step 6: original DatetimeIndex
    return pd.DataFrame(
        {
            "open": new_open,
            "high": new_high,
            "low": new_low,
            "close": new_close,
            "volume": new_volume,
        },
        index=data.index,
    )


def _normal_paths(
    data: pd.DataFrame,
    n_paths: int,
    rng: np.random.Generator,
) -> List[pd.DataFrame]:
    """Generate paths from fitted normal distribution.

    1. Fit mu, sigma from historical close-to-close returns.
    2. Generate i.i.d. normal returns per path.
    3. Reconstruct close from cumulative returns (with epsilon guard).
    4. Scale OHLC using MEDIAN intra-bar ratios from historical data
       (median is robust to outlier bars with extreme ratios).
    5. Volume: sample with replacement from historical volume.
    """
    close = data["close"].values
    n = len(close)

    # Fit mu/sigma from close returns
    rets = close[1:] / close[:-1]
    mu = np.mean(rets)
    sigma = np.std(rets)

    # Compute median intra-bar ratios
    open_ratio = np.median(data["open"].values / close)
    high_ratio = np.median(data["high"].values / close)
    low_ratio = np.median(data["low"].values / close)
    volume_pool = data["volume"].values

    anchor = data.iloc[0]["close"]
    paths: List[pd.DataFrame] = []

    for _ in range(n_paths):
        # Generate i.i.d. normal returns
        sampled_rets = rng.normal(mu, max(sigma, 1e-10), size=n)
        sampled_rets[0] = 1.0  # anchor bar

        new_close = anchor * np.cumprod(sampled_rets)
        new_close = np.maximum(new_close, 1e-8)

        new_open = new_close * open_ratio
        new_high = new_close * high_ratio
        new_low = new_close * low_ratio

        # Ensure OHLC validity: high >= max(open, close), low <= min(open, close)
        new_high = np.maximum(new_high, np.maximum(new_open, new_close))
        new_low = np.minimum(new_low, np.minimum(new_open, new_close))

        # Volume: sample with replacement
        new_volume = rng.choice(volume_pool, size=n, replace=True)

        paths.append(pd.DataFrame(
            {
                "open": new_open,
                "high": new_high,
                "low": new_low,
                "close": new_close,
                "volume": new_volume.astype(float),
            },
            index=data.index,
        ))

    return paths


def _run_backtest_and_extract(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    strategy,
    signals: Optional[Union[pd.Series, Dict[str, pd.Series]]],
    hooks: Optional[List],
    metric_fn: Callable,
    engine_kwargs: dict,
) -> float:
    """Run a single backtest and extract the metric value.

    Handles strategy/signals/hooks setup, runs engine, extracts
    returns from equity_curve, computes metric via metric_fn.
    """
    from .engine import BacktestEngine

    kwargs = dict(engine_kwargs)
    if hooks is not None:
        kwargs['hooks'] = hooks
    engine = BacktestEngine(**kwargs)
    if strategy is not None:
        engine.set_strategy(strategy)
    elif signals is not None:
        engine.set_signals(signals)
    result = engine.run(data)
    returns = utils.calculate_returns(result.equity_curve)
    return float(metric_fn(pd.Series(returns.values, dtype=float)))


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


# ---------------------------------------------------------------------------
# v1.6 Internal helpers — Multiple Comparison Correction
# ---------------------------------------------------------------------------


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
    """Wrapper around arch.bootstrap.MCS.

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


# ---------------------------------------------------------------------------
# v1.6 Public API — Monte Carlo Path Simulation
# ---------------------------------------------------------------------------


def generate_paths(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    n_paths: int = 1000,
    method: str = "block_bootstrap",
    block_size: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Union[List[pd.DataFrame], List[Dict[str, pd.DataFrame]]]:
    """Generate synthetic OHLCV paths from historical data.

    Parameters:
        data: Historical OHLCV DataFrame or multi-asset dict.
        n_paths: Number of synthetic paths to generate.
        method: Path generation method:
            - "block_bootstrap" (default): stationary block bootstrap
              on full bars. Preserves intra-bar OHLC ratios and
              short-term autocorrelation.
            - "normal": parametric normal (mu, sigma from history).
              Simplest, ignores fat tails and autocorrelation.
        block_size: Expected block length for block_bootstrap.
            None (default) -> auto: max(1, int(sqrt(N))).
        random_state: Seed for reproducibility.

    Returns:
        List of DataFrames (single-asset) or List of dicts (multi-asset).
        Each has same shape, columns, and DatetimeIndex as input.

    Raises:
        ValueError: If n_paths < 1, unknown method, block_size < 1,
            or multi-asset data has mismatched bar counts.
    """
    # Input validation
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    if method not in ("block_bootstrap", "normal"):
        raise ValueError(
            f"Unknown method: {method!r}. "
            f"Must be 'block_bootstrap' or 'normal'.")
    if block_size is not None and block_size < 1:
        raise ValueError("block_size must be >= 1")

    rng = np.random.default_rng(random_state)
    is_multi = isinstance(data, dict)

    if is_multi:
        # Validate all assets have same bar count
        lengths = {sym: len(df) for sym, df in data.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(
                f"Multi-asset data has mismatched bar counts: {lengths}. "
                f"All assets must have the same number of bars.")
        n_bars = next(iter(unique_lengths))
    else:
        n_bars = len(data)

    # Auto block_size
    if block_size is None:
        block_size = max(1, int(np.sqrt(n_bars)))

    if method == "block_bootstrap":
        if is_multi:
            paths: List[Dict[str, pd.DataFrame]] = []
            for _ in range(n_paths):
                # Single shared index sequence for all assets
                indices = _block_bootstrap_indices(n_bars, block_size, rng)
                path_dict = {}
                for sym, df in data.items():
                    path_dict[sym] = _reconstruct_ohlcv(df, indices)
                paths.append(path_dict)
            return paths
        else:
            paths_single: List[pd.DataFrame] = []
            for _ in range(n_paths):
                indices = _block_bootstrap_indices(n_bars, block_size, rng)
                paths_single.append(_reconstruct_ohlcv(data, indices))
            return paths_single

    else:  # method == "normal"
        if is_multi:
            # Generate independently per asset (no cross-correlation)
            per_asset_paths: Dict[str, List[pd.DataFrame]] = {}
            for sym, df in data.items():
                per_asset_paths[sym] = _normal_paths(df, n_paths, rng)

            paths_multi: List[Dict[str, pd.DataFrame]] = []
            for i in range(n_paths):
                path_dict = {
                    sym: per_asset_paths[sym][i]
                    for sym in data
                }
                paths_multi.append(path_dict)
            return paths_multi
        else:
            return _normal_paths(data, n_paths, rng)


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


# ---------------------------------------------------------------------------
# v1.6 Public API — Multiple Comparison Correction
# ---------------------------------------------------------------------------


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
        initial_capital = 100000  # default
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
