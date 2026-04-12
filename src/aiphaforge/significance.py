"""
Statistical Significance Testing
=================================

Bootstrap confidence intervals and permutation tests for backtest results.

Answers: "Is this strategy's performance real or luck?"

Two foundational methods:

1. **Bootstrap CI**: Confidence intervals for any performance metric via
   stationary block bootstrap (Politis-Romano).
2. **Permutation Test**: p-value for strategy alpha by shuffling signal timing.

Both work with BacktestResult or raw data + signals. Pure computation,
no engine changes.

Example::

    from aiphaforge.significance import bootstrap_ci, permutation_test

    ci = bootstrap_ci(result, metric="sharpe_ratio", confidence=0.95)
    print(f"Sharpe: {ci.observed:.2f} [{ci.ci_lower:.2f}, {ci.ci_upper:.2f}]")

    perm = permutation_test(data, strategy=my_strategy, metric="sharpe_ratio")
    print(f"p-value: {perm.p_value:.4f}")
"""

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
