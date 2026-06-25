"""Monte Carlo synthetic-path generation."""
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .. import utils
from ..stats import _block_bootstrap_indices, _reconstruct_ohlcv


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
    from ..engine import BacktestEngine

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
