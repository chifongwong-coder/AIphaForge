"""
Parameter Optimizer
===================

Grid search and walk-forward validation for strategy parameter tuning.
Standalone module — uses only the public BacktestEngine API. Zero changes
to the engine core.

Usage::

    from aiphaforge import optimize, walk_forward

    results = optimize(
        data, signals=my_signals,
        param_grid={'stop_loss': [0.03, 0.05], 'position_size': [0.5, 0.95]},
        metric='sharpe_ratio',
    )
"""

import inspect
import itertools
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from .engine import BacktestEngine
from .results import BacktestResult


def optimize(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    param_grid: Dict[str, List],
    *,
    signals=None,
    strategy=None,
    strategy_factory: Optional[Callable[[Dict], Any]] = None,
    metric: str = "sharpe_ratio",
    mode: str = "vectorized",
    **engine_kwargs,
) -> pd.DataFrame:
    """Grid search over engine and strategy parameters.

    Parameters:
        data: OHLCV data (single or multi-asset).
        param_grid: Dict of param_name to list of values.
            Engine params (e.g., 'stop_loss') go to BacktestEngine constructor.
            Strategy params go to ``strategy_factory`` if provided.
        signals: Pre-computed signals (mutually exclusive with strategy).
        strategy: Fixed strategy object (no param sweep on strategy).
        strategy_factory: Callable that receives a dict of strategy-level
            params and returns a strategy object. Allows sweeping strategy
            parameters alongside engine parameters.
        metric: Metric to sort results by (descending). Default 'sharpe_ratio'.
        mode: Execution mode for all runs.
        **engine_kwargs: Fixed engine parameters for all runs.

    Returns:
        pd.DataFrame: One row per param combination, sorted by metric desc.
    """
    if not param_grid:
        raise ValueError("param_grid must not be empty")

    if signals is None and strategy is None and strategy_factory is None:
        raise ValueError(
            "Must provide signals, strategy, or strategy_factory")

    # Identify which params are engine-level vs strategy-level
    engine_param_names = set(inspect.signature(BacktestEngine.__init__).parameters)

    # Warn if strategy params in grid but no factory to consume them
    non_engine = [k for k in param_grid if k not in engine_param_names]
    if non_engine and strategy_factory is None:
        import warnings
        warnings.warn(
            f"param_grid contains non-engine params {non_engine} but "
            f"strategy_factory is not set. These params will be ignored. "
            f"Use strategy_factory to sweep strategy-level parameters.")

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combos = list(itertools.product(*param_values))

    rows = []
    results_cache: Dict[int, BacktestResult] = {}

    for i, combo in enumerate(combos):
        params = dict(zip(param_names, combo))

        # Split into engine params and strategy params
        eng_params = {}
        strat_params = {}
        for k, v in params.items():
            if k in engine_param_names:
                eng_params[k] = v
            else:
                strat_params[k] = v

        # Build engine
        merged = {**engine_kwargs, **eng_params, 'mode': mode}
        engine = BacktestEngine(**merged)

        # Set signals or strategy
        if strategy_factory is not None:
            strat = strategy_factory(strat_params)
            if strat is None:
                raise ValueError(
                    f"strategy_factory returned None for params {strat_params}")
            engine.set_strategy(strat)
        elif signals is not None:
            engine.set_signals(signals)
        elif strategy is not None:
            engine.set_strategy(strategy)

        result = engine.run(data)
        results_cache[i] = result

        row = dict(params)
        row['_combo_idx'] = i  # for cache lookup
        row['sharpe_ratio'] = result.sharpe_ratio
        row['total_return'] = result.total_return
        row['max_drawdown'] = result.max_drawdown
        row['num_trades'] = result.num_trades
        row['final_capital'] = result.final_capital
        rows.append(row)

    df = pd.DataFrame(rows)
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False).reset_index(drop=True)
    df.attrs['_results_cache'] = results_cache
    return df


def walk_forward(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    param_grid: Dict[str, List],
    *,
    signals=None,
    strategy=None,
    strategy_factory: Optional[Callable[[Dict], Any]] = None,
    train_pct: float = 0.7,
    metric: str = "sharpe_ratio",
    mode: str = "vectorized",
    **engine_kwargs,
) -> Dict[str, Any]:
    """Walk-forward optimization: optimize on train, validate on test.

    Parameters:
        data: OHLCV data.
        param_grid: Same as optimize().
        signals: Pre-computed signals.
        strategy: Fixed strategy object.
        strategy_factory: Strategy factory for param sweeps.
        train_pct: Fraction of data for training (0, 1).
        metric: Metric to optimize on train set.
        mode: Execution mode.
        **engine_kwargs: Fixed engine parameters.

    Returns:
        Dict with keys: 'best_params', 'train_result', 'test_result',
        'train_metrics'.
    """
    if not 0 < train_pct < 1:
        raise ValueError(f"train_pct must be in (0, 1), got {train_pct}")

    # Date-based split (not positional) for multi-asset safety
    if isinstance(data, dict):
        first_key = next(iter(data))
        ref_index = data[first_key].index
    else:
        ref_index = data.index

    split_pos = int(len(ref_index) * train_pct)
    if split_pos == 0 or split_pos >= len(ref_index):
        raise ValueError(
            f"train_pct={train_pct} produces empty train or test set "
            f"(data has {len(ref_index)} rows)")
    split_date = ref_index[split_pos]

    # Split data by date.
    # .loc[:split_date] is inclusive, so iloc[:-1] trims the overlap row
    # to ensure train and test are non-overlapping.
    if isinstance(data, dict):
        train_data = {k: df.loc[:split_date].iloc[:-1]
                      for k, df in data.items()}
        test_data = {k: df.loc[split_date:] for k, df in data.items()}
    else:
        train_data = data.loc[:split_date].iloc[:-1]
        test_data = data.loc[split_date:]

    # Split signals by same date
    train_signals, test_signals = None, None
    if signals is not None:
        if isinstance(signals, dict):
            train_signals = {k: s.loc[:split_date].iloc[:-1]
                            for k, s in signals.items()}
            test_signals = {k: s.loc[split_date:]
                           for k, s in signals.items()}
        else:
            train_signals = signals.loc[:split_date].iloc[:-1]
            test_signals = signals.loc[split_date:]

    # Optimize on train
    train_df = optimize(
        train_data, param_grid,
        signals=train_signals, strategy=strategy,
        strategy_factory=strategy_factory,
        metric=metric, mode=mode, **engine_kwargs)

    # Best params (top row after sorting)
    param_names = list(param_grid.keys())
    best_params = {k: train_df.iloc[0][k] for k in param_names}

    # Get cached train result (avoid re-run)
    cache = train_df.attrs.get('_results_cache', {})
    best_combo_idx = int(train_df.iloc[0].get('_combo_idx', -1))
    train_result = cache.get(best_combo_idx)

    if train_result is None:
        # Fallback: re-run (shouldn't happen with cache)
        engine_param_names = set(inspect.signature(BacktestEngine.__init__).parameters)
        eng_p = {k: v for k, v in best_params.items()
                 if k in engine_param_names}
        strat_p = {k: v for k, v in best_params.items()
                   if k not in engine_param_names}
        merged = {**engine_kwargs, **eng_p, 'mode': mode}
        e = BacktestEngine(**merged)
        if strategy_factory:
            e.set_strategy(strategy_factory(strat_p))
        elif train_signals is not None:
            e.set_signals(train_signals)
        elif strategy:
            e.set_strategy(strategy)
        train_result = e.run(train_data)

    # Validate on test
    engine_param_names = set(inspect.signature(BacktestEngine.__init__).parameters)
    eng_p = {k: v for k, v in best_params.items()
             if k in engine_param_names}
    strat_p = {k: v for k, v in best_params.items()
               if k not in engine_param_names}
    merged = {**engine_kwargs, **eng_p, 'mode': mode}
    engine_test = BacktestEngine(**merged)
    if strategy_factory:
        engine_test.set_strategy(strategy_factory(strat_p))
    elif test_signals is not None:
        engine_test.set_signals(test_signals)
    elif strategy:
        engine_test.set_strategy(strategy)
    test_result = engine_test.run(test_data)

    return {
        'best_params': best_params,
        'train_result': train_result,
        'test_result': test_result,
        'train_metrics': train_df,
    }
