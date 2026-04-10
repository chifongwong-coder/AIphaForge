"""
End-to-end tests for v0.9.2: parameter sweep (optimize + walk_forward).
"""

import numpy as np
import pandas as pd
import pytest

from aiphaforge import BacktestResult, optimize, walk_forward
from aiphaforge.fees import ZeroFeeModel

from .conftest import make_ohlcv


class TestOptimize:
    """Grid search over engine parameters."""

    def test_grid_search_returns_sorted_dataframe(self):
        """optimize() returns DataFrame sorted by metric."""
        data = make_ohlcv(50)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1
        signals.iloc[40] = 0

        results = optimize(
            data, signals=signals,
            param_grid={
                'stop_loss': [0.03, 0.05, 0.10],
                'position_size': [0.5, 0.95],
            },
            metric='sharpe_ratio',
            fee_model=ZeroFeeModel(),
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 6  # 3 × 2 combinations
        assert 'stop_loss' in results.columns
        assert 'position_size' in results.columns
        assert 'sharpe_ratio' in results.columns
        # Sorted descending by sharpe
        if len(results) > 1:
            assert results['sharpe_ratio'].iloc[0] >= results['sharpe_ratio'].iloc[1]

    def test_empty_param_grid_raises(self):
        """Empty param_grid → ValueError."""
        data = make_ohlcv(10)
        signals = pd.Series(np.nan, index=data.index, dtype=float)

        with pytest.raises(ValueError, match="param_grid"):
            optimize(data, signals=signals, param_grid={})

    def test_no_signals_no_strategy_raises(self):
        """No signals or strategy → ValueError."""
        data = make_ohlcv(10)

        with pytest.raises(ValueError, match="Must provide"):
            optimize(data, param_grid={'stop_loss': [0.05]})

    def test_single_param_single_value(self):
        """Single param with one value → one-row result."""
        data = make_ohlcv(30)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[20] = 0

        results = optimize(
            data, signals=signals,
            param_grid={'stop_loss': [0.05]},
            fee_model=ZeroFeeModel(),
        )
        assert len(results) == 1


class TestWalkForward:
    """Walk-forward: optimize on train, validate on test."""

    def test_walk_forward_returns_structure(self):
        """walk_forward() returns dict with expected keys."""
        data = make_ohlcv(100)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[5] = 1
        signals.iloc[50] = 0
        signals.iloc[75] = 1
        signals.iloc[95] = 0

        wf = walk_forward(
            data, signals=signals,
            param_grid={'stop_loss': [0.03, 0.05, 0.10]},
            train_pct=0.7,
            metric='sharpe_ratio',
            fee_model=ZeroFeeModel(),
        )

        assert 'best_params' in wf
        assert 'train_result' in wf
        assert 'test_result' in wf
        assert 'train_metrics' in wf
        assert isinstance(wf['train_result'], BacktestResult)
        assert isinstance(wf['test_result'], BacktestResult)
        assert isinstance(wf['train_metrics'], pd.DataFrame)
        assert 'stop_loss' in wf['best_params']

    def test_walk_forward_invalid_train_pct(self):
        """train_pct outside (0, 1) → ValueError."""
        data = make_ohlcv(50)
        signals = pd.Series(np.nan, index=data.index, dtype=float)

        with pytest.raises(ValueError, match="train_pct"):
            walk_forward(
                data, signals=signals,
                param_grid={'stop_loss': [0.05]},
                train_pct=1.0,
            )

    def test_walk_forward_test_data_not_empty(self):
        """Test set should have data after split."""
        data = make_ohlcv(50)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[20] = 0
        signals.iloc[40] = 1

        wf = walk_forward(
            data, signals=signals,
            param_grid={'stop_loss': [0.05, 0.10]},
            train_pct=0.6,
            fee_model=ZeroFeeModel(),
        )

        # Test result should have an equity curve
        assert len(wf['test_result'].equity_curve) > 0
