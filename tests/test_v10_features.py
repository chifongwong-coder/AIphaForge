"""
End-to-end tests for v1.0: data loaders and plotting.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from aiphaforge import BacktestEngine, BacktestResult
from aiphaforge.data import load_csv
from aiphaforge.fees import ZeroFeeModel

from .conftest import make_ohlcv


class TestLoadCSV:
    """CSV data loader."""

    def test_load_csv_auto_detect(self):
        """Auto-detect date and OHLCV columns."""
        df = make_ohlcv(20)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index_label='Date')
            loaded = load_csv(f.name)
            os.unlink(f.name)

        assert isinstance(loaded, pd.DataFrame)
        assert isinstance(loaded.index, pd.DatetimeIndex)
        assert 'open' in loaded.columns
        assert 'close' in loaded.columns
        assert len(loaded) == 20

    def test_load_csv_explicit_date_col(self):
        """Explicit date_col parameter."""
        df = make_ohlcv(10)
        df.index.name = 'trade_date'
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df.to_csv(f.name)
            loaded = load_csv(f.name, date_col='trade_date')
            os.unlink(f.name)

        assert len(loaded) == 10

    def test_load_csv_no_date_col_raises(self):
        """No recognizable date column → clear error."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            f.write("x,open,high,low,close,volume\n")
            f.write("1,100,105,95,102,1000\n")
            path = f.name

        with pytest.raises(ValueError, match="Cannot auto-detect"):
            load_csv(path)
        os.unlink(path)

    def test_load_csv_ready_for_engine(self):
        """Loaded CSV can be passed directly to engine.run()."""
        df = make_ohlcv(30)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index_label='Date')
            loaded = load_csv(f.name)
            os.unlink(f.name)

        signals = pd.Series(np.nan, index=loaded.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[20] = 0

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode='event_driven',
            initial_capital=100_000,
            include_benchmark=False,
        )
        engine.set_signals(signals)
        result = engine.run(loaded)
        assert isinstance(result, BacktestResult)

    def test_sample_data_exists(self):
        """Bundled sample data files exist."""
        base = os.path.join(os.path.dirname(__file__), '..', 'examples', 'sample_data')
        assert os.path.exists(os.path.join(base, 'AAPL.csv'))
        assert os.path.exists(os.path.join(base, 'TSLA.csv'))


class TestPlotting:
    """Plotting (requires matplotlib)."""

    def test_plot_result_returns_figure(self):
        """plot_result returns a matplotlib Figure."""
        try:
            from aiphaforge.plotting import plot_result
        except ImportError:
            pytest.skip("matplotlib not installed")

        data = make_ohlcv(30)
        signals = pd.Series(np.nan, index=data.index, dtype=float)
        signals.iloc[1] = 1
        signals.iloc[20] = 0

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(),
            mode='event_driven',
            initial_capital=100_000,
            include_benchmark=True,
        )
        engine.set_signals(signals)
        result = engine.run(data)

        fig = plot_result(result)
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_comparison_returns_figure(self):
        """plot_comparison returns a matplotlib Figure."""
        try:
            from aiphaforge.plotting import plot_comparison
        except ImportError:
            pytest.skip("matplotlib not installed")

        data = make_ohlcv(30)
        sig1 = pd.Series(np.nan, index=data.index, dtype=float)
        sig1.iloc[1] = 1
        sig1.iloc[20] = 0
        sig2 = pd.Series(np.nan, index=data.index, dtype=float)
        sig2.iloc[5] = 1
        sig2.iloc[25] = 0

        engine = BacktestEngine(
            fee_model=ZeroFeeModel(), mode='event_driven',
            initial_capital=100_000, include_benchmark=False)

        engine.set_signals(sig1)
        r1 = engine.run(data)
        engine.set_signals(sig2)
        r2 = engine.run(data)

        fig = plot_comparison({"Strategy A": r1, "Strategy B": r2})
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)
