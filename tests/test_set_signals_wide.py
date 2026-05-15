"""v2.7 Commit A — set_signals_wide() entry point + materialization.

Covers the 11 tests required by v2.7 plan §"Commit A":

- basic multi-asset / single-asset / non-DataFrame rejection / extra
  columns dropped / missing-symbol all-NaN  → conversion plumbing
- warn_on_inf default-True / False-silent / strict promotes-to-error
  / strict + warn_on_inf=False raises (v3-decision #20)
- duplicate-index validation
- chained-run re-materialization (v3-decision #4)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from aiphaforge.engine import BacktestEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bdays(n: int = 30) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def _ohlcv(idx: pd.DatetimeIndex, base: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    closes = base + np.cumsum(rng.normal(0, 1, len(idx)))
    return pd.DataFrame(
        {
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "volume": 1_000_000.0,
        },
        index=idx,
    )


def _multi_data(symbols=("AAPL", "MSFT", "TSLA"), n: int = 30):
    idx = _bdays(n)
    return {sym: _ohlcv(idx, base=100.0 + 10.0 * i) for i, sym in enumerate(symbols)}


# ---------------------------------------------------------------------------
# Conversion plumbing
# ---------------------------------------------------------------------------


def test_set_signals_wide_basic_multi_asset():
    # Contract: set_signals_wide(df) is equivalent to
    # set_signals(wide_to_signal_dict(df, data_dict)).
    data = _multi_data()
    idx = next(iter(data.values())).index
    signal = pd.DataFrame(1.0, index=idx, columns=list(data.keys()))
    signal.iloc[0] = 0.0

    engine_dict = BacktestEngine(initial_capital=100_000.0)
    engine_dict.set_signals({sym: signal[sym] for sym in signal.columns})
    baseline = engine_dict.run(data)

    engine_wide = BacktestEngine(initial_capital=100_000.0)
    engine_wide.set_signals_wide(signal)
    result = engine_wide.run(data)

    assert result.total_return == pytest.approx(baseline.total_return, abs=1e-12)
    assert len(result.trades) == len(baseline.trades)


def test_set_signals_wide_single_asset_DataFrame():
    # Single-asset path: wide DF with one column = "default" should
    # produce the same result as set_signals(series).
    idx = _bdays()
    data = _ohlcv(idx)
    series = pd.Series(1.0, index=idx)
    series.iloc[0] = 0.0
    series.iloc[15] = 0.0  # mid-period exit so a trade actually closes

    engine_series = BacktestEngine(initial_capital=100_000.0)
    engine_series.set_signals(series)
    baseline = engine_series.run(data)

    engine_wide = BacktestEngine(initial_capital=100_000.0)
    engine_wide.set_signals_wide(series.to_frame("default"))
    result = engine_wide.run(data)

    assert result.total_return == pytest.approx(baseline.total_return, abs=1e-12)
    assert len(result.trades) == len(baseline.trades)


def test_set_signals_wide_rejects_non_dataframe():
    engine = BacktestEngine()
    with pytest.raises(TypeError, match="set_signals_wide expects pd.DataFrame"):
        engine.set_signals_wide(pd.Series([1.0, 0.0]))
    with pytest.raises(TypeError, match="set_signals_wide expects pd.DataFrame"):
        engine.set_signals_wide({"AAPL": pd.Series([1.0])})
    with pytest.raises(TypeError, match="set_signals_wide expects pd.DataFrame"):
        engine.set_signals_wide(None)


def test_set_signals_wide_extra_columns_dropped():
    data = _multi_data(symbols=("AAPL", "MSFT"))  # 2 symbols
    idx = next(iter(data.values())).index
    # Wide DF has an extra column not in data — should be silently dropped
    # (per wide_to_signal_dict v2.3 contract).
    signal = pd.DataFrame(
        {"AAPL": 1.0, "MSFT": 1.0, "GHOST": 1.0}, index=idx,
    )
    signal.iloc[0] = 0.0
    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_signals_wide(signal)
    result = engine.run(data)
    # Trades only on AAPL/MSFT — GHOST silently absent.
    assert all(t.symbol in {"AAPL", "MSFT"} for t in result.trades)


def test_set_signals_wide_missing_symbol_all_nan():
    data = _multi_data(symbols=("AAPL", "MSFT"))
    idx = next(iter(data.values())).index
    # Wide DF only covers AAPL — MSFT should receive all-NaN signals
    # → engine holds (no trades on MSFT).
    signal = pd.DataFrame({"AAPL": 1.0}, index=idx)
    signal.iloc[0] = 0.0
    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_signals_wide(signal)
    result = engine.run(data)
    assert all(t.symbol == "AAPL" for t in result.trades)


# ---------------------------------------------------------------------------
# warn_on_inf / strict semantics (v3-decision #20)
# ---------------------------------------------------------------------------


def test_set_signals_wide_warn_on_inf_default_true_emits_warning():
    data = _multi_data(symbols=("AAPL",))
    idx = next(iter(data.values())).index
    signal = pd.DataFrame({"AAPL": 1.0}, index=idx)
    signal.iloc[0] = np.inf  # ±Inf in wide DF
    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_signals_wide(signal)  # default warn_on_inf=True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        engine.run(data)
    assert any(
        "Inf values" in str(w.message) for w in caught
    ), f"expected ±Inf warning; got: {[str(w.message) for w in caught]}"


def test_set_signals_wide_warn_on_inf_false_silent():
    data = _multi_data(symbols=("AAPL",))
    idx = next(iter(data.values())).index
    signal = pd.DataFrame({"AAPL": 1.0}, index=idx)
    signal.iloc[0] = np.inf
    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_signals_wide(signal, warn_on_inf=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        engine.run(data)
    assert not any(
        "Inf values" in str(w.message) for w in caught
    ), f"expected silent; got: {[str(w.message) for w in caught]}"


def test_set_signals_wide_strict_promotes_inf_to_error():
    data = _multi_data(symbols=("AAPL",))
    idx = next(iter(data.values())).index
    signal = pd.DataFrame({"AAPL": 1.0}, index=idx)
    signal.iloc[0] = -np.inf
    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_signals_wide(signal, strict=True)
    with pytest.raises(ValueError, match="strict=True.*Inf"):
        engine.run(data)


def test_set_signals_wide_strict_with_warn_on_inf_false_raises():
    # v3-decision #20: strict + explicit conflicting kwarg → setter raises.
    idx = _bdays()
    signal = pd.DataFrame({"AAPL": 1.0}, index=idx)
    engine = BacktestEngine()
    with pytest.raises(
        ValueError,
        match="strict=True.*incompatible.*warn_on_inf=False",
    ):
        engine.set_signals_wide(signal, strict=True, warn_on_inf=False)


def test_set_signals_wide_validates_duplicate_index_raises():
    # validate_signal_wide raises on duplicate timestamps.
    idx = pd.DatetimeIndex(
        ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"],
    )
    signal = pd.DataFrame({"AAPL": [1.0, 0.0, 1.0, 0.0]}, index=idx)
    engine = BacktestEngine()
    with pytest.raises(ValueError):
        engine.set_signals_wide(signal)


# ---------------------------------------------------------------------------
# Chained-run regression (v3-decision #4 — wide DF retained across runs)
# ---------------------------------------------------------------------------


def test_set_signals_wide_chained_run_re_materializes_per_data():
    # Per v3-decision #4: wide DF is RETAINED across chained run() calls
    # and re-materialized against each call's data_dict. The contract
    # under test: a single set_signals_wide(...) followed by two run()
    # calls produces the same outputs as two independent
    # set_signals_wide(...).run() invocations.
    sig_idx = pd.bdate_range("2024-01-01", periods=60)
    signal = pd.DataFrame({"AAPL": 1.0, "MSFT": 1.0}, index=sig_idx)
    signal.iloc[0] = 0.0

    data_q1 = {
        "AAPL": _ohlcv(sig_idx[:30], base=100),
        "MSFT": _ohlcv(sig_idx[:30], base=200),
    }
    data_q2 = {
        "AAPL": _ohlcv(sig_idx[30:], base=100),
        "MSFT": _ohlcv(sig_idx[30:], base=200),
    }

    # Independent baseline: two engines, two set_signals_wide calls.
    engine_a = BacktestEngine(initial_capital=100_000.0)
    engine_a.set_signals_wide(signal)
    baseline_q1 = engine_a.run(data_q1)
    engine_b = BacktestEngine(initial_capital=100_000.0)
    engine_b.set_signals_wide(signal)
    baseline_q2 = engine_b.run(data_q2)

    # Chained: one engine, one set_signals_wide, two runs.
    engine = BacktestEngine(initial_capital=100_000.0)
    engine.set_signals_wide(signal)
    result_q1 = engine.run(data_q1)
    # Wide DF MUST be retained for the second run to re-materialize.
    assert engine._signals_wide is signal, "wide DF must persist across runs"
    result_q2 = engine.run(data_q2)

    assert result_q1.total_return == pytest.approx(
        baseline_q1.total_return, abs=1e-12,
    )
    assert result_q2.total_return == pytest.approx(
        baseline_q2.total_return, abs=1e-12,
    )
