"""
Shared test fixtures for AIphaForge test suite.
"""

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(
    n_bars: int,
    start_price: float = 100.0,
    trend: float = 0.001,
    volatility: float = 0.02,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with a DatetimeIndex.

    Parameters:
        n_bars: Number of bars to generate.
        start_price: Opening price of the first bar.
        trend: Per-bar drift applied to the close price.
        volatility: Per-bar standard deviation of log returns.
        start_date: Start date string for the DatetimeIndex.

    Returns:
        pd.DataFrame with columns [open, high, low, close, volume] and a
        DatetimeIndex at business-day frequency.
    """
    rng = np.random.default_rng(42)
    log_returns = trend + volatility * rng.standard_normal(n_bars)
    # First bar return is zero so that close[0] == start_price
    log_returns[0] = 0.0
    close = start_price * np.exp(np.cumsum(log_returns))

    # Derive OHLV from close
    intraday_noise = volatility * rng.uniform(0.2, 1.0, size=n_bars)
    high = close * (1 + intraday_noise)
    low = close * (1 - intraday_noise)
    open_ = np.empty_like(close)
    open_[0] = start_price
    open_[1:] = close[:-1]  # open equals previous close

    # Clamp open to [low, high] so OHLCV validation never warns
    open_ = np.clip(open_, low, high)
    volume = rng.integers(100_000, 1_000_000, size=n_bars).astype(float)

    dates = pd.bdate_range(start=start_date, periods=n_bars, freq="B")

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def make_probe_ohlcv(
    n: int = 60,
    seed: int = 0,
    start: float = 100.0,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV frame for the v2.0 probe tests.

    Distinct from :func:`make_ohlcv` because the probe tests need a
    seed parameter for reproducibility checks AND a drifting open
    price to exercise the close_vs_open template (where the conftest
    helper pins ``open[i] = close[i-1]`` and would always classify
    the bar by sign of the prior return).

    The returned frame is guaranteed valid OHLCV: positive prices,
    monotonic business-day index, ``high >= max(open, close, low)``
    and ``low <= min(open, close, high)`` by construction.
    """
    rng = np.random.default_rng(seed)
    closes = start * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n))
    opens = closes * (1.0 + rng.normal(0.0, 0.003, size=n))
    spreads = np.abs(rng.normal(0.0, 0.005, size=n)) * closes
    highs = np.maximum(opens, closes) + spreads
    lows = np.minimum(opens, closes) - spreads
    vol = rng.integers(1_000_000, 10_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vol},
        index=pd.bdate_range("2024-01-01", periods=n),
    )


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """100-bar OHLCV DataFrame with moderate upward trend."""
    return make_ohlcv(100)


@pytest.fixture
def sample_signals(sample_data: pd.DataFrame) -> pd.Series:
    """Signal series matching sample_data index: buy-hold-sell pattern.

    Pattern: buy at bar 10, hold through bar 70, sell at bar 70, flat until end.
    """
    signals = pd.Series(np.nan, index=sample_data.index, dtype=float)
    signals.iloc[10] = 1   # buy
    signals.iloc[70] = -1  # sell
    return signals
