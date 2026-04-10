"""
Technical Indicators
====================

Pure computation functions for common technical indicators.
Input ``pd.Series`` (or high/low/close), output ``pd.Series`` or dict.

Not imported by the main package — use directly::

    from aiphaforge.indicators import SMA, EMA, RSI, MACD, BBANDS

All functions are stateless and depend only on pandas/numpy.
"""

from typing import Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

def SMA(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def EMA(series: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=window, adjust=False).mean()


def MACD(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Dict[str, pd.Series]:
    """Moving Average Convergence Divergence.

    Returns:
        Dict with keys 'macd', 'signal', 'histogram'.
    """
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram,
    }


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def RSI(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (0-100)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def ROC(series: pd.Series, window: int = 10) -> pd.Series:
    """Rate of Change (percentage)."""
    return series.pct_change(periods=window) * 100


def STOCH(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> Dict[str, pd.Series]:
    """Stochastic Oscillator (%K, %D).

    Returns:
        Dict with keys 'k' and 'd'.
    """
    lowest = low.rolling(window=k_window, min_periods=k_window).min()
    highest = high.rolling(window=k_window, min_periods=k_window).max()
    denom = (highest - lowest).replace(0, np.nan)
    k = 100 * (close - lowest) / denom
    d = k.rolling(window=d_window, min_periods=d_window).mean()
    return {'k': k, 'd': d}


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def BBANDS(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> Dict[str, pd.Series]:
    """Bollinger Bands.

    Returns:
        Dict with keys 'upper', 'middle', 'lower'.
    """
    middle = SMA(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return {'upper': upper, 'middle': middle, 'lower': lower}


def ATR(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def VWAP(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume Weighted Average Price (cumulative intraday)."""
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (volume * direction).cumsum()
