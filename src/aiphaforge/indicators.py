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

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # When avg_loss=0 (all gains): RS=inf, RSI should be 100
    rsi = rsi.fillna(100.0)
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
    """Average True Range (Wilder's EMA smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def VWAP(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume Weighted Average Price (running cumulative).

    Note: this is a running cumulative VWAP with no daily reset.
    Correct for daily bars or single-session intraday. For multi-day
    intraday data, VWAP should reset each day — group by date and
    apply this function per group.
    """
    typical_price = (high + low + close) / 3
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (volume * direction).cumsum()


# ---------------------------------------------------------------------------
# Advanced Trend
# ---------------------------------------------------------------------------

def SUPERTREND(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 10,
    mult: float = 3.0,
) -> Dict[str, pd.Series]:
    """Supertrend indicator.

    Returns:
        Dict with keys 'supertrend' (the line) and 'direction'
        (1 = bullish, -1 = bearish).
    """
    atr = ATR(high, low, close, window)
    hl2 = (high + low) / 2
    upper_basic = hl2 + mult * atr
    lower_basic = hl2 - mult * atr

    n = len(close)
    upper_band = np.empty(n)
    lower_band = np.empty(n)
    supertrend = np.empty(n)
    direction = np.empty(n)

    upper_band[:] = np.nan
    lower_band[:] = np.nan
    supertrend[:] = np.nan
    direction[:] = np.nan

    ub_vals = upper_basic.values
    lb_vals = lower_basic.values
    cl_vals = close.values

    for i in range(1, n):
        if np.isnan(ub_vals[i]):
            continue

        # Upper band: only decrease (tighten), never widen
        if not np.isnan(upper_band[i - 1]) and cl_vals[i - 1] <= upper_band[i - 1]:
            upper_band[i] = min(ub_vals[i], upper_band[i - 1])
        else:
            upper_band[i] = ub_vals[i]

        # Lower band: only increase (tighten), never widen
        if not np.isnan(lower_band[i - 1]) and cl_vals[i - 1] >= lower_band[i - 1]:
            lower_band[i] = max(lb_vals[i], lower_band[i - 1])
        else:
            lower_band[i] = lb_vals[i]

        # Direction
        if np.isnan(direction[i - 1]):
            direction[i] = 1 if cl_vals[i] > upper_band[i] else -1
        elif direction[i - 1] == 1:
            direction[i] = -1 if cl_vals[i] < lower_band[i] else 1
        else:
            direction[i] = 1 if cl_vals[i] > upper_band[i] else -1

        supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

    idx = close.index
    return {
        'supertrend': pd.Series(supertrend, index=idx),
        'direction': pd.Series(direction, index=idx),
    }


def ICHIMOKU(
    high: pd.Series,
    low: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> Dict[str, pd.Series]:
    """Ichimoku Cloud components (no displacement/shift applied).

    Returns:
        Dict with keys 'tenkan_sen', 'kijun_sen', 'senkou_a', 'senkou_b'.
    """
    tenkan_sen = (
        high.rolling(window=tenkan, min_periods=tenkan).max()
        + low.rolling(window=tenkan, min_periods=tenkan).min()
    ) / 2

    kijun_sen = (
        high.rolling(window=kijun, min_periods=kijun).max()
        + low.rolling(window=kijun, min_periods=kijun).min()
    ) / 2

    senkou_a_val = (tenkan_sen + kijun_sen) / 2

    senkou_b_val = (
        high.rolling(window=senkou_b, min_periods=senkou_b).max()
        + low.rolling(window=senkou_b, min_periods=senkou_b).min()
    ) / 2

    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_a': senkou_a_val,
        'senkou_b': senkou_b_val,
    }


def ADX(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Average Directional Index (0-100).

    Measures trend strength regardless of direction.
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_smooth = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(
        alpha=1 / window, min_periods=window, adjust=False
    ).mean() / atr_smooth
    minus_di = 100 * minus_dm.ewm(
        alpha=1 / window, min_periods=window, adjust=False
    ).mean() / atr_smooth

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return adx


def DONCHIAN(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
) -> Dict[str, pd.Series]:
    """Donchian Channel.

    Returns:
        Dict with keys 'upper', 'lower', 'middle'.
    """
    upper = high.rolling(window=window, min_periods=window).max()
    lower = low.rolling(window=window, min_periods=window).min()
    middle = (upper + lower) / 2
    return {'upper': upper, 'lower': lower, 'middle': middle}


# ---------------------------------------------------------------------------
# Extended Trend (v1.0.3)
# ---------------------------------------------------------------------------

def WMA(series: pd.Series, window: int) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window=window, min_periods=window).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def DEMA(series: pd.Series, window: int) -> pd.Series:
    """Double Exponential Moving Average."""
    ema1 = EMA(series, window)
    ema2 = EMA(ema1, window)
    return 2 * ema1 - ema2


def TEMA(series: pd.Series, window: int) -> pd.Series:
    """Triple Exponential Moving Average."""
    ema1 = EMA(series, window)
    ema2 = EMA(ema1, window)
    ema3 = EMA(ema2, window)
    return 3 * ema1 - 3 * ema2 + ema3


def PSAR(
    high: pd.Series,
    low: pd.Series,
    af: float = 0.02,
    max_af: float = 0.2,
) -> Dict[str, pd.Series]:
    """Parabolic SAR.

    Returns:
        Dict with 'psar' (SAR values) and 'direction' (1=bull, -1=bear).
    """
    n = len(high)
    psar = np.empty(n)
    direction = np.empty(n)
    h = high.values.astype(float)
    lo = low.values.astype(float)

    psar[0] = lo[0]
    direction[0] = 1
    ep = h[0]
    cur_af = af

    for i in range(1, n):
        prev_psar = psar[i - 1]
        prev_dir = direction[i - 1]

        if prev_dir == 1:
            sar = prev_psar + cur_af * (ep - prev_psar)
            sar = min(sar, lo[i - 1])
            if i >= 2:
                sar = min(sar, lo[i - 2])
            if lo[i] < sar:
                direction[i] = -1
                psar[i] = ep
                ep = lo[i]
                cur_af = af
            else:
                direction[i] = 1
                psar[i] = sar
                if h[i] > ep:
                    ep = h[i]
                    cur_af = min(cur_af + af, max_af)
        else:
            sar = prev_psar + cur_af * (ep - prev_psar)
            sar = max(sar, h[i - 1])
            if i >= 2:
                sar = max(sar, h[i - 2])
            if h[i] > sar:
                direction[i] = 1
                psar[i] = ep
                ep = h[i]
                cur_af = af
            else:
                direction[i] = -1
                psar[i] = sar
                if lo[i] < ep:
                    ep = lo[i]
                    cur_af = min(cur_af + af, max_af)

    return {
        'psar': pd.Series(psar, index=high.index),
        'direction': pd.Series(direction, index=high.index),
    }


def KELTNER(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    atr_window: int = 10,
    mult: float = 1.5,
) -> Dict[str, pd.Series]:
    """Keltner Channels.

    Returns:
        Dict with 'upper', 'middle', 'lower'.
    """
    middle = EMA(close, window)
    atr = ATR(high, low, close, atr_window)
    return {
        'upper': middle + mult * atr,
        'middle': middle,
        'lower': middle - mult * atr,
    }


# ---------------------------------------------------------------------------
# Extended Momentum (v1.0.3)
# ---------------------------------------------------------------------------

def CCI(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    sma_tp = SMA(tp, window)
    mean_dev = (tp - sma_tp).abs().rolling(
        window=window, min_periods=window).mean()
    return (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))


def WILLR(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Williams %R (-100 to 0)."""
    highest = high.rolling(window=window, min_periods=window).max()
    lowest = low.rolling(window=window, min_periods=window).min()
    denom = (highest - lowest).replace(0, np.nan)
    return (highest - close) / denom * -100


def MFI(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Money Flow Index (0-100). Volume-weighted RSI."""
    tp = (high + low + close) / 3
    raw_mf = tp * volume
    delta = tp.diff()
    pos_mf = raw_mf.where(delta > 0, 0.0)
    neg_mf = raw_mf.where(delta < 0, 0.0)
    pos_sum = pos_mf.rolling(window=window, min_periods=window).sum()
    neg_sum = neg_mf.rolling(window=window, min_periods=window).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def STOCHRSI(
    series: pd.Series,
    rsi_window: int = 14,
    stoch_window: int = 14,
) -> Dict[str, pd.Series]:
    """Stochastic RSI.

    Returns:
        Dict with 'k' (0-1 range) and 'd' (3-period SMA of k).
    """
    rsi = RSI(series, rsi_window)
    lowest = rsi.rolling(window=stoch_window, min_periods=stoch_window).min()
    highest = rsi.rolling(window=stoch_window, min_periods=stoch_window).max()
    denom = (highest - lowest).replace(0, np.nan)
    k = (rsi - lowest) / denom
    d = k.rolling(window=3, min_periods=3).mean()
    return {'k': k, 'd': d}


# ---------------------------------------------------------------------------
# Extended Volume (v1.0.3)
# ---------------------------------------------------------------------------

def AD(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Accumulation/Distribution Line."""
    denom = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / denom
    return (clv.fillna(0) * volume).cumsum()


def CMF(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Chaikin Money Flow (approx -1 to 1)."""
    denom = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / denom
    mf_volume = clv.fillna(0) * volume
    return (
        mf_volume.rolling(window=window, min_periods=window).sum()
        / volume.rolling(window=window, min_periods=window).sum().replace(
            0, np.nan)
    )
