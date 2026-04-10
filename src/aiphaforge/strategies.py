"""
Strategy Templates
==================

Pre-built strategies combining indicators into signals. One-line backtest::

    from aiphaforge.strategies import MACrossover
    result = MACrossover(short=10, long=30).backtest(data, fee_model='china')

Or generate signals for custom engine configuration::

    signals = MACrossover(short=10, long=30).generate_signals(data)

Supports single-asset (pd.DataFrame) and multi-asset (Dict[str, pd.DataFrame]).
"""

from typing import Dict, Union

import numpy as np
import pandas as pd

from .indicators import BBANDS, EMA, MACD, RSI, SMA


def _transitions_only(raw: pd.Series) -> pd.Series:
    """Convert a raw position series (1/-1/NaN on every bar) to
    transition-only signals (emit only when direction changes, NaN=hold).

    This prevents micro-rebalancing: the engine only sees a signal at
    the crossover point, not on every bar where the condition holds.
    """
    filled = raw.ffill()
    changed = filled != filled.shift(1)
    signals = pd.Series(np.nan, index=raw.index, dtype=float)
    signals[changed] = filled[changed]
    return signals


class BaseStrategy:
    """Base class for strategy templates.

    Subclasses implement ``_compute(df)`` to convert a single OHLCV
    DataFrame into a signal Series. ``generate_signals`` and ``backtest``
    handle single/multi-asset dispatch automatically.
    """

    name: str = "Unknown"

    def generate_signals(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    ) -> Union[pd.Series, Dict[str, pd.Series]]:
        """Generate signals from OHLCV data.

        Returns:
            pd.Series for single-asset, Dict[str, pd.Series] for multi.
        """
        if isinstance(data, dict):
            return {sym: self._compute(df) for sym, df in data.items()}
        return self._compute(data)

    def _compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute signals for a single DataFrame. Override in subclass."""
        raise NotImplementedError

    def backtest(self, data, **engine_kwargs):
        """One-line backtest.

        Parameters:
            data: OHLCV DataFrame or dict.
            **engine_kwargs: Passed to BacktestEngine constructor
                (fee_model, initial_capital, mode, etc.).

        Returns:
            BacktestResult
        """
        from .engine import BacktestEngine
        engine = BacktestEngine(**engine_kwargs)
        engine.set_strategy(self)
        return engine.run(data)

    def __repr__(self):
        params = ', '.join(
            f'{k}={v!r}' for k, v in self.__dict__.items()
            if not k.startswith('_')
        )
        return f"{self.__class__.__name__}({params})"


class MACrossover(BaseStrategy):
    """Moving Average Crossover.

    Buy when short MA crosses above long MA, sell when it crosses below.

    Parameters:
        short: Short MA window (default 10).
        long: Long MA window (default 30).
        ma_type: 'sma' or 'ema' (default 'sma').
    """

    name = "MA Crossover"

    def __init__(self, short: int = 10, long: int = 30, ma_type: str = "sma"):
        self.short = short
        self.long = long
        self.ma_type = ma_type

    def _compute(self, df):
        close = df['close']
        ma_fn = EMA if self.ma_type == 'ema' else SMA
        ma_short = ma_fn(close, self.short)
        ma_long = ma_fn(close, self.long)

        raw = pd.Series(np.nan, index=df.index, dtype=float)
        raw[ma_short > ma_long] = 1
        raw[ma_short < ma_long] = -1
        return _transitions_only(raw)


class RSIMeanReversion(BaseStrategy):
    """RSI Mean Reversion.

    Buy when RSI drops below oversold, sell when it rises above overbought.

    Parameters:
        period: RSI window (default 14).
        oversold: Buy threshold (default 30).
        overbought: Sell threshold (default 70).
    """

    name = "RSI Mean Reversion"

    def __init__(self, period: int = 14, oversold: float = 30,
                 overbought: float = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def _compute(self, df):
        rsi = RSI(df['close'], self.period)

        raw = pd.Series(np.nan, index=df.index, dtype=float)
        raw[rsi < self.oversold] = 1
        raw[rsi > self.overbought] = -1
        return _transitions_only(raw)


class BollingerBreakout(BaseStrategy):
    """Bollinger Band Breakout.

    Buy when close breaks above upper band, sell when below lower band.

    Parameters:
        window: Bollinger window (default 20).
        num_std: Number of standard deviations (default 2.0).
    """

    name = "Bollinger Breakout"

    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std

    def _compute(self, df):
        bands = BBANDS(df['close'], self.window, self.num_std)

        raw = pd.Series(np.nan, index=df.index, dtype=float)
        raw[df['close'] > bands['upper']] = 1
        raw[df['close'] < bands['lower']] = -1
        return _transitions_only(raw)


class MACDStrategy(BaseStrategy):
    """MACD Histogram Strategy.

    Buy when MACD histogram > 0, sell when < 0.

    Parameters:
        fast: Fast EMA window (default 12).
        slow: Slow EMA window (default 26).
        signal: Signal EMA window (default 9).
    """

    name = "MACD Strategy"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def _compute(self, df):
        macd = MACD(df['close'], self.fast, self.slow, self.signal)

        raw = pd.Series(np.nan, index=df.index, dtype=float)
        raw[macd['histogram'] > 0] = 1
        raw[macd['histogram'] < 0] = -1
        return _transitions_only(raw)
