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

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .indicators import (
    ADX,
    BBANDS,
    DONCHIAN,
    EMA,
    ICHIMOKU,
    MACD,
    ROC,
    RSI,
    SMA,
    SUPERTREND,
    VWAP,
)


def _transitions_only(raw: pd.Series) -> pd.Series:
    """Convert a raw position series (1/-1/0/NaN on every bar) to
    transition-only signals (emit only when direction changes, NaN=hold).

    Supports 0 (flatten) as a valid signal value.
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

    @property
    def params(self) -> Dict[str, Any]:
        """Current strategy parameters (read-only view)."""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}

    def update_params(self, **kwargs) -> None:
        """Update parameters. Takes effect on next signal generation.

        Raises ValueError for unknown parameter names. This is the
        foundation for v1.2 MetaController where an Agent adjusts
        strategy params mid-backtest.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k) or k.startswith('_'):
                raise ValueError(
                    f"Unknown parameter '{k}' for "
                    f"{self.__class__.__name__}. "
                    f"Valid: {list(self.params.keys())}")
            setattr(self, k, v)

    def default_param_grid(self) -> Dict[str, list]:
        """Suggested parameter ranges for optimization.

        Override in subclass. Returns dict suitable for optimize().
        """
        return {}

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
            f'{k}={v!r}' for k, v in self.params.items()
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

    def default_param_grid(self):
        return {'short': [5, 10, 20], 'long': [20, 30, 50]}

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

    def default_param_grid(self):
        return {'period': [7, 14, 21], 'oversold': [20, 30], 'overbought': [70, 80]}

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

    def default_param_grid(self):
        return {'window': [10, 20, 30], 'num_std': [1.5, 2.0, 2.5]}

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

    def default_param_grid(self):
        return {'fast': [8, 12, 16], 'slow': [21, 26, 30], 'signal': [7, 9, 12]}

    def _compute(self, df):
        macd = MACD(df['close'], self.fast, self.slow, self.signal)

        raw = pd.Series(np.nan, index=df.index, dtype=float)
        raw[macd['histogram'] > 0] = 1
        raw[macd['histogram'] < 0] = -1
        return _transitions_only(raw)


# ---------------------------------------------------------------------------
# Advanced strategies
# ---------------------------------------------------------------------------

class SupertrendStrategy(BaseStrategy):
    """Supertrend Trend Following.

    Follow the Supertrend direction: long when bullish, short when bearish.

    Parameters:
        window: ATR window for Supertrend calculation (default 10).
        mult: ATR multiplier for band width (default 3.0).
    """

    name = "Supertrend"

    def __init__(self, window: int = 10, mult: float = 3.0):
        self.window = window
        self.mult = mult

    def default_param_grid(self):
        return {'window': [7, 10, 14], 'mult': [2.0, 3.0, 4.0]}

    def _compute(self, df):
        st = SUPERTREND(df['high'], df['low'], df['close'],
                        self.window, self.mult)

        raw = pd.Series(np.nan, index=df.index, dtype=float)
        raw[st['direction'] == 1] = 1
        raw[st['direction'] == -1] = -1
        return _transitions_only(raw)


class IchimokuStrategy(BaseStrategy):
    """Ichimoku Cloud Strategy.

    Long when tenkan > kijun AND close above senkou_a (above cloud).
    Short when tenkan < kijun AND close below senkou_b (below cloud).

    Parameters:
        tenkan: Tenkan-sen (conversion line) window (default 9).
        kijun: Kijun-sen (base line) window (default 26).
        senkou_b: Senkou Span B window (default 52).
    """

    name = "Ichimoku Cloud"

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b

    def default_param_grid(self):
        return {'tenkan': [7, 9, 12], 'kijun': [22, 26, 30],
                'senkou_b': [44, 52, 60]}

    def _compute(self, df):
        ichi = ICHIMOKU(df['high'], df['low'],
                        self.tenkan, self.kijun, self.senkou_b,
                        close=df['close'])
        close = df['close']

        raw = pd.Series(np.nan, index=df.index, dtype=float)
        buy = (ichi['tenkan_sen'] > ichi['kijun_sen']) & (close > ichi['senkou_a'])
        sell = (ichi['tenkan_sen'] < ichi['kijun_sen']) & (close < ichi['senkou_b'])
        raw[buy] = 1
        raw[sell] = -1
        return _transitions_only(raw)


class ADXTrendFollowing(BaseStrategy):
    """ADX Trend Following.

    Trade only when ADX indicates a strong trend (above threshold).
    Direction determined by close vs. SMA. No signal when ADX is low.

    Parameters:
        adx_window: ADX calculation window (default 14).
        adx_threshold: Minimum ADX value for trend trading (default 25).
        ma_window: SMA window for direction (default 20).
    """

    name = "ADX Trend Following"

    def __init__(self, adx_window: int = 14, adx_threshold: float = 25,
                 ma_window: int = 20):
        self.adx_window = adx_window
        self.adx_threshold = adx_threshold
        self.ma_window = ma_window

    def default_param_grid(self):
        return {'adx_window': [10, 14, 20], 'adx_threshold': [20, 25, 30],
                'ma_window': [10, 20, 50]}

    def _compute(self, df):
        adx = ADX(df['high'], df['low'], df['close'], self.adx_window)
        ma = SMA(df['close'], self.ma_window)
        close = df['close']

        raw = pd.Series(np.nan, index=df.index, dtype=float)
        trending = adx > self.adx_threshold
        raw[trending & (close > ma)] = 1
        raw[trending & (close < ma)] = -1
        # When ADX < threshold: exit position (flat), not hold
        raw[~trending & adx.notna()] = 0
        return _transitions_only(raw)


class DonchianBreakout(BaseStrategy):
    """Donchian Channel Breakout (Turtle Trading variant).

    Buy when close breaks above the upper channel (previous bar).
    Sell when close breaks below the lower channel (previous bar).

    Parameters:
        window: Donchian channel lookback window (default 20).
    """

    name = "Donchian Breakout"

    def __init__(self, window: int = 20):
        self.window = window

    def default_param_grid(self):
        return {'window': [10, 20, 55]}

    def _compute(self, df):
        dc = DONCHIAN(df['high'], df['low'], self.window)
        close = df['close']

        # Compare against previous bar's channel to avoid lookahead
        upper_prev = dc['upper'].shift(1)
        lower_prev = dc['lower'].shift(1)

        raw = pd.Series(np.nan, index=df.index, dtype=float)
        raw[close > upper_prev] = 1
        raw[close < lower_prev] = -1
        return _transitions_only(raw)


class MeanReversionBollinger(BaseStrategy):
    """Bollinger Band Mean Reversion.

    Opposite of BollingerBreakout: buy at the lower band (oversold),
    sell at the upper band (overbought).

    Parameters:
        window: Bollinger window (default 20).
        num_std: Number of standard deviations (default 2.0).
    """

    name = "Mean Reversion Bollinger"

    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std

    def default_param_grid(self):
        return {'window': [10, 20, 30], 'num_std': [1.5, 2.0, 2.5]}

    def _compute(self, df):
        bands = BBANDS(df['close'], self.window, self.num_std)
        close = df['close']

        raw = pd.Series(np.nan, index=df.index, dtype=float)
        raw[close < bands['lower']] = 1
        raw[close > bands['upper']] = -1
        return _transitions_only(raw)


class VWAPReversion(BaseStrategy):
    """VWAP Mean Reversion.

    Buy when price is below VWAP (undervalued), sell when above (overvalued).
    No tunable parameters — uses raw cumulative VWAP.

    Requires 'high', 'low', 'close', and 'volume' columns.
    """

    name = "VWAP Reversion"

    def __init__(self):
        pass

    def default_param_grid(self):
        return {}

    def _compute(self, df):
        vwap = VWAP(df['high'], df['low'], df['close'], df['volume'])
        close = df['close']

        raw = pd.Series(np.nan, index=df.index, dtype=float)
        raw[close < vwap] = 1
        raw[close > vwap] = -1
        return _transitions_only(raw)


class MomentumRank(BaseStrategy):
    """Cross-Sectional Momentum Rank.

    Multi-asset: rank all assets by ROC, go long the top N.
    Single-asset: fallback to simple ROC > 0 = buy.

    This strategy overrides ``generate_signals`` because cross-sectional
    ranking requires access to all assets simultaneously.

    Parameters:
        roc_window: Rate of Change lookback (default 20).
        top_n: Number of top-ranked assets to go long (default 3).
    """

    name = "Momentum Rank"

    def __init__(self, roc_window: int = 20, top_n: int = 3):
        self.roc_window = roc_window
        self.top_n = top_n

    def default_param_grid(self):
        return {'roc_window': [10, 20, 60], 'top_n': [1, 3, 5]}

    def generate_signals(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    ) -> Union[pd.Series, Dict[str, pd.Series]]:
        """Cross-sectional ranking for multi-asset, ROC sign for single."""
        if isinstance(data, dict):
            return self._compute_multi(data)
        return self._compute(data)

    def _compute(self, df):
        """Single-asset fallback: long when ROC > 0."""
        roc = ROC(df['close'], self.roc_window)

        raw = pd.Series(np.nan, index=df.index, dtype=float)
        raw[roc > 0] = 1
        raw[roc <= 0] = -1
        return _transitions_only(raw)

    def _compute_multi(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.Series]:
        """Rank all assets by ROC each bar, long the top N."""
        # Build ROC DataFrame: columns = symbols, index = dates
        roc_df = pd.DataFrame({
            sym: ROC(df['close'], self.roc_window) for sym, df in data.items()
        })

        # Rank each row (ascending: highest ROC = highest rank number)
        ranks = roc_df.rank(axis=1, ascending=True, na_option='bottom')
        n_assets = len(data)
        cutoff = n_assets - self.top_n  # top_n assets have rank > cutoff

        signals = {}
        for sym in data:
            raw = pd.Series(np.nan, index=roc_df.index, dtype=float)
            raw[ranks[sym] > cutoff] = 1
            raw[ranks[sym] <= cutoff] = -1
            signals[sym] = _transitions_only(raw)

        return signals


class PairsTrading(BaseStrategy):
    """Pairs Trading (Spread Mean Reversion).

    Requires exactly 2 assets. Computes the price spread and trades
    on z-score of the spread reverting to the mean.

    Note: Uses raw price difference (A - B) as the spread. Works best
    when both assets have similar price scales. For assets with very
    different price levels, consider using log-prices or a ratio spread
    externally before feeding to this strategy.

    This strategy overrides ``generate_signals`` because it needs
    simultaneous access to both assets.

    Parameters:
        window: Rolling window for spread mean and std (default 30).
        entry_z: Z-score threshold to enter a trade (default 2.0).
        exit_z: Z-score threshold to flatten position (default 0.5).
    """

    name = "Pairs Trading"

    def __init__(self, window: int = 30, entry_z: float = 2.0,
                 exit_z: float = 0.5):
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def default_param_grid(self):
        return {'window': [20, 30, 60], 'entry_z': [1.5, 2.0, 2.5],
                'exit_z': [0.0, 0.5, 1.0]}

    def generate_signals(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    ) -> Union[pd.Series, Dict[str, pd.Series]]:
        """Pairs trading requires exactly 2 assets passed as a dict."""
        if isinstance(data, dict):
            return self._compute_pair(data)
        raise ValueError(
            "PairsTrading requires a dict with exactly 2 symbols. "
            "Got a single DataFrame."
        )

    def _compute(self, df):
        raise NotImplementedError(
            "PairsTrading requires 2 assets. Use generate_signals(dict)."
        )

    def _compute_pair(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.Series]:
        """Compute spread z-score between two assets."""
        symbols = list(data.keys())
        if len(symbols) != 2:
            raise ValueError(
                f"PairsTrading requires exactly 2 symbols, got {len(symbols)}: "
                f"{symbols}"
            )

        sym_a, sym_b = symbols
        close_a = data[sym_a]['close']
        close_b = data[sym_b]['close']

        # Align on common index
        spread = close_a - close_b
        spread_mean = SMA(spread, self.window)
        spread_std = spread.rolling(
            window=self.window, min_periods=self.window
        ).std()

        z = (spread - spread_mean) / spread_std.replace(0, np.nan)

        # Asset A signals: buy spread (long A, short B) when z is very negative
        raw_a = pd.Series(np.nan, index=spread.index, dtype=float)
        raw_a[z < -self.entry_z] = 1    # Spread too low → long A
        raw_a[z > self.entry_z] = -1    # Spread too high → short A
        raw_a[z.abs() < self.exit_z] = 0  # Flatten near mean

        # Asset B is the opposite leg
        raw_b = pd.Series(np.nan, index=spread.index, dtype=float)
        raw_b[z < -self.entry_z] = -1   # Short B when long A
        raw_b[z > self.entry_z] = 1     # Long B when short A
        raw_b[z.abs() < self.exit_z] = 0  # Flatten near mean

        return {
            sym_a: _transitions_only(raw_a),
            sym_b: _transitions_only(raw_b),
        }


class MultiIndicatorStrategy(BaseStrategy):
    """User-Defined Multi-Indicator Strategy.

    Combines multiple indicator conditions into buy/sell rules.
    Default: RSI < 30 AND close > SMA(50) = buy,
             RSI > 70 AND close < SMA(50) = sell.

    Users can customize by providing indicator/threshold combinations
    via the ``indicators`` and ``thresholds`` parameters.

    Parameters:
        indicators: List of indicator names to use (default: ['rsi', 'sma']).
        thresholds: Dict of threshold values keyed by indicator
            (default: {'rsi_oversold': 30, 'rsi_overbought': 70,
                       'sma_window': 50}).
    """

    name = "Multi-Indicator"

    def __init__(
        self,
        indicators: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.indicators = indicators or ['rsi', 'sma']
        self.thresholds = thresholds or {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'sma_window': 50,
        }

    def default_param_grid(self):
        return {'thresholds': [
            {'rsi_oversold': 20, 'rsi_overbought': 80, 'sma_window': 50},
            {'rsi_oversold': 30, 'rsi_overbought': 70, 'sma_window': 50},
            {'rsi_oversold': 30, 'rsi_overbought': 70, 'sma_window': 200},
        ]}

    def _compute(self, df):
        close = df['close']
        buy_conditions: List[pd.Series] = []
        sell_conditions: List[pd.Series] = []

        for ind in self.indicators:
            if ind == 'rsi':
                rsi_window = int(self.thresholds.get('rsi_window', 14))
                oversold = self.thresholds.get('rsi_oversold', 30)
                overbought = self.thresholds.get('rsi_overbought', 70)
                rsi = RSI(close, rsi_window)
                buy_conditions.append(rsi < oversold)
                sell_conditions.append(rsi > overbought)

            elif ind == 'sma':
                sma_window = int(self.thresholds.get('sma_window', 50))
                ma = SMA(close, sma_window)
                buy_conditions.append(close > ma)
                sell_conditions.append(close < ma)

            elif ind == 'ema':
                ema_window = int(self.thresholds.get('ema_window', 50))
                ma = EMA(close, ema_window)
                buy_conditions.append(close > ma)
                sell_conditions.append(close < ma)

            elif ind == 'bbands':
                bb_window = int(self.thresholds.get('bb_window', 20))
                bb_std = self.thresholds.get('bb_num_std', 2.0)
                bands = BBANDS(close, bb_window, bb_std)
                buy_conditions.append(close < bands['lower'])
                sell_conditions.append(close > bands['upper'])

            elif ind == 'macd':
                fast = int(self.thresholds.get('macd_fast', 12))
                slow = int(self.thresholds.get('macd_slow', 26))
                sig = int(self.thresholds.get('macd_signal', 9))
                macd = MACD(close, fast, slow, sig)
                buy_conditions.append(macd['histogram'] > 0)
                sell_conditions.append(macd['histogram'] < 0)

        # Combine all conditions with AND logic
        if not buy_conditions:
            combined_buy = pd.Series(False, index=df.index)
            combined_sell = pd.Series(False, index=df.index)
        else:
            combined_buy = buy_conditions[0]
            combined_sell = sell_conditions[0]
            for cond in buy_conditions[1:]:
                combined_buy = combined_buy & cond
            for cond in sell_conditions[1:]:
                combined_sell = combined_sell & cond

        raw = pd.Series(np.nan, index=df.index, dtype=float)
        raw[combined_buy] = 1
        raw[combined_sell] = -1
        return _transitions_only(raw)


# ---------------------------------------------------------------------------
# Strategy Composition Tree (v1.4)
# ---------------------------------------------------------------------------


class StrategyNode(BaseStrategy):
    """Base for composite strategy nodes.

    A composite node combines multiple child strategies into a single
    signal stream. Composites are BaseStrategy subclasses, so they work
    everywhere a strategy works: engine.set_strategy(), backtest(),
    generate_signals(), MetaController, and optimizer.

    Parameters:
        children: List of child strategies (BaseStrategy instances).
    """

    name = "Strategy Node"

    def __init__(self, children: List[BaseStrategy]):
        self.children = children

    @property
    def params(self) -> Dict[str, Any]:
        return {
            'children': [c.name for c in self.children],
            **{k: v for k, v in self.__dict__.items()
               if k not in ('children',) and not k.startswith('_')},
        }

    def update_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k) and k != 'children':
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown param '{k}' for {self.name}")

    def default_param_grid(self) -> Dict[str, list]:
        return {}

    def _compute(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class WeightedBlend(StrategyNode):
    """Blend child signals with fixed weights.

    Final signal = sum(weight_i * signal_i) with re-normalized weights
    per bar (NaN children excluded). Continuous output discretized to
    signal_precision decimal places, then transition-only filtered.

    Parameters:
        children: List of child strategies.
        weights: Per-child weights (default: equal weight).
        signal_precision: Decimal places for rounding (default 2).
    """

    name = "Weighted Blend"

    def __init__(self, children: List[BaseStrategy],
                 weights: Optional[List[float]] = None,
                 signal_precision: int = 2):
        super().__init__(children)
        self.weights = weights or [1.0 / len(children)] * len(children)
        self.signal_precision = signal_precision

    def _compute(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.concat(
            [child._compute(df) for child in self.children], axis=1)
        signals.columns = range(len(self.children))
        w = pd.Series(self.weights, index=signals.columns)

        # Re-normalize weights per bar: exclude NaN children
        valid = signals.notna()
        w_valid = valid.mul(w, axis=1)
        w_sum = w_valid.sum(axis=1).replace(0, np.nan)
        w_norm = w_valid.div(w_sum, axis=0)

        # Weighted sum with re-normalized weights
        result = (signals.fillna(0) * w_norm.fillna(0)).sum(axis=1)
        all_nan = valid.sum(axis=1) == 0
        result[all_nan] = np.nan

        # Discretize + transition-only to prevent micro-rebalancing
        result = result.round(self.signal_precision)
        return _transitions_only(result)


class SelectBest(StrategyNode):
    """Run all children, select the one with the strongest signal.

    On each bar, the child with the largest abs(signal) wins.
    Tie-breaking: first child in list (idxmax behavior).
    """

    name = "Select Best"

    def _compute(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.concat(
            [child._compute(df) for child in self.children], axis=1)
        signals.columns = range(len(self.children))
        # Vectorized: select column with max abs value per row
        abs_signals = signals.abs()
        # Use fillna(-1) to avoid FutureWarning on all-NaN rows;
        # all-NaN rows get idxmax=0 but are excluded by valid check below
        best_col = abs_signals.fillna(-1).idxmax(axis=1)
        has_any = abs_signals.notna().any(axis=1)
        result = pd.Series(np.nan, index=df.index, dtype=float)
        for col in range(len(self.children)):
            mask = (best_col == col) & has_any
            result[mask] = signals.loc[mask, col]
        return _transitions_only(result)


class PriorityCascade(StrategyNode):
    """Try children in order. First non-NaN signal wins.

    Use case: primary strategy signals most bars; fallback strategy
    covers gaps when primary is NaN (no opinion).
    """

    name = "Priority Cascade"

    def _compute(self, df: pd.DataFrame) -> pd.Series:
        result = pd.Series(np.nan, index=df.index, dtype=float)
        for child in self.children:
            sig = child._compute(df)
            mask = result.isna() & sig.notna()
            result[mask] = sig[mask]
        return _transitions_only(result)


class VoteEnsemble(StrategyNode):
    """Majority vote: signal = sign(sum(sign(signals))).

    3 children: 2 say buy, 1 says sell -> final = buy.
    All NaN -> NaN. Tie -> NaN (no consensus).
    """

    name = "Vote Ensemble"

    def _compute(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.concat(
            [child._compute(df) for child in self.children], axis=1)
        signals.columns = range(len(self.children))
        votes = signals.apply(np.sign)
        vote_sum = votes.sum(axis=1)
        n_valid = votes.notna().sum(axis=1)

        result = pd.Series(np.nan, index=df.index, dtype=float)
        majority = vote_sum.abs() > n_valid / 2
        result[majority] = np.sign(vote_sum[majority])
        return _transitions_only(result)


class ConditionalSwitch(StrategyNode):
    """Regime-based strategy switching.

    condition_fn receives a DataFrame and returns a pd.Series of int
    indices (which child to use on each bar). Use for regime detection:
    e.g., low-vol -> trend following, high-vol -> mean reversion.

    Indices outside [0, len(children)) produce NaN (no signal) for
    those bars -- treated as "no opinion" by the engine.

    Parameters:
        children: List of child strategies.
        condition_fn: Callable(DataFrame) -> pd.Series of int indices.
    """

    name = "Conditional Switch"

    def __init__(self, children: List[BaseStrategy], condition_fn):
        super().__init__(children)
        self.condition_fn = condition_fn

    def _compute(self, df: pd.DataFrame) -> pd.Series:
        selection = self.condition_fn(df)
        child_signals = [c._compute(df) for c in self.children]
        result = pd.Series(np.nan, index=df.index, dtype=float)
        for i, sig in enumerate(child_signals):
            mask = selection == i
            result[mask] = sig[mask]
        return _transitions_only(result)
