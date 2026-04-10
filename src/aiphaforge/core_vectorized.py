"""
Vectorized Backtest Core

Standalone function that runs a vectorized (array-based) backtest.
Extracted from BacktestEngine._run_vectorized to keep engine.py thin.
"""

import pandas as pd

from .config import BacktestConfig
from .utils import extract_trades_vectorized


def run_vectorized(
    data: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig,
    symbol: str = "default",
) -> dict:
    """Run a vectorized backtest and return raw results.

    Parameters:
        data: OHLCV DataFrame (already validated and sorted).
        signals: Trading signal series aligned with *data*.
        config: Backtest configuration bundle.
        symbol: Instrument symbol.

    Returns:
        Dictionary with keys: equity_curve, trades, positions_df,
        net_returns.
    """
    # Apply signal transform if configured
    if config.signal_transform is not None:
        signals = signals.apply(
            lambda s: config.signal_transform(s) if not pd.isna(s) else s)

    # Compute positions: NaN = hold (forward-fill), 0 = flat
    positions = signals.ffill().fillna(0)

    # Clip short positions if not allowed
    if not config.allow_short:
        positions = positions.clip(lower=0)

    # Compute returns
    returns = data['close'].pct_change()

    # Strategy returns = position * returns (lagged by 1 to avoid lookahead bias)
    strategy_returns = positions.shift(1) * returns

    # Apply trade costs via module
    net_returns = config.trade_cost.apply_vectorized(
        strategy_returns, positions, data, config.fee_model, config.initial_capital,
    )

    # Apply stop loss via module
    if config.stop_loss_rule is not None:
        net_returns = config.stop_loss_rule.apply_vectorized(
            net_returns, positions, data,
        )

    # Fill NaN values (first row from pct_change and shift)
    net_returns = net_returns.fillna(0)

    # Compute equity curve
    equity_curve = config.initial_capital * (1 + net_returns).cumprod()

    # Extract trade records
    trades = extract_trades_vectorized(
        data, positions, signals, equity_curve,
        config.fee_model, config.initial_capital, symbol=symbol,
    )

    # Build positions DataFrame
    positions_df = pd.DataFrame(
        {
            'position': positions,
            'value': positions * data['close'],
            'signal': signals,
        },
        index=data.index,
    )

    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'positions_df': positions_df,
        'net_returns': net_returns,
    }
