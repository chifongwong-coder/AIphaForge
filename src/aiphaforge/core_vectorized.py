"""
Vectorized Backtest Core

Standalone function that runs a vectorized (array-based) backtest.
Extracted from BacktestEngine._run_vectorized to keep engine.py thin.
"""

import inspect

import pandas as pd

from .config import BacktestConfig
from .utils import extract_trades_vectorized


def _apply_stop_loss(rule, returns: pd.Series, positions: pd.Series,
                      data: pd.DataFrame):
    """Call ``rule.apply_vectorized`` with ``return_mask=True`` if the
    subclass supports it; otherwise fall back to the single-Series
    legacy contract.

    PercentageStopLoss (the only stop_loss_rule the engine constructs)
    supports return_mask. External BaseExitRule subclasses passed via
    BacktestConfig directly may not — falling back keeps them working
    (their stop exits stay invisible to per-trade attribution, matching
    pre-v1.9.7 behavior).
    """
    sig = inspect.signature(rule.apply_vectorized)
    if "return_mask" in sig.parameters:
        result = rule.apply_vectorized(
            returns, positions, data, return_mask=True)
        if isinstance(result, tuple) and len(result) == 4:
            net, mask, prices, threshold = result
            return net, (mask, prices, threshold)
        # Subclass overrode the signature but didn't return the tuple.
        return result, None
    # Legacy subclass — single Series return, no mask available.
    return rule.apply_vectorized(returns, positions, data), None


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

    # Apply stop loss via module. v1.9.7 commit 7b: ask for the trigger
    # mask so extract_trades_vectorized can emit stop_loss Trade entries
    # at the correct exit price (was invisible to per-trade attribution).
    # External BaseExitRule subclasses that don't accept return_mask
    # are handled gracefully — _apply_stop_loss falls back to legacy
    # single-Series return for them; their stop exits stay invisible
    # to per-trade attribution (legacy v1.9.6 behavior preserved).
    stop_loss_info = None
    if config.stop_loss_rule is not None:
        net_returns, stop_loss_info = _apply_stop_loss(
            config.stop_loss_rule, net_returns, positions, data)

    # Apply risk rules (v1.1)
    if config.risk_rules:
        temp_equity = config.initial_capital * (1 + net_returns.fillna(0)).cumprod()
        positions = config.risk_rules.apply_vectorized_all(
            temp_equity, positions, data)
        strategy_returns = positions.shift(1) * returns
        net_returns = config.trade_cost.apply_vectorized(
            strategy_returns, positions, data, config.fee_model,
            config.initial_capital,
        )
        # Re-apply stop-loss on recomputed returns (risk rules may have
        # modified positions, changing which stop-loss triggers fire).
        # v1.9.7 R2: when both calls fire, the SECOND mask wins because
        # the second call sees the post-risk-rules positions; the first
        # mask is stale.
        if config.stop_loss_rule is not None:
            net_returns, stop_loss_info = _apply_stop_loss(
                config.stop_loss_rule, net_returns, positions, data)

    # Fill NaN values (first row from pct_change and shift)
    net_returns = net_returns.fillna(0)

    # Cap per-bar return at -100% so a fee/slippage shock cannot flip the
    # cumprod sign and produce nonsensical positive equity (B6 root cause).
    net_returns = net_returns.clip(lower=-1.0)

    # Compute equity curve
    equity_curve = config.initial_capital * (1 + net_returns).cumprod()

    # Bankruptcy detection: once equity reaches 0 (or below from float
    # error), freeze the curve and stop attributing returns afterwards.
    bankrupt_mask = equity_curve <= 0
    bankruptcy_time = None
    if bankrupt_mask.any():
        bankruptcy_time = bankrupt_mask.idxmax()
        equity_curve.loc[bankruptcy_time:] = 0.0

    # Extract trade records (truncate at bankruptcy if it occurred)
    trade_data = data
    trade_positions = positions
    trade_signals = signals
    trade_equity = equity_curve
    if bankruptcy_time is not None:
        trade_data = data.loc[:bankruptcy_time]
        trade_positions = positions.loc[:bankruptcy_time]
        trade_signals = signals.loc[:bankruptcy_time]
        trade_equity = equity_curve.loc[:bankruptcy_time]
    trades = extract_trades_vectorized(
        trade_data, trade_positions, trade_signals, trade_equity,
        config.fee_model, config.initial_capital, symbol=symbol,
        stop_loss_info=stop_loss_info,
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
