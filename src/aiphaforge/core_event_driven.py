"""
Event-Driven Backtest Core

Standalone function that runs an event-driven (bar-by-bar) backtest.
Extracted from BacktestEngine._run_event_driven to keep engine.py thin.
"""

import warnings
from typing import Dict, Optional

import pandas as pd

from .broker import Broker
from .config import BacktestConfig
from .hooks import HookContext
from .portfolio import Portfolio
from .utils import calculate_returns


def run_event_driven(
    data: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig,
    symbol: str = "default",
    strategy=None,
    full_data: Optional[pd.DataFrame] = None,
) -> dict:
    """Run an event-driven backtest and return raw results.

    Parameters:
        data: OHLCV DataFrame (already validated and sorted).
        signals: Trading signal series aligned with *data*.
        config: Backtest configuration bundle.
        symbol: Instrument symbol.
        strategy: Strategy object (used to check for risk_manager attributes).
        full_data: Full dataset reference (for risk manager slicing).

    Returns:
        Dictionary with keys: equity_curve, trades, positions_df,
        orders_df, daily_returns, final_capital.
    """
    # Initialize components
    portfolio = Portfolio(
        initial_capital=config.initial_capital,
        max_position_size=config.max_position_size,
        allow_short=config.allow_short,
        fee_allocation=config.fee_allocation,
    )

    broker = Broker(
        fee_model=config.fee_model,
        fill_model=config.fill_model,
        session_end_time=config.session_end_time,
    )
    broker.set_portfolio(portfolio)

    # Notify hooks: backtest start
    for hook in config.hooks:
        hook.on_backtest_start(data, symbol, config=config)

    # Process each bar
    for i, (timestamp, bar) in enumerate(data.iterrows()):
        # 1. Update prices (don't record equity yet)
        portfolio.update_prices({symbol: bar['close']}, timestamp, record=False)

        # 2. Process pending orders
        filled_orders = broker.process_bar(bar, timestamp, symbol)
        for order in filled_orders:
            portfolio.update_from_order(order, timestamp)

        # 3. Check stop loss / take profit
        if config.stop_loss_rule is not None:
            config.stop_loss_rule.check_event_driven(
                portfolio, broker, symbol, bar, timestamp,
            )

        if config.take_profit_rule is not None:
            config.take_profit_rule.check_event_driven(
                portfolio, broker, symbol, bar, timestamp,
            )

        # 4. Call hooks: on_pre_signal (before signal processing)
        pending_before_hooks = (
            len(broker.get_pending_orders(symbol)) if config.hooks else 0
        )
        if config.hooks:
            ctx = HookContext(
                bar_index=i,
                timestamp=timestamp,
                bar_data=bar,
                data=data.iloc[:i + 1],
                portfolio=portfolio,
                symbol=symbol,
                broker=broker,
            )
            for hook in config.hooks:
                hook.on_pre_signal(ctx)

        # 5. Process new signals
        signal = signals.iloc[i] if i < len(signals) else 0

        # Warn if hooks submitted orders AND signal is non-zero
        if config.hooks and signal != 0:
            pending_now = len(broker.get_pending_orders(symbol))
            if pending_now > pending_before_hooks:
                warnings.warn(
                    f"Bar {i}: hook submitted orders while signal={signal}. "
                    f"Both will execute. Set signals to 0 if hooks manage orders."
                )

        if signal != 0:
            _process_signal(
                signal, portfolio, broker, symbol, bar, timestamp,
                config=config,
                bar_index=i,
                full_data=full_data,
            )

        # 6. Call hooks: on_bar (after signal processing)
        if config.hooks:
            for hook in config.hooks:
                hook.on_bar(ctx)

        # 7. Record equity AFTER all position changes
        portfolio._record_equity(timestamp)

    # Notify hooks: backtest end
    for hook in config.hooks:
        hook.on_backtest_end()

    # Build results
    equity_curve = portfolio.get_equity_curve()
    trades = portfolio.trade_history
    positions_df = portfolio.get_positions_df()
    orders_df = broker.get_orders_df()

    # Compute daily returns
    daily_returns = None
    if len(equity_curve) > 0:
        daily_returns = calculate_returns(equity_curve)

    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'positions_df': positions_df,
        'orders_df': orders_df,
        'daily_returns': daily_returns,
        'final_capital': portfolio.total_equity,
    }


def _process_signal(
    signal: int,
    portfolio: Portfolio,
    broker: Broker,
    symbol: str,
    bar: pd.Series,
    timestamp: pd.Timestamp,
    config: BacktestConfig,
    bar_index: Optional[int] = None,
    full_data: Optional[pd.DataFrame] = None,
):
    """Process a single trading signal."""
    current_pos = portfolio.get_position_size(symbol)

    # Build sliced market data visible up to the current bar
    if bar_index is not None and full_data is not None:
        sliced_data = full_data.iloc[:bar_index + 1]
        market_data_dict: Dict[str, pd.DataFrame] = {symbol: sliced_data}
    else:
        sliced_data = full_data
        market_data_dict = (
            {symbol: full_data} if full_data is not None else {}
        )

    # Risk manager check (if available)
    if config.risk_manager:
        try:
            config.risk_manager.sync_from_portfolio(portfolio)

            risk_signals = config.risk_manager.check_and_apply_risk_rules(
                portfolio, market_data_dict,
            )

            for risk_signal in risk_signals:
                if (
                    risk_signal.severity == 'critical'
                    and risk_signal.action in ['reject_new', 'close_all']
                ):
                    warnings.warn(
                        f"Risk limit triggered: {risk_signal.message}"
                    )
                    return
        except Exception as e:
            warnings.warn(f"Risk check failed: {e}")

    # Calculate target position
    if signal == 1:  # Buy signal
        target_pos = _calculate_target_position(
            portfolio, bar['close'], 1, symbol, config,
            market_data=sliced_data,
        )
    elif signal == -1:  # Sell signal
        if config.allow_short:
            target_pos = _calculate_target_position(
                portfolio, bar['close'], -1, symbol, config,
                market_data=sliced_data,
            )
        else:
            target_pos = 0
    else:
        return

    # Calculate required trade
    size_change = target_pos - current_pos

    if abs(size_change) < 0.001:  # Ignore tiny changes
        return

    # Create order
    if size_change > 0:
        order = broker.create_market_order(
            symbol, "buy", size_change, "signal", timestamp,
        )
    else:
        order = broker.create_market_order(
            symbol, "sell", abs(size_change), "signal", timestamp,
        )

    broker.submit_order(order, timestamp)


def _calculate_target_position(
    portfolio: Portfolio,
    price: float,
    direction: int,
    symbol: str,
    config: BacktestConfig,
    market_data: Optional[pd.DataFrame] = None,
) -> float:
    """Calculate target position size."""
    if price <= 0:
        return 0

    equity = portfolio.total_equity

    # Use risk manager if available (pass sliced data to avoid look-ahead)
    data_for_rm = market_data
    if (
        config.risk_manager
        and data_for_rm is not None
        and hasattr(data_for_rm, 'index')
    ):
        try:
            quantity = config.risk_manager.calculate_position_size(
                symbol=symbol,
                signal=direction,
                current_price=price,
                market_data=data_for_rm,
            )
            return quantity
        except Exception as e:
            warnings.warn(
                f"Risk manager sizing failed: {e}, using default method"
            )

    # Delegate to position sizer module
    return config.position_sizer.calculate(
        equity, price, direction, config.max_position_size,
    )
