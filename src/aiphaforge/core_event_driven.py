"""
Event-Driven Backtest Core
==========================

Unified function that runs an event-driven (bar-by-bar) backtest for
both single-asset and multi-asset modes.  The engine always passes
dict-based data; single-asset input is wrapped before calling.
"""

import warnings
from typing import Dict, List, Optional

import pandas as pd

from .broker import Broker
from .config import BacktestConfig, resolve_config
from .hooks import HookContext
from .portfolio import Portfolio
from .utils import build_unified_timeline, calculate_returns

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_event_driven(
    data_dict: Dict[str, pd.DataFrame],
    signals_dict: Dict[str, pd.Series],
    config: BacktestConfig,
    symbols: List[str],
    strategy=None,
) -> dict:
    """Run an event-driven backtest and return raw results.

    Parameters:
        data_dict: Per-symbol OHLCV DataFrames (validated and sorted).
        signals_dict: Per-symbol trading signal Series.
        config: Backtest configuration bundle.
        symbols: Ordered list of symbol names.
        strategy: Strategy object (for risk_manager attributes).

    Returns:
        Dictionary with keys: equity_curve, trades, positions_df,
        orders_df, daily_returns, final_capital, and optionally
        per_asset_pnl.
    """
    if not symbols:
        raise ValueError("No symbols provided (empty data_dict)")

    # Ensure signals_dict covers all symbols
    for sym in symbols:
        if sym not in signals_dict:
            signals_dict[sym] = pd.Series(dtype=float)

    is_single = len(symbols) == 1

    # --- Initialize portfolio ---
    portfolio = Portfolio(
        initial_capital=config.initial_capital,
        max_position_size=config.max_position_size,
        allow_short=config.allow_short,
        fee_allocation=config.fee_allocation,
        margin_config=config.margin_config,
    )

    # --- Initialize per-symbol brokers ---
    brokers: Dict[str, Broker] = {}
    for symbol in sorted(symbols):
        brokers[symbol] = Broker(
            fee_model=resolve_config(
                config.fee_model, config.asset_fee_models, symbol),
            fill_model=resolve_config(
                config.fill_model, config.asset_fill_models, symbol),
            session_end_time=config.session_end_time,
            immediate_fill_price=config.immediate_fill_price,
            assigned_symbol=symbol,
        )
        brokers[symbol].set_portfolio(portfolio)

    # --- Build unified timeline ---
    timeline, bar_avail = build_unified_timeline(data_dict)

    # --- Convert signal Series to dicts for O(1) lookup ---
    signals_as_dict: Dict[str, dict] = {
        sym: sig.to_dict() for sym, sig in signals_dict.items()
    }

    # --- Validate signal / data overlap ---
    for sym in symbols:
        sig_idx = signals_dict[sym].index
        data_idx = data_dict[sym].index
        if len(sig_idx.intersection(data_idx)) == 0 and len(sig_idx) > 0:
            warnings.warn(
                f"Zero overlap between signal index and data index "
                f"for symbol '{sym}'."
            )

    # --- Initialize last known prices ---
    last_known: Dict[str, float] = {}
    for sym in symbols:
        df = data_dict[sym]
        if len(df) > 0:
            last_known[sym] = float(df.iloc[0]['close'])
        else:
            last_known[sym] = 0.0

    # --- Per-asset PnL tracker (multi-asset only) ---
    # NOTE: per_asset_pnl tracks GROSS PnL (before fees/slippage).
    # This is intentional for attribution: each asset's raw P&L
    # contribution.  sum(per_asset_pnl) will exceed portfolio equity
    # change by the total fees paid.  Portfolio equity curve is net.
    pnl_tracker: Optional[dict] = None
    if not is_single:
        pnl_tracker = {
            'series': {sym: [] for sym in symbols},
            'prev_unrealized': {sym: 0.0 for sym in symbols},
            'bar_realized': {sym: 0.0 for sym in symbols},
        }

    # --- Notify hooks: backtest start (per symbol) ---
    for sym in sorted(symbols):
        for hook in config.hooks:
            hook.on_backtest_start(data_dict[sym], sym, config=config)

    # --- Build exit rules list ---
    exit_rules = [r for r in [config.stop_loss_rule,
                               config.take_profit_rule] if r is not None]

    # ===================================================================
    # Main event loop
    # ===================================================================
    ctx = None  # HookContext, built per-bar when hooks are present
    for idx, timestamp in enumerate(timeline):
        active = sorted(
            [s for s in symbols if timestamp in bar_avail[s]]
        )

        # 1. Update prices (forward-fill for inactive assets)
        prices: Dict[str, float] = {}
        for sym in symbols:
            if sym in active:
                prices[sym] = float(
                    data_dict[sym].loc[timestamp, 'close'])
                last_known[sym] = prices[sym]
            else:
                prices[sym] = last_known[sym]
        portfolio.update_prices(prices, timestamp, record=False)

        # Reset per-bar realized PnL tracker
        if pnl_tracker is not None:
            for sym in symbols:
                pnl_tracker['bar_realized'][sym] = 0.0

        # 2. Process pending orders (per active symbol)
        for sym in active:
            bar = data_dict[sym].loc[timestamp]
            filled = brokers[sym].process_bar(bar, timestamp, sym)
            for order in filled:
                trade = portfolio.update_from_order(order, timestamp)
                if trade and pnl_tracker is not None:
                    pnl_tracker['bar_realized'][sym] = (
                        pnl_tracker['bar_realized'].get(sym, 0.0)
                        + trade.pnl + trade.commission
                        + trade.slippage_cost
                    )

        # 3. Exit rules (per active symbol)
        for sym in active:
            bar = data_dict[sym].loc[timestamp]
            for rule in exit_rules:
                rule.check_event_driven(
                    portfolio, brokers[sym], sym, bar, timestamp)

        # 3b. Portfolio-level exit rules (margin call, etc.)
        if config.portfolio_exit_rules:
            for rule in config.portfolio_exit_rules:
                rule.check_portfolio(
                    portfolio, brokers, symbols, prices, timestamp)

        # 4. Hooks: on_pre_signal
        pending_before_hooks: Dict[str, int] = {}
        if config.hooks:
            for sym in active:
                pending_before_hooks[sym] = len(
                    brokers[sym].get_pending_orders(sym))

            if is_single:
                sym0 = symbols[0]
                ctx = HookContext(
                    bar_index=idx,
                    timestamp=timestamp,
                    portfolio=portfolio,
                    bar_data=data_dict[sym0].loc[timestamp],
                    data=data_dict[sym0].loc[:timestamp],
                    symbol=sym0,
                    broker=brokers[sym0],
                )
            else:
                ctx = HookContext(
                    bar_index=idx,
                    timestamp=timestamp,
                    portfolio=portfolio,
                    active_symbols=active,
                    all_bar_data={s: data_dict[s].loc[timestamp]
                                  for s in active},
                    all_data={s: data_dict[s].loc[:timestamp]
                              for s in symbols},
                    all_brokers=brokers,
                )
            for hook in config.hooks:
                hook.on_pre_signal(ctx)

        # 4b. Process immediate IOC/FOK from hooks
        for sym in active:
            bar = data_dict[sym].loc[timestamp]
            immediate = brokers[sym].process_immediate_orders(
                bar, timestamp, sym)
            for order in immediate:
                trade = portfolio.update_from_order(order, timestamp)
                if trade and pnl_tracker is not None:
                    pnl_tracker['bar_realized'][sym] = (
                        pnl_tracker['bar_realized'].get(sym, 0.0)
                        + trade.pnl + trade.commission
                        + trade.slippage_cost
                    )

        # 5. Two-phase signal processing
        # Phase A: collect signals (NaN = hold → skip; 0 = flat → include)
        current_signals: Dict[str, float] = {}
        for sym in active:
            raw_sig = signals_as_dict[sym].get(timestamp, float('nan'))
            if pd.isna(raw_sig):
                continue
            # Apply signal transform if configured
            if config.signal_transform is not None:
                raw_sig = config.signal_transform(raw_sig)
            current_signals[sym] = raw_sig

        # Warn if hooks submitted orders AND signals are active
        if config.hooks and current_signals:
            for sym, sig in current_signals.items():
                if sym in pending_before_hooks:
                    pending_now = len(
                        brokers[sym].get_pending_orders(sym))
                    if pending_now > pending_before_hooks.get(sym, 0):
                        warnings.warn(
                            f"Bar {idx}: hook submitted orders for "
                            f"'{sym}' while signal={sig}. Both will "
                            f"execute. Set signals to NaN if hooks "
                            f"manage orders."
                        )

        # Phase B: allocate then execute
        budgets: Dict[str, Optional[float]] = {}
        if current_signals and config.capital_allocator is not None:
            budgets = config.capital_allocator.allocate(
                current_signals, prices, portfolio, config)

        for sym, sig in sorted(current_signals.items()):
            bar = data_dict[sym].loc[timestamp]
            _process_signal(
                sig, portfolio, brokers[sym], sym, bar, timestamp,
                config=config,
                bar_index=idx,
                full_data=data_dict[sym],
                budget=budgets.get(sym),
            )

        # 5b. Process immediate IOC/FOK from signals
        for sym in active:
            bar = data_dict[sym].loc[timestamp]
            immediate = brokers[sym].process_immediate_orders(
                bar, timestamp, sym)
            for order in immediate:
                trade = portfolio.update_from_order(order, timestamp)
                if trade and pnl_tracker is not None:
                    pnl_tracker['bar_realized'][sym] = (
                        pnl_tracker['bar_realized'].get(sym, 0.0)
                        + trade.pnl + trade.commission
                        + trade.slippage_cost
                    )

        # 6. Hooks: on_bar
        if config.hooks and ctx is not None:
            for hook in config.hooks:
                hook.on_bar(ctx)

        # 6.5. Periodic costs (borrowing, funding)
        if config.periodic_cost_model is not None:
            for sym in symbols:
                pos = portfolio.positions.get(sym)
                if pos and not pos.is_flat:
                    mc = resolve_config(
                        config.margin_config,
                        config.asset_margin_configs, sym)
                    cost = config.periodic_cost_model.calculate_cost(
                        pos, prices[sym], timestamp, mc)
                    if cost > 0:
                        portfolio.deduct_cost(cost)

        # 7. Record equity + per-asset PnL
        portfolio._record_equity(timestamp)
        if pnl_tracker is not None:
            _record_per_asset_pnl(
                timestamp, portfolio, symbols, pnl_tracker)

        # 7b. Negative equity termination (gap risk)
        if (config.margin_config is not None
                and config.margin_config.stop_on_negative_equity
                and portfolio.total_equity < 0):
            break

    # --- Post-loop: notify hooks ---
    for hook in config.hooks:
        hook.on_backtest_end()

    # --- Build results ---
    equity_curve = portfolio.get_equity_curve()
    trades = portfolio.trade_history
    positions_df = portfolio.get_positions_df()

    # Merge order DataFrames from all brokers
    orders_dfs = [b.get_orders_df() for b in brokers.values()]
    orders_df = (pd.concat(orders_dfs, ignore_index=True)
                 if orders_dfs else pd.DataFrame())

    daily_returns = None
    if len(equity_curve) > 0:
        daily_returns = calculate_returns(equity_curve)

    result = {
        'equity_curve': equity_curve,
        'trades': trades,
        'positions_df': positions_df,
        'orders_df': orders_df,
        'daily_returns': daily_returns,
        'final_capital': portfolio.total_equity,
    }

    # Per-asset PnL (multi-asset only)
    if pnl_tracker is not None:
        per_asset_pnl = {}
        for sym in symbols:
            entries = pnl_tracker['series'][sym]
            if entries:
                idx_list, vals = zip(*entries)
                per_asset_pnl[sym] = pd.Series(
                    vals, index=pd.DatetimeIndex(idx_list), name=sym)
            else:
                per_asset_pnl[sym] = pd.Series(
                    dtype=float, name=sym)
        result['per_asset_pnl'] = per_asset_pnl

    return result


# ---------------------------------------------------------------------------
# Per-asset PnL tracking
# ---------------------------------------------------------------------------

def _record_per_asset_pnl(
    timestamp: pd.Timestamp,
    portfolio: Portfolio,
    symbols: List[str],
    tracker: dict,
) -> None:
    """Track per-symbol GROSS PnL contribution each bar.

    Gross = before fees/slippage.  This is standard for per-asset
    attribution in shared-capital portfolios.  The sum of all assets'
    PnL will exceed the portfolio equity change by total fees paid.
    """
    for sym in symbols:
        pos = portfolio.positions.get(sym)
        current_unrealized = pos.unrealized_pnl if pos else 0.0
        prev_unrealized = tracker['prev_unrealized'].get(sym, 0.0)
        realized_this_bar = tracker['bar_realized'].get(sym, 0.0)
        delta = (current_unrealized - prev_unrealized) + realized_this_bar
        tracker['series'][sym].append((timestamp, delta))
        tracker['prev_unrealized'][sym] = current_unrealized


# ---------------------------------------------------------------------------
# Signal processing (largely unchanged from v0.6)
# ---------------------------------------------------------------------------

def _process_signal(
    signal: float,
    portfolio: Portfolio,
    broker: Broker,
    symbol: str,
    bar: pd.Series,
    timestamp: pd.Timestamp,
    config: BacktestConfig,
    bar_index: Optional[int] = None,
    full_data: Optional[pd.DataFrame] = None,
    budget: Optional[float] = None,
):
    """Process a single trading signal.

    Signal semantics (v0.9):
    - NaN: hold (filtered before reaching here)
    - 0: go flat (close position)
    - positive: go long (value = exposure fraction of max target)
    - negative: go short (value = exposure fraction of max target)
    """
    if pd.isna(signal):
        return  # NaN = hold

    price = bar['close']
    current_pos = portfolio.get_position_size(symbol)

    # --- signal=0: go flat (close position) ---
    if abs(signal) < 1e-8:
        # Skip if margin call in progress (liquidation already pending)
        if portfolio.is_margin_call:
            return
        if abs(current_pos) < 0.001:
            return  # already flat
        side = "sell" if current_pos > 0 else "buy"
        order = broker.create_market_order(
            symbol, side, abs(current_pos), "signal_flat", timestamp)
        order.metadata['estimated_price'] = price
        broker.submit_order(order, timestamp)
        return

    # --- Directional signal ---
    direction = 1 if signal > 0 else -1
    fraction = abs(signal)

    # allow_short guard
    if direction == -1 and not config.allow_short:
        warnings.warn(
            f"Short signal for '{symbol}' ignored (allow_short=False)")
        return

    # Build sliced market data visible up to the current bar
    if bar_index is not None and full_data is not None:
        sliced_data = full_data.iloc[:bar_index + 1]
        market_data_dict: Dict[str, pd.DataFrame] = {symbol: sliced_data}
    else:
        sliced_data = full_data
        market_data_dict = (
            {symbol: full_data} if full_data is not None else {}
        )

    # Risk manager check (preserved from v0.8)
    if config.risk_manager:
        try:
            config.risk_manager.sync_from_portfolio(portfolio)
            risk_signals = config.risk_manager.check_and_apply_risk_rules(
                portfolio, market_data_dict)
            for risk_signal in risk_signals:
                if (
                    risk_signal.severity == 'critical'
                    and risk_signal.action in ['reject_new', 'close_all']
                ):
                    warnings.warn(
                        f"Risk limit triggered: {risk_signal.message}")
                    return
        except Exception as e:
            warnings.warn(f"Risk check failed: {e}")

    # Calculate max target position for this direction
    max_target = _calculate_target_position(
        portfolio, price, direction, symbol, config,
        market_data=sliced_data,
    )

    # Scale by signal fraction (continuous signal support)
    target_pos = max_target * fraction

    # Calculate required trade
    size_change = target_pos - current_pos

    # Per-asset position limit (cap total position to equity * max_pct)
    max_pct = resolve_config(
        config.max_position_pct, config.asset_max_position_pcts, symbol)
    if max_pct < 1.0 and price > 0:
        max_pos = portfolio.total_equity * max_pct / price
        new_pos = current_pos + size_change
        if abs(new_pos) > max_pos:
            capped_pos = max_pos * (1 if new_pos > 0 else -1)
            size_change = capped_pos - current_pos

    # Budget cap on incremental cost (direction-agnostic)
    if budget is not None and price > 0:
        max_size = budget / price
        if abs(size_change) > max_size:
            size_change = max_size * (1 if size_change > 0 else -1)

    # Lot-size rounding (e.g., A-share 100-share lots)
    lot = resolve_config(config.lot_size, config.asset_lot_sizes, symbol)
    if lot > 1 and size_change != 0:
        sign = 1 if size_change > 0 else -1
        size_change = (int(abs(size_change)) // lot) * lot * sign

    if abs(size_change) < 0.001:  # Ignore tiny changes
        return

    # Create order
    if size_change > 0:
        order = broker.create_market_order(
            symbol, "buy", size_change, "signal", timestamp)
    else:
        order = broker.create_market_order(
            symbol, "sell", abs(size_change), "signal", timestamp)

    order.metadata['estimated_price'] = price
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
