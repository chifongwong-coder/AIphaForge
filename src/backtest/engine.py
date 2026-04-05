"""
Backtest Engine

Main backtest executor supporting both vectorized and event-driven modes.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union, Callable, Any
from enum import Enum
import warnings

from .fees import BaseFeeModel, SimpleFeeModel, get_fee_model
from .hooks import BacktestHook, HookContext
from .orders import Order, OrderSide, OrderType
from .portfolio import Portfolio
from .broker import Broker, FillModel, SimpleBroker
from .results import BacktestResult, Trade, trades_to_dataframe

# Import utility functions
from ..utils import (
    TRADING_DAYS_STOCK,
    validate_ohlcv,
    ensure_datetime_index,
    calculate_returns,
    sharpe_ratio as calc_sharpe,
    sortino_ratio as calc_sortino,
    max_drawdown as calc_max_drawdown,
    annualize_return,
)


class ExecutionMode(Enum):
    """Execution mode."""
    VECTORIZED = "vectorized"        # Vectorized mode (fast)
    EVENT_DRIVEN = "event_driven"    # Event-driven mode (precise)


class PositionSizing(Enum):
    """Position sizing method."""
    FIXED_FRACTION = "fixed_fraction"  # Fixed fraction of equity
    FIXED_SIZE = "fixed_size"          # Fixed quantity
    ALL_IN = "all_in"                  # Full position
    RISK_BASED = "risk_based"          # Risk-based sizing


class BacktestEngine:
    """
    Backtest engine supporting vectorized and event-driven execution modes.

    Parameters:
        fee_model: Fee model instance.
        initial_capital: Starting capital.
        mode: Execution mode.
        position_sizing: Position sizing method.
        position_size: Position size (fraction or fixed quantity).
        max_position_size: Maximum single position as fraction of equity.
        stop_loss: Stop loss percentage.
        take_profit: Take profit percentage.
        allow_short: Whether short selling is allowed.
        fill_model: Fill model (event-driven mode).
        risk_manager: External risk manager (optional).
        agent_expert: AI Agent expert (optional).
        agent_trigger_interval: Agent trigger interval (every N bars).
        agent_enabled_strategies: Agent-controlled strategy enable states.
        hooks: List of backtest hooks (optional).
        include_benchmark: Whether to compute buy-and-hold benchmark.

    Example:
        >>> engine = BacktestEngine(
        ...     fee_model=ChinaAShareFeeModel(),
        ...     initial_capital=100000,
        ...     stop_loss=0.05
        ... )
        >>> engine.set_strategy(my_strategy)
        >>> results = engine.run(data)
        >>> print(results.summary())
    """

    def __init__(
        self,
        fee_model: Optional[BaseFeeModel] = None,
        initial_capital: float = 100000,
        mode: Union[str, ExecutionMode] = ExecutionMode.VECTORIZED,
        position_sizing: Union[str, PositionSizing] = PositionSizing.FIXED_FRACTION,
        position_size: float = 0.95,
        max_position_size: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        allow_short: bool = True,
        fill_model: FillModel = FillModel.NEXT_BAR_OPEN,
        risk_manager=None,
        agent_expert=None,
        agent_trigger_interval: int = 1,
        agent_enabled_strategies: Optional[Dict[str, bool]] = None,
        hooks: Optional[List[BacktestHook]] = None,
        include_benchmark: bool = True,
    ):
        # Fee model
        self.fee_model = fee_model or SimpleFeeModel()

        # Capital
        self.initial_capital = initial_capital

        # Execution mode
        if isinstance(mode, str):
            mode = ExecutionMode(mode.lower())
        self.mode = mode

        # Position sizing
        if isinstance(position_sizing, str):
            position_sizing = PositionSizing(position_sizing.lower())
        self.position_sizing = position_sizing
        self.position_size = position_size
        self.max_position_size = max_position_size

        # Risk management
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.allow_short = allow_short

        # Fill model
        self.fill_model = fill_model

        # Risk manager (optional)
        self.risk_manager = risk_manager
        if risk_manager:
            risk_manager.initialize(initial_capital)

        # AI Agent (optional)
        self.agent_expert = agent_expert
        self.agent_trigger_interval = agent_trigger_interval
        self.agent_enabled_strategies = agent_enabled_strategies or {}
        self._agent_bar_count = 0

        # Hooks (optional)
        self.hooks: List[BacktestHook] = hooks or []

        # Benchmark
        self.include_benchmark = include_benchmark

        # Internal state
        self._strategy = None
        self._signals = None
        self._data = None

    # ========== Setup Methods ==========

    def set_strategy(self, strategy) -> 'BacktestEngine':
        """
        Set the trading strategy.

        Parameters:
            strategy: Strategy object with a ``generate_signals`` method.

        Returns:
            self: For method chaining.
        """
        self._strategy = strategy
        self._signals = None
        return self

    def set_signals(self, signals: pd.Series) -> 'BacktestEngine':
        """
        Set pre-computed trading signals directly.

        Parameters:
            signals: Signal series (index=time, values={1, -1, 0}).

        Returns:
            self: For method chaining.
        """
        self._signals = signals
        self._strategy = None
        return self

    def set_fee_model(self, fee_model: Union[BaseFeeModel, str]) -> 'BacktestEngine':
        """
        Set the fee model.

        Parameters:
            fee_model: Fee model instance or market name string.

        Returns:
            self: For method chaining.
        """
        if isinstance(fee_model, str):
            self.fee_model = get_fee_model(fee_model)
        else:
            self.fee_model = fee_model
        return self

    # ========== Run Methods ==========

    def run(
        self,
        data: pd.DataFrame,
        start: Optional[str] = None,
        end: Optional[str] = None,
        symbol: str = "default"
    ) -> BacktestResult:
        """
        Run the backtest.

        Parameters:
            data: OHLCV data with open, high, low, close, volume columns.
            start: Start date (optional).
            end: End date (optional).
            symbol: Instrument symbol.

        Returns:
            BacktestResult: Backtest results.
        """
        # Validate and prepare data
        data = self._prepare_data(data, start, end)
        self._data = data

        # Generate signals
        signals = self._get_signals(data)

        # Run based on mode
        if self.mode == ExecutionMode.VECTORIZED:
            return self._run_vectorized(data, signals, symbol)
        else:
            return self._run_event_driven(data, signals, symbol)

    def _prepare_data(
        self,
        data: pd.DataFrame,
        start: Optional[str],
        end: Optional[str]
    ) -> pd.DataFrame:
        """Validate and prepare data."""
        validate_ohlcv(data, required=['open', 'high', 'low', 'close'])
        data = ensure_datetime_index(data)
        data = data.sort_index()

        if start:
            data = data[data.index >= pd.Timestamp(start)]
        if end:
            data = data[data.index <= pd.Timestamp(end)]

        if len(data) == 0:
            raise ValueError("No data after date filtering")

        return data.copy()

    def _get_signals(self, data: pd.DataFrame) -> pd.Series:
        """Get trading signals."""
        if self._signals is not None:
            signals = self._signals.reindex(data.index).fillna(0)
        elif self._strategy is not None:
            signals = self._strategy.generate_signals(data)
        else:
            raise ValueError("Must set either a strategy or signals")

        return signals

    # ========== Vectorized Backtest ==========

    def _run_vectorized(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        symbol: str
    ) -> BacktestResult:
        """
        Vectorized backtest.

        Fast but simplified backtest mode.
        """
        # Compute positions (forward-fill signals)
        positions = signals.replace(0, np.nan).ffill().fillna(0)

        # Clip short positions if not allowed
        if not self.allow_short:
            positions = positions.clip(lower=0)

        # Compute returns
        returns = data['close'].pct_change()

        # Strategy returns = position * returns (lagged by 1 to avoid lookahead bias)
        strategy_returns = positions.shift(1) * returns

        # Compute trade signals (position changes)
        trades_signal = positions.diff().abs()
        trade_days = trades_signal > 0

        # Compute transaction costs (simplified: fixed rate)
        if hasattr(self.fee_model, 'commission_rate'):
            commission_rate = self.fee_model.commission_rate
        else:
            commission_rate = 0.001

        slippage_rate = self.fee_model.slippage_pct if hasattr(self.fee_model, 'slippage_pct') else 0.001

        # Transaction costs (applied on trade days, both sides)
        trade_cost = trade_days * (commission_rate + slippage_rate) * 2

        # Net returns
        net_returns = strategy_returns - trade_cost

        # Apply stop loss
        if self.stop_loss is not None:
            net_returns = self._apply_vectorized_stop_loss(
                net_returns, positions, data, self.stop_loss
            )

        # Fill NaN values (first row from pct_change and shift)
        net_returns = net_returns.fillna(0)

        # Compute equity curve
        equity_curve = self.initial_capital * (1 + net_returns).cumprod()

        # Extract trade records
        trades = self._extract_trades_vectorized(data, positions, signals, equity_curve)

        # Compute performance metrics
        metrics = self._calculate_metrics(net_returns, equity_curve, trades)

        # Build positions DataFrame
        positions_df = pd.DataFrame({
            'position': positions,
            'value': positions * data['close'],
            'signal': signals
        }, index=data.index)

        # Strategy name
        strategy_name = self._strategy.name if self._strategy else "Custom"

        # Buy-and-hold benchmark
        benchmark_equity = None
        benchmark_metrics = None
        if self.include_benchmark:
            benchmark_equity, benchmark_metrics = self._compute_benchmark(data)

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            positions=positions_df,
            metrics=metrics,
            initial_capital=self.initial_capital,
            strategy_name=strategy_name,
            parameters=self._strategy.params if self._strategy else {},
            daily_returns=net_returns,
            benchmark_equity=benchmark_equity,
            benchmark_metrics=benchmark_metrics,
        )

    def _apply_vectorized_stop_loss(
        self,
        returns: pd.Series,
        positions: pd.Series,
        data: pd.DataFrame,
        stop_loss: float
    ) -> pd.Series:
        """Simplified vectorized stop loss."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        # Zero out returns when stop loss is triggered
        stop_triggered = drawdown < -stop_loss
        returns_with_stop = returns.copy()
        returns_with_stop[stop_triggered] = 0

        return returns_with_stop

    def _extract_trades_vectorized(
        self,
        data: pd.DataFrame,
        positions: pd.Series,
        signals: pd.Series,
        equity: pd.Series
    ) -> List[Trade]:
        """Extract trade records from vectorized results."""
        trades = []

        # Find position change points
        pos_diff = positions.diff()
        entries = pos_diff[pos_diff != 0].dropna()

        entry_time = None
        entry_price = None
        entry_direction = None
        entry_size = None
        trade_id = 0

        for idx, change in entries.items():
            price = data.loc[idx, 'close']

            if entry_time is None:
                # Open position
                if change != 0:
                    entry_time = idx
                    entry_price = price
                    entry_direction = 1 if change > 0 else -1
                    entry_size = abs(change)
            else:
                # Check if closing
                new_pos = positions.loc[idx]
                if new_pos == 0 or (entry_direction > 0 and change < 0) or (entry_direction < 0 and change > 0):
                    # Close position
                    trade_id += 1
                    pnl = entry_direction * (price - entry_price) * entry_size * self.initial_capital / entry_price

                    trades.append(Trade(
                        trade_id=f"VT{trade_id:04d}",
                        symbol="default",
                        direction=entry_direction,
                        entry_time=entry_time,
                        exit_time=idx,
                        entry_price=entry_price,
                        exit_price=price,
                        size=entry_size,
                        pnl=pnl,
                        pnl_pct=(price / entry_price - 1) * entry_direction,
                        reason="signal"
                    ))

                    # If reversing position
                    if new_pos != 0:
                        entry_time = idx
                        entry_price = price
                        entry_direction = 1 if new_pos > 0 else -1
                        entry_size = abs(new_pos)
                    else:
                        entry_time = None

        return trades

    # ========== Event-Driven Backtest ==========

    def _run_event_driven(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        symbol: str
    ) -> BacktestResult:
        """
        Event-driven backtest.

        Processes each bar sequentially for precise simulation.
        """
        # Initialize components
        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            max_position_size=self.max_position_size,
            allow_short=self.allow_short
        )

        broker = Broker(
            fee_model=self.fee_model,
            fill_model=self.fill_model
        )
        broker.set_portfolio(portfolio)

        # Notify hooks: backtest start
        for hook in self.hooks:
            hook.on_backtest_start(data, symbol)

        # Process each bar
        for i, (timestamp, bar) in enumerate(data.iterrows()):
            # 1. Update prices
            portfolio.update_prices({symbol: bar['close']}, timestamp)

            # 2. Process pending orders
            filled_orders = broker.process_bar(bar, timestamp, symbol)
            for order in filled_orders:
                portfolio.update_from_order(order, timestamp)

            # 3. Check stop loss / take profit
            if self.stop_loss is not None:
                self._check_and_execute_stop_loss(
                    portfolio, broker, symbol, bar, timestamp
                )

            if self.take_profit is not None:
                self._check_and_execute_take_profit(
                    portfolio, broker, symbol, bar, timestamp
                )

            # 4. Process new signals
            signal = signals.iloc[i] if i < len(signals) else 0

            if signal != 0:
                self._process_signal(
                    signal, portfolio, broker, symbol, bar, timestamp
                )

            # 5. Call hooks (no-op when hooks list is empty)
            if self.hooks:
                ctx = HookContext(
                    bar_index=i,
                    timestamp=timestamp,
                    bar_data=bar,
                    data=data.iloc[:i + 1],
                    portfolio=portfolio,
                    symbol=symbol
                )
                for hook in self.hooks:
                    hook.on_bar(ctx)

        # Notify hooks: backtest end
        for hook in self.hooks:
            hook.on_backtest_end()

        # Build results
        equity_curve = portfolio.get_equity_curve()
        trades = portfolio.trade_history
        positions_df = portfolio.get_positions_df()
        orders_df = broker.get_orders_df()

        # Compute metrics
        daily_returns = None
        if len(equity_curve) > 0:
            returns = calculate_returns(equity_curve)
            metrics = self._calculate_metrics(returns, equity_curve, trades)
            daily_returns = returns
        else:
            metrics = {}

        strategy_name = self._strategy.name if self._strategy else "Custom"

        # Buy-and-hold benchmark
        benchmark_equity = None
        benchmark_metrics = None
        if self.include_benchmark:
            benchmark_equity, benchmark_metrics = self._compute_benchmark(data)

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            positions=positions_df,
            metrics=metrics,
            orders=orders_df,
            initial_capital=self.initial_capital,
            final_capital=portfolio.total_equity,
            strategy_name=strategy_name,
            parameters=self._strategy.params if self._strategy else {},
            daily_returns=daily_returns,
            benchmark_equity=benchmark_equity,
            benchmark_metrics=benchmark_metrics,
        )

    def _maybe_call_agent(
        self,
        data: pd.DataFrame,
        current_idx: int,
        portfolio: Portfolio,
        symbol: str,
        interval: str = "daily"
    ) -> Optional[Dict]:
        """
        Optionally call the AI Agent (if enabled).

        Returns:
            Optional[Dict]: Agent decision result including strategy_changes.
        """
        if self.agent_expert is None:
            return None

        # Check trigger interval
        self._agent_bar_count += 1
        if self._agent_bar_count % self.agent_trigger_interval != 0:
            return None

        try:
            from ..agent import create_strategy_states, create_portfolio_state, MarketSnapshotGenerator
        except ImportError:
            warnings.warn("Agent module not installed, skipping agent call")
            return None

        # Generate agent input
        generator = MarketSnapshotGenerator(lookback_bars=20)

        strategies = create_strategy_states(
            enabled_strategies=list(self.agent_enabled_strategies.keys()),
            strategy_params={}
        )

        portfolio_state = create_portfolio_state(portfolio, data.iloc[current_idx]['close'])

        agent_input = generator.generate(
            data=data,
            current_idx=current_idx,
            symbol=symbol,
            interval=interval,
            strategies=strategies,
            portfolio=portfolio_state
        )

        agent_output = self.agent_expert.analyze(agent_input, verbose=False)

        from ..agent import execute_agent_decision
        result = execute_agent_decision(agent_output, self.agent_enabled_strategies)

        if result['changed']:
            self.agent_enabled_strategies.update(result['strategy_changes'])

        return result

    def _process_signal(
        self,
        signal: int,
        portfolio: Portfolio,
        broker: Broker,
        symbol: str,
        bar: pd.Series,
        timestamp: pd.Timestamp
    ):
        """Process a trading signal."""
        current_pos = portfolio.get_position_size(symbol)

        # Risk manager check (if available)
        if self.risk_manager:
            try:
                self.risk_manager.sync_from_portfolio(portfolio)

                market_data_dict = {symbol: self._data} if hasattr(self, '_data') else {}
                risk_signals = self.risk_manager.check_and_apply_risk_rules(
                    portfolio, market_data_dict
                )

                for risk_signal in risk_signals:
                    if risk_signal.severity == 'critical' and risk_signal.action in ['reject_new', 'close_all']:
                        warnings.warn(f"Risk limit triggered: {risk_signal.message}")
                        return
            except Exception as e:
                warnings.warn(f"Risk check failed: {e}")

        # Calculate target position
        if signal == 1:  # Buy signal
            target_pos = self._calculate_target_position(portfolio, bar['close'], 1, symbol)
        elif signal == -1:  # Sell signal
            if self.allow_short:
                target_pos = -self._calculate_target_position(portfolio, bar['close'], -1, symbol)
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
                symbol, "buy", size_change, "signal", timestamp
            )
        else:
            order = broker.create_market_order(
                symbol, "sell", abs(size_change), "signal", timestamp
            )

        broker.submit_order(order, timestamp)

    def _calculate_target_position(
        self,
        portfolio: Portfolio,
        price: float,
        direction: int,
        symbol: str = "default"
    ) -> float:
        """Calculate target position size."""
        equity = portfolio.total_equity

        # Use risk manager if available
        if self.risk_manager and hasattr(self._data, 'index'):
            try:
                quantity = self.risk_manager.calculate_position_size(
                    symbol=symbol,
                    signal=direction,
                    current_price=price,
                    market_data=self._data
                )
                return quantity
            except Exception as e:
                warnings.warn(f"Risk manager sizing failed: {e}, using default method")

        # Default position sizing logic
        if self.position_sizing == PositionSizing.ALL_IN:
            position_value = equity * self.position_size
        elif self.position_sizing == PositionSizing.FIXED_SIZE:
            return self.position_size * direction
        elif self.position_sizing == PositionSizing.FIXED_FRACTION:
            position_value = equity * self.position_size
        else:
            position_value = equity * self.position_size

        # Convert to quantity
        size = position_value / price

        # Apply max position limit
        max_size = (equity * self.max_position_size) / price
        size = min(size, max_size)

        return size * direction

    def _check_and_execute_stop_loss(
        self,
        portfolio: Portfolio,
        broker: Broker,
        symbol: str,
        bar: pd.Series,
        timestamp: pd.Timestamp
    ):
        """Check and execute stop loss."""
        position = portfolio.get_position(symbol)
        if position is None or position.is_flat:
            return

        if position.unrealized_pnl_pct <= -self.stop_loss:
            order = broker.create_market_order(
                symbol,
                "sell" if position.is_long else "buy",
                abs(position.size),
                "stop_loss",
                timestamp
            )
            broker.submit_order(order, timestamp)

    def _check_and_execute_take_profit(
        self,
        portfolio: Portfolio,
        broker: Broker,
        symbol: str,
        bar: pd.Series,
        timestamp: pd.Timestamp
    ):
        """Check and execute take profit."""
        position = portfolio.get_position(symbol)
        if position is None or position.is_flat:
            return

        if position.unrealized_pnl_pct >= self.take_profit:
            order = broker.create_market_order(
                symbol,
                "sell" if position.is_long else "buy",
                abs(position.size),
                "take_profit",
                timestamp
            )
            broker.submit_order(order, timestamp)

    # ========== Performance Calculation ==========

    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity: pd.Series,
        trades: List[Trade]
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}

        if len(returns) == 0:
            return metrics

        # Basic return metrics
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 0 else 0
        metrics['total_return'] = total_return

        # Annualized return
        n_days = len(returns)
        if n_days > 0:
            metrics['annualized_return'] = annualize_return(
                total_return, n_days, TRADING_DAYS_STOCK
            )
        else:
            metrics['annualized_return'] = 0

        # Sharpe ratio
        metrics['sharpe_ratio'] = calc_sharpe(returns, trading_days=TRADING_DAYS_STOCK)

        # Sortino ratio
        metrics['sortino_ratio'] = calc_sortino(returns, trading_days=TRADING_DAYS_STOCK)

        # Max drawdown
        metrics['max_drawdown'] = calc_max_drawdown(equity)

        # Calmar ratio
        if metrics['max_drawdown'] > 0:
            metrics['calmar_ratio'] = metrics['annualized_return'] / metrics['max_drawdown']
        else:
            metrics['calmar_ratio'] = 0

        # Trade statistics
        metrics['num_trades'] = len(trades)

        if len(trades) > 0:
            winners = [t for t in trades if t.pnl > 0]
            losers = [t for t in trades if t.pnl <= 0]

            metrics['win_rate'] = len(winners) / len(trades)
            metrics['num_winners'] = len(winners)
            metrics['num_losers'] = len(losers)

            # Average win/loss
            metrics['avg_win'] = np.mean([t.pnl for t in winners]) if winners else 0
            metrics['avg_loss'] = np.mean([t.pnl for t in losers]) if losers else 0

            # Profit factor
            total_wins = sum(t.pnl for t in winners)
            total_losses = abs(sum(t.pnl for t in losers))
            if total_losses > 0:
                metrics['profit_factor'] = total_wins / total_losses
            else:
                metrics['profit_factor'] = float('inf') if total_wins > 0 else 0

            # Average holding time
            holding_times = [(t.exit_time - t.entry_time).days for t in trades]
            metrics['avg_holding_days'] = np.mean(holding_times) if holding_times else 0

        else:
            metrics['win_rate'] = 0
            metrics['num_winners'] = 0
            metrics['num_losers'] = 0
            metrics['avg_win'] = 0
            metrics['avg_loss'] = 0
            metrics['profit_factor'] = 0
            metrics['avg_holding_days'] = 0

        # Volatility & daily return
        metrics['volatility'] = returns.std() * np.sqrt(TRADING_DAYS_STOCK)
        metrics['mean_daily_return'] = returns.mean()

        # Win/lose/flat days
        metrics['win_days'] = int((returns > 1e-8).sum())
        metrics['lose_days'] = int((returns < -1e-8).sum())
        metrics['flat_days'] = int(len(returns) - metrics['win_days'] - metrics['lose_days'])

        return metrics

    def _compute_benchmark(
        self,
        data: pd.DataFrame,
    ) -> tuple:
        """Compute buy-and-hold benchmark."""
        close = data['close']
        benchmark_equity = self.initial_capital * (close / close.iloc[0])
        bh_returns = close.pct_change().fillna(0)
        benchmark_metrics = self._calculate_metrics(bh_returns, benchmark_equity, trades=[])
        return benchmark_equity, benchmark_metrics

    def __repr__(self):
        return (f"BacktestEngine(mode={self.mode.value}, "
                f"capital={self.initial_capital:,.0f}, "
                f"fee_model={self.fee_model.name})")


# ========== Convenience Functions ==========

def backtest(
    data: pd.DataFrame,
    strategy=None,
    signals: pd.Series = None,
    initial_capital: float = 100000,
    fee_model: Union[BaseFeeModel, str] = None,
    mode: str = "vectorized",
    stop_loss: float = None,
    **kwargs
) -> BacktestResult:
    """
    Convenience backtest function.

    Parameters:
        data: OHLCV data.
        strategy: Strategy object.
        signals: Signal series (mutually exclusive with strategy).
        initial_capital: Starting capital.
        fee_model: Fee model.
        mode: Execution mode.
        stop_loss: Stop loss percentage.
        **kwargs: Additional engine parameters.

    Returns:
        BacktestResult: Backtest results.

    Example:
        >>> result = backtest(data, strategy=MAStrategy())
        >>> result = backtest(data, signals=my_signals, fee_model='china')
    """
    engine = BacktestEngine(
        initial_capital=initial_capital,
        mode=mode,
        stop_loss=stop_loss,
        **kwargs
    )

    if fee_model:
        engine.set_fee_model(fee_model)

    if strategy:
        engine.set_strategy(strategy)
    elif signals is not None:
        engine.set_signals(signals)
    else:
        raise ValueError("Must provide either strategy or signals")

    return engine.run(data)
