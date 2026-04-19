"""
Backtest Result Data Structures

Defines trade records, backtest results, and related data classes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """
    A complete trade record (entry + exit).

    Attributes:
        trade_id: Unique trade identifier.
        symbol: Instrument symbol.
        direction: Direction (1=long, -1=short).
        entry_time: Entry timestamp.
        exit_time: Exit timestamp.
        entry_price: Entry price.
        exit_price: Exit price.
        size: Trade quantity (shares / contracts). For vectorized-mode
            trades this is reconstructed as
            ``entry_equity * position_size / entry_price``.
        pnl: Profit/loss in account currency. In **vectorized** mode this
            is a *linear* per-trade approximation
            ``direction * shares * (exit_price - entry_price)``; fees are
            not double-deducted because they are already embedded in the
            geometric ``equity_curve``. As a result, ``sum(trade.pnl)``
            and ``equity_curve.iloc[-1] - initial_capital`` diverge by
            ``O(σ²·T·notional)`` whenever there are reversals or
            fractional positions — this is a mathematical property of
            geometric vs. linear PnL, not a bug. In bankruptcy runs
            ``sum(trade.pnl)`` may exceed the actual equity loss because
            the linear formula does not see the equity-zero floor.
        pnl_pct: P&L percentage.
        commission: Commission paid.
        slippage_cost: Slippage cost.
        reason: Exit reason ('signal', 'stop_loss', 'take_profit', 'timeout').
        holding_bars: Number of bars held.
        metadata: Additional information.

    Example:
        >>> trade = Trade(
        ...     trade_id="T001", symbol="AAPL", direction=1,
        ...     entry_time=pd.Timestamp("2024-01-01"),
        ...     exit_time=pd.Timestamp("2024-01-05"),
        ...     entry_price=150.0, exit_price=155.0,
        ...     size=100, pnl=500.0, pnl_pct=0.0333,
        ...     commission=2.0, slippage_cost=1.5, reason="signal"
        ... )
    """
    trade_id: str
    symbol: str
    direction: int  # 1=long, -1=short
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    commission: float = 0.0
    slippage_cost: float = 0.0
    reason: str = "signal"
    holding_bars: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.direction not in (1, -1):
            raise ValueError(f"direction must be 1 or -1, got: {self.direction}")
        if self.size <= 0:
            raise ValueError(f"size must be positive, got: {self.size}")

    @property
    def net_pnl_pct(self) -> float:
        """Net P&L as a percentage of entry notional value."""
        if self.entry_price == 0 or self.size == 0:
            return 0.0
        return self.pnl / (self.entry_price * self.size)

    @property
    def gross_pnl(self) -> float:
        """Gross P&L (before fees)."""
        return self.pnl + self.commission + self.slippage_cost

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def holding_period(self) -> pd.Timedelta:
        return self.exit_time - self.entry_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'size': self.size,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'commission': self.commission,
            'slippage_cost': self.slippage_cost,
            'reason': self.reason,
            'holding_bars': self.holding_bars,
            'net_pnl_pct': self.net_pnl_pct,
            'is_winner': self.is_winner,
            'gross_pnl': self.gross_pnl
        }

    def __repr__(self):
        direction_str = "LONG" if self.direction == 1 else "SHORT"
        pnl_sign = "+" if self.pnl >= 0 else ""
        return (f"Trade({self.trade_id}: {direction_str} {self.symbol} "
                f"{self.entry_time.strftime('%Y-%m-%d')} -> {self.exit_time.strftime('%Y-%m-%d')} "
                f"PnL: {pnl_sign}{self.pnl:.2f} ({pnl_sign}{self.pnl_pct*100:.2f}%))")


@dataclass
class BacktestResult:
    """
    Complete backtest output including equity curve, trades, and metrics.

    Attributes:
        equity_curve: Equity curve (pd.Series, index=time).
        trades: List of trade records.
        positions: Position history (pd.DataFrame).
        metrics: Performance metrics dictionary.
        orders: Order history (pd.DataFrame).
        initial_capital: Starting capital.
        final_capital: Ending capital.
        start_date: Backtest start date.
        end_date: Backtest end date.
        strategy_name: Strategy name.
        parameters: Strategy parameters.
        metadata: Additional information.

    Example:
        >>> result = engine.run(data)
        >>> print(result.summary())
        >>> result.equity_curve.plot()
    """
    equity_curve: pd.Series
    trades: List[Trade]
    positions: pd.DataFrame
    metrics: Dict[str, float]
    orders: pd.DataFrame = field(default_factory=pd.DataFrame)
    initial_capital: float = 100000.0
    final_capital: float = 0.0
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    strategy_name: str = "Unknown"
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    daily_returns: Optional[pd.Series] = None
    benchmark_equity: Optional[pd.Series] = None
    benchmark_metrics: Optional[Dict[str, float]] = None
    benchmark_name: str = "Buy & Hold"
    # Multi-asset fields (v0.7)
    per_asset_pnl: Optional[Dict[str, pd.Series]] = None
    per_asset_trades: Optional[Dict[str, List['Trade']]] = None
    per_asset_metrics: Optional[Dict[str, Dict]] = None
    symbols: List[str] = field(default_factory=list)
    turnover_history: Optional[List[float]] = None
    # Annualisation (v1.9.5)
    trading_days: int = 252
    per_asset_trading_days: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.final_capital == 0.0 and len(self.equity_curve) > 0:
            self.final_capital = self.equity_curve.iloc[-1]
        if self.start_date is None and len(self.equity_curve) > 0:
            self.start_date = self.equity_curve.index[0]
        if self.end_date is None and len(self.equity_curve) > 0:
            self.end_date = self.equity_curve.index[-1]

    # ========== Computed Properties ==========

    @property
    def total_return(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.final_capital - self.initial_capital) / self.initial_capital

    @property
    def total_pnl(self) -> float:
        return self.final_capital - self.initial_capital

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def num_winners(self) -> int:
        return sum(1 for t in self.trades if t.is_winner)

    @property
    def num_losers(self) -> int:
        return sum(1 for t in self.trades if not t.is_winner)

    @property
    def win_rate(self) -> float:
        if self.num_trades == 0:
            return 0.0
        return self.num_winners / self.num_trades

    @property
    def total_commission(self) -> float:
        return sum(t.commission for t in self.trades)

    @property
    def total_slippage(self) -> float:
        return sum(t.slippage_cost for t in self.trades)

    @property
    def sharpe_ratio(self) -> float:
        return self.metrics.get('sharpe_ratio', 0.0)

    @property
    def benchmark_return(self) -> Optional[float]:
        """Buy-and-hold benchmark total return."""
        if self.benchmark_metrics is None:
            return None
        return self.benchmark_metrics.get('total_return', 0.0)

    @property
    def max_drawdown(self) -> float:
        return self.metrics.get('max_drawdown', 0.0)

    @property
    def profit_factor(self) -> float:
        return self.metrics.get('profit_factor', 0.0)

    # ========== Conversion Methods ==========

    @property
    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        d = {
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'total_pnl': self.total_pnl,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'start_date': str(self.start_date) if self.start_date else None,
            'end_date': str(self.end_date) if self.end_date else None,
            'strategy_name': self.strategy_name,
            'parameters': self.parameters,
            'metrics': self.metrics,
            'trading_days': self.trading_days,
        }
        if self.per_asset_trading_days:
            d['per_asset_trading_days'] = dict(self.per_asset_trading_days)
        if self.benchmark_metrics is not None:
            d['benchmark_metrics'] = self.benchmark_metrics
        return d

    # ========== Output Methods ==========

    def summary(self) -> str:
        """Generate a formatted backtest summary."""
        lines = [
            "=" * 60,
            f"Backtest Summary - {self.strategy_name}",
            "=" * 60,
            "",
            "[General]",
            f"  Period:          {self.start_date} ~ {self.end_date}",
            f"  Initial Capital: {self.initial_capital:,.2f}",
            f"  Final Capital:   {self.final_capital:,.2f}",
            "",
            "[Returns]",
            f"  Total Return:    {self.total_return*100:+.2f}%",
            f"  Total P&L:       {self.total_pnl:+,.2f}",
            f"  Sharpe Ratio:    {self.sharpe_ratio:.3f}",
            f"  Max Drawdown:    {self.max_drawdown*100:.2f}%",
        ]

        vol = self.metrics.get('volatility')
        if vol is not None:
            lines.append(f"  Ann. Volatility: {vol*100:.2f}%")

        win_days = self.metrics.get('win_days')
        if win_days is not None:
            lose_days = self.metrics.get('lose_days', 0)
            flat_days = self.metrics.get('flat_days', 0)
            lines.append(f"  Win/Lose/Flat:   {win_days}/{lose_days}/{flat_days} days")

        lines += [
            "",
            "[Trade Statistics]",
            f"  Num Trades:      {self.num_trades}",
            f"  Winners:         {self.num_winners} ({self.win_rate*100:.1f}%)",
            f"  Losers:          {self.num_losers}",
            f"  Profit Factor:   {self.profit_factor:.2f}",
            "",
            "[Costs]",
            f"  Total Commission: {self.total_commission:,.2f}",
            f"  Total Slippage:   {self.total_slippage:,.2f}",
        ]

        if self.benchmark_metrics is not None:
            bm_ret = self.benchmark_metrics.get('total_return', 0)
            bm_sharpe = self.benchmark_metrics.get('sharpe_ratio', 0)
            bm_mdd = self.benchmark_metrics.get('max_drawdown', 0)
            excess = self.total_return - bm_ret
            lines += [
                "",
                f"[Benchmark - {self.benchmark_name}]",
                f"  Benchmark Return: {bm_ret*100:+.2f}%",
                f"  Benchmark Sharpe: {bm_sharpe:.3f}",
                f"  Benchmark MDD:    {bm_mdd*100:.2f}%",
                f"  Excess Return:    {excess*100:+.2f}%",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self):
        return (f"BacktestResult(strategy={self.strategy_name}, "
                f"return={self.total_return*100:+.2f}%, "
                f"trades={self.num_trades}, "
                f"sharpe={self.sharpe_ratio:.2f})")


@dataclass
class PositionSnapshot:
    """
    A snapshot of a position at a specific point in time.

    Attributes:
        timestamp: Snapshot time.
        symbol: Instrument symbol.
        size: Position size (positive=long, negative=short).
        avg_price: Average entry price.
        current_price: Current market price.
        market_value: Market value.
        unrealized_pnl: Unrealized P&L.
        realized_pnl: Realized P&L.
    """
    timestamp: pd.Timestamp
    symbol: str
    size: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0

    @property
    def is_flat(self) -> bool:
        return self.size == 0

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_price == 0 or self.size == 0:
            return 0.0
        return (self.current_price - self.avg_price) / self.avg_price * np.sign(self.size)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'size': self.size,
            'avg_price': self.avg_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct
        }


@dataclass
class EquityPoint:
    """
    Account equity state at a specific point in time.

    Attributes:
        timestamp: Snapshot time.
        total_equity: Total equity.
        cash: Cash balance.
        position_value: Total position market value.
        unrealized_pnl: Unrealized P&L.
        realized_pnl: Cumulative realized P&L.
        drawdown: Current drawdown amount.
        drawdown_pct: Current drawdown percentage.
    """
    timestamp: pd.Timestamp
    total_equity: float
    cash: float
    position_value: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    drawdown: float = 0.0
    drawdown_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'total_equity': self.total_equity,
            'cash': self.cash,
            'position_value': self.position_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'drawdown': self.drawdown,
            'drawdown_pct': self.drawdown_pct
        }


def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    """
    Convert a list of Trade objects to a DataFrame.

    Parameters:
        trades: List of Trade objects.

    Returns:
        pd.DataFrame: Trade records table.

    Example:
        >>> df = trades_to_dataframe(result.trades)
        >>> df.groupby('symbol').agg({'pnl': 'sum'})
    """
    if not trades:
        return pd.DataFrame(columns=[
            'trade_id', 'symbol', 'direction', 'entry_time', 'exit_time',
            'entry_price', 'exit_price', 'size', 'pnl', 'pnl_pct',
            'commission', 'slippage_cost', 'reason', 'holding_bars',
            'net_pnl_pct', 'is_winner', 'gross_pnl'
        ])

    return pd.DataFrame([t.to_dict() for t in trades])
