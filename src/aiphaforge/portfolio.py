"""
Portfolio Management

Tracks positions, cash, equity, and account state.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from .orders import Order, OrderSide
from .results import Trade, PositionSnapshot, EquityPoint


@dataclass
class Position:
    """
    A single instrument position.

    Attributes:
        symbol: Instrument symbol.
        size: Position size (positive=long, negative=short).
        avg_entry_price: Average entry price.
        current_price: Current market price.
        realized_pnl: Realized P&L.
        open_time: Position open time.
        last_update_time: Last update time.

    Example:
        >>> pos = Position(symbol="AAPL", size=100, avg_entry_price=150.0)
        >>> pos.update_price(155.0)
        >>> print(f"Unrealized P&L: ${pos.unrealized_pnl:.2f}")
    """
    symbol: str
    size: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    realized_pnl: float = 0.0
    open_time: Optional[pd.Timestamp] = None
    last_update_time: Optional[pd.Timestamp] = None

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
    def market_value(self) -> float:
        """Absolute market value."""
        return abs(self.size) * self.current_price

    @property
    def notional_value(self) -> float:
        """Signed notional value."""
        return self.size * self.current_price

    @property
    def cost_basis(self) -> float:
        return abs(self.size) * self.avg_entry_price

    @property
    def unrealized_pnl(self) -> float:
        if self.size == 0:
            return 0.0
        return self.size * (self.current_price - self.avg_entry_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    def update_price(self, price: float, timestamp: Optional[pd.Timestamp] = None):
        """Update current price."""
        self.current_price = price
        if timestamp:
            self.last_update_time = timestamp

    def add(
        self,
        size: float,
        price: float,
        timestamp: Optional[pd.Timestamp] = None
    ) -> float:
        """
        Add to or reduce the position.

        Parameters:
            size: Change in quantity (positive=buy, negative=sell).
            price: Execution price.
            timestamp: Timestamp.

        Returns:
            float: Realized P&L (if the trade closes part of the position).
        """
        realized = 0.0

        # New position
        if self.size == 0:
            self.size = size
            self.avg_entry_price = price
            self.open_time = timestamp
        # Same direction (adding)
        elif (self.size > 0 and size > 0) or (self.size < 0 and size < 0):
            total_cost = self.avg_entry_price * abs(self.size) + price * abs(size)
            self.size += size
            self.avg_entry_price = total_cost / abs(self.size)
        # Opposite direction (reducing or reversing)
        else:
            close_size = min(abs(self.size), abs(size))

            # Calculate realized P&L
            if self.size > 0:  # Was long
                realized = close_size * (price - self.avg_entry_price)
            else:  # Was short
                realized = close_size * (self.avg_entry_price - price)

            self.realized_pnl += realized
            remaining = abs(size) - close_size

            # Full close or reverse
            if abs(size) >= abs(self.size):
                if remaining > 0:
                    # Reverse position
                    self.size = size + self.size
                    self.avg_entry_price = price
                    self.open_time = timestamp
                else:
                    # Fully closed
                    self.size = 0
                    self.avg_entry_price = 0
                    self.open_time = None
            else:
                # Partial close
                self.size += size

        self.current_price = price
        self.last_update_time = timestamp
        return realized

    def close(self, price: float, timestamp: Optional[pd.Timestamp] = None) -> float:
        """
        Close the entire position.

        Parameters:
            price: Close price.
            timestamp: Timestamp.

        Returns:
            float: Realized P&L.
        """
        if self.size == 0:
            return 0.0

        if self.size > 0:
            realized = self.size * (price - self.avg_entry_price)
        else:
            realized = abs(self.size) * (self.avg_entry_price - price)

        self.realized_pnl += realized
        self.size = 0
        self.avg_entry_price = 0
        self.current_price = price
        self.last_update_time = timestamp
        self.open_time = None

        return realized

    def to_snapshot(self, timestamp: pd.Timestamp) -> PositionSnapshot:
        """Generate a position snapshot."""
        return PositionSnapshot(
            timestamp=timestamp,
            symbol=self.symbol,
            size=self.size,
            avg_price=self.avg_entry_price,
            current_price=self.current_price,
            market_value=self.market_value,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl
        )

    def __repr__(self):
        if self.is_flat:
            return f"Position({self.symbol}: FLAT)"
        direction = "LONG" if self.is_long else "SHORT"
        pnl_sign = "+" if self.unrealized_pnl >= 0 else ""
        return (f"Position({self.symbol}: {direction} {abs(self.size):.2f} "
                f"@{self.avg_entry_price:.2f} -> {self.current_price:.2f} "
                f"PnL: {pnl_sign}{self.unrealized_pnl:.2f})")


class Portfolio:
    """
    Portfolio manager tracking multiple positions, cash, and total equity.

    Attributes:
        initial_capital: Starting capital.
        cash: Current cash balance.
        positions: Position dict {symbol: Position}.
        max_position_size: Max single position as fraction of equity.
        allow_short: Whether short selling is allowed.
        margin_requirement: Margin requirement multiplier.

    Example:
        >>> portfolio = Portfolio(initial_capital=100000)
        >>> portfolio.update_position("AAPL", 100, 150.0, timestamp)
        >>> print(portfolio.total_equity)
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        max_position_size: float = 1.0,
        allow_short: bool = True,
        margin_requirement: float = 1.0,
        fee_allocation: str = "proportional",
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.max_position_size = max_position_size
        self.allow_short = allow_short
        self.margin_requirement = margin_requirement
        self.fee_allocation = fee_allocation

        # History
        self.equity_history: List[EquityPoint] = []
        self.trade_history: List[Trade] = []
        self.position_snapshots: List[PositionSnapshot] = []

        # Internal state
        self._peak_equity = initial_capital
        self._trade_counter = 0
        self._pending_entries: Dict[str, dict] = {}  # Tracks entry info

    # ========== Equity Calculations ==========

    @property
    def position_value(self) -> float:
        """Total position market value."""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def total_equity(self) -> float:
        """Total equity = cash + net position value (signed)."""
        return self.cash + sum(pos.notional_value for pos in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def realized_pnl(self) -> float:
        return sum(pos.realized_pnl for pos in self.positions.values())

    @property
    def total_return(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.total_equity - self.initial_capital) / self.initial_capital

    @property
    def buying_power(self) -> float:
        return self.cash / self.margin_requirement

    @property
    def current_drawdown(self) -> float:
        if self._peak_equity == 0:
            return 0.0
        return (self._peak_equity - self.total_equity) / self._peak_equity

    # ========== Position Operations ==========

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def get_position_size(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return pos.size if pos else 0.0

    def has_position(self, symbol: str) -> bool:
        pos = self.positions.get(symbol)
        return pos is not None and not pos.is_flat

    def update_position(
        self,
        symbol: str,
        size_change: float,
        price: float,
        timestamp: pd.Timestamp,
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> Optional[Trade]:
        """
        Update a position.

        Handles new entries, add-to-position, partial close, full close,
        and reversal scenarios.  Returns a Trade record when a position is
        fully or partially closed.

        Parameters:
            symbol: Instrument symbol.
            size_change: Position change (positive=buy, negative=sell).
            price: Execution price.
            timestamp: Timestamp.
            commission: Commission.
            slippage: Slippage cost.

        Returns:
            Optional[Trade]: Trade record if a (partial or full) close occurred.
        """
        if size_change == 0:
            return None

        trade = None

        # Get or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        position = self.positions[symbol]
        old_size = position.size

        # Determine scenario before mutating position
        is_new_entry = (old_size == 0 and size_change != 0)
        is_same_direction = (
            old_size != 0
            and size_change != 0
            and ((old_size > 0 and size_change > 0)
                 or (old_size < 0 and size_change < 0))
        )
        is_reducing = (
            old_size != 0
            and size_change != 0
            and not is_same_direction
        )

        # --- New entry ---
        if is_new_entry:
            self._pending_entries[symbol] = {
                'entry_time': timestamp,
                'entry_price': price,
                'direction': 1 if size_change > 0 else -1,
                'size': abs(size_change),
                'entry_commission': commission,
                'entry_slippage': slippage,
            }

        # --- Add-to-position (same direction) ---
        if is_same_direction and symbol in self._pending_entries:
            # Position.add will be called below; capture pending info for update
            pass  # handled after position.add

        # Update position (this computes realized PnL for reductions)
        realized = position.add(size_change, price, timestamp)

        # Update cash
        trade_value = abs(size_change) * price
        if size_change > 0:
            self.cash -= trade_value + commission + slippage
        else:
            self.cash += trade_value - commission - slippage

        # --- Post-update bookkeeping ---

        if is_same_direction and symbol in self._pending_entries:
            # Sync pending entry info with the updated average position
            entry_info = self._pending_entries[symbol]
            entry_info['entry_price'] = position.avg_entry_price
            entry_info['size'] = abs(position.size)
            entry_info['entry_commission'] += commission
            entry_info['entry_slippage'] += slippage

        elif is_reducing and symbol in self._pending_entries:
            entry_info = self._pending_entries[symbol]
            closed_qty = min(abs(old_size), abs(size_change))
            pending_size = entry_info['size']

            # Calculate proportional entry fees for the closed portion
            if pending_size > 0:
                if self.fee_allocation == "first_close":
                    alloc_entry_commission = entry_info['entry_commission']
                    alloc_entry_slippage = entry_info['entry_slippage']
                else:
                    # Proportional allocation
                    close_ratio = closed_qty / pending_size
                    alloc_entry_commission = entry_info['entry_commission'] * close_ratio
                    alloc_entry_slippage = entry_info['entry_slippage'] * close_ratio
            else:
                alloc_entry_commission = 0.0
                alloc_entry_slippage = 0.0

            total_fees = commission + slippage + alloc_entry_commission + alloc_entry_slippage

            # Determine whether this is a full close, partial close, or reversal
            if position.size == 0:
                # --- Full close ---
                self._pending_entries.pop(symbol)
                self._trade_counter += 1
                trade = Trade(
                    trade_id=f"T{self._trade_counter:06d}",
                    symbol=symbol,
                    direction=entry_info['direction'],
                    entry_time=entry_info['entry_time'],
                    exit_time=timestamp,
                    entry_price=entry_info['entry_price'],
                    exit_price=price,
                    size=closed_qty,
                    pnl=realized - total_fees,
                    pnl_pct=(price / entry_info['entry_price'] - 1) * entry_info['direction'],
                    commission=commission + alloc_entry_commission,
                    slippage_cost=slippage + alloc_entry_slippage,
                    reason="signal",
                )
                self.trade_history.append(trade)

            elif (old_size > 0 and position.size > 0) or (old_size < 0 and position.size < 0):
                # --- Partial close (position reduced but not flipped) ---
                self._trade_counter += 1
                trade = Trade(
                    trade_id=f"T{self._trade_counter:06d}",
                    symbol=symbol,
                    direction=entry_info['direction'],
                    entry_time=entry_info['entry_time'],
                    exit_time=timestamp,
                    entry_price=entry_info['entry_price'],
                    exit_price=price,
                    size=closed_qty,
                    pnl=realized - total_fees,
                    pnl_pct=(price / entry_info['entry_price'] - 1) * entry_info['direction'],
                    commission=commission + alloc_entry_commission,
                    slippage_cost=slippage + alloc_entry_slippage,
                    reason="signal",
                )
                self.trade_history.append(trade)

                # Update pending entry to reflect remaining size
                entry_info['size'] -= closed_qty
                if self.fee_allocation == "first_close":
                    entry_info['entry_commission'] = 0.0
                    entry_info['entry_slippage'] = 0.0
                else:
                    entry_info['entry_commission'] -= alloc_entry_commission
                    entry_info['entry_slippage'] -= alloc_entry_slippage

            else:
                # --- Reversal (position crossed zero) ---
                # Split exit-side fees between close and new entry proportionally
                total_change = abs(size_change)
                if total_change > 0:
                    close_share = abs(old_size) / total_change
                else:
                    close_share = 1.0
                new_entry_share = 1.0 - close_share

                close_commission = commission * close_share
                close_slippage = slippage * close_share
                new_entry_commission = commission * new_entry_share
                new_entry_slippage = slippage * new_entry_share

                close_total_fees = close_commission + close_slippage + alloc_entry_commission + alloc_entry_slippage

                # Close Trade for old position (full close)
                self._trade_counter += 1
                trade = Trade(
                    trade_id=f"T{self._trade_counter:06d}",
                    symbol=symbol,
                    direction=entry_info['direction'],
                    entry_time=entry_info['entry_time'],
                    exit_time=timestamp,
                    entry_price=entry_info['entry_price'],
                    exit_price=price,
                    size=abs(old_size),
                    pnl=realized - close_total_fees,
                    pnl_pct=(price / entry_info['entry_price'] - 1) * entry_info['direction'],
                    commission=close_commission + alloc_entry_commission,
                    slippage_cost=close_slippage + alloc_entry_slippage,
                    reason="signal",
                )
                self.trade_history.append(trade)

                # Create new pending entry for the reversed direction
                self._pending_entries[symbol] = {
                    'entry_time': timestamp,
                    'entry_price': price,
                    'direction': 1 if position.size > 0 else -1,
                    'size': abs(position.size),
                    'entry_commission': new_entry_commission,
                    'entry_slippage': new_entry_slippage,
                }

        return trade

    def update_from_order(
        self,
        order: Order,
        timestamp: pd.Timestamp
    ) -> Optional[Trade]:
        """
        Update position from a filled order.

        Parameters:
            order: Filled order.
            timestamp: Timestamp.

        Returns:
            Optional[Trade]: Trade record if a round-trip was completed.
        """
        if not order.is_filled:
            return None

        size_change = order.filled_size
        if order.is_sell:
            size_change = -size_change

        return self.update_position(
            symbol=order.symbol,
            size_change=size_change,
            price=order.filled_price,
            timestamp=timestamp,
            commission=order.commission,
            slippage=order.slippage
        )

    # ========== Price Updates ==========

    def update_prices(
        self,
        prices: Dict[str, float],
        timestamp: pd.Timestamp,
        record: bool = True
    ):
        """
        Update prices for all held positions.

        Parameters:
            prices: Price dict {symbol: price}.
            timestamp: Timestamp.
            record: Whether to record an equity snapshot (default True).
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price, timestamp)

        # Update peak equity
        equity = self.total_equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Record equity snapshot
        if record:
            self._record_equity(timestamp)

    def _record_equity(self, timestamp: pd.Timestamp):
        """Record an equity snapshot."""
        equity_point = EquityPoint(
            timestamp=timestamp,
            total_equity=self.total_equity,
            cash=self.cash,
            position_value=self.position_value,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            drawdown=self._peak_equity - self.total_equity,
            drawdown_pct=self.current_drawdown
        )
        self.equity_history.append(equity_point)

    # ========== Risk Checks ==========

    def check_position_limit(self, symbol: str, size: float, price: float) -> bool:
        """Check if adding to a position would exceed limits."""
        trade_value = abs(size) * price
        max_allowed = self.total_equity * self.max_position_size

        current = self.get_position(symbol)
        current_value = current.market_value if current else 0

        return (current_value + trade_value) <= max_allowed

    def check_buying_power(self, size: float, price: float) -> bool:
        """Check if there is sufficient buying power."""
        required = abs(size) * price
        return required <= self.buying_power

    def get_stop_loss_symbols(self, stop_loss_pct: float) -> List[str]:
        """
        Get symbols that have triggered stop loss.

        Parameters:
            stop_loss_pct: Stop loss percentage (e.g. 0.05 for 5%).

        Returns:
            List[str]: Symbols that triggered stop loss.
        """
        triggered = []
        for symbol, pos in self.positions.items():
            if pos.is_flat:
                continue
            if pos.unrealized_pnl_pct <= -stop_loss_pct:
                triggered.append(symbol)
        return triggered

    def calculate_position_size(
        self,
        price: float,
        method: str = "fixed_fraction",
        fraction: float = 0.1,
        risk_per_trade: float = 0.02,
        stop_distance: Optional[float] = None
    ) -> float:
        """
        Calculate suggested position size.

        Parameters:
            price: Current price.
            method: Sizing method ('fixed_fraction', 'risk_based', 'kelly').
            fraction: Fraction of equity.
            risk_per_trade: Risk per trade as fraction of equity.
            stop_distance: Stop loss distance in price units.

        Returns:
            float: Suggested quantity.
        """
        equity = self.total_equity

        if method == "fixed_fraction":
            position_value = equity * fraction
            return position_value / price

        elif method == "risk_based":
            if stop_distance is None or stop_distance <= 0:
                return 0
            risk_amount = equity * risk_per_trade
            return risk_amount / stop_distance

        else:
            position_value = equity * fraction
            return position_value / price

    # ========== Output Methods ==========

    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as a pd.Series."""
        if not self.equity_history:
            return pd.Series(dtype=float)

        data = [(e.timestamp, e.total_equity) for e in self.equity_history]
        df = pd.DataFrame(data, columns=['timestamp', 'equity'])
        return df.set_index('timestamp')['equity']

    def get_positions_df(self) -> pd.DataFrame:
        """Get active positions as a DataFrame."""
        if not self.positions:
            return pd.DataFrame()

        data = []
        for symbol, pos in self.positions.items():
            if not pos.is_flat:
                data.append({
                    'symbol': symbol,
                    'size': pos.size,
                    'direction': 'LONG' if pos.is_long else 'SHORT',
                    'avg_entry_price': pos.avg_entry_price,
                    'current_price': pos.current_price,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct * 100,
                    'realized_pnl': pos.realized_pnl
                })

        return pd.DataFrame(data)

    def summary(self) -> str:
        """Generate a portfolio summary."""
        lines = [
            "=" * 50,
            "Portfolio Summary",
            "=" * 50,
            f"Initial Capital:  ${self.initial_capital:,.2f}",
            f"Current Equity:   ${self.total_equity:,.2f}",
            f"Cash:             ${self.cash:,.2f}",
            f"Position Value:   ${self.position_value:,.2f}",
            "-" * 50,
            f"Total Return:     {self.total_return*100:+.2f}%",
            f"Unrealized P&L:   ${self.unrealized_pnl:+,.2f}",
            f"Realized P&L:     ${self.realized_pnl:+,.2f}",
            f"Current Drawdown: {self.current_drawdown*100:.2f}%",
            "-" * 50,
            f"Open Positions:   {len([p for p in self.positions.values() if not p.is_flat])}",
            f"Completed Trades: {len(self.trade_history)}",
            "=" * 50
        ]
        return "\n".join(lines)

    def reset(self):
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.equity_history.clear()
        self.trade_history.clear()
        self.position_snapshots.clear()
        self._peak_equity = self.initial_capital
        self._trade_counter = 0
        self._pending_entries.clear()

    def __repr__(self):
        return (f"Portfolio(equity=${self.total_equity:,.2f}, "
                f"positions={len([p for p in self.positions.values() if not p.is_flat])}, "
                f"return={self.total_return*100:+.2f}%)")
