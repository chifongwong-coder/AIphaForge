"""
Order Management

Defines order types, order statuses, and the order manager.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """
    Represents a trading order with all its attributes and state.

    Attributes:
        order_id: Unique order identifier.
        symbol: Instrument symbol.
        side: Direction (buy/sell).
        order_type: Order type.
        size: Order quantity.
        price: Limit price (for limit orders).
        stop_price: Trigger price (for stop orders).
        status: Current order status.
        created_time: Creation timestamp.
        filled_time: Fill timestamp.
        filled_size: Filled quantity.
        filled_price: Average fill price.
        commission: Commission charged.
        slippage: Slippage cost.
        reason: Order reason/tag.
        time_in_force: Validity type ('GTC', 'IOC', 'FOK', 'DAY').
        metadata: Additional information.

    Example:
        >>> order = Order(
        ...     symbol="AAPL",
        ...     side=OrderSide.BUY,
        ...     order_type=OrderType.MARKET,
        ...     size=100
        ... )
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    size: float
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_time: Optional[pd.Timestamp] = None
    filled_time: Optional[pd.Timestamp] = None
    filled_size: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    reason: str = ""
    time_in_force: str = "GTC"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate order parameters."""
        if self.size <= 0:
            raise ValueError(f"Order size must be positive, got: {self.size}")

        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders must specify a price")

        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError("Stop orders must specify a stop price")

        # Convert string types
        if isinstance(self.side, str):
            self.side = OrderSide(self.side.lower())
        if isinstance(self.order_type, str):
            self.order_type = OrderType(self.order_type.lower())
        if isinstance(self.status, str):
            self.status = OrderStatus(self.status.lower())

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL

    @property
    def is_pending(self) -> bool:
        return self.status == OrderStatus.PENDING

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_active(self) -> bool:
        """Whether the order can still be filled."""
        return self.status in (OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED)

    @property
    def remaining_size(self) -> float:
        return self.size - self.filled_size

    @property
    def fill_ratio(self) -> float:
        return self.filled_size / self.size if self.size > 0 else 0.0

    @property
    def notional_value(self) -> float:
        """Filled notional value."""
        return self.filled_size * self.filled_price

    @property
    def total_cost(self) -> float:
        """Total cost (commission + slippage)."""
        return self.commission + self.slippage

    def fill(
        self,
        price: float,
        size: float,
        timestamp: pd.Timestamp,
        commission: float = 0.0,
        slippage: float = 0.0
    ):
        """
        Fill the order (fully or partially).

        Parameters:
            price: Fill price.
            size: Fill quantity.
            timestamp: Fill timestamp.
            commission: Commission charged.
            slippage: Slippage cost.
        """
        if size > self.remaining_size:
            raise ValueError(f"Fill size {size} exceeds remaining size {self.remaining_size}")

        # Update average fill price
        total_filled = self.filled_size + size
        if total_filled > 0:
            self.filled_price = (
                (self.filled_price * self.filled_size + price * size) / total_filled
            )

        self.filled_size = total_filled
        self.commission += commission
        self.slippage += slippage
        self.filled_time = timestamp

        # Update status
        if self.filled_size >= self.size:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self):
        """Cancel the order."""
        if not self.is_active:
            raise ValueError(f"Cannot cancel inactive order, current status: {self.status}")
        self.status = OrderStatus.CANCELLED

    def reject(self, reason: str = ""):
        """Reject the order."""
        self.status = OrderStatus.REJECTED
        if reason:
            self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'size': self.size,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'created_time': self.created_time,
            'filled_time': self.filled_time,
            'filled_size': self.filled_size,
            'filled_price': self.filled_price,
            'commission': self.commission,
            'slippage': self.slippage,
            'reason': self.reason,
            'remaining_size': self.remaining_size,
            'notional_value': self.notional_value
        }

    def __repr__(self):
        status_sym = {
            OrderStatus.PENDING: "[PENDING]",
            OrderStatus.FILLED: "[FILLED]",
            OrderStatus.PARTIALLY_FILLED: "[PARTIAL]",
            OrderStatus.CANCELLED: "[CANCELLED]",
            OrderStatus.REJECTED: "[REJECTED]",
            OrderStatus.EXPIRED: "[EXPIRED]"
        }
        sym = status_sym.get(self.status, "")
        side_str = "BUY" if self.is_buy else "SELL"

        if self.order_type == OrderType.MARKET:
            price_str = "MARKET"
        elif self.order_type == OrderType.LIMIT:
            price_str = f"@{self.price:.2f}"
        elif self.order_type == OrderType.STOP:
            price_str = f"STOP@{self.stop_price:.2f}"
        else:
            price_str = f"STOP@{self.stop_price:.2f}->@{self.price:.2f}"

        return f"Order({self.order_id}: {side_str} {self.size} {self.symbol} {price_str} {sym})"


class OrderManager:
    """
    Manages order lifecycle including submission, cancellation, and querying.

    Example:
        >>> manager = OrderManager()
        >>> order = manager.create_market_order("AAPL", "buy", 100)
        >>> manager.submit(order)
        >>> pending = manager.get_pending_orders()
    """

    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self._order_counter = 0

    def _generate_order_id(self) -> str:
        self._order_counter += 1
        return f"ORD{self._order_counter:06d}"

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        size: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reason: str = "",
        timestamp: Optional[pd.Timestamp] = None,
        **kwargs
    ) -> Order:
        """
        Create an order.

        Parameters:
            symbol: Instrument symbol.
            side: Direction ('buy' or 'sell').
            order_type: Type ('market', 'limit', 'stop', 'stop_limit').
            size: Quantity.
            price: Limit price.
            stop_price: Stop trigger price.
            reason: Order reason/tag.
            timestamp: Creation time.
            **kwargs: Additional metadata.

        Returns:
            Order: The created order.
        """
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=OrderSide(side.lower()),
            order_type=OrderType(order_type.lower()),
            size=size,
            price=price,
            stop_price=stop_price,
            reason=reason,
            created_time=timestamp,
            metadata=kwargs
        )
        return order

    def create_market_order(
        self,
        symbol: str,
        side: str,
        size: float,
        reason: str = "",
        timestamp: Optional[pd.Timestamp] = None
    ) -> Order:
        """Create a market order."""
        return self.create_order(
            symbol=symbol, side=side, order_type="market",
            size=size, reason=reason, timestamp=timestamp
        )

    def create_limit_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        reason: str = "",
        timestamp: Optional[pd.Timestamp] = None
    ) -> Order:
        """Create a limit order."""
        return self.create_order(
            symbol=symbol, side=side, order_type="limit",
            size=size, price=price, reason=reason, timestamp=timestamp
        )

    def create_stop_order(
        self,
        symbol: str,
        side: str,
        size: float,
        stop_price: float,
        reason: str = "",
        timestamp: Optional[pd.Timestamp] = None
    ) -> Order:
        """Create a stop order."""
        return self.create_order(
            symbol=symbol, side=side, order_type="stop",
            size=size, stop_price=stop_price, reason=reason, timestamp=timestamp
        )

    def submit(self, order: Order) -> str:
        """
        Submit an order to the manager.

        Returns:
            str: Order ID.
        """
        self.orders[order.order_id] = order
        return order.order_id

    def cancel(self, order_id: str) -> bool:
        """
        Cancel an order.

        Returns:
            bool: Whether cancellation was successful.
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.is_active:
            order.cancel()
            return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.orders.get(order_id)

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get active (pending or partially filled) orders."""
        pending = [o for o in self.orders.values() if o.is_active]
        if symbol:
            pending = [o for o in pending if o.symbol == symbol]
        return pending

    def get_filled_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get filled orders."""
        filled = [o for o in self.orders.values() if o.is_filled]
        if symbol:
            filled = [o for o in filled if o.symbol == symbol]
        return filled

    def get_all_orders(self) -> List[Order]:
        return list(self.orders.values())

    def to_dataframe(self) -> pd.DataFrame:
        if not self.orders:
            return pd.DataFrame()
        return pd.DataFrame([o.to_dict() for o in self.orders.values()])

    def clear(self):
        """Clear all orders."""
        self.orders.clear()
        self._order_counter = 0

    def __len__(self) -> int:
        return len(self.orders)

    def __repr__(self):
        pending = len(self.get_pending_orders())
        filled = len(self.get_filled_orders())
        return f"OrderManager(total={len(self)}, pending={pending}, filled={filled})"


def should_trigger_stop(
    stop_price: float,
    side: OrderSide,
    high: float,
    low: float
) -> bool:
    """
    Determine whether a stop order should be triggered.

    Parameters:
        stop_price: Stop price.
        side: Order side.
        high: Bar high price.
        low: Bar low price.

    Returns:
        bool: Whether the stop is triggered.
    """
    if side == OrderSide.SELL:
        # Sell stop: triggered when price drops to or below stop price
        return low <= stop_price
    else:
        # Buy stop (short cover): triggered when price rises to or above stop price
        return high >= stop_price


def should_fill_limit(
    limit_price: float,
    side: OrderSide,
    high: float,
    low: float
) -> bool:
    """
    Determine whether a limit order can be filled.

    Parameters:
        limit_price: Limit price.
        side: Order side.
        high: Bar high price.
        low: Bar low price.

    Returns:
        bool: Whether the limit can be filled.
    """
    if side == OrderSide.BUY:
        # Buy limit: fills when price drops to or below limit
        return low <= limit_price
    else:
        # Sell limit: fills when price rises to or above limit
        return high >= limit_price
