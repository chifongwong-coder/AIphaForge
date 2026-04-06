"""
Broker Simulation

Simulates order execution, slippage, and fill logic.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from enum import Enum

from .orders import (
    Order, OrderType, OrderSide, OrderStatus, OrderManager,
    should_trigger_stop, should_fill_limit
)
from .fees import BaseFeeModel, SimpleFeeModel
from .portfolio import Portfolio
from .results import Trade


class FillModel(Enum):
    """Fill price model."""
    NEXT_BAR_OPEN = "next_bar_open"    # Next bar's open price
    CURRENT_CLOSE = "current_close"    # Current bar's close price
    VWAP = "vwap"                      # Approximate VWAP
    WORST_CASE = "worst_case"          # Worst case (buy at high, sell at low)


class SlippageModel(Enum):
    """Slippage model."""
    FIXED = "fixed"
    VOLUME_BASED = "volume_based"
    VOLATILITY_BASED = "volatility_based"


class Broker:
    """
    Broker simulator.

    Handles order execution, fill simulation, and fee calculation.

    Attributes:
        fee_model: Fee model.
        fill_model: Fill price model.
        slippage_model: Slippage model.
        partial_fills: Whether partial fills are supported.
        volume_limit_pct: Max order size as fraction of bar volume.

    Example:
        >>> broker = Broker(fee_model=ChinaAShareFeeModel())
        >>> order = broker.create_market_order("AAPL", "buy", 100)
        >>> broker.submit_order(order, timestamp)
        >>> filled = broker.process_bar(bar_data, timestamp)
    """

    def __init__(
        self,
        fee_model: Optional[BaseFeeModel] = None,
        fill_model: FillModel = FillModel.NEXT_BAR_OPEN,
        slippage_model: SlippageModel = SlippageModel.FIXED,
        partial_fills: bool = False,
        volume_limit_pct: float = 0.1,
        check_buying_power: bool = True,
        stop_fill_pessimistic: bool = True
    ):
        self.fee_model = fee_model or SimpleFeeModel()
        self.fill_model = fill_model
        self.slippage_model = slippage_model
        self.partial_fills = partial_fills
        self.volume_limit_pct = volume_limit_pct
        self.check_buying_power = check_buying_power
        self.stop_fill_pessimistic = stop_fill_pessimistic

        # Order management
        self.order_manager = OrderManager()

        # Associated portfolio (for buying power checks)
        self._portfolio: Optional[Portfolio] = None

        # Statistics
        self.total_orders = 0
        self.filled_orders = 0
        self.rejected_orders = 0

    def set_portfolio(self, portfolio: Portfolio):
        """Set the associated portfolio."""
        self._portfolio = portfolio

    # ========== Order Creation ==========

    def create_market_order(
        self,
        symbol: str,
        side: str,
        size: float,
        reason: str = "",
        timestamp: Optional[pd.Timestamp] = None
    ) -> Order:
        """Create a market order."""
        return self.order_manager.create_market_order(
            symbol, side, size, reason, timestamp
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
        return self.order_manager.create_limit_order(
            symbol, side, size, price, reason, timestamp
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
        return self.order_manager.create_stop_order(
            symbol, side, size, stop_price, reason, timestamp
        )

    # ========== Order Submission ==========

    def submit_order(
        self,
        order: Order,
        timestamp: Optional[pd.Timestamp] = None
    ) -> str:
        """
        Submit an order.

        Parameters:
            order: Order object.
            timestamp: Submission time.

        Returns:
            str: Order ID.
        """
        if timestamp and order.created_time is None:
            order.created_time = timestamp

        # Buying power check
        if self.check_buying_power and self._portfolio and order.is_buy:
            estimated_price = order.price or 0  # Market order price unknown
            if estimated_price > 0:
                if not self._portfolio.check_buying_power(order.size, estimated_price):
                    order.reject("Insufficient buying power")
                    self.rejected_orders += 1
                    return order.order_id

        self.order_manager.submit(order)
        self.total_orders += 1
        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        return self.order_manager.cancel(order_id)

    def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all orders (optionally filtered by symbol)."""
        pending = self.order_manager.get_pending_orders(symbol)
        for order in pending:
            order.cancel()

    # ========== Order Execution ==========

    def process_bar(
        self,
        bar: pd.Series,
        timestamp: pd.Timestamp,
        symbol: str = "default"
    ) -> List[Order]:
        """
        Process a bar and attempt to fill pending orders.

        Parameters:
            bar: Bar data containing open, high, low, close, volume.
            timestamp: Timestamp.
            symbol: Instrument symbol.

        Returns:
            List[Order]: Orders filled during this bar.
        """
        filled_orders = []
        pending = self.order_manager.get_pending_orders(symbol)

        for order in pending:
            if order.symbol != symbol:
                continue

            filled = self._try_fill_order(order, bar, timestamp)
            if filled:
                filled_orders.append(order)
                self.filled_orders += 1

        return filled_orders

    def _try_fill_order(
        self,
        order: Order,
        bar: pd.Series,
        timestamp: pd.Timestamp
    ) -> bool:
        """
        Attempt to fill an order against a bar.

        Returns:
            bool: Whether the order was filled.
        """
        high = bar['high']
        low = bar['low']

        if order.order_type == OrderType.MARKET:
            fill_price = self._get_fill_price(order, bar)
            return self._execute_fill(order, fill_price, order.size, timestamp, bar.get('volume', float('inf')))

        elif order.order_type == OrderType.LIMIT:
            if should_fill_limit(order.price, order.side, high, low):
                fill_price = order.price
                return self._execute_fill(order, fill_price, order.size, timestamp, bar.get('volume', float('inf')))

        elif order.order_type == OrderType.STOP:
            if should_trigger_stop(order.stop_price, order.side, high, low):
                # Triggered: fill at market price
                fill_price = self._get_stop_fill_price(order, bar)
                return self._execute_fill(order, fill_price, order.size, timestamp, bar.get('volume', float('inf')))

        elif order.order_type == OrderType.STOP_LIMIT:
            if should_trigger_stop(order.stop_price, order.side, high, low):
                # Stop triggered: check if limit can also be filled on this bar
                if should_fill_limit(order.price, order.side, high, low):
                    fill_price = order.price
                    return self._execute_fill(order, fill_price, order.size, timestamp, bar.get('volume', float('inf')))
                else:
                    # Stop triggered but limit not filled: convert to limit order
                    order.order_type = OrderType.LIMIT

        return False

    def _get_fill_price(self, order: Order, bar: pd.Series) -> float:
        """Get fill price for a market order."""
        if self.fill_model == FillModel.NEXT_BAR_OPEN:
            return bar['open']
        elif self.fill_model == FillModel.CURRENT_CLOSE:
            return bar['close']
        elif self.fill_model == FillModel.VWAP:
            return (bar['high'] + bar['low'] + bar['close']) / 3
        elif self.fill_model == FillModel.WORST_CASE:
            if order.is_buy:
                return bar['high']
            else:
                return bar['low']
        return bar['open']

    def _get_stop_fill_price(self, order: Order, bar: pd.Series) -> float:
        """Get fill price after a stop is triggered."""
        if self.stop_fill_pessimistic:
            # Conservative: fill at bar extreme (worst case for stops)
            if order.is_sell:
                return min(order.stop_price, bar['low'])
            else:
                return max(order.stop_price, bar['high'])
        else:
            # Realistic: fill at stop price, or gap open if price gaps past
            if order.is_sell:
                if bar['open'] <= order.stop_price:
                    return bar['open']  # Gapped past stop
                return order.stop_price
            else:
                if bar['open'] >= order.stop_price:
                    return bar['open']  # Gapped past stop
                return order.stop_price

    def _execute_fill(
        self,
        order: Order,
        price: float,
        size: float,
        timestamp: pd.Timestamp,
        volume: float
    ) -> bool:
        """
        Execute an order fill.

        Returns:
            bool: Whether the fill was successful.
        """
        # Volume limit check
        if self.partial_fills and volume < float('inf'):
            max_fill = volume * self.volume_limit_pct
            if size > max_fill:
                size = max_fill
                if size <= 0:
                    return False

        # Calculate slippage
        slippage = self._calculate_slippage(price, size, order.side, volume)

        # Adjust fill price for slippage
        if order.is_buy:
            adjusted_price = price + slippage / size if size > 0 else price
        else:
            adjusted_price = price - slippage / size if size > 0 else price

        # Calculate commission
        commission = self.fee_model.calculate_commission(
            adjusted_price, size, order.side.value
        )

        # Execute fill
        order.fill(
            price=adjusted_price,
            size=size,
            timestamp=timestamp,
            commission=commission,
            slippage=slippage
        )

        return True

    def _calculate_slippage(
        self,
        price: float,
        size: float,
        side: OrderSide,
        volume: float
    ) -> float:
        """Calculate slippage cost."""
        if self.slippage_model == SlippageModel.FIXED:
            return self.fee_model.calculate_slippage(price, size, side.value, volume)

        elif self.slippage_model == SlippageModel.VOLUME_BASED:
            # Volume-based: larger orders relative to volume get more slippage
            base_slippage = self.fee_model.calculate_slippage(price, size, side.value, volume)
            if volume > 0:
                volume_ratio = size / volume
                multiplier = 1 + volume_ratio * 10
                return base_slippage * multiplier
            return base_slippage

        elif self.slippage_model == SlippageModel.VOLATILITY_BASED:
            # Volatility-based (simplified: falls back to fixed)
            return self.fee_model.calculate_slippage(price, size, side.value, volume)

        return self.fee_model.calculate_slippage(price, size, side.value, volume)

    # ========== Query Methods ==========

    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        return self.order_manager.get_pending_orders(symbol)

    def get_filled_orders(self, symbol: Optional[str] = None) -> List[Order]:
        return self.order_manager.get_filled_orders(symbol)

    def get_order(self, order_id: str) -> Optional[Order]:
        return self.order_manager.get_order(order_id)

    def get_orders_df(self) -> pd.DataFrame:
        return self.order_manager.to_dataframe()

    # ========== Reset and Stats ==========

    def reset(self):
        """Reset broker state."""
        self.order_manager.clear()
        self.total_orders = 0
        self.filled_orders = 0
        self.rejected_orders = 0

    def get_stats(self) -> Dict:
        """Get broker statistics."""
        return {
            'total_orders': self.total_orders,
            'filled_orders': self.filled_orders,
            'rejected_orders': self.rejected_orders,
            'pending_orders': len(self.get_pending_orders()),
            'fill_rate': self.filled_orders / self.total_orders if self.total_orders > 0 else 0
        }

    def __repr__(self):
        pending = len(self.get_pending_orders())
        return (f"Broker(fee_model={self.fee_model.name}, "
                f"pending={pending}, filled={self.filled_orders})")


class SimpleBroker:
    """
    Simplified broker for vectorized backtesting.

    Does not maintain order state; only calculates costs.

    Example:
        >>> broker = SimpleBroker(commission_rate=0.001)
        >>> cost = broker.calculate_trade_cost(100.0, 100, 'buy')
    """

    def __init__(
        self,
        fee_model: Optional[BaseFeeModel] = None,
        slippage_pct: float = 0.001
    ):
        self.fee_model = fee_model or SimpleFeeModel(slippage_pct=slippage_pct)

    def calculate_trade_cost(
        self,
        price: float,
        size: float,
        side: str,
        volume: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate trade cost.

        Returns:
            Tuple[float, float]: (commission, slippage).
        """
        commission = self.fee_model.calculate_commission(price, size, side)
        slippage = self.fee_model.calculate_slippage(price, size, side, volume)
        return commission, slippage

    def get_execution_price(
        self,
        price: float,
        side: str
    ) -> float:
        """Get execution price after slippage."""
        return self.fee_model.get_execution_price(price, side)
