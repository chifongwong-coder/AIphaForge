"""
Broker Simulation

Simulates order execution, slippage, and fill logic.
"""

import warnings
from datetime import time
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .fees import BaseFeeModel, SimpleFeeModel
from .orders import Order, OrderManager, OrderSide, OrderType, should_fill_limit, should_trigger_stop
from .portfolio import Portfolio


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
        stop_fill_pessimistic: bool = True,
        session_end_time: Optional[time] = None,
        immediate_fill_price: str = "close",
        assigned_symbol: Optional[str] = None,
    ):
        self.fee_model = fee_model or SimpleFeeModel()
        self.fill_model = fill_model
        self.slippage_model = slippage_model
        self.partial_fills = partial_fills
        self.volume_limit_pct = volume_limit_pct
        self.check_buying_power = check_buying_power
        self.stop_fill_pessimistic = stop_fill_pessimistic
        self.session_end_time = session_end_time
        self.assigned_symbol = assigned_symbol
        if immediate_fill_price not in ("close", "open", "vwap"):
            raise ValueError(
                f"immediate_fill_price must be 'close', 'open', or 'vwap', "
                f"got {immediate_fill_price!r}"
            )
        self.immediate_fill_price = immediate_fill_price

        # Market impact model (v1.9.4)
        self._impact_model = None  # BaseImpactModel or None
        self._adv: float = 0.0    # updated per-bar by event loop
        self._volatility: float = 0.0  # updated per-bar by event loop

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

    def set_impact_model(self, model: object) -> None:
        """Assign a market impact model and auto-override VOLUME_BASED slippage."""
        self._impact_model = model
        if self.slippage_model == SlippageModel.VOLUME_BASED:
            warnings.warn(
                "VOLUME_BASED slippage auto-replaced with FIXED when "
                "impact_model is set (both model the same effect)."
            )
            self.slippage_model = SlippageModel.FIXED

    # ========== Order Creation ==========

    def create_market_order(
        self,
        symbol: str,
        side: str,
        size: float,
        reason: str = "",
        timestamp: Optional[pd.Timestamp] = None,
        time_in_force: str = "GTC",
    ) -> Order:
        """Create a market order."""
        return self.order_manager.create_market_order(
            symbol, side, size, reason, timestamp,
            time_in_force=time_in_force,
        )

    def create_limit_order(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        reason: str = "",
        timestamp: Optional[pd.Timestamp] = None,
        time_in_force: str = "GTC",
    ) -> Order:
        """Create a limit order."""
        return self.order_manager.create_limit_order(
            symbol, side, size, price, reason, timestamp,
            time_in_force=time_in_force,
        )

    def create_stop_order(
        self,
        symbol: str,
        side: str,
        size: float,
        stop_price: float,
        reason: str = "",
        timestamp: Optional[pd.Timestamp] = None,
        time_in_force: str = "GTC",
    ) -> Order:
        """Create a stop order."""
        return self.order_manager.create_stop_order(
            symbol, side, size, stop_price, reason, timestamp,
            time_in_force=time_in_force,
        )

    def create_trailing_stop_order(
        self,
        symbol: str,
        side: str,
        size: float,
        trail_amount: Optional[float] = None,
        trail_percent: Optional[float] = None,
        initial_price: Optional[float] = None,
        reason: str = "",
        timestamp: Optional[pd.Timestamp] = None,
        time_in_force: str = "GTC",
    ) -> Order:
        """Create a trailing stop order.

        Parameters:
            symbol: Instrument symbol.
            side: Direction ('buy' or 'sell').
            size: Order quantity.
            trail_amount: Absolute trail distance.
            trail_percent: Percentage trail distance (fraction in (0, 1)).
            initial_price: Current market price for computing initial
                stop_price.
            reason: Order reason/tag.
            timestamp: Creation time.
            time_in_force: Order validity type.

        Returns:
            Order: The created trailing stop order.
        """
        return self.order_manager.create_trailing_stop_order(
            symbol, side, size,
            trail_amount=trail_amount,
            trail_percent=trail_percent,
            initial_price=initial_price,
            reason=reason,
            timestamp=timestamp,
            time_in_force=time_in_force,
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

        # Symbol guard for multi-asset mode
        if (self.assigned_symbol is not None
                and order.symbol != self.assigned_symbol):
            order.reject(f"Symbol mismatch: broker assigned to "
                         f"'{self.assigned_symbol}', got '{order.symbol}'")
            self.rejected_orders += 1
            return order.order_id

        # DAY orders must have a created_time for session expiration logic
        if order.time_in_force == "DAY" and order.created_time is None:
            raise ValueError(
                "DAY orders must have created_time set at submission"
            )

        # Buying power check: only for orders that INCREASE exposure.
        # Closing/reducing orders (buy-to-close-short, sell-to-close-long)
        # do not consume buying power and are always allowed.
        needs_buying_power = False
        if self._portfolio:
            pos = self._portfolio.get_position(order.symbol)
            if order.is_buy:
                # Buy + no position or long → opens/increases → needs check
                # Buy + short → closing short → no check
                needs_buying_power = (pos is None or pos.is_flat
                                      or pos.is_long)
            else:
                # Sell + no position or short → opens/increases → needs check
                # Sell + long → closing long → no check
                needs_buying_power = (pos is None or pos.is_flat
                                      or pos.is_short)

        if self.check_buying_power and self._portfolio and needs_buying_power:
            # Block new opens during margin call
            if self._portfolio.is_margin_call:
                order.reject("Margin call in effect — new opens blocked")
                self.rejected_orders += 1
                return order.order_id

            # Estimate price for market orders
            estimated_price = order.price or 0
            if estimated_price <= 0:
                pos = self._portfolio.positions.get(order.symbol)
                if pos and pos.current_price > 0:
                    estimated_price = pos.current_price
            if estimated_price <= 0:
                estimated_price = order.metadata.get(
                    'estimated_price', 0)

            # Check buying power
            if estimated_price > 0:
                estimated_cost = order.size * estimated_price
                if estimated_cost > self._portfolio.buying_power:
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

        Uses a two-phase approach:
        - Pre-phase: expire stale DAY orders from a previous session.
        - Phase 1: process IOC/FOK orders (one-shot semantics).
        - Phase 2: process GTC/DAY orders (normal persistent logic).

        Parameters:
            bar: Bar data containing open, high, low, close, volume.
            timestamp: Timestamp.
            symbol: Instrument symbol.

        Returns:
            List[Order]: Orders filled during this bar.
        """
        filled_orders: List[Order] = []
        pending = self.order_manager.get_pending_orders(symbol)

        # Pre-phase: expire stale DAY orders
        for order in pending:
            if order.symbol != symbol:
                continue
            if order.time_in_force != "DAY":
                continue
            if self._is_day_order_stale(order, timestamp):
                order.expire("day_session_end")

        # Phase 0: Update trailing stop prices before fill checking
        self._update_trailing_stops(bar, symbol)

        # Refresh the pending list after expiring DAY orders
        pending = self.order_manager.get_pending_orders(symbol)

        # Phase 1: IOC and FOK orders (one chance to fill)
        processed = self._process_ioc_fok_orders(bar, timestamp, symbol)
        filled_orders.extend(processed)

        # Phase 2: GTC and DAY orders (normal processing)
        pending = self.order_manager.get_pending_orders(symbol)
        for order in pending:
            if order.symbol != symbol:
                continue
            if order.time_in_force in ("GTC", "DAY"):
                filled = self._try_fill_order(order, bar, timestamp)
                if filled:
                    filled_orders.append(order)
                    self.filled_orders += 1

        return filled_orders

    def _process_ioc_fok_orders(
        self,
        bar: pd.Series,
        timestamp: pd.Timestamp,
        symbol: str,
        fill_price_override: Optional[float] = None,
    ) -> List[Order]:
        """Process pending IOC and FOK orders.

        Shared helper used by both ``process_bar`` (Phase 1) and
        ``process_immediate_orders`` (second pass).

        Parameters:
            bar: Bar OHLCV data.
            timestamp: Current bar timestamp.
            symbol: Instrument symbol.
            fill_price_override: If provided, market orders use this price
                instead of the configured fill model.  Limit/stop orders
                keep their own price logic.

        Returns:
            List[Order]: All processed IOC/FOK orders (filled, partially
                expired, or fully expired).
        """
        processed: List[Order] = []
        pending = self.order_manager.get_pending_orders(symbol)

        for order in pending:
            if order.symbol != symbol:
                continue
            if order.time_in_force == "IOC":
                if fill_price_override is not None and order.order_type == OrderType.MARKET:
                    filled = self._execute_fill(
                        order, fill_price_override, order.size, timestamp,
                        bar.get('volume', float('inf')),
                    )
                else:
                    filled = self._try_fill_order(order, bar, timestamp)
                if filled and order.is_filled:
                    processed.append(order)
                    self.filled_orders += 1
                elif filled and order.is_active:
                    # Partially filled IOC: expire the remainder
                    order.expire("ioc_timeout")
                    processed.append(order)
                else:
                    # Not filled at all
                    order.expire("ioc_timeout")
                    processed.append(order)
            elif order.time_in_force == "FOK":
                volume = bar.get('volume', float('inf'))
                if volume < float('inf'):
                    available = volume * self.volume_limit_pct
                    if available < order.size:
                        order.expire("fok_volume")
                        processed.append(order)
                        continue
                # FOK pre-check passed: attempt full fill
                if fill_price_override is not None and order.order_type == OrderType.MARKET:
                    filled = self._execute_fill(
                        order, fill_price_override, order.size, timestamp,
                        volume, bypass_partial_clip=True,
                    )
                else:
                    filled = self._try_fill_order(
                        order, bar, timestamp, bypass_partial_clip=True,
                    )
                if filled:
                    processed.append(order)
                    self.filled_orders += 1
                else:
                    order.expire("fok_price")
                    processed.append(order)

        return processed

    def process_immediate_orders(
        self,
        bar: pd.Series,
        timestamp: pd.Timestamp,
        symbol: str = "default",
    ) -> List[Order]:
        """Process pending IOC/FOK orders submitted during the current bar.

        This is the "second pass" for same-bar IOC/FOK semantics.  Only
        IOC and FOK orders participate; GTC/DAY orders are untouched.

        The fill price for market orders is determined by
        ``self.immediate_fill_price``:
            - ``"close"``: current bar's close price.
            - ``"open"``: current bar's open price.
            - ``"vwap"``: ``(high + low + close) / 3``.

        Parameters:
            bar: Current bar OHLCV data.
            timestamp: Current bar timestamp.
            symbol: Instrument symbol.

        Returns:
            List[Order]: Orders that were processed (filled/expired).
        """
        if self.immediate_fill_price == "open":
            price = bar['open']
        elif self.immediate_fill_price == "vwap":
            price = (bar['high'] + bar['low'] + bar['close']) / 3
        else:  # "close" (default)
            price = bar['close']

        return self._process_ioc_fok_orders(
            bar, timestamp, symbol, fill_price_override=price,
        )

    def _is_day_order_stale(
        self, order: Order, current_timestamp: pd.Timestamp
    ) -> bool:
        """Check whether a DAY order belongs to a previous session.

        If ``session_end_time`` is configured, the order is stale when the
        current bar timestamp is past that time on the same day *or* on a
        later calendar day.  Without ``session_end_time``, stale simply
        means the order's creation date differs from the bar's date.
        """
        if order.created_time is None:
            return False
        if self.session_end_time is not None:
            # Session-based: stale if bar is past session_end_time on the
            # creation day, or on a later calendar day.
            if current_timestamp.date() > order.created_time.date():
                return True
            if (
                current_timestamp.date() == order.created_time.date()
                and current_timestamp.time() > self.session_end_time
            ):
                return True
            return False
        # Calendar-day based: stale when date changes
        return current_timestamp.date() != order.created_time.date()

    def _update_trailing_stops(
        self,
        bar: pd.Series,
        symbol: str,
    ) -> None:
        """Ratchet trailing stop prices using current bar data.

        For sell trailing stops (long protection):
            new_stop = high * (1 - trail_percent)  [or high - trail_amount]
            stop_price = max(stop_price, new_stop)  # only ratchets UP

        For buy trailing stops (short protection):
            new_stop = low * (1 + trail_percent)  [or low + trail_amount]
            stop_price = min(stop_price, new_stop)  # only ratchets DOWN

        Called at the start of process_bar, before fill checking.
        """
        pending = self.order_manager.get_pending_orders(symbol)
        for order in pending:
            if order.order_type != OrderType.TRAILING_STOP:
                continue
            if order.side == OrderSide.SELL:
                high = bar['high']
                if order.trail_percent is not None:
                    new_stop = high * (1 - order.trail_percent)
                else:
                    new_stop = high - order.trail_amount
                order.stop_price = max(order.stop_price, new_stop)
            else:  # BUY (short protection)
                low = bar['low']
                if order.trail_percent is not None:
                    new_stop = low * (1 + order.trail_percent)
                else:
                    new_stop = low + order.trail_amount
                order.stop_price = min(order.stop_price, new_stop)

    def _try_fill_order(
        self,
        order: Order,
        bar: pd.Series,
        timestamp: pd.Timestamp,
        bypass_partial_clip: bool = False
    ) -> bool:
        """
        Attempt to fill an order against a bar.

        Parameters:
            order: The order to fill.
            bar: Bar OHLCV data.
            timestamp: Current bar timestamp.
            bypass_partial_clip: If True, skip the partial_fills volume
                clipping inside _execute_fill (used by FOK after its own
                pre-check).

        Returns:
            bool: Whether (any part of) the order was filled.
        """
        high = bar['high']
        low = bar['low']
        volume = bar.get('volume', float('inf'))

        if order.order_type == OrderType.MARKET:
            fill_price = self._get_fill_price(order, bar)
            return self._execute_fill(order, fill_price, order.size, timestamp, volume, bypass_partial_clip)

        elif order.order_type == OrderType.LIMIT:
            if should_fill_limit(order.price, order.side, high, low):
                fill_price = order.price
                return self._execute_fill(order, fill_price, order.size, timestamp, volume, bypass_partial_clip)

        elif order.order_type in (OrderType.STOP, OrderType.TRAILING_STOP):
            if should_trigger_stop(order.stop_price, order.side, high, low):
                fill_price = self._get_stop_fill_price(order, bar)
                return self._execute_fill(order, fill_price, order.size, timestamp, volume, bypass_partial_clip)

        elif order.order_type == OrderType.STOP_LIMIT:
            if should_trigger_stop(order.stop_price, order.side, high, low):
                if should_fill_limit(order.price, order.side, high, low):
                    fill_price = order.price
                    return self._execute_fill(order, fill_price, order.size, timestamp, volume, bypass_partial_clip)
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
        volume: float,
        bypass_partial_clip: bool = False
    ) -> bool:
        """
        Execute an order fill.

        Parameters:
            order: The order to fill.
            price: Fill price.
            size: Requested fill quantity.
            timestamp: Fill timestamp.
            volume: Bar volume.
            bypass_partial_clip: If True, skip the partial_fills volume
                clipping (used by FOK which performs its own pre-check).

        Returns:
            bool: Whether the fill was successful.
        """
        # Volume limit check (skipped for FOK which does its own pre-check)
        if not bypass_partial_clip and self.partial_fills and volume < float('inf'):
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

        # Apply market impact (v1.9.4)
        if self._impact_model is not None and self._adv > 0:
            impact = self._impact_model.estimate_impact(
                size, adjusted_price, self._adv, self._volatility)
            if order.is_buy:
                adjusted_price *= (1 + impact)
            else:
                adjusted_price *= (1 - impact)
            # Clamp limit/stop-limit orders to their limit price
            # (impact cannot make fills worse than the limit guarantee)
            if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
                if order.is_buy and order.price is not None:
                    adjusted_price = min(adjusted_price, order.price)
                elif not order.is_buy and order.price is not None:
                    adjusted_price = max(adjusted_price, order.price)
            if 'market_impact_bps' not in order.metadata:
                order.metadata['market_impact_bps'] = 0.0
            order.metadata['market_impact_bps'] += impact * 10000

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

    def get_orders_with_fills(self, symbol: Optional[str] = None) -> List[Order]:
        """Get orders that have any fills (FILLED + PARTIALLY_EXPIRED).

        Passthrough to ``OrderManager.get_orders_with_fills``.
        """
        return self.order_manager.get_orders_with_fills(symbol)

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
