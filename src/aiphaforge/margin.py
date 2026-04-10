"""
Margin Trading
==============

Margin account model, margin call rules, and periodic cost models
for leveraged and short-selling simulations.

The unified margin mode uses ``initial_margin_ratio`` (IMR) as the
single parameter: IMR=1.0 is cash-only (v0.7 behavior), IMR=0.5 is
2x leverage, IMR=0.1 is 10x leverage.
"""

from dataclasses import dataclass
from typing import Any, Dict, Set

from .orders import OrderStatus

# ---------------------------------------------------------------------------
# MarginConfig
# ---------------------------------------------------------------------------

@dataclass
class MarginConfig:
    """Margin account configuration.

    Parameters:
        initial_margin_ratio: Fraction of position value required as equity
            when opening. 1.0 = cash-only, 0.5 = 2x leverage.
        maintenance_margin_ratio: Minimum equity as fraction of position
            value. 0.0 = no margin call.
        borrowing_rate: Annual interest rate on borrowed funds.
        stop_on_negative_equity: Terminate backtest if equity goes negative
            (gap risk / 穿仓).
    """
    initial_margin_ratio: float = 1.0
    maintenance_margin_ratio: float = 0.0
    borrowing_rate: float = 0.0
    stop_on_negative_equity: bool = False


# ---------------------------------------------------------------------------
# Portfolio-level exit rule base
# ---------------------------------------------------------------------------

class BasePortfolioExitRule:
    """Abstract base for portfolio-level exit rules (e.g., margin call).

    Unlike ``BaseExitRule`` (per-symbol), this is called once per bar
    with access to the full portfolio and all brokers.
    """

    def check_portfolio(
        self,
        portfolio: Any,
        brokers: Dict[str, Any],
        symbols: list,
        prices: Dict[str, float],
        timestamp: Any,
    ) -> None:
        """Portfolio-level exit check. Called once per bar at step 3b."""
        pass


# ---------------------------------------------------------------------------
# MarginCallExitRule
# ---------------------------------------------------------------------------

class MarginCallExitRule(BasePortfolioExitRule):
    """Force-liquidate positions when a margin call is triggered.

    Liquidation orders are GTC market orders that fill at the next bar's
    open price. Within a single bar, all positions receive liquidation
    orders (since ``is_margin_call`` does not update until orders fill).
    Across bars, deduplication prevents re-submitting for positions that
    already have pending liquidation orders.

    Parameters:
        liquidation_strategy: ``"largest_first"`` (submit in descending
            market value order) or ``"all"`` (submit for all positions).
    """

    def __init__(self, liquidation_strategy: str = "largest_first"):
        self.strategy = liquidation_strategy
        self._pending_liquidations: Set[str] = set()

    def check_portfolio(self, portfolio, brokers, symbols, prices, ts):
        if not portfolio.is_margin_call:
            self._pending_liquidations.clear()
            return

        positions = list(portfolio.positions.items())
        if self.strategy == "largest_first":
            positions = sorted(
                positions,
                key=lambda x: abs(x[1].market_value),
                reverse=True,
            )

        for sym, pos in positions:
            if pos.is_flat or sym not in brokers:
                continue
            if sym in self._pending_liquidations:
                continue
            side = "sell" if pos.is_long else "buy"
            order = brokers[sym].create_market_order(
                sym, side, abs(pos.size), "margin_call", ts)
            order.metadata['estimated_price'] = prices.get(sym, 0)
            brokers[sym].submit_order(order, ts)
            if order.status != OrderStatus.REJECTED:
                self._pending_liquidations.add(sym)


# ---------------------------------------------------------------------------
# Periodic cost models
# ---------------------------------------------------------------------------

class PeriodicCostModel:
    """Abstract base for per-bar periodic costs (borrowing, funding).

    Subclasses implement :meth:`calculate_cost` to return the dollar
    cost to deduct from cash for a given position on a given bar.
    """

    def calculate_cost(
        self,
        position: Any,
        price: float,
        timestamp: Any,
        margin_config: MarginConfig,
    ) -> float:
        """Return cost to deduct this bar. Positive = cost."""
        return 0.0


class BorrowingCostModel(PeriodicCostModel):
    """Daily interest on borrowed funds.

    For leveraged longs: borrowed amount is fixed at entry
    (``entry_value * (1 - IMR)``), not affected by price changes.
    This ensures fair comparison across strategies — winning positions
    are not penalized with higher interest.

    For shorts: borrowed = current market value of shorted shares
    (real brokers charge on current value since you must return shares
    at current price).
    """

    def calculate_cost(self, position, price, timestamp, margin_config):
        if margin_config is None:
            return 0.0  # no margin = no borrowing
        annual_rate = margin_config.borrowing_rate
        if annual_rate <= 0:
            return 0.0
        daily_rate = annual_rate / 365
        imr = margin_config.initial_margin_ratio
        if position.is_short:
            # Short: borrow cost on current market value of shares
            borrowed = abs(position.market_value)
        else:
            # Long: borrow cost on entry-time borrowed amount (fixed)
            entry_value = abs(position.size * position.avg_entry_price)
            borrowed = entry_value * (1 - imr)
        return max(0.0, borrowed * daily_rate)


class FundingRateModel(PeriodicCostModel):
    """Perpetual futures funding rate (applied per bar).

    The ``funding_rate_per_bar`` must be scaled by the user to match
    bar frequency:
    - 8h bars (standard crypto): use raw rate (e.g. 0.0001)
    - Daily bars: multiply by 3 (3 funding periods per day)
    - 1-min bars: divide by 480 (480 minutes per 8h)
    """

    def __init__(self, funding_rate_per_bar: float = 0.0001):
        self.funding_rate_per_bar = funding_rate_per_bar

    def calculate_cost(self, position, price, timestamp, margin_config):
        return abs(position.notional_value) * self.funding_rate_per_bar
