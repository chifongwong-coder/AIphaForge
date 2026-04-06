"""
Position Sizing

Pluggable position sizing strategies for the backtest engine.
"""

class BasePositionSizer:
    """Abstract base for position sizing.

    Subclasses implement ``calculate()`` to determine the target position
    size given the current equity, price, and direction.
    """

    def calculate(
        self,
        equity: float,
        price: float,
        direction: int,
        max_position_size: float,
    ) -> float:
        """Calculate the target position size in shares.

        Parameters:
            equity: Current portfolio equity.
            price: Current price per share.
            direction: 1 for long, -1 for short.
            max_position_size: Maximum position as a fraction of equity.

        Returns:
            Signed target position size (positive=long, negative=short).
        """
        raise NotImplementedError


class FractionSizer(BasePositionSizer):
    """Fixed fraction of equity.

    Parameters:
        fraction: Fraction of equity to allocate (e.g. 0.95 = 95%).
    """

    def __init__(self, fraction: float):
        self.fraction = fraction

    def calculate(
        self,
        equity: float,
        price: float,
        direction: int,
        max_position_size: float,
    ) -> float:
        if price <= 0:
            return 0.0
        size = equity * self.fraction / price
        max_size = equity * max_position_size / price
        size = min(size, max_size)
        return size * direction


class FixedSizer(BasePositionSizer):
    """Fixed quantity regardless of equity.

    Parameters:
        size: Fixed number of shares.
    """

    def __init__(self, size: float):
        self.size = size

    def calculate(
        self,
        equity: float,
        price: float,
        direction: int,
        max_position_size: float,
    ) -> float:
        if price <= 0:
            return 0.0
        max_size = equity * max_position_size / price
        size = min(self.size, max_size)
        return size * direction


class AllInSizer(BasePositionSizer):
    """All-in position sizing (same formula as FractionSizer).

    Parameters:
        fraction: Fraction of equity to allocate (default 0.95).
    """

    def __init__(self, fraction: float = 0.95):
        self.fraction = fraction

    def calculate(
        self,
        equity: float,
        price: float,
        direction: int,
        max_position_size: float,
    ) -> float:
        if price <= 0:
            return 0.0
        size = equity * self.fraction / price
        max_size = equity * max_position_size / price
        size = min(size, max_size)
        return size * direction
