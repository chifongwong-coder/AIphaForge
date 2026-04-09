"""
Capital Allocator
=================

Distributes available capital among multiple assets when simultaneous
signals occur.  The allocator is called once per timestamp with all
non-zero signals and returns a budget (max capital) per symbol.

The return type ``Dict[str, Optional[float]]`` uses ``None`` to mean
"no budget constraint" (e.g. sell/close signals that free capital).
A ``float`` value caps the incremental trade cost for that symbol.

This interface is forward-compatible with v0.8 margin/leverage:
a margin-aware allocator can return margin-based budgets without
any change to the core event loop or budget-cap logic.
"""

from typing import Any, Dict, Optional


class BaseCapitalAllocator:
    """Abstract base for capital allocation strategies.

    Subclasses implement :meth:`allocate` to distribute capital among
    symbols that have active signals on a given timestamp.
    """

    def allocate(
        self,
        signals: Dict[str, int],
        prices: Dict[str, float],
        portfolio: Any,
        config: Any,
    ) -> Dict[str, Optional[float]]:
        """Return budget per symbol.

        Parameters:
            signals: Non-zero signals for this timestamp.
                     Positive = buy, negative = sell.
            prices: Last known price per symbol.
            portfolio: Portfolio instance (use ``portfolio.cash``).
            config: BacktestConfig instance.

        Returns:
            Dict mapping each signal's symbol to either a ``float``
            budget (max capital for the trade) or ``None`` (no limit).
        """
        raise NotImplementedError


class EqualWeightAllocator(BaseCapitalAllocator):
    """Divide available cash equally among buy signals.

    Sell signals receive ``None`` (no budget constraint).
    """

    def allocate(
        self,
        signals: Dict[str, int],
        prices: Dict[str, float],
        portfolio: Any,
        config: Any,
    ) -> Dict[str, Optional[float]]:
        buys = [s for s, sig in signals.items() if sig > 0]
        budgets: Dict[str, Optional[float]] = {}
        if buys:
            per_symbol = portfolio.cash / len(buys)
            for s in buys:
                budgets[s] = per_symbol
        for s, sig in signals.items():
            if sig <= 0:
                budgets[s] = None
        return budgets


class FixedWeightAllocator(BaseCapitalAllocator):
    """Allocate cash proportional to predefined per-symbol weights.

    Symbols not in *weights* receive equal share of the remainder.
    Sell signals receive ``None``.

    Parameters:
        weights: Mapping of symbol to weight (0, 1].  Weights are
                 fractions of available cash, not of total equity.
    """

    def __init__(self, weights: Dict[str, float]) -> None:
        for sym, w in weights.items():
            if w <= 0:
                raise ValueError(
                    f"Weight for '{sym}' must be > 0, got {w}")
        total = sum(weights.values())
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"Sum of weights ({total:.4f}) exceeds 1.0")
        self.weights = dict(weights)

    def allocate(
        self,
        signals: Dict[str, int],
        prices: Dict[str, float],
        portfolio: Any,
        config: Any,
    ) -> Dict[str, Optional[float]]:
        cash = portfolio.cash
        buys = [s for s, sig in signals.items() if sig > 0]
        budgets: Dict[str, Optional[float]] = {}

        # Pre-compute weighted vs unweighted split
        weighted_buys = [b for b in buys if b in self.weights]
        unweighted_buys = [b for b in buys if b not in self.weights]
        allocated_frac = sum(self.weights[b] for b in weighted_buys)
        remainder = cash * max(0.0, 1.0 - allocated_frac)
        per_unweighted = (
            remainder / len(unweighted_buys) if unweighted_buys else 0
        )

        for s in buys:
            if s in self.weights:
                budgets[s] = cash * self.weights[s]
            else:
                budgets[s] = per_unweighted

        for s, sig in signals.items():
            if sig <= 0:
                budgets[s] = None
        return budgets


class ProRataAllocator(BaseCapitalAllocator):
    """Allocate cash proportional to signal strength for buy signals.

    Stronger signals receive a larger share of available cash.
    Sell signals receive ``None``.
    """

    def allocate(
        self,
        signals: Dict[str, int],
        prices: Dict[str, float],
        portfolio: Any,
        config: Any,
    ) -> Dict[str, Optional[float]]:
        buys = {s: abs(sig) for s, sig in signals.items() if sig > 0}
        budgets: Dict[str, Optional[float]] = {}
        if buys:
            total_strength = sum(buys.values())
            if total_strength > 0:
                cash = portfolio.cash
                for s, strength in buys.items():
                    budgets[s] = cash * strength / total_strength
            else:
                per_symbol = portfolio.cash / len(buys)
                for s in buys:
                    budgets[s] = per_symbol
        for s, sig in signals.items():
            if sig <= 0:
                budgets[s] = None
        return budgets
