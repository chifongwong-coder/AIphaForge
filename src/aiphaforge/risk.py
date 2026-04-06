"""
Risk Management Framework

Defines the abstract base class for risk managers and the RiskSignal data
structure used to communicate risk decisions within the engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd


@dataclass
class RiskSignal:
    """
    A risk management signal produced by a risk manager.

    Attributes:
        severity: Signal severity ('info', 'warning', 'critical').
        action: Recommended action ('none', 'reduce', 'reject_new', 'close_all').
        message: Human-readable description of the risk event.
    """
    severity: str    # 'info', 'warning', 'critical'
    action: str      # 'none', 'reduce', 'reject_new', 'close_all'
    message: str


class BaseRiskManager(ABC):
    """
    Abstract base class for risk managers.

    Implement this interface to plug custom risk management logic into the
    backtest engine.  The engine calls these methods during signal processing
    to enforce position-sizing rules and portfolio-level risk limits.
    """

    @abstractmethod
    def initialize(self, initial_capital: float) -> None:
        """
        Initialize the risk manager with the starting capital.

        Parameters:
            initial_capital: Portfolio starting capital.
        """
        ...

    @abstractmethod
    def sync_from_portfolio(self, portfolio) -> None:
        """
        Synchronize internal state from the current portfolio.

        Called before ``check_and_apply_risk_rules`` so the risk manager
        has up-to-date position and equity information.

        Parameters:
            portfolio: The current Portfolio instance.
        """
        ...

    @abstractmethod
    def check_and_apply_risk_rules(
        self,
        portfolio,
        market_data: Dict[str, pd.DataFrame],
    ) -> List[RiskSignal]:
        """
        Evaluate portfolio-level risk rules.

        Parameters:
            portfolio: The current Portfolio instance.
            market_data: Dict mapping symbol to a DataFrame of historical
                bars visible up to (and including) the current bar.

        Returns:
            A list of RiskSignal objects.  Signals with severity 'critical'
            and action 'reject_new' or 'close_all' will cause the engine
            to skip signal processing for the current bar.
        """
        ...

    @abstractmethod
    def calculate_position_size(
        self,
        symbol: str,
        signal: int,
        current_price: float,
        market_data: pd.DataFrame,
    ) -> float:
        """
        Calculate the desired position size for a new signal.

        Parameters:
            symbol: Instrument symbol.
            signal: Trade direction (1 = long, -1 = short).
            current_price: Current market price.
            market_data: Historical bar data for *symbol* visible up to
                (and including) the current bar.

        Returns:
            Target position size (signed: positive for long, negative for short).
        """
        ...
