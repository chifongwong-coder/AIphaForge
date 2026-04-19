"""
Risk Management Framework

Defines the abstract base class for risk managers, composable risk rules,
and the RiskSignal data structure used to communicate risk decisions
within the engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Composable Risk Rules (v1.1)
# ---------------------------------------------------------------------------


class BaseRiskRule:
    """Single risk rule. Composable via CompositeRiskManager.

    Subclass and override ``check`` for event-driven mode and/or
    ``apply_vectorized`` for vectorized mode.  Override ``reset``
    if the rule carries state across bars.
    """

    name: str = "Unknown"

    def reset(self) -> None:
        """Reset state for a new backtest run. Override if stateful."""
        pass

    def check(self, portfolio, prices: Dict[str, float],
              timestamp) -> Optional[RiskSignal]:
        """Check this rule against the current portfolio state.

        Parameters:
            portfolio: Current Portfolio instance.
            prices: Dict mapping symbol to current price.
            timestamp: Current bar timestamp.

        Returns:
            A RiskSignal if the rule triggers, or None (pass).
        """
        return None

    def apply_vectorized(self, equity_curve: pd.Series,
                         positions: pd.Series,
                         data: pd.DataFrame) -> pd.Series:
        """Vectorized mode: modify positions series. Default: no-op."""
        return positions


class CompositeRiskManager(BaseRiskManager):
    """Compose multiple risk rules into one manager.

    Inherits :class:`BaseRiskManager` so it is a drop-in replacement for
    user-defined risk managers passed via ``BacktestEngine(risk_manager=...)``.
    The recommended way to use a composite, however, is via
    ``BacktestEngine(risk_rules=...)`` because the engine has a more
    direct fast-path for the rule-based interface.
    """

    def __init__(self, rules: List[BaseRiskRule]):
        self.rules = list(rules)
        self.history: List[Dict] = []

    # ------------------------------------------------------------------
    # BaseRiskManager interface (used when passed via risk_manager=)
    # ------------------------------------------------------------------

    def initialize(self, initial_capital: float) -> None:
        """No-op: composite rules don't track total capital."""
        pass

    def sync_from_portfolio(self, portfolio) -> None:
        """No-op: rules read state directly from the portfolio they're given."""
        pass

    def check_and_apply_risk_rules(
        self,
        portfolio,
        market_data: Dict[str, pd.DataFrame],
    ) -> List[RiskSignal]:
        """Adapt the BaseRiskManager interface to ``check_all``.

        Builds a ``prices`` dict from the latest bar in each symbol's
        market_data and forwards to :meth:`check_all`.
        """
        prices: Dict[str, float] = {}
        timestamp = None
        for sym, df in market_data.items():
            if len(df) == 0:
                continue
            prices[sym] = float(df.iloc[-1]['close'])
            timestamp = df.index[-1]
        return self.check_all(portfolio, prices, timestamp)

    def calculate_position_size(
        self,
        symbol: str,
        signal: int,
        current_price: float,
        market_data: pd.DataFrame,
    ) -> float:
        """Default sizing: pass the signal through unchanged.

        Composite *rules* are about gating / clipping risk, not
        determining base position size. Use the engine's
        ``position_sizer`` argument for that.
        """
        return float(signal)

    # ------------------------------------------------------------------
    # Composite-specific API (used when passed via risk_rules=)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all rules for a new backtest run."""
        for rule in self.rules:
            rule.reset()
        self.history.clear()

    def check_all(self, portfolio, prices: Dict[str, float],
                  timestamp) -> List[RiskSignal]:
        """Run all rules and collect triggered signals.

        Parameters:
            portfolio: Current Portfolio instance.
            prices: Dict mapping symbol to current price.
            timestamp: Current bar timestamp.

        Returns:
            List of RiskSignal objects from rules that triggered.
        """
        signals: List[RiskSignal] = []
        for rule in self.rules:
            sig = rule.check(portfolio, prices, timestamp)
            if sig is not None:
                signals.append(sig)
                self.history.append({
                    'timestamp': timestamp,
                    'rule': rule.name,
                    'signal': sig,
                })
        return signals

    def apply_vectorized_all(self, equity_curve: pd.Series,
                             positions: pd.Series,
                             data: pd.DataFrame) -> pd.Series:
        """Apply all rules sequentially in vectorized mode."""
        for rule in self.rules:
            positions = rule.apply_vectorized(equity_curve, positions, data)
        return positions


# ---------------------------------------------------------------------------
# Built-in Risk Rules
# ---------------------------------------------------------------------------


class MaxDrawdownHalt(BaseRiskRule):
    """Halt new trades when portfolio drawdown exceeds a threshold.

    In event-driven mode, trading resumes when drawdown recovers below
    ``reset_drawdown`` (hysteresis).

    In vectorized mode, positions are zeroed once drawdown exceeds the
    threshold and stay zeroed for the remainder (no hysteresis — this is
    a simplification; use event-driven mode for recovery support).

    Parameters:
        max_drawdown: Drawdown fraction that triggers the halt (e.g. 0.15).
        reset_drawdown: Drawdown fraction below which trading resumes
            (e.g. 0.10).  Must be <= max_drawdown.
    """

    name = "Max Drawdown Halt"

    def __init__(self, max_drawdown: float = 0.15,
                 reset_drawdown: float = 0.10):
        self.max_drawdown = max_drawdown
        self.reset_drawdown = reset_drawdown
        self._halted = False

    def reset(self) -> None:
        self._halted = False

    def check(self, portfolio, prices, timestamp) -> Optional[RiskSignal]:
        dd = portfolio.current_drawdown
        if not self._halted and dd >= self.max_drawdown:
            self._halted = True
            return RiskSignal(
                'critical', 'reject_new',
                f'Drawdown {dd:.1%} >= {self.max_drawdown:.1%}')
        if self._halted and dd < self.reset_drawdown:
            self._halted = False
            return RiskSignal(
                'info', 'none',
                f'Drawdown recovered to {dd:.1%}, trading resumed')
        if self._halted:
            return RiskSignal(
                'critical', 'reject_new',
                f'Trading halted (drawdown {dd:.1%})')
        return None

    def apply_vectorized(self, equity_curve, positions, data):
        """Zero positions when drawdown exceeds threshold.

        Note: vectorized mode has no hysteresis (reset_drawdown recovery).
        Once halted, stays halted for the rest of the period.  This is a
        simplification — event-driven mode supports recovery via check().
        """
        peak = equity_curve.cummax()
        dd = (peak - equity_curve) / peak
        halted = dd >= self.max_drawdown
        return positions.where(~halted, 0)


class ExposureLimit(BaseRiskRule):
    """Limit total long/short/net exposure as a fraction of equity.

    Parameters:
        max_long: Maximum long exposure as fraction of equity.
        max_short: Maximum short exposure as fraction of equity.
        max_net: Maximum absolute net exposure as fraction of equity.
    """

    name = "Exposure Limit"

    def __init__(self, max_long: float = 1.0, max_short: float = 0.5,
                 max_net: float = 1.0):
        self.max_long = max_long
        self.max_short = max_short
        self.max_net = max_net

    def check(self, portfolio, prices, timestamp) -> Optional[RiskSignal]:
        equity = portfolio.total_equity
        if equity <= 0:
            return RiskSignal('critical', 'reject_new', 'Zero equity')

        long_val = sum(
            p.market_value for p in portfolio.positions.values()
            if p.is_long)
        short_val = sum(
            abs(p.market_value) for p in portfolio.positions.values()
            if p.is_short)
        net_val = long_val - short_val

        if long_val / equity > self.max_long:
            return RiskSignal(
                'critical', 'reject_new',
                f'Long exposure {long_val / equity:.1%} > {self.max_long:.1%}')
        if short_val / equity > self.max_short:
            return RiskSignal(
                'critical', 'reject_new',
                f'Short exposure {short_val / equity:.1%} > {self.max_short:.1%}')
        if abs(net_val) / equity > self.max_net:
            return RiskSignal(
                'critical', 'reject_new',
                f'Net exposure {net_val / equity:.1%} exceeds limit')
        return None


class DailyLossLimit(BaseRiskRule):
    """Stop trading when daily loss exceeds a threshold.

    Resets at the start of each new trading day.

    Parameters:
        max_daily_loss: Maximum allowed daily loss as a fraction of
            start-of-day equity (e.g. 0.03 for 3%).
    """

    name = "Daily Loss Limit"

    def __init__(self, max_daily_loss: float = 0.03):
        self.max_daily_loss = max_daily_loss
        self._day_start_equity: Optional[float] = None
        self._current_date = None
        self._halted_today = False

    def reset(self) -> None:
        self._day_start_equity = None
        self._current_date = None
        self._halted_today = False

    def check(self, portfolio, prices, timestamp) -> Optional[RiskSignal]:
        today = timestamp.date() if hasattr(timestamp, 'date') else None
        if today != self._current_date:
            self._current_date = today
            self._day_start_equity = portfolio.total_equity
            self._halted_today = False

        if self._halted_today:
            return RiskSignal(
                'critical', 'reject_new',
                'Daily loss limit hit — halted until next day')

        if self._day_start_equity and self._day_start_equity > 0:
            daily_pnl = portfolio.total_equity - self._day_start_equity
            daily_loss = -daily_pnl / self._day_start_equity
            if daily_loss >= self.max_daily_loss:
                self._halted_today = True
                return RiskSignal(
                    'critical', 'reject_new',
                    f'Daily loss {daily_loss:.1%} >= {self.max_daily_loss:.1%}')
        return None


class ConcentrationLimit(BaseRiskRule):
    """Warn when a single position exceeds a fraction of equity.

    Emits a 'warning' severity signal with 'reduce' action when any
    single position's weight exceeds ``max_weight``.

    Parameters:
        max_weight: Maximum weight of a single position as fraction
            of total equity (e.g. 0.3 for 30%).
    """

    name = "Concentration Limit"

    def __init__(self, max_weight: float = 0.3):
        self.max_weight = max_weight

    def check(self, portfolio, prices, timestamp) -> Optional[RiskSignal]:
        equity = portfolio.total_equity
        if equity <= 0:
            return None
        for sym, pos in portfolio.positions.items():
            if pos.is_flat:
                continue
            weight = abs(pos.market_value) / equity
            if weight > self.max_weight:
                return RiskSignal(
                    'warning', 'reduce',
                    f'{sym} weight {weight:.1%} > {self.max_weight:.1%}')
        return None
