"""
MetaContext — Agent Control Interface
=====================================

Mutable config proxy that allows AI agent hooks to dynamically adjust
strategy, risk parameters, and position sizing during an event-driven
backtest.

Attached to HookContext as ``ctx.meta``. Changes persist across bars
until explicitly changed again. The base BacktestConfig is never mutated;
overrides are applied via ``dataclasses.replace`` each bar.

Event-driven mode only. Vectorized mode has no per-bar hook mechanism.
"""

from typing import Any, Dict, List, Optional


class MetaContext:
    """Mutable config proxy for Agent control within a bar.

    Attached to HookContext as ``ctx.meta``. Changes apply to the current
    bar's signal processing and persist until changed again.

    NOT a replacement for BacktestConfig -- it wraps the config and
    applies per-bar overrides. Base config is never mutated.
    """

    def __init__(self, config: Any, strategy: Any = None) -> None:
        self._config = config
        self._strategy = strategy
        self._overrides: Dict[str, Any] = {}
        self._suppress: bool = False
        self._target_weights: Optional[Dict[str, float]] = None
        self._audit: List[Dict[str, Any]] = []
        self._audit_cursor: int = 0

    # --- Internal logging ---

    def _log(self, action: str, value: Any) -> None:
        """Append to audit trail. Enriched with context at step 4.1."""
        self._audit.append({'action': action, 'value': value})

    # --- Strategy control ---

    def set_strategy(self, strategy: Any) -> None:
        """Swap the active strategy. Takes effect this bar."""
        self._strategy = strategy
        self._log('set_strategy', strategy.name)

    def adjust_strategy_params(self, **kwargs: Any) -> None:
        """Adjust current strategy parameters via update_params()."""
        if self._strategy is not None:
            self._strategy.update_params(**kwargs)
            self._log('adjust_strategy_params', kwargs)

    # --- Position sizing ---

    def adjust_sizing(self, fraction: float) -> None:
        """Override position sizer fraction for this bar onwards."""
        from .position_sizing import FractionSizer
        self._overrides['position_sizer'] = FractionSizer(fraction)
        self._log('adjust_sizing', fraction)

    # --- Risk control ---

    def adjust_stop_loss(self, threshold: float) -> None:
        """Override stop-loss threshold."""
        from .exit_rules import PercentageStopLoss
        self._overrides['stop_loss_rule'] = PercentageStopLoss(threshold)
        self._log('adjust_stop_loss', threshold)

    def adjust_take_profit(self, threshold: float) -> None:
        """Override take-profit threshold."""
        from .exit_rules import PercentageTakeProfit
        self._overrides['take_profit_rule'] = PercentageTakeProfit(threshold)
        self._log('adjust_take_profit', threshold)

    # --- Signal control ---

    def suppress_signals(self) -> None:
        """Suppress signal processing for this bar (hold all positions)."""
        self._suppress = True
        self._log('suppress_signals', True)

    def resume_signals(self) -> None:
        """Resume signal processing (undo suppress)."""
        self._suppress = False
        self._log('resume_signals', True)

    def set_target_weights(self, weights: Dict[str, float]) -> None:
        """Override signal processing with target weights for this bar."""
        self._target_weights = dict(weights)
        self._log('set_target_weights', dict(weights))

    # --- Read state ---

    @property
    def current_strategy(self) -> Any:
        """The currently active strategy."""
        return self._strategy

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        """Copy of the full audit trail."""
        return list(self._audit)

    # --- Internal: apply overrides to config ---

    def _apply_overrides(self, config: Any) -> Any:
        """Return config with current overrides applied.

        Called by the event loop before signal processing.
        Does NOT mutate the original config.
        """
        import dataclasses
        if not self._overrides:
            return config
        return dataclasses.replace(config, **self._overrides)
