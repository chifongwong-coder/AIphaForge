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
        self._needs_regeneration: bool = False
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
        self._needs_regeneration = True
        self._log('set_strategy', strategy.name)

    def adjust_strategy_params(self, **kwargs: Any) -> None:
        """Adjust current strategy parameters via update_params().

        Triggers signal regeneration so new params take effect.
        """
        if self._strategy is not None:
            self._strategy.update_params(**kwargs)
            self._needs_regeneration = True
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

    def adjust_trailing_stop(self, trail_percent: float) -> None:
        """Override trailing stop-loss rule.

        Creates a new TrailingStopLoss with the given trail_percent.
        Fresh watermark — starts tracking from the current bar.
        Independent of adjust_stop_loss (both can be active).
        """
        from .exit_rules import TrailingStopLoss
        self._overrides['trailing_stop_rule'] = TrailingStopLoss(trail_percent)
        self._log('adjust_trailing_stop', trail_percent)

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

    # --- Strategy tree control (v1.4) ---

    def _is_composite(self) -> bool:
        """Check if the current strategy is a composite (StrategyNode)."""
        from .strategies import StrategyNode
        return isinstance(self._strategy, StrategyNode)

    def set_weights(self, weights: list) -> None:
        """Adjust composite strategy weights. Auto-regenerates.

        Warns and no-ops if current strategy is not a weighted composite.
        """
        if not self._is_composite() or not hasattr(self._strategy, 'weights'):
            import warnings
            warnings.warn("set_weights: current strategy is not a weighted composite")
            return
        if len(weights) != len(self._strategy.children):
            raise ValueError(
                f"len(weights)={len(weights)} != "
                f"len(children)={len(self._strategy.children)}")
        self._strategy.weights = list(weights)
        self._needs_regeneration = True
        self._log('set_weights', weights)

    def swap_child(self, index: int, new_child: Any) -> None:
        """Replace a child strategy in the tree. Auto-regenerates.

        Warns and no-ops if current strategy is not a composite.
        """
        if not self._is_composite():
            import warnings
            warnings.warn("swap_child: current strategy is not a composite")
            return
        if not 0 <= index < len(self._strategy.children):
            raise IndexError(
                f"child index {index} out of range "
                f"[0, {len(self._strategy.children)})")
        self._strategy.children[index] = new_child
        self._needs_regeneration = True
        self._log('swap_child', {'index': index, 'child': new_child.name})

    def get_children(self) -> list:
        """List child strategies (empty list for leaf strategies)."""
        if self._is_composite():
            return list(self._strategy.children)
        return []

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
