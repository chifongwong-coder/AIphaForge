"""
Agent Latency Simulation Hook
==============================

Simulates the delay between an agent's decision and the actual order
submission.  LLM-based agents typically take 5-120 seconds to produce a
decision; this hook lets backtests model that latency by delaying order
submissions by N bars.

Architecture: decorator pattern.  ``LatencyHook`` wraps any
``BacktestHook``, intercepting its ``submit_order()`` calls and replaying
them after a configurable delay.
"""

import dataclasses
import random
import warnings
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .hooks import BacktestHook, HookContext
from .orders import Order

# ---------------------------------------------------------------------------
# Internal broker proxy
# ---------------------------------------------------------------------------

class _CapturingBrokerProxy:
    """Wraps a real broker, capturing ``submit_order`` calls.

    All other attribute accesses (``create_market_order``, etc.) are
    forwarded to the underlying broker so the inner hook can create
    orders normally.
    """

    def __init__(self, real_broker: Any) -> None:
        self._real_broker = real_broker
        self.captured_orders: List[Tuple[Order, Any]] = []

    def submit_order(
        self, order: Order, timestamp: Any = None
    ) -> str:
        """Capture instead of submitting."""
        self.captured_orders.append((order, timestamp))
        return order.order_id

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_broker, name)


# ---------------------------------------------------------------------------
# Queued-order record
# ---------------------------------------------------------------------------

_QueuedOrder = Tuple[int, Order, Any]  # (submit_at_bar, order, decision_timestamp)


# ---------------------------------------------------------------------------
# LatencyHook
# ---------------------------------------------------------------------------

class LatencyHook(BacktestHook):
    """Decorator hook that delays an inner hook's order submissions.

    Wraps any ``BacktestHook`` and intercepts its ``submit_order`` calls
    during ``on_pre_signal``.  Captured orders are held in a queue and
    submitted to the real broker after a configurable delay measured in
    bars.

    Parameters:
        inner_hook: The hook whose orders will be delayed.
        latency_model: ``"fixed"``, ``"statistical"``, or ``"custom"``.
        latency_params: Model-specific parameters.
            - fixed: ``{"bars": int}`` (>= 1).
            - statistical: ``{"distribution": "normal"|"uniform", ...}``.
              For ``"normal"``: ``{"mean": float, "std": float}``.
              For ``"uniform"``: ``{"low": float, "high": float}``.
            - custom: ``{"fn": Callable[[int, HookContext], int]}``.

    Example::

        hook = LatencyHook(
            inner_hook=my_agent_hook,
            latency_model="fixed",
            latency_params={"bars": 3},
        )
        engine = BacktestEngine(mode="event_driven", hooks=[hook])
    """

    def __init__(
        self,
        inner_hook: BacktestHook,
        latency_model: str = "fixed",
        latency_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.inner_hook = inner_hook
        self.latency_model = latency_model
        self.latency_params: Dict[str, Any] = latency_params or {}
        self._queue: Deque[_QueuedOrder] = deque()
        self._warned_signal_conflict = False

        # Eagerly validate parameters
        self._validate_params()

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def _validate_params(self) -> None:
        if self.latency_model == "fixed":
            bars = self.latency_params.get("bars", 1)
            if not isinstance(bars, int) or bars < 1:
                raise ValueError(
                    f"fixed latency requires 'bars' >= 1, got {bars!r}"
                )
        elif self.latency_model == "statistical":
            dist = self.latency_params.get("distribution")
            if dist not in ("normal", "uniform"):
                raise ValueError(
                    f"statistical latency requires distribution "
                    f"'normal' or 'uniform', got {dist!r}"
                )
        elif self.latency_model == "custom":
            fn = self.latency_params.get("fn")
            if not callable(fn):
                raise ValueError(
                    "custom latency requires 'fn' to be callable"
                )
        else:
            raise ValueError(
                f"Unknown latency_model: {self.latency_model!r}. "
                f"Must be 'fixed', 'statistical', or 'custom'."
            )

    # ------------------------------------------------------------------
    # Lifecycle callbacks
    # ------------------------------------------------------------------

    def on_backtest_start(
        self,
        data: Any,
        symbol: str,
        *,
        config: Any = None,
    ) -> None:
        """Validate execution mode and forward to the inner hook."""
        self._queue.clear()
        self._warned_signal_conflict = False

        if config is None:
            raise ValueError(
                "LatencyHook requires EVENT_DRIVEN mode. "
                "Pass the backtest config to on_backtest_start."
            )

        # Validate execution mode
        mode = getattr(config, "mode", None)
        if mode is not None and mode != "event_driven":
            raise ValueError(
                f"LatencyHook requires EVENT_DRIVEN mode, got '{mode}'. "
                "Hooks are only called in event-driven execution."
            )

        # Warn once if the engine also has signals or a strategy configured
        has_signals = getattr(config, "has_signals", False)
        has_strategy = getattr(config, "has_strategy", False)
        if (has_signals or has_strategy) and not self._warned_signal_conflict:
            warnings.warn(
                "LatencyHook detected while a strategy/signals are also "
                "configured. Both the hook and signals will generate orders. "
                "Set signals to 0 if the hook manages all orders.",
                stacklevel=2,
            )
            self._warned_signal_conflict = True

        self.inner_hook.on_backtest_start(data, symbol, config=config)

    def on_pre_signal(self, context: HookContext) -> None:
        """Intercept the inner hook's orders and manage the delay queue."""
        # Step 1 — run the inner hook with a capturing proxy
        proxy = _CapturingBrokerProxy(context.broker)
        proxied_ctx = dataclasses.replace(context, broker=proxy)
        self.inner_hook.on_pre_signal(proxied_ctx)

        # Step 2 — queue newly captured orders with computed delay
        for order, ts in proxy.captured_orders:
            delay = self._calculate_delay(context.bar_index, context)
            submit_at = context.bar_index + delay - 1
            decision_ts = context.timestamp
            self._queue.append((submit_at, order, decision_ts))

        # Step 3 — submit any orders whose delay has elapsed
        self._flush_ready_orders(context)

    def on_bar(self, context: HookContext) -> None:
        """Forward to the inner hook (no proxying needed)."""
        self.inner_hook.on_bar(context)

    def on_backtest_end(self) -> None:
        """Forward to the inner hook."""
        self.inner_hook.on_backtest_end()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_ready_orders(self, context: HookContext) -> None:
        """Submit queued orders that are due on or before the current bar."""
        remaining: Deque[_QueuedOrder] = deque()
        while self._queue:
            submit_at, order, decision_ts = self._queue.popleft()
            if submit_at <= context.bar_index:
                # Preserve decision-time TIF semantics
                if order.created_time is None:
                    order.created_time = decision_ts
                context.broker.submit_order(order, decision_ts)
            else:
                remaining.append((submit_at, order, decision_ts))
        self._queue = remaining

    def _calculate_delay(self, bar_index: int, ctx: HookContext) -> int:
        """Return the delay in bars (always >= 1)."""
        if self.latency_model == "fixed":
            return self.latency_params.get("bars", 1)

        if self.latency_model == "statistical":
            dist = self.latency_params["distribution"]
            if dist == "normal":
                mean = self.latency_params.get("mean", 3.0)
                std = self.latency_params.get("std", 1.0)
                sample = random.gauss(mean, std)
            else:  # uniform
                low = self.latency_params.get("low", 1.0)
                high = self.latency_params.get("high", 5.0)
                sample = random.uniform(low, high)
            return max(1, round(sample))

        if self.latency_model == "custom":
            fn: Callable[[int, HookContext], int] = self.latency_params["fn"]
            value = fn(bar_index, ctx)
            return max(1, int(value))

        # Should not be reachable after __init__ validation
        return 1  # pragma: no cover


# ---------------------------------------------------------------------------
# SimpleLatencyHook — convenience subclass
# ---------------------------------------------------------------------------

class SimpleLatencyHook(LatencyHook):
    """Convenience base for agents that only need ``make_decision``.

    Subclass and override ``make_decision(ctx)`` instead of creating a
    separate ``BacktestHook`` and wrapping it with ``LatencyHook``.

    Example::

        class MyAgent(SimpleLatencyHook):
            def make_decision(self, ctx):
                if ctx.bar_data['close'] > ctx.data['close'].mean():
                    order = ctx.broker.create_market_order(
                        ctx.symbol, 'buy', 100,
                    )
                    ctx.broker.submit_order(order, ctx.timestamp)
    """

    def __init__(
        self,
        latency_model: str = "fixed",
        latency_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Build an anonymous inner hook that delegates to make_decision
        outer = self

        class _InnerHook(BacktestHook):
            def on_pre_signal(self, context: HookContext) -> None:
                result = outer.make_decision(context)
                if result is not None:
                    context.broker.submit_order(result, context.timestamp)

        super().__init__(
            inner_hook=_InnerHook(),
            latency_model=latency_model,
            latency_params=latency_params,
        )

    def make_decision(self, ctx: HookContext) -> Optional[Order]:
        """Override this to implement agent decision logic.

        Return an ``Order`` to have it submitted (with latency), or
        ``None`` to skip this bar.  The returned order will be passed
        to ``broker.submit_order`` automatically.  You may also call
        ``ctx.broker.submit_order`` directly for multiple orders.
        """
        raise NotImplementedError(
            "Subclasses must implement make_decision()"
        )
