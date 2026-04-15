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
import math
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
# Internal meta proxy
# ---------------------------------------------------------------------------

class _CapturingMetaProxy:
    """Wraps a real MetaContext, capturing mutable method calls.

    Non-callable attributes (bools, dicts, etc.) always pass through
    to the real MetaContext — this is safe because reading an attribute
    is never a mutation.

    Callable attributes (methods) are captured by default, EXCEPT
    those in ``_READ_ONLY_METHODS`` which are known query methods.

    This design:
    - Correctly handles attribute reads (e.g., ``_suppress``, ``_strategy``)
    - Captures new mutable methods added to MetaContext in the future
    - Forwards known read-only methods (``get_children``, ``_is_composite``)
    """

    _READ_ONLY_METHODS = {
        'get_children', '_is_composite',
    }

    def __init__(self, real_meta: Any) -> None:
        object.__setattr__(self, '_real_meta', real_meta)
        object.__setattr__(self, 'captured_ops', [])

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._real_meta, name)
        if not callable(attr):
            # Non-callable (bool, dict, list, object, etc.): pass through.
            # Properties like current_strategy, audit_log resolve here.
            return attr
        if name in self._READ_ONLY_METHODS:
            # Known read-only method: forward directly.
            return attr
        # Callable and not read-only: capture for delayed replay.
        def _capture(*args: Any, **kwargs: Any) -> None:
            self.captured_ops.append((name, args, kwargs))
        return _capture


# ---------------------------------------------------------------------------
# Queued-order record
# ---------------------------------------------------------------------------

_QueuedOrder = Tuple[int, Order, Any]  # (submit_at_bar, order, decision_timestamp)
_MetaQueueEntry = Tuple[int, List[Tuple[str, tuple, dict]], Any]  # (submit_at_bar, meta_ops, decision_timestamp)


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
        self._order_queue: Deque[_QueuedOrder] = deque()
        self._meta_queue: Deque[_MetaQueueEntry] = deque()
        self._warned_signal_conflict = False
        self._warned_custom_clamp = False
        self._current_order: Optional[Order] = None

        # Eagerly validate parameters
        self._validate_params()

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_model_params(model: str, params: Dict[str, Any]) -> None:
        """Validate latency model and parameters.

        Can be used standalone (e.g., for per-symbol override validation)
        without needing a full LatencyHook instance.
        """
        if model == "fixed":
            bars = params.get("bars", 1)
            if not isinstance(bars, int) or bars < 0:
                raise ValueError(
                    f"fixed latency requires 'bars' >= 0, got {bars!r}"
                )
        elif model == "statistical":
            dist = params.get("distribution")
            if dist not in ("normal", "uniform"):
                raise ValueError(
                    f"statistical latency requires distribution "
                    f"'normal' or 'uniform', got {dist!r}"
                )
        elif model == "custom":
            fn = params.get("fn")
            if not callable(fn):
                raise ValueError(
                    "custom latency requires 'fn' to be callable"
                )
        else:
            raise ValueError(
                f"Unknown latency_model: {model!r}. "
                f"Must be 'fixed', 'statistical', or 'custom'."
            )

    def _validate_params(self) -> None:
        self._validate_model_params(self.latency_model, self.latency_params)

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
        self._order_queue.clear()
        self._meta_queue.clear()
        self._warned_signal_conflict = False
        self._warned_custom_clamp = False

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
                "Set signals to NaN if the hook manages all orders.",
                stacklevel=2,
            )
            self._warned_signal_conflict = True

        self.inner_hook.on_backtest_start(data, symbol, config=config)

    def on_pre_signal(self, context: HookContext) -> None:
        """Intercept the inner hook's orders and manage the delay queue."""
        if context.all_brokers is not None:
            # Multi-asset: proxy all brokers
            self._on_pre_signal_multi(context)
        else:
            # Single-asset: proxy the single broker
            self._on_pre_signal_single(context)

    def _on_pre_signal_single(self, context: HookContext) -> None:
        """Single-asset path: proxy context.broker and context.meta."""
        proxy = _CapturingBrokerProxy(context.broker)
        meta_proxy = (
            _CapturingMetaProxy(context.meta)
            if context.meta is not None else None
        )
        proxied_ctx = dataclasses.replace(
            context,
            broker=proxy,
            meta=meta_proxy if meta_proxy is not None else context.meta,
        )
        self.inner_hook.on_pre_signal(proxied_ctx)

        # Queue captured orders (per-order delay, same as before)
        for order, ts in proxy.captured_orders:
            self._current_order = order
            delay = self._calculate_delay(context.bar_index, context)
            self._current_order = None
            submit_at = context.bar_index + delay - 1
            decision_ts = context.timestamp
            self._order_queue.append((submit_at, order, decision_ts))

        # Queue captured meta ops (one batch, default delay)
        if meta_proxy is not None and meta_proxy.captured_ops:
            delay = self._calculate_delay(context.bar_index, context)
            submit_at = context.bar_index + delay - 1
            self._meta_queue.append(
                (submit_at, list(meta_proxy.captured_ops), context.timestamp))

        self._flush_ready_actions(context)

    def _on_pre_signal_multi(self, context: HookContext) -> None:
        """Multi-asset path: proxy all brokers and meta."""
        proxied_brokers = {}
        for sym, broker in context.all_brokers.items():
            proxied_brokers[sym] = _CapturingBrokerProxy(broker)
        meta_proxy = (
            _CapturingMetaProxy(context.meta)
            if context.meta is not None else None
        )
        proxied_ctx = dataclasses.replace(
            context,
            all_brokers=proxied_brokers,
            meta=meta_proxy if meta_proxy is not None else context.meta,
        )
        self.inner_hook.on_pre_signal(proxied_ctx)

        # Collect captured orders from all proxies
        for sym, proxy in proxied_brokers.items():
            for order, ts in proxy.captured_orders:
                self._current_order = order
                delay = self._calculate_delay(
                    context.bar_index, context)
                self._current_order = None
                submit_at = context.bar_index + delay - 1
                decision_ts = context.timestamp
                self._order_queue.append((submit_at, order, decision_ts))

        # Queue captured meta ops (one batch, default delay)
        if meta_proxy is not None and meta_proxy.captured_ops:
            delay = self._calculate_delay(context.bar_index, context)
            submit_at = context.bar_index + delay - 1
            self._meta_queue.append(
                (submit_at, list(meta_proxy.captured_ops), context.timestamp))

        self._flush_ready_actions(context)

    def on_bar(self, context: HookContext) -> None:
        """Forward to the inner hook (no proxying needed)."""
        self.inner_hook.on_bar(context)

    def on_backtest_end(self) -> None:
        """Warn about pending queue items, then forward to the inner hook."""
        n_pending = len(self._order_queue) + len(self._meta_queue)
        if n_pending > 0:
            warnings.warn(
                f"LatencyHook: {n_pending} queued actions were not "
                f"executed (backtest ended before delay elapsed).")
        self._order_queue.clear()
        self._meta_queue.clear()
        self.inner_hook.on_backtest_end()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_ready_actions(self, context: HookContext) -> None:
        """Flush meta operations and orders that are due on or before the current bar."""
        # 1. Flush meta queue first (strategy changes take effect before orders)
        remaining_meta: Deque[_MetaQueueEntry] = deque()
        while self._meta_queue:
            submit_at, meta_ops, decision_ts = self._meta_queue.popleft()
            if submit_at <= context.bar_index:
                if context.meta is not None:
                    for method_name, args, kwargs in meta_ops:
                        getattr(context.meta, method_name)(*args, **kwargs)
            else:
                remaining_meta.append((submit_at, meta_ops, decision_ts))
        self._meta_queue = remaining_meta

        # 2. Then flush order queue (submit orders)
        remaining_orders: Deque[_QueuedOrder] = deque()
        while self._order_queue:
            submit_at, order, decision_ts = self._order_queue.popleft()
            if submit_at <= context.bar_index:
                # Preserve decision-time TIF semantics
                if order.created_time is None:
                    order.created_time = decision_ts
                # Route to correct broker
                if context.all_brokers is not None:
                    broker = context.all_brokers.get(order.symbol)
                    if broker is not None:
                        broker.submit_order(order, decision_ts)
                else:
                    context.broker.submit_order(order, decision_ts)
            else:
                remaining_orders.append((submit_at, order, decision_ts))
        self._order_queue = remaining_orders

    @staticmethod
    def _calculate_delay_for_model(
        bar_index: int,
        ctx: HookContext,
        model: str,
        params: Dict[str, Any],
    ) -> int:
        """Compute delay for a given latency model and params.

        This is a pure function that does not read or mutate instance state,
        except that the ``"custom"`` model's callable receives the bar_index
        and context as-is.

        Returns:
            int: Delay in bars (always >= 1 for fixed/statistical).
                 For custom, returns the raw int(value) — caller is
                 responsible for clamping and validation.
        """
        if model == "fixed":
            return params.get("bars", 1)

        if model == "statistical":
            dist = params["distribution"]
            if dist == "normal":
                mean = params.get("mean", 3.0)
                std = params.get("std", 1.0)
                sample = random.gauss(mean, std)
            else:  # uniform
                low = params.get("low", 1.0)
                high = params.get("high", 5.0)
                sample = random.uniform(low, high)
            return max(1, round(sample))

        if model == "custom":
            fn: Callable[[int, HookContext], int] = params["fn"]
            value = fn(bar_index, ctx)
            # Caller handles inf/nan guard and clamp warning
            return value  # type: ignore[return-value]

        # Should not be reachable after __init__ validation
        return 1  # pragma: no cover

    def _calculate_delay(self, bar_index: int, ctx: HookContext) -> int:
        """Return the delay in bars (always >= 1)."""
        value = self._calculate_delay_for_model(
            bar_index, ctx, self.latency_model, self.latency_params,
        )

        # For custom model, validate and clamp with a warning
        if self.latency_model == "custom":
            if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                raise ValueError(
                    f"Custom latency callable returned invalid value: {value}"
                )
            clamped = max(1, int(value))
            if clamped != int(value) and not self._warned_custom_clamp:
                warnings.warn(
                    f"Custom latency callable returned {value} at bar {bar_index}, "
                    f"clamped to {clamped} (minimum delay is 1 bar). "
                    f"Further clamp warnings suppressed.",
                    stacklevel=2,
                )
                self._warned_custom_clamp = True
            return clamped

        return value


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


# ---------------------------------------------------------------------------
# SymbolRoutingLatencyHook — per-symbol latency routing
# ---------------------------------------------------------------------------

class SymbolRoutingLatencyHook(LatencyHook):
    """LatencyHook subclass that routes delay calculation by symbol.

    Different assets can have different latency profiles (e.g., an LLM
    agent may take longer to analyze Chinese stocks than US stocks).
    This hook lets you specify per-symbol overrides while sharing a
    single inner hook instance.

    Parameters:
        inner_hook: The hook whose orders will be delayed.
        default_latency_model: Default latency model for unmatched symbols.
        default_latency_params: Default latency parameters.
        symbol_overrides: Mapping of symbol to ``(model, params)`` tuples.

    Example::

        hook = SymbolRoutingLatencyHook(
            inner_hook=agent_hook,
            default_latency_model="fixed",
            default_latency_params={"bars": 3},
            symbol_overrides={
                "AAPL": ("fixed", {"bars": 1}),
                "600519.SH": ("fixed", {"bars": 5}),
            },
        )
    """

    def __init__(
        self,
        inner_hook: BacktestHook,
        default_latency_model: str = "fixed",
        default_latency_params: Optional[Dict[str, Any]] = None,
        symbol_overrides: Optional[Dict[str, Tuple[str, Dict[str, Any]]]] = None,
    ) -> None:
        super().__init__(
            inner_hook=inner_hook,
            latency_model=default_latency_model,
            latency_params=default_latency_params,
        )
        self._symbol_overrides: Dict[str, Tuple[str, Dict[str, Any]]] = (
            symbol_overrides or {}
        )

        # Validate each override's parameters eagerly
        for sym, (model, params) in self._symbol_overrides.items():
            try:
                # Create a temporary instance to validate
                LatencyHook._validate_model_params(model, params)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid override for symbol {sym!r}: {exc}"
                ) from exc

    def _calculate_delay(self, bar_index: int, ctx: HookContext) -> int:
        """Route delay calculation by the current order's symbol."""
        order = self._current_order
        if order is not None and order.symbol in self._symbol_overrides:
            model, params = self._symbol_overrides[order.symbol]
            value = self._calculate_delay_for_model(bar_index, ctx, model, params)
            # Apply same guard+clamp as base class for custom callables
            if model == "custom":
                if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                    raise ValueError(
                        f"Custom latency callable returned invalid value: {value}"
                    )
                clamped = max(1, int(value))
                if clamped != int(value) and not self._warned_custom_clamp:
                    warnings.warn(
                        f"Custom latency callable returned {value} at bar "
                        f"{bar_index}, clamped to {clamped} (minimum delay "
                        f"is 1 bar). Further clamp warnings suppressed.",
                        stacklevel=2,
                    )
                    self._warned_custom_clamp = True
                return clamped
            return value
        return super()._calculate_delay(bar_index, ctx)
