"""
Backtest Engine

Main backtest executor supporting both vectorized and event-driven modes.
"""

import warnings
from dataclasses import dataclass
from datetime import time
from enum import Enum
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .broker import FillModel
from .config import BacktestConfig
from .core_event_driven import run_event_driven
from .core_vectorized import run_vectorized
from .costs import DefaultTradeCost
from .exit_rules import PercentageStopLoss, PercentageTakeProfit
from .fees import BaseFeeModel, SimpleFeeModel, get_fee_model
from .hooks import (
    BacktestHook,
    LifecycleContext,
    call_hook_lifecycle_end,
    call_hook_lifecycle_start,
)
from .latency import LatencyHook
from .position_sizing import AllInSizer, FixedSizer, FractionSizer
from .results import BacktestResult, Trade
from .signals import (
    target_weight_wide_to_schedule,
    validate_signal_wide,
    wide_to_signal_dict,
)

# Import utility functions
from .utils import (
    TRADING_DAYS_STOCK,
    _normalize_trading_days,
    annualize_return,
    calculate_trade_metrics,
    compute_buy_and_hold,
    ensure_datetime_index,
    infer_bars_per_year,
    validate_ohlcv,
)
from .utils import (
    max_drawdown as calc_max_drawdown,
)
from .utils import (
    sharpe_ratio as calc_sharpe,
)
from .utils import (
    sortino_ratio as calc_sortino,
)

# v2.8: public surface lock. Anything not listed here is internal and
# may move in v3.0. `_TargetWeightsWideConfig` (v2.7) is deliberately
# excluded as a private implementation detail; tests that need it
# import it explicitly.
__all__ = [
    "BacktestEngine",
    "ExecutionMode",
    "PositionSizing",
    "backtest",
]


@dataclass(frozen=True)
class _TargetWeightsWideConfig:
    """v2.7 frozen bundle of set_target_weights_wide kwargs.

    Frozen for pickle stability across the v2.6 parallel-backtest path.
    Holds the EFFECTIVE values after default-resolution and strict-
    conflict checks (the setter never stores raw kwargs — see
    set_target_weights_wide for the resolution rules).
    """
    # rebalance_dates is stored as a concrete list (or None) — generators
    # would silently break two paths: pickle (per v2.6 parallel-backtest)
    # and chained run() (exhausted after first materialization).
    rebalance_dates: Optional[List]
    snap: Literal["exact", "next", "previous"]
    universe_alignment: Literal["union", "intersection"]
    on_collision: Literal["warn", "raise", "first", "last"]
    strict: bool


class ExecutionMode(Enum):
    """Execution mode."""
    VECTORIZED = "vectorized"        # Vectorized mode (fast)
    EVENT_DRIVEN = "event_driven"    # Event-driven mode (precise)


class PositionSizing(Enum):
    """Position sizing method."""
    FIXED_FRACTION = "fixed_fraction"  # Fixed fraction of equity
    FIXED_SIZE = "fixed_size"          # Fixed quantity
    ALL_IN = "all_in"                  # Full position
    RISK_BASED = "risk_based"          # Risk-based sizing


class BacktestEngine:
    """
    Backtest engine supporting vectorized and event-driven execution modes.

    Parameters:
        fee_model: Fee model instance.
        initial_capital: Starting capital.
        mode: Execution mode.
        position_sizing: Position sizing method.
        position_size: Position size (fraction or fixed quantity).
        max_position_size: Maximum single position as fraction of equity.
        stop_loss: Stop loss percentage.
        take_profit: Take profit percentage.
        allow_short: Whether short selling is allowed.
        fill_model: Fill model (event-driven mode).
        risk_manager: External risk manager (optional).
        agent_expert: AI Agent expert (optional).
        agent_trigger_interval: Agent trigger interval (every N bars).
        agent_enabled_strategies: Agent-controlled strategy enable states.
        hooks: List of backtest hooks (optional).
        include_benchmark: Whether to compute buy-and-hold benchmark.

    Example:
        >>> engine = BacktestEngine(
        ...     fee_model=ChinaAShareFeeModel(),
        ...     initial_capital=100000,
        ...     stop_loss=0.05
        ... )
        >>> engine.set_strategy(my_strategy)
        >>> results = engine.run(data)
        >>> print(results.summary())
    """

    def __init__(
        self,
        fee_model: Optional[BaseFeeModel] = None,
        initial_capital: float = 100000,
        mode: Union[str, ExecutionMode] = ExecutionMode.VECTORIZED,
        position_sizing: Union[str, PositionSizing] = PositionSizing.FIXED_FRACTION,
        position_size: float = 0.95,
        max_position_size: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        allow_short: bool = True,
        fill_model: FillModel = FillModel.NEXT_BAR_OPEN,
        risk_manager=None,
        agent_expert=None,
        agent_trigger_interval: int = 1,
        agent_enabled_strategies: Optional[Dict[str, bool]] = None,
        hooks: Optional[List[BacktestHook]] = None,
        include_benchmark: bool = True,
        fee_allocation: str = "proportional",
        data_validation: str = "warn",
        session_end_time: Optional[time] = None,
        immediate_fill_price: str = "close",
        capital_allocator=None,
        asset_fee_models: Optional[Dict] = None,
        asset_fill_models: Optional[Dict] = None,
        margin_config=None,
        asset_margin_configs: Optional[Dict] = None,
        periodic_cost_model=None,
        portfolio_exit_rules: Optional[List] = None,
        lot_size: int = 1,
        asset_lot_sizes: Optional[Dict] = None,
        max_position_pct: float = 1.0,
        asset_max_position_pcts: Optional[Dict] = None,
        signal_transform=None,
        turnover_config=None,
        risk_rules=None,
        trailing_stop_rule=None,
        initial_universe: Optional[List[str]] = None,
        impact_model=None,
        impact_adv_lookback: int = 20,
        impact_vol_lookback: int = 20,
        trading_days: Union[int, Dict[str, int]] = TRADING_DAYS_STOCK,
        portfolio_trading_days: Optional[int] = None,
        representative_notional: Optional[float] = None,
        representative_size: Optional[float] = None,
        settlement: str = "t+0",
        asset_settlements: Optional[Dict[str, str]] = None,
        spread_model=None,
        asset_spread_models: Optional[Dict] = None,
    ):
        # Fee model
        if isinstance(fee_model, str):
            self.fee_model = get_fee_model(fee_model)
        else:
            self.fee_model = fee_model or SimpleFeeModel()

        # Capital
        self.initial_capital = initial_capital

        # Execution mode
        if isinstance(mode, str):
            mode = ExecutionMode(mode.lower())
        self.mode = mode

        # Position sizing
        if isinstance(position_sizing, str):
            position_sizing = PositionSizing(position_sizing.lower())
        self.position_sizing = position_sizing
        self.position_size = position_size
        self.max_position_size = max_position_size

        # Risk management
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.allow_short = allow_short

        # Fill model
        self.fill_model = fill_model

        # Risk manager (optional). risk_manager= and risk_rules= cover
        # the same need via two interfaces; passing both is ambiguous.
        if risk_manager is not None and risk_rules is not None:
            raise ValueError(
                "Pass either risk_manager= or risk_rules=, not both. "
                "Prefer risk_rules= for new code; risk_manager= remains "
                "for users with a custom BaseRiskManager subclass."
            )
        self.risk_manager = risk_manager
        if risk_manager:
            risk_manager.initialize(initial_capital)

        # AI Agent (optional)
        self.agent_expert = agent_expert
        self.agent_trigger_interval = agent_trigger_interval
        self.agent_enabled_strategies = agent_enabled_strategies or {}
        self._agent_bar_count = 0

        # Hooks (optional)
        self.hooks: List[BacktestHook] = hooks or []

        # Benchmark
        self.include_benchmark = include_benchmark

        # Fee allocation for partial close trades
        self.fee_allocation = fee_allocation

        # Data validation level ('strict', 'warn', 'none')
        self.data_validation = data_validation

        # Session end time for DAY order expiration
        self.session_end_time = session_end_time

        # Fill price for same-bar IOC/FOK second pass
        self.immediate_fill_price = immediate_fill_price

        # Multi-asset (v0.7)
        self.capital_allocator = capital_allocator
        self.asset_fee_models: Dict = asset_fee_models or {}
        self.asset_fill_models: Dict = asset_fill_models or {}

        # Margin (v0.8)
        self.margin_config = margin_config
        self.asset_margin_configs: Dict = asset_margin_configs or {}
        self.periodic_cost_model = periodic_cost_model
        self.portfolio_exit_rules: List = portfolio_exit_rules or []

        # Lot sizes (v0.8)
        if not isinstance(lot_size, int) or lot_size < 1:
            raise ValueError(f"lot_size must be an int >= 1, got {lot_size!r}")
        self.lot_size = lot_size
        self.signal_transform = signal_transform
        self.turnover_config = turnover_config
        self.risk_rules = risk_rules
        self.trailing_stop_rule = trailing_stop_rule
        self.initial_universe = initial_universe

        # Market impact (v1.9.4)
        self.impact_model = impact_model
        self.impact_adv_lookback = impact_adv_lookback
        self.impact_vol_lookback = impact_vol_lookback

        # Annualisation (v1.9.5).
        # bool is a subclass of int in Python; reject it first so
        # BacktestEngine(trading_days=True) doesn't silently become 1.
        if isinstance(trading_days, bool):
            raise TypeError(
                "trading_days must be int or Dict[str, int], not bool")
        if not isinstance(trading_days, (int, dict)):
            raise TypeError(
                f"trading_days must be int or Dict[str, int], "
                f"got {type(trading_days).__name__}"
            )
        if isinstance(trading_days, int) and trading_days < 1:
            raise ValueError(f"trading_days must be >= 1, got {trading_days}")
        if isinstance(trading_days, dict):
            if not trading_days:
                raise ValueError(
                    "trading_days dict is empty; pass a scalar int or "
                    "populate the dict with {symbol: int}")
            for k, v in trading_days.items():
                if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                    raise ValueError(
                        f"trading_days[{k!r}] must be int >= 1, got {v!r}")
        if portfolio_trading_days is not None:
            if isinstance(portfolio_trading_days, bool):
                raise TypeError("portfolio_trading_days must be int, not bool")
            if portfolio_trading_days < 1:
                raise ValueError(
                    f"portfolio_trading_days must be >= 1, "
                    f"got {portfolio_trading_days}")
        self.trading_days: Union[int, Dict[str, int]] = trading_days
        self.portfolio_trading_days_override: Optional[int] = portfolio_trading_days

        # Resolved at run time in run() once active symbols are known:
        self._portfolio_trading_days: int = (
            portfolio_trading_days if portfolio_trading_days is not None
            else (trading_days if isinstance(trading_days, int) else TRADING_DAYS_STOCK)
        )
        self._resolved_per_asset_td: Dict[str, int] = {}
        self.asset_lot_sizes: Dict = asset_lot_sizes or {}
        for sym, ls in self.asset_lot_sizes.items():
            if not isinstance(ls, int) or ls < 1:
                raise ValueError(
                    f"lot_size for '{sym}' must be an int >= 1, got {ls!r}")

        # Per-asset position limits (v0.8)
        if not 0 < max_position_pct <= 1.0:
            raise ValueError(
                f"max_position_pct must be in (0, 1.0], got {max_position_pct}")
        self.max_position_pct = max_position_pct
        self.asset_max_position_pcts: Dict = asset_max_position_pcts or {}
        for sym, pct in self.asset_max_position_pcts.items():
            if not 0 < pct <= 1.0:
                raise ValueError(
                    f"max_position_pct for '{sym}' must be in (0, 1.0], "
                    f"got {pct}")

        # v2.8.6: settlement constraint ("t+0" or "t+1").
        # T+1 is an exchange-level rule (SSE/SZSE cash equities): shares
        # bought today cannot be sold the same calendar day. It applies
        # to all participants, so it is a market-microstructure config,
        # not an account attribute. Note t+1 does NOT imply no-shorting;
        # pass allow_short=False as well for cash A-share realism.
        _valid_settlements = ("t+0", "t+1")
        if settlement not in _valid_settlements:
            raise ValueError(
                f"settlement must be one of {_valid_settlements}, "
                f"got {settlement!r}")
        self.settlement = settlement
        self.asset_settlements: Dict[str, str] = asset_settlements or {}
        for sym, stl in self.asset_settlements.items():
            if stl not in _valid_settlements:
                raise ValueError(
                    f"asset_settlements[{sym!r}] must be one of "
                    f"{_valid_settlements}, got {stl!r}")

        # v2.8.6: bid-ask spread models (see spread.py). Event-driven
        # honors any BaseSpreadModel; vectorized folds a global
        # FixedSpread into the linear cost approximation and ignores
        # everything else with a per-run warning.
        self.spread_model = spread_model
        self.asset_spread_models: Dict = asset_spread_models or {}

        # Custom benchmark config defaults
        self._config_benchmark: Optional[pd.Series] = None
        self._config_benchmark_type: str = "auto"
        self._config_benchmark_name: str = "Buy & Hold"

        # Feature modules
        self._stop_loss_rule = (
            PercentageStopLoss(stop_loss) if stop_loss else None
        )
        self._take_profit_rule = (
            PercentageTakeProfit(take_profit) if take_profit else None
        )
        self._trade_cost = DefaultTradeCost()
        self._position_sizer = self._create_position_sizer()

        # v2.8.1: representative trade size for vectorized cost estimation.
        # User-passed values win; otherwise derived from the sizer in
        # _build_config() at run time.
        self.representative_notional = representative_notional
        self.representative_size = representative_size

        # Internal state
        self._strategy = None
        self._signals = None
        self._data = None
        self._target_weights = None

        # v2.7 wide-input state — see Engine state vector in v2.7 plan.
        # Declared upfront so each setter can zero them independently
        # without ordering coupling across commits.
        self._signals_wide: Optional[pd.DataFrame] = None
        self._signals_wide_inf_action: Literal["silent", "warn", "raise"] = "warn"
        self._signals_wide_strict: bool = False
        self._target_weights_wide: Optional[pd.DataFrame] = None
        self._target_weights_wide_config = None  # _TargetWeightsWideConfig | None — set in v2.7 Commit C

    def _create_position_sizer(self):
        """Create the appropriate position sizer based on config."""
        if self.position_sizing == PositionSizing.FIXED_SIZE:
            return FixedSizer(self.position_size)
        elif self.position_sizing == PositionSizing.ALL_IN:
            return AllInSizer(self.position_size)
        elif self.position_sizing == PositionSizing.FIXED_FRACTION:
            return FractionSizer(self.position_size)
        else:
            # RISK_BASED falls back to FractionSizer
            warnings.warn(
                "RISK_BASED position sizing is not yet implemented, "
                "falling back to FIXED_FRACTION"
            )
            return FractionSizer(self.position_size)

    # ========== Setup Methods ==========

    def _clear_wide_input_state(self) -> None:
        """Zero the v2.7 wide-input state fields. Called by every
        setter for mutual exclusion across the 8-field state vector.
        """
        self._signals_wide = None
        self._signals_wide_inf_action = "warn"
        self._signals_wide_strict = False
        self._target_weights_wide = None
        self._target_weights_wide_config = None

    def set_strategy(self, strategy) -> 'BacktestEngine':
        """
        Set the trading strategy.

        Parameters:
            strategy: Strategy object with a ``generate_signals`` method.

        Returns:
            self: For method chaining.
        """
        self._strategy = strategy
        self._signals = None
        self._target_weights = None
        self._clear_wide_input_state()
        return self

    def set_signals(
        self, signals: Union[pd.Series, Dict[str, pd.Series]],
    ) -> 'BacktestEngine':
        """
        Set pre-computed trading signals directly.

        Parameters:
            signals: Signal series (single-asset) or dict of signal
                series keyed by symbol (multi-asset).

        Returns:
            self: For method chaining.

        Raises:
            TypeError: if ``signals`` is a ``pd.DataFrame``. v2.7
                tightens the contract — wide-layout input goes through
                :meth:`set_signals_wide` instead. Previously a DataFrame
                here would crash deep in the engine with a confusing
                ``AttributeError``; v2.7 refuses cleanly at the
                boundary. This is a breaking change from v2.6 only for
                callers who relied on the prior crash being caught
                upstream.
            TypeError: if ``signals`` is neither ``pd.Series`` nor
                ``dict[str, pd.Series]`` (v2.8.2).
            TypeError: from ``validate_signal_series`` if the Series
                (or any dict value) has a non-``DatetimeIndex`` or
                non-numeric dtype (v2.8.2).
            ValueError: from ``validate_signal_series`` on duplicate
                timestamps (v2.8.2). Engine-level ``data_validation
                ="none"`` skips this validation for parity with
                :func:`aiphaforge.utils.validate_ohlcv`.
        """
        if isinstance(signals, pd.DataFrame):
            raise TypeError(
                "set_signals only accepts pd.Series (single-asset) or "
                "dict[str, pd.Series] (multi-asset). For wide-layout "
                "DataFrame input, use set_signals_wide(df) instead. "
                "(See README v2.7 release notes — this is a breaking "
                "change from v2.6.)"
            )
        # v2.8.2 M3: boundary validation. set_signals_wide already
        # calls validate_signal_wide; this closes the gap on the
        # single-Series / per-symbol-dict path. Respects engine
        # data_validation="none" for parity with validate_ohlcv.
        if self.data_validation != "none":
            from .signals import validate_signal_series
            if isinstance(signals, dict):
                for sym, s in signals.items():
                    try:
                        validate_signal_series(s)
                    except (ValueError, TypeError) as exc:
                        raise type(exc)(
                            f"set_signals: invalid signals[{sym!r}] — {exc}"
                        ) from exc
            elif isinstance(signals, pd.Series):
                validate_signal_series(signals)
            else:
                raise TypeError(
                    f"set_signals expected pd.Series or "
                    f"dict[str, pd.Series]; got "
                    f"{type(signals).__name__}"
                )
        self._signals = signals
        self._strategy = None
        self._target_weights = None
        self._clear_wide_input_state()
        return self

    def set_signals_wide(
        self,
        signal_wide: pd.DataFrame,
        *,
        warn_on_inf: bool = True,
        strict: bool = False,
    ) -> 'BacktestEngine':
        """Set pre-computed wide-layout signals (index=datetime, columns=symbol).

        Materialization is deferred to ``run()``: at run time the wide
        DataFrame is converted to ``dict[str, pd.Series]`` against the
        actual ``data`` argument (single-asset path synthesizes the
        dict via ``{symbol: data}``).

        Parameters
        ----------
        signal_wide
            Wide-layout signal DataFrame. ``index`` MUST be a
            ``DatetimeIndex`` without duplicates (validated via
            :func:`signals.validate_signal_wide`).
        warn_on_inf
            If True (default), ±Inf in the wide DF emits a warning at
            materialization (and is coerced to NaN per the signal
            contract). If False, coerce silently.
        strict
            CI / fail-fast mode. When True: ±Inf raises instead of
            warning. **Incompatible with ``warn_on_inf=False``** —
            passing both raises ``ValueError`` immediately at this
            setter call (per v2.7 plan v3-decision #20: strict always
            wins, conflicting explicit kwargs raise at setter time).
        """
        if not isinstance(signal_wide, pd.DataFrame):
            raise TypeError(
                f"set_signals_wide expects pd.DataFrame, got "
                f"{type(signal_wide).__name__}; use set_signals for "
                f"Series/dict input"
            )
        if strict and not warn_on_inf:
            raise ValueError(
                "set_signals_wide(strict=True) is incompatible with "
                "warn_on_inf=False; pass strict=False to opt out of "
                "Inf checks"
            )
        validate_signal_wide(signal_wide, forbid_tz=True)

        # Compute tri-state inf_action from (strict, warn_on_inf):
        # (True, True) → raise (strict short-circuits)
        # (True, False) → already rejected above
        # (False, True) → warn
        # (False, False) → silent
        if strict:
            inf_action: Literal["silent", "warn", "raise"] = "raise"
        elif warn_on_inf:
            inf_action = "warn"
        else:
            inf_action = "silent"

        self._signals_wide = signal_wide
        self._signals_wide_inf_action = inf_action
        self._signals_wide_strict = strict
        # Mutual exclusion against the other 5 setter groups:
        self._strategy = None
        self._signals = None
        self._target_weights = None
        self._target_weights_wide = None
        self._target_weights_wide_config = None
        return self

    def set_score_wide(
        self,
        score_wide: pd.DataFrame,
        rule,
        *,
        warn_on_inf: bool = True,
        strict: bool = False,
    ) -> 'BacktestEngine':
        """Set wide-layout scores; ``rule.transform(scores)`` → signals.

        Routes raw scores through an explicit ``ScoreToSignalRule``
        (e.g. :class:`signal_rules.ThresholdScoreRule`,
        :class:`signal_rules.CrossSectionalQuantileRule`) before
        delegating to :meth:`set_signals_wide`. Makes the
        score-to-signal gate visible at the engine boundary so an ML
        model emitting probabilities cannot be silently routed as
        direction signals.

        Parameters
        ----------
        score_wide
            Wide-layout score DataFrame.
        rule
            Object with a ``.transform(scores) → DataFrame`` method.
        warn_on_inf, strict
            Forwarded to :meth:`set_signals_wide`. Same strict
            semantics — strict + warn_on_inf=False raises immediately.

        Notes
        -----
        Score-rule default asymmetry (per plan v3-decision #9):
        ``ThresholdScoreRule`` defaults to ``neutral_action="hold"``
        (NaN); ``CrossSectionalQuantileRule`` defaults to
        ``neutral_action="flat"`` (0). The two emit different
        post-warmup behavior on the same score frame — read each
        rule's docstring before picking.
        """
        if not isinstance(score_wide, pd.DataFrame):
            raise TypeError(
                f"set_score_wide expects pd.DataFrame, got "
                f"{type(score_wide).__name__}"
            )
        # Validate the INPUT score frame here so tz-aware / duplicate-
        # index errors are attributed to the user's score, not blamed
        # on the rule via the downstream set_signals_wide wrapper.
        validate_signal_wide(score_wide, forbid_tz=True)
        if not hasattr(rule, "transform"):
            raise TypeError(
                f"set_score_wide rule must have a .transform(scores) "
                f"method (use ThresholdScoreRule, "
                f"CrossSectionalQuantileRule, or a callable wrapped in "
                f"such); got {type(rule).__name__}"
            )
        if strict and not warn_on_inf:
            raise ValueError(
                "set_score_wide(strict=True) is incompatible with "
                "warn_on_inf=False; pass strict=False to opt out of "
                "Inf checks"
            )

        # Atomicity (per plan v2-decision #6): zero out the 5 other
        # setter fields BEFORE invoking rule.transform so a transform
        # raise cannot leave stale _signals from a prior call.
        self._strategy = None
        self._signals = None
        self._target_weights = None
        self._target_weights_wide = None
        self._target_weights_wide_config = None

        # Wrap rule.transform so any failure clearly blames the rule.
        try:
            signal_wide = rule.transform(score_wide)
        except Exception as exc:
            raise type(exc)(
                f"{type(rule).__name__}.transform raised: {exc}"
            ) from exc

        if not isinstance(signal_wide, pd.DataFrame):
            raise TypeError(
                f"{type(rule).__name__}.transform returned "
                f"{type(signal_wide).__name__}, expected pd.DataFrame. "
                f"set_score_wide requires a wide-DataFrame-returning "
                f"rule (e.g. ThresholdScoreRule, "
                f"CrossSectionalQuantileRule)."
            )

        # Delegate with re-raise wrapping so set_signals_wide-side
        # validation failures (duplicate index, etc) blame the rule
        # that produced the malformed frame.
        try:
            return self.set_signals_wide(
                signal_wide, warn_on_inf=warn_on_inf, strict=strict,
            )
        except (TypeError, ValueError) as exc:
            raise type(exc)(
                f"{type(rule).__name__}.transform produced an invalid "
                f"frame: {exc}"
            ) from exc

    def set_target_weights(
        self,
        weights_schedule: Dict[str, Dict[str, float]],
    ) -> 'BacktestEngine':
        """Set target portfolio weights for rebalancing.

        Parameters:
            weights_schedule: Mapping of date string to per-symbol
                weight dict.  Example::

                    {
                        "2024-01-01": {"AAPL": 0.3, "TSLA": 0.7},
                        "2024-02-01": {"AAPL": 0.5, "TSLA": 0.5},
                    }

                Between rebalance dates, positions are held (NaN signal).
                Weight=0 on a rebalance date closes the position.

        Returns:
            self: For method chaining.
        """
        self._target_weights = weights_schedule
        self._signals = None
        self._strategy = None
        self._clear_wide_input_state()
        return self

    def set_target_weights_wide(
        self,
        weights: pd.DataFrame,
        *,
        rebalance_dates: Optional[Sequence] = None,
        snap: Optional[Literal["exact", "next", "previous"]] = None,
        universe_alignment: Literal["union", "intersection"] = "union",
        on_collision: Optional[Literal["warn", "raise", "first", "last"]] = None,
        strict: bool = False,
    ) -> 'BacktestEngine':
        """Set wide-layout target weights; materialized to schedule at run().

        Wraps :func:`signals.target_weight_wide_to_schedule` at the
        engine surface. ``snap`` and ``on_collision`` use ``None`` as
        the "default not explicitly passed" sentinel (per plan
        v3-decision #20 default-detection note) so strict-conflict
        checks can distinguish explicit-vs-default.

        Parameters
        ----------
        weights
            Wide-layout target-weight DataFrame, ``index=datetime,
            columns=symbol``. NaN weights are coerced to 0 by the
            underlying schedule adapter (explicit close on that bar).
        rebalance_dates
            Optional explicit rebalance schedule. If None, every row of
            ``weights`` is treated as a candidate rebalance date —
            common footgun: a daily-index ``weights`` frame with no
            ``rebalance_dates`` will fire 252 rebalances/year.
        snap, on_collision
            Forwarded to the adapter. Defaults are ``"exact"`` and
            ``"warn"`` respectively (resolved inside the body).
        strict
            CI / fail-fast. When True: ``snap`` defaults to ``"exact"``
            and ``on_collision`` defaults to ``"raise"`` if not
            explicitly passed; explicit conflicting values
            (``snap != "exact"`` or
            ``on_collision in {"warn", "first", "last"}``) raise
            ``ValueError`` immediately at this setter call.
        """
        if not isinstance(weights, pd.DataFrame):
            raise TypeError(
                f"set_target_weights_wide expects pd.DataFrame, got "
                f"{type(weights).__name__}"
            )
        validate_signal_wide(weights, forbid_tz=True)

        # Strict-conflict checks (per plan v3-decision #20).
        if strict:
            if on_collision is None:
                on_collision = "raise"
            elif on_collision in {"warn", "first", "last"}:
                raise ValueError(
                    f"set_target_weights_wide(strict=True) is "
                    f"incompatible with on_collision={on_collision!r}; "
                    f"strict mode raises on collision; pass "
                    f"strict=False to use that policy"
                )
            if snap is None:
                snap = "exact"
            elif snap != "exact":
                raise ValueError(
                    f"set_target_weights_wide(strict=True) is "
                    f"incompatible with snap={snap!r}; strict requires "
                    f"snap='exact'; pass strict=False to use that snap "
                    f"mode"
                )
        else:
            if on_collision is None:
                on_collision = "warn"
            if snap is None:
                snap = "exact"

        # Eagerly materialize rebalance_dates to list so:
        # (a) the frozen config is pickle-safe (generators aren't),
        # (b) chained run() calls don't see an exhausted iterator.
        rebalance_dates_list: Optional[List] = (
            list(rebalance_dates) if rebalance_dates is not None else None
        )
        cfg = _TargetWeightsWideConfig(
            rebalance_dates=rebalance_dates_list,
            snap=snap,
            universe_alignment=universe_alignment,
            on_collision=on_collision,
            strict=strict,
        )
        self._target_weights_wide = weights
        self._target_weights_wide_config = cfg
        # Mutual exclusion against the other 5 setter groups.
        self._strategy = None
        self._signals = None
        self._target_weights = None
        self._signals_wide = None
        self._signals_wide_inf_action = "warn"
        self._signals_wide_strict = False
        return self

    def set_fee_model(self, fee_model: Union[BaseFeeModel, str]) -> 'BacktestEngine':
        """
        Set the fee model.

        Parameters:
            fee_model: Fee model instance or market name string.

        Returns:
            self: For method chaining.
        """
        if isinstance(fee_model, str):
            self.fee_model = get_fee_model(fee_model)
        else:
            self.fee_model = fee_model
        return self

    # ========== Run Methods ==========

    def run(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        symbol: str = "default",
        *,
        benchmark: Optional[pd.Series] = None,
        benchmark_type: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        secondary_data: Optional[Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]] = None,
        secondary_bar_align: str = "close",
    ) -> BacktestResult:
        """
        Run the backtest.

        Parameters:
            data: OHLCV data (single-asset ``pd.DataFrame``) or dict of
                DataFrames keyed by symbol (multi-asset).
            start: Start date (optional, single-asset only).
            end: End date (optional, single-asset only).
            symbol: Instrument symbol (single-asset only).
            benchmark: Custom benchmark series (prices or returns).
            benchmark_type: Type of benchmark data.
            weights: Per-symbol weights for vectorized multi-asset.
            secondary_data: Secondary timeframe data (event-driven only).
                Mapping of timeframe name to DataFrame (global) or dict
                of per-symbol DataFrames.
            secondary_bar_align: Alignment mode for secondary bars.
                ``"close"`` (default) or ``"open"``.

        Returns:
            BacktestResult: Backtest results.
        """
        is_multi = isinstance(data, dict)

        # --- Multi-asset path ---
        if is_multi:
            # Resolve annualisation before delegating so _run_multi can
            # read self._portfolio_trading_days / _resolved_per_asset_td.
            self._portfolio_trading_days, self._resolved_per_asset_td = \
                _normalize_trading_days(
                    self.trading_days, sorted(data.keys()),
                    portfolio_override=self.portfolio_trading_days_override,
                )
            # v2.8.6: per-symbol annualization sanity check.
            for _sym in sorted(data.keys()):
                self._warn_annualization_mismatch(
                    data[_sym].index,
                    self._resolved_per_asset_td.get(
                        _sym, self._portfolio_trading_days),
                    symbol=_sym,
                )
            return self._run_multi(
                data, benchmark=benchmark,
                benchmark_type=benchmark_type, weights=weights,
                secondary_data=secondary_data,
                secondary_bar_align=secondary_bar_align,
            )

        # --- Single-asset path ---
        # Validate and prepare data
        data = self._prepare_data(data, start, end)
        self._data = data

        # Reset per-run state
        self._agent_bar_count = 0

        # Resolve annualisation for this run (single-asset)
        self._portfolio_trading_days, self._resolved_per_asset_td = \
            _normalize_trading_days(
                self.trading_days, [symbol],
                portfolio_override=self.portfolio_trading_days_override,
            )

        # v2.8.6: annualization sanity check (warn-only, never
        # auto-corrects — the inference is a heuristic).
        self._warn_annualization_mismatch(
            data.index, self._portfolio_trading_days)

        # Generate signals
        signals = self._get_signals(data, symbol=symbol)

        # Build config bundle (with run-time benchmark overrides)
        config = self._build_config(
            benchmark=benchmark,
            benchmark_type=benchmark_type,
            symbols=[symbol],
        )
        if self._target_weights is not None:
            config.is_weight_mode = True

        # Guard: multiple LatencyHook instances wrapping the same inner_hook
        latency_hooks = [h for h in self.hooks if isinstance(h, LatencyHook)]
        if len(latency_hooks) > 1:
            inner_ids: List[int] = []
            for lh in latency_hooks:
                iid = id(lh.inner_hook)
                if iid in inner_ids:
                    raise ValueError(
                        "Multiple LatencyHook instances wrap the same "
                        "inner_hook. Use a single LatencyHook per agent, or "
                        "SymbolRoutingLatencyHook for per-symbol latency."
                    )
                inner_ids.append(iid)

        # Validate secondary data
        if secondary_data is not None:
            for tf_name, tf_data in secondary_data.items():
                if isinstance(tf_data, pd.DataFrame):
                    validate_ohlcv(
                        tf_data,
                        required=['open', 'high', 'low', 'close'],
                        validation_level=self.data_validation,
                    )
                else:
                    for sym_name, sdf in tf_data.items():
                        validate_ohlcv(
                            sdf,
                            required=['open', 'high', 'low', 'close'],
                            validation_level=self.data_validation,
                        )

        # Dispatch to execution core
        if self.mode == ExecutionMode.VECTORIZED:
            self._fire_vectorized_lifecycle(
                config, {symbol: data}, [symbol], phase="start")
            try:
                raw = run_vectorized(data, signals, config, symbol)
            finally:
                self._fire_vectorized_lifecycle(
                    config, {symbol: data}, [symbol], phase="end")
        else:
            # Wrap single-asset as dict for the unified core
            raw = run_event_driven(
                data_dict={symbol: data},
                signals_dict={symbol: signals},
                config=config,
                symbols=[symbol],
                strategy=self._strategy,
                secondary_data=secondary_data,
                secondary_bar_align=secondary_bar_align,
            )

        return self._build_result(raw, data, config)

    def _run_multi(
        self,
        data_dict: Dict[str, pd.DataFrame],
        *,
        benchmark: Optional[pd.Series] = None,
        benchmark_type: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        secondary_data: Optional[Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]] = None,
        secondary_bar_align: str = "close",
    ) -> BacktestResult:
        """Run a multi-asset backtest."""
        from .capital_allocator import EqualWeightAllocator

        symbols = sorted(data_dict.keys())

        # v2.2.2 Commit D: validate AND normalize into a local dict
        # rather than mutating the caller's data_dict in place. Prior
        # behavior was to write normalized frames back via
        # `data_dict[sym] = ...`, which silently replaced the user's
        # frames with sorted/dtype-converted copies — breaking
        # patterns where the same data_dict is reused (e.g. running
        # multiple backtests on the same input).
        normalized: Dict[str, pd.DataFrame] = {}
        for sym, df in data_dict.items():
            validate_ohlcv(
                df,
                required=['open', 'high', 'low', 'close'],
                validation_level=self.data_validation,
            )
            normalized[sym] = ensure_datetime_index(df).sort_index().copy()
        # All downstream references must use `normalized`, not
        # `data_dict`. Replace the local binding to keep the rest of
        # this method's body unchanged. The CALLER's dict is unchanged.
        data_dict = normalized

        # Validate secondary data
        if secondary_data is not None:
            for tf_name, tf_data in secondary_data.items():
                if isinstance(tf_data, pd.DataFrame):
                    validate_ohlcv(
                        tf_data,
                        required=['open', 'high', 'low', 'close'],
                        validation_level=self.data_validation,
                    )
                else:
                    for sym_name, sdf in tf_data.items():
                        validate_ohlcv(
                            sdf,
                            required=['open', 'high', 'low', 'close'],
                            validation_level=self.data_validation,
                        )

        # Generate signals
        signals_dict = self._get_signals_multi(data_dict)

        # Build config
        config = self._build_config(
            benchmark=benchmark,
            benchmark_type=benchmark_type,
            symbols=symbols,
        )
        if self._target_weights is not None:
            config.is_weight_mode = True

        # Auto-set allocator for multi-asset if not provided
        if config.capital_allocator is None:
            if config.margin_config is not None:
                from .capital_allocator import MarginAllocator
                warnings.warn(
                    "No capital_allocator set for multi-asset margin mode. "
                    "Using MarginAllocator (buying_power based). "
                    "Set capital_allocator explicitly to suppress."
                )
                config.capital_allocator = MarginAllocator()
            else:
                warnings.warn(
                    "No capital_allocator set for multi-asset mode. "
                    "Using EqualWeightAllocator (equal budget per signal). "
                    "Set capital_allocator explicitly to suppress."
                )
                config.capital_allocator = EqualWeightAllocator()

        # Dispatch
        if self.mode == ExecutionMode.VECTORIZED:
            self._fire_vectorized_lifecycle(
                config, data_dict, symbols, phase="start")
            try:
                raw = self._run_vectorized_multi(
                    data_dict, signals_dict, config, weights)
            finally:
                self._fire_vectorized_lifecycle(
                    config, data_dict, symbols, phase="end")
        else:
            raw = run_event_driven(
                data_dict=data_dict,
                signals_dict=signals_dict,
                config=config,
                symbols=symbols,
                strategy=self._strategy,
                secondary_data=secondary_data,
                secondary_bar_align=secondary_bar_align,
            )

        # Build result (use first asset's data for benchmark alignment)
        first_df = data_dict[symbols[0]]
        result = self._build_result(raw, first_df, config)

        # Attach multi-asset fields
        if 'per_asset_pnl' in raw:
            result.per_asset_pnl = raw['per_asset_pnl']
        result.symbols = symbols

        # Group trades by symbol
        if result.trades:
            per_asset_trades = {}
            for t in result.trades:
                per_asset_trades.setdefault(t.symbol, []).append(t)
            result.per_asset_trades = per_asset_trades

        # Populate per_asset_metrics (v1.9.5 — fix of pre-existing gap).
        # _build_result already set result.trading_days and
        # per_asset_trading_days from self._resolved_per_asset_td.
        if result.per_asset_pnl:
            from .performance import PerformanceAnalyzer
            analyzer = PerformanceAnalyzer(
                result,
                trading_days=self._portfolio_trading_days,
                per_asset_trading_days=self._resolved_per_asset_td,
            )
            result.per_asset_metrics = analyzer.per_asset_analysis()

        return result

    def _materialize_target_weights_wide(
        self, data_dict: Dict[str, pd.DataFrame],
    ) -> Dict:
        """v2.7: convert ``self._target_weights_wide`` → schedule dict.

        Captures helper warnings via ``catch_warnings(record=True)`` so
        they can be re-emitted (preserving original stacklevel) AND a
        SECOND engine-layer warning can be emitted on collision-context
        per plan v3-decision #22.
        """
        cfg = self._target_weights_wide_config
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", UserWarning)
            schedule = target_weight_wide_to_schedule(
                self._target_weights_wide, data_dict,
                rebalance_dates=cfg.rebalance_dates,
                snap=cfg.snap,
                universe_alignment=cfg.universe_alignment,
                on_collision=cfg.on_collision,
            )
        # Re-emit captured warnings at original stacklevel for downstream
        # callers' compatibility.
        for w in captured:
            warnings.warn(w.message, w.category, stacklevel=2)
        # Engine-layer enriched warning per plan v3.2: detect via
        # substring "collision" in the captured message text. Helper at
        # signals.py:652 uses the literal word "collision(s)" which is
        # a controlled-text contract (doc-comment in signals.py calls
        # this out for future maintainers). Filename-based detection
        # is unreliable because helper warnings use stacklevel=2.
        if cfg.on_collision == "warn" and any(
            "collision" in str(w.message).lower() for w in captured
        ):
            warnings.warn(
                "set_target_weights_wide: collision warning above "
                "from target_weight_wide_to_schedule. For CI / strict "
                "pipelines pass on_collision='raise' or strict=True.",
                UserWarning,
                stacklevel=2,
            )
        return schedule

    def _materialize_signals_wide(
        self, data_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.Series]:
        """v2.7: convert ``self._signals_wide`` → dict[str, Series].

        Honors ``self._signals_wide_inf_action`` tri-state. Called by
        both ``_get_signals`` (single-asset, after data_dict synthesis)
        and ``_get_signals_multi`` (multi-asset). Per v2.7 plan
        v3-decision #4 the wide DataFrame is RETAINED across chained
        ``run()`` calls, so this method may be called multiple times.
        """
        action = self._signals_wide_inf_action
        if action == "raise":
            # Pre-check for Inf with a clean strict-mode error.
            arr = self._signals_wide.to_numpy(dtype=float)
            if np.isinf(arr).any():
                raise ValueError(
                    "set_signals_wide(strict=True): ±Inf values in "
                    "signal_wide are not permitted in strict mode."
                )
            return wide_to_signal_dict(
                self._signals_wide, data_dict, warn_on_inf=False,
            )
        return wide_to_signal_dict(
            self._signals_wide, data_dict,
            warn_on_inf=(action == "warn"),
        )

    def _get_signals_multi(
        self,
        data_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.Series]:
        """Get signals for multi-asset mode."""
        # v2.7 wide-input materialization: must run BEFORE the existing
        # dispatch so the materialized dict / schedule feeds the
        # standard `isinstance(self._signals, dict)` /
        # `self._target_weights is not None` branches below.
        if self._signals_wide is not None:
            self._signals = self._materialize_signals_wide(data_dict)
        if self._target_weights_wide is not None:
            self._target_weights = self._materialize_target_weights_wide(
                data_dict,
            )

        # Target weights mode: convert schedule to signal series
        if self._target_weights is not None:
            return self._weights_to_signals(
                self._target_weights, data_dict)

        if isinstance(self._signals, dict):
            signals_dict = {}
            for sym, df in data_dict.items():
                if sym in self._signals:
                    sig = self._signals[sym].reindex(df.index)
                else:
                    sig = pd.Series(np.nan, index=df.index, dtype=float)
                signals_dict[sym] = sig.replace(
                    [np.inf, -np.inf], np.nan)
            return signals_dict
        elif self._strategy is not None:
            result = self._strategy.generate_signals(data_dict)
            if isinstance(result, dict):
                return result
            raise TypeError(
                "Strategy.generate_signals() must return "
                "Dict[str, pd.Series] for multi-asset mode"
            )
        else:
            raise ValueError(
                "Must set either a strategy or signals (via set_signals "
                "or set_strategy) before running a multi-asset backtest"
            )

    @staticmethod
    def _weights_to_signals(
        weights_schedule: Dict[str, Dict[str, float]],
        data_dict: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.Series]:
        """Convert target weight schedule to per-symbol signal Series.

        Between rebalance dates: NaN (hold). On rebalance dates: weight value.
        """
        all_syms = set()
        for w_dict in weights_schedule.values():
            all_syms.update(w_dict.keys())
        all_syms.update(data_dict.keys())

        signals = {}
        for sym in data_dict:
            sig = pd.Series(np.nan, index=data_dict[sym].index, dtype=float)
            for date_str, w_dict in weights_schedule.items():
                ts = pd.Timestamp(date_str)
                if ts in sig.index:
                    sig.loc[ts] = w_dict.get(sym, 0.0)
            signals[sym] = sig
        return signals

    def _run_vectorized_multi(
        self,
        data_dict: Dict[str, pd.DataFrame],
        signals_dict: Dict[str, pd.Series],
        config: BacktestConfig,
        weights: Optional[Dict[str, float]] = None,
    ) -> dict:
        """Run vectorized multi-asset: per-asset runs + merge."""
        import dataclasses

        symbols = sorted(data_dict.keys())
        if weights is None:
            weights = {s: 1.0 / len(symbols) for s in symbols}

        # Validate weights
        for s, w in weights.items():
            if w <= 0:
                raise ValueError(
                    f"Weight for '{s}' must be > 0, got {w}")
        total_w = sum(weights.values())
        if total_w > 1.0 + 1e-9:
            raise ValueError(
                f"Sum of weights ({total_w:.4f}) exceeds 1.0")
        if total_w < 1.0 - 1e-9:
            warnings.warn(
                f"Sum of weights ({total_w:.4f}) < 1.0. "
                f"{1 - total_w:.1%} of capital held as cash."
            )

        per_asset = {}
        for sym in symbols:
            w = weights.get(sym, 0)
            asset_capital = config.initial_capital * w
            asset_config = dataclasses.replace(
                config, initial_capital=asset_capital)
            per_asset[sym] = run_vectorized(
                data_dict[sym], signals_dict[sym],
                asset_config, sym)

        return self._merge_vectorized_results(
            per_asset, config.initial_capital)

    @staticmethod
    def _fire_vectorized_lifecycle(
        config: BacktestConfig,
        data_dict: Dict[str, pd.DataFrame],
        symbols: List[str],
        *,
        phase: str,
    ) -> None:
        """Fire on_backtest_start / on_backtest_end once per vectorized run.

        Vectorized mode skips on_pre_signal and on_bar (no broker /
        portfolio is constructed), but lifecycle hooks fire so that
        users with stateful hooks (resets, dashboards, etc.) get called.

        For phase='end', each hook call is wrapped in try/except so
        a buggy end-hook cannot mask the primary engine exception
        (Python's try/finally re-raise semantics would otherwise lose
        the original RuntimeError from run_vectorized). Symmetric with
        the event-driven core (core_event_driven.py finally block).
        """
        if not config.hooks:
            return
        sorted_symbols = sorted(symbols)
        primary_sym = sorted_symbols[0]
        primary_data = data_dict[primary_sym]
        ts = primary_data.index[0] if phase == "start" else primary_data.index[-1]
        ctx = LifecycleContext(
            phase=phase,  # type: ignore[arg-type]
            timestamp=ts,
            symbols=sorted_symbols,
            config=config,
            data_dict=data_dict,
            primary_symbol=primary_sym,
            primary_data=primary_data,
        )
        dispatch = call_hook_lifecycle_start if phase == "start" else call_hook_lifecycle_end
        # End-hook exception policy (v1.9.8): suppress only when a
        # primary exception is in flight (so we don't mask the loop's
        # crash). On the success path, end-hook exceptions propagate
        # normally so buggy hooks fail visibly.
        import sys as _sys
        primary_in_flight = (
            phase == "end" and _sys.exc_info()[1] is not None
        )
        for hook in config.hooks:
            if phase == "end" and primary_in_flight:
                try:
                    dispatch(hook, ctx)
                except Exception as exc:
                    warnings.warn(
                        f"on_backtest_end raised on "
                        f"{type(hook).__name__}: {exc!r}. "
                        f"Suppressed so the primary exception "
                        f"propagates; original is still in __context__."
                    )
            else:
                dispatch(hook, ctx)

    @staticmethod
    def _merge_vectorized_results(
        per_asset: Dict[str, dict],
        initial_capital: float,
    ) -> dict:
        """Merge per-asset vectorized results into a portfolio result."""
        # Align and sum equity curves
        equity_curves = {}
        all_trades = []
        all_orders = []
        for sym, raw in per_asset.items():
            eq = raw['equity_curve']
            equity_curves[sym] = eq
            all_trades.extend(raw.get('trades', []))
            odf = raw.get('orders_df')
            if odf is not None and len(odf) > 0:
                all_orders.append(odf)

        eq_df = pd.DataFrame(equity_curves)
        # Forward-fill and sum
        eq_df = eq_df.ffill()
        portfolio_equity = eq_df.sum(axis=1)

        orders_df = (pd.concat(all_orders, ignore_index=True)
                     if all_orders else pd.DataFrame())

        from .utils import calculate_returns
        daily_returns = (calculate_returns(portfolio_equity)
                         if len(portfolio_equity) > 0 else None)

        # Per-asset PnL from independent equity curves
        per_asset_pnl = {}
        for sym, eq in equity_curves.items():
            per_asset_pnl[sym] = eq.diff().fillna(0.0)
            per_asset_pnl[sym].name = sym

        return {
            'equity_curve': portfolio_equity,
            'trades': all_trades,
            'positions_df': pd.DataFrame(),
            'orders_df': orders_df,
            'daily_returns': daily_returns,
            'final_capital': (float(portfolio_equity.iloc[-1])
                              if len(portfolio_equity) > 0 else 0.0),
            'per_asset_pnl': per_asset_pnl,
        }

    def _prepare_data(
        self,
        data: pd.DataFrame,
        start: Optional[str],
        end: Optional[str]
    ) -> pd.DataFrame:
        """Validate and prepare data."""
        validate_ohlcv(
            data,
            required=['open', 'high', 'low', 'close'],
            validation_level=self.data_validation,
        )
        data = ensure_datetime_index(data)
        data = data.sort_index()

        if start:
            data = data[data.index >= pd.Timestamp(start)]
        if end:
            data = data[data.index <= pd.Timestamp(end)]

        if len(data) == 0:
            raise ValueError("No data after date filtering")

        return data.copy()

    def _get_signals(
        self, data: pd.DataFrame, *, symbol: str = "default",
    ) -> pd.Series:
        """Get trading signals. NaN = hold, 0 = flat, nonzero = trade."""
        # v2.7 wide-input materialization (single-asset path). Synthesizes
        # data_dict = {symbol: data} per plan v3-decision #19.
        if self._signals_wide is not None:
            data_dict_single = {symbol: data}
            signal_dict = self._materialize_signals_wide(data_dict_single)
            self._signals = signal_dict[symbol]
        if self._target_weights_wide is not None:
            data_dict_single = {symbol: data}
            self._target_weights = self._materialize_target_weights_wide(
                data_dict_single,
            )

        if self._target_weights is not None:
            signals_dict = self._weights_to_signals(
                self._target_weights, {symbol: data})
            signals = signals_dict[symbol]
        elif self._signals is not None:
            signals = self._signals.reindex(data.index)
            # NaN from reindex means "no signal" = hold (preserve NaN)
        elif self._strategy is not None:
            signals = self._strategy.generate_signals(data)
        else:
            raise ValueError("Must set either a strategy or signals")

        signals = signals.replace([np.inf, -np.inf], np.nan)
        return signals

    # ========== Config and Result Building ==========

    # Fields the vectorized core silently ignores. Names are the kwargs
    # of __init__ (so we can read defaults via inspect.signature). The
    # `position_sizing` and `position_size` pair is handled jointly.
    # Note: vectorized DOES honor stop_loss (via stop_loss_rule),
    # risk_rules (via apply_vectorized_all), trade_cost (apply_vectorized),
    # signal_transform, and allow_short. Everything else listed here is
    # silently dropped — surface that to the user.
    # v2.8.1 H3: expanded from 7 to 21 entries after a line-by-line audit
    # of core_vectorized.py. The new entries fall into 4 buckets:
    #   - portfolio-shape: fill_model, max_position_size, session_end_time,
    #     immediate_fill_price, fee_allocation
    #   - multi-asset: capital_allocator (H4 — special-cased), asset_fee_models,
    #     asset_fill_models, asset_max_position_pcts, asset_lot_sizes
    #   - per-asset limits / lots: lot_size, max_position_pct
    #   - portfolio-level rules: portfolio_exit_rules, asset_margin_configs
    # Note: vectorized DOES honor stop_loss (via stop_loss_rule),
    # risk_rules (via apply_vectorized_all), trade_cost (apply_vectorized),
    # signal_transform, and allow_short. max_position_size is partially
    # honored — see _warn_vectorized_max_position_size_partial (H4 / FR-G2).
    _VECTORIZED_UNSUPPORTED_FIELDS: Tuple[str, ...] = (
        # v2.8.0 (7):
        "take_profit",
        "trailing_stop_rule",
        "impact_model",
        "margin_config",
        "periodic_cost_model",
        "turnover_config",
        "risk_manager",
        # v2.8.1 (14):
        "fill_model",
        "max_position_size",  # special partial-honor message
        "session_end_time",
        "immediate_fill_price",
        "fee_allocation",
        "capital_allocator",  # special divergence message
        "asset_fee_models",
        "asset_fill_models",
        "asset_max_position_pcts",
        "asset_lot_sizes",
        "lot_size",
        "max_position_pct",
        "portfolio_exit_rules",
        "asset_margin_configs",
    )

    def _warn_vectorized_capital_allocator_divergence(self) -> None:
        """v2.8.1 H4 — special message for capital_allocator.

        Vectorized multi-asset uses a static equal-weight split; the
        capital_allocator is silently ignored. This produces wrong PnL
        for users who set up dynamic allocators.
        """
        warnings.warn(
            "vectorized multi-asset mode uses static equal-weight "
            f"capital split; your capital_allocator="
            f"{type(self.capital_allocator).__name__} is ignored. Switch "
            "to mode='event_driven' for dynamic per-bar allocation."
        )

    def _warn_annualization_mismatch(
        self,
        index,
        trading_days: int,
        symbol: Optional[str] = None,
    ) -> None:
        """v2.8.6: warn when trading_days is far off the bar density.

        ``trading_days`` is bars-per-year throughout this codebase
        (sqrt(trading_days) Sharpe scaling), so intraday data with the
        default 252 silently mis-annualizes by an order of magnitude
        (crypto 1h is ~8760 bars/year). Warn-only beyond a 3x band in
        either direction; emitted unconditionally at this single call
        site per run (plain warnings.warn, no once-flags). Never
        auto-corrects: the density inference is a heuristic and an
        explicit in-band user value always wins silently.
        """
        inferred = infer_bars_per_year(index)
        if inferred is None or trading_days <= 0:
            return
        ratio = inferred / trading_days
        if ratio >= 3.0 or ratio <= 1.0 / 3.0:
            where = f" for {symbol!r}" if symbol is not None else ""
            warnings.warn(
                f"annualization mismatch{where}: trading_days="
                f"{trading_days}, but the data's bar density implies "
                f"~{round(inferred)} bars/year ({ratio:.1f}x off). "
                f"Sharpe and annualized metrics scale with "
                f"sqrt(trading_days); pass trading_days="
                f"{round(inferred)} if the data frequency is intended."
            )

    def _warn_vectorized_max_position_size_partial(self) -> None:
        """v2.8.1 FR-G2 — special message for max_position_size.

        Vectorized mode uses max_position_size only for representative
        cost-estimation sizing via the position_sizer (per v2.8.1 H1).
        Explicit per-bar position clamping requires event-driven mode.
        """
        warnings.warn(
            "vectorized mode uses max_position_size only for "
            "cost-estimation sizing via the position_sizer (per "
            "v2.8.1 H1); explicit per-bar position clamping requires "
            "mode='event_driven'."
        )

    def _warn_vectorized_unsupported(self) -> None:
        """Warn when vectorized mode is given config it silently ignores.

        Reads field defaults from __init__'s signature so this stays
        in sync if someone changes the kwarg defaults later. The
        (position_sizing, position_size) pair is treated as one
        composite warning (a user changing one usually changes the
        other together — no need to fire twice).
        """
        import inspect
        init_defaults = {
            name: param.default
            for name, param in
            inspect.signature(BacktestEngine.__init__).parameters.items()
            if param.default is not inspect.Parameter.empty
        }

        # Composite (position_sizing, position_size) warning.
        ps_def = init_defaults.get("position_sizing")
        sz_def = init_defaults.get("position_size")
        if (self.position_sizing != ps_def
                or self.position_size != sz_def):
            warnings.warn(
                "vectorized mode takes positions directly from signals; "
                f"(position_sizing={self.position_sizing}, "
                f"position_size={self.position_size}) is ignored. "
                "Switch to mode='event_driven' to honor sizing config."
            )

        # Per-field warnings for the rest. Field-specific dispatchers
        # take precedence over the generic message.
        special = {
            "capital_allocator": self._warn_vectorized_capital_allocator_divergence,
            "max_position_size": self._warn_vectorized_max_position_size_partial,
        }
        for field in self._VECTORIZED_UNSUPPORTED_FIELDS:
            default = init_defaults.get(field)
            value = getattr(self, field, default)
            # Fields stored with `X or {}` / `X or []` normalization
            # show up as {} / [] in self even when the user passed
            # None. Treat empty collections as "not configured".
            if isinstance(value, (dict, list)) and not value:
                continue
            if value != default:
                handler = special.get(field)
                if handler is not None:
                    handler()
                else:
                    warnings.warn(
                        f"vectorized mode ignores {field}={value!r}; "
                        "switch to mode='event_driven' to honor it."
                    )

        # v2.8.6: spread models — standalone check, NOT part of
        # _VECTORIZED_UNSUPPORTED_FIELDS: a global FixedSpread IS
        # honored (folded into the linear cost approximation); only
        # dynamic models and per-asset overrides are ignored.
        from .spread import FixedSpread
        if (self.spread_model is not None
                and not isinstance(self.spread_model, FixedSpread)):
            warnings.warn(
                "vectorized mode folds FixedSpread only; "
                f"spread_model={type(self.spread_model).__name__} is "
                "ignored. Switch to mode='event_driven' for dynamic "
                "spread models."
            )
        if self.asset_spread_models:
            warnings.warn(
                "vectorized mode cannot fold per-asset spread "
                "overrides; asset_spread_models is ignored (the global "
                "FixedSpread, if any, still folds). Switch to "
                "mode='event_driven' to honor per-asset spreads."
            )

    def _build_config(
        self,
        benchmark: Optional[pd.Series] = None,
        benchmark_type: Optional[str] = None,
        symbols: Optional[List[str]] = None,
    ) -> BacktestConfig:
        """Build a BacktestConfig from the engine's attributes.

        Parameters:
            benchmark: Run-time benchmark override (takes precedence over
                the engine-level config).
            benchmark_type: Run-time benchmark_type override.
            symbols: List of symbols for this run.
        """
        # v1.9.7: warn if vectorized mode + non-default unsupported config.
        # Vectorized core only honors a subset of engine config; the rest
        # are silently dropped today. Surface these explicitly so users
        # don't get misleading "I set X but nothing changed" results.
        if self.mode == ExecutionMode.VECTORIZED:
            # v2.8.6: T+1 cannot be honored by the vectorized core;
            # silently ignoring it would yield optimistic results, so
            # raise instead of warn (any "t+1", global or per-asset).
            if (self.settlement == "t+1"
                    or "t+1" in self.asset_settlements.values()):
                raise ValueError(
                    "settlement='t+1' is not supported in vectorized "
                    "mode; use mode='event_driven'")
            self._warn_vectorized_unsupported()

        # v2.8.1: resolve representative notional/size for vectorized
        # cost estimation. User-passed engine kwargs win; otherwise
        # derive from the sizer.
        rep_notional, rep_size = self._resolve_representative_trade()

        # v2.8.6: fold a global FixedSpread into the vectorized cost
        # approximation (one half-spread per side). Assigned
        # UNCONDITIONALLY on every call so a spread_model mutation
        # between runs never leaves a stale rate on the shared
        # DefaultTradeCost instance.
        from .spread import FixedSpread
        self._trade_cost.half_spread_rate = (
            self.spread_model.spread_bps / 2.0 / 1e4
            if isinstance(self.spread_model, FixedSpread) else 0.0)

        return BacktestConfig(
            initial_capital=self.initial_capital,
            fee_model=self.fee_model,
            allow_short=self.allow_short,
            fee_allocation=self.fee_allocation,
            fill_model=self.fill_model,
            stop_loss_rule=self._stop_loss_rule,
            take_profit_rule=self._take_profit_rule,
            trade_cost=self._trade_cost,
            position_sizer=self._position_sizer,
            risk_manager=self.risk_manager,
            hooks=self.hooks,
            include_benchmark=self.include_benchmark,
            data_validation=self.data_validation,
            max_position_size=self.max_position_size,
            session_end_time=self.session_end_time,
            immediate_fill_price=self.immediate_fill_price,
            mode=self.mode.value,
            has_signals=self._signals is not None,
            has_strategy=self._strategy is not None,
            benchmark=benchmark if benchmark is not None else self._config_benchmark,
            benchmark_type=benchmark_type if benchmark_type is not None else self._config_benchmark_type,
            benchmark_name=self._config_benchmark_name,
            symbols=symbols or [],
            capital_allocator=self.capital_allocator,
            asset_fee_models=self.asset_fee_models,
            asset_fill_models=self.asset_fill_models,
            margin_config=self.margin_config,
            asset_margin_configs=self.asset_margin_configs,
            periodic_cost_model=self.periodic_cost_model,
            portfolio_exit_rules=self.portfolio_exit_rules,
            lot_size=self.lot_size,
            asset_lot_sizes=self.asset_lot_sizes,
            max_position_pct=self.max_position_pct,
            asset_max_position_pcts=self.asset_max_position_pcts,
            signal_transform=self.signal_transform,
            turnover_config=self.turnover_config,
            risk_rules=self.risk_rules,
            trailing_stop_rule=self.trailing_stop_rule,
            initial_universe=self.initial_universe,
            impact_model=self.impact_model,
            impact_adv_lookback=self.impact_adv_lookback,
            impact_vol_lookback=self.impact_vol_lookback,
            representative_notional=rep_notional,
            representative_size=rep_size,
            settlement=self.settlement,
            asset_settlements=self.asset_settlements,
            spread_model=self.spread_model,
            asset_spread_models=self.asset_spread_models,
        )

    def _resolve_representative_trade(self):
        """Resolve (representative_notional, representative_size) for
        vectorized cost estimation.

        Q3 precedence: ANY user-passed engine kwarg wins independently.
        If only ``representative_notional`` is set, use it (size derived
        in apply_vectorized). If only ``representative_size`` is set,
        use it (notional derived). If both, pass both through. Sizer
        dispatch only runs when BOTH are None.

        Sizer dispatch: Fraction/AllIn → notional via ``initial_capital
        * min(fraction, max_position_size)`` (per ``position_sizing.py``'s
        effective allocation); Fixed → size; anything else → both None
        (apply_vectorized then takes the zero-cost-warned degenerate
        branch).
        """
        # v2.8.1 post-review fix: honor user-passed values independently.
        # The earlier `if representative_notional is not None: return
        # (notional, size)` branch silently dropped a user-passed
        # representative_size when notional was unset.
        if (self.representative_notional is not None
                or self.representative_size is not None):
            return self.representative_notional, self.representative_size
        sizer = self._position_sizer
        if isinstance(sizer, (FractionSizer, AllInSizer)):
            notional = self.initial_capital * min(
                sizer.fraction, self.max_position_size
            )
            return notional, None
        if isinstance(sizer, FixedSizer):
            return None, sizer.size
        return None, None

    def _build_result(
        self,
        raw: dict,
        data: pd.DataFrame,
        config: Optional[BacktestConfig] = None,
    ) -> BacktestResult:
        """Build a BacktestResult from raw core output."""
        equity_curve = raw['equity_curve']
        trades = raw['trades']
        positions_df = raw['positions_df']
        net_returns = raw.get('net_returns')
        daily_returns = raw.get('daily_returns')
        orders_df = raw.get('orders_df', pd.DataFrame())
        final_capital = raw.get('final_capital', 0.0)

        # Determine which returns series to use for metrics
        returns_for_metrics = net_returns if net_returns is not None else daily_returns

        # Compute metrics
        if returns_for_metrics is not None and len(returns_for_metrics) > 0:
            metrics = self._calculate_metrics(returns_for_metrics, equity_curve, trades)
        else:
            metrics = {}

        # Strategy name
        strategy_name = (
            getattr(self._strategy, 'name', 'Custom')
            if self._strategy else "Custom"
        )

        # Benchmark
        benchmark_equity = None
        benchmark_metrics = None
        benchmark_name = "Buy & Hold"
        if self.include_benchmark:
            benchmark_equity, benchmark_metrics, benchmark_name = (
                self._compute_benchmark(data, config)
            )

        # Use net_returns or daily_returns for the result
        result_returns = net_returns if net_returns is not None else daily_returns

        result_kwargs = dict(
            equity_curve=equity_curve,
            trades=trades,
            positions=positions_df,
            metrics=metrics,
            initial_capital=self.initial_capital,
            strategy_name=strategy_name,
            parameters=(
                getattr(self._strategy, 'params', {})
                if self._strategy else {}
            ),
            daily_returns=result_returns,
            benchmark_equity=benchmark_equity,
            benchmark_metrics=benchmark_metrics,
            benchmark_name=benchmark_name,
        )

        if orders_df is not None and len(orders_df) > 0:
            result_kwargs['orders'] = orders_df

        if final_capital is not None:
            result_kwargs['final_capital'] = final_capital

        result = BacktestResult(**result_kwargs)

        # Attach turnover history if present
        if 'turnover_history' in raw:
            result.turnover_history = raw['turnover_history']

        # Attach MetaContext audit trail (v1.2)
        if 'meta_audit' in raw and raw['meta_audit']:
            result.metadata['meta_audit'] = raw['meta_audit']

        # v2.8.6: record spread-model presence for event-driven
        # summaries (vectorized suppresses the Total Spread line — the
        # folded spread lives in returns, not per-trade costs).
        if (self.mode == ExecutionMode.EVENT_DRIVEN
                and (self.spread_model is not None
                     or self.asset_spread_models)):
            result.metadata['spread_model'] = repr(
                self.spread_model or self.asset_spread_models)

        # Annualisation (v1.9.5). Multi-asset path overrides per_asset_trading_days
        # and populates per_asset_metrics after _build_result returns.
        result.trading_days = self._portfolio_trading_days
        result.per_asset_trading_days = dict(self._resolved_per_asset_td)

        # v1.9.7: populate result.symbols. Pre-fix this was empty for
        # single-asset runs (only multi-asset set it via _run_multi),
        # which silently broke any consumer that read result.symbols
        # (e.g. market_impact.estimate_capacity). config.symbols is set
        # in _build_config from the caller's symbols list, so it's
        # reliable for both single- and multi-asset paths.
        if config is not None and config.symbols:
            result.symbols = list(config.symbols)

        return result

    # ========== Performance Calculation ==========

    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity: pd.Series,
        trades: List[Trade],
        *,
        trading_days: Optional[int] = None,
    ) -> Dict[str, float]:
        """Calculate performance metrics.

        Delegates to shared utility functions so that the engine and
        PerformanceAnalyzer use the same calculations.

        Parameters:
            trading_days: Annualisation factor. Defaults to
                ``self._portfolio_trading_days`` — override when computing
                metrics for a single-asset benchmark inside a multi-asset
                run so the benchmark uses its own annualisation rather
                than the (possibly dict-max) portfolio value.
        """
        metrics: Dict[str, float] = {}

        if len(returns) == 0:
            return metrics

        # --- Return metrics ---
        if len(equity) > 0 and equity.iloc[0] != 0:
            total_return = equity.iloc[-1] / equity.iloc[0] - 1
        else:
            total_return = 0.0
        metrics['total_return'] = total_return

        n_days = len(returns)
        td = trading_days if trading_days is not None else self._portfolio_trading_days
        metrics['annualized_return'] = (
            annualize_return(total_return, n_days, td) if n_days > 0 else 0.0
        )

        # --- Risk metrics ---
        metrics['sharpe_ratio'] = calc_sharpe(returns, trading_days=td)
        metrics['sortino_ratio'] = calc_sortino(returns, trading_days=td)
        metrics['max_drawdown'] = calc_max_drawdown(equity)
        metrics['calmar_ratio'] = (
            metrics['annualized_return'] / metrics['max_drawdown']
            if metrics['max_drawdown'] > 0 else 0.0
        )

        # --- Trade metrics (delegated to utils) ---
        metrics.update(calculate_trade_metrics(trades))

        # --- Simple inline metrics ---
        metrics['volatility'] = float(returns.std() * np.sqrt(td))
        metrics['mean_daily_return'] = float(returns.mean())
        metrics['win_days'] = int((returns > 1e-8).sum())
        metrics['lose_days'] = int((returns < -1e-8).sum())
        metrics['flat_days'] = int(len(returns) - metrics['win_days'] - metrics['lose_days'])

        return metrics

    def _compute_benchmark(
        self,
        data: pd.DataFrame,
        config: Optional[BacktestConfig] = None,
    ) -> tuple:
        """Compute benchmark equity and metrics.

        If a custom benchmark series is available (via *config*), it is
        used after type detection and alignment.  Otherwise the default
        buy-and-hold benchmark is computed via :func:`compute_buy_and_hold`.

        Returns:
            (benchmark_equity, benchmark_metrics, benchmark_name) tuple.
        """
        custom = config.benchmark if config is not None else None
        btype = config.benchmark_type if config is not None else "auto"
        bname = config.benchmark_name if config is not None else "Buy & Hold"

        if custom is not None:
            # --- Determine benchmark type ---
            if btype == "auto":
                # Heuristic: all positive and minimum > 1.0 → prices
                if (custom > 0).all() and custom.min() > 1.0:
                    detected = "prices"
                else:
                    detected = "returns"
                warnings.warn(
                    f"benchmark_type='auto': detected as '{detected}'. "
                    "Consider specifying benchmark_type explicitly.",
                    stacklevel=2,
                )
                btype = detected

            # --- Convert to equity curve ---
            if btype == "prices":
                if custom.iloc[0] != 0:
                    benchmark_equity = custom / custom.iloc[0] * self.initial_capital
                else:
                    benchmark_equity = pd.Series(
                        self.initial_capital, index=custom.index
                    )
            else:
                # returns
                benchmark_equity = (1 + custom).cumprod() * self.initial_capital

            # --- Align to data index ---
            benchmark_equity = benchmark_equity.reindex(data.index).ffill()

            # Warn if >5% missing after alignment
            n_missing = int(benchmark_equity.isna().sum())
            if n_missing > 0:
                pct_missing = n_missing / len(data.index)
                if pct_missing > 0.05:
                    warnings.warn(
                        f"Custom benchmark has {pct_missing:.1%} missing values "
                        f"after alignment ({n_missing}/{len(data.index)} bars). "
                        "Results may be unreliable.",
                        stacklevel=2,
                    )
                # Fill any remaining leading NaN with the initial capital
                benchmark_equity = benchmark_equity.bfill()

        else:
            # Default: buy-and-hold
            benchmark_equity = compute_buy_and_hold(data, self.initial_capital)
            bname = "Buy & Hold"

        bh_returns = benchmark_equity.pct_change().fillna(0)
        # Benchmark annualisation: use the first symbol's trading_days
        # when we have a per-symbol map (otherwise the buy-and-hold of
        # AAPL inside an AAPL+BTC portfolio would get annualised with
        # 365 instead of 252, inflating benchmark Sharpe by √(365/252)).
        benchmark_td = self._portfolio_trading_days
        if config is not None and config.symbols and self._resolved_per_asset_td:
            first_sym = config.symbols[0]
            benchmark_td = self._resolved_per_asset_td.get(
                first_sym, self._portfolio_trading_days)
        benchmark_metrics = self._calculate_metrics(
            bh_returns, benchmark_equity, trades=[],
            trading_days=benchmark_td,
        )
        return benchmark_equity, benchmark_metrics, bname

    def __repr__(self):
        return (f"BacktestEngine(mode={self.mode.value}, "
                f"capital={self.initial_capital:,.0f}, "
                f"fee_model={self.fee_model.name})")


# ========== Convenience Functions ==========

def backtest(
    data: pd.DataFrame,
    strategy=None,
    signals: pd.Series = None,
    initial_capital: float = 100000,
    fee_model: Union[BaseFeeModel, str] = None,
    mode: str = "vectorized",
    stop_loss: float = None,
    benchmark: Optional[pd.Series] = None,
    benchmark_type: Optional[str] = None,
    benchmark_name: Optional[str] = None,
    **kwargs
) -> BacktestResult:
    """
    Convenience backtest function.

    Parameters:
        data: OHLCV data.
        strategy: Strategy object.
        signals: Signal series (mutually exclusive with strategy).
        initial_capital: Starting capital.
        fee_model: Fee model.
        mode: Execution mode.
        stop_loss: Stop loss percentage.
        benchmark: Custom benchmark series (prices or returns).
        benchmark_type: Benchmark type — ``"prices"``, ``"returns"``,
            or ``"auto"``.
        benchmark_name: Display name for the benchmark in results.
        **kwargs: Additional engine parameters.

    Returns:
        BacktestResult: Backtest results.

    Example:
        >>> result = backtest(data, strategy=MAStrategy())
        >>> result = backtest(data, signals=my_signals, fee_model='china')
    """
    engine = BacktestEngine(
        initial_capital=initial_capital,
        mode=mode,
        stop_loss=stop_loss,
        **kwargs
    )

    if fee_model:
        engine.set_fee_model(fee_model)

    if benchmark_name is not None:
        engine._config_benchmark_name = benchmark_name

    if strategy:
        engine.set_strategy(strategy)
    elif signals is not None:
        engine.set_signals(signals)
    else:
        raise ValueError("Must provide either strategy or signals")

    return engine.run(data, benchmark=benchmark, benchmark_type=benchmark_type)
