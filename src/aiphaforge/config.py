"""
Backtest Configuration

Dataclass bundling all configuration needed by the vectorized and
event-driven execution cores.
"""

from dataclasses import dataclass, field
from datetime import time
from typing import Any, Dict, List, Optional

import pandas as pd

from .broker import FillModel
from .costs import BaseTradeCost, DefaultTradeCost
from .exit_rules import BaseExitRule
from .fees import BaseFeeModel, SimpleFeeModel
from .hooks import BacktestHook
from .position_sizing import BasePositionSizer, FractionSizer


@dataclass
class BacktestConfig:
    """Configuration bundle passed to execution cores.

    Groups the ~15 configuration fields that both the vectorized and
    event-driven cores need, avoiding long parameter lists.
    """

    initial_capital: float = 100000.0
    fee_model: BaseFeeModel = field(default_factory=SimpleFeeModel)
    allow_short: bool = True
    fee_allocation: str = "proportional"
    fill_model: FillModel = FillModel.NEXT_BAR_OPEN
    stop_loss_rule: Optional[BaseExitRule] = None
    take_profit_rule: Optional[BaseExitRule] = None
    trade_cost: BaseTradeCost = field(default_factory=DefaultTradeCost)
    position_sizer: BasePositionSizer = field(
        default_factory=lambda: FractionSizer(0.95)
    )
    risk_manager: Any = None
    hooks: List[BacktestHook] = field(default_factory=list)
    include_benchmark: bool = True
    data_validation: str = "warn"
    max_position_size: float = 1.0
    session_end_time: Optional[time] = None
    immediate_fill_price: str = "close"
    mode: str = "event_driven"
    has_signals: bool = False
    has_strategy: bool = False
    benchmark: Optional[pd.Series] = None
    benchmark_type: str = "auto"
    benchmark_name: str = "Buy & Hold"
    # Multi-asset fields (v0.7)
    asset_fee_models: Dict[str, BaseFeeModel] = field(default_factory=dict)
    asset_fill_models: Dict[str, FillModel] = field(default_factory=dict)
    capital_allocator: Any = None  # BaseCapitalAllocator; typed Any to avoid cycle
    symbols: List[str] = field(default_factory=list)
    # Margin (v0.8)
    margin_config: Any = None  # MarginConfig; typed Any to avoid cycle
    asset_margin_configs: Dict = field(default_factory=dict)
    periodic_cost_model: Any = None  # PeriodicCostModel
    portfolio_exit_rules: List = field(default_factory=list)
    # Lot sizes (v0.8)
    lot_size: int = 1  # default 1 = fractional allowed
    asset_lot_sizes: Dict[str, int] = field(default_factory=dict)
    # Per-asset position limits (v0.8) — fraction of equity
    max_position_pct: float = 1.0  # 1.0 = no per-asset limit
    asset_max_position_pcts: Dict[str, float] = field(default_factory=dict)
    # Signal (v0.9)
    signal_transform: Optional[Any] = None  # Callable[[float], float]
    is_weight_mode: bool = False
    # Turnover (v0.9.1)
    turnover_config: Any = None  # TurnoverConfig
    # Risk rules (v1.1)
    risk_rules: Any = None  # CompositeRiskManager


@dataclass
class TurnoverConfig:
    """Turnover constraint configuration.

    Turnover is TWO-SIDED: buys + sells both counted.
    A round-trip = 2x one-sided turnover. More conservative than
    industry-standard one-sided reporting.
    """
    max_turnover_per_bar: float = 0.2  # fraction of equity


def resolve_config(default: Any, overrides: Dict, symbol: str) -> Any:
    """Return per-symbol override if present, else default."""
    return overrides.get(symbol, default)
