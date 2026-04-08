"""
Backtest Configuration

Dataclass bundling all configuration needed by the vectorized and
event-driven execution cores.
"""

from dataclasses import dataclass, field
from datetime import time
from typing import Any, List, Optional

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
