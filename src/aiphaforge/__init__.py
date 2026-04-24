"""
AIphaForge
==========

A high-performance backtest engine for AI agent-driven quantitative trading.

Contents:
---------
- BacktestEngine: Backtest engine supporting both vectorized and event-driven modes
- Fee models: Support for US stocks, China A-shares, crypto, and more
- Performance analysis: Comprehensive metrics calculation and report generation

Quick Start:
------------
>>> from aiphaforge import BacktestEngine, ChinaAShareFeeModel
>>> engine = BacktestEngine(
...     fee_model=ChinaAShareFeeModel(),
...     initial_capital=100000,
...     stop_loss=0.05
... )
>>> engine.set_strategy(my_strategy)
>>> results = engine.run(data)
>>> print(results.summary())

Using ML model signals:
-----------------------
>>> from aiphaforge import backtest
>>> signals = model.predict(features)  # pd.Series {1, -1, 0}
>>> results = backtest(data, signals=signals, fee_model='china')

Fee Models:
-----------
>>> from aiphaforge import (
...     SimpleFeeModel,
...     USStockFeeModel,
...     ChinaAShareFeeModel,
...     CryptoSpotFeeModel,
...     CryptoFuturesFeeModel,
...     get_fee_model
... )
>>> fee_model = get_fee_model('china')
>>> fee_model = get_fee_model('crypto', maker_fee=0.0008)

Performance Analysis:
--------------------
>>> from aiphaforge import PerformanceAnalyzer, analyze, compare_strategies
>>> analyzer = PerformanceAnalyzer(results)
>>> print(analyzer.summary())
>>> report = analyzer.generate_report()
>>> comparison = compare_strategies({
...     'MA': ma_results,
...     'RSI': rsi_results
... })
"""

from .broker import Broker, FillModel, SimpleBroker, SlippageModel
from .capital_allocator import (
    BaseCapitalAllocator,
    EqualWeightAllocator,
    FixedWeightAllocator,
    MarginAllocator,
    ProRataAllocator,
)
from .config import BacktestConfig, TurnoverConfig, resolve_config
from .corporate_actions import CorporateActionHook
from .costs import BaseTradeCost, DefaultTradeCost
from .engine import BacktestEngine, ExecutionMode, PositionSizing, backtest
from .exit_rules import BaseExitRule, PercentageStopLoss, PercentageTakeProfit, TrailingStopLoss
from .fees import (
    BaseFeeModel,
    ChinaAShareFeeModel,
    CryptoFuturesFeeModel,
    CryptoSpotFeeModel,
    MarketType,
    SimpleFeeModel,
    USStockFeeModel,
    ZeroFeeModel,
    get_fee_model,
)
from .hooks import (
    BacktestHook,
    BandRebalanceHook,
    CostAwareRebalanceHook,
    DriftRebalanceHook,
    HookContext,
    LifecycleContext,
    OptimizedRebalanceHook,
    ScheduleHook,
    SecondaryTimeframe,
    schedule_rebalance,
)
from .latency import LatencyHook, SimpleLatencyHook, SymbolRoutingLatencyHook
from .margin import (
    BasePortfolioExitRule,
    BorrowingCostModel,
    FundingRateModel,
    MarginCallExitRule,
    MarginConfig,
    PeriodicCostModel,
)
from .market_impact import (
    BaseImpactModel,
    CapacityResult,
    LinearImpactModel,
    PowerLawImpactModel,
    SquareRootImpactModel,
    corwin_schultz_spread,
    estimate_capacity,
    suggested_impact_params,
)
from .meta import MetaContext
from .optimizer import BayesianResult, optimize, optimize_bayesian, walk_forward
from .orders import Order, OrderManager, OrderSide, OrderStatus, OrderType
from .performance import PerformanceAnalyzer, analyze, compare_strategies
from .portfolio import Portfolio, Position
from .portfolio_optimizer import (
    BasePortfolioOptimizer,
    EqualWeightOptimizer,
    InverseVolatilityOptimizer,
    MeanVarianceOptimizer,
    MinimumVarianceOptimizer,
    RiskParityOptimizer,
)
from .position_sizing import AllInSizer, BasePositionSizer, FixedSizer, FractionSizer
from .results import BacktestResult, EquityPoint, PositionSnapshot, Trade, trades_to_dataframe
from .risk import (
    BaseRiskManager,
    BaseRiskRule,
    CompositeRiskManager,
    ConcentrationLimit,
    DailyLossLimit,
    ExposureLimit,
    MaxDrawdownHalt,
    RiskSignal,
)
from .significance import (
    BootstrapResult,
    CorrectionResult,
    DSRResult,
    MonteCarloResult,
    PermutationResult,
    PSRResult,
    bootstrap_ci,
    bootstrap_metrics,
    build_returns_matrix,
    deflated_sharpe_ratio,
    generate_paths,
    monte_carlo_test,
    multiple_comparison_correction,
    permutation_test,
    probabilistic_sharpe_ratio,
)
from .strategies import (
    ConditionalSwitch,
    PriorityCascade,
    SelectBest,
    StrategyNode,
    VoteEnsemble,
    WeightedBlend,
)

__version__ = '2.0.0'

__all__ = [
    # Main engine
    'BacktestEngine',
    'ExecutionMode',
    'PositionSizing',
    'backtest',

    # Fee models
    'BaseFeeModel',
    'SimpleFeeModel',
    'USStockFeeModel',
    'ChinaAShareFeeModel',
    'CryptoSpotFeeModel',
    'CryptoFuturesFeeModel',
    'ZeroFeeModel',
    'MarketType',
    'get_fee_model',

    # Orders
    'Order',
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'OrderManager',

    # Portfolio
    'Position',
    'Portfolio',

    # Broker
    'Broker',
    'SimpleBroker',
    'FillModel',
    'SlippageModel',

    # Exit rules
    'BaseExitRule',
    'PercentageStopLoss',
    'PercentageTakeProfit',
    'TrailingStopLoss',

    # Trade costs
    'BaseTradeCost',
    'DefaultTradeCost',

    # Position sizing
    'BasePositionSizer',
    'FractionSizer',
    'FixedSizer',
    'AllInSizer',

    # Hooks
    'HookContext',
    'LifecycleContext',
    'BacktestHook',
    'SecondaryTimeframe',
    'ScheduleHook',
    'schedule_rebalance',
    'DriftRebalanceHook',
    'BandRebalanceHook',
    'CostAwareRebalanceHook',
    'OptimizedRebalanceHook',
    'LatencyHook',
    'SimpleLatencyHook',
    'SymbolRoutingLatencyHook',

    # MetaController (v1.2)
    'MetaContext',

    # Config
    'BacktestConfig',
    'resolve_config',

    # Capital allocation
    'BaseCapitalAllocator',
    'EqualWeightAllocator',
    'FixedWeightAllocator',
    'MarginAllocator',
    'ProRataAllocator',

    # Margin
    'MarginConfig',
    'MarginCallExitRule',
    'BasePortfolioExitRule',
    'PeriodicCostModel',
    'BorrowingCostModel',
    'FundingRateModel',
    'CorporateActionHook',
    'TurnoverConfig',
    'optimize',
    'walk_forward',
    'optimize_bayesian',
    'BayesianResult',

    # Risk management
    'BaseRiskManager',
    'BaseRiskRule',
    'CompositeRiskManager',
    'MaxDrawdownHalt',
    'ExposureLimit',
    'DailyLossLimit',
    'ConcentrationLimit',
    'RiskSignal',

    # Strategy composition tree (v1.4)
    'StrategyNode',
    'WeightedBlend',
    'SelectBest',
    'PriorityCascade',
    'VoteEnsemble',
    'ConditionalSwitch',

    # Significance testing (v1.5)
    'BootstrapResult',
    'PermutationResult',
    'bootstrap_ci',
    'bootstrap_metrics',
    'permutation_test',

    # Monte Carlo & Multiple Comparison Correction (v1.6)
    'MonteCarloResult',
    'CorrectionResult',
    'generate_paths',
    'monte_carlo_test',
    'multiple_comparison_correction',
    'build_returns_matrix',

    # PSR / DSR (v1.9.5)
    'PSRResult',
    'DSRResult',
    'probabilistic_sharpe_ratio',
    'deflated_sharpe_ratio',

    # Market impact (v1.9.4)
    'BaseImpactModel',
    'LinearImpactModel',
    'SquareRootImpactModel',
    'PowerLawImpactModel',
    'estimate_capacity',
    'CapacityResult',
    'suggested_impact_params',
    'corwin_schultz_spread',

    # Portfolio optimization
    'BasePortfolioOptimizer',
    'EqualWeightOptimizer',
    'InverseVolatilityOptimizer',
    'MeanVarianceOptimizer',
    'RiskParityOptimizer',
    'MinimumVarianceOptimizer',

    # Results
    'Trade',
    'BacktestResult',
    'PositionSnapshot',
    'EquityPoint',
    'trades_to_dataframe',

    # Analysis
    'PerformanceAnalyzer',
    'analyze',
    'compare_strategies',
]
