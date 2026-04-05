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

# Main engine
from .engine import (
    BacktestEngine,
    ExecutionMode,
    PositionSizing,
    backtest
)

# Fee models
from .fees import (
    BaseFeeModel,
    SimpleFeeModel,
    USStockFeeModel,
    ChinaAShareFeeModel,
    CryptoSpotFeeModel,
    CryptoFuturesFeeModel,
    ZeroFeeModel,
    MarketType,
    get_fee_model
)

# Order management
from .orders import (
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    OrderManager
)

# Portfolio management
from .portfolio import (
    Position,
    Portfolio
)

# Broker simulation
from .broker import (
    Broker,
    SimpleBroker,
    FillModel,
    SlippageModel
)

# Hook framework
from .hooks import (
    HookContext,
    BacktestHook,
)

# Result data structures
from .results import (
    Trade,
    BacktestResult,
    PositionSnapshot,
    EquityPoint,
    trades_to_dataframe
)

# Performance analysis
from .performance import (
    PerformanceAnalyzer,
    analyze,
    compare_strategies
)

__version__ = '0.1.0'

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

    # Hooks
    'HookContext',
    'BacktestHook',

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
