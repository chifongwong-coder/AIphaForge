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
# Broker simulation
from .broker import Broker, FillModel, SimpleBroker, SlippageModel
from .engine import BacktestEngine, ExecutionMode, PositionSizing, backtest

# Fee models
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

# Hook framework
from .hooks import (
    BacktestHook,
    HookContext,
)

# Order management
from .orders import Order, OrderManager, OrderSide, OrderStatus, OrderType

# Performance analysis
from .performance import PerformanceAnalyzer, analyze, compare_strategies

# Portfolio management
from .portfolio import Portfolio, Position

# Result data structures
from .results import BacktestResult, EquityPoint, PositionSnapshot, Trade, trades_to_dataframe

# Risk management
from .risk import (
    BaseRiskManager,
    RiskSignal,
)

__version__ = '0.3.0'

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

    # Risk management
    'BaseRiskManager',
    'RiskSignal',

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
