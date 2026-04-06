# AIphaForge

A high-performance backtest engine designed for AI agent-driven quantitative trading systems.

## Overview

AIphaForge is purpose-built for backtesting trading strategies controlled by AI agents (LLM-based meta-controllers). Unlike traditional backtest frameworks that assume deterministic strategies, this engine provides the infrastructure needed to evaluate agent-based decision systems where:

- **Decisions are non-deterministic** — the same market state may produce different agent outputs
- **Decisions have latency** — LLM inference takes seconds to minutes, not milliseconds
- **Knowledge leakage is a risk** — LLMs may "remember" historical events from training data

AIphaForge also works perfectly well as a general-purpose backtest framework for traditional rule-based and ML strategies.

## Features

### Core Engine
- **Dual execution modes**: Vectorized (fast, for parameter sweeps) and Event-Driven (precise, bar-by-bar simulation)
- **Realistic order simulation**: Market, limit, stop, and stop-limit orders with configurable fill and slippage models
- **Multi-market fee models**: US stocks, China A-shares, crypto spot, and crypto futures with accurate fee structures
- **Portfolio management**: Multi-asset position tracking, cash management, equity curve recording

### Extensibility
- **Hook framework**: Plug in custom logic at every bar — agent triggers, monitors, custom risk rules
- **Strategy interface**: Any object with a `generate_signals()` method works as a strategy
- **Risk manager interface**: Inject your own risk management system via constructor
- **Agent integration interface**: Built-in hooks for connecting LLM-based decision agents
- **Fee model extensibility**: Subclass `BaseFeeModel` to support any market

### Performance Analysis
- 30+ metrics: Sharpe, Sortino, Calmar, max drawdown, VaR, CVaR, profit factor, and more
- Monthly/yearly return breakdowns
- Multi-strategy comparison
- Buy-and-hold benchmark

## Quick Start

```python
from aiphaforge import BacktestEngine, backtest

# Option 1: Using the convenience function
result = backtest(
    data,                          # pd.DataFrame with OHLCV columns
    signals=my_signals,            # pd.Series of {1, -1, 0}
    fee_model='crypto',
    initial_capital=100000,
    stop_loss=0.05
)
print(result.summary())

# Option 2: Using the engine directly
engine = BacktestEngine(
    fee_model='china',
    initial_capital=500000,
    mode='event_driven',
    stop_loss=0.05,
    take_profit=0.15
)
engine.set_strategy(my_strategy)
result = engine.run(data)
```

### Using with an AI Agent

```python
from aiphaforge import BacktestEngine, BacktestHook, HookContext

class AgentHook(BacktestHook):
    def __init__(self, agent, trigger_interval=20):
        self.agent = agent
        self.trigger_interval = trigger_interval

    def on_bar(self, ctx: HookContext):
        if ctx.bar_index % self.trigger_interval == 0:
            decision = self.agent.analyze(ctx.data, ctx.portfolio)
            # Apply agent decision to strategy parameters...
        return None

engine = BacktestEngine(
    mode='event_driven',
    hooks=[AgentHook(my_agent, trigger_interval=20)]
)
```

## Installation

```bash
pip install aiphaforge
```

### Requirements

- Python >= 3.10
- pandas
- numpy

## Project Structure

```
src/aiphaforge/
├── engine.py          # Main backtest engine (vectorized + event-driven)
├── broker.py          # Order execution and fill simulation
├── portfolio.py       # Position and cash tracking
├── orders.py          # Order types and lifecycle management
├── fees.py            # Multi-market fee models
├── hooks.py           # Hook framework for extensibility
├── results.py         # Result data structures
├── performance.py     # Performance analysis and reporting
└── utils.py           # Common utilities and financial calculations
```

## Fee Models

| Model | Market | Key Features |
|-------|--------|-------------|
| `USStockFeeModel` | US Equities | Per-share commission, minimum fee |
| `ChinaAShareFeeModel` | China A-Shares | Commission + stamp duty (sell) + transfer fee |
| `CryptoSpotFeeModel` | Crypto Spot | Maker/taker fee rates |
| `CryptoFuturesFeeModel` | Crypto Futures | Maker/taker + funding rate |
| `SimpleFeeModel` | Generic | Flat commission rate |
| `ZeroFeeModel` | Testing | No fees |

```python
from aiphaforge import get_fee_model

fee_model = get_fee_model('china')
fee_model = get_fee_model('crypto', maker_fee=0.0008, taker_fee=0.001)
```

## Execution Modes

### Vectorized Mode
Fast NumPy/Pandas vectorized computation. Best for parameter optimization and quick validation.

```python
engine = BacktestEngine(mode='vectorized')
```

### Event-Driven Mode
Bar-by-bar simulation with full order lifecycle. Supports complex order types, realistic fills, and hook-based extensions.

```python
engine = BacktestEngine(mode='event_driven', fill_model=FillModel.NEXT_BAR_OPEN)
```

## Performance Analysis

```python
from aiphaforge import PerformanceAnalyzer, compare_strategies

analyzer = PerformanceAnalyzer(result)
print(analyzer.summary())

# Compare multiple strategies
comparison = compare_strategies({
    'Momentum': momentum_result,
    'MeanRevert': mean_revert_result,
})
print(comparison)
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
