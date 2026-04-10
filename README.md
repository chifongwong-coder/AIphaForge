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
- **Unified multi-asset**: Single-asset and multi-asset backtests share one code path. Pass a `pd.DataFrame` or a `Dict[str, pd.DataFrame]`
- **Realistic order simulation**: Market, limit, stop, and stop-limit orders with configurable fill and slippage models
- **Time-in-force support**: GTC, IOC, FOK, and DAY order expiration with session-aware DAY semantics
- **Continuous signals**: Fractional signals in [-1, 1] (z-scores, alpha values). `0 = flat`, `NaN = hold`, with optional `signal_transform` for custom mapping
- **Target-weight rebalancing**: `set_target_weights()` for institutional portfolio management workflows

### Multi-Asset
- **Shared capital pool** (event-driven) or **weighted split** (vectorized)
- **Capital allocators**: EqualWeight, FixedWeight, ProRata, Margin — or build your own via `BaseCapitalAllocator`
- **Per-asset overrides**: Fee models, fill models, margin configs, lot sizes, and position limits per symbol via `resolve_config()` pattern
- **Per-asset PnL attribution**: Gross PnL time series, correlation matrix, per-asset Sharpe

### Margin & Leverage
- **Unified margin mode**: `initial_margin_ratio=1.0` is cash-only, `0.5` is 2x leverage, `0.1` is 10x
- **Margin calls**: Portfolio-level `MarginCallExitRule` with forced liquidation
- **Periodic costs**: `BorrowingCostModel` (entry-based for longs, market-value for shorts), `FundingRateModel` (perpetual futures)
- **Buying power**: Equity-based formula, distinguishes open vs close orders, blocks new opens during margin call

### AI Agent Integration
- **Hook framework**: `on_pre_signal` / `on_bar` callbacks with full broker and portfolio access
- **Latency simulation**: `LatencyHook` models LLM inference delay (fixed, statistical, or custom distributions)
- **Per-symbol routing**: `SymbolRoutingLatencyHook` for different latency per asset
- **Convenience subclass**: `SimpleLatencyHook` — override `make_decision()` for quick agent prototyping
- **Multi-asset hooks**: Single-asset and multi-asset HookContext shapes, dual-path broker proxy

### Costs & Fees
- **Multi-market presets**: US stocks, China A-shares, crypto spot, crypto futures — `get_fee_model("china")`
- **Slippage models**: Fixed, volume-based, volatility-based
- **Lot sizes**: Per-asset minimum trade units (e.g., A-share 100-share lots)
- **Corporate actions**: `CorporateActionHook` for dividends and stock splits

### Performance Analysis
- 30+ metrics: Sharpe, Sortino, Calmar, max drawdown, VaR, CVaR, profit factor, and more
- Monthly/yearly return breakdowns, multi-strategy comparison
- Custom benchmark comparison (or automatic buy-and-hold)
- Per-asset analysis with correlation matrix

## Quick Start

### Single-Asset Backtest

```python
from aiphaforge import BacktestEngine, backtest

result = backtest(
    data,                          # pd.DataFrame with OHLCV columns
    signals=my_signals,            # pd.Series: 1=buy, -1=sell, 0=flat, NaN=hold
    fee_model='crypto',
    initial_capital=100000,
    stop_loss=0.05
)
print(result.summary())
```

### Multi-Asset Backtest

```python
from aiphaforge import BacktestEngine, EqualWeightAllocator

data = {
    "600519.SH": maotai_df,
    "000858.SZ": wuliangye_df,
    "601318.SH": pingan_df,
}
signals = {sym: sig_series for sym, sig_series in ...}

engine = BacktestEngine(
    fee_model='china',
    mode='event_driven',
    initial_capital=1_000_000,
    capital_allocator=EqualWeightAllocator(),
    asset_lot_sizes={"600519.SH": 100, "000858.SZ": 100, "601318.SH": 100},
)
engine.set_signals(signals)
result = engine.run(data)
print(result.per_asset_pnl)
```

### Target-Weight Rebalancing

```python
weights = {
    "2024-01-01": {"AAPL": 0.3, "TSLA": 0.2, "GOOG": 0.5},
    "2024-02-01": {"AAPL": 0.4, "TSLA": 0.1, "GOOG": 0.5},
}
engine = BacktestEngine(mode='event_driven', initial_capital=500_000)
engine.set_target_weights(weights)
result = engine.run({"AAPL": aapl_df, "TSLA": tsla_df, "GOOG": goog_df})
```

### Leveraged Trading with Margin

```python
from aiphaforge import BacktestEngine, MarginAllocator
from aiphaforge.margin import MarginConfig, BorrowingCostModel, MarginCallExitRule

engine = BacktestEngine(
    fee_model='china',
    mode='event_driven',
    initial_capital=100_000,
    margin_config=MarginConfig(
        initial_margin_ratio=0.5,        # 2x leverage
        maintenance_margin_ratio=0.65,   # A-share 130% maintenance line
        borrowing_rate=0.08,
    ),
    capital_allocator=MarginAllocator(),
    portfolio_exit_rules=[MarginCallExitRule("largest_first")],
    periodic_cost_model=BorrowingCostModel(),
)
```

### AI Agent with Latency Simulation

```python
from aiphaforge import BacktestEngine, BacktestHook, LatencyHook, HookContext

class AgentHook(BacktestHook):
    def __init__(self, agent):
        self.agent = agent

    def on_pre_signal(self, ctx: HookContext):
        decision = self.agent.analyze(ctx.data, ctx.portfolio)
        if decision.action == 'buy':
            order = ctx.broker.create_market_order(
                ctx.symbol, 'buy', decision.size, 'agent', ctx.timestamp)
            ctx.broker.submit_order(order, ctx.timestamp)

# 3-bar latency to simulate LLM inference delay
latency_hook = LatencyHook(
    inner_hook=AgentHook(my_agent),
    latency_model="fixed",
    latency_params={"bars": 3},
)

engine = BacktestEngine(mode='event_driven', hooks=[latency_hook])
engine.set_signals(signals)
result = engine.run(data)
```

### Continuous Signals with Transform

```python
import numpy as np

# Z-score signals clipped to [-1, 1]
engine = BacktestEngine(
    mode='event_driven',
    signal_transform=lambda s: np.clip(s, -1, 1),
)
engine.set_signals(zscore_signals)  # values like -2.3, 0.5, 1.8, ...
result = engine.run(data)
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
├── engine.py              # Backtest engine orchestrator
├── config.py              # BacktestConfig dataclass
├── core_vectorized.py     # Vectorized execution core
├── core_event_driven.py   # Event-driven execution core (unified single/multi-asset)
├── capital_allocator.py   # Capital allocation (EqualWeight, FixedWeight, ProRata, Margin)
├── margin.py              # Margin/leverage (MarginConfig, MarginCall, BorrowingCost, FundingRate)
├── corporate_actions.py   # Dividend and stock split handling
├── exit_rules.py          # Stop-loss / take-profit modules
├── costs.py               # Trade cost modules (vectorized mode)
├── position_sizing.py     # Position sizing modules
├── broker.py              # Order execution and fill simulation
├── portfolio.py           # Position, cash, and margin tracking
├── orders.py              # Order types and lifecycle management
├── fees.py                # Multi-market fee models
├── hooks.py               # Hook framework (HookContext with single/multi-asset shapes)
├── latency.py             # Agent latency simulation (single + multi-asset)
├── risk.py                # Risk manager ABC
├── results.py             # Result data structures
├── performance.py         # Performance analysis and reporting
└── utils.py               # Common utilities and financial calculations
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

## Performance Analysis

```python
from aiphaforge import PerformanceAnalyzer, compare_strategies

analyzer = PerformanceAnalyzer(result)
print(analyzer.summary())

# Per-asset analysis (multi-asset backtests)
per_asset = analyzer.per_asset_analysis()
corr = analyzer.correlation_matrix()

# Compare multiple strategies
comparison = compare_strategies({
    'Momentum': momentum_result,
    'MeanRevert': mean_revert_result,
})
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
