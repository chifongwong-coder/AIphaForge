# AIphaForge

A high-performance backtest engine designed for AI agent-driven quantitative trading systems.

## Overview

AIphaForge is purpose-built for backtesting trading strategies controlled by AI agents (LLM-based meta-controllers). Unlike traditional backtest frameworks that assume deterministic strategies, this engine provides the infrastructure needed to evaluate agent-based decision systems where:

- **Decisions are non-deterministic** — the same market state may produce different agent outputs
- **Decisions have latency** — LLM inference takes seconds to minutes, not milliseconds
- **Knowledge leakage is a risk** — LLMs may "remember" historical events from training data

AIphaForge also works perfectly well as a general-purpose backtest framework for traditional rule-based and ML strategies.

```python
from aiphaforge import BacktestEngine

# A backtest in five lines — single-asset or multi-asset, same API.
engine = BacktestEngine(initial_capital=100_000, fee_model="us", stop_loss=0.05)
engine.set_signals(signals)        # pd.Series in [-1, 1]; 0=flat, NaN=hold
result = engine.run(data)           # pd.DataFrame OR Dict[str, pd.DataFrame]
print(result.summary())             # Sharpe, drawdown, win rate, …
```

More patterns (AI agent with MetaController, portfolio rebalancing,
bootstrap CIs, market-impact capacity, Bayesian optimization) live
under [Quick Start](#quick-start).

## Features

### Core Engine
- **Dual execution modes**: Vectorized (fast, for parameter sweeps) and Event-Driven (precise, bar-by-bar simulation) — the engine warns when vectorized mode is given config it doesn't enforce, so what runs matches what you wrote
- **Unified multi-asset**: single-asset and multi-asset share one code path. Pass a `pd.DataFrame` or a `Dict[str, pd.DataFrame]`
- **Realistic order simulation**: market, limit, stop, stop-limit, and trailing stop orders with configurable fill and slippage models
- **Time-in-force support**: GTC, IOC, FOK, and DAY with session-aware DAY semantics
- **Continuous signals**: fractional signals in [-1, 1]; `0 = flat`, `NaN = hold`, with optional `signal_transform`
- **Target-weight rebalancing**: `set_target_weights()` for institutional portfolio workflows

### Multi-Asset
- **Shared capital pool** (event-driven) or **weighted split** (vectorized)
- **Capital allocators**: EqualWeight, FixedWeight, ProRata, Margin — or build your own via `BaseCapitalAllocator`
- **Per-asset overrides**: Fee models, fill models, margin configs, lot sizes, and position limits per symbol
- **Per-asset PnL attribution**: Gross PnL time series, correlation matrix, per-asset Sharpe

### Margin & Leverage
- **Unified margin mode**: `initial_margin_ratio=1.0` is cash-only, `0.5` is 2x leverage, `0.1` is 10x
- **Margin calls**: Portfolio-level `MarginCallExitRule` with forced liquidation
- **Periodic costs**: `BorrowingCostModel` (entry-based for longs, market-value for shorts), `FundingRateModel` (perpetual futures)

### AI Agent Integration
- **Hook framework**: `on_pre_signal` / `on_bar` / `on_backtest_start` / `on_backtest_end` callbacks with a `LifecycleContext`, full broker and portfolio access, and exception-safe cleanup (end-hooks fire even on engine error)
- **MetaController**: Agent dynamically adjusts strategy, risk, sizing, and target weights mid-backtest via `ctx.meta`
- **Strategy composition tree**: `WeightedBlend`, `SelectBest`, `PriorityCascade`, `VoteEnsemble`, `ConditionalSwitch` — composable strategy nodes that work with MetaController
- **Latency simulation**: `LatencyHook` models LLM inference delay with decision/execution delay separation — decision latency applies to both orders and MetaController operations, per-symbol execution latency is additive
- **Dynamic universe selection**: `add_to_universe()` / `remove_from_universe()` / `set_universe()` — agent decides what to trade at runtime, with automatic position closing on removal
- **Multi-timeframe**: `secondary_data` for daily trend analysis while executing on minute bars
- **Scheduled rebalancing**: `ScheduleHook` for periodic callbacks (daily/weekly/monthly/quarterly/N-bar)
- **Rebalancing hooks**: `DriftRebalanceHook` (threshold-based), `BandRebalanceHook` (per-asset band), `CostAwareRebalanceHook` (turnover vs. cost) — all support static or dynamic weights
- **Portfolio optimization**: `OptimizedRebalanceHook` with pluggable optimizers — Equal Weight, Inverse Volatility, Mean-Variance, Risk Parity, Minimum Variance. Integrates with dynamic universe selection

### Technical Indicators & Strategies
- **25 indicators**: SMA, EMA, WMA, DEMA, TEMA, MACD, ADX, Parabolic SAR, Supertrend, Ichimoku, RSI, ROC, Stochastic, CCI, Williams %R, MFI, StochRSI, Bollinger Bands, ATR, Keltner, Donchian, VWAP, OBV, A/D Line, CMF
- **19 strategy templates**: 13 leaf strategies (MA Crossover, MACD, RSI Mean Reversion, Bollinger, Supertrend, etc.) + 6 composite strategy nodes
- **One-line backtest**: `MACrossover(short=10, long=30).backtest(data, fee_model='china')`

### Risk Management
- **Exit rules**: percentage stop-loss, take-profit, and trailing stop. All produce per-trade records (with `reason='stop_loss'` / `'take_profit'` / `'trailing_stop_exit'`) in both execution modes
- **Composable risk rules**: `CompositeRiskManager` with `MaxDrawdownHalt`, `ExposureLimit`, `DailyLossLimit`, `ConcentrationLimit`
- **Agent-controlled risk**: MetaController adjusts stop-loss, take-profit, sizing, and signals per bar

### Parameter Optimization
- **Grid search**: `optimize()` with walk-forward validation
- **Bayesian optimization**: `optimize_bayesian()` via Optuna with automatic train/test split, constraint support, and trial caching (optional dependency)

### Statistical Significance Testing
- **Bootstrap CI**: `bootstrap_ci()` / `bootstrap_metrics()` — stationary block bootstrap (Politis-Romano) for Sharpe, drawdown, and custom metrics
- **Permutation test**: `permutation_test()` — shuffle signal timing to test alpha significance (Phipson-Smyth corrected p-values)
- **PSR / DSR**: `probabilistic_sharpe_ratio()` and `deflated_sharpe_ratio()` — Bailey & López de Prado significance tests with Pearson kurtosis adjustment
- **Monte Carlo simulation**: `monte_carlo_test()` — generate synthetic market paths, run strategy/agent on each to test robustness
- **Multiple comparison correction**: `multiple_comparison_correction()` — Bonferroni, Benjamini-Hochberg, or Model Confidence Set (optional `arch` dependency)
- **Path generation**: `generate_paths()` — block bootstrap or parametric normal synthetic OHLCV data

### Market Impact & Capacity
- **Market impact models**: `LinearImpactModel`, `SquareRootImpactModel` (Almgren-Chriss with permanent impact), `PowerLawImpactModel` — pluggable via `BaseImpactModel` ABC
- **Strategy capacity estimation**: `estimate_capacity()` scales trade sizes, computes impact drag, uses bisection to find max capital before Sharpe degrades
- **Volatility & liquidity tools**: Parkinson high-low volatility, Corwin-Schultz spread estimator, rolling ADV — all from OHLCV data, no order book needed
- **Calibration presets**: `suggested_impact_params()` for US large/small cap, China A-shares, crypto spot/futures

### Costs & Fees
- **Multi-market presets**: US stocks, China A-shares, crypto spot, crypto futures — `get_fee_model("china")`
- **Slippage models**: Fixed, volume-based, volatility-based
- **Lot sizes**: Per-asset minimum trade units (e.g., A-share 100-share lots)
- **Corporate actions**: `CorporateActionHook` for dividends and stock splits

### Performance Analysis
- **30+ metrics**: Sharpe, Sortino, Calmar, max drawdown, VaR, CVaR, profit factor, win rate, and more
- **Per-symbol annualization**: `trading_days=` accepts a scalar (252 / 365) or a per-symbol dict; mixed-asset portfolios (e.g. AAPL + BTC-USD) annualise per-asset metrics correctly
- **Per-asset attribution**: `BacktestResult.per_asset_metrics` on every multi-asset run, with correlation matrix
- **Breakdowns**: monthly / yearly return tables, multi-strategy comparison
- **Benchmark overlay**: custom series or automatic buy-and-hold

## Quick Start

### Strategy One-Line Backtest

```python
from aiphaforge.strategies import MACrossover

result = MACrossover(short=10, long=30).backtest(data, fee_model='china')
print(result.summary())
```

### Signal-Based Backtest

```python
from aiphaforge import BacktestEngine

engine = BacktestEngine(
    fee_model='crypto',
    initial_capital=100000,
    stop_loss=0.05,
)
engine.set_signals(signals)  # pd.Series: 1=buy, -1=sell, 0=flat, NaN=hold
result = engine.run(data)
```

### AI Agent with MetaController

```python
from aiphaforge import BacktestEngine, BacktestHook
from aiphaforge.strategies import WeightedBlend, MACrossover, RSIMeanReversion

class AdaptiveAgent(BacktestHook):
    def on_pre_signal(self, ctx):
        if ctx.meta:
            vol = ctx.data['close'].pct_change().std()
            if vol > 0.03:
                ctx.meta.set_weights([0.3, 0.7])  # favor mean reversion
            else:
                ctx.meta.set_weights([0.7, 0.3])  # favor trend

tree = WeightedBlend(
    children=[MACrossover(), RSIMeanReversion()],
    weights=[0.5, 0.5],
)

engine = BacktestEngine(
    mode='event_driven',
    hooks=[AdaptiveAgent()],
)
engine.set_strategy(tree)
result = engine.run(data)
```

### Monthly Rebalancing

```python
from aiphaforge import BacktestEngine
from aiphaforge.hooks import schedule_rebalance

engine = BacktestEngine(
    mode='event_driven',
    hooks=[schedule_rebalance({"AAPL": 0.5, "TSLA": 0.5}, "monthly")],
)
result = engine.run({"AAPL": aapl_df, "TSLA": tsla_df})
```

### Statistical Validation

```python
from aiphaforge.significance import bootstrap_ci, permutation_test, monte_carlo_test

# Confidence interval on Sharpe ratio
ci = bootstrap_ci(result, metric="sharpe_ratio", confidence=0.95)
print(f"Sharpe: {ci.observed:.2f} [{ci.ci_lower:.2f}, {ci.ci_upper:.2f}]")

# Is the strategy's alpha significant?
perm = permutation_test(data, strategy=MACrossover(), n_permutations=1000)
print(f"p-value: {perm.p_value:.4f}")

# Monte Carlo robustness (agent re-executes on synthetic paths)
mc = monte_carlo_test(data, strategy=tree, hooks=[AdaptiveAgent()], n_paths=500)
print(f"MC Sharpe: {mc.mean:.2f} ± {mc.std:.2f}, worst: {mc.worst_case:.2f}")
```

### Bayesian Parameter Optimization

```python
from aiphaforge.optimizer import optimize_bayesian

result = optimize_bayesian(
    data,
    param_ranges={'short': (5, 30), 'long': (20, 80), 'ma_type': ['sma', 'ema']},
    strategy_factory=lambda p: MACrossover(**p),
    n_trials=50,
    train_pct=0.7,  # automatic overfitting protection
)
print(f"Best: {result.best_params}")
print(f"In-sample Sharpe:  {result.in_sample_result.sharpe_ratio:.2f}")
print(f"Out-of-sample:     {result.out_of_sample_result.sharpe_ratio:.2f}")
```

### Dynamic Universe Selection

```python
from aiphaforge import BacktestEngine, BacktestHook

class UniverseRotator(BacktestHook):
    def on_pre_signal(self, ctx):
        if ctx.meta and ctx.bar_index % 20 == 0:
            # Rotate: keep top 3 by recent momentum
            momentum = {}
            for sym in ctx.meta._all_symbols:
                df = ctx.all_data.get(sym)
                if df is not None and len(df) > 20:
                    momentum[sym] = df['close'].iloc[-1] / df['close'].iloc[-20] - 1
            top3 = sorted(momentum, key=momentum.get, reverse=True)[:3]
            ctx.meta.set_universe(top3)

engine = BacktestEngine(mode='event_driven', hooks=[UniverseRotator()],
                         initial_universe=["AAPL", "TSLA"])
result = engine.run(data_dict)  # data_dict has 10+ symbols
```

### Market Impact Estimation

```python
from aiphaforge import BacktestEngine
from aiphaforge.market_impact import SquareRootImpactModel, estimate_capacity

# Backtest with realistic market impact
engine = BacktestEngine(
    impact_model=SquareRootImpactModel(eta=0.5, gamma=0.1),
    fee_model='us',
)
engine.set_strategy(strategy)
result = engine.run(data)

# Estimate how much capital this strategy can handle
capacity = estimate_capacity(result, data, min_sharpe=1.0)
print(f"Max capacity: ${capacity.estimated_capacity:,.0f}")
```

## Installation

```bash
pip install aiphaforge
```

Optional dependencies:

```bash
pip install aiphaforge[plot]          # matplotlib for visualization
pip install aiphaforge[data]          # yfinance for data loading
pip install aiphaforge[optimize]      # optuna for Bayesian optimization
pip install aiphaforge[significance]  # scipy (PSR / DSR) + arch (Model Confidence Set)
pip install aiphaforge[portfolio]     # scipy for portfolio optimization
pip install aiphaforge[all]           # everything
```

### Requirements

- Python >= 3.10
- pandas >= 1.5
- numpy >= 1.23

## Fee Models

| Model | Market | Key Features |
|-------|--------|-------------|
| `USStockFeeModel` | US Equities | Per-share commission, minimum fee |
| `ChinaAShareFeeModel` | China A-Shares | Commission + stamp duty (sell) + transfer fee |
| `CryptoSpotFeeModel` | Crypto Spot | Maker/taker fee rates |
| `CryptoFuturesFeeModel` | Crypto Futures | Maker/taker + funding rate |
| `SimpleFeeModel` | Generic | Flat commission rate |
| `ZeroFeeModel` | Testing | No fees |

## Testing

```bash
pytest tests/ -v
```

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE) — you are free to use, modify, and distribute this software, but any derivative work must also be distributed under the same license.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. By contributing, you agree that your contributions will be licensed under the GPL v3.
