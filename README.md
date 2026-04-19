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
- **Realistic order simulation**: Market, limit, stop, stop-limit, and **trailing stop** orders with configurable fill and slippage models
- **Time-in-force support**: GTC, IOC, FOK, and DAY order expiration with session-aware DAY semantics
- **Continuous signals**: Fractional signals in [-1, 1] (z-scores, alpha values). `0 = flat`, `NaN = hold`, with optional `signal_transform` for custom mapping
- **Target-weight rebalancing**: `set_target_weights()` for institutional portfolio management workflows

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
- **Hook framework**: `on_pre_signal` / `on_bar` callbacks with full broker and portfolio access
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
- **Composable risk rules**: `CompositeRiskManager` with `MaxDrawdownHalt`, `ExposureLimit`, `DailyLossLimit`, `ConcentrationLimit`
- **Trailing stop loss**: `TrailingStopLoss` exit rule tracks price highs/lows and exits on pullback
- **Agent-controlled risk**: MetaController adjusts stop-loss, take-profit, sizing, and signals per bar

### Parameter Optimization
- **Grid search**: `optimize()` with walk-forward validation
- **Bayesian optimization**: `optimize_bayesian()` via Optuna with automatic train/test split, constraint support, and trial caching (optional dependency)

### Statistical Significance Testing
- **Bootstrap CI**: `bootstrap_ci()` / `bootstrap_metrics()` — stationary block bootstrap (Politis-Romano) for Sharpe, drawdown, and custom metrics
- **Permutation test**: `permutation_test()` — shuffle signal timing to test alpha significance (Phipson-Smyth corrected p-values)
- **PSR / DSR (v1.9.5)**: `probabilistic_sharpe_ratio()` and `deflated_sharpe_ratio()` — Bailey & López de Prado significance tests with Pearson kurtosis adjustment
- **Monte Carlo simulation**: `monte_carlo_test()` — generate synthetic market paths, run strategy/agent on each to test robustness
- **Multiple comparison correction**: `multiple_comparison_correction()` — Bonferroni, Benjamini-Hochberg, or Model Confidence Set (optional `arch` dependency)
- **Path generation**: `generate_paths()` — block bootstrap or parametric normal synthetic OHLCV data

### Per-Symbol Annualization (v1.9.5)
- `BacktestEngine(trading_days=...)` accepts a scalar (252 / 365 / etc.) or a per-symbol dict
- Mixed-asset portfolios (e.g. AAPL + BTC-USD) annualise per-asset metrics correctly; portfolio-level requires an explicit `portfolio_trading_days` (no silent auto-infer — a single scalar cannot be objectively chosen for stocks+crypto)
- `BacktestResult.per_asset_metrics` is now populated on every multi-asset run (previously declared but never set)

### v1.9.5 compatibility notes
- Default `trading_days=252` reproduces v1.9.4 numbers exactly.
- **Pickle compatibility across versions is not guaranteed.** `BacktestResult` gained new fields (`trading_days`, `per_asset_trading_days`); pickles created with v1.9.4 should be regenerated rather than loaded into v1.9.5. If you need long-term persistence, use `result.to_dict()` + JSON.

### v1.9.6 — Engine stability release
A focused round of correctness fixes covering 10 long-standing bugs found in an engine-wide audit. No new features; existing public APIs are preserved.

- **Hook lifecycle (B1, B2, R5)**: `on_backtest_start` and `on_backtest_end` now both fire **once per backtest** (previously start fired per-symbol while end fired once) and receive a `LifecycleContext` with `phase`, `symbols`, `data_dict`, `primary_symbol`, and `primary_data`. Pre-v1.9.6 subclasses that use the legacy `(data, symbol, *, config)` signature still work via a backward-compat adapter that emits a one-time `DeprecationWarning` per subclass. Vectorized mode now also fires the lifecycle callbacks (previously skipped entirely).
- **Vectorized cumprod safety (B6)**: `net_returns` is clipped at -1.0 and equity is frozen at 0 once bankruptcy is detected; pathological costs / tiny capital can no longer flip the cumprod sign and produce nonsensical positive equity.
- **Vectorized trade reconstruction (Q1, Q3)**: `Trade.size` from a vectorized run now means **shares** (`entry_equity * position_size / entry_price`), not signal magnitude. `Trade.pnl` no longer double-deducts commission + slippage — it is now a linear path-independent approximation that matches the geometric equity curve to machine epsilon for single-trade no-fee cases. See `Trade.__doc__` for the full discrepancy contract.
- **Risk API (B3)**: `BacktestEngine(risk_manager=CompositeRiskManager(...))` no longer crashes — `CompositeRiskManager` now inherits `BaseRiskManager`. Passing both `risk_manager=` and `risk_rules=` simultaneously raises `ValueError` instead of silently picking one.
- **Data validation (B5)**: `validate_ohlcv(level='strict')` now rejects non-finite (`inf` / `-inf`) and non-positive (`<= 0`) prices. Volume of `0` remains valid.
- **Time-aware borrowing cost (Q2)**: `BorrowingCostModel` now uses the engine-derived `bar_seconds`; hourly bars correctly charge `daily_rate / 24` rather than a full day's interest. `days_per_year` is exposed on the constructor (365 / 360 / 252). `FundingRateModel` ignores `bar_seconds` by design (its rate is already per-bar).
- **Impact model guards (Q4)**: `LinearImpactModel`, `SquareRootImpactModel`, and `PowerLawImpactModel` early-return 0 when `order_size <= 0` (previously the square-root model raised on `math.sqrt(<0)`).
- **Metadata lock (S1)**: `aiphaforge.__version__` is now CI-locked to match `pyproject.toml` via `tests/test_metadata.py`.

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
- 30+ metrics: Sharpe, Sortino, Calmar, max drawdown, VaR, CVaR, profit factor, and more
- Monthly/yearly return breakdowns, multi-strategy comparison
- Custom benchmark comparison (or automatic buy-and-hold)
- Per-asset analysis with correlation matrix

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
pip install aiphaforge[significance]  # arch for Model Confidence Set
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
