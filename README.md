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

### Pick Your Entry Point

`BacktestEngine` accepts six input-shape setters.  Pick the one matching
your data shape; the rest are mutually exclusive (calling a different
setter clears the previous input state).

| Setter | Input shape | Use when | Mutual exclusion |
|---|---|---|---|
| `set_strategy(strategy)` | `BaseStrategy` instance | You have a strategy class that computes signals each bar from price and state. | Calls `_clear_wide_input_state()`. |
| `set_signals(signals)` | `pd.Series` (single-asset) or `Dict[str, pd.Series]` (multi-asset) | You have pre-computed continuous signals in [-1, 1] (or NaN-hold). | Calls `_clear_wide_input_state()`. |
| `set_signals_wide(signal_wide)` | `pd.DataFrame` (rows=bars, cols=symbols) | You have a wide DataFrame of signals — single setter for the multi-asset case. | Inline clears `_strategy`, `_signals`, `_target_weights`, `_target_weights_wide`, `_target_weights_wide_config`. |
| `set_score_wide(scores, rule)` | `pd.DataFrame` of ML scores + a `ScoreToSignalRule` | You have model scores and want an explicit thresholding rule applied each bar. | Inline clears `_strategy`, `_signals`, `_target_weights`, `_target_weights_wide`, `_target_weights_wide_config`; the resulting signal frame is then set via `set_signals_wide(...)`. |
| `set_target_weights(target_weights)` | `pd.Series` or `Dict[str, pd.Series]` | You have target portfolio weights (institutional rebalancing). | Calls `_clear_wide_input_state()`. |
| `set_target_weights_wide(target_weights, ...)` | `pd.DataFrame` of target weights, optional `rebalance_frequency` | Multi-asset target-weight rebalancing on a daily DataFrame with quarterly (or similar) cadence. | Inline clears `_strategy`, `_signals`, `_target_weights`, `_signals_wide`. |

`set_fee_model(...)` is unrelated — it configures cost simulation
independently of input shape.

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

### LLM Memory Probes
A screening and inspection toolkit for measuring training-data leakage in LLM-driven backtests. Two orthogonal probes, no verdicts, no certification — descriptive numbers only.
- **Q&A Probe**: `KnowledgeProbe` generates objective same-bar OHLC + direction questions from a dataset; user runs them through any LLM externally; engine scores answers against ground truth with banded relative-error tolerance
- **A/B Probe**: `run_ab_probe()` runs an AI agent and a comparable baseline on raw vs transformed data, computes per-metric `excess_drop = ai_rel_drop − baseline_rel_drop` with symmetric normalization, low-anchor handling, and an optional AI-on-AI noise control to separate transform-induced sensitivity from generic LLM brittleness
- **7 built-in transforms** across three canonical stages: `SymbolMasker` and `DateShift` (metadata), `PriceScale` and `PriceRebase` (level), `OHLCJitter`, `BlockBootstrap`, `WindowShuffle` (series). Pipeline enforces stage order, mode compatibility, and OHLC integrity validation
- **Two execution modes**: `view_only` (engine fills at real prices, agent sees transformed view — Strategy-based agents only in v2.0) and `market_level` (transformed dataset becomes the execution market)
- **Anti-gaming protection**: `max_range_width` cap demotes "predict everything" interval answers to `miss`; auto-injected `transform_detectability_warning` for transforms an LLM may behaviorally react to; capacity-parity check for AI vs baseline turnover mismatch
- **User-attested manifest**: `provider_config` recommended-keys list (`model`, `snapshot_id`, `temperature`, `prompt_template_hash`, `tool_policy`, …) for cross-paper comparability — engine never verifies these claims, the user owns the publication-grade attestation

### KnowledgeProbe Pillar (v2.2)
The second leakage-diagnostic pillar: a standalone `knowledge_check()` workflow with forward-extrapolation probes, a held-out synthetic anchor, vol-scaled tolerance, and a typed `KnowledgeCheckReport`. Deliberately not folded into a single `audit()` — see "Pillar non-transitivity" below.
- **`ContinuationProbe`**: forward-extrapolation Q&A. Three templates (`NextCloseContinuation`, `NextRangeContinuation`, `NextReturnContinuation`). Both `context_bars` and `forward_horizon` are required positional args (no engine-side defaults — the values are task-design choices and silent-default-driven drift between papers is the failure mode being prevented).
- **`RankContinuationProbe` + `RankAnswer`**: cross-sectional ranking probe. `RankAnswer` has explicit partial/tie/extra semantics (omitted OR extra → "invalid", symmetric scoring eliminates the perverse incentive where hallucination scores better than partial answer). Tie-corrected ρ matches scipy's `spearmanr` to 1e-12. N-dependent quantile cutoffs (Monte-Carlo lookup for N≤20, Gaussian for N>20).
- **`build_synthetic_anchor()`**: held-out vol-matched fabricated series. Three methods: `garch_resample` (GJR-GARCH(1,1,1) for equity when n≥500 bars, falls back to GARCH(1,1) below the threshold for determinism); `block_bootstrap`; `random_walk_volmatched`. Equity-vs-crypto auto-detector uses three diagnostics (full leverage corr <-0.10 + sign-stable across two halves + block-bootstrap CI excludes zero), with bootstrap SE and classification confidence in provenance.
- **Vol-scaled tolerance** (`VolScalingSpec`): per-asset auto-picker for the σ estimator (US equity → Garman-Klass; crypto / futures → Parkinson; penny stocks → stdev_returns). Causality is strict: point-in-time uses bars `< t`, continuation uses bars `≤ context_window[-1]` held fixed across forward horizons. H==L fallback fires when ≥50% of window bars are degenerate (with a labeled-heuristic bias estimate in provenance).
- **`AttestedAnswers`** + dual hash: `parsing_schema_hash` (rejects "unpinned" version for built-in parsers) AND `prompt_template_hash` (covers system prompt, user template, closed-schema `ContextSerializationSpec` with CSV escaping/JSON scalars/decimal-separator collision check, NFC + Cf-scrub + variation-selector strip + CRLF normalization).
- **`knowledge_check()` orchestrator**: pure idempotent function (only impurities are `report_uuid` and `wall_clock_utc`). 9-key `provider_config` validation (temperature, top_p, model_id, model_version, max_output_tokens, tokenizer_id, seed, reasoning_effort, stop_sequences — typed `None` for nullable). PCG64 BitGenerator pinned for bootstrap reproducibility. Tango (1998) paired score interval for per-bucket CIs (the anchor question set mirrors real position-by-position so independent-samples Newcombe Method 10 was the wrong shape). Sign-test variant pinning (exact binomial < 25, normal+continuity 25–39, plain normal ≥ 40).
- **Refusal detection**: `looks_like_refusal()` keyword-in-leading-50-chars short-circuit + length/digit-ratio fallback. Joint `compute_effective_rate()` per AnswerRecord (parsed AND not refused) — replaces product-of-marginals which double-discounted the common case where paragraph refusals trigger BOTH parse failure and refusal-keyword presence. Symmetric refusal threshold with explicit 0/0 handling.
- **Persistence baseline** (continuation only): per-template rule. `NextReturnContinuation` predicts zero log-return (martingale hypothesis); close/range templates predict the last-context-bar value. `persistence_caveat` is a typed field on the report, not buried in notes — paper readers cannot mistake `persistence_vs_real_sign_test_p` (deterministic vs sampled) for like-for-like with `anchor_vs_real_sign_test_p` (LLM vs LLM).

#### Pillar non-transitivity
The Q&A pillar (v2.2 `knowledge_check`) is **non-transitive** with the obfuscation pillar (v2.0 `run_ab_probe`) and the differential-bootstrap pillar (v2.1). The same agent passing one does not imply it passes another. They measure different things:
- **Knowledge-check pass + obfuscation-bootstrap fail**: An agent answers point-in-time OHLC trivia about historical bars correctly (the LLM has read the company's stock pages) but its trading decisions are sensitive to symbol masking — the agent's *decision pipeline* uses the symbol identity. Pillar A says "the model knows the data"; pillar B says "the model's strategy depends on knowing it". Both are true, neither implies the other.
- **Knowledge-check fail + obfuscation-bootstrap pass**: An agent cannot recall historical OHLC values (low Q&A score) but its strategy is invariant under symbol masking. Pillar A says "the model has not memorized the bars"; pillar B says "the model's strategy doesn't lean on identity". Both are true; the model is honest *for trading purposes* even though it has weak factual recall.

Every `KnowledgeCheckReport.notes` carries this warning verbatim. Reports are intentionally not aggregated into a single 🟢/🟡/🔴 verdict — `KnowledgeCheckReport.is_pillar_summary` is `False` and the `__post_init__` rejects any attempt to flip it.

#### v2.8 release notes — Cleanup + Public API Lock

v2.8 is the final v2.x release. Three cleanups in service of the v2.x → v3.0 transition: (1) hard-delete all 4 deprecated aliases scheduled for v2.8; (2) lock the public `__all__` surface on 30 previously-implicit module APIs; (3) draft a private v3.0 deprecation roadmap. Engine source diff: zero lines.

**Headline breaking change — `KnowledgeCheckReport.bucket_delta_tango_ci` removed**

This was the only non-underscore-prefixed alias in the v2.8 cleanup batch. It was deprecation-scheduled at the comment level since v2.2.1 but never emitted a runtime warning during the v2.x line, so the v2.8 deletion is your first concrete signal. Migration is a literal-text sed:

```bash
# GNU sed (Linux):
sed -i 's/bucket_delta_tango_ci/bucket_delta_ci/g' your_code.py
# BSD sed (macOS):
sed -i '' 's/bucket_delta_tango_ci/bucket_delta_ci/g' your_code.py
```

`bucket_delta_ci` and `bucket_delta_tango_ci` always shared the same dict object during v2.x (cross-populated in `__post_init__`); the rename is semantic-equivalent.

**Failure modes** (so you can pattern-match if your upgrade breaks):

- `AttributeError: 'KnowledgeCheckReport' object has no attribute 'bucket_delta_tango_ci'` — direct attribute read.
- `TypeError: __init__() got an unexpected keyword argument 'bucket_delta_tango_ci'` — constructor usage.
- `TypeError: replace() got an unexpected keyword argument 'bucket_delta_tango_ci'` — `dataclasses.replace`.
- `report.to_dict()` and `dataclasses.asdict(report)` no longer contain the key — **silent**. Find affected consumers before upgrading:
  ```bash
  grep -rn 'bucket_delta_tango_ci' your_pipeline/
  ```

**3 underscore-prefixed aliases removed** (private by convention, low blast radius). All three fail with `ImportError: cannot import name '<alias>' from 'aiphaforge.probes.orchestrator'` on direct import, or `AttributeError` on `getattr`-style access:

- `aiphaforge.probes.orchestrator._BUCKET_ORDINAL_WEIGHTS` → import `LEAKAGE_INDEX_BUCKET_WEIGHTS` from the same module instead.
- `aiphaforge.probes.orchestrator._apply_vol_scaling_to_question_set` → import `apply_vol_scaling_to_question_set` from `aiphaforge.probes._vol`.
- `aiphaforge.probes.orchestrator._pair_scores_by_position` → use `_pair_scores_by_question_id` (qid-based pairing degrades gracefully on dropped/reordered questions; position-based silently misaligned).

**`__all__` user contract** — v2.8 adds `__all__` to 30 previously-implicit module-level `.py` files. Combined with the 9 modules that already declared `__all__` (signals, signal_rules, signal_strategy, factors, factor_strategy, strategy_factors, diagnostics, factor_library, package `__init__`), **every** module under `aiphaforge.*` now has a declared public API. **If you import a symbol from `aiphaforge.<module>` that is NOT in `<module>.__all__`, your code depends on an internal that may move in v3.0.** Check via:

```python
import aiphaforge.engine
print(aiphaforge.engine.__all__)
```

To audit every module at once:

```python
import importlib, pkgutil, aiphaforge
for m in pkgutil.iter_modules(aiphaforge.__path__):
    if m.ispkg:
        continue
    mod = importlib.import_module(f"aiphaforge.{m.name}")
    print(f"{m.name}: {getattr(mod, '__all__', '(NO __all__)')}")
```

Subpackage `__init__.py` exports (`probes/__init__.py`, `alpha/__init__.py`, `calendars/__init__.py`) were already curated and are unchanged.

**Explicit non-change callout: `BaseStrategy._compute(df)` is NOT deprecated.** Earlier drafts of the v2.x roadmap considered soft-deprecating it; that decision was reversed during v2.8 planning. `_compute` and `generate_signals` form a deliberate two-layer override surface (single-asset hook + multi-asset dispatch), not alternatives. Both are permanent public API.

**What v2.8 does NOT include**:

- Any new functionality. v2.8 is cleanup only.
- IncrementalFactor engine integration (deferred to v3.0).
- StrategyNode 5-class consolidation (deferred to v3.0).
- `dict[str, Series]` → wide DataFrame default migration (deferred to v3.0).
- Any DeprecationWarning emissions — v2.8 alias removals are hard deletes; the README and resulting AttributeError / TypeError are your only signals.

#### v2.7 release notes — Engine Signal-Input Widening

v2.7 adds three wide-DataFrame entry points to `BacktestEngine` that route through the v2.3 / v2.4 adapters internally. The existing `set_signals(...)` contract (Series / dict[str, Series] only) is preserved. Per-bar engine loop diff: zero lines.

**New entry points**:

```python
# 1. Pre-computed wide signals (single setter for the multi-asset case).
engine.set_signals_wide(
    signal_wide_df,          # index=datetime, columns=symbol
    warn_on_inf=True,        # default: ±Inf in the DF emits a warning
    strict=False,            # CI mode — see below
).run(data)

# 2. Wide ML scores + an explicit ScoreToSignalRule gate.
from aiphaforge import ThresholdScoreRule
engine.set_score_wide(
    scores_df,               # raw model output (any float values)
    ThresholdScoreRule(long_threshold=0.7, short_threshold=0.3),
).run(data)

# 3. Wide target weights with quarterly rebalancing on a daily DF.
quarterly_dates = [df.index[0], df.index[63], df.index[126], df.index[189]]
engine.set_target_weights_wide(
    weights_df,
    rebalance_dates=quarterly_dates,   # OMIT this and you fire 252 rebalances/year
    snap="exact",
    on_collision="warn",
).run(data)
```

**Score-rule default asymmetry — important when picking a rule**: `ThresholdScoreRule` defaults to `neutral_action="hold"` (NaN — engine ffills the prior position). `CrossSectionalQuantileRule` defaults to `neutral_action="flat"` (0 — engine closes the position on every bar where the name is in the middle quantile). On a daily score frame with monthly rebalances, `CrossSectionalQuantileRule`'s default emits explicit closes daily, which is almost certainly NOT what you want for sparse rebalancing — pre-mask the score frame to your rebalance dates, OR compute weights yourself and use `set_target_weights_wide(rebalance_dates=...)`.

**`set_score_wide` does NOT down-sample**: the rule is applied per-bar across the full input frame. If your scores are daily but rebalances are monthly, the score-to-signal gate fires daily.

**NaN semantics, by entry point**:

- `set_signals_wide` — NaN in the wide DF means **hold** (engine ffills prior position).
- `set_target_weights_wide` — NaN weight in the wide DF is coerced to **0** (explicit close) by the underlying schedule adapter. This is intentional but asymmetric with `set_signals_wide`'s NaN-as-hold; document your fixture explicitly when both setters appear in one pipeline.

**CI / strict mode**: `strict=True` is a one-kwarg fail-fast switch on all three setters. It promotes Inf-warnings to errors and (for `set_target_weights_wide`) defaults `on_collision` to `"raise"` and `snap` to `"exact"`. Strict ALWAYS wins — passing `strict=True` together with a conflicting explicit kwarg (e.g. `strict=True, warn_on_inf=False` or `strict=True, on_collision="warn"`) raises `ValueError` immediately at the setter call rather than silently overriding your value. If you want partial strictness, set the individual kwargs explicitly without `strict=True`.

```python
# Fail-fast for CI / production pipelines:
engine.set_signals_wide(df, strict=True)
engine.set_target_weights_wide(weights, strict=True)
```

**Breaking change — `set_signals(df)` now raises `TypeError`**: previously a `pd.DataFrame` passed to `set_signals` would crash deep inside `_get_signals` with a confusing `AttributeError`. v2.7 refuses cleanly at the boundary with a message pointing to `set_signals_wide`. The exception type changes from `AttributeError` to `TypeError`, and the failure point moves from inside `run()` to inside `set_signals` — downstream `try/except` handlers may notice. Migration: replace `engine.set_signals(wide_df)` with `engine.set_signals_wide(wide_df)`.

**What v2.7 does NOT include**: engine integration with `IncrementalFactor` (still deferred to v3.0); no auto-inference between Series / dict / DataFrame in `set_signals` — each entry point has one accepted shape per the master plan §5.5 anti-list.

#### v2.6 release notes — Incremental Factor MVP

v2.6 ships the `IncrementalFactor` API for stateful per-bar factor computation alongside v2.4's batch `BaseFactor`. Engine source diff: zero lines.

> **`BacktestEngine` does NOT yet consume `IncrementalFactor`** in v2.6 (and v2.7 did not add it — that integration is deferred to v3.0; v2.7 widened the signal-input API instead, see the v2.7 release notes above). Users wanting to drive incremental factors today must call `factor.update(bar, state)` manually from their own code, or use `factor.run_all(data)` for batch-mode replay.

**New top-level exports**:

- `IncrementalFactor` — abstract base for stateful per-bar factors
- `FactorState` — per-(factor, symbol) state dataclass; subclassable
- 5 concrete factors: `RollingMeanIncremental`, `RollingStdIncremental`, `MomentumIncremental`, `RSIIncremental`, `VolumeZScoreIncremental`

**Usage** (single-symbol):

```python
from aiphaforge import RSIIncremental

factor = RSIIncremental(period=14)
sig = factor.run_all(df)              # batch-equivalent for full-data replay

# or per-bar driven:
state = factor.initial_state()
for _, bar in df.iterrows():
    value, state = factor.update(bar, state)
```

**MetaController param-change recipe** (clear+rebuild via `rewarmup`):

```python
factor.update_params(period=20)  # clears self._state
factor.rewarmup(history)         # rebuilds state under new params
```

**Per-factor warmup** (number of leading NaN bars before the first non-NaN value):

| Factor | Warmup bars |
|---|---|
| `RollingMeanIncremental(window)` | `window - 1` |
| `RollingStdIncremental(window, ddof=1)` | `window - 1` |
| `MomentumIncremental(window)` | `window` |
| `RSIIncremental(period)` | `period - 1` (matches pandas `ewm(min_periods=period)`) |
| `VolumeZScoreIncremental(window)` | `window - 1` |

**Numerical equivalence**: incremental factors are tested against the v2.4 batch reference (or a pandas one-liner where no v2.4 batch exists). Closed-form factors (`RollingMean`, `Momentum`) match within `rtol=1e-12`. Recursive factors (`RollingStd`, `RSI`, `VolumeZScore`) match within `rtol=1e-9, atol=1e-12` — pandas batch recomputes from the raw window each bar (no recursion), so even a numerically stable Welford / Wilder smoother differs at the `~1e-10` scale on adversarial price series.

**What v2.6 does NOT include**: cross-sectional incremental (deferred to v2.7+ / v3.0), engine integration (v2.7), `MetaController` automated rewarmup hook (v3.0), incremental neutralization / regression-residual factors (v3.0), and dynamic universe symbol add/remove integration (v3.0).

**Design decisions callers should be aware of**:

- **State ownership**: the caller manages one `FactorState` per `(factor, symbol)` pair. A single factor instance is meant to be reused across symbols by passing different states into `update(bar, state)`. The factor itself MUST NOT carry per-symbol state in instance fields (cross-symbol contamination risk).
- **Parallel APIs, not subclassed**: `IncrementalFactor` (v2.6) and `BaseFactor` (v2.4) are independent. Pick one per factor based on workflow — batch for research / one-shot backtests, incremental for live trading or event-driven simulation. A future v3.0 may unify via a shared protocol.
- **Dynamic universe**: symbols added mid-run must be warmed up via replay-from-history — caller calls `factor.rewarmup(symbol_history)` before consuming the symbol's first value. Symbols leaving the universe should have their state dropped; re-entry is treated as a fresh add (any retained stale state would be a footgun across delisting / ticker reuse).
- **`available_at` semantics**: a factor computed from bar `t`'s OHLC is only available AFTER bar `t`'s close. The earliest decision that may use it is bar `t+1`'s open. Using bar `t`'s factor value at bar `t`'s open is lookahead bias — the v2.4 anti-lookahead invariant carries over.
- **Hook coupling**: a hook that mutates strategy params is responsible for propagating the change to the incremental factor (`factor.update_params(...)` then `factor.rewarmup(history)`); risk-state hooks (positions, capital, exposure) have no effect on the factor cache — risk and factor computation are orthogonal subsystems.

#### v2.5 release notes — StrategyNode 3-Mode Rewrite

v2.5 rewrites the 5 `StrategyNode` composite classes (`WeightedBlend`, `SelectBest`, `PriorityCascade`, `VoteEnsemble`, `ConditionalSwitch`) so they can host modern `generate_signals(data)` children alongside the existing `_compute(df)` legacy children. Engine source diff: zero lines.

**New `mode` parameter** (keyword-only, defaults to `"auto"`):

- `"auto"` — try each child's `generate_signals`; on any `Exception`, fall back to `_compute` (single-asset only). The first fallback for each `(composite_class, child_class)` pair logs a `WARNING`; subsequent fallbacks for the same pair drop to `DEBUG`.
- `"generate_signals"` — always route through `child.generate_signals(data)`. No fallback; child exceptions propagate.
- `"legacy_compute"` — always route through `child._compute(df)`. Multi-asset dict input raises `TypeError` (legacy `_compute` is single-asset only). This is the v2.4-equivalent safety hatch.

Existing user code constructing composites without `mode=` is unchanged: bit-equal output against v2.4 (snapshot-pinned).

**Direction-only contract** (v3.0 will widen): If any child carries a `SignalSpec` with `kind != "direction"` (e.g. `"target_weight"`), the composite raises `ValueError` at `generate_signals` entry. Legacy children without `spec` are trusted as direction. Mixed-kind composition is a v3.0 design point.

**Multi-asset symbol mismatch is a hard error**: If two children of `WeightedBlend(mode="generate_signals")` return per-symbol dicts with different key sets, the composite raises `ValueError` naming the missing/extra symbols. Pre-pad via `child_output.reindex(union)` is the user's responsibility — silent renormalization across universes is deferred to v3.0.

**Nested-composite mode is independent**: `WeightedBlend(mode="auto", children=[SelectBest(mode="legacy_compute"), ...])` — the inner `SelectBest` runs its children via `_compute` regardless of the parent's `auto` mode. Each composite reads its own `.mode`; parents do not propagate.

**`extract_strategy_factors` contract is preserved**: the adapter still returns `FactorSet.empty()` for any `StrategyNode` subclass, independent of `mode`. Recursive factor extraction is a v3.0 opt-in feature (separate function).

#### v2.4 release notes — Factor + Alpha Layer

v2.4 adds the **Factor + Alpha layer** on top of v2.3's Signal Layer Foundation. Engine source diff: zero lines (parallel API surface only).

**New top-level exports**:

- `SignalSpec` / `SignalFrame` typed wrappers (deferred from v2.3 — earn their keep alongside the factor layer where typed metadata flows through factor → rule → signal pipelines).
- Factor layer: `BaseFactor`, plus 5 reference factors (`RSIFactor`, `MomentumFactor`, `MASpreadFactor`, `VWAPDistanceFactor`, `VolumeZScoreFactor`) with parametrised `name` format pinning (`"rsi_14"`, `"momentum_20"`, etc.).
- `FactorRuleStrategy(factor, rule)` — single-factor MVP per master plan §6.4. Multi-factor composition deferred to v2.5+ when `FactorSpec.is_primary` is honored.
- `extract_strategy_factors(strategy, data)` — adapter exposing implicit factors of 5 built-in strategies (MACrossover, RSIMeanReversion, VWAPReversion, MomentumRank, PairsTrading). Composite strategies (StrategyNode subclasses) explicitly return `FactorSet.empty()` (no recursion). Point-in-time snapshot — uses strategy's CURRENT params; documented in docstring.
- `aiphaforge.alpha` subpackage: `AlphaScreener` MVP (IC / RankIC / ICIR / coverage / quantile returns), `forward_returns` (with `return_type="simple"` default + `"log"` opt-in), `ic` / `rank_ic` / `coverage` per-timestamp metrics, `signal_forward_return` / `signal_hit_rate` / `signal_turnover` for signal-only strategies (no factors required).
- `assert_factor_no_lookahead` / `assert_signal_no_lookahead` (already in v2.3; promoted to top-level in v2.4).
- `FactorSet.to_json()` / `from_json()` — sanctioned cross-version persistence path (R10). JSON via `pandas orient='split'` preserves DatetimeIndex / column / values dtypes. Schema version 1.

**Architectural firewall** (R7 + R11): `aiphaforge.alpha.*` and `factor_library.py` / `factor_strategy.py` / `strategy_factors.py` MUST NOT import any of 13 execution-layer modules (engine, fees, broker, market_impact, position_sizing, risk, portfolio, optimizer, portfolio_optimizer, capital_allocator, margin, costs, latency). AST-based guard tests verify each module.

**Closed v2.2.2 G methodology gap**: canonical-formula tests for `_vol.py` (Parkinson + Garman-Klass + stdev_returns ddof=1), `_rank.py` (tie-corrected Spearman ρ vs hand-derived Kendall formula), `anchors.py` (block bootstrap seed determinism + GJR-GARCH parameter recovery with empirically-pinned per-parameter tolerance bands), and a true end-to-end Phipson-Smyth invocation of `permutation_test` (the v2.2.2 G test reproduced the formula inline only).

**v2.4 reference-factor formulas pinned in plan + tests**:

- `RSIFactor(period)` wraps `indicators.RSI`. Output [0, 100].
- `MomentumFactor(window)` is gross simple return `close[t]/close[t-w] - 1` (NOT `indicators.ROC`'s 100x-scaled).
- `MASpreadFactor(short, long, ma_type)` is `MA_short / MA_long - 1`.
- `VWAPDistanceFactor(window)` uses **rolling** VWAP (NOT the running-cumulative `indicators.VWAP`). Zero-volume window naturally returns NaN via `0/0`.
- `VolumeZScoreFactor(window)` uses `min_periods=window` rolling z-score with `ddof=1`. Explicitly NOT `.expanding()` (would be lookahead).

**Default behaviour preservation**: all v2.3 / v2.2.x APIs are untouched. `wide_to_signal_dict(warn_on_inf=False)` default preserves v2.3 silent Inf coercion; the opt-in warning is for users debugging upstream factor bugs.

#### v2.3 release notes — Signal Layer Foundation

v2.3 establishes the **Signal Layer** as a first-class abstraction without touching `BacktestEngine`. This is the first step of the multi-version refactor (v2.3 → v2.8) decoupling factors from strategies; see `docs/AIphaForge_Framework_Refactor_Master_Plan_v1.0.md` for the full roadmap.

**New top-level exports**:

- Signal-layer utilities: `transitions_only`, `prepare_signals_for_engine` (with `broadcast=True` for global market-wide signals), `dict_to_signal_wide`, `wide_to_signal_dict`, `target_weight_wide_to_schedule` (with `union`/`intersection` universe alignment + `on_collision` policy for snap collisions).
- Score → signal rules: `ThresholdScoreRule` (default `neutral_action="hold"` — preserves existing positions on uncertain scores) and `CrossSectionalQuantileRule` (default `neutral_action="flat"` — Alphalens / Qlib convention closes middle quantiles).
- `DirectSignalStrategy` — wraps ML / AI / external precomputed signals or scores+rule. `update_signals(...)` / `update_scores(...)` instance methods support live MetaController-driven updates.

**New subpackages** (subpackage-import only in v2.3, top-level exports planned for v2.4):

- `aiphaforge.factors` — `FactorSpec`, `FactorSet`, `FactorProvider` Protocol (data structures only; no compute logic). **Note**: these dataclasses are `frozen=True`; field additions in future minor versions will break pickled instances. Persist via JSON if cross-version stability is required.
- `aiphaforge.diagnostics` — `assert_factor_no_lookahead` and `assert_signal_no_lookahead` test/research tools using prefix-slice semantics with `atol=1e-12` default. NOT runtime checks.

`SignalSpec` / `SignalFrame` typed wrappers are deferred from the original v2.3 scope to **v2.4** alongside the factor layer — they earn their keep when factor → rule → signal pipelines need typed metadata flowing through the stack.

**Internal**: the four backward-compat aliases (`_BUCKET_ORDINAL_WEIGHTS`, `bucket_delta_tango_ci` field, `_apply_vol_scaling_to_question_set` re-export, `_pair_scores_by_position`) had their removal-version annotations updated from v2.3.0 to **v2.8.0** to match the revised v2.x → v3.0 roadmap. All four were hard-deleted in v2.8 cleanup — see v2.8 release notes above for the migration paths.

`BacktestEngine` source diff: zero lines. The v2.3 release adds parallel API surface only.

#### v2.2.2 patch additions

##### BREAKING NUMERIC: PSR / DSR default σ convention
`probabilistic_sharpe_ratio` and `deflated_sharpe_ratio` previously used `rets.std()` (pandas default `ddof=1`, sample σ) but paired it with the `√(T-1)` standardization in the z-statistic — internally inconsistent with the cited Bailey & López de Prado (2012, eq. 14), which derives the formula using the **biased** σ estimator (ddof=0, divides by T). v2.2.2 flips the default to `std_ddof=0` (canonical Bailey-LdP form). Numeric drift relative to v2.2.1: ~0.2% at T=252, ~2% at T=20. Users who pinned an exact v2.2.1 PSR/DSR value in their tests can preserve the historic number by passing `std_ddof=1` explicitly. Both functions also gained a docstring note that the DSR's variance formula assumes i.i.d. returns; the Lo (2002) autocorrelation correction is currently un-applied and flagged for v2.3.

#### v2.2.1 patch additions
- **`LEAKAGE_INDEX_BUCKET_WEIGHTS`** public export (`MappingProxyType`, immutable) — paper authors should import the locked weights instead of transcribing the literals so citations track the version-locked schedule (changing the weights requires a `__version__` bump per § 14).
- **`bucket_delta_ci`** as the canonical name (the v2.2.1-historic `_tango_ci`-suffixed alias was hard-deleted in v2.8 — see v2.8 release notes above). The historic name was misleading: the implementation is Wald-style with a sentinel-on-zero-width fallback, not real Tango. Real Tango lands in v2.2.2.
- **`anchor_validity`** typed as `Literal["NO_ANCHOR", "OK", "REFUSAL_SUSPECTED", "PAIRING_FAILED"]` (was bare `str`). `PAIRING_FAILED` is a new tag distinct from `REFUSAL_SUSPECTED` — the former fires when the real-side and anchor-side question_id sets do not overlap, the latter when one side's effective rate is much lower than the other.
- **`parser_used_distribution_real` / `_anchor`** typed `Mapping[str, int]` fields on `KnowledgeCheckReport` (rank probes only) — per-side `Counter` of which parser path resolved each `AnswerRecord` (e.g., `user_provided_list`, `json_array`, `numbered`, `raw_text_unparseable`). When the distribution is non-degenerate (different parser paths fired across the probe set), a drift-note appears in `notes` pointing readers at `manifest.provider_config` to check for prompt-template churn.
- **Structured-input refusal carve-out**: when `AnswerRecord.parsed_answer` is a `RankAnswer` / `Sequence[str]` / dict-with-ranking AND `parse_status == "valid"`, `compute_effective_rate` and `compute_refusal_rate` skip the refusal heuristic on `raw_answer` (the user already did the parsing work; the engine's refusal heuristic is moot for them).
- **`KnowledgeCheckReport.to_dict()`** helper that unwraps the `MappingProxyType` fields so users following the standard `json.dumps(asdict(report))` pattern don't crash. Nested dataclass fields (`real_score`, `anchor_score`, `persistence_baseline_score`) contain no `MappingProxyType` members and remain safe to pass to `dataclasses.asdict()` directly.
- **Sign-test continuity correction** uses the textbook `max(|n_pos − n/2| − 0.5, 0)` form (floors the numerator at 0 before dividing) — same numeric output as the prior post-hoc clamp, cleaner derivation; tied `n_pos == n/2` produces `z = 0 → p = 1` directly.

### Trading Calendar
Daily-resolution trading-day primitive shipped as a self-contained `aiphaforge.calendars` package. Note: the module name is **plural** (`calendars`) to avoid shadowing the Python stdlib `calendar`.
- **`TradingCalendar`** dataclass with `is_trading_day` / `next_trading_day` / `prev_trading_day` / `snap` / `is_conformant` and a `stable_fingerprint` for cross-instance value-equality
- **4 predefined exchange calendars**: `US_EQUITY` (NYSE-style), `CHINA_A_SHARE` (SSE), `CRYPTO_24_7` (every date a trading day), `US_FUTURES_ES` (CME equity futures). Holiday data covers 1990–2035, vendored from MIT-licensed `pandas_market_calendars` (NOT a runtime dependency — generated offline by `scripts/generate_holidays_json.py`)
- **`DateShift(calendar=, snap=, on_collision=)`** wires the calendar into the existing memory-probe transform: `snap` chooses `"forward"` / `"backward"` / `"nearest"` / `"error"` for non-trading shifted dates, and `on_collision` handles duplicates after snap (`"error"` / `"keep_first"` / `"keep_last"`)
- **`validate_ohlcv_integrity(calendar=)`** and **`TransformPipeline(calendar=)`** thread calendar conformance into the existing OHLC validators; calendar mismatches between explicit and inferred sources fail fast with `CalendarConflictError`
- **Per-scenario manifest warnings**: `transform_detectability_warnings` (calendar-snap fingerprint caveat) and `calendar_snap_collisions` (when row-dropping policies fire — with per-arm breakdown so identical-arm fixtures don't double-count) are serialised into the per-scenario A/B report so JSON readers see them
- **Per-call diagnostics** (v2.1): `TransformPipeline.apply_with_diagnostics()` returns a frozen `TransformApplyResult(data, diagnostics)` so callers can inspect collision diagnostics outside `run_ab_probe`. `apply()` keeps the v2.0 `pd.DataFrame` return type
- **Vendor-attested provenance**: every predefined calendar's `.provenance` mapping carries the source package version, generation script SHA-256, `last_verified` date, and `next_refresh_target` (currently 2033-12-31) so audit consumers can trace every shipped holiday back to its generation step
- **Daily resolution only**: no early-close / lunch-break / partial-holiday / intraday-session modelling — that's deferred to v2.2 alongside the Hook-based view-only broker-proxy wrapper

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

### LLM Memory Probes — Q&A

The engine generates objective questions, the user runs them through
any LLM externally, then submits parsed answers back for scoring.
The engine never calls the LLM.

```python
from aiphaforge.probes import (
    KnowledgeProbe, OpenQuestion, ToleranceProfile,
    DEFAULT_TEMPLATES, sample_dates,
    AnswerRecord, serialize_answer_records,
    parse_numeric_answer,
)

# 1. Generate a question set from the dataset. Mix the default
#    templates with a strict-mode opt-in template — the
#    ``*_strict()`` presets use bp-scale exact thresholds calibrated
#    for memorization detection (the loose preset matches v2.0
#    defaults; opt-in only, no behavior change for existing callers).
templates = list(DEFAULT_TEMPLATES) + [
    OpenQuestion(tolerance=ToleranceProfile.us_equity_price_strict()),
]
probe = KnowledgeProbe(symbol="AAPL", templates=templates)
ts_list = sample_dates(data, n=200, seed=42, start=1)
qs = probe.build(data, ts_list)
qs.export_questions("questions.jsonl")     # safe to feed to the LLM
qs.export_answer_key("answer_key.jsonl")   # KEEP PRIVATE — truth values

# 2. ... user runs questions.jsonl through their LLM externally,
#    uses ``parse_numeric_answer(reply)`` (or the choice/binary
#    parsers) to coerce free-text replies into typed values,
#    and writes answers.jsonl as a list of AnswerRecord rows ...

# 3. Score, attesting to provider configuration for cross-paper
#    comparability. Recommended keys (model, snapshot_id,
#    temperature, prompt_template_hash, system_fingerprint,
#    prompt_cache_disclosed, ...) are listed in
#    ``RECOMMENDED_PROVIDER_CONFIG_KEYS`` — the engine never
#    verifies these claims; the user owns the attestation.
report = probe.score(
    "answers.jsonl", question_set=qs,
    provider_config={
        "model": "claude-opus-4-7",
        "temperature": 0.0,
        "prompt_cache_disclosed": True,
        "seed_attestation": False,
    },
)
print(f"score_real bands: {report.bands_breakdown}")
print(f"band_index_arbitrary: {report.band_index_arbitrary:.3f}")
# Per-template breakdown surfaces selective-memorization signals
# (e.g. "model nails close prices but is hopeless on highs").
for tid, entry in (report.by_template or {}).items():
    print(f"  {tid}: exact={entry['exact_rate']:.2%} "
          f"miss={entry['miss_rate']:.2%}")
```

### LLM Memory Probes — A/B

Compare an AI agent and a comparable baseline on raw vs transformed
data; the runner returns descriptive `excess_drop` distributions —
no verdicts, no p-values.

```python
from aiphaforge.probes import (
    run_ab_probe, ABScenario, MACrossBaseline,
    SymbolMasker, BlockBootstrap,
)

scenarios = [
    ABScenario(
        scenario_id="metadata_only",
        mode="market_level",
        transforms=[SymbolMasker(symbols=["AAPL"], seed=42)],
    ),
    ABScenario(
        scenario_id="bootstrap_block_20",
        mode="market_level",
        transforms=[BlockBootstrap(block_size=20)],
    ),
]

result = run_ab_probe(
    ai_factory=lambda: my_llm_strategy,           # any BaseStrategy or BacktestHook
    baseline_factory=lambda: MACrossBaseline(short=10, long=30),
    data=data,
    scenarios=scenarios,
    n_repeat=10,
    enable_ai_noise_control=True,                  # for stochastic transforms
    # v2.0.1: per-scenario determinism check catches Hooks whose
    # non-determinism is exposed only by transformed inputs.
    # ``determinism_profile="auto"`` resolves to ``v2_compat`` for
    # ``raw_only`` (preserves v2.0 behavior) and ``llm_balanced``
    # for ``per_scenario`` — which checks total_return + num_trades
    # + win_rate at rel_tol=1e-3, the practical LLM noise floor.
    agent_determinism_check="per_scenario",
    determinism_profile="auto",
    # ``agent_implementation_contract`` declares the agent's shape so
    # unsupported combinations (e.g. view_only + plain hook) are
    # reported as ``status="unsupported"`` rather than as failures.
    agent_implementation_contract="strategy",
    provider_config={"model": "claude-opus-4-7", "temperature": 0.0},
)

for sc in result.scenarios:
    for m, s in sc.metric_summaries.items():
        print(f"{sc.scenario_id} {m}: excess_drop ~ {s.mean_excess_drop}")
    for w in sc.warnings:
        print(f"  warning: {w}")
```

### Trading Calendar (v2.1)

Daily-resolution only — no early closes, no intraday sessions.

```python
from aiphaforge.calendars import US_EQUITY, TradingCalendar
import pandas as pd

# Membership + boundary helpers
US_EQUITY.is_trading_day(pd.Timestamp("2024-12-25"))   # False (Christmas)
US_EQUITY.next_trading_day(pd.Timestamp("2024-12-25")) # 2024-12-26
US_EQUITY.snap(pd.Timestamp("2024-01-06"), "nearest")  # 2024-01-05 (Fri)

# Calendar-aware DateShift for the A/B memory probe
from aiphaforge.probes import ABScenario
from aiphaforge.probes.transforms import DateShift

scen = ABScenario(
    scenario_id="us_back_3y",
    mode="market_level",
    transforms=[
        DateShift(
            offset=pd.DateOffset(years=-3),
            calendar=US_EQUITY,
            snap="forward",        # holiday → next trading day
            # Multi-year shifts WILL collide on NYSE: e.g. 2018-12-26
            # and 2018-12-22 both forward-snap to 2015-12-28 (Mon)
            # because 2015-12-25 was a Friday holiday. `keep_last`
            # accepts the data loss and surfaces a structured
            # collision warning into the per-scenario manifest. Use
            # `"error"` (the default) for validation contexts where
            # silent row drops are unacceptable.
            on_collision="keep_last",
        ),
    ],
)

# `_run_scenario` infers the calendar from the transform via the
# explicit `_aiphaforge_calendar_provider` marker protocol; user
# transforms with an unrelated `.calendar` attribute are ignored.
# Calendar conflicts (two DateShift with different calendars) fail
# fast with `CalendarConflictError`.
```

For per-call diagnostics outside `run_ab_probe` (e.g. when applying a
pipeline directly in user code), use `apply_with_diagnostics`:

```python
from aiphaforge.probes.transforms import (
    DateShift, TransformPipeline, TransformApplyResult,
)

ds = DateShift(
    offset=pd.DateOffset(years=-3),
    calendar=US_EQUITY,
    snap="forward",
    on_collision="keep_last",
)
pipeline = TransformPipeline(transforms=[ds], mode="market_level")

# Backward-compatible: apply() returns just the DataFrame
out_df = pipeline.apply(data)

# Opt-in: apply_with_diagnostics() returns a frozen result object
result: TransformApplyResult = pipeline.apply_with_diagnostics(data)
out_df = result.data
for diag in result.diagnostics:
    if diag.code == "calendar_snap_collision_rows_dropped":
        print(f"dropped {diag.details['collision_count']} rows")
```

### Factor Research (alpha screening)

Use the alpha layer to rank a custom factor before plugging it into a
`FactorRuleStrategy`.  `AlphaScreener.compute_ic` returns the
information-coefficient series across forward-return horizons;
`FactorReport` summarises IC stats, decile spreads, and turnover.

```python
from aiphaforge.alpha.evaluator import AlphaScreener
from aiphaforge.alpha.report import FactorReport
from aiphaforge.factor_strategy import FactorRuleStrategy
from aiphaforge.factors import BaseFactor


class MyMomentum(BaseFactor):
    def compute(self, df):
        return df["close"].pct_change(20)


screener = AlphaScreener(prices=price_df)
ic_series = screener.compute_ic(MyMomentum())
report = FactorReport.from_screener(screener, factor=MyMomentum())
# Once the factor passes screening, drop it into a strategy:
strategy = FactorRuleStrategy(factor=MyMomentum(), top_k=10)
```

### Hook-driven Order Submission

`BacktestHook.on_pre_signal` runs each bar before the engine processes
its own signals, so a hook can inject orders into the broker for that
bar.  The hook calls `context.broker.submit_order(order, timestamp=...)`
directly — `BacktestHook` itself does not expose `submit_order`.

```python
from aiphaforge.hooks import BacktestHook
from aiphaforge.orders import Order, OrderSide, OrderType


class CustomEntryHook(BacktestHook):
    def on_pre_signal(self, context):
        # context.broker, context.timestamp, context.portfolio, etc.
        # are available to inspect; submit an Order when your rule fires.
        if my_condition(context):
            order = Order(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                size=100,
            )
            context.broker.submit_order(order, timestamp=context.timestamp)


engine.add_hook(CustomEntryHook())
```

### LLM Memory Probes — `knowledge_check` orchestrator

`knowledge_check` is the high-level entry point for the LLM memory
probe pillar: feed it a `KnowledgeProbe`, an `AttestedAnswers` bundle,
and a provider config, and it runs the probe + comparison and returns
a `KnowledgeCheckReport` with leakage indices, anchor validity, and
persistence validity.

```python
from aiphaforge.probes import (
    AttestedAnswers,
    KnowledgeCheckReport,
    KnowledgeProbe,
    knowledge_check,
)

probe = KnowledgeProbe(name="AAPL_Q1_2024_close", ...)
answers = AttestedAnswers.attest(...)
provider_config = {
    "endpoint": "...",
    "api_key": "...",
    "model": "your-llm-model",
    "temperature": 0.0,
    "max_tokens": 100,
    "max_retries": 3,
    "request_timeout_seconds": 30,
    "rate_limit_rps": 5,
    "system_prompt": "...",
}

report: KnowledgeCheckReport = knowledge_check(
    probe=probe,
    attested=answers,
    provider_config=provider_config,
)
assert report.anchor_validity == "OK"
```

## v2.8.3 Release Notes

See [CHANGELOG.md](./CHANGELOG.md#283---2026-05-24).

## v2.8.2 Release Notes

See [CHANGELOG.md](./CHANGELOG.md#282---2026-05-21).

## v2.8.1 Release Notes

See [CHANGELOG.md](./CHANGELOG.md#281---2026-05-20).

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
