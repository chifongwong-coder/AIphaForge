# Changelog

All notable changes to AIphaForge are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Releases prior to v2.8.1 are not documented here; consult the git history
(`git log --oneline`) for earlier changes.

## [Unreleased]

## [2.8.6] - 2026-06-12

### Headline — framework gap fixes (T+1 settlement, spread models)

Closes the gaps surfaced by application-layer intraday usage. Core
modules change at the source level, but every new capability is
opt-in: default-configuration results are numerically identical to
v2.8.5 (snapshot-pinned). Default-visible changes are limited to the
`load_yahoo` bug fix, the annualization warning, and the `(approx)`
label on vectorized cost lines.

### Added

- `settlement="t+1"` + `asset_settlements` on `BacktestEngine` /
  `BacktestConfig` / `Broker`: SSE/SZSE-style same-day sell freeze,
  enforced at fill time inside `Broker._execute_fill` for every fill
  path (GTC, IOC/FOK direct, immediate orders). Oversized sells fill
  the settled portion and expire the remainder (`t+1_settlement`);
  FOK sells exceeding the settled quantity are killed outright.
  Vectorized mode + any `"t+1"` raises (silent optimism refused).
  First T+1 event per symbol emits one `UserWarning`.
- `aiphaforge.spread` (new public module): `BaseSpreadModel`,
  `FixedSpread(spread_bps)`, `VolatilitySpread(k, min_bps, max_bps)`.
  Event-driven fills cross half the quoted spread per side;
  `requires_volatility` models activate the rolling Parkinson vol
  pipeline without needing an impact model (separate per-bar channel
  with warmup-`None` semantics). Realized spread is side-signed and
  post-clamp; reported via `Trade.spread_cost`,
  `BacktestResult.total_spread`, and a `Total Spread` summary line
  (event-driven only).
- Vectorized cost path folds a global `FixedSpread`
  (`DefaultTradeCost.half_spread_rate`, one half-spread per side,
  applied even on the degenerate-notional branch); dynamic models and
  per-asset overrides warn per run and are ignored.
- `utils.infer_bars_per_year` + engine annualization-mismatch warning
  (3x density band, suggests the inferred bars-per-year, never
  auto-corrects).
- `FundingRateModel(funding_rate_8h=..., bar_interval_seconds=...)`:
  exact linear conversion of venue-quoted 8h funding rates; mutually
  exclusive with `funding_rate_per_bar`; legacy usage unchanged.
- README "Modeling Boundaries" section (session gaps, T+1
  approximations, spread modeling choices, vectorized cost
  approximation).

### Fixed

- `load_yahoo` crashed on yfinance >= 1.x (`'tuple' object has no
  attribute 'lower'`): single-symbol MultiIndex columns are flattened
  before lowercasing; the multi-symbol fallback no longer assumes
  MultiIndex columns.
- `MarginCallExitRule` dedup now tracks order ids and releases
  symbols whose liquidation order terminated without filling in full
  (fill-time rejections previously left the position stuck,
  never re-liquidated, while the margin call persisted).

### Changed

- Vectorized summaries label `Total Commission` / `Total Slippage`
  with `(approx)` via `result.metadata['cost_model']` (the
  once-per-process warning drowns in parameter sweeps).
- Limit/stop-limit fills are clamped to their limit price after all
  price adjustments whenever a spread or impact model is configured
  (previously the clamp only ran inside the impact branch; the
  legacy slippage-only path is untouched, preserving default-config
  results).
- `Trade` gains a `spread_cost` field (appended last; default 0.0 —
  positional construction unaffected); `gross_pnl` and `to_dict`
  include it.

## [2.8.5] - 2026-05-25

### Headline — test hygiene + factor LOW patches

Eight deliverables grouped under "test hygiene + factor LOW" plus a
carry-forward reverse-lock tightening from v2.8.2. The engine
zero-diff invariant holds: `engine.py`, `broker.py`, `portfolio.py`,
`orders.py`, `fees.py`, `hooks.py`, `results.py`, and `performance.py`
are untouched.

### Added

- L1: `tests/_helpers/ohlcv.py` with 3 shared OHLCV constructors
  (`make_ohlcv`, `make_close_only`, `make_ohlcv_from_closes`); 10
  module-level fixtures migrated.
- L2: `tests/test_engine_edge_cases.py` pinning N=1 bar behavior,
  all-NaN signal no-trades, and constant-drift closed-form total
  return.
- L5: `test_running_sum_no_drift_at_100k_bars` in
  `tests/test_incremental_rolling_mean.py` (rtol=1e-9, atol=1e-6).
- L7: meta-test
  `tests/test_probes_private_symbols_test_only.py` asserting the 3
  probes private helpers stay test-internal; module docstring tag in
  `src/aiphaforge/probes/__init__.py`.
- Reverse-lock: `_ALLOWED_FOREIGN_ORIGINS` constant in
  `tests/test_v2_8_public_api_lock.py`; two regression guards
  (`test_reverse_lock_allowlist_matches_audit_script`,
  `test_reverse_lock_other_category_remains_zero`).

### Changed

- L3: `src/aiphaforge/alpha/signal_analysis.py` appended to the R11
  alpha-firewall path list in `tests/test_alpha_screener.py`.
- L4: seed-transition cliff comment in
  `tests/test_incremental_rsi.py` rewritten to read "period-2,
  period-1, period" (matches the 3 existing boundary tests).
- L6 (**behavior change**): `AlphaScreener._check_columns` raises
  `TypeError` (was `AttributeError`) when `factor` or `prices` is not
  a `pd.DataFrame`. The error message names the offending argument
  and the actual type.
- Reverse-lock: rule 2 of
  `test_module_no_accidentally_public_symbols` requires
  `inspect.isclass(val) or inspect.isfunction(val)` (or a
  typing/__future__ allowlist hit) before short-circuiting on
  foreign `__module__`. Foreign-class-instance leaks now fall
  through to the lock assertion.

### Removed

- `tests/test_v2_8_2_deferred_test_hardening.py::test_reverse_lock_currently_passes_foreign_class_instances`
  — the v2.8.2-era pin no longer applies; v2.8.5 actively catches
  the case the pin documented.

### Notes

- ruff: no new rules; `ruff check src/` passes CLEAN at v2.8.5.
- pytest: net +8 across the suite (full breakdown in the README v2.8.5
  release notes section).
- mypy: scoped-blocking `mypy src/aiphaforge/alpha/` still at 0
  errors. Broad `mypy src/aiphaforge/` remains advisory.
- L6 escape hatch: wrap inputs in `.to_frame()` before calling, or
  catch `TypeError` instead of `AttributeError`.

### Lockfile recipe

```bash
pip install git+https://github.com/chifongwong-coder/AIphaForge@<v2.8.5-merge-sha>
```

### What v2.8.5 does NOT include

- V2.9-S1 `aiphaforge.stats` hoist; V2.9-S2 `significance.py` split;
  V2.9-S3 `set_factor_provider` + `IncrementalSMA`/`IncrementalEMA`;
  V2.9-S5 curated probes promote; mypy Phase 2; new features.

## [2.8.4] - 2026-05-24

### Headline — UX + API hygiene

Six docs-and-CI items grouped as "UX + API hygiene" plus one carve-out
from v2.8.3.  The engine source is untouched; the surface public API
is unchanged.  Single user-visible behavior change:
`parse_numeric_answer` gains a `decimal_separator: Literal["us", "eu"]`
kwarg that defaults to `"us"` (preserves v2.8.3 behavior verbatim).

### Per-M one-liner

| M | What changed |
|---|---|
| M11 | v2.8.1 / v2.8.2 / v2.8.3 release notes extracted from `README.md` into this `CHANGELOG.md` (Keep-a-Changelog 1.1.0 format).  README keeps thin anchor-preserving stubs. |
| M11b | `USStockFeeModel.__repr__` parameterizes SEC FY2026 and FINRA 2026 schedule labels via two module-level constants in `fees.py`.  Default repr text unchanged. |
| M12 | New "Pick Your Entry Point" decision table in README listing all six input-shape `BacktestEngine.set_*` setters with shape, when-to-use, and mutual-exclusion cells.  Parity test pins setter coverage. |
| M13 | Three new Quick Start sub-sections: factor research (AlphaScreener + FactorReport + FactorRuleStrategy), hook-driven order submission (BacktestHook.on_pre_signal + context.broker.submit_order), and `knowledge_check` orchestrator.  Parity test pins symbol imports. |
| M14 | New mypy step in `.github/workflows/ci.yml` — advisory broad `mypy src/aiphaforge/` (continue-on-error: true) + scoped-blocking `mypy src/aiphaforge/alpha/` (no continue-on-error; gates the currently-clean subpackage). |
| M15 | `parse_numeric_answer` gains `decimal_separator: Literal["us", "eu"] = "us"`.  Default `"us"` preserves v2.8.3 behavior (including known silent-wrong outputs).  Under `"eu"`: comma=decimal, period=thousands.  Bracketed ranges accept both `;` (canonical) and `,` (warns).  v2.8.3 M7 broad warn-shim removed under default. |

### URL changes in v2.8.4

- README.md anchors `#v281`, `#v282`, `#v283` still resolve (the h2 stubs are preserved); each stub links to the canonical CHANGELOG.md anchor.
- Sub-anchors inside the v2.8.3 block (e.g., `### Lost-data playbook for v2.8.2 ContinuationProbe users`) are preserved verbatim inside CHANGELOG.md so existing deep-links continue to resolve inside the new file.

### CI engineer triage block

Upgrading from v2.8.3:

- ruff: no new rules; `ruff check src/` passes CLEAN at v2.8.4.
- pytest: +18 tests; total 1762 -> 1780.
- mypy: new advisory step `mypy src/aiphaforge/` (non-blocking); new scoped-blocking `mypy src/aiphaforge/alpha/` (blocking, 0 errors at v2.8.4 release).
- AttestedAnswers users re-attest against the new release string (`__version__ == '2.8.4'`).
- **MED-E silent-shape caveat**: under `decimal_separator="eu"`, the input `"[1,234, 5,678]"` (US thousands-separator pair) parses as `(1.234, 5.678)` — a 1000x smaller pair than the US default `(1234.0, 5678.0)`.  A UserWarning IS emitted (R5), but pick the locale matching your inputs to avoid the trap.

### Lockfile-pin recipe

```bash
pip install git+https://github.com/chifongwong-coder/AIphaForge@<v2.8.4-merge-sha>
```

### What v2.8.4 does NOT include

- Reverse-lock skip-rule tightening (v2.8.5).
- Test fixture consolidation (v2.8.5 L1).
- `aiphaforge.stats` neutral-primitives module (v2.9).
- `significance.py` subpackage split (v2.9).
- `IncrementalSMA` / `IncrementalEMA` (no committed milestone).
- mypy Phase 2 (per-file allowlist + blocking on top modules) — v2.9.
- v3.0 LLM / AI factor mining (separate major track).

## [2.8.3] - 2026-05-24

### Headline — LLM-pillar diagnostic patches + a fabricated-roadmap retraction

v2.8.3 ships five MEDIUM-severity LLM-pillar patches (M6-M10),
corrects a fabricated v2.9 design claim in
`MetaContext.adjust_strategy_params`, and clears the v2.8.2
Commit I deferred-items backlog. No engine source change beyond
documentation in `fees.py` and the `meta.py` docstring; the
public API surface and `BacktestResult` field set are unchanged.

### Per-M one-liner

| ID | File | Symptom |
|----|------|---------|
| M6 | `probes/anchors.py:_block_bootstrap_corr_ci` | Floor-division on `n // block_size` left bootstrap samples short by up to `block_size-1` elements when `n` was not a multiple. Now uses ceiling division so the concatenated sample (after the `[:n]` slice) matches the original length exactly. Leverage-corr CI / SE estimates may shift by ~1-3% on unaligned windows. |
| M7 | `probes/scoring.py:parse_numeric_answer` | European decimal-comma inputs (`"1,5"`, `"12,50"`) previously tokenized to two numbers and returned `None` under the default `permissive=False` — a silent loss. v2.8.3 emits a `UserWarning` naming the offending substring + suggesting `str.replace(',', '.')`. Parser output is intentionally unchanged. |
| M8 | `probes/_vol.py:estimate_sigma` | Parkinson H==L fallback to `stdev_returns` still triggers at fraction ≥ 50%. The new `[0.4, 0.5)` warning band adds an `h_eq_l_warning_band` provenance entry (with closed-form bias estimate) without changing the chosen estimator — surfaces borderline windows for audit. |
| M9 | `probes/orchestrator.py:looks_like_refusal` | Leading-window keyword scan widened 50 → 80 characters. Captures refusals that arrive after a longer preface — e.g. when a reply opens with two filler clauses before the refusal keyword. Concrete: the prefix `"The price was approximately 100. The price was approximately 100. "` is 66 chars, so a refusal keyword (`"i don't know"`) at position 66 sits past v2.8.2's 50-char window but inside v2.8.3's 80-char window. False-positive class from in-quote refusal phrases past position 80 is regression-pinned. |
| M10 | `probes/orchestrator.py:KnowledgeCheckReport` | New `persistence_validity: Literal["NO_PERSISTENCE", "OK", "PAIRING_FAILED", "UNKNOWN"]` field. Mirrors `anchor_validity`. When ≠ `OK`, `real_minus_persistence_bucket_delta`, `real_vs_persistence_sign_test_p`, **and `persistence_caveat`** are suppressed to `None`. Legacy pickles (v2.8.2 and earlier) backfill via `__setstate__`: `persistence_baseline_score is not None` → `"UNKNOWN"`; otherwise → `"NO_PERSISTENCE"` (the latter case covers Knowledge / RankContinuation pickles for which no baseline ever existed). |
| meta.py docstring | `meta.py:MetaContext.adjust_strategy_params` | Removes the fabricated "designed jointly with v2.9 IncrementalFactor" claim. The performance note now reflects the actual current behavior (O(N·K) full-timeline regeneration) and an honest "on the roadmap, no committed milestone" pointer. |

### M10 pickle migration recipe

v2.8.2 pickles do not carry `persistence_validity`. The new
`__setstate__` backfills the field **conditionally** with no
warning, so existing load sites continue to work:

- If the original report had a persistence baseline
  (`persistence_baseline_score is not None`, i.e. the probe was a
  `ContinuationProbe`): backfill to `"UNKNOWN"`. The legacy pickle
  cannot distinguish a trivial-OK pairing from a `PAIRING_FAILED`
  one (both produced `real_vs_persistence_sign_test_p = 1.0` via
  `sign_test_p(0, 0) = (1.0, "trivial_n_zero")`), so re-run the
  orchestrator to recover the real `OK` / `PAIRING_FAILED` status.
- If no persistence baseline was ever computed (`KnowledgeProbe`
  or `RankContinuationProbe`): backfill to `"NO_PERSISTENCE"`.
  There is nothing to be "unknown" about; no re-run is needed.

```python
import pickle
from aiphaforge.probes.orchestrator import KnowledgeCheckReport

with open("v2.8.2_report.pickle", "rb") as f:
    report = pickle.load(f)
    # ContinuationProbe pickle → persistence_validity="UNKNOWN"
    # Knowledge / RankContinuation pickle → "NO_PERSISTENCE"

if report.persistence_validity == "UNKNOWN":
    # Legacy ContinuationProbe pickle: cannot distinguish OK from
    # PAIRING_FAILED because sign_test_p(0, 0) returns (1.0,
    # "trivial_n_zero") under both. Re-run the orchestrator on the
    # original probe to upgrade.
    ...
elif report.persistence_validity == "NO_PERSISTENCE":
    # Knowledge / RankContinuation pickle — never had a persistence
    # baseline. No upgrade needed; downstream stats already None.
    ...
```

Rationale: the v2.8.2 `real_vs_persistence_sign_test_p` field
silently collapsed both `OK` (paired but no signal) and
`PAIRING_FAILED` (no qid overlap → empty pair list → `(0, 0)`)
into the same `1.0` p-value. Consumers reading a v2.8.2 pickle
have no way to tell which case produced the `1.0`. The `UNKNOWN`
tag flags exactly this ambiguity so users can choose to re-run
rather than treat the legacy `1.0` as a clean null result.

### M7 honesty paragraph

If you submitted European-format numeric inputs to
`parse_numeric_answer` in v2.8.2 (single scalar like `"1,5"`),
the parser returned `None` and your `parse_status` showed
`"invalid"`. There was no warning. Range/bracket paths that
contained a comma-decimal token were similarly mis-tokenized.
v2.8.3 emits a `UserWarning` so the silent loss surfaces in
test logs. Full locale support (parsing European decimals
correctly given an explicit `locale=` argument) is on the
v2.8.4 M15 roadmap; v2.8.3 is warning-only.

### Lost-data playbook for v2.8.2 ContinuationProbe users

For paper authors who used v2.8.2 `ContinuationProbe` pickles in
published work AND cannot re-run (raw data deleted, dataset
embargo, post-publication audit, etc.):

**Suggested citation wording**:

> This work used AIphaForge v2.8.2 for the persistence-baseline
> analysis. v2.8.3 added a `persistence_validity` field to
> `KnowledgeCheckReport` that retroactively cannot distinguish
> `OK` from `PAIRING_FAILED` for v2.8.2-saved pickles (per the
> AIphaForge v2.8.3 release notes, M10). Where this paper reports
> `real_minus_persistence_bucket_delta` or
> `real_vs_persistence_sign_test_p` from a v2.8.2-loaded report,
> the underlying `persistence_validity` is `UNKNOWN`, and results
> should be treated as having unknown pairing-success state.

**Retraction / correction guidance**:

- If your published claim used `real_minus_persistence_bucket_delta`
  as evidence of a specific pairing state, and you cannot re-run:
  publish a correction or erratum stating the underlying validity
  is `UNKNOWN` per v2.8.3 release notes M10.
- If your published claim used `real_vs_persistence_sign_test_p`
  for a hypothesis test: same — note the underlying validity is
  `UNKNOWN`. The `1.0` p-value cannot distinguish trivial-OK from
  pairing-failure.
- If your published claim used only `bucket_delta` magnitudes
  without explicit pairing-success assertions: a footnote
  disclosing the v2.8.2 → v2.8.3 ambiguity is sufficient (no
  retraction needed).

### CI engineer triage block

Expected failure modes when CI re-baselines on v2.8.3, in order
of likelihood:

1. **M6 leverage-corr CI / SE shifts** — bootstrap samples on
   unaligned windows are now full-length. CI half-widths and SEs
   may move by ~1-3% on the affected windows.
2. **M7 European-pattern warnings** — fixtures containing
   `"\d,\d"` or `"\d,\d\d"` patterns now emit `UserWarning`.
   `pytest.warns` blocks may need to widen, or input fixtures
   need pre-localization via `str.replace(',', '.')`.
3. **M8 H==L `[0.4, 0.5)` warnings** — windows in the new
   warning band gain an extra provenance entry
   (`h_eq_l_warning_band`); test fixtures that snapshot the
   provenance dict shape need updating.
4. **M9 refusal_rate metric up for borderline fixtures** — if
   your fixtures contain refusal-shaped phrases starting in the
   character-position 50-79 band, `looks_like_refusal` now
   returns `True` where it previously returned `False`.
   Downstream `refusal_rate` aggregates rise; `effective_rate`
   falls correspondingly.
5. **M10 `persistence_validity` field on
   `KnowledgeCheckReport`** — pickled reports gain a new
   field; consumers using `dataclasses.fields()` to enumerate
   surface need updating. The `to_dict()` method already covers
   the field automatically.
6. **M3 dict-path validation respects `data_validation="none"`**
   — regression-pin only, no code change. The v2.8.2
   `set_signals` dict path already honors the
   `data_validation="none"` escape hatch (the outer guard at
   `engine.py:453` covers both single-Series and dict branches).
   v2.8.3 adds a test that pins this behavior so a future
   refactor that accidentally narrows the guard would surface
   immediately.

### Reverse-pickle caveat

v2.8.3 → v2.8.2 reverse pickle is NOT supported. The new
`persistence_validity` field has no `__init__` translation back
to the v2.8.2 dataclass; a v2.8.3 pickle loaded under v2.8.2 will
raise `TypeError: KnowledgeCheckReport got unexpected keyword
arguments: ['persistence_validity']`. This is per the v2.8.1 H7
forward-only contract.

### Lockfile-pin recipe

```bash
pip install \
  git+https://github.com/chifongwong-coder/AIphaForge@<v2.8.3-merge-sha>
```

The merge SHA is set when v2.8.3 lands on `main`; until then, pin
to the feature branch via
`@feature/v2.8.3-llm-medium`.

### Deprecation removal commitment

`bucket_delta_tango_ci` legacy alias remains scheduled for hard
removal in v2.9 (unchanged from v2.8.1+v2.8.2). v2.8.3 does not
move the schedule.

### What v2.8.3 does NOT include

- Full locale support for European decimal-comma parsing (M7 is
  warning-only; locale support is on the v2.8.4 M15 roadmap).
- Reverse-lock skip-rule tightening (v2.8.5).
- `IncrementalSMA` / `IncrementalEMA` / partial-timeline
  regeneration (no committed milestone — see the corrected
  `MetaContext.adjust_strategy_params` docstring).
- `__pickle_version__` constant on `KnowledgeCheckReport` (does
  not exist; out of scope for v2.8.3).
- `dataclasses.asdict` for `KnowledgeCheckReport` serialization
  (use `report.to_dict()` instead — `asdict` rejects the
  `MappingProxyType`-wrapped fields).


## [2.8.2] - 2026-05-21

### Headline — parallel-backtest users: your strategy instance is mutated in place

If you used `MetaContext.adjust_strategy_params` with a shared
strategy instance across parallel backtest workers
(`multiprocessing.Pool`, threading, or a simple loop), the
adjustment from one worker leaked to ALL other workers holding
the same reference. v2.8.2 documents this in the
`adjust_strategy_params` docstring + ships the
factory-per-worker recipe.

```python
# WRONG — shared instance leaks adjustments across workers:
strategy = MACrossover(short=5, long=20)
with Pool() as pool:
    pool.map(lambda d: run(d, strategy), datasets)

# RIGHT — factory per worker isolates state:
def make_strategy():
    return MACrossover(short=5, long=20)
with Pool() as pool:
    pool.map(lambda d: run(d, make_strategy()), datasets)
```

If your historical v2.8.1-or-earlier results came from the
"shared instance" pattern with mid-run `adjust_strategy_params`
calls, those results are cross-contaminated and should be
re-run with the factory pattern. **There is NO runtime
detection in v2.8.2** — if you upgrade via dependabot / auto-pin
without reading these release notes, silent contamination from
prior runs persists. You must audit your parallel-backtest
pipelines and switch to the factory pattern manually.

**Performance note — `adjust_strategy_params` regen cost**: every
call re-runs the strategy from bar 0 (O(N·K) for N bars × K
adjustments). Callers performing many small adjustments on a long
timeline should batch them where possible. A partial-timeline
(incremental) regeneration path is on the roadmap; no committed
milestone yet. *(The v2.8.2 release notes initially announced a
v2.8.3 ship "designed jointly with a v2.9 IncrementalFactor
engine integration" — there is no IncrementalFactor module or
v2.9 plan, and v2.8.3 corrects this fabrication in both the
`MetaContext.adjust_strategy_params` docstring and these release
notes. See v2.8.3 Commit H.)*

### CI engineer triage block

Five expected failure modes when CI re-baselines on v2.8.2, in
order of impact:

1. **US-stock sell-side cost shift (M2)** — `USStockFeeModel`
   sell commission rises by ~$2.3 per $100k due to FY2026 SEC §31
   + 2026 FINRA TAF defaults now applying. Buy-side unchanged.
   Golden fixtures with US-stock sell-heavy strategies will
   rebaseline. Opt out (e.g., pre-2003 backtests):
   `USStockFeeModel(sec_fee_rate=0, finra_taf_per_share=0)`.
2. **`set_signals` validation (M3)** — duplicate-index Series,
   non-`DatetimeIndex` Series, and non-numeric dtype now raise at
   `set_signals` time. Tests that worked silently in v2.8.1 with
   these shapes now fail at the boundary instead of crashing
   later. See "M3 rejection set" below for the enumerated cases.
   Escape hatch: `BacktestEngine(data_validation="none")`.
3. **Parallel-backtest contamination (M4)** — covered in the
   headline above. No automatic detection ships in v2.8.2; the
   migration is to switch to the factory-per-worker pattern.
4. **`cost_normalization` opt-in (M5)** — `DefaultTradeCost` now
   accepts `cost_normalization="current_equity"` (default
   `"initial_capital"` preserves v2.8.1). Decision rubric below.
5. **`PercentageStopLoss` divergence pin (M1)** — no behavior
   change; new regression test pins existing event-driven /
   vectorized divergence magnitude on a gap-bar fixture. CI
   suites mirroring that fixture pattern may need to update
   their expected divergence range. **Note for live-vs-backtest
   users**: vectorized PnL is OPTIMISTIC vs event-driven on
   gap-down triggers; if you tune thresholds against vectorized
   and run live event-driven, expect realized worse fills.
6. **New tests + test regressions fixed (Commit F + v2.8.1
   deferred items)** — 9 new test files / additions land in
   v2.8.2 (M1-M5 + 4 v2.8.1 follow-up pins). 2 pre-existing tests
   were updated to honor M3's stricter `set_signals` validation
   (`test_e2e.py::test_empty_data_raises`,
   `test_v2_8_1_h2_dup_index.py::test_engine_event_driven_rejects_dup_data_at_validation_not_loop`).
   A CI workflow running `pytest --collect-only` diff will see
   these new entries; no behavior change in the production code.

### Per-M one-liner

| ID | File | Symptom |
|----|------|---------|
| M1 | `exit_rules.py:PercentageStopLoss` | Event-driven submits market order (next-bar open fill); vectorized exits at theoretical threshold on trigger bar. Documents-only fix; alignment requires next-bar OHLC plumbing into vectorized (v2.9). |
| M2 | `fees.py:USStockFeeModel` | Sell side now includes SEC §31 (FY2026 `20.60e-6`) + FINRA TAF (2026 `0.000195/share, cap $9.79`). Buy-side unchanged. Opt out with zeros. |
| M3 | `engine.py:set_signals` | Single-Series + per-symbol-dict path now calls `validate_signal_series`. Respects `data_validation="none"`. |
| M4 | `meta.py:MetaContext.adjust_strategy_params` | In-place mutation documented; factory-per-worker recipe ships. Performance note: each call re-runs the strategy from bar 0 (O(N·K)). A partial-timeline regeneration path is on the roadmap, no committed milestone. *(v2.8.2 originally promised a v2.8.3 ship "designed jointly with v2.9 IncrementalFactor"; this claim was fabricated — there is no IncrementalFactor engine integration plan. v2.8.3 Commit H retracted the claim in the meta.py docstring; this row supersedes the v2.8.2 row text.)* |
| M5 | `costs.py:DefaultTradeCost` | Opt-in `cost_normalization="current_equity"`. Default unchanged. First-order approximation (uses pre-cost gross returns). |

### M3 rejection set

`set_signals` now rejects these Series shapes at the boundary
(authoritative source: `tests/_helpers/expected_rejections.md`):

- Non-`DatetimeIndex` (e.g. `RangeIndex`, `Int64Index`, `MultiIndex`) → `TypeError`
- Duplicate `DatetimeIndex` timestamps → `ValueError`
- Non-numeric dtype (e.g. `object`, `string`) → `TypeError`

These shapes PASS default validation (NOT rejected):
- All-NaN Series
- Out-of-order timestamps
- Empty Series
- Fractional values (when `allow_fractional=True`, the default)

Escape hatch: `BacktestEngine(data_validation="none")` skips boundary
validation for parity with the `validate_ohlcv` convention.

### M2 era-specific overrides

Recipes for backtests outside the FY2026 default era:

```python
# Pre-2003-04-01 (no NASD/FINRA TAF; fee introduced 2003-04):
# FINRA TAF was originally the NASD TAF, renamed in 2007 when
# FINRA was formed from NASD + NYSE Regulation. Same fee math.
fee = USStockFeeModel(finra_taf_per_share=0)

# Pre-1971 (no SEC §31; fee introduced 1971):
fee = USStockFeeModel(sec_fee_rate=0)

# Cross-boundary backtest spanning the FY2026-2 effective date
# (2026-04-04): SEC §31 jumped from 8.00e-6 to 20.60e-6.
# Run two segments with separate USStockFeeModel instances and
# concatenate equity curves; a single scalar can't express both.
# NOTE: this is a CALLER-side recipe — the engine has no in-engine
# path for switching fee model mid-run. Split your data, run two
# BacktestEngine instances, then concatenate the equity curves
# manually.
fee_pre  = USStockFeeModel(sec_fee_rate=8.00e-6)   # 2025-10-01..2026-04-03
fee_post = USStockFeeModel(sec_fee_rate=20.60e-6)  # 2026-04-04..present
```

For historic rates outside v2.8.2's defaults, consult the SEC Fee
Rate Advisory archive and the FINRA Trading Activity Fee schedule.

### M5 decision rubric

When to use which `cost_normalization`:

- **`"initial_capital"` (default)** — path-independent cost
  reporting (most use cases); your strategy stays within ±50%
  of starting capital; you compare across strategies and want
  apples-to-apples cost figures.
- **`"current_equity"`** — backtest has > 50% expected drawdown;
  you compute Sharpe / max-drawdown over deep-loss regimes; you
  need cost ratio to reflect realized equity at each bar.

The `"current_equity"` mode is a first-order approximation
(`running_equity` uses gross pre-cost returns). The bias under-
states cost; magnitude grows with turnover × horizon. The
iteratively-correct version is a v2.9 follow-up.

### Lockfile-pin

```bash
# v2.8.1 (previous release):
pip install git+https://github.com/chifongwong-coder/AIphaForge@5862eb7

# v2.8.2 (current release): pin the immutable commit SHA below.
# Until the v2.8.2 branch is merged to main, this is the HEAD SHA
# of feature/v2.8.2-strategy-medium; after merge, swap to the
# merge-commit SHA from `git log main` for the immutable reference.
# Avoid pinning to a branch name — branch names are mutable.
pip install git+https://github.com/chifongwong-coder/AIphaForge@2240d2b
```

### Deprecation removal commitment

`tango_paired_diff_ci` alias + `bucket_delta_tango_ci` kwarg
hard-removal stays on v2.9 schedule (unchanged from v2.8.1).

### What v2.8.2 does NOT include

- **M4 perf optimization** — full-timeline regen on every
  `adjust_strategy_params` call. A partial-timeline regeneration
  path is on the roadmap; no committed milestone.
  *(v2.8.2 originally cited a v2.8.3 ship "designed jointly with
  v2.9 IncrementalFactor engine integration so both consumers
  share a single API"; this paragraph was fabricated. v2.8.3
  Commit H + the v2.8.3 release notes retract the claim. This
  bullet supersedes the v2.8.2 original.)*
- **LLM probe MEDIUMs** (anchors, orchestrator, scoring) —
  v2.8.3.
- **UX + API hygiene** (CHANGELOG, mypy CI) — v2.8.4.
- **Test fixture consolidation + factor LOWs** — v2.8.5.
- **Architectural items** (`significance.py` split, neutral
  primitives module) — v2.9 (the final v2.x release).
  *(v2.8.2 originally included "IncrementalFactor engine
  integration" in this list; that integration is not on a v2.9
  plan and was retracted in v2.8.3.)*
- **v3.0 LLM/AI factor mining** — separate major track.

---


## [2.8.1] - 2026-05-20

### Two silent-correctness bugs are fixed. Your historical results may have been wrong.

**Headline 1 — vectorized US-stock equity curves WILL improve after
upgrade (H1).** v2.8.0's `DefaultTradeCost.apply_vectorized` called
`fee_model.estimate_commission_rate()` with no arguments, getting the
default `(price=100, size=100)` pair. For `USStockFeeModel` with the
`min_commission=$1` floor, that produced a ~1% trade cost — about
100× the real per-trade rate. v2.8.1 derives a representative
`(price, size)` from your `position_sizer + initial_capital`. Cost
drag drops from inflated ≈1% to correct ≈0.01-0.1%. **This is a bug
fix, not a behavior change in your strategy.** Caveat for live
traders: if you tuned thresholds against the inflated backtest cost,
your live edge may now appear *worse* than your new backtest — your
strategy was implicitly over-paying for the wrong reason.

**Headline 2 — vectorized multi-asset PnL with a `capital_allocator`
was silently wrong (H4).** The vectorized core ignored
`capital_allocator` and ran static equal-weight; the event-driven
core honored it. Users running vectorized multi-asset backtests with
a custom allocator got PnL that did NOT match the equivalent
event-driven run. v2.8.1 does NOT implement dynamic allocation in
vectorized (that's v2.8.2+); instead, it emits a loud `UserWarning`
naming your allocator class and pointing at `mode='event_driven'`.
**If you ran vectorized multi-asset with a non-trivial
`capital_allocator` in v2.8.0, your historical backtest PnL is wrong
and you should re-run on event-driven to get the right numbers.**

### Lockfile-pin for academic reproducibility of v2.8.0-pinned runs

AIphaForge is not currently published to PyPI. For exact
reproducibility of a v2.8.0-pinned analysis, pin the git commit SHA
directly:

```bash
pip install git+https://github.com/chifongwong-coder/AIphaForge@fd4b34f
```

```text
# requirements.txt
aiphaforge @ git+https://github.com/chifongwong-coder/AIphaForge@fd4b34f
```

Commit `fd4b34f` is the v2.8.0 release commit (`__version__ ==
'2.8.0'`). If you cited a v2.8.0 result in a paper, this is the SHA
to reference.

### CI engineer triage block

Seven expected failure modes when CI re-baselines on v2.8.1, listed
in order of impact (silent-correctness bugs first, then noisy breaks,
then opt-in `-W error` breaks):

1. **Equity curve drift (H1)** — most common. Cost rate fixed;
   rebaseline golden fixtures for any single-asset vectorized run.
2. **Multi-asset PnL drift (H4)** — silent bug; previously wrong
   numbers. If your golden fixtures cover vectorized multi-asset
   with a `capital_allocator`, decide whether to (a) accept the new
   `UserWarning` and re-baseline against event-driven, or (b) switch
   the fixture to `mode='event_driven'` outright.
3. **Duplicate-timestamp fixture `ValueError` (H2)** —
   `validate_ohlcv` now hard-fails on duplicate index regardless of
   `validation_level`. Fix the data, not the test:
   `df = df[~df.index.duplicated(keep='first')]`.
4. **New vectorized warnings break `-W error` runs (H3 + H4)** —
   v2.8.1 expanded `_VECTORIZED_UNSUPPORTED_FIELDS` from 7 to 21.
   Setting any of `fill_model`, `session_end_time`,
   `immediate_fill_price`, `fee_allocation`, `capital_allocator`,
   `lot_size`, `max_position_pct`, the multi-asset `asset_*` dicts,
   etc. on a `vectorized` engine now warns. Under `pytest -W error`
   these become exceptions. Either move to `event_driven` or relax
   the warning filter for the affected modules.
5. **`DeprecationWarning` as error (H6)** — code that still imports
   `tango_paired_diff_ci` (renamed to `wald_paired_diff_ci`) emits a
   `DeprecationWarning`. Under `-W error` this is a hard fail.
   Migrate the import or relax the filter.
6. **Pickle bytes-hash pinning (H7)** —
   `KnowledgeCheckReport.__getstate__` changes dict shape. Any
   caller pinning the pickle bytes-hash needs re-pinning.
7. **ABC additive break** — external subclasses of `BaseTradeCost`
   that override `apply_vectorized` with the v2.8.0 (kwargs-less)
   signature will `TypeError` on the first vectorized run:
   `TypeError: apply_vectorized() got an unexpected keyword argument 'representative_notional'`.
   Fix in your override: add
   `*, representative_notional=None, representative_size=None`
   (or `**_kwargs`) to the signature. Forward both kwargs if you
   call `super().apply_vectorized(...)` from your subclass — silently
   dropping them re-introduces the H1 bug at the call site. No
   fallback shim ships in v2.8.1 per the v2.8.x "no compat flag"
   precedent.

### Per-H one-liner

| ID | File | Symptom |
|----|------|---------|
| H1 | `costs.py:DefaultTradeCost.apply_vectorized` | Vectorized US-stock cost over-billed ≈100× via no-args `estimate_commission_rate()`. |
| H2 | `utils.py:validate_ohlcv` | Duplicate-timestamp OHLCV slipped past `warn` mode and crashed event-driven mid-loop. |
| H3 | `engine.py:_VECTORIZED_UNSUPPORTED_FIELDS` | 14 fields silently dropped by vectorized; no warning surfaced them. |
| H4 | `engine.py:_warn_vectorized_capital_allocator_divergence` | Vectorized multi-asset ignored `capital_allocator`; PnL silently diverged from event-driven. |
| H5 | `probes/anchors.py:_build_ohlcv_from_returns` | Anchor H/L hardcoded ±1.5% → Parkinson/GK vol estimates were deterministic noise. Synthetic spread ratio now bar-for-bar matches real; Parkinson is approximately (not exactly) equal — see docstring. |
| H6 | `probes/orchestrator.py:tango_paired_diff_ci` | Function named after Tango (1998) but body is Wald — caused mis-citation. |
| H7 | `probes/orchestrator.py:KnowledgeCheckReport` | `MappingProxyType` fields broke pickle round-trip; multiprocessing pipelines crashed. |
| H8 | `tests/test_v2_8_public_api_lock.py` | v2.8 lock was one-way; symbols without `_` prefix slipped public. |

### Anchor-probe users (H5): your reported Parkinson values will change

Anchor-side Parkinson `(ln(H/L))^2` and Garman-Klass intra-bar vol
estimates in v2.8.0 were deterministic functions of the constant
±1.5% spread the helper hardcoded — not anything tied to your real
symbol. v2.8.1 derives synthetic H/L from the real bar's
`(H - L) / close` ratio per timestamp, so the anchor's vol stats now
reflect the real bar. **Numerical impact**: any leakage-test
sensitivity calibrated on v2.8.0 Parkinson values needs
re-calibration. The synthetic spread RATIO is bar-for-bar identical
to the real spread; Parkinson is approximately equal (the synthetic
centers H/L symmetrically around close, which a real bar generally
does not — typical relative error < 1% at spread ≤ 5%, see the
`_build_ohlcv_from_returns` docstring).

### Breaking changes + migration recipes

- **H2** dedupe: `df = df[~df.index.duplicated(keep='first')]`
- **H6** rename — GNU sed:
  `sed -i 's/tango_paired_diff_ci/wald_paired_diff_ci/g' your_file.py`
  BSD/macOS:
  `sed -i '' 's/tango_paired_diff_ci/wald_paired_diff_ci/g' your_file.py`
- **H7** pickle: old v2.7.x pickles carrying `bucket_delta_tango_ci`
  load with a `DeprecationWarning` (the translation now fires on
  both `__init__` and pickle restore via `__setstate__` — v2.8.1
  Commit J fix). Re-save with `pickle.dump(report, ...)` after a
  clean load to silence.
- **H7** kwargs-only: `KnowledgeCheckReport` accepts ONLY keyword
  arguments since v2.8.1. Positional construction raises `TypeError`.
  The dataclass has 24 required fields plus several optional defaults;
  the skeleton below is illustrative — copy-paste will raise
  `TypeError: missing required argument 'paired_sign_test_n_positive'`
  (or similar) until every required field is supplied. Inspect
  `dataclasses.fields(KnowledgeCheckReport)` for the authoritative
  list:
  ```python
  # v2.8.0 (worked, no longer):
  # KnowledgeCheckReport("knowledge", real_score, anchor_score, ...)

  # v2.8.1+ (skeleton — supply ALL 24 required fields):
  KnowledgeCheckReport(
      probe_kind="knowledge",
      real_score=real_score,
      anchor_score=anchor_score,
      bucket_delta=bucket_delta,
      paired_sign_test_p=sign_test_p_val,
      # ... other ~14 required fields by keyword ...
      anchor_validity="OK",
      parsing_schema_hash=schema_hash,
      parsing_schema_description=schema_desc,
      prompt_template_hash=template_hash,
      prompt_template_description=template_desc,
  )
  ```
- **H8** promoted symbols (your existing imports are now blessed):
  `serialize_answer_records`, `resolve_determinism_config`.

### Advanced knobs you might not have noticed

- `BacktestEngine(representative_notional=...)` and / or
  `BacktestEngine(representative_size=...)` — override the engine-
  derived cost-estimation values. Engine default is
  `initial_capital * min(sizer.fraction, max_position_size)` for
  `FractionSizer` / `AllInSizer`, or `sizer.size` for `FixedSizer`.
  Either or both kwargs can be passed independently — whatever you
  pass wins; the engine fills only the unset side from the sizer.
- `build_synthetic_anchor(..., hl_spread_source="real_distribution_shuffled")`
  — opt out of bar-by-bar real-spread propagation. Permutes the real
  spread ratios via a seed-derived RNG; destroys bar-level
  autocorrelation while preserving the marginal distribution. Useful
  for verifying "no per-bar leak" via `autocorr(spread) ≈ 0`. True
  symbol anonymization still goes through `SymbolMasker`.

### Deprecation removal commitment

`tango_paired_diff_ci` and the `bucket_delta_tango_ci` legacy kwarg
to `KnowledgeCheckReport` ship with `DeprecationWarning` in v2.8.1.
Both **hard-remove in v2.9** — the next minor version after the
v2.8.x patch series. If you maintain a downstream that pins
`tango_paired_diff_ci` import, migrate before bumping past v2.8.x.

### What v2.8.1 does NOT include

v2.8.1 is the **HIGH-severity batch only**. MEDIUM-severity follow-
ups (strategy / LLM / UX-API / test-factor housekeeping) ship across
v2.8.2 – v2.8.5; architectural work (`IncrementalFactor` engine
integration, `significance.py` split, neutral primitives module)
lands in v2.9 (the final v2.x release). A separate v3.0 track is
reserved for LLM/AI factor mining.

