"""A/B probe runner for the v2.0 memory-probe toolkit.

Runs an AI agent and a comparable baseline through a panel of
transform scenarios, computes per-metric absolute and relative
drops, the diff-in-diff ``excess_drop`` series, and (optionally)
an AI-on-AI noise control to separate transform-induced sensitivity
from generic LLM brittleness.

The runner is purely descriptive — no p-values, no verdicts. It
reports distributions, low-anchor counts, comparability warnings,
and a manifest. Interpretation is the user's responsibility.

See `docs/plans/v2.0-plan.md` §3 for the full spec, including the
populated/None invariant for ``MetricDropSummary`` and the manifest
schema.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from aiphaforge.engine import BacktestEngine
from aiphaforge.hooks import BacktestHook
from aiphaforge.probes.models import (
    ABProbeResult,
    ABScenario,
    AgentContract,
    MetricConfig,
    MetricDropSummary,
    ScenarioABReport,
)
from aiphaforge.probes.transforms import TransformPipeline
from aiphaforge.results import BacktestResult
from aiphaforge.strategies import (
    BaseStrategy,
    MACrossover,
    MeanReversionBollinger,
    MomentumRank,
)

# ---------- Defaults ----------

DEFAULT_METRICS: tuple[str, ...] = (
    "total_return",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "trade_count",
)


# Per-metric config. Defaults are calibrated for daily-frequency
# strategies on liquid US equities; users running at other
# frequencies / asset classes should override.
DEFAULT_METRIC_CONFIG: Dict[str, MetricConfig] = {
    "total_return": MetricConfig(
        higher_is_better=True,
        normalization_floor=1e-3,
        low_anchor_threshold=0.01,
    ),
    "sharpe_ratio": MetricConfig(
        higher_is_better=True,
        normalization_floor=1e-2,
        low_anchor_threshold=0.10,
    ),
    "max_drawdown": MetricConfig(
        higher_is_better=False,  # positive magnitude where larger is worse
        normalization_floor=1e-3,
        low_anchor_threshold=0.005,
    ),
    "win_rate": MetricConfig(
        higher_is_better=True,
        normalization_floor=1e-2,
        low_anchor_threshold=0.02,
    ),
    "profit_factor": MetricConfig(
        higher_is_better=True,
        normalization_floor=1e-2,
        low_anchor_threshold=0.10,
    ),
    "trade_count": MetricConfig(
        higher_is_better=True,
        normalization_floor=1.0,
        low_anchor_threshold=5.0,
    ),
}


# Transforms whose presence in a scenario should auto-inject a
# detectability warning into the report (per plan §2.6). These are
# transforms an LLM may detect and behaviorally react to (refuse,
# hedge, regime-confuse) regardless of whether actual leakage is
# being probed.  ``ReturnsOnlyView`` is listed in the plan but not
# yet implemented in transforms.py — kept here so the warning fires
# automatically when v2.0.x ships it.
_DETECTABLE_TRANSFORM_NAMES = frozenset({
    "SymbolMasker", "DateShift", "ReturnsOnlyView",
    "OHLCJitter", "BlockBootstrap", "WindowShuffle",
})


# Type aliases.
AgentLike = Union[BaseStrategy, BacktestHook]
AgentFactory = Callable[[], AgentLike]


# ---------- Built-in baseline strategies ----------

class MACrossBaseline(MACrossover):
    """Default trend-following baseline: 10/30 MA crossover."""

    def __init__(self, short: int = 10, long: int = 30):
        super().__init__(short=short, long=long)


class MeanRevBaseline(MeanReversionBollinger):
    """Default contrarian baseline: Bollinger mean-reversion."""

    def __init__(self, window: int = 20):
        super().__init__(window=window)


class MomentumBaseline(MomentumRank):
    """Default momentum baseline: cross-sectional rank for multi-asset,
    falls back to simple ROC > 0 = buy for single-asset.
    """

    def __init__(self, roc_window: int = 20, top_n: int = 3):
        super().__init__(roc_window=roc_window, top_n=top_n)


# ---------- Metric extraction ----------

def _extract_metric(result: BacktestResult, metric: str) -> float:
    """Pull a single metric out of a BacktestResult.

    Tries property access first (`total_return`, `num_trades`,
    `max_drawdown`, ...), then falls back to `result.metrics[metric]`.
    For ``trade_count``, returns the integer trade count as a float.
    """
    if metric == "trade_count":
        return float(result.num_trades)
    if hasattr(result, metric):
        try:
            value = getattr(result, metric)
            if isinstance(value, (int, float)):
                return float(value)
        except Exception:
            pass
    if result.metrics and metric in result.metrics:
        v = result.metrics[metric]
        return float(v) if v is not None else float("nan")
    return float("nan")


def _orient(value: float, cfg: MetricConfig) -> float:
    """Apply higher-is-better orientation."""
    return value if cfg.higher_is_better else -value


# ---------- Single-arm execution ----------

def _engine_for(engine_kwargs: Optional[Dict[str, Any]]) -> BacktestEngine:
    return BacktestEngine(**(engine_kwargs or {}))


def _run_one_arm(
    factory: AgentFactory,
    data: pd.DataFrame,
    *,
    transforms: Sequence,
    mode: str,
    seed: Optional[int],
    engine_kwargs: Optional[Dict[str, Any]] = None,
) -> BacktestResult:
    """Run a single arm of the A/B at one transform pipeline.

    For ``market_level`` mode the transformed data is the engine's
    market. For ``view_only`` mode the agent sees the transformed
    data but the engine fills at the original real prices/dates;
    this requires a Strategy-based factory in v2.0 because the
    Hook-driven broker-proxy wrapper is deferred to v2.0.x.

    Empty ``transforms`` runs the raw arm (no transformation).
    """
    agent = factory()
    engine = _engine_for(engine_kwargs)

    if not transforms:
        # Raw arm.
        if isinstance(agent, BaseStrategy):
            engine.set_strategy(agent)
        else:
            # Hook-driven agents emit orders via ctx.broker; the engine
            # still requires a primary signal source. Hand it an
            # all-NaN ("hold") series so the hook is the sole driver.
            engine.hooks.append(agent)
            engine.set_signals(
                pd.Series(np.nan, index=data.index, dtype=float)
            )
        return engine.run(data)

    pipeline = TransformPipeline(
        transforms=list(transforms), mode=mode,
    )

    if mode == "market_level":
        transformed = pipeline.apply(data, seed=seed)
        if isinstance(agent, BaseStrategy):
            engine.set_strategy(agent)
        else:
            engine.hooks.append(agent)
            engine.set_signals(
                pd.Series(np.nan, index=transformed.index, dtype=float)
            )
        return engine.run(transformed)

    # view_only mode.
    if not isinstance(agent, BaseStrategy):
        raise NotImplementedError(
            "view_only mode with a Hook-based agent requires the "
            "broker-proxy wrapper, which is deferred to v2.0.x. "
            "For v2.0, view_only is supported only for Strategy-based "
            "agents (factories returning BaseStrategy subclasses)."
        )
    view = pipeline.apply(data, seed=seed)
    signals = agent.generate_signals(view)
    # Transforms like DateShift change the index; align signals back
    # to the real-data index so the engine can fill at real bars.
    # Length must match — bar i in the view corresponds to bar i in
    # the real frame for all built-in view-only transforms (which
    # only relabel labels, never reorder rows). A different length
    # means the strategy emitted an aligned-but-trimmed series we
    # cannot safely reindex; reject explicitly.
    if isinstance(signals, pd.Series):
        if len(signals) != len(data):
            raise ValueError(
                f"view_only signal length ({len(signals)}) does not "
                f"match real-data length ({len(data)}); cannot safely "
                "reindex. Ensure the strategy returns a same-length "
                "series, or use mode='market_level'."
            )
        signals = signals.copy()
        signals.index = data.index
    elif isinstance(signals, dict):
        signals = {sym: s.copy() for sym, s in signals.items()}
        for s in signals.values():
            if len(s) != len(data):
                raise ValueError(
                    "view_only multi-asset signal length mismatch; "
                    "cannot safely reindex."
                )
            s.index = data.index
    engine.set_signals(signals)
    return engine.run(data)


# ---------- Determinism check ----------

def _check_agent_determinism(
    factory: AgentFactory,
    data: pd.DataFrame,
    *,
    seed: Optional[int],
    engine_kwargs: Optional[Dict[str, Any]],
    metric: str = "total_return",
) -> bool:
    """Run the same factory twice on identical inputs.

    Returns True if both runs produce the same metric value (within
    1e-12 relative tolerance), False otherwise. The runner cannot
    police what a stateful Hook does internally — this surfaces
    non-determinism so the user sees it as a warning rather than
    silently absorbing it into the obfuscation noise term.
    """
    try:
        r1 = _run_one_arm(
            factory, data, transforms=[], mode="market_level",
            seed=seed, engine_kwargs=engine_kwargs,
        )
        r2 = _run_one_arm(
            factory, data, transforms=[], mode="market_level",
            seed=seed, engine_kwargs=engine_kwargs,
        )
    except Exception:
        return False
    v1 = _extract_metric(r1, metric)
    v2 = _extract_metric(r2, metric)
    if np.isnan(v1) and np.isnan(v2):
        return True
    if np.isnan(v1) or np.isnan(v2):
        return False
    denom = max(abs(v1), abs(v2), 1e-12)
    return abs(v1 - v2) / denom < 1e-12


# ---------- Drop math ----------

def _compute_relative_drop(
    raw: float, test: float, cfg: MetricConfig,
) -> tuple[float, Optional[float], bool]:
    """Compute (abs_drop, rel_drop_or_None, is_low_anchor) for one repeat.

    Symmetric normalization denominator per plan §3.7:
        denom = max(|raw|, |test|, normalization_floor)
    Low-anchor flag fires before relative computation; flagged repeats
    return rel_drop = None.
    """
    o_raw = _orient(raw, cfg)
    o_test = _orient(test, cfg)
    abs_drop = o_raw - o_test
    if max(abs(o_raw), abs(o_test)) < cfg.low_anchor_threshold:
        return abs_drop, None, True
    denom = max(abs(o_raw), abs(o_test), cfg.normalization_floor)
    return abs_drop, abs_drop / denom, False


def _summary_stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


# ---------- Main runner ----------

def _hash_data(data: pd.DataFrame) -> str:
    """Stable hash of an OHLCV DataFrame for the manifest."""
    h = hashlib.sha256()
    h.update(data.values.astype(np.float64).tobytes())
    h.update(np.asarray(data.index.values).tobytes())
    h.update(",".join(map(str, data.columns)).encode("utf-8"))
    return h.hexdigest()[:16]


def _scenario_warnings(
    scenario: ABScenario,
    *,
    parity_warning: Optional[str],
    determinism_ok_ai: bool,
    determinism_ok_baseline: bool,
) -> List[str]:
    out: List[str] = []
    detectable = sorted({
        getattr(t, "name", type(t).__name__)
        for t in scenario.transforms
        if getattr(t, "name", type(t).__name__) in _DETECTABLE_TRANSFORM_NAMES
    })
    if detectable:
        out.append(
            "transform_detectability_warning: scenario uses "
            f"{detectable}, which an LLM may detect and behaviorally "
            "react to (refusal, hedging, regime confusion). Treat "
            "excess_drop as suggestive, not conclusive."
        )
    if parity_warning:
        out.append(parity_warning)
    if not determinism_ok_ai:
        out.append(
            "agent_determinism_check_failed_ai: re-running the AI factory "
            "on identical inputs produced different metric values. "
            "Stochastic agents bias paired comparisons; consider "
            "temperature=0 with a deterministic seed."
        )
    if not determinism_ok_baseline:
        out.append(
            "agent_determinism_check_failed_baseline: baseline factory "
            "is non-deterministic on identical inputs."
        )
    return out


def _capacity_parity_warning(
    ai_real: BacktestResult,
    baseline_real: BacktestResult,
    *,
    parity_band: tuple[float, float] = (0.5, 2.0),
) -> Optional[str]:
    """Warn when AI and baseline differ in turnover by more than the band.

    Uses ``num_trades`` as the practical proxy for turnover when an
    explicit ``turnover_history`` is unavailable. The band is
    log-symmetric: ``[0.5, 2.0]`` linear == ``[-ln 2, +ln 2]`` log.
    """
    ai_t = max(ai_real.num_trades, 1)
    base_t = max(baseline_real.num_trades, 1)
    ratio = ai_t / base_t
    lo, hi = parity_band
    if not (lo <= ratio <= hi):
        return (
            f"capacity_parity_warning: AI turnover ({ai_t} trades) and "
            f"baseline turnover ({base_t} trades) differ by ratio "
            f"{ratio:.2f}× (parity band {lo}–{hi}×). The diff-in-diff "
            "may be uninformative when the two arms operate at different "
            "trade frequencies."
        )
    return None


def run_ab_probe(
    ai_factory: AgentFactory,
    baseline_factory: AgentFactory,
    data: pd.DataFrame,
    scenarios: Sequence[ABScenario],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    n_repeat: int = 10,
    seeds: Optional[Sequence[int]] = None,
    metric_config: Optional[Dict[str, MetricConfig]] = None,
    min_valid_repeats: int = 5,
    agent_contract: Optional[AgentContract] = None,
    enable_ai_noise_control: bool = False,
    engine_kwargs: Optional[Dict[str, Any]] = None,
    provider_config: Optional[Dict[str, Any]] = None,
    parity_band: tuple[float, float] = (0.5, 2.0),
) -> ABProbeResult:
    """Run the v2.0 A/B probe across one or more scenarios.

    Each scenario produces four arms per repeat (AI raw, AI test,
    baseline raw, baseline test); paired seeds within a repeat keep
    AI and baseline aligned on the same transform realization.
    Output is descriptive — no p-values, no verdicts.
    """
    cfg_map: Dict[str, MetricConfig] = dict(DEFAULT_METRIC_CONFIG)
    if metric_config:
        cfg_map.update(metric_config)
    for m in metrics:
        if m not in cfg_map:
            raise KeyError(
                f"metric {m!r} requires a MetricConfig (pass via "
                "metric_config=)."
            )

    if seeds is not None and len(seeds) != n_repeat:
        raise ValueError(
            f"seeds length ({len(seeds)}) must equal n_repeat ({n_repeat})"
        )
    if min_valid_repeats > n_repeat:
        raise ValueError(
            f"min_valid_repeats ({min_valid_repeats}) cannot exceed "
            f"n_repeat ({n_repeat}); summary scalars would always be None."
        )
    seeds_used = (
        list(seeds) if seeds is not None
        else list(range(n_repeat))
    )

    # Determinism check (one factory each, separate from main loop).
    # Note: this is a once-per-probe check on the raw path. A Hook
    # whose non-determinism is triggered only by transformed inputs
    # (e.g., a regime-flag-driven random call) will not be caught
    # here — surface that limitation in the warning string.
    determinism_ok_ai = _check_agent_determinism(
        ai_factory, data, seed=seeds_used[0],
        engine_kwargs=engine_kwargs,
    )
    determinism_ok_baseline = _check_agent_determinism(
        baseline_factory, data, seed=seeds_used[0],
        engine_kwargs=engine_kwargs,
    )

    # Capture baseline + AI configuration for cross-paper manifest
    # comparability — two probes using `MACrossBaseline(short=10,
    # long=30)` vs `MACrossBaseline(short=20, long=50)` produce
    # non-comparable excess_drops; the repr distinguishes them.
    try:
        ai_repr = repr(ai_factory())
    except Exception as e:  # pragma: no cover - defensive
        ai_repr = f"<unrepresentable: {e!r}>"
    try:
        baseline_repr = repr(baseline_factory())
    except Exception as e:  # pragma: no cover
        baseline_repr = f"<unrepresentable: {e!r}>"

    scenario_reports: List[ScenarioABReport] = []
    for scenario in scenarios:
        scenario_reports.append(
            _run_scenario(
                scenario=scenario,
                ai_factory=ai_factory,
                baseline_factory=baseline_factory,
                data=data,
                metrics=metrics,
                cfg_map=cfg_map,
                seeds_used=seeds_used,
                n_repeat=n_repeat,
                min_valid_repeats=min_valid_repeats,
                agent_contract=agent_contract,
                enable_ai_noise_control=enable_ai_noise_control,
                engine_kwargs=engine_kwargs,
                parity_band=parity_band,
                determinism_ok_ai=determinism_ok_ai,
                determinism_ok_baseline=determinism_ok_baseline,
            )
        )

    manifest = {
        "data_hash": _hash_data(data),
        "data_rows": int(len(data)),
        "data_date_range": [
            str(data.index[0]) if len(data) else None,
            str(data.index[-1]) if len(data) else None,
        ],
        "n_repeat_requested": n_repeat,
        "seeds_used": list(seeds_used),
        "metrics": list(metrics),
        "metric_config": {
            m: asdict(cfg_map[m]) for m in metrics
        },
        "min_valid_repeats": min_valid_repeats,
        "agent_contract": agent_contract,
        "enable_ai_noise_control": enable_ai_noise_control,
        "parity_band": list(parity_band),
        "ai_determinism_check_passed": determinism_ok_ai,
        "baseline_determinism_check_passed": determinism_ok_baseline,
        "ai_factory_repr": ai_repr,
        "baseline_factory_repr": baseline_repr,
        "engine_kwargs": dict(engine_kwargs or {}),
        "provider_config": dict(provider_config or {}),
        "scenarios": [
            {
                "scenario_id": s.scenario_id,
                "mode": s.mode,
                "transforms": [
                    {
                        "name": getattr(t, "name", type(t).__name__),
                        "category": getattr(t, "category", None),
                        "stochastic": getattr(t, "stochastic", False),
                        "repr": repr(t),
                    }
                    for t in s.transforms
                ],
                "notes": s.notes,
            }
            for s in scenarios
        ],
    }
    return ABProbeResult(scenarios=scenario_reports, manifest=manifest)


def _run_scenario(
    *,
    scenario: ABScenario,
    ai_factory: AgentFactory,
    baseline_factory: AgentFactory,
    data: pd.DataFrame,
    metrics: Sequence[str],
    cfg_map: Dict[str, MetricConfig],
    seeds_used: Sequence[int],
    n_repeat: int,
    min_valid_repeats: int,
    agent_contract: Optional[AgentContract],
    enable_ai_noise_control: bool,
    engine_kwargs: Optional[Dict[str, Any]],
    parity_band: tuple[float, float],
    determinism_ok_ai: bool,
    determinism_ok_baseline: bool,
) -> ScenarioABReport:
    """Run one ABScenario across n_repeat repeats."""

    # Pre-check: TransformPipeline construction validates mode and
    # invertibility via the agent_contract gate.
    if scenario.transforms:
        TransformPipeline(
            transforms=list(scenario.transforms),
            mode=scenario.mode,
            agent_contract=agent_contract,
        )

    is_stochastic_pipeline = any(
        getattr(t, "stochastic", False) for t in scenario.transforms
    )

    # Per-repeat raw values (seed-paired; identical seed for AI and
    # baseline within a repeat).
    raw_records: Dict[str, Dict[str, List[float]]] = {
        m: {"ai_raw": [], "ai_test": [],
            "baseline_raw": [], "baseline_test": []}
        for m in metrics
    }
    ai_test_a_records: Dict[str, List[float]] = {m: [] for m in metrics}
    ai_test_b_records: Dict[str, List[float]] = {m: [] for m in metrics}

    parity_warning: Optional[str] = None

    for r, seed_r in enumerate(seeds_used):
        # Raw arms.
        ai_raw = _run_one_arm(
            ai_factory, data, transforms=[], mode=scenario.mode,
            seed=seed_r, engine_kwargs=engine_kwargs,
        )
        baseline_raw = _run_one_arm(
            baseline_factory, data, transforms=[], mode=scenario.mode,
            seed=seed_r, engine_kwargs=engine_kwargs,
        )
        if r == 0:
            parity_warning = _capacity_parity_warning(
                ai_raw, baseline_raw, parity_band=parity_band,
            )

        # Test arms (transformed). For stochastic pipelines, AI and
        # baseline share the same seed so the transform realization
        # is identical within the repeat.
        ai_test = _run_one_arm(
            ai_factory, data, transforms=scenario.transforms,
            mode=scenario.mode, seed=seed_r, engine_kwargs=engine_kwargs,
        )
        baseline_test = _run_one_arm(
            baseline_factory, data, transforms=scenario.transforms,
            mode=scenario.mode, seed=seed_r, engine_kwargs=engine_kwargs,
        )

        for m in metrics:
            raw_records[m]["ai_raw"].append(_extract_metric(ai_raw, m))
            raw_records[m]["ai_test"].append(_extract_metric(ai_test, m))
            raw_records[m]["baseline_raw"].append(_extract_metric(baseline_raw, m))
            raw_records[m]["baseline_test"].append(_extract_metric(baseline_test, m))

        # Optional AI-on-AI noise control. Hard-gated on stochastic
        # pipelines: refuse to run on deterministic pipelines even
        # when explicitly requested, since the control is mathematically
        # vacuous (test_a == test_b by construction).
        if enable_ai_noise_control and is_stochastic_pipeline:
            # Two independent realizations spawned from `seed_r` per
            # plan §3.8 ("two independent transformed realizations").
            # Using SeedSequence.spawn keeps the noise sub-seeds
            # deterministically derived from the outer seed.
            child_a, child_b = np.random.SeedSequence(seed_r).spawn(2)
            sub_a = int(child_a.generate_state(1)[0])
            sub_b = int(child_b.generate_state(1)[0])
            ai_test_a = _run_one_arm(
                ai_factory, data, transforms=scenario.transforms,
                mode=scenario.mode, seed=sub_a,
                engine_kwargs=engine_kwargs,
            )
            ai_test_b = _run_one_arm(
                ai_factory, data, transforms=scenario.transforms,
                mode=scenario.mode, seed=sub_b,
                engine_kwargs=engine_kwargs,
            )
            for m in metrics:
                ai_test_a_records[m].append(_extract_metric(ai_test_a, m))
                ai_test_b_records[m].append(_extract_metric(ai_test_b, m))

    # Build per-metric summaries.
    metric_summaries: Dict[str, MetricDropSummary] = {}
    for m in metrics:
        cfg = cfg_map[m]
        ai_raw_list = raw_records[m]["ai_raw"]
        ai_test_list = raw_records[m]["ai_test"]
        base_raw_list = raw_records[m]["baseline_raw"]
        base_test_list = raw_records[m]["baseline_test"]

        ai_abs: List[float] = []
        base_abs: List[float] = []
        ai_rel: List[Optional[float]] = []
        base_rel: List[Optional[float]] = []
        excess: List[Optional[float]] = []
        n_low_ai = 0
        n_low_base = 0

        for r in range(n_repeat):
            ai_a, ai_r, ai_low = _compute_relative_drop(
                ai_raw_list[r], ai_test_list[r], cfg
            )
            base_a, base_r, base_low = _compute_relative_drop(
                base_raw_list[r], base_test_list[r], cfg
            )
            ai_abs.append(ai_a)
            base_abs.append(base_a)
            ai_rel.append(ai_r)
            base_rel.append(base_r)
            if ai_low:
                n_low_ai += 1
            if base_low:
                n_low_base += 1
            if ai_r is not None and base_r is not None:
                excess.append(ai_r - base_r)
            else:
                excess.append(None)

        valid_excess = [v for v in excess if v is not None]
        n_valid = len(valid_excess)

        # Min-valid-repeats gate around summary scalars.
        if n_valid >= min_valid_repeats:
            stats = _summary_stats(valid_excess)
            dom = float(np.mean(np.asarray(valid_excess) > 0))
        else:
            stats = {}
            dom = None

        # Optional AI-on-AI noise control aggregation.
        ai_noise_abs: Optional[List[float]] = None
        ai_noise_rel: Optional[List[float]] = None
        mean_ai_noise_abs = median_ai_noise_abs = iqr_ai_noise_abs = None
        mean_ai_noise_rel = median_ai_noise_rel = iqr_ai_noise_rel = None
        if (
            enable_ai_noise_control
            and is_stochastic_pipeline
            and ai_test_a_records[m]
        ):
            ai_noise_abs = []
            ai_noise_rel = []
            for a, b in zip(ai_test_a_records[m], ai_test_b_records[m]):
                o_a = _orient(a, cfg)
                o_b = _orient(b, cfg)
                abs_d = abs(o_a - o_b)
                denom = max(abs(o_a), abs(o_b), cfg.normalization_floor)
                ai_noise_abs.append(abs_d)
                ai_noise_rel.append(abs_d / denom)
            if ai_noise_abs:
                mean_ai_noise_abs = float(np.mean(ai_noise_abs))
                median_ai_noise_abs = float(np.median(ai_noise_abs))
                iqr_ai_noise_abs = float(
                    np.percentile(ai_noise_abs, 75)
                    - np.percentile(ai_noise_abs, 25)
                )
                mean_ai_noise_rel = float(np.mean(ai_noise_rel))
                median_ai_noise_rel = float(np.median(ai_noise_rel))
                iqr_ai_noise_rel = float(
                    np.percentile(ai_noise_rel, 75)
                    - np.percentile(ai_noise_rel, 25)
                )

        metric_summaries[m] = MetricDropSummary(
            metric=m,
            ai_raw=ai_raw_list,
            ai_test=ai_test_list,
            baseline_raw=base_raw_list,
            baseline_test=base_test_list,
            ai_abs_drop=ai_abs,
            baseline_abs_drop=base_abs,
            ai_rel_drop=ai_rel,
            baseline_rel_drop=base_rel,
            excess_drop=excess,
            n_valid_relative=n_valid,
            min_valid_repeats=min_valid_repeats,
            n_low_anchor_ai=n_low_ai,
            n_low_anchor_baseline=n_low_base,
            dominance_rate=dom,
            mean_excess_drop=stats.get("mean") if stats else None,
            median_excess_drop=stats.get("median") if stats else None,
            std_excess_drop=stats.get("std") if stats else None,
            p10_excess_drop=stats.get("p10") if stats else None,
            p90_excess_drop=stats.get("p90") if stats else None,
            ai_noise_abs=ai_noise_abs,
            ai_noise_rel=ai_noise_rel,
            mean_ai_noise_abs=mean_ai_noise_abs,
            median_ai_noise_abs=median_ai_noise_abs,
            iqr_ai_noise_abs=iqr_ai_noise_abs,
            mean_ai_noise_rel=mean_ai_noise_rel,
            median_ai_noise_rel=median_ai_noise_rel,
            iqr_ai_noise_rel=iqr_ai_noise_rel,
        )

    # Per-repeat table for audit.
    rows: List[Dict[str, Any]] = []
    for r in range(n_repeat):
        for m in metrics:
            rows.append({
                "repeat": r,
                "seed": seeds_used[r],
                "metric": m,
                "ai_raw": raw_records[m]["ai_raw"][r],
                "ai_test": raw_records[m]["ai_test"][r],
                "baseline_raw": raw_records[m]["baseline_raw"][r],
                "baseline_test": raw_records[m]["baseline_test"][r],
            })
    per_repeat_table = pd.DataFrame(rows) if rows else None

    if enable_ai_noise_control and not is_stochastic_pipeline:
        warnings.warn(
            f"scenario {scenario.scenario_id!r}: enable_ai_noise_control=True "
            "but pipeline contains no stochastic transforms; control is "
            "mathematically vacuous and was not run.",
            stacklevel=2,
        )

    n_unique = n_repeat if is_stochastic_pipeline else 1
    warning_list = _scenario_warnings(
        scenario,
        parity_warning=parity_warning,
        determinism_ok_ai=determinism_ok_ai,
        determinism_ok_baseline=determinism_ok_baseline,
    )
    if enable_ai_noise_control and not is_stochastic_pipeline:
        warning_list.append(
            "ai_noise_control_skipped: pipeline is fully deterministic; "
            "AI-on-AI control was hard-disabled even though "
            "enable_ai_noise_control=True (control is vacuous)."
        )

    return ScenarioABReport(
        scenario_id=scenario.scenario_id,
        mode=scenario.mode,
        n_repeat_requested=n_repeat,
        n_unique_transform_realizations=n_unique,
        metric_summaries=metric_summaries,
        per_repeat_table=per_repeat_table,
        warnings=warning_list,
    )


__all__ = [
    "DEFAULT_METRICS",
    "DEFAULT_METRIC_CONFIG",
    "MACrossBaseline",
    "MeanRevBaseline",
    "MomentumBaseline",
    "run_ab_probe",
]
