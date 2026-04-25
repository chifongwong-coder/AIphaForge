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
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from aiphaforge.engine import BacktestEngine
from aiphaforge.hooks import BacktestHook
from aiphaforge.probes.models import (
    ABProbeResult,
    ABScenario,
    AgentContract,
    AgentImplementationContract,
    DeterminismCheckResult,
    MetricConfig,
    MetricDropSummary,
    ResolvedDeterminismConfig,
    ScenarioABReport,
    UnsupportedScenarioError,
)
from aiphaforge.probes.scoring import _merge_provider_config
from aiphaforge.probes.transforms import (
    TransformPipeline,
    _effective_calendar_from_transforms,
)
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

    # v2.1: thread the inferred effective calendar so the pipeline's
    # final integrity validator runs calendar conformance. This is
    # the user-observable A/B execution path.
    effective_calendar = _effective_calendar_from_transforms(transforms)
    pipeline = TransformPipeline(
        transforms=list(transforms), mode=mode,
        calendar=effective_calendar,
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


# ---------- v2.0.1 r5: determinism check ----------

# Profile defaults — see plan §5.3.
_PROFILE_V2_COMPAT = (("total_return",), 1e-12)
_PROFILE_LLM_BALANCED = (("total_return", "num_trades", "win_rate"), 1e-3)


def resolve_determinism_config(
    *,
    mode: Literal["off", "raw_only", "per_scenario"],
    profile: Literal["auto", "v2_compat", "llm_balanced"],
    determinism_metrics: Optional[Sequence[str]],
    determinism_rel_tol: Optional[float],
) -> ResolvedDeterminismConfig:
    """Resolve user kwargs into a concrete metrics/tol pair.

    Plan §5.3 r5. ``auto + raw_only`` → ``v2_compat``; ``auto +
    per_scenario`` → ``llm_balanced``; ``auto + off`` → no metrics.
    Explicit ``determinism_metrics=`` and ``determinism_rel_tol=``
    override the profile.
    """
    requested_metrics: Optional[tuple[str, ...]] = (
        tuple(determinism_metrics) if determinism_metrics is not None else None
    )

    if mode == "off":
        resolved_profile: Literal["v2_compat", "llm_balanced", "off"] = "off"
        base_metrics: tuple[str, ...] = ()
        base_tol: Optional[float] = None
    else:
        if profile == "auto":
            resolved_profile = (
                "v2_compat" if mode == "raw_only" else "llm_balanced"
            )
        else:
            resolved_profile = profile
        if resolved_profile == "v2_compat":
            base_metrics, base_tol = _PROFILE_V2_COMPAT
        else:
            base_metrics, base_tol = _PROFILE_LLM_BALANCED

    metrics_out = (
        requested_metrics if requested_metrics is not None else base_metrics
    )
    tol_out = (
        determinism_rel_tol if determinism_rel_tol is not None else base_tol
    )

    return ResolvedDeterminismConfig(
        profile=resolved_profile,
        determinism_metrics=metrics_out,
        determinism_rel_tol=tol_out,
        requested_profile=profile,
        requested_metrics=requested_metrics,
        requested_rel_tol=determinism_rel_tol,
    )


def _stable_view_fingerprint(view: pd.DataFrame) -> str:
    """Deterministic hash of a transformed view, for replayability check.

    Same shape + same byte content → same fingerprint. Used by the
    per-scenario transformed-arm check to refuse user transforms
    that ignore ``seed=`` (plan §5.8).

    v2.0.2 hardening: hash per-column instead of casting the whole
    frame to float64. The previous implementation raised
    ``ValueError`` on user transforms that emit metadata string
    columns; the caller swallowed it and treated the view as
    replayable, hiding genuine non-determinism. We now hash numeric
    columns by their bytes and non-numeric columns by the bytes of
    their pandas string repr.
    """
    h = hashlib.sha256()
    # Column names + order — fingerprint changes if either changes.
    h.update(",".join(map(str, view.columns)).encode("utf-8"))
    h.update(b"|")
    # Index — works for datetime, int, str alike via numpy bytes.
    h.update(np.asarray(view.index.values).tobytes())
    h.update(b"|")
    # Per-column content hashing keeps numeric paths fast (bytes
    # roundtrip) while supporting metadata string columns without
    # raising.
    for col in view.columns:
        series = view[col]
        if np.issubdtype(series.dtype, np.number):
            h.update(series.to_numpy(dtype=np.float64).tobytes())
        else:
            # bool / object / datetime / str — encode the repr.
            h.update(series.astype(str).str.encode("utf-8").sum())
        h.update(b";")
    return h.hexdigest()[:16]


def _is_finite(x: float) -> bool:
    return not (np.isnan(x) or np.isinf(x))


def _values_match(v1: float, v2: float, *, rel_tol: float) -> bool:
    """Symmetric relative-tolerance equality with NaN handling.

    Two NaNs match (both runs failed identically). One NaN does not
    match a finite value. Both finite values use the existing
    symmetric denominator.
    """
    nan1 = np.isnan(v1)
    nan2 = np.isnan(v2)
    if nan1 and nan2:
        return True
    if nan1 or nan2:
        return False
    denom = max(abs(v1), abs(v2), 1e-12)
    return abs(v1 - v2) / denom < rel_tol


def _check_agent_determinism(
    factory: AgentFactory,
    data: pd.DataFrame,
    *,
    resolved_config: ResolvedDeterminismConfig,
    seed: Optional[int],
    engine_kwargs: Optional[Dict[str, Any]],
    transforms: Optional[Sequence] = None,
    mode: str = "market_level",
) -> DeterminismCheckResult:
    """Run the factory twice on identical inputs; return result object.

    Plan §5.7 r5 contract:
    1. Exactly one engine pair per arm (extracted metrics share runs).
    2. ``UnsupportedScenarioError`` re-raised to the orchestrator.
    3. Ordinary exceptions → ``status="error"``, ``passed=False``.
    4. Returns metric-level pass/fail accounting in ``failed_metrics``.
    """
    transforms = list(transforms) if transforms else []
    metrics = resolved_config.determinism_metrics
    rel_tol = resolved_config.determinism_rel_tol or 0.0

    try:
        r1 = _run_one_arm(
            factory, data, transforms=transforms, mode=mode,
            seed=seed, engine_kwargs=engine_kwargs,
        )
        r2 = _run_one_arm(
            factory, data, transforms=transforms, mode=mode,
            seed=seed, engine_kwargs=engine_kwargs,
        )
    except UnsupportedScenarioError:
        # Framework preflight signals — orchestrator records as
        # status="unsupported", not as a determinism failure.
        raise
    except Exception as exc:
        return DeterminismCheckResult(
            passed=False,
            status="error",
            metric_values_run_1={},
            metric_values_run_2={},
            failed_metrics=[],
            determinism_metrics=tuple(metrics),
            determinism_rel_tol=rel_tol,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    values_1: dict[str, float] = {}
    values_2: dict[str, float] = {}
    failed: list[str] = []
    for m in metrics:
        v1 = _extract_metric(r1, m)
        v2 = _extract_metric(r2, m)
        # Internal dataclass dicts are finite-float-only; let the
        # JSON serializer expand to None / sentinels.
        if _is_finite(v1):
            values_1[m] = float(v1)
        if _is_finite(v2):
            values_2[m] = float(v2)
        if not _values_match(v1, v2, rel_tol=rel_tol):
            failed.append(m)

    if failed:
        return DeterminismCheckResult(
            passed=False,
            status="failed",
            metric_values_run_1=values_1,
            metric_values_run_2=values_2,
            failed_metrics=failed,
            determinism_metrics=tuple(metrics),
            determinism_rel_tol=rel_tol,
        )
    return DeterminismCheckResult(
        passed=True,
        status="passed",
        metric_values_run_1=values_1,
        metric_values_run_2=values_2,
        failed_metrics=[],
        determinism_metrics=tuple(metrics),
        determinism_rel_tol=rel_tol,
    )


def _arm_result_to_json(
    result: Optional[DeterminismCheckResult],
    *,
    metrics: Sequence[str],
    rel_tol: Optional[float],
) -> Optional[dict[str, Any]]:
    """Serialize one ``DeterminismCheckResult`` to JSON-safe ``ArmResult``.

    Plan §5.10 r5: every requested metric appears as a key (missing
    → None, never omitted); NaN/inf normalize to string sentinels.
    Returns ``None`` when ``result is None`` (the off-mode shape).
    """
    if result is None:
        return None
    metric_keys = tuple(metrics)

    def _project(values: dict[str, float]) -> dict[str, Optional[Union[float, str]]]:
        out: dict[str, Optional[Union[float, str]]] = {}
        for k in metric_keys:
            if k not in values:
                out[k] = None
                continue
            v = values[k]
            if np.isnan(v):
                out[k] = "nan"
            elif np.isposinf(v):
                out[k] = "inf"
            elif np.isneginf(v):
                out[k] = "-inf"
            else:
                out[k] = float(v)
        return out

    return {
        "passed": result.passed,
        "status": result.status,
        "metric_values_run_1": _project(result.metric_values_run_1),
        "metric_values_run_2": _project(result.metric_values_run_2),
        "failed_metrics": list(result.failed_metrics),
        "determinism_metrics": list(metric_keys),
        "determinism_rel_tol": rel_tol,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "metadata": dict(result.metadata),
    }


def _legacy_pass(
    result: Optional[DeterminismCheckResult],
    *,
    mode: str,
) -> Optional[bool]:
    """Plan §5.11: legacy bool mirror conversion."""
    if mode == "off":
        return None
    if result is None:
        return False
    return result.status == "passed"


def _preflight_unsupported(
    *,
    contract: Optional[AgentImplementationContract],
    scenario_mode: str,
) -> Optional[str]:
    """Return an UnsupportedScenarioError reason or None.

    Plan §5.5 / §5.15: preflight detects the v2.0.1 unsupported
    combinations. Currently only ``view_only`` + plain ``hook``.
    Other contracts are admitted; the orchestrator surfaces ordinary
    runtime errors as ``status="error"``.
    """
    if scenario_mode == "view_only" and contract == "hook":
        return (
            "view_only with a plain hook is unsupported in v2.0.1; "
            "use hook_view_only_capable or wait for the v2.2 broker-proxy "
            "wrapper."
        )
    return None


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


def _format_determinism_warning(
    *,
    subject: str,
    scenario_id: str,
    arm: str,
    result: DeterminismCheckResult,
) -> Optional[str]:
    """Build a status-based warning per plan §5.14.

    Returns ``None`` for ``passed`` results. Warning text always
    carries ``subject=``, ``scenario=``, ``arm=``, and either
    ``failed_metrics=`` or ``reason=``.
    """
    status = result.status
    if status == "passed":
        return None
    if status == "failed":
        return (
            f"agent_determinism_check_failed: subject={subject} "
            f"scenario={scenario_id} arm={arm} "
            f"failed_metrics={result.failed_metrics}; "
            f"determinism_rel_tol={result.determinism_rel_tol}; "
            "see manifest for values."
        )
    if status == "error":
        return (
            f"agent_determinism_check_error: subject={subject} "
            f"scenario={scenario_id} arm={arm} "
            f"reason='{result.error_type}: {result.error_message}'."
        )
    if status == "unsupported":
        return (
            f"agent_determinism_check_unsupported: subject={subject} "
            f"scenario={scenario_id} arm={arm} "
            f"reason='{result.error_message}'; "
            "this is not a determinism failure."
        )
    return None


def _scenario_warnings(
    scenario: ABScenario,
    *,
    parity_warning: Optional[str],
    transformed_ai: Optional[DeterminismCheckResult] = None,
    transformed_baseline: Optional[DeterminismCheckResult] = None,
) -> List[str]:
    """Per-scenario warnings.

    Raw-arm warnings emit ONCE per subject at the top level (plan
    §5.14: "raw warnings emit once per subject, not once per
    scenario"). This function only handles transform-detectability,
    capacity-parity, and the transformed-arm determinism warnings.
    """
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
    if transformed_ai is not None:
        w = _format_determinism_warning(
            subject="ai", scenario_id=scenario.scenario_id,
            arm="transformed", result=transformed_ai,
        )
        if w:
            out.append(w)
    if transformed_baseline is not None:
        w = _format_determinism_warning(
            subject="baseline", scenario_id=scenario.scenario_id,
            arm="transformed", result=transformed_baseline,
        )
        if w:
            out.append(w)
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
    agent_determinism_check: Literal["off", "raw_only", "per_scenario"] = "raw_only",
    determinism_profile: Literal["auto", "v2_compat", "llm_balanced"] = "auto",
    determinism_metrics: Optional[Sequence[str]] = None,
    determinism_rel_tol: Optional[float] = None,
    agent_implementation_contract: Optional[AgentImplementationContract] = None,
    engine_kwargs: Optional[Dict[str, Any]] = None,
    provider_config: Optional[Dict[str, Any]] = None,
    parity_band: tuple[float, float] = (0.5, 2.0),
) -> ABProbeResult:
    """Run the v2.0 A/B probe across one or more scenarios.

    Each scenario produces four arms per repeat (AI raw, AI test,
    baseline raw, baseline test); paired seeds within a repeat keep
    AI and baseline aligned on the same transform realization.
    Output is descriptive — no p-values, no verdicts.

    Determinism check (v2.0.1 r5):
        ``agent_determinism_check`` chooses ``off`` / ``raw_only``
        (default; v2.0-compatible) / ``per_scenario``.

        ``determinism_profile`` resolves into a concrete
        (metrics, rel_tol) pair. ``"auto"`` + ``raw_only`` →
        ``v2_compat`` (``("total_return",)`` + ``1e-12``); ``"auto"``
        + ``per_scenario`` → ``llm_balanced``
        (``("total_return", "num_trades", "win_rate")`` + ``1e-3``).
        Explicit ``determinism_metrics=`` and
        ``determinism_rel_tol=`` override the profile.

        ``agent_implementation_contract`` declares the agent's
        implementation shape (``"strategy"`` / ``"hook"`` /
        ``"hook_view_only_capable"`` / ``"callable_factory"``);
        unsupported combinations (e.g. ``view_only`` + plain
        ``"hook"``) are reported as ``status="unsupported"`` rather
        than as a determinism failure.

        Distinct from the existing ``agent_contract`` kwarg, which
        is the *order-shape* literal governing transform
        admissibility under ``view_only``.

        The canonical determinism schema lives at
        ``manifest["determinism_check"]`` with ``schema_version``;
        legacy flat mirror fields (``ai_determinism_check_passed``,
        ...) are derived from it.
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

    # Resolve determinism config once (plan §5.3).
    resolved = resolve_determinism_config(
        mode=agent_determinism_check,
        profile=determinism_profile,
        determinism_metrics=determinism_metrics,
        determinism_rel_tol=determinism_rel_tol,
    )

    # Raw determinism — one engine pair per subject, shared across
    # all scenarios in per_scenario mode (plan §5.4).
    raw_ai: Optional[DeterminismCheckResult] = None
    raw_baseline: Optional[DeterminismCheckResult] = None
    if agent_determinism_check != "off":
        raw_ai = _check_agent_determinism(
            ai_factory, data,
            resolved_config=resolved,
            seed=seeds_used[0], engine_kwargs=engine_kwargs,
        )
        raw_baseline = _check_agent_determinism(
            baseline_factory, data,
            resolved_config=resolved,
            seed=seeds_used[0], engine_kwargs=engine_kwargs,
        )
        # Raw warnings emit once per subject at the top level (plan
        # §5.14) — not duplicated under each scenario_id.
        for subj, res in (("ai", raw_ai), ("baseline", raw_baseline)):
            w = _format_determinism_warning(
                subject=subj, scenario_id="__raw__",
                arm="raw", result=res,
            )
            if w:
                warnings.warn(w, stacklevel=2)

    # Capture baseline + AI configuration for cross-paper manifest
    # comparability.
    try:
        ai_repr = repr(ai_factory())
    except Exception as e:  # pragma: no cover - defensive
        ai_repr = f"<unrepresentable: {e!r}>"
    try:
        baseline_repr = repr(baseline_factory())
    except Exception as e:  # pragma: no cover
        baseline_repr = f"<unrepresentable: {e!r}>"

    scenario_reports: List[ScenarioABReport] = []
    transformed_ai_per_scenario: Dict[str, Optional[DeterminismCheckResult]] = {}
    transformed_baseline_per_scenario: Dict[str, Optional[DeterminismCheckResult]] = {}
    for scenario in scenarios:
        report, t_ai, t_base = _run_scenario(
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
            agent_implementation_contract=agent_implementation_contract,
            enable_ai_noise_control=enable_ai_noise_control,
            engine_kwargs=engine_kwargs,
            parity_band=parity_band,
            agent_determinism_check=agent_determinism_check,
            resolved_config=resolved,
        )
        scenario_reports.append(report)
        transformed_ai_per_scenario[scenario.scenario_id] = t_ai
        transformed_baseline_per_scenario[scenario.scenario_id] = t_base

    # Build canonical determinism_check first (plan §5.9, §5.16 #6).
    metric_keys = resolved.determinism_metrics
    rel_tol_out = resolved.determinism_rel_tol

    def _per_scenario_dict(
        transformed_map: Dict[str, Optional[DeterminismCheckResult]],
        raw_result: Optional[DeterminismCheckResult],
    ) -> Dict[str, Dict[str, Optional[Dict[str, Any]]]]:
        # Per-scenario raw fields are intentionally duplicated under
        # every scenario_id for self-contained export (plan §5.13);
        # the raw check is still executed only once.
        raw_arm = _arm_result_to_json(
            raw_result, metrics=metric_keys, rel_tol=rel_tol_out,
        )
        out: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = {}
        for sid, t_res in transformed_map.items():
            out[sid] = {
                "raw": raw_arm,
                "transformed": _arm_result_to_json(
                    t_res, metrics=metric_keys, rel_tol=rel_tol_out,
                ),
            }
        return out

    canonical = {
        "schema_version": "2.0.1",
        "mode": agent_determinism_check,
        "agent_implementation_contract": agent_implementation_contract,
        "requested": {
            "determinism_profile": resolved.requested_profile,
            "determinism_metrics": (
                list(resolved.requested_metrics)
                if resolved.requested_metrics is not None
                else None
            ),
            "determinism_rel_tol": resolved.requested_rel_tol,
        },
        "resolved": {
            "profile": resolved.profile,
            "determinism_metrics": list(metric_keys),
            "determinism_rel_tol": rel_tol_out,
        },
        "subjects": {
            "ai": {
                "raw": _arm_result_to_json(
                    raw_ai, metrics=metric_keys, rel_tol=rel_tol_out,
                ),
                "per_scenario": (
                    _per_scenario_dict(transformed_ai_per_scenario, raw_ai)
                    if agent_determinism_check == "per_scenario"
                    else {}
                ),
            },
            "baseline": {
                "raw": _arm_result_to_json(
                    raw_baseline, metrics=metric_keys, rel_tol=rel_tol_out,
                ),
                "per_scenario": (
                    _per_scenario_dict(
                        transformed_baseline_per_scenario, raw_baseline,
                    )
                    if agent_determinism_check == "per_scenario"
                    else {}
                ),
            },
        },
        "controls": {},
        "extension": {},
    }

    # Legacy flat mirrors derived from canonical (plan §5.11, §5.12).
    def _flat_per_scenario(
        subject_block: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        flat: Dict[str, Dict[str, Any]] = {}
        for sid, arms in subject_block.get("per_scenario", {}).items():
            raw_arm = arms.get("raw") or {}
            t_arm = arms.get("transformed") or {}
            flat[sid] = {
                "raw_passed": raw_arm.get("passed"),
                "raw_status": raw_arm.get("status"),
                "raw_failed_metrics": raw_arm.get("failed_metrics", []),
                "raw_metric_values_run_1": raw_arm.get(
                    "metric_values_run_1", {}
                ),
                "raw_metric_values_run_2": raw_arm.get(
                    "metric_values_run_2", {}
                ),
                "raw_error_type": raw_arm.get("error_type"),
                "raw_error_message": raw_arm.get("error_message"),
                "transformed_passed": t_arm.get("passed"),
                "transformed_status": t_arm.get("status"),
                "transformed_failed_metrics": t_arm.get(
                    "failed_metrics", []
                ),
                "transformed_metric_values_run_1": t_arm.get(
                    "metric_values_run_1", {}
                ),
                "transformed_metric_values_run_2": t_arm.get(
                    "metric_values_run_2", {}
                ),
                "transformed_error_type": t_arm.get("error_type"),
                "transformed_error_message": t_arm.get("error_message"),
            }
        return flat

    is_off = agent_determinism_check == "off"
    legacy_status_ai = (
        "off" if is_off
        else (raw_ai.status if raw_ai is not None else "error")
    )
    legacy_status_baseline = (
        "off" if is_off
        else (raw_baseline.status if raw_baseline is not None else "error")
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
        "determinism_check": canonical,
        # Legacy mirror fields (plan §5.11) — derived from canonical.
        "agent_determinism_check_mode": agent_determinism_check,
        "determinism_profile": resolved.profile,
        "determinism_metrics": list(metric_keys),
        "determinism_rel_tol": rel_tol_out,
        "agent_implementation_contract": agent_implementation_contract,
        "ai_determinism_check_passed": _legacy_pass(
            raw_ai, mode=agent_determinism_check,
        ),
        "baseline_determinism_check_passed": _legacy_pass(
            raw_baseline, mode=agent_determinism_check,
        ),
        "ai_determinism_check_status": legacy_status_ai,
        "baseline_determinism_check_status": legacy_status_baseline,
        "ai_determinism_failed_metrics": (
            [] if is_off
            else (raw_ai.failed_metrics if raw_ai is not None else [])
        ),
        "baseline_determinism_failed_metrics": (
            [] if is_off
            else (raw_baseline.failed_metrics if raw_baseline is not None else [])
        ),
        "ai_determinism_check_per_scenario": _flat_per_scenario(
            canonical["subjects"]["ai"]
        ),
        "baseline_determinism_check_per_scenario": _flat_per_scenario(
            canonical["subjects"]["baseline"]
        ),
        "ai_factory_repr": ai_repr,
        "baseline_factory_repr": baseline_repr,
        "engine_kwargs": dict(engine_kwargs or {}),
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
    # Provider config plumbed through the same merge helper as Q&A so
    # the empty-bucket strip rule (plan §3.4) applies uniformly.
    merged_manifest, _prov = _merge_provider_config(manifest, provider_config)
    return ABProbeResult(
        scenarios=scenario_reports,
        manifest=merged_manifest if merged_manifest is not None else manifest,
    )


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
    agent_implementation_contract: Optional[AgentImplementationContract],
    enable_ai_noise_control: bool,
    engine_kwargs: Optional[Dict[str, Any]],
    parity_band: tuple[float, float],
    agent_determinism_check: str,
    resolved_config: ResolvedDeterminismConfig,
) -> tuple[
    ScenarioABReport,
    Optional[DeterminismCheckResult],
    Optional[DeterminismCheckResult],
]:
    """Run one ABScenario across n_repeat repeats."""

    # Pre-check: TransformPipeline construction validates mode and
    # invertibility via the agent_contract gate. v2.1: also threads
    # the inferred effective calendar so calendar conflicts (e.g.
    # two DateShift transforms with different calendars) fail fast
    # before the per-repeat work begins.
    if scenario.transforms:
        TransformPipeline(
            transforms=list(scenario.transforms),
            mode=scenario.mode,
            agent_contract=agent_contract,
            calendar=_effective_calendar_from_transforms(scenario.transforms),
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

    # Per-scenario determinism check (plan §5.4 r5). The raw-arm
    # check ran once at the top level; here we run the transformed
    # arm. Plan §5.5: framework preflight detects unsupported
    # combinations (e.g. view_only + plain hook) and surfaces them as
    # status="unsupported", not as failures. Plan §5.8: the gate is
    # replayability, not stochasticity — a seeded stochastic
    # transform that produces the same fingerprint twice IS valid.
    transformed_ai: Optional[DeterminismCheckResult] = None
    transformed_baseline: Optional[DeterminismCheckResult] = None
    if (
        agent_determinism_check == "per_scenario"
        and scenario.transforms
    ):
        unsupported_reason = _preflight_unsupported(
            contract=agent_implementation_contract,
            scenario_mode=scenario.mode,
        )
        if unsupported_reason is not None:
            # Build two independent instances. Sharing the same
            # frozen DeterminismCheckResult would also share its
            # mutable fields (metadata dict, failed_metrics list);
            # downstream code that mutates one would corrupt the
            # other (v2.0.2 hardening).
            def _unsupported(reason: str) -> DeterminismCheckResult:
                return DeterminismCheckResult(
                    passed=None,
                    status="unsupported",
                    metric_values_run_1={},
                    metric_values_run_2={},
                    failed_metrics=[],
                    determinism_metrics=resolved_config.determinism_metrics,
                    determinism_rel_tol=resolved_config.determinism_rel_tol or 0.0,
                    error_type="UnsupportedScenarioError",
                    error_message=reason,
                )
            transformed_ai = _unsupported(unsupported_reason)
            transformed_baseline = _unsupported(unsupported_reason)
        else:
            # Replayability fingerprint test (plan §5.8). User
            # transforms that ignore seed get reported as
            # status="unsupported".
            replay_seed = seeds_used[0]
            replayable = True
            try:
                pipeline = TransformPipeline(
                    transforms=list(scenario.transforms),
                    mode=scenario.mode,
                )
                view_1 = pipeline.apply(data, seed=replay_seed)
                view_2 = pipeline.apply(data, seed=replay_seed)
                fp_1 = _stable_view_fingerprint(view_1)
                fp_2 = _stable_view_fingerprint(view_2)
                replayable = fp_1 == fp_2
            except Exception:
                # Pipeline construction failure isn't a determinism
                # question; the per-arm check below will surface it
                # with the appropriate status.
                replayable = True

            if not replayable:
                # See unsupported branch above for the no-shared-mutable
                # rationale (v2.0.2 hardening).
                _msg = (
                    "transform pipeline produced different views for the "
                    "same seed; cannot fix the arm for a determinism check."
                )

                def _non_replayable() -> DeterminismCheckResult:
                    return DeterminismCheckResult(
                        passed=None,
                        status="unsupported",
                        metric_values_run_1={},
                        metric_values_run_2={},
                        failed_metrics=[],
                        determinism_metrics=resolved_config.determinism_metrics,
                        determinism_rel_tol=resolved_config.determinism_rel_tol or 0.0,
                        error_type="NonReplayableViewError",
                        error_message=_msg,
                    )
                transformed_ai = _non_replayable()
                transformed_baseline = _non_replayable()
            else:
                try:
                    transformed_ai = _check_agent_determinism(
                        ai_factory, data,
                        resolved_config=resolved_config,
                        seed=replay_seed,
                        engine_kwargs=engine_kwargs,
                        transforms=scenario.transforms,
                        mode=scenario.mode,
                    )
                except UnsupportedScenarioError as exc:
                    transformed_ai = DeterminismCheckResult(
                        passed=None, status="unsupported",
                        metric_values_run_1={}, metric_values_run_2={},
                        failed_metrics=[],
                        determinism_metrics=resolved_config.determinism_metrics,
                        determinism_rel_tol=resolved_config.determinism_rel_tol or 0.0,
                        error_type="UnsupportedScenarioError",
                        error_message=str(exc),
                    )
                try:
                    transformed_baseline = _check_agent_determinism(
                        baseline_factory, data,
                        resolved_config=resolved_config,
                        seed=replay_seed,
                        engine_kwargs=engine_kwargs,
                        transforms=scenario.transforms,
                        mode=scenario.mode,
                    )
                except UnsupportedScenarioError as exc:
                    transformed_baseline = DeterminismCheckResult(
                        passed=None, status="unsupported",
                        metric_values_run_1={}, metric_values_run_2={},
                        failed_metrics=[],
                        determinism_metrics=resolved_config.determinism_metrics,
                        determinism_rel_tol=resolved_config.determinism_rel_tol or 0.0,
                        error_type="UnsupportedScenarioError",
                        error_message=str(exc),
                    )

    n_unique = n_repeat if is_stochastic_pipeline else 1
    warning_list = _scenario_warnings(
        scenario,
        parity_warning=parity_warning,
        transformed_ai=transformed_ai,
        transformed_baseline=transformed_baseline,
    )
    if enable_ai_noise_control and not is_stochastic_pipeline:
        warning_list.append(
            "ai_noise_control_skipped: pipeline is fully deterministic; "
            "AI-on-AI control was hard-disabled even though "
            "enable_ai_noise_control=True (control is vacuous)."
        )

    # v2.1 §6.1 / §6.2: structured manifest warnings.
    detectability_warnings = _build_calendar_detectability_warnings(
        scenario.transforms,
    )
    collision_warnings = _build_collision_warnings(scenario.transforms)

    report = ScenarioABReport(
        scenario_id=scenario.scenario_id,
        mode=scenario.mode,
        n_repeat_requested=n_repeat,
        n_unique_transform_realizations=n_unique,
        metric_summaries=metric_summaries,
        per_repeat_table=per_repeat_table,
        warnings=warning_list,
        transform_detectability_warnings=detectability_warnings,
        calendar_snap_collisions=collision_warnings,
    )
    return report, transformed_ai, transformed_baseline


def _build_calendar_detectability_warnings(
    transforms: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Plan §6.1 r3-final — calendar_snap_fingerprint warning.

    Fires when any DateShift in the scenario has a calendar AND
    snap != "error" (i.e. the snap can rewrite dates and produce
    fingerprintable post-holiday clustering).
    """
    warnings_out: List[Dict[str, Any]] = []
    for t in transforms:
        if (
            getattr(t, "name", None) == "DateShift"
            and getattr(t, "calendar", None) is not None
            and getattr(t, "snap", None) != "error"
        ):
            warnings_out.append({
                "code": "calendar_snap_fingerprint",
                "severity": "info",
                "source": "DateShift",
                "message": (
                    "Calendar snapping can create post-holiday "
                    "Monday/Tuesday clustering. A frontier LLM may "
                    "detect the transformed calendar rather than "
                    "recall the original series. Interpret behavior "
                    "changes as a possible false-positive pathway."
                ),
            })
    return warnings_out


def _serialize_collision_examples(
    examples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Stringify the in-memory pd.Timestamp objects in the collision
    report's ``examples`` list so the manifest is JSON-safe.

    Plan §4.4 / §6.2 r3-final defect #4 fix: in-memory the
    DateShift report holds pd.Timestamp; the manifest serializer
    converts to YYYY-MM-DD strings here.
    """
    out = []
    for ex in examples:
        out.append({
            "target_ts": ex["target_ts"].strftime("%Y-%m-%d"),
            "source_ts": [
                ts.strftime("%Y-%m-%d") for ts in ex["source_ts"]
            ],
            "kept_source_ts": ex["kept_source_ts"].strftime("%Y-%m-%d"),
            "dropped_source_ts": [
                ts.strftime("%Y-%m-%d") for ts in ex["dropped_source_ts"]
            ],
        })
    return out


def _build_collision_warnings(
    transforms: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Plan §6.2 r3-final — calendar_snap_collision_rows_dropped.

    Reads ``last_collision_report`` populated by DateShift.apply
    when a non-error collision policy actually dropped rows.
    """
    out: List[Dict[str, Any]] = []
    for t in transforms:
        report = getattr(t, "last_collision_report", None)
        if not report:
            continue
        details = {
            "transform": report["transform"],
            "on_collision": report["on_collision"],
            "collision_count": report["collision_count"],
            "collision_group_count": report["collision_group_count"],
            "examples": _serialize_collision_examples(report["examples"]),
            "examples_truncated": report["examples_truncated"],
        }
        out.append({
            "code": "calendar_snap_collision_rows_dropped",
            "severity": "warning",
            "source": report["transform"],
            "message": (
                f"{report['transform']} calendar snapping produced "
                f"duplicate target dates; on_collision="
                f"{report['on_collision']!r} dropped "
                f"{report['collision_count']} source rows. "
                "This changes the transformed sample length."
            ),
            "details": details,
        })
    return out


__all__ = [
    "DEFAULT_METRICS",
    "DEFAULT_METRIC_CONFIG",
    "MACrossBaseline",
    "MeanRevBaseline",
    "MomentumBaseline",
    "run_ab_probe",
]
