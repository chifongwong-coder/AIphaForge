"""v2.2 M2: volatility estimators for vol-scaled tolerance bands.

Three estimators of σ — fractional per-bar log-return volatility:

- ``stdev_returns``: stdev of log(close[t]/close[t-1]) — gold standard.
- ``parkinson``: from intra-bar log(H/L); under-estimates by overnight
  return; ~5× efficient on Brownian motion (Garman-Klass 1980).
- ``garman_klass``: includes log(C/O) — partially captures overnight.

Asset-class auto-picker (per v2.2-plan-r6 § 3.3.1):
  us_equity → garman_klass; crypto → parkinson; futures → parkinson;
  penny_stock → stdev_returns; unknown → stdev_returns.

Causality (per § 3.2):
  point-in-time: σ from bars `< answer_timestamp`.
  continuation: σ from bars `≤ context_window[-1].timestamp`.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .models import ToleranceProfile

# Per v2.2-plan-r6 § 3.3.1
_PRESET_TO_METHOD = {
    "us_equity_price": "garman_klass",
    "us_equity_price_strict": "garman_klass",
    "crypto_price": "parkinson",
    "crypto_price_strict": "parkinson",
    "futures_price": "parkinson",
    "futures_price_strict": "parkinson",
    "penny_stock_price": "stdev_returns",
    "penny_stock_price_strict": "stdev_returns",
}


def resolve_method(method: str, preset_name: str | None) -> str:
    """Resolve ``method='auto'`` to a concrete estimator using the
    preset name. Unknown preset → ``stdev_returns`` (safe default).
    """
    if method != "auto":
        return method
    if preset_name is None:
        return "stdev_returns"
    return _PRESET_TO_METHOD.get(preset_name, "stdev_returns")


def estimate_sigma(
    bars: pd.DataFrame,
    method: str,
    *,
    log_returns: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Estimate σ on the given bars window using the chosen method.

    Returns ``(sigma, provenance)`` where provenance is a dict
    suitable for embedding in a report's ``vol_scale_provenance``.
    All three methods produce σ in units of fractional per-bar
    log-return volatility (close-to-close conceptually).

    H==L fallback (per § 3.3): when ``method`` ∈ {parkinson,
    garman_klass} and high == low for ≥ 50% of window bars,
    automatically falls back to ``stdev_returns`` and records the
    fallback in provenance, along with the closed-form Parkinson
    under-estimate ``1 - sqrt(1 - h_eq_l_fraction)`` derived from
    the (1-f) variance loss of the empirical estimator on H==L
    bars (Parkinson 1980).

    Parkinson event-day warning: when |return| > 5σ in any window
    bar, provenance gains a ``parkinson_event_day_warning`` entry.
    Overnight-gap warning: |log(open[i]/close[i-1])| > 3 × σ →
    additional warning entry (per F6 follow-up).
    """
    if len(bars) < 2:
        return 0.0, {
            "method_requested": method,
            "method_used": method,
            "n_bars": len(bars),
            "window_too_small": True,
        }

    high = bars["high"].astype(float).to_numpy()
    low = bars["low"].astype(float).to_numpy()
    close = bars["close"].astype(float).to_numpy()
    has_open = "open" in bars.columns
    open_ = (
        bars["open"].astype(float).to_numpy() if has_open else close
    )

    provenance: dict[str, Any] = {
        "method_requested": method,
        "n_bars": int(len(bars)),
        "log_returns": log_returns,
    }

    method_used = method

    # v2.2.1 #10: when method='garman_klass' but `open` column is
    # missing, GK silently falls back to using close (zeroing the
    # log(C/O) overnight term). This produces a quietly biased σ
    # with no provenance flag. r7 fix: fall back to Parkinson
    # (which uses only intra-bar H/L and doesn't need open) and
    # record the fallback in provenance.
    if method == "garman_klass" and not has_open:
        method_used = "parkinson"
        provenance["garman_klass_missing_open_fallback"] = (
            "open column missing; Garman-Klass requires log(C/O); "
            "falling back to parkinson which uses only intra-bar H/L."
        )

    # H==L fallback decision for parkinson / garman_klass.
    if method in ("parkinson", "garman_klass"):
        h_eq_l_mask = (high == low)
        h_eq_l_fraction = float(h_eq_l_mask.mean())
        provenance["h_eq_l_fraction"] = h_eq_l_fraction
        if h_eq_l_fraction >= 0.5:
            method_used = "stdev_returns"
            provenance["fallback_reason"] = (
                f"H==L for {h_eq_l_fraction:.0%} of window bars; "
                f"{method} degenerates. Fell back to stdev_returns."
            )
            # Parkinson estimator with H==L on fraction f of bars
            # captures only (1-f) of true variance. Derivation:
            #   σ²_P = (1/(4 ln 2)) · E[(ln H/L)²]
            # H==L bars contribute 0 to the mean → σ̂² = (1-f)·σ²_true
            # → σ̂/σ_true = √(1-f) → under-estimate = 1 - √(1-f).
            # Closed-form, not heuristic; cite Parkinson (1980).
            provenance["estimated_underestimate_pct"] = (
                1.0 - math.sqrt(1.0 - h_eq_l_fraction)
            )
            provenance["underestimate_formula"] = (
                "1 - sqrt(1 - h_eq_l_fraction) — closed-form derivation "
                "from the Parkinson (1980) estimator: H==L bars "
                "contribute 0 to mean((ln H/L)^2), so the empirical "
                "estimator captures only (1-f) of true variance, "
                "giving sigma_hat / sigma_true = sqrt(1-f)."
            )
        elif 0.4 <= h_eq_l_fraction < 0.5:
            # v2.8.3 Commit C (M8): warning band just below the
            # fallback threshold. The estimator does not switch to
            # stdev_returns (insufficient evidence of degeneracy
            # given the 50% gate), but the empirical bias is already
            # material — by the closed-form 1 - sqrt(1-f) the
            # under-estimate at f=0.4 is ~22.5% and at f=0.5 is
            # ~29.3%. Surface a provenance entry so users can
            # audit borderline windows. Does NOT change the chosen
            # estimator.
            provenance["h_eq_l_warning_band"] = {
                "h_eq_l_fraction": h_eq_l_fraction,
                "estimated_underestimate_pct": (
                    1.0 - math.sqrt(1.0 - h_eq_l_fraction)
                ),
                "underestimate_formula": (
                    "1 - sqrt(1 - h_eq_l_fraction) — closed-form "
                    "Parkinson (1980) bias estimate."
                ),
                "guidance": (
                    f"H==L for {h_eq_l_fraction:.0%} of window bars "
                    f"(below the {50:.0%} fallback threshold). "
                    f"{method} is still in use but its sigma is "
                    f"likely under-estimated by "
                    f"{(1.0 - math.sqrt(1.0 - h_eq_l_fraction)) * 100:.1f}%. "
                    f"Consider stdev_returns for sensitivity analysis."
                ),
            }

    # Compute close-to-close log returns (used by stdev_returns and
    # by event-day warning math regardless of chosen method).
    close_to_close = np.log(close[1:] / close[:-1])

    if method_used == "stdev_returns":
        # stdev with N-1 denominator (sample stdev).
        sigma = float(np.std(close_to_close, ddof=1)) if len(
            close_to_close
        ) >= 2 else 0.0
    elif method_used == "parkinson":
        # σ²_P = (1 / (4 ln 2)) * mean( (ln H/L)^2 )
        with np.errstate(divide="ignore", invalid="ignore"):
            log_hl = np.log(high / low)
        log_hl = np.where(np.isfinite(log_hl), log_hl, 0.0)
        sigma2 = float(np.mean(log_hl ** 2) / (4.0 * math.log(2.0)))
        sigma = math.sqrt(max(sigma2, 0.0))
    elif method_used == "garman_klass":
        # σ²_GK = 0.5 * (ln H/L)^2 - (2 ln 2 - 1) * (ln C/O)^2
        with np.errstate(divide="ignore", invalid="ignore"):
            log_hl = np.log(high / low)
            log_co = np.log(close / open_)
        log_hl = np.where(np.isfinite(log_hl), log_hl, 0.0)
        log_co = np.where(np.isfinite(log_co), log_co, 0.0)
        sigma2_gk = (
            0.5 * np.mean(log_hl ** 2)
            - (2.0 * math.log(2.0) - 1.0) * np.mean(log_co ** 2)
        )
        sigma = math.sqrt(max(float(sigma2_gk), 0.0))
    else:
        raise ValueError(f"unknown vol estimator method: {method_used!r}")

    provenance["method_used"] = method_used
    provenance["sigma"] = sigma

    # Parkinson / GK event-day runtime warning.
    if method_used in ("parkinson", "garman_klass") and sigma > 0:
        max_abs_ret = float(np.max(np.abs(close_to_close)))
        if max_abs_ret > 5.0 * sigma:
            provenance["parkinson_event_day_warning"] = {
                "max_abs_return_in_window": max_abs_ret,
                "sigma_estimate": sigma,
                "max_return_over_sigma": max_abs_ret / sigma,
                "guidance": (
                    f"Window contains an event-day move "
                    f"(|return| > 5σ). {method_used} under-estimates "
                    f"σ on event days; consider rerunning with "
                    f"method='stdev_returns' for sensitivity analysis."
                ),
            }
        # Overnight-gap trigger (per F6 follow-up).
        if "open" in bars.columns and len(close) >= 2:
            with np.errstate(divide="ignore", invalid="ignore"):
                gap = np.abs(np.log(open_[1:] / close[:-1]))
            gap = gap[np.isfinite(gap)]
            if len(gap) > 0:
                max_gap = float(np.max(gap))
                if max_gap > 3.0 * sigma:
                    provenance["parkinson_overnight_gap_warning"] = {
                        "max_overnight_gap_in_window": max_gap,
                        "sigma_estimate": sigma,
                        "guidance": (
                            f"Window contains overnight gap > 3σ. "
                            f"{method_used} uses only intra-bar H/L "
                            f"and misses overnight returns; the "
                            f"under-estimate is worse than the "
                            f"event-day case."
                        ),
                    }

    return sigma, provenance


def estimate_sigma_for_pointintime(
    data: pd.DataFrame,
    answer_timestamp: pd.Timestamp,
    window: int,
    method: str,
    preset_name: str | None,
    *,
    log_returns: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Strictly-causal σ for point-in-time questions: bars
    ``< answer_timestamp``, last ``window`` of them.
    """
    method_resolved = resolve_method(method, preset_name)
    prior = data[data.index < answer_timestamp]
    if len(prior) < 2:
        return 0.0, {
            "mode": "pointintime",
            "method_requested": method_resolved,
            "n_bars": int(len(prior)),
            "window_too_small": True,
        }
    window_bars = prior.iloc[-window:]
    sigma, prov = estimate_sigma(
        window_bars, method_resolved, log_returns=log_returns
    )
    prov["mode"] = "pointintime"
    return sigma, prov


def estimate_sigma_for_continuation(
    data: pd.DataFrame,
    last_context_timestamp: pd.Timestamp,
    window: int,
    method: str,
    preset_name: str | None,
    *,
    log_returns: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Strictly-causal σ for continuation questions: bars
    ``≤ last_context_timestamp``, last ``window`` of them. The
    forward-target bar is excluded.
    """
    method_resolved = resolve_method(method, preset_name)
    in_window = data[data.index <= last_context_timestamp]
    if len(in_window) < 2:
        return 0.0, {
            "mode": "continuation",
            "method_requested": method_resolved,
            "n_bars": int(len(in_window)),
            "window_too_small": True,
        }
    window_bars = in_window.iloc[-window:]
    sigma, prov = estimate_sigma(
        window_bars, method_resolved, log_returns=log_returns
    )
    prov["mode"] = "continuation"
    return sigma, prov


def apply_vol_scale(
    base_tolerance: "ToleranceProfile",
    sigma: float,
) -> "ToleranceProfile":
    """Return a new ToleranceProfile whose numeric thresholds are
    ``factor × sigma``, floor-capped by ``base_tolerance.absolute_floor``.

    The factors come from ``base_tolerance.vol_scale``. Range-form
    widths are scaled identically to point-form thresholds. The
    ``max_range_width``, ``sign_sensitive``, ``sign_epsilon`` are
    preserved unchanged.

    Per § 3.4 of v2.2-plan-r6, ``max_range_width`` and
    ``miss_floor_factor`` do NOT interact (different stages of
    scoring): max_range_width gates the user's submitted range,
    miss_floor_factor is the engine-side band lower bound.
    """
    from .models import ToleranceProfile  # late import to avoid cycle
    spec = base_tolerance.vol_scale
    if spec is None:
        return base_tolerance
    floor = base_tolerance.absolute_floor
    near = max(spec.near_factor * sigma, floor)
    rough = max(spec.rough_factor * sigma, floor)
    # NOTE: miss_floor_factor would be max(spec.miss_floor_factor *
    # sigma, floor) but the existing ToleranceProfile schema does not
    # carry a miss-floor field — the "miss" bucket is implicit
    # (anything beyond rough_threshold). The miss_floor_factor is
    # surfaced as the lower bound of the miss bucket via M6's
    # bucket-counting logic (see _vol_scale_provenance.notes).
    # Build a fresh ToleranceProfile with scaled bands. We keep
    # absolute_floor at the same value (defines the floor itself);
    # max_range_width / sign_sensitive / sign_epsilon untouched.
    return ToleranceProfile(
        absolute_floor=floor,
        exact_threshold=near,  # "exact" same as "near" in vol-scaled
                                # mode (a numeric answer within 1σ
                                # is the user's "exact" expectation).
        near_threshold=near,
        rough_threshold=rough,
        exact_range_width=near,
        near_range_width=near,
        rough_range_width=rough,
        max_range_width=base_tolerance.max_range_width,
        sign_sensitive=base_tolerance.sign_sensitive,
        sign_epsilon=base_tolerance.sign_epsilon,
        vol_scale=None,  # already applied; clear to avoid re-scaling
        preset_name=base_tolerance.preset_name,
    )


def apply_vol_scaling_to_question_set(
    probe: Any,
    question_set: Any,
    data: "pd.DataFrame | dict[str, pd.DataFrame]",
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """v2.2.1 audit-fix Commit D: rewrite each QuestionSpec.tolerance
    to the vol-scaled form when its tolerance carries a vol_scale
    spec. Moved from orchestrator.py — this module is the canonical
    home for vol-related transforms.

    Returns ``(new_question_set, provenance_by_qid)``.

    Skips:
      - Questions whose tolerance is None.
      - Questions whose tolerance.vol_scale is None.
      - RankContinuationProbe (rank scoring uses Spearman ρ, not
        sigma-bands; vol_scale doesn't apply).

    For each scoped question:
      - KnowledgeProbe → point-in-time mode; sigma from bars
        strictly before qs.timestamp on the side's data.
      - ContinuationProbe → continuation mode; sigma from bars
        ending at the LAST context bar (held fixed across forward
        steps).
    """
    from dataclasses import replace as _dc_replace

    # Probe-class imports are late to avoid circular imports
    # (_continuation and _rank both depend on models.py).
    from ._continuation import ContinuationProbe
    from ._rank import RankContinuationProbe
    from .questions import QuestionSet

    if isinstance(probe, RankContinuationProbe):
        # Rank scoring doesn't use vol-scaled bands.
        return question_set, {}

    rewritten: list = []
    provenance: dict[str, dict[str, Any]] = {}

    for qs in question_set:
        if qs.tolerance is None or qs.tolerance.vol_scale is None:
            rewritten.append(qs)
            continue

        # data may be a DataFrame (single-symbol probes) or a dict
        # (RankContinuationProbe — but we excluded that above).
        if isinstance(data, dict):
            symbol_data = data.get(qs.symbol)
            if symbol_data is None:
                rewritten.append(qs)
                continue
        else:
            symbol_data = data

        spec = qs.tolerance.vol_scale
        if isinstance(probe, ContinuationProbe):
            ctx = qs.metadata.get("context_window")
            if not ctx:
                rewritten.append(qs)
                continue
            # v2.2.1 hotfix: previously `pd.Timestamp(ctx[-1]["index"])`
            # which crashed with KeyError on DataFrames whose index has
            # a non-None name (e.g. data.index.name == "Date"), because
            # `df.reset_index().to_dict(orient="records")` keys the
            # column by the index name, not the literal string "index".
            # Read the anchor timestamp directly off the question spec
            # — by ContinuationProbe construction, the context window
            # ends at the anchor bar, so qs.timestamp == last_ctx_ts.
            last_ctx_ts = qs.timestamp
            sigma, prov = estimate_sigma_for_continuation(
                symbol_data, last_ctx_ts,
                window=spec.window,
                method=spec.method,
                preset_name=qs.tolerance.preset_name,
                log_returns=spec.log_returns,
            )
        else:
            # Point-in-time probes (KnowledgeProbe).
            sigma, prov = estimate_sigma_for_pointintime(
                symbol_data, qs.timestamp,
                window=spec.window,
                method=spec.method,
                preset_name=qs.tolerance.preset_name,
                log_returns=spec.log_returns,
            )
        scaled_tol = apply_vol_scale(qs.tolerance, sigma)
        rewritten.append(_dc_replace(qs, tolerance=scaled_tol))
        provenance[qs.question_id] = prov

    return QuestionSet(rewritten), provenance


__all__ = [
    "apply_vol_scale",
    "apply_vol_scaling_to_question_set",
    "estimate_sigma",
    "estimate_sigma_for_continuation",
    "estimate_sigma_for_pointintime",
    "resolve_method",
]
