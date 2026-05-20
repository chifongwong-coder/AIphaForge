"""v2.2 M5: held-out synthetic anchor for the KnowledgeProbe pillar.

A fabricated OHLCV series matching real_data's volatility profile but
with no relationship to the real symbol. Used by ``knowledge_check``
(M6) to compute ``score_minus_anchor`` as the leakage estimate.

See ``docs/plans/v2.2-plan-r6.md`` § 6.

Three generation methods:
  - ``garch_resample`` (default): fit GJR-GARCH(1,1,1) for equity
    when n>=500 bars, else GARCH(1,1); simulate new returns,
    integrate to OHLCV.
  - ``block_bootstrap``: resample fixed-length blocks of real
    returns (preserves autocorrelation, destroys date alignment).
  - ``random_walk_volmatched``: iid Normal returns with σ matched
    to real σ. No autocorrelation; the ``unseen`` anchor whose vol
    clustering is independent of the real symbol's GARCH params.

Equity-vs-crypto auto-detector (per § 6.3.1):
  Three diagnostics — full corr(r_{t-1}, |r_t|) < -0.10 AND both
  half-correlations < -0.05 AND block-bootstrap CI of full corr
  excludes zero. classification_confidence is the studentized
  distance from -0.10.

GJR sample-size gate (per § 6.3): n >= 500 required for GJR; below,
fall back to GARCH. Convergence-failure cascade (per § 6.4):
GJR -> GARCH -> random_walk_volmatched.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

# Block-bootstrap parameters per § 6.3.1.
_BLOCK_SIZE = 20      # daily-frequency convention
_N_RESAMPLES = 500    # MC SE ~2% on 0.95 quantile
_GJR_MIN_BARS = 500   # Determinism gate per § 6.3


def _stable_label(seed: int) -> str:
    """Deterministic synthetic-ticker label from the seed.

    Format: ``SYN_<8-hex-chars>``. Never contains a real ticker.
    """
    h = hashlib.sha256(f"SYN_seed_{seed}".encode("ascii")).hexdigest()
    return f"SYN_{h[:8].upper()}"


def _leverage_corr(returns: np.ndarray) -> float:
    """corr(r_{t-1}, |r_t|). Negative on equity (leverage effect)."""
    if len(returns) < 3:
        return float("nan")
    lagged = returns[:-1]
    abs_curr = np.abs(returns[1:])
    if np.std(lagged) == 0 or np.std(abs_curr) == 0:
        return 0.0
    return float(np.corrcoef(lagged, abs_curr)[0, 1])


def _block_bootstrap_corr_ci(
    returns: np.ndarray,
    *,
    block_size: int,
    n_resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Block-bootstrap percentile CI of leverage_corr. Returns
    (lo, hi, se).
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    n = len(returns)
    n_blocks = max(1, n // block_size)
    samples = []
    for _ in range(n_resamples):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        boot = np.concatenate(
            [returns[s: s + block_size] for s in starts]
        )[:n]
        samples.append(_leverage_corr(boot))
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return float("nan"), float("nan"), float("nan")
    alpha = 1.0 - confidence
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1.0 - alpha / 2))
    se = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
    return lo, hi, se


def _detect_equity_class(
    returns: np.ndarray,
    *,
    seed: int,
) -> tuple[bool, dict[str, Any]]:
    """Auto-detect equity (vs crypto) using three diagnostics.

    Returns (is_equity, provenance_dict) per § 6.3.1.
    """
    full = _leverage_corr(returns)
    half_size = len(returns) // 2
    first_half = _leverage_corr(returns[:half_size])
    second_half = _leverage_corr(returns[half_size:])
    ci_lo, ci_hi, se = _block_bootstrap_corr_ci(
        returns, block_size=_BLOCK_SIZE,
        n_resamples=_N_RESAMPLES, seed=seed,
    )

    # Three-diagnostic gate.
    sign_stable = (
        first_half < -0.05 and second_half < -0.05
    )
    ci_excludes_zero = math.isfinite(ci_hi) and ci_hi < 0.0
    is_equity = (
        full < -0.10
        and sign_stable
        and ci_excludes_zero
    )

    classification_confidence = (
        ((-0.10 - full) / se) if (math.isfinite(se) and se > 0)
        else float("nan")
    )

    return is_equity, {
        "full_corr": full,
        "first_half_corr": first_half,
        "second_half_corr": second_half,
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "bootstrap_se": se,
        "classification_confidence": classification_confidence,
        "block_size_used": _BLOCK_SIZE,
        "n_resamples_used": _N_RESAMPLES,
        "is_equity": is_equity,
    }


def _fit_garch_with_fallback(
    returns: np.ndarray,
    gate_says_gjr: bool,
) -> tuple[Any, str]:
    """Fit GARCH/GJR with the documented fallback chain:
    GJR (if gate) -> GARCH(1,1) -> raise (caller falls back to
    random walk).

    Returns (fitted_result, model_name). model_name is one of
    "gjr_garch", "garch", or "convergence_failure" (caller catches
    None and falls back).
    """
    try:
        from arch import arch_model  # type: ignore[import-not-found]
    except ImportError:
        return None, "arch_unavailable"

    if gate_says_gjr:
        try:
            am = arch_model(
                returns * 100.0,  # arch expects pct returns
                mean="Zero", vol="GARCH", p=1, o=1, q=1,
                rescale=False,
            )
            res = am.fit(disp="off", show_warning=False)
            return res, "gjr_garch"
        except Exception:
            pass  # fall through to symmetric GARCH

    try:
        am = arch_model(
            returns * 100.0,
            mean="Zero", vol="GARCH", p=1, q=1,
            rescale=False,
        )
        res = am.fit(disp="off", show_warning=False)
        return res, "garch"
    except Exception:
        return None, "convergence_failure"


def _simulate_garch_returns(
    fitted_result: Any,
    n: int,
    seed: int,
) -> np.ndarray:
    """Simulate n returns from a fitted GARCH model."""
    sim = fitted_result.model.simulate(
        params=fitted_result.params,
        nobs=n,
        random_state=np.random.RandomState(seed),
    )
    # arch returns pct returns; convert back to fractional
    return np.asarray(sim["data"].values) / 100.0


def _build_ohlcv_from_returns(
    returns: np.ndarray,
    real_data: pd.DataFrame,
    *,
    label: str,
    hl_spread_source: str = "real_bar_ratio",
    spread_rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Integrate a return series into an OHLCV DataFrame matching
    the real_data index. Open/high/low are derived from the real
    bar's ``(high - low) / close`` spread RATIO so that intra-bar
    volatility tracks the real bar's range (v2.8.1 H5 — the v2.8.0
    constant ±1.5% produced deterministic noise that broke vol
    estimators).

    Accuracy notes (v2.8.1 post-review):

    - The synthetic spread ratio ``(H_syn - L_syn) / close_syn`` is
      bar-for-bar IDENTICAL to the real spread ratio. Tests that
      compare those values pass to machine precision.
    - Parkinson ``(ln(H/L))^2`` on the synthetic series is
      APPROXIMATELY equal to the real Parkinson, not exact. The
      synthetic bar centers H/L symmetrically around close
      (``H = close*(1 + spread/2)``, ``L = close*(1 - spread/2)``);
      a real bar is not in general symmetric around close. The
      deviation is small at typical equity spreads (<1% relative
      error at spread ≤ 5%) and dominated by the v2.8.0 constant
      ±1.5% bias it replaces, but a user computing Parkinson with
      strict accuracy guarantees should reach for ``real_data``
      directly. v2.9 may switch to ``half = spread/(2 + spread)``
      (Parkinson-exact reconstruction) if this matters.

    Parameters
    ----------
    hl_spread_source
        ``"real_bar_ratio"`` (default) — use the real bar's H/L
        spread per timestamp. Bar-for-bar identical spread RATIO
        series.
        ``"real_distribution_shuffled"`` — permute the real spread
        ratios with ``spread_rng``. Destroys bar-level autocorrelation
        while preserving the marginal distribution; used as a
        leak-verification opt-out (``autocorr(spread) ≈ 0`` after).
    spread_rng
        Required when ``hl_spread_source == "real_distribution_shuffled"``.
        A fresh ``Generator`` constructed from the anchor seed in
        ``build_synthetic_anchor`` — independent of any future
        close-path stochasticity (defensive future-proofing per v2.1.3).
    """
    base_close = float(real_data["close"].iloc[0])
    closes = base_close * np.exp(np.cumsum(returns))
    if len(closes) < len(real_data):
        # Pad to match index length
        closes = np.concatenate(
            [closes, [closes[-1]] * (len(real_data) - len(closes))]
        )
    closes = closes[: len(real_data)]

    # v2.8.1 H5: derive H/L from the real bar's spread ratio.
    real_high = real_data["high"].astype(float).to_numpy()
    real_low = real_data["low"].astype(float).to_numpy()
    real_close = real_data["close"].astype(float).to_numpy()
    spread_ratio = (real_high - real_low) / np.clip(
        np.abs(real_close), 1e-9, None
    )
    if hl_spread_source == "real_distribution_shuffled":
        if spread_rng is None:
            raise ValueError(
                "hl_spread_source='real_distribution_shuffled' requires a "
                "spread_rng argument"
            )
        spread_ratio = spread_rng.permutation(spread_ratio)
    elif hl_spread_source != "real_bar_ratio":
        raise ValueError(
            f"hl_spread_source must be 'real_bar_ratio' or "
            f"'real_distribution_shuffled'; got {hl_spread_source!r}"
        )

    opens = closes
    half = spread_ratio / 2.0
    highs = closes * (1.0 + half)
    lows = closes * (1.0 - half)
    # Enforce OHLC ordering. opens == closes today, so the clamp is
    # a no-op in practice; included for defensive correctness.
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": real_data["volume"].astype(float).to_numpy(),
        },
        index=real_data.index,
    )
    df.attrs["synthetic_label"] = label
    # v2.8.1 M2: key-assignment, NOT wholesale dict replace, so any
    # vol_scale / n_anchor_bars keys set upstream survive.
    if "vol_scale_provenance" not in df.attrs:
        df.attrs["vol_scale_provenance"] = {}
    df.attrs["vol_scale_provenance"]["hl_spread_source"] = hl_spread_source
    return df


def _block_bootstrap_returns(
    real_returns: np.ndarray,
    n: int,
    seed: int,
    block_size: int = _BLOCK_SIZE,
) -> np.ndarray:
    """Block-bootstrap resample of real_returns to length n."""
    rng = np.random.Generator(np.random.PCG64(seed))
    n_blocks = (n + block_size - 1) // block_size
    starts = rng.integers(
        0, len(real_returns) - block_size + 1, size=n_blocks
    )
    boot = np.concatenate(
        [real_returns[s: s + block_size] for s in starts]
    )
    return boot[:n]


def _random_walk_volmatched_returns(
    real_returns: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    """iid Normal returns with σ matched to real σ. No autocorrelation."""
    rng = np.random.Generator(np.random.PCG64(seed))
    sigma = float(np.std(real_returns, ddof=1))
    return rng.normal(0.0, sigma, n)


def build_synthetic_anchor(
    real_data: pd.DataFrame,
    seed: int,
    *,
    asset_class: str = "auto",
    method: str = "garch_resample",
    label: str | None = None,
    hl_spread_source: str = "real_bar_ratio",
) -> pd.DataFrame:
    """Return a fabricated OHLCV series matching real_data's
    volatility profile but with no relationship to the real symbol.

    Parameters
    ----------
    real_data
        Real OHLCV DataFrame; the synthetic series will match
        ``real_data.index`` length and frequency.
    seed
        Deterministic seed for the random generator. The synthetic
        ticker label is also derived from the seed so users can
        rerun anchor probes without re-issuing prompts.
    asset_class
        ``"equity"``, ``"crypto"``, or ``"auto"``. ``"auto"`` runs
        the three-diagnostic detector per § 6.3.1.
    method
        ``"garch_resample"`` (default), ``"block_bootstrap"``, or
        ``"random_walk_volmatched"``.
    label
        Optional override for the synthetic ticker label. Default
        is ``SYN_<8-hex>`` derived from ``seed``.
    hl_spread_source
        v2.8.1: ``"real_bar_ratio"`` (default) propagates the real
        bar's H/L spread ratio onto the synthetic series so that
        Parkinson / Garman-Klass intra-bar volatility tracks reality.
        ``"real_distribution_shuffled"`` permutes the real spread
        ratios via a seed-derived RNG, destroying bar-level
        autocorrelation while preserving the marginal distribution
        — used as a light-weight leak-mitigation opt-out. True
        symbol anonymization should go through ``SymbolMasker``.

    Returns
    -------
    pd.DataFrame
        Synthetic OHLCV with same shape and index as ``real_data``.
        Provenance is in ``df.attrs``: ``synthetic_label``,
        ``vol_model_chosen``, ``vol_model_provenance``,
        ``leverage_corr_provenance`` (when ``asset_class="auto"``).
    """
    if hl_spread_source not in ("real_bar_ratio", "real_distribution_shuffled"):
        raise ValueError(
            f"hl_spread_source must be 'real_bar_ratio' or "
            f"'real_distribution_shuffled'; got {hl_spread_source!r}"
        )

    # v2.8.1 post-review fix: H/L spread derivation needs the real bar's
    # high/low columns. Validate up front so a missing column surfaces
    # as a clean ValueError, not a deep KeyError inside
    # _build_ohlcv_from_returns.
    missing_cols = [c for c in ("close", "high", "low", "volume")
                    if c not in real_data.columns]
    if missing_cols:
        raise ValueError(
            f"build_synthetic_anchor: real_data missing required "
            f"column(s) {missing_cols}. v2.8.1 derives synthetic H/L "
            f"from the real bar's (H-L)/close spread ratio, so all of "
            f"open/high/low/close/volume must be present."
        )

    if label is None:
        label = _stable_label(seed)

    closes = real_data["close"].astype(float).to_numpy()
    if len(closes) < 3:
        raise ValueError(
            "real_data must have >= 3 bars for synthetic anchor"
        )
    real_returns = np.log(closes[1:] / closes[:-1])

    provenance: dict[str, Any] = {
        "method_requested": method,
        "asset_class_requested": asset_class,
        "seed": int(seed),
        "n_bars": len(real_data),
    }

    if asset_class == "auto":
        is_equity, leverage_prov = _detect_equity_class(
            real_returns, seed=seed,
        )
        provenance["leverage_corr_provenance"] = leverage_prov
        resolved_class = "equity" if is_equity else "crypto"
    elif asset_class in ("equity", "crypto"):
        resolved_class = asset_class
    else:
        raise ValueError(
            f"asset_class must be 'auto', 'equity', or 'crypto'; "
            f"got {asset_class!r}"
        )
    provenance["asset_class_resolved"] = resolved_class

    if method == "block_bootstrap":
        sim_returns = _block_bootstrap_returns(
            real_returns, n=len(real_data) - 1, seed=seed,
        )
        provenance["vol_model_chosen"] = "block_bootstrap"
    elif method == "random_walk_volmatched":
        sim_returns = _random_walk_volmatched_returns(
            real_returns, n=len(real_data) - 1, seed=seed,
        )
        provenance["vol_model_chosen"] = "random_walk_volmatched"
    elif method == "garch_resample":
        # GJR sample-size gate per § 6.3
        gate_says_gjr = (
            len(real_data) >= _GJR_MIN_BARS
            and resolved_class == "equity"
        )
        if not gate_says_gjr and len(real_data) < _GJR_MIN_BARS:
            provenance["gjr_gate_note"] = (
                f"GJR not attempted (n={len(real_data)} < "
                f"{_GJR_MIN_BARS})"
            )
        fitted, model_name = _fit_garch_with_fallback(
            real_returns, gate_says_gjr=gate_says_gjr,
        )
        provenance["vol_model_chosen"] = model_name
        if fitted is None:
            # Fallback chain bottom: random walk vol-matched.
            sim_returns = _random_walk_volmatched_returns(
                real_returns, n=len(real_data) - 1, seed=seed,
            )
            provenance["vol_model_chosen"] = (
                "random_walk_volmatched_fallback"
            )
            provenance["fallback_reason"] = (
                "GARCH/GJR convergence failure or arch package "
                "unavailable"
            )
        else:
            try:
                sim_returns = _simulate_garch_returns(
                    fitted, n=len(real_data) - 1, seed=seed,
                )
            except Exception:
                sim_returns = _random_walk_volmatched_returns(
                    real_returns, n=len(real_data) - 1, seed=seed,
                )
                provenance["vol_model_chosen"] = (
                    "random_walk_volmatched_fallback"
                )
                provenance["fallback_reason"] = (
                    "GARCH simulation failure"
                )
    else:
        raise ValueError(
            f"unknown method {method!r}; must be 'garch_resample', "
            f"'block_bootstrap', or 'random_walk_volmatched'"
        )

    # v2.8.1 H5 / M1: construct a SEPARATE rng for spread permutation,
    # independent of any future close-path stochasticity. Today the
    # close-path is deterministic but the separation is defensive
    # future-proofing.
    spread_rng = np.random.Generator(np.random.PCG64(seed))
    df = _build_ohlcv_from_returns(
        sim_returns, real_data, label=label,
        hl_spread_source=hl_spread_source, spread_rng=spread_rng,
    )
    df.attrs["vol_model_chosen"] = provenance["vol_model_chosen"]
    df.attrs["vol_model_provenance"] = provenance
    if "leverage_corr_provenance" in provenance:
        df.attrs["leverage_corr_provenance"] = (
            provenance["leverage_corr_provenance"]
        )
    return df


__all__ = [
    "build_synthetic_anchor",
]
