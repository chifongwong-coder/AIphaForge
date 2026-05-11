"""v2.2 M4: RankContinuationProbe + RankAnswer + N-dependent quantile
lookup with tie-corrected rho.

See ``docs/plans/v2.2-plan-r6.md`` § 5.

Key design (per r6 review, Q-B1):
- Tie correction is applied to the OBSERVED rho (matching scipy's
  spearmanr formula), NOT to N for table-row lookup. The null
  distribution of rho at N=20 with one tied pair is NOT identical
  to the null at N=18 with no ties.
- Cutoffs are quantile-based (default 0.99 / 0.90 / 0.75) of the
  null Spearman distribution at the actual N.
- Lookup uses Gaussian closed form for N>20; small-N permutation
  table is computed lazily (in-memory, deterministic).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from aiphaforge.probes.models import QuestionSpec
from aiphaforge.probes.questions import (
    QuestionSet,
    _make_question_id,
)

# ---------- Symbol normalization presets ----------


@dataclass(frozen=True)
class NormalizationSpec:
    """How to canonicalize a raw symbol string for comparison."""

    nfc: bool = True
    strip_whitespace: bool = True
    uppercase: bool = True
    dash_to_dot: bool = False  # US-equity convention only


_NORMALIZATION_PRESETS: dict[str, NormalizationSpec] = {
    "us_equity": NormalizationSpec(
        nfc=True, strip_whitespace=True, uppercase=True,
        dash_to_dot=True,
    ),
    "crypto": NormalizationSpec(
        nfc=True, strip_whitespace=True, uppercase=True,
        dash_to_dot=False,  # BTC-USD venue identifier preserved
    ),
    "futures": NormalizationSpec(
        nfc=True, strip_whitespace=True, uppercase=True,
        dash_to_dot=False,
    ),
    "passthrough": NormalizationSpec(
        nfc=True, strip_whitespace=False, uppercase=False,
        dash_to_dot=False,
    ),
}


def normalize_symbol(raw: str, preset: str = "passthrough") -> str:
    """Canonicalize a raw symbol string via a documented preset.

    The preset name is part of parsing_schema_description and
    contributes to parsing_schema_hash, so users opting into
    different normalization get different hashes (no silent venue
    merges).

    ``passthrough`` is the default to force users to opt into a
    preset whose semantics they understand. The dash->dot rule is
    US-equity-specific (BRK-B → BRK.B); crypto and futures preserve
    dash to avoid collapsing venue identifiers (BTC-USD ≠ BTC.USD).
    """
    if preset not in _NORMALIZATION_PRESETS:
        raise ValueError(
            f"unknown normalization preset {preset!r}; available: "
            f"{sorted(_NORMALIZATION_PRESETS)}"
        )
    spec = _NORMALIZATION_PRESETS[preset]
    s = raw
    if spec.nfc:
        import unicodedata
        s = unicodedata.normalize("NFC", s)
    if spec.strip_whitespace:
        s = s.strip()
    if spec.uppercase:
        s = s.upper()
    if spec.dash_to_dot:
        s = s.replace("-", ".")
    return s


# ---------- RankAnswer ----------


@dataclass(frozen=True)
class RankAnswer:
    """Parsed rank response with explicit partial/tie/extra semantics.

    Scoring rule (locked v2.2):
      - ``omitted_symbols`` non-empty OR ``extra_symbols`` non-empty
        → bucket = "invalid". Both are forms of LLM disobedience;
        symmetric scoring eliminates the perverse incentive (a
        hallucinator scoring better than a partial-answerer).
      - Ties resolved by mid-rank (Spearman convention).
      - Singleton tie groups rejected at construction (semantically
        empty).

    See § 5.2.1.
    """

    ranking: tuple[str, ...]
    omitted_symbols: frozenset[str] = field(default_factory=frozenset)
    extra_symbols: frozenset[str] = field(default_factory=frozenset)
    tie_groups: tuple[frozenset[str], ...] = ()

    def __post_init__(self):
        if len(set(self.ranking)) != len(self.ranking):
            raise ValueError("ranking contains duplicate symbols")
        seen: set[str] = set()
        for group in self.tie_groups:
            if len(group) < 2:
                raise ValueError(
                    f"tie group must have >=2 members; got "
                    f"{sorted(group)}"
                )
            overlap = group & seen
            if overlap:
                raise ValueError(
                    f"tie_groups not a partition: {sorted(overlap)} "
                    f"appears in multiple groups"
                )
            if not group.issubset(set(self.ranking)):
                raise ValueError(
                    f"tie group contains symbols not in ranking: "
                    f"{sorted(group - set(self.ranking))}"
                )
            seen |= group


# ---------- RankCutoffSpec ----------


@dataclass(frozen=True)
class RankCutoffSpec:
    """Bucket boundaries expressed as quantiles of the null Spearman
    distribution at the question's actual N. rho thresholds are
    looked up at scoring time.
    """

    exact_quantile: float = 0.99
    near_quantile: float = 0.90
    rough_quantile: float = 0.75

    def __post_init__(self):
        # Lock max supported quantile to 0.99 per § 5.2.3 — 10^6 MC
        # samples give SE ~0.001 at q=0.99; q=0.999 is uncalibrated.
        for name in (
            "exact_quantile", "near_quantile", "rough_quantile",
        ):
            v = getattr(self, name)
            if not (0.0 < v <= 0.99):
                raise ValueError(
                    f"{name}={v} out of supported range (0, 0.99]"
                )
        if not (
            self.exact_quantile > self.near_quantile
            > self.rough_quantile
        ):
            raise ValueError(
                "quantiles must be strictly decreasing: "
                f"exact={self.exact_quantile} > "
                f"near={self.near_quantile} > "
                f"rough={self.rough_quantile}"
            )


# ---------- Spearman tie-corrected rho ----------


def tie_corrected_spearman(
    x_ranks: np.ndarray,
    y_ranks: np.ndarray,
) -> float:
    """Compute the tie-corrected Spearman rho.

    Uses the standard Kendall (1948) / Olds (1949) form, equivalent
    to scipy's spearmanr under the hood. Matches scipy to ~1e-12.

    Formula:
      rho = ((N^3 - N) - 6*sum(d^2) - (T_x + T_y)/2)
            / sqrt( ((N^3 - N) - T_x) * ((N^3 - N) - T_y) )

    where T = sum_g (t_g^3 - t_g) over tie groups in each ranking.
    """
    n = len(x_ranks)
    if n != len(y_ranks):
        raise ValueError("rank arrays must have equal length")
    if n < 2:
        return 0.0

    def _tie_term(ranks: np.ndarray) -> float:
        # Count tie-group sizes by integer bucketing of mid-ranks.
        # Mid-ranks of tied pairs share the same value.
        unique, counts = np.unique(ranks, return_counts=True)
        return float(
            np.sum((counts.astype(float) ** 3) - counts.astype(float))
        )

    d = x_ranks.astype(float) - y_ranks.astype(float)
    sum_d2 = float(np.sum(d ** 2))
    n3_n = float(n ** 3 - n)
    t_x = _tie_term(x_ranks)
    t_y = _tie_term(y_ranks)
    numerator = n3_n - 6.0 * sum_d2 - 0.5 * (t_x + t_y)
    denom = math.sqrt(max((n3_n - t_x), 0.0) * max((n3_n - t_y), 0.0))
    if denom == 0:
        return 0.0
    return numerator / denom


def midranks(values: Sequence[float]) -> np.ndarray:
    """Return mid-ranks of values (1-indexed, ties get average rank)."""
    arr = np.asarray(values, dtype=float)
    return pd.Series(arr).rank(method="average").to_numpy()


# ---------- Null-quantile lookup ----------


def _spearman_null_quantile(n: int, q: float, seed: int = 42) -> float:
    """rho threshold s.t. P(|rho_null| >= rho_threshold) = 1 - q.

    For N <= 20: Monte Carlo with deterministic seed (10^4 samples
    is plenty for q in [0.75, 0.99]; cached per-N).
    For N > 20: Gaussian closed form, sigma = 1/sqrt(N-1).
    """
    if n < 3:
        return 0.0
    if n > 20:
        # Gaussian: rho_q = z_(q) / sqrt(N-1), where z is the
        # quantile of standard normal of |rho| under H0.
        from math import sqrt
        try:
            from scipy.stats import norm  # type: ignore[import-not-found]
            z = float(norm.ppf((1.0 + q) / 2.0))
        except ImportError:
            # Approximation table for common q values
            z_table = {0.99: 2.576, 0.95: 1.960, 0.90: 1.645, 0.75: 1.150}
            z = z_table.get(q, 1.645)
        return z / sqrt(n - 1)
    # Small-N: Monte Carlo. Cached on first call per N to avoid
    # recomputation.
    return _spearman_mc_quantile(n, q, seed=seed)


# Module-level cache.
_MC_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _spearman_mc_quantile(n: int, q: float, seed: int = 42) -> float:
    """Cached Monte Carlo of |rho| under the no-tie null."""
    cache_key = (n, seed)
    if cache_key not in _MC_CACHE:
        rng = np.random.Generator(np.random.PCG64(seed))
        n_samples = 10_000
        samples = np.empty(n_samples)
        ranks_canonical = np.arange(1, n + 1, dtype=float)
        for i in range(n_samples):
            shuf = rng.permutation(ranks_canonical)
            samples[i] = abs(
                tie_corrected_spearman(ranks_canonical, shuf)
            )
        samples.sort()
        _MC_CACHE[cache_key] = samples
    samples = _MC_CACHE[cache_key]
    idx = int(q * len(samples))
    if idx >= len(samples):
        idx = len(samples) - 1
    return float(samples[idx])


# ---------- RankContinuationProbe ----------


class RankContinuationProbe:
    """Cross-sectional continuation probe.

    Asks the LLM to rank N symbols by their forward return at
    anchor_ts + forward_horizon, given each symbol's prior
    context_bars of OHLCV data.

    See § 5.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        context_bars: int,        # NO DEFAULT
        forward_horizon: int,     # NO DEFAULT
        rank_metric: str = "log_return",
    ):
        if not isinstance(context_bars, int) or context_bars < 2:
            raise ValueError(
                "context_bars must be int >= 2"
            )
        if not isinstance(forward_horizon, int) or forward_horizon < 1:
            raise ValueError(
                "forward_horizon must be int >= 1"
            )
        if rank_metric not in {"log_return", "simple_return"}:
            raise ValueError(
                f"rank_metric must be 'log_return' or 'simple_return'; "
                f"got {rank_metric!r}"
            )
        if len(symbols) < 2:
            raise ValueError(
                f"RankContinuationProbe needs >= 2 symbols; got "
                f"{len(symbols)}"
            )
        self.symbols = tuple(symbols)
        self.context_bars = context_bars
        self.forward_horizon = forward_horizon
        self.rank_metric = rank_metric

    def build(
        self,
        data_dict: dict[str, pd.DataFrame],
        timestamps: Sequence[pd.Timestamp],
    ) -> QuestionSet:
        """Build one QuestionSpec per anchor timestamp containing
        all N symbols' context windows. Anchors too close to either
        edge of any symbol's data are silently skipped.
        """
        specs: list[QuestionSpec] = []
        for ts in timestamps:
            valid = True
            ctx_by_symbol: dict[str, list[dict]] = {}
            target_returns: dict[str, float] = {}
            for sym in self.symbols:
                if sym not in data_dict:
                    valid = False
                    break
                data = data_dict[sym]
                if ts not in data.index:
                    valid = False
                    break
                idx_loc = data.index.get_loc(ts)
                if idx_loc - self.context_bars + 1 < 0:
                    valid = False
                    break
                if idx_loc + self.forward_horizon >= len(data):
                    valid = False
                    break
                if idx_loc + self.forward_horizon - 1 < 0:
                    valid = False
                    break
                ctx = data.iloc[
                    idx_loc - self.context_bars + 1: idx_loc + 1
                ]
                ctx_by_symbol[sym] = ctx.reset_index().to_dict(
                    orient="records"
                )
                c_h = float(
                    data.iloc[idx_loc + self.forward_horizon]["close"]
                )
                c_h_minus_1 = float(
                    data.iloc[
                        idx_loc + self.forward_horizon - 1
                    ]["close"]
                )
                if c_h_minus_1 <= 0 or c_h <= 0:
                    valid = False
                    break
                if self.rank_metric == "log_return":
                    target_returns[sym] = math.log(c_h / c_h_minus_1)
                else:
                    target_returns[sym] = (c_h - c_h_minus_1) / c_h_minus_1
            if not valid:
                continue

            # Truth: list of symbols in descending order of target return
            ranked = sorted(
                self.symbols, key=lambda s: -target_returns[s]
            )
            specs.append(
                QuestionSpec(
                    question_id=_make_question_id(
                        ",".join(self.symbols), ts,
                        f"rank_continuation_h{self.forward_horizon}",
                    ),
                    symbol="MULTI",
                    timestamp=ts,
                    template_id="rank_continuation",
                    answer_type="numeric",  # answer is a ranking;
                                            # parsed by parse_rank_answer
                    prompt_text=(
                        f"Given the prior {self.context_bars} bars "
                        f"for each of {list(self.symbols)}, rank "
                        f"the symbols best-to-worst by {self.rank_metric} "
                        f"at t+{self.forward_horizon}."
                    ),
                    choices=None,
                    truth_value=tuple(ranked),
                    tolerance=None,
                    metadata={
                        "context_window_by_symbol": ctx_by_symbol,
                        "context_bars": self.context_bars,
                        "forward_horizon": self.forward_horizon,
                        "rank_metric": self.rank_metric,
                        "n_symbols": len(self.symbols),
                        "anchor_timestamp": ts.isoformat(),
                    },
                )
            )
        return QuestionSet(specs)


# ---------- Score a rank answer ----------


def score_rank_answer(
    rank_answer: RankAnswer,
    truth_ranking: Sequence[str],
    cutoffs: Optional[RankCutoffSpec] = None,
) -> tuple[str, dict]:
    """Score a single RankAnswer against the truth ranking.

    Returns (band, provenance) where band ∈
    {exact, near, rough, miss, invalid}.
    """
    if cutoffs is None:
        cutoffs = RankCutoffSpec()

    # Symmetric omitted/extra → invalid (per § 5.2.1).
    if rank_answer.omitted_symbols or rank_answer.extra_symbols:
        return "invalid", {
            "reason": "omitted_or_extra_symbols",
            "omitted_count": len(rank_answer.omitted_symbols),
            "extra_count": len(rank_answer.extra_symbols),
        }

    # Build mid-ranks for both sides.
    n = len(truth_ranking)
    truth_idx = {s: n - i for i, s in enumerate(truth_ranking)}  # 1=best
    user_ranking = list(rank_answer.ranking)
    user_idx = {s: n - i for i, s in enumerate(user_ranking)}

    # Apply tie-group mid-rank correction in the user ranking.
    # For each tie group, replace each member's rank with the average
    # of their assigned ranks.
    if rank_answer.tie_groups:
        for group in rank_answer.tie_groups:
            avg = sum(user_idx[s] for s in group) / len(group)
            for s in group:
                user_idx[s] = avg

    truth_arr = np.array([truth_idx[s] for s in truth_ranking])
    user_arr = np.array([user_idx[s] for s in truth_ranking])

    rho = tie_corrected_spearman(truth_arr, user_arr)

    # Look up rho thresholds at the actual N.
    rho_exact = _spearman_null_quantile(n, cutoffs.exact_quantile)
    rho_near = _spearman_null_quantile(n, cutoffs.near_quantile)
    rho_rough = _spearman_null_quantile(n, cutoffs.rough_quantile)

    if rho >= rho_exact:
        band = "exact"
    elif rho >= rho_near:
        band = "near"
    elif rho >= rho_rough:
        band = "rough"
    else:
        band = "miss"

    return band, {
        "rho": rho,
        "n": n,
        "rho_exact_threshold": rho_exact,
        "rho_near_threshold": rho_near,
        "rho_rough_threshold": rho_rough,
    }


__all__ = [
    "NormalizationSpec",
    "RankAnswer",
    "RankContinuationProbe",
    "RankCutoffSpec",
    "midranks",
    "normalize_symbol",
    "score_rank_answer",
    "tie_corrected_spearman",
]
