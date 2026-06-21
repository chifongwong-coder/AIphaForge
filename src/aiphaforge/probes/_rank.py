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
import threading as _threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
import pandas as pd

from aiphaforge.probes.models import QuestionSpec
from aiphaforge.probes.questions import (
    QuestionSet,
    _make_question_id,
)

# Rank-correlation primitives now live in aiphaforge.stats (v2.9.0
# hoist). Re-imported here so internal callers and the historical
# ``from aiphaforge.probes._rank import midranks, tie_corrected_spearman``
# path keep working; both stay in this module's ``__all__``.
from aiphaforge.stats import midranks, tie_corrected_spearman  # noqa: F401

if TYPE_CHECKING:
    from aiphaforge.probes.models import AnswerRecord

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

    v2.2.1 (#2 + #4):
      - ``raw_ranking`` preserves the user's pre-normalization
        submission for paper-grade audit.
      - ``parser_used`` records which path produced this RankAnswer
        — one of: ``user_provided``, ``user_provided_list``,
        ``user_provided_dict``, ``json_array``, ``json_object``,
        ``json_array_codefence``, ``numbered``, ``bulleted_dash``,
        ``bulleted_asterisk``, ``comma``, ``newline``, or None.

    See § 5.2.1.
    """

    ranking: tuple[str, ...]
    omitted_symbols: frozenset[str] = field(default_factory=frozenset)
    extra_symbols: frozenset[str] = field(default_factory=frozenset)
    tie_groups: tuple[frozenset[str], ...] = ()
    raw_ranking: tuple[str, ...] = ()
    parser_used: Optional[str] = None

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
# ``tie_corrected_spearman`` and ``midranks`` moved to aiphaforge.stats
# (v2.9.0 hoist); imported above and re-exported via ``__all__``.


# ---------- Null-quantile lookup ----------


def _spearman_null_quantile(n: int, q: float, seed: int = 42) -> float:
    """rho threshold s.t. P(|rho_null| >= rho_threshold) = 1 - q.

    v2.2.1 #6: looked up from precomputed table at
    ``src/aiphaforge/probes/_rank_null.npz``:
      - N ∈ [3, 10]: exact enumeration of all N! permutations.
      - N ∈ [11, 20]: Monte Carlo with 10⁶ samples per N,
        PCG64-seeded via SeedSequence(20260512).spawn(10).
      - N > 20: Gaussian closed form (asymptotic).

    The table is generated once by
    ``scripts/generate_rank_null_table.py`` and shipped as a
    release artifact. Lazy-loaded into module state on first use.
    """
    if n < 3:
        return 0.0
    if n > 20:
        from math import sqrt
        try:
            from scipy.stats import norm  # type: ignore[import-not-found]
            z = float(norm.ppf((1.0 + q) / 2.0))
        except ImportError:
            z_table = {0.99: 2.576, 0.95: 1.960, 0.90: 1.645, 0.75: 1.150}
            z = z_table.get(q, 1.645)
        return z / sqrt(n - 1)
    # 3 <= n <= 20: lookup table.
    table = _load_rank_null_table()
    samples = table[f"abs_rho_n{n}"]
    idx = int(q * len(samples))
    if idx >= len(samples):
        idx = len(samples) - 1
    return float(samples[idx])


# Module-level cache: loaded once on first call.
_RANK_NULL_TABLE: dict[str, np.ndarray] | None = None
# v2.2.2 Commit C: lock guarding the cache check-and-set so concurrent
# callers don't both pass the `is None` check and both load the table.
# Per-call lock acquisition is ~30ns when uncontested; negligible on
# the probe hot path.
_RANK_NULL_TABLE_LOCK = _threading.Lock()


def _load_rank_null_table() -> dict[str, np.ndarray]:
    """Lazy-load the precomputed rank-null table.

    v2.2.2 Commit C: thread-safe via a module-level lock with
    double-checked locking, and FD-safe via a ``with np.load(...)``
    context manager. The arrays are eagerly copied out of the NpzFile
    so the cache holds plain ndarrays even after the underlying file
    handle closes. (NpzFile in the non-mmap default mode already
    materializes each array on access, so the eager-copy is mostly
    defensive — if a future numpy version switches the default to
    `mmap_mode='r'`, the explicit copy keeps the cached arrays valid.)
    """
    global _RANK_NULL_TABLE
    # Fast path: already loaded, no lock needed.
    if _RANK_NULL_TABLE is not None:
        return _RANK_NULL_TABLE
    with _RANK_NULL_TABLE_LOCK:
        # Re-check inside the lock (another thread may have loaded
        # the table while we were waiting).
        if _RANK_NULL_TABLE is None:
            from pathlib import Path
            npz_path = (
                Path(__file__).resolve().parent / "_rank_null.npz"
            )
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"rank-null table not found at {npz_path}; "
                    "regenerate via scripts/generate_rank_null_table.py"
                )
            with np.load(npz_path) as data:
                _RANK_NULL_TABLE = {
                    key: np.asarray(data[key]) for key in data.files
                }
        return _RANK_NULL_TABLE


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
        normalization_preset: str = "passthrough",
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
        if normalization_preset not in _NORMALIZATION_PRESETS:
            raise ValueError(
                f"normalization_preset must be in "
                f"{sorted(_NORMALIZATION_PRESETS)}; got "
                f"{normalization_preset!r}"
            )
        # v2.2.1 #4: normalize symbols at probe-build time so truth
        # ranking and user submissions compare on the same canonical
        # form. The preset name is also surfaced on the report so
        # paper readers see which canonicalization was used.
        self.normalization_preset = normalization_preset
        self.symbols = tuple(
            normalize_symbol(s, normalization_preset) for s in symbols
        )
        self.raw_symbols = tuple(symbols)  # pre-normalization for audit
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
            # v2.2.1 #4: data_dict is keyed by RAW user symbols
            # (whatever the user passed in). We look up data by raw,
            # but produce truth + context dicts in the canonical
            # (normalized) form so user RankAnswers compare correctly.
            for raw_sym, sym in zip(self.raw_symbols, self.symbols):
                if raw_sym not in data_dict:
                    valid = False
                    break
                data = data_dict[raw_sym]
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


def _is_sequence_of_str(x: Any) -> bool:
    """v2.2.1 #2 path-2 acceptance predicate.

    Accepts list[str], tuple[str, ...], np.ndarray (kind 'U' or
    'O' all-str), pd.Series (StringDtype, ArrowDtype string, or
    object all-str). Rejects bare str defensively (would split
    char-by-char).
    """
    if isinstance(x, str):
        return False
    if isinstance(x, (list, tuple)):
        return all(isinstance(item, str) for item in x)
    try:
        if isinstance(x, np.ndarray):
            if x.dtype.kind == 'U':
                return True
            if x.dtype.kind == 'O':
                return all(isinstance(item, str) for item in x)
            return False
        if isinstance(x, pd.Series):
            if isinstance(x.dtype, pd.StringDtype):
                return True
            try:
                arrow_dtype = pd.ArrowDtype
                if isinstance(x.dtype, arrow_dtype):
                    try:
                        import pyarrow as pa  # type: ignore
                        if pa.types.is_string(x.dtype.pyarrow_dtype):
                            return True
                    except (ImportError, AttributeError):
                        pass
            except AttributeError:
                pass  # pandas < 2.0
            if x.dtype == object:
                return all(isinstance(item, str) for item in x)
            return False
    except (ImportError, AttributeError):
        pass
    return False


def parse_rank_answer(
    raw_text: str,
    expected_symbols: Sequence[str],
    normalization_preset: str = "passthrough",
) -> Optional[RankAnswer]:
    """Lenient parser for raw LLM rank responses.

    v2.2.1 #2: tries multiple text formats in order; returns the
    first success. On failure returns None (NOT raises) — the
    orchestrator translates None to parse_status='invalid'.

    Tries in order:
      1. JSON array (with optional markdown code fence stripped).
      2. JSON object with "ranking" key.
      3. Numbered list (lines starting with "N. ").
      4. Bulleted list (lines starting with "- " or "* ").
      5. Comma-separated.
      6. Newline-separated.

    Each parsed symbol is normalized via normalize_symbol with the
    supplied preset before comparing to expected_symbols.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None
    expected_norm = {
        normalize_symbol(s, normalization_preset)
        for s in expected_symbols
    }
    text = raw_text.strip()

    # Strip markdown code fence wrappers if present.
    fence_used = False
    if text.startswith("```"):
        lines = text.split("\n", 1)
        if len(lines) == 2 and lines[1].rstrip().endswith("```"):
            text = lines[1].rstrip()[:-3].strip()
            fence_used = True

    parsed_raw: Optional[list[str]] = None
    parser_label: Optional[str] = None
    tie_groups: tuple[frozenset[str], ...] = ()

    # 1+2: JSON paths.
    try:
        import json as _json
        obj = _json.loads(text)
        if isinstance(obj, list) and all(
            isinstance(x, str) for x in obj
        ):
            parsed_raw = list(obj)
            parser_label = (
                "json_array_codefence" if fence_used else "json_array"
            )
        elif isinstance(obj, dict) and isinstance(
            obj.get("ranking"), list,
        ) and all(isinstance(x, str) for x in obj["ranking"]):
            parsed_raw = list(obj["ranking"])
            parser_label = "json_object"
            tied = obj.get("tied")
            if isinstance(tied, list):
                groups = []
                for grp in tied:
                    if (
                        isinstance(grp, list)
                        and all(isinstance(x, str) for x in grp)
                        and len(grp) >= 2
                    ):
                        groups.append(frozenset(
                            normalize_symbol(s, normalization_preset)
                            for s in grp
                        ))
                tie_groups = tuple(groups)
    except Exception:
        pass

    # 3: numbered list.
    if parsed_raw is None:
        import re
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        numbered_re = re.compile(r"^\d+[\.\)]\s+(.+)$")
        matches = [numbered_re.match(ln) for ln in lines]
        if matches and all(m is not None for m in matches):
            parsed_raw = [m.group(1).strip() for m in matches]
            parser_label = "numbered"

    # 4: bulleted list.
    if parsed_raw is None:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines and all(ln.startswith("- ") for ln in lines):
            parsed_raw = [ln[2:].strip() for ln in lines]
            parser_label = "bulleted_dash"
        elif lines and all(ln.startswith("* ") for ln in lines):
            parsed_raw = [ln[2:].strip() for ln in lines]
            parser_label = "bulleted_asterisk"

    # 5: comma-separated (single line).
    if parsed_raw is None and "," in text and "\n" not in text:
        parsed_raw = [s.strip() for s in text.split(",") if s.strip()]
        parser_label = "comma"

    # 6: newline-separated (no commas).
    if parsed_raw is None and "\n" in text:
        parsed_raw = [
            ln.strip() for ln in text.splitlines() if ln.strip()
        ]
        parser_label = "newline"

    # Final fallback: single-token text.
    if parsed_raw is None:
        return None

    # Reject duplicates.
    seen_norm: list[str] = []
    seen_set: set[str] = set()
    for raw in parsed_raw:
        norm = normalize_symbol(raw, normalization_preset)
        if norm in seen_set:
            return None  # duplicate symbol — invalid permutation
        seen_set.add(norm)
        seen_norm.append(norm)

    # Classify extra and omitted.
    extras = frozenset(s for s in seen_norm if s not in expected_norm)
    omitted = frozenset(
        s for s in expected_norm if s not in seen_set
    )

    return RankAnswer(
        ranking=tuple(seen_norm),
        omitted_symbols=omitted,
        extra_symbols=extras,
        tie_groups=tie_groups,
        raw_ranking=tuple(parsed_raw),
        parser_used=parser_label,
    )


def resolve_rank_answer(
    rec: "AnswerRecord",
    expected_symbols: Sequence[str],
    normalization_preset: str = "passthrough",
) -> tuple[Optional[RankAnswer], str, str]:
    """v2.2.1 #2 priority chain — accept structured input first;
    fall back to text parser.

    Returns (rank_answer, parse_status, parser_used).

    Paths (locked v2.2.1):
      1. parsed_answer is RankAnswer → use directly.
      2. parsed_answer is Sequence[str] (not bare str) → wrap.
      3. parsed_answer is dict with "ranking" key → construct.
      4. parsed_answer is None AND raw_answer is text → text parser.
      5. else → invalid.

    Failure does NOT raise.
    """
    parsed = rec.parsed_answer

    if isinstance(parsed, RankAnswer):
        return parsed, "valid", parsed.parser_used or "user_provided"

    if _is_sequence_of_str(parsed):
        normalized = [
            normalize_symbol(s, normalization_preset) for s in parsed
        ]
        # Reject duplicates.
        if len(set(normalized)) != len(normalized):
            return None, "invalid", "user_provided_list_duplicates"
        expected_norm = {
            normalize_symbol(s, normalization_preset)
            for s in expected_symbols
        }
        ans = RankAnswer(
            ranking=tuple(normalized),
            omitted_symbols=frozenset(
                s for s in expected_norm if s not in normalized
            ),
            extra_symbols=frozenset(
                s for s in normalized if s not in expected_norm
            ),
            raw_ranking=tuple(parsed),
            parser_used="user_provided_list",
        )
        return ans, "valid", "user_provided_list"

    if isinstance(parsed, dict) and _is_sequence_of_str(
        parsed.get("ranking")
    ):
        ranking_raw = list(parsed["ranking"])
        normalized = [
            normalize_symbol(s, normalization_preset)
            for s in ranking_raw
        ]
        if len(set(normalized)) != len(normalized):
            return None, "invalid", "user_provided_dict_duplicates"
        expected_norm = {
            normalize_symbol(s, normalization_preset)
            for s in expected_symbols
        }
        # Optional tied
        tie_groups = ()
        tied = parsed.get("tied")
        if isinstance(tied, list):
            groups = []
            for grp in tied:
                if _is_sequence_of_str(grp) and len(grp) >= 2:
                    groups.append(frozenset(
                        normalize_symbol(s, normalization_preset)
                        for s in grp
                    ))
            tie_groups = tuple(groups)
        ans = RankAnswer(
            ranking=tuple(normalized),
            omitted_symbols=frozenset(
                s for s in expected_norm if s not in normalized
            ),
            extra_symbols=frozenset(
                s for s in normalized if s not in expected_norm
            ),
            tie_groups=tie_groups,
            raw_ranking=tuple(ranking_raw),
            parser_used="user_provided_dict",
        )
        return ans, "valid", "user_provided_dict"

    if parsed is None and isinstance(rec.raw_answer, str):
        ans = parse_rank_answer(
            rec.raw_answer, expected_symbols, normalization_preset,
        )
        if ans is None:
            return None, "invalid", "raw_text_unparseable"
        return ans, "valid", ans.parser_used or "raw_text_unparseable"

    return None, "invalid", "unrecognized_parsed_answer_type"


__all__ = [
    "NormalizationSpec",
    "RankAnswer",
    "RankContinuationProbe",
    "RankCutoffSpec",
    "midranks",
    "normalize_symbol",
    "parse_rank_answer",
    "resolve_rank_answer",
    "score_rank_answer",
    "tie_corrected_spearman",
]
