"""v2.2 M4 tests — RankContinuationProbe + RankAnswer + cutoffs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aiphaforge.probes._rank import (
    RankAnswer,
    RankContinuationProbe,
    RankCutoffSpec,
    midranks,
    normalize_symbol,
    score_rank_answer,
    tie_corrected_spearman,
)

# ---------- normalize_symbol ----------


class TestNormalizeSymbol:
    def test_passthrough_default_preserves_case(self):
        assert normalize_symbol("aapl") == "aapl"

    def test_us_equity_uppercases(self):
        assert normalize_symbol("aapl", "us_equity") == "AAPL"

    def test_us_equity_dash_to_dot(self):
        assert normalize_symbol("brk-b", "us_equity") == "BRK.B"

    def test_crypto_preserves_dash(self):
        assert normalize_symbol("btc-usd", "crypto") == "BTC-USD"

    def test_crypto_uppercases(self):
        assert normalize_symbol("btc-usd", "crypto") == "BTC-USD"

    def test_us_equity_strips_whitespace(self):
        assert normalize_symbol("  aapl  ", "us_equity") == "AAPL"

    def test_passthrough_keeps_whitespace(self):
        assert normalize_symbol("  aapl  ") == "  aapl  "

    def test_rejects_unknown_preset(self):
        with pytest.raises(ValueError):
            normalize_symbol("aapl", "weird_preset")


# ---------- RankAnswer ----------


class TestRankAnswer:
    def test_minimal_construction(self):
        a = RankAnswer(ranking=("AAPL", "TSLA"))
        assert a.ranking == ("AAPL", "TSLA")
        assert a.tie_groups == ()

    def test_rejects_duplicate_ranking(self):
        with pytest.raises(ValueError, match="duplicate"):
            RankAnswer(ranking=("AAPL", "AAPL"))

    def test_rejects_singleton_tie_group(self):
        with pytest.raises(ValueError, match=">=2"):
            RankAnswer(
                ranking=("AAPL", "TSLA"),
                tie_groups=(frozenset({"AAPL"}),),
            )

    def test_rejects_overlapping_tie_groups(self):
        with pytest.raises(ValueError, match="partition"):
            RankAnswer(
                ranking=("A", "B", "C"),
                tie_groups=(
                    frozenset({"A", "B"}),
                    frozenset({"B", "C"}),
                ),
            )

    def test_rejects_tie_group_not_in_ranking(self):
        with pytest.raises(ValueError, match="not in ranking"):
            RankAnswer(
                ranking=("A", "B"),
                tie_groups=(frozenset({"A", "X"}),),
            )

    def test_is_frozen(self):
        a = RankAnswer(ranking=("A", "B"))
        with pytest.raises(Exception):
            a.ranking = ("C",)  # noqa


# ---------- RankCutoffSpec ----------


class TestRankCutoffSpec:
    def test_default_construction(self):
        c = RankCutoffSpec()
        assert c.exact_quantile == 0.99
        assert c.near_quantile == 0.90
        assert c.rough_quantile == 0.75

    def test_rejects_quantile_above_0_99(self):
        with pytest.raises(ValueError, match="0, 0.99"):
            RankCutoffSpec(exact_quantile=0.999)

    def test_rejects_non_decreasing(self):
        with pytest.raises(ValueError, match="strictly decreasing"):
            RankCutoffSpec(
                exact_quantile=0.90, near_quantile=0.95,
                rough_quantile=0.75,
            )


# ---------- tie_corrected_spearman ----------


class TestTieCorrectedSpearman:
    def test_perfect_match_rho_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        rho = tie_corrected_spearman(x, x.copy())
        assert rho == pytest.approx(1.0)

    def test_perfect_anti_match_rho_minus_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = x[::-1]
        rho = tie_corrected_spearman(x, y)
        assert rho == pytest.approx(-1.0)

    def test_matches_scipy_spearmanr_no_ties(self):
        try:
            from scipy.stats import spearmanr
        except ImportError:
            pytest.skip("scipy not available")
        rng = np.random.default_rng(0)
        for _ in range(5):
            x = rng.permutation(20).astype(float) + 1
            y = rng.permutation(20).astype(float) + 1
            r_ours = tie_corrected_spearman(x, y)
            r_scipy = spearmanr(x, y).statistic
            assert r_ours == pytest.approx(r_scipy, abs=1e-12)

    def test_matches_scipy_spearmanr_with_ties(self):
        try:
            from scipy.stats import spearmanr
        except ImportError:
            pytest.skip("scipy not available")
        # Construct ranks with ties.
        x_raw = np.array([1.0, 2.0, 2.0, 4.0, 5.0, 5.0, 5.0, 8.0])
        y_raw = np.array([2.0, 1.0, 3.0, 3.0, 5.0, 5.0, 6.0, 7.0])
        x_ranks = midranks(x_raw)
        y_ranks = midranks(y_raw)
        r_ours = tie_corrected_spearman(x_ranks, y_ranks)
        r_scipy = spearmanr(x_raw, y_raw).statistic
        assert r_ours == pytest.approx(r_scipy, abs=1e-12)


# ---------- score_rank_answer ----------


class TestScoreRankAnswer:
    def test_perfect_ranking_scores_exact(self):
        truth = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")
        ans = RankAnswer(ranking=truth)
        band, prov = score_rank_answer(ans, truth)
        assert band == "exact"
        assert prov["rho"] == pytest.approx(1.0)

    def test_perfect_anti_ranking_scores_miss(self):
        truth = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")
        ans = RankAnswer(ranking=truth[::-1])
        band, _ = score_rank_answer(ans, truth)
        assert band == "miss"

    def test_omitted_symbols_score_invalid(self):
        truth = ("A", "B", "C")
        ans = RankAnswer(
            ranking=("A", "B"),
            omitted_symbols=frozenset({"C"}),
        )
        band, prov = score_rank_answer(ans, truth)
        assert band == "invalid"
        assert prov["reason"] == "omitted_or_extra_symbols"

    def test_extra_symbols_score_invalid(self):
        truth = ("A", "B")
        ans = RankAnswer(
            ranking=("A", "B", "X"),
            extra_symbols=frozenset({"X"}),
        )
        band, _ = score_rank_answer(ans, truth)
        assert band == "invalid"


# ---------- RankContinuationProbe ----------


def _make_multi_data(
    symbols: list[str],
    n: int = 60,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = {}
    idx = pd.bdate_range("2024-01-01", periods=n)
    for i, sym in enumerate(symbols):
        rets = rng.normal(0.0, 0.01, n)
        closes = (100.0 + i * 10.0) * np.exp(np.cumsum(rets))
        out[sym] = pd.DataFrame(
            {
                "open": closes, "high": closes * 1.01,
                "low": closes * 0.99, "close": closes,
                "volume": [1e6] * n,
            },
            index=idx,
        )
    return out


class TestRankContinuationProbe:
    def test_rejects_missing_context_bars(self):
        with pytest.raises(TypeError):
            RankContinuationProbe(  # type: ignore[call-arg]
                symbols=["A", "B"], forward_horizon=1,
            )

    def test_rejects_missing_forward_horizon(self):
        with pytest.raises(TypeError):
            RankContinuationProbe(  # type: ignore[call-arg]
                symbols=["A", "B"], context_bars=10,
            )

    def test_rejects_fewer_than_2_symbols(self):
        with pytest.raises(ValueError):
            RankContinuationProbe(
                symbols=["A"], context_bars=10, forward_horizon=1,
            )

    def test_one_question_per_anchor(self):
        symbols = ["A", "B", "C"]
        data = _make_multi_data(symbols, n=60)
        probe = RankContinuationProbe(
            symbols=symbols, context_bars=10, forward_horizon=1,
        )
        anchors = list(data["A"].index[20:25])
        qs = list(probe.build(data, anchors))
        assert len(qs) == 5  # 5 anchors → 5 questions

    def test_truth_value_is_descending_return_ranking(self):
        symbols = ["A", "B", "C"]
        # Construct deterministic returns: A>B>C at t+1
        idx = pd.bdate_range("2024-01-01", periods=30)
        data = {
            "A": pd.DataFrame(
                {
                    "open": [100.0] * 30, "high": [101.0] * 30,
                    "low": [99.0] * 30,
                    "close": [100.0] * 20 + [105.0] + [100.0] * 9,
                    "volume": [1e6] * 30,
                }, index=idx,
            ),
            "B": pd.DataFrame(
                {
                    "open": [100.0] * 30, "high": [101.0] * 30,
                    "low": [99.0] * 30,
                    "close": [100.0] * 20 + [102.0] + [100.0] * 9,
                    "volume": [1e6] * 30,
                }, index=idx,
            ),
            "C": pd.DataFrame(
                {
                    "open": [100.0] * 30, "high": [101.0] * 30,
                    "low": [99.0] * 30,
                    "close": [100.0] * 20 + [99.0] + [100.0] * 9,
                    "volume": [1e6] * 30,
                }, index=idx,
            ),
        }
        probe = RankContinuationProbe(
            symbols=symbols, context_bars=5, forward_horizon=1,
        )
        # anchor at index 19 → forward 1 = index 20 (where prices diverge)
        anchors = [idx[19]]
        qs = list(probe.build(data, anchors))
        assert len(qs) == 1
        assert qs[0].truth_value == ("A", "B", "C")

    def test_skips_anchors_too_close_to_edges(self):
        symbols = ["A", "B"]
        data = _make_multi_data(symbols, n=20)
        probe = RankContinuationProbe(
            symbols=symbols, context_bars=15, forward_horizon=1,
        )
        # context_bars=15, anchor=5 → needs index -9. Skip.
        anchors = [data["A"].index[5]]
        qs = list(probe.build(data, anchors))
        assert len(qs) == 0


# ---------- N-dependent quantile ----------


class TestSpearmanNullQuantile:
    def test_n_eq_10_quantile_exists_and_finite(self):
        from aiphaforge.probes._rank import _spearman_null_quantile
        rho_q = _spearman_null_quantile(10, 0.95)
        assert 0 < rho_q < 1.0

    def test_n_50_uses_gaussian(self):
        from aiphaforge.probes._rank import _spearman_null_quantile
        # Gaussian: rho_q = z(0.95) / sqrt(N-1) ≈ 1.96/sqrt(49) ≈ 0.28
        rho_q = _spearman_null_quantile(50, 0.95)
        assert rho_q == pytest.approx(1.96 / 7.0, abs=0.01)

    def test_smaller_n_has_larger_threshold(self):
        from aiphaforge.probes._rank import _spearman_null_quantile
        # SE of rho ~ 1/sqrt(N-1); smaller N → wider null → larger
        # required rho for the same quantile.
        rho_n10 = _spearman_null_quantile(10, 0.95)
        rho_n100 = _spearman_null_quantile(100, 0.95)
        assert rho_n10 > rho_n100
