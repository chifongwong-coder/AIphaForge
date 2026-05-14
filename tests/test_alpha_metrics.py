"""v2.4 Commit G — alpha.metrics tests."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aiphaforge.alpha.labels import forward_returns
from aiphaforge.alpha.metrics import coverage, ic, rank_ic
from aiphaforge.alpha.rank_stats import midranks, tie_corrected_spearman


def _wide(periods: int = 10, n_symbols: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=periods)
    cols = [f"S{i}" for i in range(n_symbols)]
    data = rng.normal(0.0, 1.0, (periods, n_symbols))
    return pd.DataFrame(data, index=idx, columns=cols)


class TestIC:
    def test_constructed_factor_correlates_with_returns(self):
        # When factor IS the forward return, IC == 1 exactly modulo
        # last-digit float (per quant-review v1).
        prices = pd.DataFrame(
            100.0 * np.exp(np.cumsum(
                np.random.default_rng(42).normal(0.0, 0.01, (50, 5)),
                axis=0,
            )),
            index=pd.bdate_range("2024-01-01", periods=50),
            columns=[f"S{i}" for i in range(5)],
        )
        fr = forward_returns(prices, horizon=1)
        # Use the forward returns themselves as the factor.
        ic_series = ic(fr, fr).dropna()
        for v in ic_series:
            assert v == pytest.approx(1.0, abs=1e-12)

    def test_returns_nan_when_factor_has_no_variance(self):
        # Constant factor row → std=0 → IC=NaN at that timestamp.
        idx = pd.bdate_range("2024-01-01", periods=3)
        factor = pd.DataFrame(
            {"A": [1.0, 0.5, 0.5], "B": [1.0, 0.4, 0.5],
             "C": [1.0, 0.6, 0.5]}, index=idx,
        )
        fr = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03], "B": [-0.01, 0.0, 0.01],
             "C": [0.0, -0.01, 0.0]}, index=idx,
        )
        out = ic(factor, fr)
        # Bar 0: factor all 1.0 → no variance → NaN.
        # Bar 2: factor all 0.5 → no variance → NaN.
        assert pd.isna(out.iloc[0])
        assert pd.isna(out.iloc[2])

    def test_ic_handles_partial_overlap_columns_raises(self):
        # R9: column-misalignment must raise, not silently misalign.
        f = _wide(periods=3, n_symbols=3)
        r = _wide(periods=3, n_symbols=3)
        r.columns = ["S0", "S1", "S99"]  # different symbol set
        with pytest.raises(ValueError, match="do not match"):
            ic(f, r)

    def test_ic_set_equal_but_wrong_order_emits_helpful_error(self):
        f = _wide(periods=3, n_symbols=3)  # S0, S1, S2
        r = f[["S2", "S1", "S0"]]  # same set, reversed order
        with pytest.raises(ValueError, match="reindex"):
            ic(f, r)


class TestRankIC:
    def test_rank_ic_uses_midranks_then_tie_corrected_spearman(self):
        # Cross-check production rank_ic against the documented
        # midranks → tie_corrected_spearman pipeline computed
        # directly here. NOT a re-run of production internals.
        idx = pd.bdate_range("2024-01-01", periods=2)
        # Bar 0: tie-free.
        # Bar 1: one tied pair in factor (B and C both at 2.0).
        factor = pd.DataFrame(
            {"A": [1.0, 1.0], "B": [2.0, 2.0],
             "C": [3.0, 2.0], "D": [4.0, 4.0]}, index=idx,
        )
        fr = pd.DataFrame(
            {"A": [0.01, 0.02], "B": [0.02, 0.01],
             "C": [0.03, 0.04], "D": [0.04, 0.03]}, index=idx,
        )
        rho = rank_ic(factor, fr)
        # Hand-compute via the same pipeline production should use.
        for t in idx:
            f_v = factor.loc[t].to_numpy(dtype=float)
            r_v = fr.loc[t].to_numpy(dtype=float)
            expected = tie_corrected_spearman(midranks(f_v), midranks(r_v))
            assert rho.loc[t] == pytest.approx(expected, abs=1e-12)


class TestCoverage:
    def test_coverage_counts_non_nan_fraction(self):
        idx = pd.bdate_range("2024-01-01", periods=3)
        factor = pd.DataFrame(
            {"A": [1.0, np.nan, 1.0], "B": [np.nan, np.nan, 1.0],
             "C": [1.0, 1.0, np.nan]}, index=idx,
        )
        cov = coverage(factor)
        assert cov.iloc[0] == pytest.approx(2 / 3)
        assert cov.iloc[1] == pytest.approx(1 / 3)
        assert cov.iloc[2] == pytest.approx(2 / 3)


class TestAlphaMetricsFirewall:
    def test_alpha_metrics_does_not_import_engine(self):
        path = (
            Path(__file__).resolve().parent.parent
            / "src/aiphaforge/alpha/metrics.py"
        )
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert "aiphaforge.engine" not in name
