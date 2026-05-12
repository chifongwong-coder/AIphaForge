"""v2.2.1 #6 — Precomputed Spearman null-quantile table tests."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from aiphaforge.probes._rank import (
    _load_rank_null_table,
    _spearman_null_quantile,
    tie_corrected_spearman,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
NPZ_PATH = REPO_ROOT / "src/aiphaforge/probes/_rank_null.npz"


class TestRankNullTable:
    def test_npz_file_shipped(self):
        assert NPZ_PATH.exists(), (
            "rank-null table .npz must be shipped with the package"
        )

    def test_table_covers_n_3_to_20(self):
        table = _load_rank_null_table()
        for n in range(3, 21):
            assert f"abs_rho_n{n}" in table, f"missing N={n}"

    def test_table_metadata_records_seed(self):
        table = _load_rank_null_table()
        assert "__meta_mc_root_seed" in table
        assert int(table["__meta_mc_root_seed"][0]) == 20260512

    def test_table_metadata_records_per_n_seeds(self):
        table = _load_rank_null_table()
        assert "__meta_mc_per_n_seeds" in table
        per_n = table["__meta_mc_per_n_seeds"]
        # 10 entries for N=11..20
        assert len(per_n) == 10
        # Per-N seeds are independent (no two equal — astronomically
        # unlikely from a single SeedSequence.spawn call).
        assert len(set(per_n.tolist())) == 10

    def test_table_metadata_records_n_ranges(self):
        table = _load_rank_null_table()
        exact_range = tuple(table["__meta_exact_n_range"].tolist())
        mc_range = tuple(table["__meta_mc_n_range"].tolist())
        # v2.2.1 final: exact ≤ 10, MC for 11..20 (12! is too slow
        # for pure-Python iteration; documented practical choice).
        assert exact_range == (3, 10)
        assert mc_range == (11, 20)

    def test_table_metadata_records_generator_version(self):
        table = _load_rank_null_table()
        assert "__meta_generator_version" in table
        assert "v2.2.1" in str(table["__meta_generator_version"][0])

    def test_smaller_n_has_larger_quantile_threshold(self):
        # SE(rho) ~ 1/sqrt(N-1) — smaller N → wider null →
        # higher rho needed for the same quantile.
        rho_n10 = _spearman_null_quantile(10, 0.95)
        rho_n20 = _spearman_null_quantile(20, 0.95)
        assert rho_n10 > rho_n20

    def test_n_5_quantile_finite_and_in_range(self):
        rho = _spearman_null_quantile(5, 0.95)
        assert 0 < rho < 1.0
        assert math.isfinite(rho)

    def test_n_10_uses_exact_lookup(self):
        # Sanity: N=10 row should be exactly 10! = 3628800 entries
        table = _load_rank_null_table()
        assert len(table["abs_rho_n10"]) == 3628800

    def test_n_15_uses_mc_lookup(self):
        # MC samples = 1_000_000
        table = _load_rank_null_table()
        assert len(table["abs_rho_n15"]) == 1_000_000

    def test_n_above_20_uses_gaussian(self):
        # N=50 should use closed-form Gaussian, ~ 1.96/sqrt(49)
        rho_q = _spearman_null_quantile(50, 0.95)
        assert rho_q == pytest.approx(1.96 / 7.0, abs=0.01)

    def test_quantile_lookup_matches_test_time_mc_at_n_15(self):
        # Independent fresh MC at N=15, q=0.99 should match the
        # table to within ~3e-3 (table SE is ~1e-3 at this q;
        # fresh MC SE is similar).
        rho_table = _spearman_null_quantile(15, 0.99)
        rng = np.random.Generator(np.random.PCG64(99999))
        n_test = 100_000
        canonical = np.arange(1, 16, dtype=float)
        samples = np.array([
            abs(tie_corrected_spearman(canonical, rng.permutation(canonical)))
            for _ in range(n_test)
        ])
        samples.sort()
        rho_test = float(samples[int(0.99 * n_test)])
        # Tolerance: ~1e-2 to account for both MC SEs.
        assert abs(rho_table - rho_test) < 1e-2
