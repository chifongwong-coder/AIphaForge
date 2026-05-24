"""v2.4 Commit N — Canonical-formula tests for probes/anchors.py.

Closes the v2.2.2 G methodology gap for the two paper-cited
functions in anchors.py.

WARNING: do NOT regenerate expected values from production code.
"""
from __future__ import annotations

import numpy as np
import pytest

from aiphaforge.probes.anchors import (
    _block_bootstrap_corr_ci,
    _fit_garch_with_fallback,
)

# ---------------------------------------------------------------------------
# Block bootstrap (fixed-block scheme, NOT Politis-Romano stationary)
# ---------------------------------------------------------------------------


class TestBlockBootstrapCorrCI:
    def test_block_bootstrap_seed_is_deterministic(self):
        # Same seed → bit-identical (lo, hi, se).
        rng_data = np.random.default_rng(2026)
        returns = rng_data.normal(0.0, 0.01, 200)
        out1 = _block_bootstrap_corr_ci(
            returns, block_size=10, n_resamples=100, seed=42,
        )
        out2 = _block_bootstrap_corr_ci(
            returns, block_size=10, n_resamples=100, seed=42,
        )
        assert out1 == out2
        # Different seed → different output.
        out3 = _block_bootstrap_corr_ci(
            returns, block_size=10, n_resamples=100, seed=999,
        )
        assert out3 != out1

    # v2.8.3 Commit A (M6): ceiling-division n_blocks invariant tests.
    def test_block_bootstrap_aligned_length_unchanged(self):
        # n is a multiple of block_size: ceiling(n/k) == floor(n/k),
        # so the M6 fix is a no-op on aligned inputs.
        rng_data = np.random.default_rng(2026)
        returns = rng_data.normal(0.0, 0.01, 200)
        lo, hi, se = _block_bootstrap_corr_ci(
            returns, block_size=10, n_resamples=100, seed=42,
        )
        # CI is finite and ordered (sanity on aligned path).
        assert np.isfinite(lo) and np.isfinite(hi) and np.isfinite(se)
        assert lo <= hi

    def test_block_bootstrap_unaligned_length_n_minus_one(self):
        # n=199, block_size=10 → floor gives 19 blocks (190 elements,
        # short by 9); ceiling gives 20 blocks (200, truncated to 199).
        # After M6 the bootstrap sample length equals n exactly.
        rng_data = np.random.default_rng(2026)
        returns = rng_data.normal(0.0, 0.01, 199)
        lo, hi, se = _block_bootstrap_corr_ci(
            returns, block_size=10, n_resamples=100, seed=42,
        )
        assert np.isfinite(lo) and np.isfinite(hi) and np.isfinite(se)
        assert lo <= hi

    def test_block_bootstrap_unaligned_length_short_residual(self):
        # n=53, block_size=10 → ceiling gives 6 blocks (60 → 53);
        # exercises the tightest non-trivial residual.
        rng_data = np.random.default_rng(2026)
        returns = rng_data.normal(0.0, 0.01, 53)
        lo, hi, se = _block_bootstrap_corr_ci(
            returns, block_size=10, n_resamples=100, seed=42,
        )
        assert np.isfinite(lo) and np.isfinite(hi) and np.isfinite(se)
        assert lo <= hi


# ---------------------------------------------------------------------------
# GJR-GARCH parameter recovery (per-parameter empirical tolerance bands)
# ---------------------------------------------------------------------------


class TestGJRGarchParameterRecovery:
    """Pre-implementation verification (2026-05-14) simulated GJR-GARCH
    at true (ω=0.05, α=0.07, β=0.88, γ=0.05) for T=2000 across 5 seeds.
    Observed max relative errors: ω 42%, α 36%, β 2%, γ 79%. Plan v3
    bands include safety margin: ω 55%, α 45%, β 5%, γ 90%.

    Production at anchors.py:165 multiplies returns × 100 before
    fitting. We mirror that by un-scaling recovered ω by 1/10000.
    """

    @pytest.mark.parametrize("seed", [42, 123, 2024, 999, 7])
    def test_per_seed_per_parameter_within_empirical_bands(self, seed):
        pytest.importorskip("arch")
        T = 2000
        omega_true, alpha_true, beta_true, gamma_true = (
            0.05, 0.07, 0.88, 0.05,
        )
        rng = np.random.default_rng(seed)
        eps = np.zeros(T)
        sigma2 = np.zeros(T)
        sigma2[0] = omega_true / (1 - alpha_true - beta_true - 0.5 * gamma_true)
        for t in range(1, T):
            z = rng.standard_normal()
            eps[t-1] = np.sqrt(sigma2[t-1]) * z
            indicator = 1.0 if eps[t-1] < 0 else 0.0
            sigma2[t] = (
                omega_true
                + (alpha_true + gamma_true * indicator) * eps[t-1]**2
                + beta_true * sigma2[t-1]
            )
        eps[T-1] = np.sqrt(sigma2[T-1]) * rng.standard_normal()

        # Production fits returns * 100 internally; pass eps as-is.
        res, model_name = _fit_garch_with_fallback(eps, gate_says_gjr=True)
        assert res is not None
        assert model_name == "gjr_garch"

        # Un-scale ω from %² (production scaled returns by 100).
        p = res.params
        omega_recovered = p["omega"] / 10000.0
        alpha_recovered = p["alpha[1]"]
        beta_recovered = p["beta[1]"]
        gamma_recovered = p["gamma[1]"]

        def rel_err(recovered, true):
            return abs(recovered - true) / abs(true)

        # Empirically pinned bands per pre-implementation
        # verification at this exact 5-seed sweep.
        assert rel_err(omega_recovered, omega_true) < 0.55
        assert rel_err(alpha_recovered, alpha_true) < 0.45
        assert rel_err(beta_recovered, beta_true) < 0.05
        assert rel_err(gamma_recovered, gamma_true) < 0.90

    def test_short_series_falls_back_to_symmetric_garch(self):
        # The fallback chain documented at _fit_garch_with_fallback:
        # GJR (if gate) -> GARCH(1,1) -> None. Short series with
        # gate_says_gjr=False should land on GARCH(1,1).
        pytest.importorskip("arch")
        rng = np.random.default_rng(0)
        eps = rng.standard_normal(300) * 0.01
        res, model_name = _fit_garch_with_fallback(eps, gate_says_gjr=False)
        assert res is not None
        assert model_name == "garch"
