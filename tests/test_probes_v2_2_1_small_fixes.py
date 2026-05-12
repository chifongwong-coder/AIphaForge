"""v2.2.1 #9, #10 — small fixes (sign-test fallback, GK missing-open)."""
from __future__ import annotations

import pandas as pd

from aiphaforge.probes._vol import estimate_sigma
from aiphaforge.probes.orchestrator import sign_test_p

# ---------- #9: sign-test n<25 fallback ----------


class TestSignTestSmallNFallback:
    def test_n_lt_25_uses_exact_binomial_when_scipy_present(self):
        # scipy is installed in this test env; n<25 → exact_binomial.
        p, variant = sign_test_p(5, 5)
        assert variant == "exact_binomial"

    def test_n_25_uses_continuity_correction(self):
        p, variant = sign_test_p(15, 10)
        assert variant == "normal_continuity_corrected"

    def test_n_40_uses_normal_approx(self):
        p, variant = sign_test_p(25, 15)
        assert variant == "normal_approximation"


# ---------- #10: Garman-Klass falls back to Parkinson when `open` missing ----------


class TestGarmanKlassFallbackOpenMissing:
    def test_garman_klass_uses_parkinson_when_open_column_absent(self):
        # Build a bars frame with NO 'open' column.
        idx = pd.bdate_range("2024-01-01", periods=10)
        bars = pd.DataFrame(
            {
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.0] * 10,
                "volume": [1e6] * 10,
            },
            index=idx,
        )
        sigma, prov = estimate_sigma(bars, "garman_klass")
        # The H==L fallback is also triggerable; but here H!=L,
        # so the 'missing open' fallback is what we test.
        # Parkinson on these bars is a known nonzero value.
        assert prov["method_used"] == "parkinson"
        assert "garman_klass_missing_open_fallback" in prov

    def test_garman_klass_uses_open_when_present(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        bars = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.5] * 10,
                "volume": [1e6] * 10,
            },
            index=idx,
        )
        sigma, prov = estimate_sigma(bars, "garman_klass")
        assert prov["method_used"] == "garman_klass"
        assert "garman_klass_missing_open_fallback" not in prov
